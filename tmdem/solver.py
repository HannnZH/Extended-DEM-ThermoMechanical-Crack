"""
Networks, energy losses and training loops of the thermo-mechanical solver.

The mechanical and thermal networks receive [x, y, Gamma], where Gamma is
the crack embedding.  The asymptotic enrichment of the displacement at the tip
(Williams terms) is optional and enabled by default.

Thermo-mechanical coupling is one-way through
    eps_elastic = eps_total - alpha * (T - T_ref) * I,
with alpha the coefficient returned by thermal_strain_coefficient.

Author: Han Zhang (han.zhang7@unsw.edu.au)
"""


import copy
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

try:
    from scipy.stats.qmc import LatinHypercube
except Exception:  # scipy is a requirement; numpy sampling is the fallback.
    LatinHypercube = None


def set_seed(seed: int = 2025, deterministic: bool = True, strict: bool = False) -> None:
    """Seed the python, numpy and torch generators and request deterministic kernels.

    Seeding alone does not make a GPU run reproducible, since kernels that
    reduce in a nondeterministic order give float-level differences between
    runs; ``deterministic`` therefore pins the kernels as well.

    ``strict`` selects what happens at an operation with no deterministic
    implementation: raise (True) or fall back with a warning (False), which
    the run log records.

    CUBLAS_WORKSPACE_CONFIG must be set before cuBLAS initializes, so it is
    set here unless the environment already defines it.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # benchmark picks a convolution algorithm by timing it, so leaving it on
    # would let the machine's load decide which kernel runs.
    torch.backends.cudnn.benchmark = False

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=not strict)
        except Exception as exc:      # older torch, or no such setting
            print(f"[seed] deterministic algorithms unavailable: {exc}")

    # Printed here so that the log of every run records the seed and the
    # environment.
    print(f"[seed] seed={seed}")
    print(describe_run_environment())


def describe_run_environment() -> str:
    """Return one line with the settings that can change a result between machines.

    The line names the torch and CUDA versions, the GPU and the determinism
    settings.  It is written into the run log so that runs on different
    machines can be compared on the same footing.
    """
    if torch.cuda.is_available():
        card = torch.cuda.get_device_name(0)
        cuda = torch.version.cuda
    else:
        card, cuda = "cpu", "n/a"
    try:
        det = torch.are_deterministic_algorithms_enabled()
    except Exception:
        det = "unknown"
    return (f"[env] torch {torch.__version__} cuda {cuda} | card {card} | "
            f"deterministic={det} cudnn.deterministic={torch.backends.cudnn.deterministic} "
            f"cudnn.benchmark={torch.backends.cudnn.benchmark} "
            f"CUBLAS_WORKSPACE_CONFIG={os.environ.get('CUBLAS_WORKSPACE_CONFIG', 'unset')}")


class MLP(nn.Module):
    """Small tanh MLP used for the displacement branch."""

    def __init__(self, layers: Sequence[int], activation: type[nn.Module] = nn.Tanh):
        super().__init__()
        blocks: List[nn.Module] = []
        for i in range(len(layers) - 1):
            blocks.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                blocks.append(activation())
        self.net = nn.Sequential(*blocks)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ThermalNet(nn.Module):
    """
    SIREN-lite thermal MLP.

    A sinusoidal first layer gives clean spatial derivatives for the
    conduction energy, while the later tanh layers keep training stable.
    """

    def __init__(self, layers: Sequence[int], lb: Sequence[float], ub: Sequence[float], omega_0: float = 30.0):
        super().__init__()
        assert len(layers) >= 3, "ThermalNet expects at least input, hidden, output dimensions."
        self.register_buffer("lb", torch.tensor(lb, dtype=torch.float32))
        self.register_buffer("ub", torch.tensor(ub, dtype=torch.float32))
        self.omega_0 = float(omega_0)

        self.first = nn.Linear(layers[0], layers[1])
        nn.init.uniform_(self.first.weight, -1.0 / layers[0], 1.0 / layers[0])
        nn.init.zeros_(self.first.bias)

        rest: List[nn.Module] = []
        for i in range(1, len(layers) - 1):
            rest.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                rest.append(nn.Tanh())
        self.rest = nn.Sequential(*rest)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Only x,y are geometrically normalised.  The optional crack embedding
        # channel Gamma already lies roughly in [-1,1], so it is left unchanged.
        if x.shape[1] >= 2:
            xy = x[:, :2]
            xyn = 2.0 * (xy - self.lb) / (self.ub - self.lb) - 1.0
            if x.shape[1] > 2:
                x_in = torch.cat([xyn, x[:, 2:]], dim=1)
            else:
                x_in = xyn
        else:
            x_in = x
        h = torch.sin(self.omega_0 * self.first(x_in))
        return self.rest(h)


class AxisScaleNet(nn.Module):
    """Normalise x,y to [-1,1] and keep any extra embedding channels unchanged."""

    def __init__(self, net: nn.Module, lb: Sequence[float], ub: Sequence[float]):
        super().__init__()
        self.net = net
        self.register_buffer("lb", torch.tensor(lb, dtype=torch.float32))
        self.register_buffer("ub", torch.tensor(ub, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xy = x[:, :2]
        xyn = 2.0 * (xy - self.lb) / (self.ub - self.lb) - 1.0
        if x.shape[1] > 2:
            xyn = torch.cat([xyn, x[:, 2:]], dim=1)
        return self.net(xyn)


def get_uniform_mesh(nx_elem: int, ny_elem: int, domain_ends_x: Sequence[float], domain_ends_y: Sequence[float]) -> np.ndarray:
    """Uniform rectangular element list: [x_min, x_max, y_min, y_max, level]."""
    x_edges = np.linspace(domain_ends_x[0], domain_ends_x[1], nx_elem + 1)
    y_edges = np.linspace(domain_ends_y[0], domain_ends_y[1], ny_elem + 1)
    vertex = []
    for j in range(ny_elem):
        for i in range(nx_elem):
            vertex.append([x_edges[i], x_edges[i + 1], y_edges[j], y_edges[j + 1], 0])
    return np.asarray(vertex, dtype=np.float64)


def update_adaptive_mesh(
    nx_elem: int,
    ny_elem: int,
    domain_ends_x: Sequence[float],
    domain_ends_y: Sequence[float],
    refined_elements: Optional[List[Dict]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rectangular refinement of the uniform element list.  Unused when mesh_policy='fixed'."""
    vertex = get_uniform_mesh(nx_elem, ny_elem, domain_ends_x, domain_ends_y)
    parent_indices = np.full(len(vertex), -1, dtype=np.int64)
    if not refined_elements:
        return vertex, parent_indices

    new_vertex: List[List[float]] = []
    new_parent_indices: List[int] = []
    for elem_id, elem in enumerate(vertex):
        x_min, x_max, y_min, y_max, current_level = elem
        xc = 0.5 * (x_min + x_max)
        yc = 0.5 * (y_min + y_max)
        refine_info = None
        for item in refined_elements:
            x0, x1, y0, y1 = item["coords"]
            if x0 <= xc <= x1 and y0 <= yc <= y1:
                refine_info = item
                break

        if refine_info is None or refine_info.get("level", 0) <= current_level:
            new_vertex.append(elem.tolist())
            new_parent_indices.append(-1)
            continue

        level = int(refine_info.get("level", current_level + 1))
        sx, sy = refine_info.get("subdiv", (2, 2))
        x_splits = np.linspace(x_min, x_max, sx + 1)
        y_splits = np.linspace(y_min, y_max, sy + 1)
        for jj in range(sy):
            for ii in range(sx):
                new_vertex.append([x_splits[ii], x_splits[ii + 1], y_splits[jj], y_splits[jj + 1], level])
                new_parent_indices.append(elem_id)

    return np.asarray(new_vertex, dtype=np.float64), np.asarray(new_parent_indices, dtype=np.int64)


def mark_elements_for_refinement(
    X_f: np.ndarray,
    elem_idx: np.ndarray,
    vertex: np.ndarray,
    residual_pred: np.ndarray,
    crack_gamma: Optional[np.ndarray] = None,
    error_percentile: float = 10.0,
    gamma_threshold: float = 0.2,
    max_level: int = 2,
) -> List[Dict]:
    """
    Mark elements with large mechanical residuals and elements near the crack.

    Used only with mesh_policy='adaptive'; the examples use a fixed uniform
    mesh of background elements.
    """
    elem_idx = elem_idx.astype(int).flatten()
    residual = np.abs(residual_pred).reshape(-1)
    unique_elems = np.unique(elem_idx)
    elem_error = []
    for e in unique_elems:
        ids = np.where(elem_idx == e)[0]
        elem_error.append(np.sum(residual[ids] * X_f[ids, 2]))
    elem_error = np.asarray(elem_error)
    n_top = max(1, int(round(len(unique_elems) * error_percentile / 100.0)))
    selected = set(unique_elems[np.argsort(-elem_error)[:n_top]].tolist())

    if crack_gamma is not None:
        gamma = np.abs(crack_gamma).reshape(-1)
        selected.update(np.unique(elem_idx[gamma > gamma_threshold]).astype(int).tolist())

    subdiv_schemes = {1: (2, 2), 2: (4, 4), 3: (8, 8)}
    refined = []
    for e in sorted(selected):
        if e < 0 or e >= len(vertex):
            continue
        level = int(min(vertex[e, 4] + 1, max_level))
        refined.append({
            "coords": vertex[e, :4].tolist(),
            "level": level,
            "max_level": max_level,
            "subdiv": subdiv_schemes.get(level, (2, 2)),
        })
    return refined


def plot_mesh(vertex: np.ndarray, filename: str = "Mesh", save_dir: str = "results", show: bool = False) -> None:
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(8, 8))
    for elem in vertex:
        x0, x1, y0, y1 = elem[:4]
        plt.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], linewidth=0.5, color="k")
    plt.axis("equal")
    plt.title("Mesh")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{filename}.png"), dpi=300)
    if show:
        plt.show()
    plt.close()


def plot_field(
    points: np.ndarray,
    field_values: np.ndarray,
    domain_ends_x: Sequence[float],
    domain_ends_y: Sequence[float],
    title: str,
    label: str,
    filename: str,
    save_dir: str,
    grid_size: int = 200,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "viridis",
    show: bool = False,
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(10, 8))

    points = np.asarray(points, dtype=float)
    values = np.asarray(field_values).reshape(-1)

    # If points already form a tensor-product prediction grid, reshape
    # directly instead of calling griddata.  Re-interpolating a field that
    # jumps across the crack produces jagged artefacts near the crack.
    xu = np.unique(points[:, 0])
    yu = np.unique(points[:, 1])
    is_structured = (len(xu) * len(yu) == points.shape[0])
    if is_structured:
        order = np.lexsort((points[:, 0], points[:, 1]))
        xi_grid, yi_grid = np.meshgrid(xu, yu, indexing="xy")
        field_grid = values[order].reshape(len(yu), len(xu))
    else:
        xi = np.linspace(domain_ends_x[0], domain_ends_x[1], grid_size)
        yi = np.linspace(domain_ends_y[0], domain_ends_y[1], grid_size)
        xi_grid, yi_grid = np.meshgrid(xi, yi)
        field_grid = griddata(points, values, (xi_grid, yi_grid), method="linear", fill_value=np.nan)

    mesh = plt.pcolormesh(
        xi_grid, yi_grid, field_grid,
        cmap=cmap, vmin=vmin, vmax=vmax,
        shading="auto", edgecolors="none", antialiased=False, rasterized=True,
    )
    plt.colorbar(mesh, label=label)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{filename}.png"), dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def load_curve_txt(txt_path: str) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.isfile(txt_path):
        raise FileNotFoundError(f"FEM curve file not found: {txt_path}")
    data = np.loadtxt(txt_path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise ValueError(f"{txt_path} must have at least two columns: displacement force")
    return data[:, 0], data[:, 1]


def plot_force_disp(fd_graph: np.ndarray, filename: str = "Force_Displacement", outdir: str = "results", show: bool = False) -> None:
    if fd_graph is None or len(fd_graph) == 0:
        return
    os.makedirs(outdir, exist_ok=True)
    valid = np.isfinite(fd_graph[:, 0]) & np.isfinite(fd_graph[:, 1])
    data = fd_graph[valid]
    if len(data) == 0:
        return
    plt.figure(figsize=(5, 5))
    plt.plot(data[:, 0], data[:, 1], linewidth=2.0)
    plt.xlabel("Displacement")
    plt.ylabel("Force")
    plt.title("Force-Displacement Diagram")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, filename + ".png"), dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_force_disp_compare(
    fd_graph: np.ndarray,
    fem_curves: Dict[str, Tuple[np.ndarray, np.ndarray]],
    outpath: str,
    show: bool = False,
) -> None:
    if fd_graph is None or len(fd_graph) == 0:
        return
    outdir = os.path.dirname(outpath)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    valid = np.isfinite(fd_graph[:, 0]) & np.isfinite(fd_graph[:, 1])
    data = fd_graph[valid]
    if len(data) == 0:
        return
    mask = np.concatenate(([True], np.diff(data[:, 0]) > 0))
    data = data[mask]
    plt.figure(figsize=(6, 5))
    plt.plot(data[:, 0], data[:, 1], linewidth=2.0, label="proposed method")
    for label, (x, y) in fem_curves.items():
        plt.plot(x, y, linewidth=2.0, label=label)
    plt.xlabel("Displacement")
    plt.ylabel("Force")
    plt.title("Force-displacement: proposed method and finite element reference")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def _curve_diff_data(samples: np.ndarray):
    """Compute tangents, normals and cumulative arc length of a sampled crack curve."""
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] != 2:
        raise ValueError(f"curve samples must have shape (N, 2), got {samples.shape}")
    if samples.shape[0] < 2:
        raise ValueError("curve needs at least two points")

    tan = np.empty_like(samples)
    tan[1:-1] = samples[2:] - samples[:-2]
    tan[0] = samples[1] - samples[0]
    tan[-1] = samples[-1] - samples[-2]

    nrm = np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1.0e-12)
    tan = tan / nrm
    nor = np.stack([-tan[:, 1], tan[:, 0]], axis=1)

    ds = np.linalg.norm(samples[1:] - samples[:-1], axis=1)
    cum_s = np.concatenate([[0.0], np.cumsum(ds)])
    return tan, nor, cum_s, float(cum_s[-1])


def _interp_curve(samples, tangents, normals, cum_s, s_query):
    """Interpolate crack curve position/tangent/normal by arc length."""
    samples = np.asarray(samples, dtype=np.float64)
    tangents = np.asarray(tangents, dtype=np.float64)
    cum_s = np.asarray(cum_s, dtype=np.float64)
    s_query = np.asarray(s_query, dtype=np.float64).reshape(-1)

    if samples.shape[0] < 2 or cum_s[-1] <= 1.0e-14:
        x = np.repeat(samples[:1], len(s_query), axis=0)
        t = np.tile(np.array([[1.0, 0.0]], dtype=np.float64), (len(s_query), 1))
        n = np.tile(np.array([[0.0, 1.0]], dtype=np.float64), (len(s_query), 1))
        return x, t, n

    s_clip = np.clip(s_query, cum_s[0], cum_s[-1])
    x_out = np.empty((len(s_clip), 2), dtype=np.float64)
    t_out = np.empty((len(s_clip), 2), dtype=np.float64)

    for i, sq in enumerate(s_clip):
        j = np.searchsorted(cum_s, sq, side="right") - 1
        j = int(np.clip(j, 0, len(cum_s) - 2))
        ds = max(float(cum_s[j + 1] - cum_s[j]), 1.0e-14)
        a = float((sq - cum_s[j]) / ds)
        x_out[i] = (1.0 - a) * samples[j] + a * samples[j + 1]

        t_raw = (1.0 - a) * tangents[j] + a * tangents[j + 1]
        t_out[i] = t_raw / max(float(np.linalg.norm(t_raw)), 1.0e-14)

    n_out = np.stack([-t_out[:, 1], t_out[:, 0]], axis=1)
    return x_out, t_out, n_out


@dataclass
class CrackGeometry:
    """Active crack geometry shared by the crack-face sampling, the SIF extraction and the enrichment."""

    control_points: Sequence[Sequence[float]]
    curve_points: np.ndarray
    curve_tangents: np.ndarray
    curve_normals: np.ndarray
    curve_s: np.ndarray
    total_length_value: float

    @classmethod
    def from_embedding(cls, embedding, control_points):
        control_points = [list(map(float, p)) for p in control_points]
        if hasattr(embedding, "curve_data"):
            pts, tan, nor, s, total_L = embedding.curve_data()
        else:
            pts = np.asarray(control_points, dtype=np.float64)
            tan, nor, s, total_L = _curve_diff_data(pts)

        pts = np.asarray(pts, dtype=np.float64)
        tan = np.asarray(tan, dtype=np.float64)
        nor = np.asarray(nor, dtype=np.float64)
        s = np.asarray(s, dtype=np.float64)

        # Use the embedding's own endpoint metadata when available.  This keeps
        # the Williams/SIF tip frame exactly the same frame used by getGamma().
        if hasattr(embedding, "tip_tangent") and tan.shape[0] > 0:
            t_tip = np.asarray(embedding.tip_tangent, dtype=np.float64)
            t_tip = t_tip / max(float(np.linalg.norm(t_tip)), 1.0e-12)
            tan[-1] = t_tip
            nor[-1] = np.array([-t_tip[1], t_tip[0]], dtype=np.float64)

        return cls(
            control_points=control_points,
            curve_points=pts,
            curve_tangents=tan,
            curve_normals=nor,
            curve_s=s,
            total_length_value=float(total_L),
        )

    @property
    def points(self):
        """Alias for ``control_points``, the propagated control points."""
        return self.control_points

    @property
    def tip(self) -> np.ndarray:
        return np.asarray(self.curve_points[-1], dtype=np.float64)

    @property
    def last_tangent(self) -> np.ndarray:
        t = np.asarray(self.curve_tangents[-1], dtype=np.float64)
        nrm = np.linalg.norm(t)
        if nrm < 1.0e-14:
            return np.asarray([1.0, 0.0], dtype=np.float64)
        return t / nrm

    @property
    def last_normal(self) -> np.ndarray:
        t = self.last_tangent
        return np.asarray([-t[1], t[0]], dtype=np.float64)

    @property
    def last_beta(self) -> float:
        t = self.last_tangent
        return float(np.arctan2(t[1], t[0]))

    @property
    def total_length(self) -> float:
        return float(self.total_length_value)

    @property
    def last_segment_length(self) -> float:
        pts = np.asarray(self.control_points, dtype=np.float64)
        if pts.shape[0] < 2:
            return self.total_length
        return float(np.linalg.norm(pts[-1] - pts[-2]))

    def sample_faces(self, n_per_segment: int = 200, offset: float = 1.0e-4) -> Tuple[np.ndarray, np.ndarray]:
        """Sample upper and lower faces along the active crack curve."""
        total_L = self.total_length
        if total_L <= 1.0e-14:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

        n = max(int(n_per_segment), 2)
        s = np.linspace(0.02 * total_L, 0.98 * total_L, n)
        base, _, normals = _interp_curve(
            self.curve_points,
            self.curve_tangents,
            self.curve_normals,
            self.curve_s,
            s,
        )

        pts = np.vstack([base + offset * normals, base - offset * normals]).astype(np.float32)
        nrm = np.vstack([normals, -normals]).astype(np.float32)
        return pts, nrm

    def points_behind_tip(self, r_vals):
        """Points, tangents and normals at arc-distance ``r_vals`` behind the tip."""
        s = self.total_length - np.asarray(r_vals, dtype=np.float64)
        return _interp_curve(
            self.curve_points,
            self.curve_tangents,
            self.curve_normals,
            self.curve_s,
            s,
        )


def embedding_face_attenuation(model, offset: float) -> Optional[float]:
    """Fraction of the embedded (network) crack-face jump seen at +/- offset.

    The crack embedding decays like exp(-alpha * (d/ell)^p) away from the
    crack, so two-sided face sampling at normal distance ``offset`` measures
    only this fraction of the network's full face-limit jump.  Choose the
    offset such that alpha * (offset/ell)^p << 1.
    """
    emb = getattr(getattr(model, "model_uv", None), "extendAxis", None)
    alpha = getattr(emb, "decay_alpha", None)
    if alpha is None:
        return None
    p = float(getattr(emb, "decay_p", 1.0))
    ell = getattr(emb, "decay_ell", None)
    d = float(offset) / float(ell) if ell else float(offset)
    return float(np.exp(-float(alpha) * d ** p))


def cod_face_jump_sif(
    model,
    geom: "CrackGeometry",
    load_nd: float,
    r_vals: np.ndarray,
    offset: float,
    E: float,
    nu: float,
    mu: float,
    plane_strain: bool = True,
):
    """Displacement-extrapolation K1/K2 from two-sided crack-face sampling.

    The displacement is evaluated at x +/- offset * n(s) at arc distances
    ``r_vals`` behind the active tip, so the measured jump contains both the
    embedded network discontinuity (through the Gamma input) and the jump of
    the Williams enrichment across its branch cut (through atan2 in the tip
    frame).

    Offset selection: float32 coordinate noise << offset << min(r_vals), and
    embedding_face_attenuation(model, offset) close to 1.

    Returns (K1, K2, K1_samples, K2_samples, attenuation).
    """
    r_vals = np.asarray(r_vals, dtype=np.float64).reshape(-1)
    X_mid, _, n_mid = geom.points_behind_tip(r_vals)

    # Offset along the local normal so both samples stay on their own face
    # even behind a kink; the jump is decomposed in the tip frame (LEFM
    # convention, same frame as the interaction integral).
    upper = X_mid + float(offset) * n_mid
    lower = X_mid - float(offset) * n_mid
    X_pair = np.vstack([upper, lower]).astype(np.float32)

    pts = torch.tensor(X_pair, dtype=torch.float32, device=model.device)
    load_t = torch.tensor([[float(load_nd)]], dtype=torch.float32, device=model.device)
    with torch.no_grad():
        u, v = model.net_uv(pts[:, 0:1], pts[:, 1:2], load_t)
    uv = torch.cat([u, v], dim=1).detach().cpu().numpy().astype(np.float64)

    jump = uv[: len(r_vals)] - uv[len(r_vals):]
    tangent = np.asarray(geom.last_tangent, dtype=np.float64)
    normal = np.asarray(geom.last_normal, dtype=np.float64)
    delta_t = jump @ tangent
    delta_n = jump @ normal

    kappa = 3.0 - 4.0 * nu if plane_strain else (3.0 - nu) / (1.0 + nu)
    coeff = mu / (kappa + 1.0) * np.sqrt(2.0 * np.pi)
    K1_samples = coeff * delta_n / np.sqrt(r_vals)
    K2_samples = coeff * delta_t / np.sqrt(r_vals)

    fit_slice = slice(1, -1) if len(r_vals) >= 7 else slice(None)
    try:
        K1 = float(np.polyfit(r_vals[fit_slice], K1_samples[fit_slice], 1)[-1])
        K2 = float(np.polyfit(r_vals[fit_slice], K2_samples[fit_slice], 1)[-1])
    except Exception:
        K1 = float(np.median(K1_samples[fit_slice]))
        K2 = float(np.median(K2_samples[fit_slice]))
    if not np.isfinite(K1):
        K1 = 0.0
    if not np.isfinite(K2):
        K2 = 0.0

    attenuation = embedding_face_attenuation(model, offset)
    if attenuation is not None and attenuation < 0.90:
        print(
            f"[SIF-COD] WARNING: face offset {offset:.2e} attenuates the "
            f"embedded jump to {attenuation:.2f} of its face-limit value; "
            f"reduce the offset (need decay_alpha * offset << 1)."
        )
    return K1, K2, K1_samples, K2_samples, attenuation


def plot_sif_radius_sweep(outdir, tag, radii, K1_list, K2_list,
                          K1_ref=None, K2_ref=None, ref_label="COD"):
    """Radius sweep of the interaction integral: K against contour radius.

    For a converged field the curves are flat; drift with radius indicates an
    unreliable extraction (training residuals, a contour too close to the
    enrichment decay, or a missing thermal area term).
    """
    os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    ax.plot(radii, K1_list, "o-", color="tab:blue", label=r"$K_I$ (interaction)")
    ax.plot(radii, K2_list, "s-", color="tab:orange", label=r"$K_{II}$ (interaction)")
    if K1_ref is not None:
        ax.axhline(K1_ref, color="tab:blue", ls="--", lw=1.0, label=rf"$K_I$ ({ref_label})")
    if K2_ref is not None:
        ax.axhline(K2_ref, color="tab:orange", ls="--", lw=1.0, label=rf"$K_{{II}}$ ({ref_label})")
    ax.set_xlabel("contour radius")
    ax.set_ylabel("K")
    ax.set_title(f"Interaction-integral path independence ({tag})")
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"sif_radius_sweep_{tag}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_cod_extrapolation(outdir, tag, r_vals, K1_samples, K2_samples, K1, K2):
    """Displacement-extrapolation quality: K samples vs r with the r->0 fit.

    Clean samples lie on a straight line whose intercept is the reported K;
    curvature or scatter means the sampling window (r range, face offset) or
    the near-tip field needs attention.
    """
    os.makedirs(outdir, exist_ok=True)
    r_vals = np.asarray(r_vals, dtype=np.float64).reshape(-1)
    fit_slice = slice(1, -1) if len(r_vals) >= 7 else slice(None)

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    for samples, K, color, label in (
        (np.asarray(K1_samples).reshape(-1), K1, "tab:blue", "K_I"),
        (np.asarray(K2_samples).reshape(-1), K2, "tab:orange", "K_{II}"),
    ):
        ax.plot(r_vals, samples, "o", ms=4, color=color, label=rf"${label}$ samples")
        try:
            slope = float(np.polyfit(r_vals[fit_slice], samples[fit_slice], 1)[0])
        except Exception:
            slope = 0.0
        r_line = np.linspace(0.0, float(r_vals[-1]), 50)
        ax.plot(r_line, K + slope * r_line, "-", lw=1.2, color=color, alpha=0.7)
        ax.plot([0.0], [K], "*", ms=12, color=color)
    ax.set_xlabel("distance behind tip r")
    ax.set_ylabel("K estimate")
    ax.set_title(f"COD displacement extrapolation ({tag})")
    ax.grid(True, ls="--", alpha=0.5)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"sif_cod_extrapolation_{tag}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def energy_still_improving(history, window, rel_tol):
    """Whether the energy is still decreasing, judged on windowed means.

    Consecutive losses cannot be compared directly.  The integration points
    are redrawn every few hundred iterations, so the loss is a noisy estimate
    of the energy and each redraw shows up as a step.  Means over two
    consecutive windows average this out, provided each window spans a
    resampling interval.

    Returns True while the history is shorter than two windows, and while the
    relative improvement between the two most recent windows is at least
    ``rel_tol``.  A loss that has risen gives a negative improvement and
    therefore False: the energy is no longer decreasing.
    """
    if len(history) < 2 * window:
        return True
    prev = float(np.mean(history[-2 * window:-window]))
    curr = float(np.mean(history[-window:]))
    return (prev - curr) / max(abs(prev), 1.0e-30) >= rel_tol


class ThermoSolver2D(nn.Module):
    # Stopping rule for both Adam loops.  The iteration count an example passes
    # is a ceiling, and training stops once the energy has stopped decreasing.
    # The window matches the resampling interval, so that each mean covers a
    # whole point set.  Set stop_on_energy_plateau to False to run every Adam
    # loop for its full iteration count.
    stop_on_energy_plateau = True
    energy_plateau_window = 500
    energy_plateau_rel_tol = 1.0e-4
    energy_plateau_patience = 3
    energy_plateau_min_iters = 1000
    energy_plateau_check_every = 250

    def plateau_earliest_stop(self):
        """First iteration at which the stopping rule can fire.

        The rule is evaluated only after ``energy_plateau_min_iters``, at
        multiples of ``energy_plateau_check_every``, and once the history holds
        two windows; it must then hold for ``energy_plateau_patience``
        consecutive checks.  With the defaults this is iteration 1500, so a
        training call with a smaller budget runs to its budget, and
        announce_plateau_rule prints a line when a call is in this case.
        """
        first = max(self.energy_plateau_min_iters, 2 * self.energy_plateau_window)
        step = self.energy_plateau_check_every
        first = ((first + step - 1) // step) * step
        return first + (self.energy_plateau_patience - 1) * step

    def announce_plateau_rule(self, nIter_Adam, label):
        if not self.stop_on_energy_plateau:
            print(f"[{label}] energy-plateau stopping is off; running the full {nIter_Adam}")
            return
        first = self.plateau_earliest_stop()
        if nIter_Adam < first:
            print(f"[{label}] budget {nIter_Adam} is below the rule's earliest "
                  f"possible stop ({first}); this call runs to its budget")

    def _plateau_reached(self, history, stall, label, it):
        """Update the stall counter; True when training should stop."""
        if not self.stop_on_energy_plateau:
            return False, stall
        if (it + 1) < self.energy_plateau_min_iters:
            return False, stall
        if (it + 1) % self.energy_plateau_check_every:
            return False, stall
        if energy_still_improving(history, self.energy_plateau_window,
                                  self.energy_plateau_rel_tol):
            return False, 0
        stall += 1
        if stall < self.energy_plateau_patience:
            return False, stall
        print(f"[{label}] energy plateaued at Adam iter {it + 1}: "
              f"improvement over {self.energy_plateau_window} iters fell below "
              f"{self.energy_plateau_rel_tol:.1e} on "
              f"{self.energy_plateau_patience} consecutive checks")
        return True, stall
    """Staggered thermo-mechanical solver: thermal solve, then mechanical solve."""

    def __init__(self, model_params: Dict, model_uv: nn.Module, model_T: nn.Module, crack_geometry: CrackGeometry):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_uv = model_uv.to(self.device)
        self.model_T = model_T.to(self.device)
        self.crack_geometry = crack_geometry

        self.E = float(model_params.get("E", 340.0e3))
        self.nu = float(model_params.get("nu", 0.22))
        self.alpha = float(model_params.get("alpha", 8e-6))
        self.k = float(model_params.get("k", 300.0))
        self.rho = float(model_params.get("rho", 2.45e-6))
        self.c = float(model_params.get("c", 775.0))
        self.dt = float(model_params.get("dt", 2e-8))
        self.T_ref = float(model_params.get("T_ref", 300.0))
        self.U_ref = float(model_params.get("U_ref", 1e-4))
        self.DeltaT_ref = float(model_params.get("DeltaT_ref", 100.0))
        self.lb = np.asarray(model_params.get("lb", np.array([0.0, 0.0])), dtype=np.float64)
        self.ub = np.asarray(model_params.get("ub", np.array([1.0, 1.0])), dtype=np.float64)
        # Thermal crack representation.  The default thermal network has one
        # output and represents the crack through its Gamma input;
        # use_thermal_jump adds an explicit Gamma * T_jump output.
        self.use_thermal_jump = bool(model_params.get("use_thermal_jump", False))

        # Optional weak q.n consistency penalty on sampled crack faces.
        self.adiabatic_crack_weight = float(model_params.get("adiabatic_crack_weight", 1.0e-6))
        self.adiabatic_crack_samples = int(model_params.get("adiabatic_crack_samples", 150))
        self.crack_face_offset = float(model_params.get("crack_face_offset", 1.0e-4))
        # Densification of the integration points near the tip (see
        # generate_integration_points); off by default.
        self.tip_densify_radius = float(model_params.get("tip_densify_radius", 0.0))
        self.tip_densify_factor = int(model_params.get("tip_densify_factor", 1))

        # Plane strain with the reduced 2D stiffness takes the effective thermal
        # expansion coefficient (1 + nu) * alpha, since eps_zz = 0 blocks the
        # out-of-plane expansion.  The phase-field reference (fem/pf_fem.py,
        # alpha_mode ps3d) uses the same.  Off by default; each example sets it.
        self.plane_strain_thermal_alpha_correction = bool(
            model_params.get("plane_strain_thermal_alpha_correction", False))

        # Smoothed conductivity band around the crack (see
        # crack_conductivity_factor), off by default.
        self.use_crack_conductivity_degradation = bool(model_params.get("use_crack_conductivity_degradation", False))
        self.crack_band_width = float(model_params.get("crack_band_width", 0.01))
        self.crack_min_conductivity_ratio = float(model_params.get("crack_min_conductivity_ratio", 1.0e-6))
        self.plane_type = str(model_params.get("plane_type", "plane_strain"))

        # Default displacement: NN_u(x, y, Gamma) plus the asymptotic enrichment at
        # the tip.  The explicit Gamma*jump_u/v outputs are optional and off by
        # default.
        self.use_explicit_mechanical_jump = bool(model_params.get("use_explicit_mechanical_jump", False))

        # Asymptotic enrichment at the tip (Williams terms).
        self.use_tip_enrichment = bool(model_params.get("use_tip_enrichment", True))
        self.tip_enrichment_q = float(model_params.get("tip_enrichment_q", 1.0))
        self.tip_enrichment_decay_factor = float(model_params.get("tip_enrichment_decay_factor", 5.0))
        self.tip_enrichment_u_scale = float(model_params.get("tip_enrichment_u_scale", 1.0))

        # The trainable Williams amplitudes start at zero, the value at the unloaded
        # first load step, and warm starting carries the converged value from one
        # load step to the next.  For a run that starts at a finite load, pass
        # K_I_init = K_estimate/sqrt(2*pi).
        K1_init = float(model_params.get("K_I_init", 0.0))
        K2_init = float(model_params.get("K_II_init", 0.0))
        self.K_I = nn.Parameter(torch.tensor(K1_init, dtype=torch.float32, device=self.device))
        self.K_II = nn.Parameter(torch.tensor(K2_init, dtype=torch.float32, device=self.device))

        self.set_D(self.E, self.nu, self.plane_type)
        # (geometry object, face points, face normals): sample_faces runs a
        # per-point Python interpolation loop, so cache it per crack geometry
        # instead of rebuilding on every optimizer iteration.
        self._crack_face_cache: Optional[Tuple[object, torch.Tensor, torch.Tensor]] = None
        self.prev_thermal_net: Optional[nn.Module] = None
        self.thermal_adam_loss_history: List[float] = []
        self.thermal_lbfgs_loss_history: List[float] = []
        self.elastic_adam_loss_history: List[float] = []
        self.elastic_lbfgs_loss_history: List[float] = []

    def set_D(self, E: float, nu: float, plane_type: str = "plane_strain") -> None:
        if plane_type == "plane_stress":
            self.d11 = E / (1.0 - nu * nu)
            self.d12 = E * nu / (1.0 - nu * nu)
            self.G = E / (2.0 * (1.0 + nu))
            self.lamda = E * nu / (1.0 - nu * nu)
            self.mu = self.G
        elif plane_type == "plane_strain":
            self.d11 = E * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
            self.d12 = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
            self.G = E / (2.0 * (1.0 + nu))
            self.lamda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
            self.mu = self.G
        else:
            raise ValueError("plane_type must be 'plane_stress' or 'plane_strain'.")


    def mechanical_parameters(self) -> List[nn.Parameter]:
        """Return mechanical parameters used by the elastic optimiser."""
        params = list(self.model_uv.parameters())
        if self.use_tip_enrichment:
            params += [self.K_I, self.K_II]
        return params

    def trainable_sif_values(self) -> Tuple[float, float]:
        """Return trainable Williams K values."""
        if self.use_tip_enrichment:
            return float(self.K_I.detach().cpu()), float(self.K_II.detach().cpu())
        return float("nan"), float("nan")

    def current_crack_length(self) -> float:
        """Total length of the active crack curve."""
        try:
            return float(self.crack_geometry.total_length)
        except Exception:
            return float(np.linalg.norm(self.ub - self.lb))

    def current_crack_tip_state(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return active crack tip, tangent angle and crack length."""
        try:
            tip_np = np.asarray(self.crack_geometry.tip, dtype=np.float64)
            beta = float(self.crack_geometry.last_beta)
            length = float(self.crack_geometry.total_length)
        except Exception:
            tip_np = np.asarray([self.ub[0], 0.5 * (self.lb[1] + self.ub[1])], dtype=np.float64)
            beta = 0.0
            length = np.linalg.norm(self.ub - self.lb)

        tip = torch.tensor(tip_np.reshape(1, 2), dtype=torch.float32, device=self.device)
        beta_t = torch.tensor(beta, dtype=torch.float32, device=self.device)
        length_t = torch.tensor(max(float(length), 1.0e-12), dtype=torch.float32, device=self.device)
        return tip, beta_t, length_t

    def crack_tip_asymptotic_displacement(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return Williams near-tip displacement in physical units."""
        if not self.use_tip_enrichment:
            return torch.zeros_like(x), torch.zeros_like(y)

        tip, beta, crack_len = self.current_crack_tip_state()
        c = torch.cos(beta)
        s = torch.sin(beta)

        rel_x = x - tip[:, 0:1]
        rel_y = y - tip[:, 1:2]
        x1 = c * rel_x + s * rel_y
        x2 = -s * rel_x + c * rel_y
        r = torch.sqrt(torch.clamp(x1 ** 2 + x2 ** 2, min=1.0e-18))

        theta = torch.atan2(x2, x1)
        if str(self.plane_type).lower() == "plane_strain":
            kappa = 3.0 - 4.0 * self.nu
        else:
            kappa = (3.0 - self.nu) / (1.0 + self.nu)

        root = torch.sqrt(r / (2.0 * np.pi))
        mu = self.mu

        u_I = (self.K_I / (2.0 * mu)) * root * torch.cos(theta / 2.0) * (
            kappa - 1.0 + 2.0 * torch.sin(theta / 2.0) ** 2
        )
        v_I = (self.K_I / (2.0 * mu)) * root * torch.sin(theta / 2.0) * (
            kappa + 1.0 - 2.0 * torch.cos(theta / 2.0) ** 2
        )
        u_II = (self.K_II / (2.0 * mu)) * root * torch.sin(theta / 2.0) * (
            2.0 + kappa + torch.cos(theta)
        )
        v_II = (self.K_II / (2.0 * mu)) * root * torch.cos(theta / 2.0) * (
            2.0 - kappa - torch.cos(theta)
        )

        u_local = u_I + u_II
        v_local = v_I + v_II
        u_global = u_local * c - v_local * s
        v_global = u_local * s + v_local * c

        domain_width = float(max(self.ub[0] - self.lb[0], 1.0e-12))
        decay = torch.exp(
            -self.tip_enrichment_decay_factor
            * domain_width
            / crack_len
            * torch.pow(r, self.tip_enrichment_q)
        )
        return self.tip_enrichment_u_scale * u_global * decay, self.tip_enrichment_u_scale * v_global * decay


    def variable(self, data: np.ndarray | torch.Tensor, requires_grad: bool = True) -> torch.Tensor:
        if not torch.is_tensor(data):
            data = torch.tensor(data, dtype=torch.float32)
        return data.to(self.device).float().requires_grad_(requires_grad)

    def generate_integration_points(
        self,
        vertex: np.ndarray,
        num_points_per_elem: int = 5,
        method: str = "mc",
        gauss_order: int = 2,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Element-wise integration points.  Thermal training should use method='mc'.

        With ``tip_densify_radius``/``tip_densify_factor`` set on the model, MC
        elements whose centre lies within that radius of the active crack tip
        receive ``factor`` times as many samples.  The mesh is untouched, and since
        each point carries weight ``area / n`` every element still integrates to
        its own area.
        """
        xs: List[np.ndarray] = []
        ys: List[np.ndarray] = []
        ws: List[np.ndarray] = []
        elem_ids: List[np.ndarray] = []
        method = method.lower()

        densify_factor = int(getattr(self, "tip_densify_factor", 1))
        densify_radius = float(getattr(self, "tip_densify_radius", 0.0))
        densify_r2 = densify_radius ** 2
        tip_xy = None
        if densify_factor > 1 and densify_radius > 0.0 and self.crack_geometry is not None:
            try:
                tip_xy = np.asarray(self.crack_geometry.tip, dtype=np.float64)
            except Exception:
                tip_xy = None
        for eid, elem in enumerate(vertex):
            x0, x1, y0, y1 = elem[:4]
            area = (x1 - x0) * (y1 - y0)
            if method == "gaussian":
                pts_1d, w_1d = np.polynomial.legendre.leggauss(gauss_order)
                xg = 0.5 * (x1 - x0) * pts_1d + 0.5 * (x1 + x0)
                yg = 0.5 * (y1 - y0) * pts_1d + 0.5 * (y1 + y0)
                xx, yy = np.meshgrid(xg, yg)
                ww = np.outer(w_1d, w_1d).reshape(-1, 1) * area / 4.0
                x_flat = xx.reshape(-1, 1)
                y_flat = yy.reshape(-1, 1)
            else:
                n = int(num_points_per_elem)
                if tip_xy is not None and densify_factor > 1:
                    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
                    if (cx - tip_xy[0]) ** 2 + (cy - tip_xy[1]) ** 2 <= densify_r2:
                        n *= densify_factor
                if LatinHypercube is not None:
                    sampler = LatinHypercube(d=2)
                    rnd = sampler.random(n)
                else:
                    rnd = np.random.rand(n, 2)
                x_flat = (x0 + (x1 - x0) * rnd[:, 0:1])
                y_flat = (y0 + (y1 - y0) * rnd[:, 1:2])
                ww = np.full((n, 1), area / n, dtype=np.float64)
            xs.append(x_flat)
            ys.append(y_flat)
            ws.append(ww)
            elem_ids.append(np.full((len(x_flat), 1), eid, dtype=np.float64))
        return np.vstack(xs), np.vstack(ys), np.vstack(ws), np.vstack(elem_ids)

    def xy_tensor(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, y], dim=1)

    def net_thermal(self, x: torch.Tensor, y: torch.Tensor, T_delta_nd: torch.Tensor) -> torch.Tensor:
        """Physical temperature: NN_T(x, y, Gamma) with hard top/bottom lifting."""
        xy = self.xy_tensor(x, y)
        T_out = self.model_T(xy)

        if self.use_thermal_jump and T_out.shape[1] >= 2:
            try:
                gamma = self.model_T.extendAxis.getGamma(xy)
                if gamma.dim() == 1:
                    gamma = gamma.unsqueeze(-1)
                gamma = gamma[:, 0:1]
            except Exception:
                gamma = torch.zeros_like(T_out[:, 0:1])
            T_hat = T_out[:, 0:1] + gamma * T_out[:, 1:2]
        else:
            T_hat = T_out[:, 0:1]

        theta_nd = y * (1.0 - y) * T_hat + y * T_delta_nd
        return self.T_ref + self.DeltaT_ref * theta_nd

    def net_uv(self, x: torch.Tensor, y: torch.Tensor, v_delta_nd: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Physical displacement: NN_u(x, y, Gamma) plus near-tip enrichment."""
        xy = self.xy_tensor(x, y)
        uv_hat = self.model_uv(xy)
        u_hat = uv_hat[:, 0:1]
        v_hat = uv_hat[:, 1:2]

        if self.use_explicit_mechanical_jump and uv_hat.shape[1] >= 4:
            try:
                gamma = self.model_uv.extendAxis.getGamma(xy)
                if gamma.dim() == 1:
                    gamma = gamma.unsqueeze(-1)
                gamma = gamma[:, 0:1]
            except Exception:
                gamma = torch.zeros_like(u_hat)
            u_hat = u_hat + gamma * uv_hat[:, 2:3]
            v_hat = v_hat + gamma * uv_hat[:, 3:4]

        u_tip, v_tip = self.crack_tip_asymptotic_displacement(x, y)

        # Hard boundary conditions for the [0,1]x[0,1] single-edge notched
        # geometry: u = v = 0 on y = 0, u = 0 and v = v_delta on y = 1.
        u_lift = y * (1.0 - y)
        v_lift = y * (1.0 - y)

        u_nd = u_lift * u_hat + u_lift * (u_tip / self.U_ref)
        v_nd = y * (y - 1.0) * v_hat + y * v_delta_nd + v_lift * (v_tip / self.U_ref)
        return self.U_ref * u_nd, self.U_ref * v_nd


    def compute_strain(self, u: torch.Tensor, v: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        e_xx = u_x
        e_yy = v_y
        e_xy = 0.5 * (u_y + v_x)
        return e_xx, e_yy, e_xy, u_x, u_y, v_x, v_y

    def thermal_strain_coefficient(self, alpha=None, nu=None):
        """CTE entering the 2D reduced constitutive law.

        Plain alpha by default; with plane_strain_thermal_alpha_correction=True
        and plane strain it becomes alpha*(1+nu), the exact plane-strain
        effective CTE (see the constructor).  In plane stress plain alpha is
        exact.  Accepts scalars or tensors (graded fields).
        """
        a = self.alpha if alpha is None else alpha
        if not self.plane_strain_thermal_alpha_correction:
            return a
        if str(self.plane_type).lower() != "plane_strain":
            return a
        n = self.nu if nu is None else nu
        return a * (1.0 + n)

    def constitutive_thermal(
        self,
        e_xx: torch.Tensor,
        e_yy: torch.Tensor,
        e_xy: torch.Tensor,
        T: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        eps_th = self.thermal_strain_coefficient() * (T - self.T_ref)
        ee_xx = e_xx - eps_th
        ee_yy = e_yy - eps_th
        ee_xy = e_xy
        tr_e = ee_xx + ee_yy
        sx = self.lamda * tr_e + 2.0 * self.mu * ee_xx
        sy = self.lamda * tr_e + 2.0 * self.mu * ee_yy
        sxy = 2.0 * self.mu * ee_xy
        return sx, sy, sxy, ee_xx, ee_yy, ee_xy

    def crack_conductivity_factor(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Smoothed conductivity band around the crack.

        Off by default (use_crack_conductivity_degradation=False), since the crack
        embedding of the thermal network carries the discontinuity; it then
        returns ones() and the thermal energy density is unaffected.
        """
        if not self.use_crack_conductivity_degradation or self.crack_band_width <= 0.0:
            return torch.ones_like(x)

        pts = list(self.crack_geometry.points)
        if len(pts) < 2:
            return torch.ones_like(x)

        xy = torch.cat([x, y], dim=1)
        dist_min = None
        for p0_np, p1_np in zip(pts[:-1], pts[1:]):
            p0 = torch.tensor(p0_np, dtype=xy.dtype, device=xy.device).view(1, 2)
            p1 = torch.tensor(p1_np, dtype=xy.dtype, device=xy.device).view(1, 2)
            seg = p1 - p0
            seg_len2 = torch.sum(seg * seg) + 1.0e-30
            t = torch.sum((xy - p0) * seg, dim=1, keepdim=True) / seg_len2
            t = torch.clamp(t, 0.0, 1.0)
            proj = p0 + t * seg
            dist = torch.sqrt(torch.sum((xy - proj) ** 2, dim=1, keepdim=True) + 1.0e-30)
            dist_min = dist if dist_min is None else torch.minimum(dist_min, dist)

        band = max(self.crack_band_width, 1.0e-12)
        chi = torch.exp(- (dist_min / band) ** 4)
        eta = self.crack_min_conductivity_ratio
        return eta + (1.0 - eta) * (1.0 - chi)

    def thermal_energy_density(self, x: torch.Tensor, y: torch.Tensor, T_delta_nd: torch.Tensor, Temp_n: torch.Tensor) -> torch.Tensor:
        """Backward-Euler thermal energy density (one-way coupling).

            w_T = 0.5*(rho c / dt)*(T - T_n)^2  +  0.5*k*g_crack(x,y)*|grad T|^2

        The coupling is one-way (thermal -> mechanical through eps_th), so there is
        no mechanical source term.  g_crack = 1 by default, since the crack
        embedding of the thermal network carries the discontinuity; the smoothed
        band is enabled by use_crack_conductivity_degradation=True.
        """
        x.requires_grad_(True)
        y.requires_grad_(True)
        T = self.net_thermal(x, y, T_delta_nd)
        T_x = torch.autograd.grad(T, x, grad_outputs=torch.ones_like(T), create_graph=True)[0]
        T_y = torch.autograd.grad(T, y, grad_outputs=torch.ones_like(T), create_graph=True)[0]
        w_store = 0.5 * (self.rho * self.c / self.dt) * (T - Temp_n) ** 2
        gT = self.crack_conductivity_factor(x, y)
        w_cond = 0.5 * self.k * gT * (T_x**2 + T_y**2)
        return w_store + w_cond

    def _crack_face_samples(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cached crack-face sample points/normals for the current geometry."""
        cache = self._crack_face_cache
        if cache is not None and cache[0] is self.crack_geometry:
            return cache[1], cache[2]
        pts_np, normals_np = self.crack_geometry.sample_faces(self.adiabatic_crack_samples, self.crack_face_offset)
        pts = torch.tensor(pts_np, dtype=torch.float32, device=self.device)
        normals = torch.tensor(normals_np, dtype=torch.float32, device=self.device)
        self._crack_face_cache = (self.crack_geometry, pts, normals)
        return pts, normals

    def adiabatic_crack_loss(self, T_delta_nd: torch.Tensor) -> torch.Tensor:
        """Soft consistency penalty on q.n at sampled crack-face points.

        The crack embedding of the thermal network carries the insulated crack
        faces; this penalty is a small regularizer, with a default weight of 1e-6.
        """
        if self.adiabatic_crack_weight <= 0.0:
            return torch.zeros((), device=self.device)
        pts, normals = self._crack_face_samples()
        if pts.shape[0] == 0:
            return torch.zeros((), device=self.device)
        x = pts[:, 0:1].clone().detach().requires_grad_(True)
        y = pts[:, 1:2].clone().detach().requires_grad_(True)
        T = self.net_thermal(x, y, T_delta_nd)
        T_x = torch.autograd.grad(T, x, grad_outputs=torch.ones_like(T), create_graph=True)[0]
        T_y = torch.autograd.grad(T, y, grad_outputs=torch.ones_like(T), create_graph=True)[0]
        qn = -self.k * (T_x * normals[:, 0:1] + T_y * normals[:, 1:2])
        return self.adiabatic_crack_weight * torch.mean(qn**2)

    def precompute_thermal(self, x: torch.Tensor, y: torch.Tensor, T_delta_nd: torch.Tensor) -> torch.Tensor:
        """Evaluate T once for a fixed integration batch.

        The coupling is one-way and the elastic energy uses T algebraically
        (through eps_th, no grad-T), so a detached precomputed T is exactly
        equivalent inside the elastic solve while skipping the thermal-network
        forward/backward on every optimizer iteration.
        """
        with torch.no_grad():
            return self.net_thermal(x, y, T_delta_nd).detach()

    def elastic_energy_density(self, x: torch.Tensor, y: torch.Tensor, T_delta_nd: torch.Tensor, v_delta_nd: torch.Tensor, T_precomputed: Optional[torch.Tensor] = None) -> torch.Tensor:
        x.requires_grad_(True)
        y.requires_grad_(True)
        T = self.net_thermal(x, y, T_delta_nd) if T_precomputed is None else T_precomputed
        u, v = self.net_uv(x, y, v_delta_nd)
        e_xx, e_yy, e_xy, *_ = self.compute_strain(u, v, x, y)
        sx, sy, sxy, ee_xx, ee_yy, ee_xy = self.constitutive_thermal(e_xx, e_yy, e_xy, T)
        return 0.5 * (ee_xx * sx + ee_yy * sy + 2.0 * ee_xy * sxy)

    def compute_elastic_residual(self, x: torch.Tensor, y: torch.Tensor, T_delta_nd: torch.Tensor, v_delta_nd: torch.Tensor) -> torch.Tensor:
        x.requires_grad_(True)
        y.requires_grad_(True)
        T = self.net_thermal(x, y, T_delta_nd)
        u, v = self.net_uv(x, y, v_delta_nd)
        e_xx, e_yy, e_xy, *_ = self.compute_strain(u, v, x, y)
        sx, sy, sxy, *_ = self.constitutive_thermal(e_xx, e_yy, e_xy, T)
        dsx_dx = torch.autograd.grad(sx, x, grad_outputs=torch.ones_like(sx), create_graph=True)[0]
        dsxy_dy = torch.autograd.grad(sxy, y, grad_outputs=torch.ones_like(sxy), create_graph=True)[0]
        dsxy_dx = torch.autograd.grad(sxy, x, grad_outputs=torch.ones_like(sxy), create_graph=True)[0]
        dsy_dy = torch.autograd.grad(sy, y, grad_outputs=torch.ones_like(sy), create_graph=True)[0]
        return torch.sqrt((dsx_dx + dsxy_dy) ** 2 + (dsxy_dx + dsy_dy) ** 2 + 1.0e-30)

    def train_thermal(
        self,
        X_f: np.ndarray,
        T_delta_nd: float,
        Temp_n_f: np.ndarray,
        nIter_Adam: int,
        lr_Adam: float,
        nIter_LBFGS: int,
        lr_LBFGS: float,
        nIter_log: int,
        step: int,
        vertex: np.ndarray,
        num_integration_points: int,
        resample_every_iters: int = 0,
        T_delta_nd_prev: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.model_T.train()
        self.model_uv.eval()
        self.thermal_adam_loss_history = []
        self.thermal_lbfgs_loss_history = []
        T_delta_t = torch.tensor([[T_delta_nd]], dtype=torch.float32, device=self.device)

        def make_tensors(X: np.ndarray, Temp_n: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            x = self.variable(X[:, 0:1], True)
            y = self.variable(X[:, 1:2], True)
            w = self.variable(X[:, 2:3], False)
            Tn = self.variable(Temp_n.reshape(-1, 1), False)
            return x, y, w, Tn

        x_f, y_f, wt_f, Temp_n = make_tensors(X_f, Temp_n_f)
        optimizer = optim.Adam(self.model_T.parameters(), lr=lr_Adam)
        print("Training thermal network")
        self.announce_plateau_rule(nIter_Adam, "thermal")
        t0 = time.time()
        stall = 0
        for it in range(nIter_Adam):
            optimizer.zero_grad()
            energy_T = self.thermal_energy_density(x_f, y_f, T_delta_t, Temp_n)
            loss = torch.sum(energy_T * wt_f) + self.adiabatic_crack_loss(T_delta_t)
            if not torch.isfinite(loss):
                print(f"[thermal] non-finite loss at Adam iter {it}; stop this thermal training call before optimizer update.")
                break
            loss.backward()
            optimizer.step()
            self.thermal_adam_loss_history.append(float(loss.detach().cpu()))
            if it % nIter_log == 0:
                print(f"Thermal Adam iter: {it:5d}, Loss: {loss.item():.3e}, Time: {time.time() - t0:.2f}")
                t0 = time.time()
            done, stall = self._plateau_reached(self.thermal_adam_loss_history, stall, "thermal", it)
            if done:
                break
            if resample_every_iters > 0 and (it + 1) % resample_every_iters == 0 and (it + 1) < nIter_Adam:
                x_np, y_np, wt_np, elem_np = self.generate_integration_points(vertex, num_integration_points, method="mc")
                X_f = np.column_stack([x_np, y_np, wt_np, elem_np])
                Temp_n_f = self.eval_prev_thermal(X_f[:, 0:2], T_delta_nd_prev)
                x_f, y_f, wt_f, Temp_n = make_tensors(X_f, Temp_n_f)
                print(
                    f"[thermal-resample] Adam iter {it + 1}: "
                    f"regenerated {X_f.shape[0]} MC integration points"
                )

        if nIter_LBFGS > 0:
            # L-BFGS on the current integration batch, which stays fixed.
            lbfgs = optim.LBFGS(
                self.model_T.parameters(),
                max_iter=nIter_LBFGS,
                max_eval=nIter_LBFGS,
                tolerance_grad=1.0e-9,
                tolerance_change=1.0e-9,
                history_size=50,
                line_search_fn="strong_wolfe",
                lr=lr_LBFGS,
            )

            def closure() -> torch.Tensor:
                lbfgs.zero_grad()
                energy_T = self.thermal_energy_density(x_f, y_f, T_delta_t, Temp_n)
                loss = torch.sum(energy_T * wt_f) + self.adiabatic_crack_loss(T_delta_t)
                if not torch.isfinite(loss):
                    print("[thermal] non-finite loss inside LBFGS closure; returning without backward.")
                    return loss
                loss.backward()
                self.thermal_lbfgs_loss_history.append(float(loss.detach().cpu()))
                if len(self.thermal_lbfgs_loss_history) % nIter_log == 0:
                    print(f"Thermal LBFGS iter: {len(self.thermal_lbfgs_loss_history):5d}, Loss: {loss.item():.3e}")
                return loss

            lbfgs.step(closure)
        return X_f, Temp_n_f

    def train_elastic(
        self,
        X_f: np.ndarray,
        T_delta_nd: float,
        v_delta_nd: float,
        nIter_Adam: int,
        lr_Adam: float,
        nIter_LBFGS: int,
        lr_LBFGS: float,
        nIter_log: int,
        vertex: Optional[np.ndarray] = None,
        num_integration_points: int = 5,
        resample_every_iters: int = 0,
    ) -> np.ndarray:
        self.model_T.eval()
        self.model_uv.train()
        self.elastic_adam_loss_history = []
        self.elastic_lbfgs_loss_history = []
        T_delta_t = torch.tensor([[T_delta_nd]], dtype=torch.float32, device=self.device)
        v_delta_t = torch.tensor([[v_delta_nd]], dtype=torch.float32, device=self.device)

        def make_tensors(X: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return self.variable(X[:, 0:1], True), self.variable(X[:, 1:2], True), self.variable(X[:, 2:3], False)

        x_f, y_f, wt_f = make_tensors(X_f)
        T_pre = self.precompute_thermal(x_f, y_f, T_delta_t)
        optimizer = optim.Adam(self.mechanical_parameters(), lr=lr_Adam)
        print("Training elastic network")
        self.announce_plateau_rule(nIter_Adam, "elastic")
        t0 = time.time()
        stall = 0
        for it in range(nIter_Adam):
            optimizer.zero_grad()
            energy_e = self.elastic_energy_density(x_f, y_f, T_delta_t, v_delta_t, T_precomputed=T_pre)
            loss = torch.sum(energy_e * wt_f)
            if not torch.isfinite(loss):
                print(f"[elastic] non-finite loss at Adam iter {it}; stop this elastic training call before optimizer update.")
                break
            loss.backward()
            optimizer.step()
            self.elastic_adam_loss_history.append(float(loss.detach().cpu()))
            if it % nIter_log == 0:
                print(f"Elastic Adam iter: {it:5d}, Loss: {loss.item():.3e}, Time: {time.time() - t0:.2f}")
                t0 = time.time()
            done, stall = self._plateau_reached(self.elastic_adam_loss_history, stall, "elastic", it)
            if done:
                break
            if vertex is not None and resample_every_iters > 0 and (it + 1) % resample_every_iters == 0 and (it + 1) < nIter_Adam:
                x_np, y_np, wt_np, elem_np = self.generate_integration_points(vertex, num_integration_points, method="mc")
                X_f = np.column_stack([x_np, y_np, wt_np, elem_np])
                x_f, y_f, wt_f = make_tensors(X_f)
                T_pre = self.precompute_thermal(x_f, y_f, T_delta_t)
                print(
                    f"[elastic-resample] Adam iter {it + 1}: "
                    f"regenerated {X_f.shape[0]} MC integration points"
                )

        if nIter_LBFGS > 0:
            lbfgs = optim.LBFGS(
                self.mechanical_parameters(),
                max_iter=nIter_LBFGS,
                max_eval=nIter_LBFGS,
                tolerance_grad=1.0e-9,
                tolerance_change=1.0e-9,
                history_size=50,
                line_search_fn="strong_wolfe",
                lr=lr_LBFGS,
            )

            def closure() -> torch.Tensor:
                lbfgs.zero_grad()
                energy_e = self.elastic_energy_density(x_f, y_f, T_delta_t, v_delta_t, T_precomputed=T_pre)
                loss = torch.sum(energy_e * wt_f)
                if not torch.isfinite(loss):
                    print("[elastic] non-finite loss inside LBFGS closure; returning without backward.")
                    return loss
                loss.backward()
                self.elastic_lbfgs_loss_history.append(float(loss.detach().cpu()))
                if len(self.elastic_lbfgs_loss_history) % nIter_log == 0:
                    print(f"Elastic LBFGS iter: {len(self.elastic_lbfgs_loss_history):5d}, Loss: {loss.item():.3e}")
                return loss

            lbfgs.step(closure)
        return X_f

    def snapshot_thermal_net(self) -> None:
        self.prev_thermal_net = copy.deepcopy(self.model_T)
        self.prev_thermal_net.eval()
        for p in self.prev_thermal_net.parameters():
            p.requires_grad_(False)

    def eval_prev_thermal(self, X: np.ndarray, T_delta_nd_prev: float) -> np.ndarray:
        if self.prev_thermal_net is None:
            return self.T_ref * np.ones((X.shape[0], 1), dtype=np.float32)
        with torch.no_grad():
            pts = torch.tensor(X, dtype=torch.float32, device=self.device)
            x = pts[:, 0:1]
            y = pts[:, 1:2]
            T_delta = torch.tensor([[T_delta_nd_prev]], dtype=torch.float32, device=self.device)
            old_net = self.model_T
            self.model_T = self.prev_thermal_net
            T = self.net_thermal(x, y, T_delta)
            self.model_T = old_net
        return T.detach().cpu().numpy().reshape(-1, 1)

    def predict_thermal(self, X: np.ndarray, T_delta_nd: float) -> np.ndarray:
        self.model_T.eval()
        with torch.no_grad():
            pts = torch.tensor(X, dtype=torch.float32, device=self.device)
            x = pts[:, 0:1]
            y = pts[:, 1:2]
            T_delta = torch.tensor([[T_delta_nd]], dtype=torch.float32, device=self.device)
            T = self.net_thermal(x, y, T_delta)
        return T.detach().cpu().numpy()

    def predict_fields(self, X: np.ndarray, T_delta_nd: float, v_delta_nd: float) -> Dict[str, np.ndarray]:
        self.model_T.eval()
        self.model_uv.eval()
        pts = torch.tensor(X, dtype=torch.float32, device=self.device)
        x = pts[:, 0:1].clone().detach().requires_grad_(True)
        y = pts[:, 1:2].clone().detach().requires_grad_(True)
        T_delta = torch.tensor([[T_delta_nd]], dtype=torch.float32, device=self.device)
        v_delta = torch.tensor([[v_delta_nd]], dtype=torch.float32, device=self.device)
        T = self.net_thermal(x, y, T_delta)
        u, v = self.net_uv(x, y, v_delta)
        e_xx, e_yy, e_xy, *_ = self.compute_strain(u, v, x, y)
        sx, sy, sxy, ee_xx, ee_yy, ee_xy = self.constitutive_thermal(e_xx, e_yy, e_xy, T)
        vm = torch.sqrt(sx**2 - sx * sy + sy**2 + 3.0 * sxy**2 + 1e-30)
        eps_th = self.thermal_strain_coefficient() * (T - self.T_ref)
        xy = torch.cat([x, y], dim=1)
        try:
            gamma = self.model_uv.extendAxis.getGamma(xy)
        except Exception:
            gamma = torch.zeros_like(x)

        # Split of T into its smooth and jump parts.  The default thermal mode
        # has no explicit Gamma*T_jump term, so T_jump_contrib_K is zero.
        with torch.no_grad():
            T_out = self.model_T(xy)
            T_smooth_raw = T_out[:, 0:1]
            if self.use_thermal_jump and T_out.shape[1] >= 2:
                T_jump_raw = T_out[:, 1:2]
            else:
                T_jump_raw = torch.zeros_like(T_smooth_raw)
            env = (y * (1.0 - y)) * self.DeltaT_ref
            T_smooth_phys = env * T_smooth_raw
            T_jump_phys = env * gamma * T_jump_raw

        K_I_diag, K_II_diag = self.trainable_sif_values()
        return {
            "T": T.detach().cpu().numpy(),
            "u": u.detach().cpu().numpy(),
            "v": v.detach().cpu().numpy(),
            "sx": sx.detach().cpu().numpy(),
            "sy": sy.detach().cpu().numpy(),
            "sxy": sxy.detach().cpu().numpy(),
            "von_mises": vm.detach().cpu().numpy(),
            "eps_th": eps_th.detach().cpu().numpy(),
            "ee_xx": ee_xx.detach().cpu().numpy(),
            "ee_yy": ee_yy.detach().cpu().numpy(),
            "ee_xy": ee_xy.detach().cpu().numpy(),
            "K_I": np.array([[K_I_diag]], dtype=np.float64),
            "K_II": np.array([[K_II_diag]], dtype=np.float64),
            "gamma": gamma.detach().cpu().numpy(),
            "T_smooth_contrib_K": T_smooth_phys.detach().cpu().numpy(),
            "T_jump_contrib_K": T_jump_phys.detach().cpu().numpy(),
        }

    def reaction_force(self, X_f: np.ndarray, T_delta_nd: float, v_delta_nd: float) -> float:
        x = self.variable(X_f[:, 0:1], True)
        y = self.variable(X_f[:, 1:2], True)
        wt = self.variable(X_f[:, 2:3], False)
        T_delta = torch.tensor([[T_delta_nd]], dtype=torch.float32, device=self.device)
        v_delta = torch.tensor([[v_delta_nd]], dtype=torch.float32, device=self.device, requires_grad=True)
        energy_e = self.elastic_energy_density(x, y, T_delta, v_delta)
        Pi = torch.sum(energy_e * wt)
        dPi_dvnd = torch.autograd.grad(Pi, v_delta, create_graph=False, retain_graph=False)[0]
        return float(dPi_dvnd.detach().cpu().item() / self.U_ref)

    def save_model(
        self,
        step: int,
        fd_graph: np.ndarray,
        Temp_n_f: np.ndarray,
        X_f: np.ndarray,
        vertex: np.ndarray,
        parent_indices: np.ndarray,
        step_dir: str,
        mesh_policy: str = "fixed",
        mesh_meta: Optional[Dict] = None,
    ) -> None:
        os.makedirs(step_dir, exist_ok=True)
        payload = {
            "step": step,
            "thermal_state_dict": self.model_T.state_dict(),
            "mechanical_state_dict": self.model_uv.state_dict(),
            "fdGraph": fd_graph,
            "T_n_f": Temp_n_f,
            "X_f": X_f,
            "vertex": vertex,
            "parent_indices": parent_indices,
            "mesh_policy": mesh_policy,
            "mesh_meta": mesh_meta or {},
            "thermal_adam_loss_history": self.thermal_adam_loss_history,
            "thermal_lbfgs_loss_history": self.thermal_lbfgs_loss_history,
            "elastic_adam_loss_history": self.elastic_adam_loss_history,
            "elastic_lbfgs_loss_history": self.elastic_lbfgs_loss_history,
            "crack_points": np.asarray(self.crack_geometry.points, dtype=np.float64),
            "crack_curve_points": np.asarray(self.crack_geometry.curve_points, dtype=np.float64),
            "K_I": float(self.K_I.detach().cpu()),
            "K_II": float(self.K_II.detach().cpu()),
            "use_tip_enrichment": self.use_tip_enrichment,
            "use_thermal_jump": self.use_thermal_jump,
            "use_explicit_mechanical_jump": self.use_explicit_mechanical_jump,
            # States of the python, numpy, torch and cuda generators, so that
            # a resumed run continues the random streams of an uninterrupted
            # one.  The Latin hypercube sampler of generate_integration_points
            # creates its own unseeded generator, which these states do not
            # cover.
            "rng_python": random.getstate(),
            "rng_numpy": np.random.get_state(),
            "rng_torch": torch.get_rng_state(),
            "rng_cuda": (torch.cuda.get_rng_state_all()
                         if torch.cuda.is_available() else None),
        }
        torch.save(payload, os.path.join(step_dir, f"checkpoint_step_{step}.pt"))
