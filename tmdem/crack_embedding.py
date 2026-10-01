"""
Crack embedding: the crack polyline and the scalar function Gamma(x, y) that
both networks receive as an input.

The crack is represented by the propagated control points, with one window
along the arc length, extended past the tip by the endpoint overlap, and one
normal decay.  The ``Embedding`` base class and the ``extendAxisNet`` wrapper
follow the isothermal formulation of Zhao and Shao (2025).

Author: Han Zhang (han.zhang7@unsw.edu.au)
"""


from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn


class Embedding:
    """Base class for scalar crack functions Gamma(x, y).

    Sign convention: Gamma > 0 on the negative-normal side of the crack, where
    the signed distance to the crack line is negative.  Post-processing that
    assigns the two crack faces from the sign of Gamma follows this convention.
    """

    def __init__(self, HeavisideZero=0):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.set_Heaviside(HeavisideZero)

    def getGamma(self, xy) -> torch.Tensor: ...

    def set_Heaviside(self, HeavisideZero):

        def HeavisideZero0(x):
            return torch.sign(torch.relu(x))  # 0 for x <= 0, 1 for x > 0

        def HeavisideZero1(x):
            return 1 - torch.sign(torch.relu(-x))  # 1 for x >= 0, 0 for x < 0

        Heavisides = [HeavisideZero0, HeavisideZero1]
        self.Heaviside = Heavisides[HeavisideZero]

    def sign(self, x):
        return (self.Heaviside(x) - 0.5) * 2

    def zero(self, x, y):
        return torch.zeros_like(x)

    def one(self, x, y):
        return torch.ones_like(x)

    def neg_one(self, x, y):
        return -self.one(x, y)


class extendAxisNet(nn.Module):
    """Wrap a network so it receives [x, y, Gamma(x, y)] as input."""

    def __init__(self, net: nn.Module, extendAxis: Embedding) -> None:
        super().__init__()
        self.net = net
        self.extendAxis = extendAxis

    def forward(self, xy):
        Gamma = self.extendAxis.getGamma(xy)
        axis = torch.cat((xy, Gamma), dim=1)
        return self.net(axis)

    def infer(self, axis):
        return self.net(axis)

    def set_extend_axis(self, extendAxis: Embedding):
        self.extendAxis = extendAxis


def clean_crack_points(points: Sequence[Sequence[float]], tol: float = 1.0e-12) -> List[List[float]]:
    """Remove non-finite and consecutive duplicate crack points."""
    cleaned: List[List[float]] = []
    for p in points:
        q = np.asarray(p, dtype=np.float64)
        if q.shape != (2,) or not np.all(np.isfinite(q)):
            continue
        if cleaned and np.linalg.norm(q - np.asarray(cleaned[-1], dtype=np.float64)) <= tol:
            continue
        cleaned.append(q.tolist())
    return cleaned


def _to_np_points(points: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"crack points must have shape (N, 2), got {arr.shape}")
    if arr.shape[0] < 2:
        raise ValueError("at least two crack points are required")
    return arr


def _segment_data(points: np.ndarray):
    p0 = points[:-1]
    p1 = points[1:]
    delta = p1 - p0
    length = np.maximum(np.linalg.norm(delta, axis=-1), 1.0e-12)
    tangent = delta / length[:, None]
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=-1)
    cum_s0 = np.concatenate([[0.0], np.cumsum(length)[:-1]])
    total_L = float(np.sum(length))
    return p0, p1, tangent, normal, length, cum_s0, total_L


def _curve_diff_data(samples: np.ndarray):
    samples = _to_np_points(samples)
    ds = np.linalg.norm(np.diff(samples, axis=0), axis=1)
    ds = np.maximum(ds, 1.0e-12)
    cum_s = np.concatenate([[0.0], np.cumsum(ds)])
    total_L = float(cum_s[-1])

    tangent = np.zeros_like(samples, dtype=np.float64)
    tangent[0] = samples[1] - samples[0]
    tangent[-1] = samples[-1] - samples[-2]
    if len(samples) > 2:
        tangent[1:-1] = samples[2:] - samples[:-2]
    norm = np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1.0e-12)
    tangent = tangent / norm
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    return tangent, normal, cum_s, total_L


def _interp_curve_samples(samples: np.ndarray, tangent: np.ndarray, cum_s: np.ndarray, s_query: np.ndarray):
    s_clip = np.clip(np.asarray(s_query, dtype=np.float64), 0.0, float(cum_s[-1]))
    x_out = np.empty((len(s_clip), 2), dtype=np.float64)
    t_out = np.empty((len(s_clip), 2), dtype=np.float64)

    for i, sq in enumerate(s_clip):
        j = np.searchsorted(cum_s, sq, side="right") - 1
        j = int(np.clip(j, 0, len(cum_s) - 2))
        ds = max(float(cum_s[j + 1] - cum_s[j]), 1.0e-14)
        a = float((sq - cum_s[j]) / ds)
        x_out[i] = (1.0 - a) * samples[j] + a * samples[j + 1]
        t_raw = (1.0 - a) * tangent[j] + a * tangent[j + 1]
        t_out[i] = t_raw / max(float(np.linalg.norm(t_raw)), 1.0e-14)

    n_out = np.stack([-t_out[:, 1], t_out[:, 0]], axis=1)
    return x_out, t_out, n_out


def _sample_faces_from_curve(samples, tangent, normal, cum_s, total_L, n: int = 200, offset: float = 1.0e-4):
    if total_L <= 0.0:
        s_vals = np.zeros((1,), dtype=np.float64)
    else:
        s_vals = np.linspace(0.0, total_L, max(int(n), 2))
    mid, tan, nor = _interp_curve_samples(samples, tangent, cum_s, s_vals)
    xp = mid + float(offset) * nor
    xm = mid - float(offset) * nor
    return xp, xm, tan, nor


def _points_behind_tip_from_curve(samples, tangent, normal, cum_s, total_L, r_vals):
    r = np.asarray(r_vals, dtype=np.float64).reshape(-1)
    s_query = total_L - np.clip(r, 0.0, total_L)
    return _interp_curve_samples(samples, tangent, cum_s, s_query)


def _right_tip_window_torch(s_extended: torch.Tensor, total_L: float, endpoint_overlap: float = 0.0) -> torch.Tensor:
    L = max(float(total_L) + float(endpoint_overlap), 1.0e-12)
    eta = torch.clamp((L - s_extended) / L, min=0.0, max=1.0)
    return eta ** 2


def _sign_from_levelset(ls: torch.Tensor) -> torch.Tensor:
    # Gamma = +1 on the negative-normal side of the crack (see Embedding).
    return torch.where(ls < 0.0, torch.ones_like(ls), -torch.ones_like(ls))


class PolylineUnifiedEmbedding(Embedding):
    """Polyline crack embedding with a single arc-length window and a normal decay."""

    def __init__(
        self,
        points,
        *,
        decay_alpha: float = 10.0,
        decay_p: float = 1.0,
        decay_ell: Optional[float] = None,
        endpoint_overlap: float = 0.0,
        softmin_sigma: Optional[float] = None,
    ):
        super().__init__(HeavisideZero=0)
        self._points_np = _to_np_points(clean_crack_points(points))
        p0, _, tangent, normal, length, cum_s0, total_L = _segment_data(self._points_np)

        self._p0_np = p0
        self._tan_np = tangent
        self._nor_np = normal
        self._length_np = length
        self._cum_s0_np = cum_s0
        self._total_L = total_L

        self.decay_alpha = float(decay_alpha)
        self.decay_p = float(decay_p)
        self.decay_ell = None if decay_ell is None else float(decay_ell)
        self.endpoint_overlap = float(endpoint_overlap)

        if softmin_sigma is None:
            softmin_sigma = 0.5 * float(np.min(length))
        self.softmin_sigma = max(float(softmin_sigma), 1.0e-6)


        self._tip_xy_np = self._points_np[-1].copy()
        self._tip_tangent_np = tangent[-1].copy()
        self._tip_beta = float(np.arctan2(tangent[-1, 1], tangent[-1, 0]))
        self._start_xy_np = self._points_np[0].copy()
        self._start_tangent_np = tangent[0].copy()

        self._ls_override: Optional[float] = None
        self._device_dtype_buffers_attached = False

    def _attach_buffers(self, like: torch.Tensor):
        if self._device_dtype_buffers_attached:
            try:
                if self._p0.device == like.device and self._p0.dtype == like.dtype:
                    return
            except AttributeError:
                pass
        d, dt = like.device, like.dtype
        self._p0 = torch.as_tensor(self._p0_np, device=d, dtype=dt)
        self._tan = torch.as_tensor(self._tan_np, device=d, dtype=dt)
        self._nor = torch.as_tensor(self._nor_np, device=d, dtype=dt)
        self._length = torch.as_tensor(self._length_np, device=d, dtype=dt)
        self._cum_s0 = torch.as_tensor(self._cum_s0_np, device=d, dtype=dt)
        self._tip_xy = torch.as_tensor(self._tip_xy_np, device=d, dtype=dt)
        self._tip_tan = torch.as_tensor(self._tip_tangent_np, device=d, dtype=dt)
        self._start_xy = torch.as_tensor(self._start_xy_np, device=d, dtype=dt)
        self._start_tan = torch.as_tensor(self._start_tangent_np, device=d, dtype=dt)
        self._device_dtype_buffers_attached = True

    def _project(self, xy: torch.Tensor):
        delta = xy[:, None, :] - self._p0[None, :, :]
        t_unclamped = (delta * self._tan[None, :, :]).sum(dim=-1)
        t_clamped = torch.clamp(
            t_unclamped,
            min=torch.zeros_like(t_unclamped),
            max=self._length[None, :].expand_as(t_unclamped),
        )
        foot = self._p0[None, :, :] + t_clamped.unsqueeze(-1) * self._tan[None, :, :]
        r2 = ((xy[:, None, :] - foot) ** 2).sum(dim=-1)
        signed_d = (delta * self._nor[None, :, :]).sum(dim=-1)
        s_proj = self._cum_s0[None, :] + t_clamped
        return signed_d, r2, s_proj

    def getGamma(self, xy):
        if xy.dim() == 1:
            xy = xy.unsqueeze(0)
        self._attach_buffers(xy)

        signed_d, r2, s_proj = self._project(xy)
        w = torch.softmax(-r2 / (self.softmin_sigma ** 2), dim=-1)
        ls = (w * signed_d).sum(dim=-1)
        s_proj_w = (w * s_proj).sum(dim=-1)

        past_tip = ((xy - self._tip_xy) * self._tip_tan).sum(dim=-1).clamp(min=0.0)
        before_start = (-(xy - self._start_xy) * self._start_tan).sum(dim=-1).clamp(min=0.0)
        s_extended = s_proj_w + past_tip - before_start

        envelope = _right_tip_window_torch(s_extended, self._total_L, self.endpoint_overlap)
        dist = torch.abs(ls)
        if self.decay_ell is None:
            envelope = envelope * torch.exp(-self.decay_alpha * (dist ** self.decay_p))
        else:
            envelope = envelope * torch.exp(-self.decay_alpha * ((dist / self.decay_ell) ** self.decay_p))


        sign_term = _sign_from_levelset(ls) if self._ls_override is None else torch.full_like(envelope, self._ls_override)
        return (sign_term * envelope).unsqueeze(-1)

    def set_ls(self, ls: float):
        v = float(ls)
        if v not in (-1.0, 1.0):
            raise ValueError(f"set_ls expects +1 or -1, got {v}")
        self._ls_override = v

    def restore_ls(self):
        self._ls_override = None

    @property
    def tip(self):
        return self._tip_xy_np.copy()

    @property
    def tip_tangent(self):
        return self._tip_tangent_np.copy()

    @property
    def tip_beta(self) -> float:
        return float(self._tip_beta)

    @property
    def total_length(self) -> float:
        return float(self._total_L)

    def curve_data(self):
        tangent, normal, cum_s, total_L = _curve_diff_data(self._points_np)
        return self._points_np.copy(), tangent.copy(), normal.copy(), cum_s.copy(), float(total_L)

    def sample_faces(self, n: int = 200, offset: float = 1.0e-4):
        samples, tangent, normal, cum_s, total_L = self.curve_data()
        return _sample_faces_from_curve(samples, tangent, normal, cum_s, total_L, n=n, offset=offset)

    def points_behind_tip(self, r_vals):
        samples, tangent, normal, cum_s, total_L = self.curve_data()
        return _points_behind_tip_from_curve(samples, tangent, normal, cum_s, total_L, r_vals)


def make_crack_embedding(
    points,
    *,
    decay_alpha: float = 10.0,
    decay_p: float = 1.0,
    decay_ell: Optional[float] = None,
    endpoint_overlap: float = 0.0,
    softmin_sigma: Optional[float] = None,
):
    """Build the crack embedding shared by the thermal and mechanical networks."""
    points = clean_crack_points(points)
    if len(points) < 2:
        raise ValueError("need at least two crack points")
    return PolylineUnifiedEmbedding(
        points,
        decay_alpha=decay_alpha,
        decay_p=decay_p,
        decay_ell=decay_ell,
        endpoint_overlap=endpoint_overlap,
        softmin_sigma=softmin_sigma,
    )
