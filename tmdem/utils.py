"""
Checkpoints, plotting grids and logging shared by the example scripts.

Anything specific to one example (loading schedule, fracture energy law,
boundary conditions, propagation policy) stays in that example.

Author: Han Zhang (han.zhang7@unsw.edu.au)
"""

import os
import random

import numpy as np
import torch
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Checkpoint IO
# ---------------------------------------------------------------------------
def torch_load_checkpoint(path, device):
    """Load a checkpoint across PyTorch versions.

    PyTorch 2.6+ may default to weights_only=True, which rejects the numpy
    arrays in the checkpoint payload.
    """
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def find_latest_checkpoint(results_dir):
    """Return (step, checkpoint_path) of the latest completed step."""
    candidates = []
    if not os.path.isdir(results_dir):
        return None, None

    for name in os.listdir(results_dir):
        if not name.startswith("step_"):
            continue
        step_dir = os.path.join(results_dir, name)
        if not os.path.isdir(step_dir):
            continue
        for fname in os.listdir(step_dir):
            if fname.startswith("checkpoint_step_") and fname.endswith(".pt"):
                try:
                    step = int(fname.replace("checkpoint_step_", "").replace(".pt", ""))
                    candidates.append((step, os.path.join(step_dir, fname)))
                except ValueError:
                    pass

    if not candidates:
        return None, None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1]


def restore_from_checkpoint(model, checkpoint_path, fd_graph, device, strict=False):
    """Restore model, mesh, carried thermal field, crack path, FD history and random streams.

    Returns: start_step, fd_graph, Temp_n_f, X_f, vertex, parent_indices,
    crack_points (or None).
    """
    ckpt = torch_load_checkpoint(checkpoint_path, device)
    step_done = int(ckpt["step"])

    model.model_T.load_state_dict(ckpt["thermal_state_dict"], strict=strict)
    model.model_uv.load_state_dict(ckpt["mechanical_state_dict"], strict=strict)

    X_f = ckpt["X_f"]
    Temp_n_f = ckpt["T_n_f"]
    vertex = ckpt["vertex"]
    parent_indices = ckpt["parent_indices"]

    fd_old = ckpt.get("fdGraph", None)
    if fd_old is not None:
        n = min(fd_graph.shape[0], fd_old.shape[0])
        fd_graph[:n, :] = fd_old[:n, :]

    model.thermal_adam_loss_history = list(ckpt.get("thermal_adam_loss_history", []))
    model.thermal_lbfgs_loss_history = list(ckpt.get("thermal_lbfgs_loss_history", []))
    model.elastic_adam_loss_history = list(ckpt.get("elastic_adam_loss_history", []))
    model.elastic_lbfgs_loss_history = list(ckpt.get("elastic_lbfgs_loss_history", []))

    # Restore the random streams if the checkpoint carries them, so that a
    # resumed run continues the streams of an uninterrupted one (see
    # ThermoSolver2D.save_model for what these states cover).
    if ckpt.get("rng_python") is not None:
        random.setstate(ckpt["rng_python"])
    if ckpt.get("rng_numpy") is not None:
        np.random.set_state(ckpt["rng_numpy"])
    if ckpt.get("rng_torch") is not None:
        torch.set_rng_state(ckpt["rng_torch"].cpu().to(torch.uint8))
    if ckpt.get("rng_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(
            [t.cpu().to(torch.uint8) for t in ckpt["rng_cuda"]])
        print("[resume] random streams restored from the checkpoint")

    if "K_I" in ckpt and hasattr(model, "K_I"):
        model.K_I.data.fill_(float(ckpt["K_I"]))
    if "K_II" in ckpt and hasattr(model, "K_II"):
        model.K_II.data.fill_(float(ckpt["K_II"]))

    ckpt_crack_points = ckpt.get("crack_points", None)
    if ckpt_crack_points is not None:
        ckpt_crack_points = np.asarray(ckpt_crack_points, dtype=np.float64).tolist()

    start_step = step_done + 1
    print(f"[resume] Loaded checkpoint: {checkpoint_path}")
    print(f"[resume] Last completed step = {step_done}; continuing from step {start_step}")
    return start_step, fd_graph, Temp_n_f, X_f, vertex, parent_indices, ckpt_crack_points


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def make_prediction_grid(domain_x, domain_y, n=251):
    """Structured tensor-product grid for plotting fields that jump across the crack."""
    x = np.linspace(domain_x[0], domain_x[1], n)
    y = np.linspace(domain_y[0], domain_y[1], n)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return np.column_stack([xx.reshape(-1), yy.reshape(-1)])


def mask_crack_band_for_plot(X, values, crack_geometry, width=0.004, tip_radius=0.012, domain_mask_fn=None):
    """NaN-mask a narrow crack band (plotting only).

    Removes the jagged edge produced when the sharp jump across the crack
    cuts through rectangular plotting pixels.  ``domain_mask_fn(X) -> bool mask``
    optionally NaNs points outside a non-rectangular domain first.
    """
    X = np.asarray(X, dtype=float)
    val = np.asarray(values).reshape(-1, 1).copy()

    if domain_mask_fn is not None:
        val[~np.asarray(domain_mask_fn(X), dtype=bool), 0] = np.nan

    if crack_geometry is None or not hasattr(crack_geometry, "curve_points"):
        return val
    pts = np.asarray(crack_geometry.curve_points, dtype=np.float64)
    if pts.shape[0] < 2:
        return val

    dist_min = np.full((X.shape[0],), np.inf)
    for p0, p1 in zip(pts[:-1], pts[1:]):
        seg = p1 - p0
        seg_len2 = float(np.dot(seg, seg)) + 1.0e-30
        t = np.clip(((X - p0) @ seg) / seg_len2, 0.0, 1.0)
        proj = p0[None, :] + t[:, None] * seg[None, :]
        dist_min = np.minimum(dist_min, np.linalg.norm(X - proj, axis=1))

    tip = np.asarray(crack_geometry.tip, dtype=np.float64)
    dist_tip = np.linalg.norm(X - tip[None, :], axis=1)
    val[(dist_min < width) | (dist_tip < tip_radius), 0] = np.nan
    return val


def save_crack_path(points, step_dir, step, domain_x, domain_y, crack_geometry=None, show=False):
    """Save control points, the active crack curve, and a path plot."""
    pts = np.asarray(points, dtype=np.float64)
    np.savetxt(os.path.join(step_dir, f"crack_control_points_step_{step}.txt"), pts, header="x y")

    if crack_geometry is not None and hasattr(crack_geometry, "curve_points"):
        curve = np.asarray(crack_geometry.curve_points, dtype=np.float64)
    else:
        curve = pts
    np.savetxt(os.path.join(step_dir, f"crack_path_step_{step}.txt"), curve, header="x y")

    plt.figure(figsize=(5, 5))
    plt.plot(curve[:, 0], curve[:, 1], "-", linewidth=2.0, label="active crack")
    plt.plot(pts[:, 0], pts[:, 1], "o", markersize=3, label="control points")
    plt.xlim(domain_x[0], domain_x[1])
    plt.ylim(domain_y[0], domain_y[1])
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Crack path - Step {step}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(step_dir, f"crack_path_step_{step}.png"), dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def log_resampling_change(label, X_before, X_after):
    """Print whether a training call changed its integration-point batch."""
    same_shape = X_before.shape == X_after.shape
    same_points = False
    if same_shape:
        same_points = np.allclose(X_before[:, :2], X_after[:, :2], rtol=0.0, atol=1.0e-14)

    if same_shape and same_points:
        print(f"[{label}] integration points unchanged: {X_after.shape[0]}")
    else:
        print(
            f"[{label}] integration points resampled/changed: "
            f"{X_before.shape[0]} -> {X_after.shape[0]}"
        )
