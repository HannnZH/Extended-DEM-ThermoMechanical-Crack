#!/usr/bin/env python
"""
Notched cruciform specimen under combined thermo-mechanical loading.

The benchmark of Prasad et al., in the displacement-controlled statement of
Mandal et al.: a cruciform plate whose arms are 50 mm wide and extend 50 mm
from the central square, with a 10 mm edge notch at 45 deg from the
lower-right re-entrant corner, E = 218.4e3, nu = 0.2, alpha = 6e-4 per deg C,
plane stress.  Three load types are run:

    I    mechanical only: a displacement prescribed on the top face,
         isothermal;
    II   thermal only: +T on the top face and -T on the bottom face, with the
         top face traction free;
    III  both loads at once, rising in a fixed ratio.

In all three the ends of the side arms are held at the reference temperature
with u = 0, and the bottom face carries u = v = 0.

Each load step solves the staggered thermal then mechanical problem, extracts
the stress intensity factors with the interaction integral, and extends the
crack by a crack increment in the maximum hoop stress direction when the
energy release rate of the kink G(theta_c) exceeds the fracture energy.
Coordinates are physical mm; AxisScaleNet and ThermalNet map x, y from
[lb, ub] to [-1, 1] internally.

    python examples/cruciform.py --load-type I      # mechanical load
    python examples/cruciform.py --load-type II     # thermal load
    python examples/cruciform.py --load-type III    # both (default)
    python examples/cruciform.py --load-type I --seed 1

--seed sets the network initialization (default 2025), --out the output
folder (default runs/cruciform_<load-type>), and --resume continues from the
latest checkpoint in it.

Author: Han Zhang (han.zhang7@unsw.edu.au)
"""

import os
import argparse
import random
import time
import copy
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")   # figures are written to files; no window is opened
import warnings
warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
import matplotlib.pyplot as plt

import _entry  # noqa: F401  (puts the repository root on the import path)

from tmdem.crack_embedding import extendAxisNet

from tmdem.crack_embedding import (
    clean_crack_points,
    make_crack_embedding as _build_crack_embedding,
)

from tmdem.solver import (
    AxisScaleNet,
    CrackGeometry,
    MLP,
    ThermalNet,
    ThermoSolver2D,
    get_uniform_mesh,
    plot_field,
    set_seed,
)

from tmdem.sif_estimation import (
    SIFEstimator,
    effective_energy_release_rate,
    extraction_is_plausible,
    get_active_crack_geometry,
    last_tangent_normal,
    local_material_at_tip,
    max_hoop_kink_angle as lib_max_hoop_kink_angle,
    plane_strain_flag,
    sample_crack_tip_temperature,
    set_crack_state,
    total_crack_length,
)

from tmdem.utils import (
    find_latest_checkpoint,
    log_resampling_change,
    make_prediction_grid,
    mask_crack_band_for_plot,
    torch_load_checkpoint,
)

plt.close("all")

# -----------------------------------------------------------------------------
# Command line: load case, seed and output folder.  The defaults are the
# settings of the runs reported in the paper.
# -----------------------------------------------------------------------------
cli = argparse.ArgumentParser(description='Notched cruciform specimen under combined thermo-mechanical loading.')
cli.add_argument("--load-type", choices=("I", "II", "III"), default="III",
                 help="I: prescribed top displacement only; II: temperature only, top face free; "
                      "III: both (default)")
cli.add_argument("--seed", type=int, default=2025,
                 help="seed of the network initialization (default 2025)")
cli.add_argument("--resume", action="store_true",
                 help="continue from the latest checkpoint in the output folder")
cli.add_argument("--out", default=None,
                 help="output folder (default runs/cruciform_<load-type>)")
cli_args = cli.parse_args()
out_dir = cli_args.out or os.path.join("runs", f"cruciform_{cli_args.load_type}")
os.makedirs(out_dir, exist_ok=True)
os.chdir(out_dir)
print("Output folder:", os.getcwd())


# -----------------------------------------------------------------------------
# Reproducibility and folders
# -----------------------------------------------------------------------------
set_seed(cli_args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training on device:", device)

results_dir = "results"
os.makedirs(results_dir, exist_ok=True)
start_time = time.time()

resume_from_checkpoint = cli_args.resume
# The propagation log is appended to, so a fresh run removes the log of an
# earlier run in the same folder.
_prop_log_path = os.path.join(results_dir, "crack_propagation_history.csv")
if not resume_from_checkpoint and os.path.exists(_prop_log_path):
    os.remove(_prop_log_path)
resume_checkpoint_path = "auto"
strict_resume = False

# -----------------------------------------------------------------------------
# Geometry in mm
# -----------------------------------------------------------------------------
# Cruciform = union of a vertical arm and a horizontal arm.
# Arm width is 50 mm and each arm extends 50 mm from the central square.
arm_half_width = 25.0
half_size = 75.0
Lx = Ly = half_size

domainEndsX = np.array([-half_size, half_size], dtype=np.float64)
domainEndsY = np.array([-half_size, half_size], dtype=np.float64)

cruciform_polygon = np.array([
    [-75.0, -25.0], [-25.0, -25.0], [-25.0, -75.0],
    [ 25.0, -75.0], [ 25.0, -25.0], [ 75.0, -25.0],
    [ 75.0,  25.0], [ 25.0,  25.0], [ 25.0,  75.0],
    [-25.0,  75.0], [-25.0,  25.0], [-75.0,  25.0],
], dtype=np.float64)

# The 10 mm initial notch is an edge notch from the lower-right re-entrant
# corner of the cruciform central square.  The active crack is stored from
# mouth -> tip, so the last point is the propagating tip.
#
#   notch mouth = ( 25, -25) mm
#   notch tip   = ( 25, -25) + 10/sqrt(2) * (-1, +1) mm
notch_length = 10.0
notch_mouth = np.array([arm_half_width, -arm_half_width], dtype=np.float64)
notch_tip = notch_mouth + notch_length / np.sqrt(2.0) * np.array([-1.0, 1.0], dtype=np.float64)
crack_points = [notch_mouth.tolist(), notch_tip.tolist()]


def inside_cruciform_np(X):
    """Boolean mask for the cruciform domain."""
    X = np.asarray(X, dtype=np.float64)
    x = X[:, 0]
    y = X[:, 1]
    vertical = (np.abs(x) <= arm_half_width + 1.0e-12) & (np.abs(y) <= half_size + 1.0e-12)
    horizontal = (np.abs(y) <= arm_half_width + 1.0e-12) & (np.abs(x) <= half_size + 1.0e-12)
    return vertical | horizontal


def smoothstep01(z):
    z = torch.clamp(z, 0.0, 1.0)
    return z * z * (3.0 - 2.0 * z)


def center_face_selector_torch(coord):
    """1 on the loaded central face, smoothly decays outside |coord|=25 mm."""
    a = float(arm_half_width)
    L = float(half_size)
    ac = torch.abs(coord)
    z = (L - ac) / max(L - a, 1.0e-12)
    return torch.where(ac <= a, torch.ones_like(coord), smoothstep01(z))


def distance_point_to_segment(p, a, b):
    p = np.asarray(p, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1.0e-30:
        return float(np.linalg.norm(p - a))
    t = np.clip(float(np.dot(p - a, ab) / denom), 0.0, 1.0)
    q = a + t * ab
    return float(np.linalg.norm(p - q))


def distance_to_cruciform_boundary(p):
    pts = cruciform_polygon
    return min(distance_point_to_segment(p, pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts)))


def tip_on_cruciform_boundary(p, tol=1.0e-7):
    p = np.asarray(p, dtype=np.float64).reshape(1, 2)
    if not inside_cruciform_np(p)[0]:
        return True
    return distance_to_cruciform_boundary(p[0]) <= tol


# -----------------------------------------------------------------------------
# Material and scaling
# -----------------------------------------------------------------------------
# Mandal et al. state this benchmark in N, m and deg C: E = 218.4e3 N/m^2,
# G_f = 2.0e-4 N/m, f_t = 120 N/m^2.  Lengths here are in mm and E = 218.4e3
# is read as N/mm^2, which scales the stiffness by 1e6, so the fracture
# energy is scaled by the same factor: G_c = 2.0e-7 N/mm x 1e6 = 0.2 N/mm.
# The Irwin length E G_c / f_t^2 is 3.03 mm in both systems.
Gc0 = 0.2
use_temperature_dependent_Gc = False

model_params = {
    "E": 218.4e3,
    "nu": 0.2,
    "alpha": 6.0e-4,
    "k": 1.0,
    "rho": 0.0,
    "c": 1.0,
    "dt": 1.0,
    "T_ref": 0.0,
    "plane_type": "plane_stress",
    "U_ref": 5.0e-2,       # final top displacement = 0.05 mm -> v_delta_nd = 1
    "DeltaT_ref": 10.0,    # final top temperature = +10 C -> T_delta_nd = 1
    "lb": np.array([domainEndsX[0], domainEndsY[0]], dtype=np.float64),
    "ub": np.array([domainEndsX[1], domainEndsY[1]], dtype=np.float64),
    "use_thermal_jump": False,
    "use_explicit_mechanical_jump": False,
    "adiabatic_crack_weight": 1.0e-8,
    "adiabatic_crack_samples": 200,
    "crack_face_offset": 5.0e-2,
    "use_crack_conductivity_degradation": False,
    "crack_band_width": 1.0,
    "crack_min_conductivity_ratio": 1.0e-6,
    # Near-tip densification of the Monte Carlo integration points: background
    # cells within tip_densify_radius (mm) of the tip receive tip_densify_factor
    # times as many points.
    "tip_densify_radius": 3.0,
    "tip_densify_factor": 48,
    "use_tip_enrichment": True,
    "tip_enrichment_q": 1.0,
    # Decay of the tip enrichment for coordinates in mm, an effective radius of
    # O(1 mm).
    "tip_enrichment_decay_factor": 0.05,
    "tip_enrichment_u_scale": 1.0,
}

# -----------------------------------------------------------------------------
# Mesh, integration, loading and training
# -----------------------------------------------------------------------------
mesh_policy = "fixed"
nx_elem = 150
ny_elem = 150

thermal_num_integration_points = 9
thermal_integration_method = "mc"
elastic_integration_method = "mc"
elastic_num_integration_points = 9
elastic_gauss_order = 3

# Both loads are applied in 100 equal increments, as in Mandal et al.,
# reaching u* = 0.05 mm and theta* = +-10 C.  The load type sets the increment
# that does not belong to it to zero.
deltaT_phys = 0.0 if cli_args.load_type == "I" else 0.1       # C per increment, top +, bottom -
deltaV_phys = 0.0 if cli_args.load_type == "II" else 5.0e-4   # mm per increment at the top face
numSteps = 101          # step 0 to 100

# Load type of this run (I, II or III), derived from the increments, for the
# step banner.
load_type_label = ("I" if deltaT_phys == 0.0 else
                   "II" if deltaV_phys == 0.0 else "III")

# The top face carries the mechanical load: a displacement boundary in types I
# and III, traction free in type II.  Derived from the load increment, so it
# follows the load type and does not change during the ramp.
top_face_is_free = deltaV_phys == 0.0

thermal_num_Adam_step0 = 5000
thermal_num_LBFGS_step0 = 2500
thermal_num_Adam_step1 = 2000
thermal_num_LBFGS_step1 = 0
thermal_num_Adam = 2000
thermal_num_LBFGS = 0

elastic_num_Adam_step0 = 7000
elastic_num_LBFGS_step0 = 3000
elastic_num_Adam_step1 = 3000
elastic_num_LBFGS_step1 = 0
# Elastic iterations per load step from step 2 on, each step warm started
# from the previous one.  No elastic L-BFGS after step 0, since L-BFGS runs on
# a fixed draw of the integration points.
elastic_num_Adam = 3000
elastic_num_LBFGS = 0

lr_Adam = 1.0e-3
lr_lbfgs = 0.1
num_log = 100

enable_resample_thermal = True
enable_resample_elastic = True
resample_frequency_epochs_thermal = 500
resample_frequency_epochs_elastic = 500

# -----------------------------------------------------------------------------
# Crack propagation and SIF settings
# -----------------------------------------------------------------------------
enable_crack_propagation = True
# Direction rule.  "max_hoop" takes the maximum hoop stress direction from the
# extracted K1 and K2, as in the other examples.  "energy_regularized" selects
# the kink by the relaxed elastic energy over a set of candidate angles with a
# curvature penalty, "reference_regularized" adds a penalty toward a
# prescribed angle, and "straight" continues along the last tangent.
propagation_mode = "max_hoop"
# Crack increment.  The interaction contour (sif_interaction_radius, 3.6 mm)
# spans several background cells, and with sif_limit_by_last_segment its
# radius is clipped to 0.9 times the straight run ending at the tip, which
# after a kink is one increment: 0.9 * 4 mm = 3.6 mm.
crack_increment = 4.0                 # mm per propagation event
# Upper limit on the crack increments within one load step, each followed by
# retraining and a new extraction.  With a limit of one, growth is paced by
# the load stepping, as in the phase-field solution of Mandal et al. with the
# same load increments.
max_propagations_per_step = 1
# Griffith threshold chi_c on G(theta_c)/Gc, the same in all examples.
min_G_over_Gc_to_propagate = 1.0
# Grow the crack only on an extraction that the radius sweep finds admissible
# (SIFEstimator.extraction_is_validated), as in sen_tension.py.
require_validated_extraction = True
# Cap on the kink angle, shared by the direction rule and the driving force.
max_kink_angle_deg = 60.0

sif_cod_num_points = 10
# Two-sided face-sampling offset in mm for the crack-opening estimate, with
# float32 noise << offset << sif_r_min and, for p = 1,
# offset << crack_embedding_decay_length, so that the normal decay of the
# embedding does not attenuate the sampled jump.
sif_cod_offset = 3.0e-3
# Extraction length scales are set in units of the background cell size h
# rather than of crack_increment, so that the crack increment and the
# extraction can be chosen independently.
background_cell_size = (domainEndsX[1] - domainEndsX[0]) / nx_elem   # 1.000 mm
sif_r_min = 4.0 * background_cell_size
sif_r_max = 12.0 * background_cell_size
sif_limit_by_last_segment = True
sif_estimator = "interaction_with_cod_fallback"
# Interaction-integral contour radius, 3.6 cells, kept at 0.9 * crack_increment
# or below so that it fits the straight run after a kink (see
# sif_limit_by_last_segment).
sif_interaction_radius = 3.6 * background_cell_size
# Quadrature points of the interaction integral (see SIFEstimator).
sif_interaction_num_points = 2880
print_sif_range_diagnostics = True

# Normal decay of the crack embedding.  crack_embedding_decay_length is the
# e-folding length ell/alpha of the envelope exp(-alpha * (dist/ell)^p), in
# units of the background cell size h.  Load type II uses 0.3 h with the
# larger displacement network, types I and III use 5 h with the smaller one.
crack_embedding_decay_length = (0.30 if cli_args.load_type == "II" else 5.0) * background_cell_size
crack_embedding_decay_ell = 6.0 * background_cell_size
crack_embedding_decay_alpha = crack_embedding_decay_ell / crack_embedding_decay_length
crack_embedding_decay_p = 1.0
crack_embedding_endpoint_overlap = 2.0 * background_cell_size

prop_thermal_num_Adam = 1500
prop_thermal_num_LBFGS = 0
# Elastic iterations after each crack extension.
prop_elastic_num_Adam = 3000
prop_elastic_num_LBFGS = 0

# Absolute cap on the ratio G(theta_c)/Gc accepted from one extraction
# (extraction_is_plausible, default 5.0); a larger ratio stops propagation in
# this load step.  The jump cap of 2.0 between consecutive extractions within
# one load step keeps its default.
extraction_ratio_cap = 100

plot_grid_n = 801
smooth_plot_levels = 250
smooth_plot_dpi = 350
crack_plot_band_width = 0.12 * crack_increment
crack_plot_tip_radius = 0.25 * crack_increment

# Energy-regularized direction search, used only when propagation_mode is
# "energy_regularized" or "reference_regularized".  Candidate angles are
# measured from the previous crack tangent, and the selected direction
# minimizes the elastic energy over this set plus a small curvature penalty.
energy_candidate_angle_min_deg = -45.0
energy_candidate_angle_max_deg = 45.0
energy_candidate_angle_num = 37
energy_include_max_hoop_candidate = True
energy_probe_num_points = 6000
# All integration points within this radius of the tip are kept in the probe
# set of the direction search, and the rest of the budget is a random
# far-field subset.
energy_probe_near_radius = 4.0 * crack_increment
energy_curvature_penalty_weight = 0.08
# Weight of a penalty on the deviation from the maximum hoop stress angle
# theta_mcs in the energy search; zero, so theta_mcs only seeds the candidate
# directions.
energy_mcs_deviation_penalty_weight = 0.0

# Per-candidate relaxation: each trial direction is relaxed briefly from the
# same pre-extension state before its elastic energy is read.
candidate_relax_iters = 200
candidate_relax_lr = 1.0e-3

# Penalty toward a prescribed direction (propagation_mode =
# "reference_regularized"), off.
use_reference_direction_penalty = False
reference_direction_deg = 155
reference_direction_penalty_weight = 0.0
reference_direction_power = 0.0
energy_print_candidate_table = True

# Geometric safety rules: the crack may not form loops or run back along or
# near the existing crack.
reject_self_intersection = True
min_crack_clearance = 1.25 * crack_increment
angle_retry_count = 11

# Local direction stabilization: growth stays on the notch side, and
# K_ratio_tol is a deadband: for |K2|/|K1| below it the kink is zero.
K_ratio_tol = 0.05
# Scale factor on the kink angle; 1.0 is the plain maximum hoop stress
# criterion.
kink_angle_relaxation = 1.0
use_notch_side_growth_filter = True
min_notch_inward_projection = 0.05
reject_growth_toward_notch_mouth = True
min_notch_distance_increase = 0.10 * crack_increment

# The crack path is saved as a PNG after every accepted crack extension;
# show_crack_path_on_update has no effect with the non-interactive backend.
plot_crack_path_on_update = True
show_crack_path_on_update = True


# -----------------------------------------------------------------------------
# Custom model for cruciform hard constraints
# -----------------------------------------------------------------------------
class CruciformThermoSolver2D(ThermoSolver2D):
    """Library solver on the cruciform geometry.

    Integration points outside the cruciform are discarded, and the boundary
    conditions are imposed exactly by lifting and gating the network outputs.
    """

    def generate_integration_points(self, vertex, num_points_per_elem=5, method="mc", gauss_order=2):
        x, y, w, elem = super().generate_integration_points(
            vertex, num_points_per_elem=num_points_per_elem, method=method, gauss_order=gauss_order
        )
        X = np.column_stack([x.reshape(-1), y.reshape(-1)])
        keep = inside_cruciform_np(X)
        return x[keep], y[keep], w[keep], elem[keep]

    def net_thermal(self, x, y, T_delta_nd):
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

        X = x / float(half_size)
        Y = y / float(half_size)

        # Dirichlet thermal faces:
        # top central face:    theta = +T_delta
        # bottom central face: theta = -T_delta
        # left/right end faces: theta = 0
        center_x = center_face_selector_torch(x)
        theta_lift = T_delta_nd * Y * center_x

        gate_x = torch.clamp(1.0 - X**2, min=0.0)
        gate_y = torch.clamp(1.0 - Y**2, min=0.0)
        theta_nd = theta_lift + gate_x * gate_y * T_hat
        return self.T_ref + self.DeltaT_ref * theta_nd

    def net_uv(self, x, y, v_delta_nd):
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

        X = x / float(half_size)
        Y01 = (y + float(half_size)) / (2.0 * float(half_size))
        Y01 = torch.clamp(Y01, 0.0, 1.0)

        # u = 0 on left, right, and bottom ends.
        gate_u = torch.clamp(1.0 - X**2, min=0.0) * Y01

        # v = 0 on the bottom, and v = v_delta on the top when a displacement is
        # prescribed there (types I and III).  Without a mechanical load the top
        # face is traction free.
        if top_face_is_free:
            gate_v = Y01
            v_lift = 0.0 * v_delta_nd
        else:
            gate_v = Y01 * (1.0 - Y01)
            v_lift = Y01 * v_delta_nd

        # One scalar gate for the Williams enrichment, since gating u and v
        # differently would distort its mode shape.  It vanishes on both Dirichlet
        # sets (x=+-75 via 1-X^2, y=-+75 via Y01*(1-Y01)), and its value at the tip
        # is absorbed by the amplitudes.
        gate_tip = torch.clamp(1.0 - X**2, min=0.0) * Y01 * (1.0 - Y01)

        u_nd = gate_u * u_hat + gate_tip * (u_tip / self.U_ref)
        v_nd = v_lift + gate_v * v_hat + gate_tip * (v_tip / self.U_ref)
        return self.U_ref * u_nd, self.U_ref * v_nd


def mask_outside_domain(X, values):
    vals = np.asarray(values).reshape(-1, 1).copy()
    keep = inside_cruciform_np(X)
    vals[~keep, 0] = np.nan
    return vals


def plot_field_smooth(
    X_plot,
    values,
    domain_x,
    domain_y,
    title,
    label,
    filename,
    save_dir,
    cmap="viridis",
    vmin=None,
    vmax=None,
    show=False,
    levels=None,
    dpi=None,
):
    """Smooth field plot on the structured prediction grid, for visualization only."""
    n = int(round(np.sqrt(X_plot.shape[0])))
    if n * n != X_plot.shape[0]:
        raise ValueError("X_plot must be generated from a square structured grid.")

    xx = X_plot[:, 0].reshape(n, n)
    yy = X_plot[:, 1].reshape(n, n)
    zz = np.asarray(values, dtype=np.float64).reshape(n, n)
    zz = np.ma.masked_invalid(zz)

    if levels is None:
        levels = smooth_plot_levels
    if dpi is None:
        dpi = smooth_plot_dpi

    plt.figure(figsize=(8, 6))
    img = plt.imshow(
        zz,
        extent=(domain_x[0], domain_x[1], domain_y[0], domain_y[1]),
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
        aspect="equal",
    )
    plt.colorbar(img, label=label)
    plt.xlim(domain_x)
    plt.ylim(domain_y)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{filename}.png"), dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_cruciform_outline(save_path, crack_geometry=None, title="Cruciform domain", show=False):
    poly = np.vstack([cruciform_polygon, cruciform_polygon[0]])
    plt.figure(figsize=(6, 6))
    plt.plot(poly[:, 0], poly[:, 1], "k-", linewidth=1.5)
    if crack_geometry is not None and hasattr(crack_geometry, "curve_points"):
        curve = np.asarray(crack_geometry.curve_points, dtype=np.float64)
        plt.plot(curve[:, 0], curve[:, 1], "-", linewidth=2.0, label="crack")
        ctrl = np.asarray(crack_geometry.points, dtype=np.float64)
        plt.plot(ctrl[:, 0], ctrl[:, 1], "o", markersize=3)
        plt.legend()
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlim(domainEndsX)
    plt.ylim(domainEndsY)
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_cruciform_mesh(vertex, save_path, crack_geometry=None, title="Cruciform integration mesh", show=False):
    """Plot only active background cells whose centroids lie inside the cruciform."""
    plt.figure(figsize=(7, 7))
    for elem in np.asarray(vertex, dtype=np.float64):
        x0, x1, y0, y1 = elem[:4]
        xc = 0.5 * (x0 + x1)
        yc = 0.5 * (y0 + y1)
        if not inside_cruciform_np(np.array([[xc, yc]], dtype=np.float64))[0]:
            continue
        plt.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], "k-", linewidth=0.25)

    poly = np.vstack([cruciform_polygon, cruciform_polygon[0]])
    plt.plot(poly[:, 0], poly[:, 1], "k-", linewidth=1.5)

    if crack_geometry is not None and hasattr(crack_geometry, "curve_points"):
        curve = np.asarray(crack_geometry.curve_points, dtype=np.float64)
        plt.plot(curve[:, 0], curve[:, 1], linewidth=2.0, label="crack")
        plt.legend()

    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlim(domainEndsX)
    plt.ylim(domainEndsY)
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def restore_from_checkpoint(model, checkpoint_path, load_history, device, strict=False):
    ckpt = torch_load_checkpoint(checkpoint_path, device)
    step_done = int(ckpt["step"])
    model.model_T.load_state_dict(ckpt["thermal_state_dict"], strict=strict)
    model.model_uv.load_state_dict(ckpt["mechanical_state_dict"], strict=strict)
    X_f = ckpt["X_f"]
    Temp_n_f = ckpt["T_n_f"]
    vertex = ckpt["vertex"]
    parent_indices = ckpt["parent_indices"]
    old_hist = ckpt.get("fdGraph", None)
    if old_hist is not None:
        n = min(load_history.shape[0], old_hist.shape[0])
        m = min(load_history.shape[1], old_hist.shape[1])
        load_history[:n, :m] = old_hist[:n, :m]
    model.thermal_adam_loss_history = list(ckpt.get("thermal_adam_loss_history", []))
    model.thermal_lbfgs_loss_history = list(ckpt.get("thermal_lbfgs_loss_history", []))
    model.elastic_adam_loss_history = list(ckpt.get("elastic_adam_loss_history", []))
    model.elastic_lbfgs_loss_history = list(ckpt.get("elastic_lbfgs_loss_history", []))
    # Restore the four random number streams if the checkpoint carries them,
    # so that a resumed run draws what an uninterrupted one would have drawn.
    # A checkpoint without them is resumed on fresh streams.
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
    crack_pts = ckpt.get("crack_points", None)
    if crack_pts is not None:
        crack_pts = np.asarray(crack_pts, dtype=np.float64).tolist()
    print(f"[resume] Loaded {checkpoint_path}; continuing from step {step_done + 1}")
    return step_done + 1, load_history, Temp_n_f, X_f, vertex, parent_indices, crack_pts


def select_energy_probe_points(X_f, tip=None):
    """Probe subset for the candidate-direction energy comparison.

    Keeps all integration points within energy_probe_near_radius of the
    active tip and fills the remaining budget with a random far-field subset.
    The same probe set is used for every candidate within one proposal, so
    only relative energies matter and the weights need no renormalization.
    """
    X_f = np.asarray(X_f)
    n_total = X_f.shape[0]
    n_probe = int(min(max(1000, energy_probe_num_points), n_total))
    if n_probe >= n_total:
        return X_f.copy()
    if tip is None:
        idx = np.random.choice(n_total, size=n_probe, replace=False)
        return X_f[idx].copy()

    tip = np.asarray(tip, dtype=np.float64)
    d2 = (X_f[:, 0] - tip[0]) ** 2 + (X_f[:, 1] - tip[1]) ** 2
    r2 = float(energy_probe_near_radius) ** 2
    near = np.where(d2 <= r2)[0]
    far = np.where(d2 > r2)[0]

    if near.size >= n_probe:
        order = np.argsort(d2[near])
        sel = near[order[:n_probe]]
    else:
        n_far = min(n_probe - near.size, far.size)
        far_sel = np.random.choice(far, size=n_far, replace=False)
        sel = np.concatenate([near, far_sel])
    print(
        f"[energy-probe] {sel.size} points ({near.size} within "
        f"r={energy_probe_near_radius:.2f} of the tip, {sel.size - min(near.size, n_probe)} far-field)"
    )
    return X_f[sel].copy()


def snapshot_mechanical_state(model):
    """Deep-copy the trainable mechanical state so every candidate direction is
    relaxed from the same pre-extension warm start."""
    state = copy.deepcopy(model.model_uv.state_dict())
    extras = {}
    if hasattr(model, "K_I"):
        extras["K_I"] = float(model.K_I.detach().cpu())
    if hasattr(model, "K_II"):
        extras["K_II"] = float(model.K_II.detach().cpu())
    return state, extras


def restore_mechanical_state(model, snap):
    """Restore the mechanical weights and Williams K parameters captured above.

    The embedding (extendAxis) is not part of model_uv.state_dict(), so this is
    independent of the candidate crack swap done by set_crack_state().
    """
    state, extras = snap
    model.model_uv.load_state_dict(state)
    if "K_I" in extras and hasattr(model, "K_I"):
        model.K_I.data.fill_(extras["K_I"])
    if "K_II" in extras and hasattr(model, "K_II"):
        model.K_II.data.fill_(extras["K_II"])


def relax_elastic_for_candidate(model, X_f, T_delta_nd, v_delta_nd, n_iters, lr):
    """Short warm-started Adam relaxation of the elastic field on the current
    (candidate) crack embedding.  Updates the mechanical parameters in place,
    which the caller restores afterwards.
    """
    if X_f is None or X_f.shape[0] == 0 or int(n_iters) <= 0:
        return
    model.model_T.eval()
    model.model_uv.train()
    x = torch.tensor(X_f[:, 0:1], dtype=torch.float32, device=model.device, requires_grad=True)
    y = torch.tensor(X_f[:, 1:2], dtype=torch.float32, device=model.device, requires_grad=True)
    w = torch.tensor(X_f[:, 2:3], dtype=torch.float32, device=model.device)
    T_delta_t = torch.tensor([[T_delta_nd]], dtype=torch.float32, device=model.device)
    v_delta_t = torch.tensor([[v_delta_nd]], dtype=torch.float32, device=model.device)
    opt = torch.optim.Adam(model.mechanical_parameters(), lr=float(lr))
    for _ in range(int(n_iters)):
        opt.zero_grad()
        energy_e = model.elastic_energy_density(x, y, T_delta_t, v_delta_t)
        loss = torch.sum(energy_e * w)
        if not torch.isfinite(loss):
            break
        loss.backward()
        opt.step()


def compute_elastic_energy_snapshot(model, X_f, T_delta_nd, v_delta_nd):
    """Evaluate elastic strain energy for the current crack embedding.

    The caller relaxes the field for each candidate direction first (see
    relax_elastic_for_candidate), so this returns the relaxed energy used as
    the energy-release direction indicator.
    """
    if X_f is None or X_f.shape[0] == 0:
        return np.inf

    xy = torch.tensor(X_f[:, :2], dtype=torch.float32, device=model.device, requires_grad=True)
    w = torch.tensor(X_f[:, 2:3], dtype=torch.float32, device=model.device)
    T_delta_t = torch.tensor([[T_delta_nd]], dtype=torch.float32, device=model.device)
    v_delta_t = torch.tensor([[v_delta_nd]], dtype=torch.float32, device=model.device)

    x = xy[:, 0:1]
    y = xy[:, 1:2]
    u, v = model.net_uv(x, y, v_delta_t)
    T = model.net_thermal(x, y, T_delta_t)

    du = torch.autograd.grad(u.sum(), xy, create_graph=False, retain_graph=True)[0]
    dv = torch.autograd.grad(v.sum(), xy, create_graph=False, retain_graph=False)[0]

    eps_th = model.thermal_strain_coefficient() * (T - model.T_ref)
    e_xx = du[:, 0:1] - eps_th
    e_yy = dv[:, 1:2] - eps_th
    gamma_xy = du[:, 1:2] + dv[:, 0:1]

    tr_e = e_xx + e_yy
    sx = model.lamda * tr_e + 2.0 * model.mu * e_xx
    sy = model.lamda * tr_e + 2.0 * model.mu * e_yy
    sxy = model.mu * gamma_xy

    energy_density = 0.5 * (sx * e_xx + sy * e_yy + sxy * gamma_xy)
    energy = torch.sum(w * energy_density)
    return float(energy.detach().cpu().numpy())


def build_energy_candidate_thetas(theta_mcs):
    vals = np.deg2rad(np.linspace(
        energy_candidate_angle_min_deg,
        energy_candidate_angle_max_deg,
        int(energy_candidate_angle_num),
    ))
    extra = [0.0]
    if energy_include_max_hoop_candidate and np.isfinite(theta_mcs):
        extra += [float(theta_mcs), 0.5 * float(theta_mcs)]
    vals = np.concatenate([vals, np.asarray(extra, dtype=np.float64)])
    vals = np.unique(np.round(vals, decimals=12))
    return vals.astype(np.float64)


def segment_intersects_np(a, b, c, d, tol=1.0e-12):
    """Closed segment intersection test with a small numerical tolerance."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)

    def orient(p, q, r):
        return float(np.cross(q - p, r - p))

    def on_segment(p, q, r):
        return (
            min(p[0], r[0]) - tol <= q[0] <= max(p[0], r[0]) + tol and
            min(p[1], r[1]) - tol <= q[1] <= max(p[1], r[1]) + tol and
            abs(orient(p, q, r)) <= tol
        )

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)

    if (o1 * o2 < -tol) and (o3 * o4 < -tol):
        return True
    if abs(o1) <= tol and on_segment(a, c, b):
        return True
    if abs(o2) <= tol and on_segment(a, d, b):
        return True
    if abs(o3) <= tol and on_segment(c, a, d):
        return True
    if abs(o4) <= tol and on_segment(c, b, d):
        return True
    return False


def max_hoop_kink_angle(K1, K2):
    """Maximum hoop stress kink angle with this example's K_II deadband,
    relaxation factor and kink cap (see sif_estimation.max_hoop_kink_angle
    for the plain form)."""
    return lib_max_hoop_kink_angle(
        K1, K2,
        max_kink_angle_deg=max_kink_angle_deg,
        k_ratio_deadband=K_ratio_tol,
        relaxation=kink_angle_relaxation,
    )


def clip_extension_to_cruciform(tip, direction, length):
    tip = np.asarray(tip, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    end = tip + length * direction
    if inside_cruciform_np(end.reshape(1, 2))[0]:
        return end, False
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        p = tip + mid * length * direction
        if inside_cruciform_np(p.reshape(1, 2))[0]:
            lo = mid
        else:
            hi = mid
    clipped = tip + lo * length * direction
    return clipped, True


def segment_distance_np(p, a, b):
    p = np.asarray(p, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ab = b - a
    den = float(np.dot(ab, ab))
    if den <= 1.0e-30:
        return float(np.linalg.norm(p - a))
    t = np.clip(float(np.dot(p - a, ab) / den), 0.0, 1.0)
    q = a + t * ab
    return float(np.linalg.norm(p - q))


def candidate_too_close_to_old_crack(points, new_tip, exclude_last=2):
    if not reject_self_intersection:
        return False
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 4:
        return False
    new_tip = np.asarray(new_tip, dtype=np.float64)
    # The last one or two segments connect to the current tip and are not
    # checked.
    nseg_check = max(0, pts.shape[0] - 1 - int(exclude_last))
    for i in range(nseg_check):
        d = segment_distance_np(new_tip, pts[i], pts[i + 1])
        if d < min_crack_clearance:
            return True
    return False


def candidate_segment_intersects_old_crack(points, new_tip, exclude_last=1):
    if not reject_self_intersection:
        return False
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 4:
        return False
    a = pts[-1]
    b = np.asarray(new_tip, dtype=np.float64)
    nseg_check = max(0, pts.shape[0] - 1 - int(exclude_last))
    for i in range(nseg_check):
        if segment_intersects_np(a, b, pts[i], pts[i + 1]):
            return True
    return False


def candidate_fails_notch_side_filter(points, direction, new_tip):
    """Reject candidates that grow back into the notch mouth or the right arm.

    A valid extension keeps a positive projection on the inward notch
    direction, which points from the notch mouth into the specimen.
    """
    if not use_notch_side_growth_filter:
        return False

    pts = np.asarray(points, dtype=np.float64)
    tip = pts[-1]
    mouth = np.asarray(notch_mouth, dtype=np.float64)
    inward = np.asarray(notch_tip, dtype=np.float64) - mouth
    inward = inward / max(float(np.linalg.norm(inward)), 1.0e-14)
    direction = np.asarray(direction, dtype=np.float64)
    direction = direction / max(float(np.linalg.norm(direction)), 1.0e-14)

    if float(np.dot(direction, inward)) < min_notch_inward_projection:
        return True

    if reject_growth_toward_notch_mouth:
        d_old = float(np.linalg.norm(tip - mouth))
        d_new = float(np.linalg.norm(np.asarray(new_tip, dtype=np.float64) - mouth))
        if d_new < d_old + min_notch_distance_increase:
            return True

    return False


def signed_angle_difference(a, b):
    """Return a-b wrapped to [-pi, pi]."""
    return float(np.arctan2(np.sin(a - b), np.cos(a - b)))


def reference_direction_penalty(beta_new):
    if not use_reference_direction_penalty:
        return 0.0
    beta_ref = np.deg2rad(reference_direction_deg)
    d_beta = signed_angle_difference(float(beta_new), float(beta_ref))
    return reference_direction_penalty_weight * (abs(d_beta) / np.pi) ** reference_direction_power


def propose_crack_extension(model, points, K1, K2, X_energy_probe=None, T_delta_nd=None, v_delta_nd=None):
    _, _, beta = last_tangent_normal(points, model=model)
    theta_mcs = 0.0 if propagation_mode == "straight" else max_hoop_kink_angle(K1, K2)
    tip = np.asarray(get_active_crack_geometry(model, points).tip, dtype=np.float64)

    if propagation_mode in ("energy_regularized", "reference_regularized") and X_energy_probe is not None:
        candidate_thetas = build_energy_candidate_thetas(theta_mcs)
        if propagation_mode == "reference_regularized" and use_reference_direction_penalty:
            beta_ref = np.deg2rad(reference_direction_deg)
            theta_ref = signed_angle_difference(beta_ref, beta)
            theta_ref = np.clip(
                theta_ref,
                np.deg2rad(energy_candidate_angle_min_deg),
                np.deg2rad(energy_candidate_angle_max_deg),
            )
            extra = np.asarray([theta_ref, 0.75 * theta_ref, 0.50 * theta_ref], dtype=np.float64)
            candidate_thetas = np.unique(np.round(np.concatenate([candidate_thetas, extra]), decimals=12))
    else:
        candidate_thetas = [
            theta_mcs,
            0.50 * theta_mcs,
            0.25 * theta_mcs,
            0.0,
            -0.25 * theta_mcs,
            -0.50 * theta_mcs,
        ]
        if abs(theta_mcs) < 1.0e-12:
            small = np.deg2rad(3.0)
            candidate_thetas += [small, -small]

    candidate_data = []
    original_points = [list(map(float, p)) for p in points]

    # Same pre-extension warm start for every candidate, so the relaxed energies
    # are directly comparable as an energy-release indicator.
    do_energy_search = (
        propagation_mode in ("energy_regularized", "reference_regularized")
        and X_energy_probe is not None
    )
    baseline_state = snapshot_mechanical_state(model) if do_energy_search else None

    try:
        for theta in candidate_thetas:
            theta = float(theta)
            beta_new = beta + theta
            direction = np.array([np.cos(beta_new), np.sin(beta_new)], dtype=np.float64)
            new_tip, hit_boundary = clip_extension_to_cruciform(tip, direction, crack_increment)

            if np.linalg.norm(new_tip - tip) < 1.0e-10:
                continue
            if candidate_fails_notch_side_filter(points, direction, new_tip):
                continue
            if candidate_too_close_to_old_crack(points, new_tip):
                continue
            if candidate_segment_intersects_old_crack(points, new_tip):
                continue

            pi_val = np.nan
            if do_energy_search:
                trial_points = original_points + [new_tip.tolist()]
                # Warm start from the identical pre-extension state, install the
                # trial crack, relax briefly, then read the relaxed energy.
                restore_mechanical_state(model, baseline_state)
                set_crack_state(model, trial_points, make_crack_embedding)
                relax_elastic_for_candidate(
                    model, X_energy_probe, T_delta_nd, v_delta_nd,
                    candidate_relax_iters, candidate_relax_lr,
                )
                pi_val = compute_elastic_energy_snapshot(model, X_energy_probe, T_delta_nd, v_delta_nd)

            candidate_data.append({
                "theta": theta,
                "beta_new": beta_new,
                "direction": direction,
                "new_tip": new_tip.tolist(),
                "hit_boundary": hit_boundary,
                "energy": pi_val,
            })
    finally:
        # Restore both the mechanical weights and the original crack so the
        # accepted extension is retrained from the clean pre-search state.
        if baseline_state is not None:
            restore_mechanical_state(model, baseline_state)
        set_crack_state(model, original_points, make_crack_embedding)

    if not candidate_data:
        print("[propagation] all candidate crack directions were rejected")
        return None, beta + theta_mcs, theta_mcs, False

    if propagation_mode not in ("energy_regularized", "reference_regularized") or X_energy_probe is None:
        c = candidate_data[0]
        return c["new_tip"], c["beta_new"], c["theta"], c["hit_boundary"]

    energies = np.array([c["energy"] for c in candidate_data], dtype=np.float64)
    finite = np.isfinite(energies)
    if not np.any(finite):
        print("[propagation] all candidate energies are non-finite; falling back to first valid candidate")
        c = candidate_data[0]
        return c["new_tip"], c["beta_new"], c["theta"], c["hit_boundary"]

    e_min = float(np.min(energies[finite]))
    e_max = float(np.max(energies[finite]))
    e_scale = max(e_max - e_min, abs(e_min), 1.0e-12)
    theta_scale = max(np.deg2rad(max(abs(energy_candidate_angle_min_deg), abs(energy_candidate_angle_max_deg))), 1.0e-12)

    best_idx = None
    best_score = np.inf
    for i, c in enumerate(candidate_data):
        e_norm = (float(c["energy"]) - e_min) / e_scale if np.isfinite(c["energy"]) else 1.0e6
        curvature_penalty = energy_curvature_penalty_weight * (float(c["theta"]) / theta_scale) ** 2
        mcs_penalty = energy_mcs_deviation_penalty_weight * ((float(c["theta"]) - float(theta_mcs)) / theta_scale) ** 2
        ref_penalty = reference_direction_penalty(c["beta_new"]) if propagation_mode == "reference_regularized" else 0.0
        score = e_norm + curvature_penalty + mcs_penalty + ref_penalty
        c["score"] = score
        c["ref_penalty"] = ref_penalty
        if score < best_score:
            best_score = score
            best_idx = i

    if energy_print_candidate_table:
        print("[energy-direction] candidate directions:")
        for i, c in enumerate(candidate_data):
            mark = "*" if i == best_idx else " "
            print(
                f"  {mark} theta={np.rad2deg(c['theta']):7.2f} deg, "
                f"beta={np.rad2deg(c['beta_new']):7.2f} deg, "
                f"energy={c['energy']:.6e}, ref={c.get('ref_penalty', 0.0):.3e}, "
                f"score={c['score']:.4e}, tip=({c['new_tip'][0]:.3f},{c['new_tip'][1]:.3f})"
            )

    c = candidate_data[best_idx]
    return c["new_tip"], c["beta_new"], c["theta"], c["hit_boundary"]


def crack_reached_boundary(points, model=None, tol=1.0e-7):
    if model is not None:
        try:
            tip = np.asarray(model.crack_geometry.tip, dtype=np.float64)
        except Exception:
            tip = np.asarray(points[-1], dtype=np.float64)
    else:
        tip = np.asarray(points[-1], dtype=np.float64)
    return tip_on_cruciform_boundary(tip, tol=tol)


def driving_force(K1, K2, crack_tip_xy):
    """Crack driving force on the kinked configuration, and the kink angle.

    The energy release rate of the kink G(theta_c), as in the other examples;
    it equals the straight-ahead J = (K1^2 + K2^2)/E' in pure mode I.  The kink
    angle uses the same cap as the direction search.  Plane stress, so E' = E.
    """
    E_tip, nu_tip, _ = local_material_at_tip(model, float(crack_tip_xy[0]))
    E_eff = E_tip / (1.0 - nu_tip ** 2) if plane_strain_flag(model) else E_tip
    return effective_energy_release_rate(
        K1, K2, E_eff,
        max_kink_angle_deg=max_kink_angle_deg,
        relaxation=kink_angle_relaxation)


def get_tip_fracture_properties(model, points, T_delta_nd):
    geom = get_active_crack_geometry(model, points)
    tip = np.asarray(geom.tip, dtype=np.float64)
    T_tip = sample_crack_tip_temperature(model, tip, T_delta_nd, radius=0.25 * crack_increment, n_ring=24, keep_fn=inside_cruciform_np)
    return tip, T_tip, float(Gc0)


def append_propagation_history(row):
    path = os.path.join(results_dir, "crack_propagation_history.csv")
    write_header = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if write_header:
            f.write("step,prop_iter,v_top_mm,T_amp_C,K1,K2,J,T_tip_C,Gc,J_over_Gc,beta,theta,tip_x,tip_y,propagated\n")
        f.write(",".join(str(x) for x in row) + "\n")


def save_crack_path(points, step_dir, step, crack_geometry=None, prop_iter=None, show=False):
    """Save and optionally display the current crack path.

    prop_iter numbers the PNGs of several extensions within one load step.
    """
    pts = np.asarray(points, dtype=np.float64)
    if prop_iter is None:
        suffix = f"step_{step}"
        title = f"Crack path, step {step}"
    else:
        suffix = f"step_{step}_prop_{prop_iter:02d}"
        title = f"Crack path, step {step}, propagation {prop_iter}"

    np.savetxt(os.path.join(step_dir, f"crack_control_points_{suffix}.txt"), pts, header="x y")
    if crack_geometry is not None and hasattr(crack_geometry, "curve_points"):
        curve = np.asarray(crack_geometry.curve_points, dtype=np.float64)
    else:
        curve = pts

    np.savetxt(os.path.join(step_dir, f"crack_path_{suffix}.txt"), curve, header="x y")
    png_path = os.path.join(step_dir, f"crack_path_{suffix}.png")
    plot_cruciform_outline(png_path, crack_geometry=crack_geometry, title=title, show=show)

    # Also keep a convenient latest-path copy in the main results folder.
    plot_cruciform_outline(
        os.path.join(results_dir, "crack_path_latest.png"),
        crack_geometry=crack_geometry,
        title=title,
        show=False,
    )


def make_crack_embedding(points):
    points = clean_crack_points(points)
    if len(points) < 2:
        points = [notch_mouth.tolist(), notch_tip.tolist()]
    return _build_crack_embedding(
        points,
        decay_alpha=crack_embedding_decay_alpha,
        decay_p=crack_embedding_decay_p,
        decay_ell=crack_embedding_decay_ell,
        endpoint_overlap=crack_embedding_endpoint_overlap,
    )


def make_elastic_integration_points(model, vertex, X_f_thermal):
    if elastic_integration_method.lower() == "mc":
        return X_f_thermal.copy()
    x_np, y_np, wt_np, elem_np = model.generate_integration_points(
        vertex, elastic_num_integration_points, method="gaussian", gauss_order=elastic_gauss_order
    )
    return np.column_stack([x_np, y_np, wt_np, elem_np])


def elastic_resampling_args(vertex):
    if elastic_integration_method.lower() == "mc" and enable_resample_elastic:
        return vertex, elastic_num_integration_points, resample_frequency_epochs_elastic
    return None, elastic_num_integration_points, 0


def save_convergence_plot(history_adam, history_lbfgs, path, title):
    plt.figure(figsize=(8, 5))
    if history_adam:
        plt.semilogy(history_adam, label="Adam")
    if history_lbfgs:
        offset = len(history_adam)
        plt.semilogy(range(offset, offset + len(history_lbfgs)), history_lbfgs, label="LBFGS")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


# -----------------------------------------------------------------------------
# Build model and integration mesh
# -----------------------------------------------------------------------------
crack_embedding = make_crack_embedding(crack_points)
crack_geometry = CrackGeometry.from_embedding(crack_embedding, crack_points)

uv_core = AxisScaleNet(
    # Displacement network.  Load type II uses four hidden layers of 100 with the
    # short embedding decay length, types I and III three hidden layers of 50
    # with the long one.
    MLP([3, 100, 100, 100, 100, 2] if cli_args.load_type == "II" else [3, 50, 50, 50, 2],
        activation=nn.Tanh),
    lb=model_params["lb"],
    ub=model_params["ub"],
)
model_uv = extendAxisNet(uv_core, crack_embedding)

thermal_core = ThermalNet(
    [3, 50, 50, 50, 1],
    lb=model_params["lb"],
    ub=model_params["ub"],
    omega_0=30.0,
)
model_T = extendAxisNet(thermal_core, crack_embedding)

model = CruciformThermoSolver2D(
    model_params,
    model_uv=model_uv,
    model_T=model_T,
    crack_geometry=crack_geometry,
)

# Shared K/J extraction: interaction integral with guards, COD fallback,
# cross-check and per-step validation plots (see tmdem/sif_estimation.py).
sif = SIFEstimator(
    model,
    tip_boundary_distance=distance_to_cruciform_boundary,
    interaction_radius=sif_interaction_radius,
    interaction_num_points=sif_interaction_num_points,
    cod_r_min=sif_r_min,
    cod_r_max=sif_r_max,
    cod_num_points=sif_cod_num_points,
    cod_offset=sif_cod_offset,
    crack_increment=crack_increment,
    limit_r_by_last_segment=sif_limit_by_last_segment,
    estimator=sif_estimator,
    print_range_diagnostics=print_sif_range_diagnostics,
    j_label="J",
)

vertex = get_uniform_mesh(nx_elem, ny_elem, domainEndsX, domainEndsY)
parent_indices = np.full(len(vertex), -1, dtype=np.int64)

x_np, y_np, wt_np, elem_np = model.generate_integration_points(
    vertex, thermal_num_integration_points, method=thermal_integration_method
)
X_f_T = np.column_stack([x_np, y_np, wt_np, elem_np])
Temp_n_f = model.T_ref * np.ones((X_f_T.shape[0], 1), dtype=np.float32)
X_f_U = make_elastic_integration_points(model, vertex, X_f_T)

load_history = np.full((numSteps, 6), np.nan, dtype=np.float64)
# columns: step, v_top_mm, T_amp_C, J_tip, J_over_Gc, crack_length

deltaT = deltaT_phys / model.DeltaT_ref
deltaV = deltaV_phys / model.U_ref

start_step = 0
if resume_from_checkpoint:
    if resume_checkpoint_path == "auto":
        _, checkpoint_path = find_latest_checkpoint(results_dir)
    else:
        checkpoint_path = resume_checkpoint_path
    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        start_step, load_history, Temp_n_f, X_f_T, vertex, parent_indices, ckpt_crack_points = restore_from_checkpoint(
            model, checkpoint_path, load_history, device, strict=strict_resume
        )
        X_f_U = make_elastic_integration_points(model, vertex, X_f_T)
        if ckpt_crack_points is not None:
            crack_points = ckpt_crack_points
            crack_points, crack_embedding = set_crack_state(model, crack_points, make_crack_embedding)
    else:
        print("[resume] No checkpoint found. Starting from step 0.")

plot_cruciform_outline(
    os.path.join(results_dir, "cruciform_geometry_initial.png"),
    crack_geometry=model.crack_geometry,
    title="Initial cruciform geometry",
    show=True,
)
plot_cruciform_mesh(
    vertex,
    os.path.join(results_dir, "cruciform_mesh_initial.png"),
    crack_geometry=model.crack_geometry,
    title="Initial cruciform mesh",
    show=True,
)

# -----------------------------------------------------------------------------
# Main simulation loop
# -----------------------------------------------------------------------------
for step in range(start_step, numSteps):
    step_dir = os.path.join(results_dir, f"step_{step}")
    os.makedirs(step_dir, exist_ok=True)

    v_delta_nd = deltaV * step
    T_delta_nd = deltaT * step
    T_delta_nd_prev = deltaT * (step - 1) if step >= 1 else 0.0

    v_delta_phys = v_delta_nd * model.U_ref
    T_delta_phys = T_delta_nd * model.DeltaT_ref

    print(f"\n=== Cruciform mode {load_type_label} step {step}/{numSteps - 1} ===")
    print(f"top v = {v_delta_phys:.6e} mm | top theta = +{T_delta_phys:.3f} C | bottom theta = -{T_delta_phys:.3f} C")
    print(f"thermal points = {X_f_T.shape[0]} | elastic points = {X_f_U.shape[0]} ({elastic_integration_method})")

    if step >= 1:
        model.snapshot_thermal_net()

    if step == 0:
        n_adam_T, n_lbfgs_T = thermal_num_Adam_step0, thermal_num_LBFGS_step0
        n_adam_U, n_lbfgs_U = elastic_num_Adam_step0, elastic_num_LBFGS_step0
    elif step == 1:
        n_adam_T, n_lbfgs_T = thermal_num_Adam_step1, thermal_num_LBFGS_step1
        n_adam_U, n_lbfgs_U = elastic_num_Adam_step1, elastic_num_LBFGS_step1
    else:
        n_adam_T, n_lbfgs_T = thermal_num_Adam, thermal_num_LBFGS
        n_adam_U, n_lbfgs_U = elastic_num_Adam, elastic_num_LBFGS

    print("Training thermal network")
    t0 = time.time()
    X_f_T_before = X_f_T.copy()
    X_f_T, Temp_n_f = model.train_thermal(
        X_f=X_f_T,
        T_delta_nd=T_delta_nd,
        Temp_n_f=Temp_n_f,
        nIter_Adam=n_adam_T,
        lr_Adam=lr_Adam,
        nIter_LBFGS=n_lbfgs_T,
        lr_LBFGS=lr_lbfgs,
        nIter_log=num_log,
        step=step,
        vertex=vertex,
        num_integration_points=thermal_num_integration_points,
        resample_every_iters=(resample_frequency_epochs_thermal if enable_resample_thermal else 0),
        T_delta_nd_prev=T_delta_nd_prev,
    )
    print(f"Thermal training time: {time.time() - t0:.2f} s")
    log_resampling_change("thermal", X_f_T_before, X_f_T)

    X_f_U = make_elastic_integration_points(model, vertex, X_f_T)
    elastic_vertex, elastic_npts, elastic_resample_every = elastic_resampling_args(vertex)

    print("Training elastic network")
    t0 = time.time()
    X_f_U_before = X_f_U.copy()
    X_f_U = model.train_elastic(
        X_f=X_f_U,
        T_delta_nd=T_delta_nd,
        v_delta_nd=v_delta_nd,
        nIter_Adam=n_adam_U,
        lr_Adam=lr_Adam,
        nIter_LBFGS=n_lbfgs_U,
        lr_LBFGS=lr_lbfgs,
        nIter_log=num_log,
        vertex=elastic_vertex,
        num_integration_points=elastic_npts,
        resample_every_iters=elastic_resample_every,
    )
    print(f"Elastic training time: {time.time() - t0:.2f} s")
    log_resampling_change("elastic", X_f_U_before, X_f_U)

    prop_iter = 0
    prev_ratio = None
    while enable_crack_propagation and step > 0 and prop_iter < max_propagations_per_step:
        if crack_reached_boundary(crack_points, model=model):
            print("[propagation] crack has reached the cruciform boundary; stop propagation.")
            break

        K1, K2, J, beta_tip = sif.estimate(crack_points, T_delta_nd, v_delta_nd)
        crack_tip_xy, T_tip_prop, Gc_tip_prop = get_tip_fracture_properties(model, crack_points, T_delta_nd)
        G_drive, theta_drive = driving_force(K1, K2, crack_tip_xy)
        ratio = G_drive / max(Gc_tip_prop, 1.0e-30)
        if not all(np.isfinite([K1, K2, J, G_drive, ratio])):
            print("[propagation] non-finite K/J/G. Stop this propagation loop.")
            break
        ok, why = extraction_is_plausible(ratio, prev_ratio,
                                          ratio_cap=extraction_ratio_cap)
        if not ok:
            print(f"[propagation] Rejected extraction at step {step} iter {prop_iter}: {why}. "
                  f"K1={K1:.4e}, K2={K2:.4e}. Stop this propagation loop.")
            break
        prev_ratio = ratio

        print(
            f"[prop-check step {step} iter {prop_iter}] "
            f"K1={K1:.4e}, K2={K2:.4e}, J_straight={J:.4e}, "
            f"G_eff={G_drive:.4e} (theta={np.rad2deg(theta_drive):+.2f} deg), "
            f"Gc={Gc_tip_prop:.4e}, G_eff/Gc={ratio:.3f}, "
            f"Lcrack={total_crack_length(crack_points, model=model):.3f} mm"
        )
        append_propagation_history([
            step, prop_iter, f"{v_delta_phys:.12e}", f"{T_delta_phys:.12e}",
            f"{K1:.12e}", f"{K2:.12e}", f"{J:.12e}",
            f"{T_tip_prop:.12e}", f"{Gc_tip_prop:.12e}", f"{ratio:.12e}",
            f"{beta_tip:.12e}", 0.0,
            f"{crack_tip_xy[0]:.12e}", f"{crack_tip_xy[1]:.12e}", 0,
        ])

        # The extraction is logged above whether or not it is admitted.
        if require_validated_extraction:
            ok, why = sif.extraction_is_validated()
            if ok and why:
                # Extension on a clipped sweep that did not pass (see
                # SIFEstimator.extraction_is_validated), reported here.
                print(f"[propagation] Released extraction at step {step} iter {prop_iter}: {why}")
            if not ok:
                print(f"[propagation] Rejected extraction at step {step} iter {prop_iter}: "
                      f"{why}. K1={K1:.4e}, K2={K2:.4e}. Stop propagation at this step.")
                break

        if ratio < min_G_over_Gc_to_propagate:
            break

        X_energy_probe = select_energy_probe_points(X_f_U, tip=model.crack_geometry.tip)
        new_tip, beta_new, theta_kink, hit_boundary = propose_crack_extension(
            model,
            crack_points,
            K1,
            K2,
            X_energy_probe=X_energy_probe,
            T_delta_nd=T_delta_nd,
            v_delta_nd=v_delta_nd,
        )
        if new_tip is None:
            print("[propagation] proposed extension is too small or outside the domain.")
            break

        crack_points.append(new_tip)
        crack_points, crack_embedding = set_crack_state(model, crack_points, make_crack_embedding)
        prop_iter += 1
        print(f"[propagation] new tip = {new_tip}, kink = {np.rad2deg(theta_kink):.2f} deg")
        if plot_crack_path_on_update:
            save_crack_path(
                crack_points,
                step_dir,
                step,
                crack_geometry=model.crack_geometry,
                prop_iter=prop_iter,
                show=show_crack_path_on_update,
            )
        append_propagation_history([
            step, prop_iter, f"{v_delta_phys:.12e}", f"{T_delta_phys:.12e}",
            f"{K1:.12e}", f"{K2:.12e}", f"{J:.12e}",
            f"{T_tip_prop:.12e}", f"{Gc_tip_prop:.12e}", f"{ratio:.12e}",
            f"{beta_new:.12e}", f"{theta_kink:.12e}",
            f"{new_tip[0]:.12e}", f"{new_tip[1]:.12e}", 1,
        ])

        if hit_boundary or crack_reached_boundary(crack_points, model=model):
            print("[propagation] crack reached the cruciform boundary after this extension.")
            break

        print("[propagation] retraining thermal network after crack extension")
        X_f_T_before = X_f_T.copy()
        X_f_T, Temp_n_f = model.train_thermal(
            X_f=X_f_T,
            T_delta_nd=T_delta_nd,
            Temp_n_f=Temp_n_f,
            nIter_Adam=prop_thermal_num_Adam,
            lr_Adam=lr_Adam,
            nIter_LBFGS=prop_thermal_num_LBFGS,
            lr_LBFGS=lr_lbfgs,
            nIter_log=num_log,
            step=step,
            vertex=vertex,
            num_integration_points=thermal_num_integration_points,
            resample_every_iters=(resample_frequency_epochs_thermal if enable_resample_thermal else 0),
            T_delta_nd_prev=T_delta_nd_prev,
        )
        log_resampling_change("thermal-propagation", X_f_T_before, X_f_T)

        print("[propagation] retraining elastic network after crack extension")
        X_f_U = make_elastic_integration_points(model, vertex, X_f_T)
        elastic_vertex, elastic_npts, elastic_resample_every = elastic_resampling_args(vertex)
        X_f_U_before = X_f_U.copy()
        X_f_U = model.train_elastic(
            X_f=X_f_U,
            T_delta_nd=T_delta_nd,
            v_delta_nd=v_delta_nd,
            nIter_Adam=prop_elastic_num_Adam,
            lr_Adam=lr_Adam,
            nIter_LBFGS=prop_elastic_num_LBFGS,
            lr_LBFGS=lr_lbfgs,
            nIter_log=num_log,
            vertex=elastic_vertex,
            num_integration_points=elastic_npts,
            resample_every_iters=elastic_resample_every,
        )
        log_resampling_change("elastic-propagation", X_f_U_before, X_f_U)

    if prop_iter >= max_propagations_per_step:
        print(f"[propagation] reached max_propagations_per_step={max_propagations_per_step}.")

    T_pred = model.predict_thermal(X_f_T[:, :2], T_delta_nd)
    Temp_n_f = T_pred.reshape(-1, 1)

    crack_tip_xy, T_tip, Gc_tip = get_tip_fracture_properties(model, crack_points, T_delta_nd)
    if crack_reached_boundary(crack_points, model=model):
        K1_tip = K2_tip = J_tip = G_tip = theta_tip = 0.0
    else:
        K1_tip, K2_tip, J_tip, beta_tip = sif.estimate(crack_points, T_delta_nd, v_delta_nd)
        G_tip, theta_tip = driving_force(K1_tip, K2_tip, crack_tip_xy)
        sif.save_validation(step_dir, step, crack_points, T_delta_nd, v_delta_nd)

    # The J_tip and J_over_Gc columns hold the energy release rate of the kink
    # G(theta_c) and its ratio to Gc; the straight-ahead J is kept as
    # J_straight, as in sen_shear_fgm.py.
    J_straight_tip = J_tip
    J_tip = G_tip

    load_history[step, :] = [
        step,
        v_delta_phys,
        T_delta_phys,
        J_tip,
        J_tip / max(Gc_tip, 1.0e-30),
        total_crack_length(crack_points, model=model),
    ]
    np.save(os.path.join(results_dir, "load_history.npy"), load_history)
    with open(os.path.join(results_dir, "load_history.csv"), "w", encoding="utf-8") as f:
        f.write("step,v_top_mm,T_amp_C,J_tip,J_over_Gc,crack_length\n")
        for row in load_history[np.isfinite(load_history[:, 0])]:
            f.write(",".join(f"{v:.12e}" for v in row) + "\n")

    X_plot = make_prediction_grid(domainEndsX, domainEndsY, n=plot_grid_n)
    fields_plot = model.predict_fields(X_plot, T_delta_nd, v_delta_nd)
    fields_display = dict(fields_plot)
    for key in ["T", "u", "v", "sx", "sy", "sxy", "von_mises"]:
        fields_display[key] = mask_crack_band_for_plot(
            X_plot,
            fields_plot[key],
            model.crack_geometry,
            width=crack_plot_band_width,
            tip_radius=crack_plot_tip_radius,
            domain_mask_fn=inside_cruciform_np,
        )
    fields_display["gamma"] = mask_outside_domain(X_plot, fields_plot["gamma"])

    np.savez(
        os.path.join(step_dir, f"fields_step_{step}.npz"),
        X_f_T=X_f_T,
        X_f_U=X_f_U,
        X_plot=X_plot,
        T_plot=fields_plot["T"],
        u_plot=fields_plot["u"],
        v_plot=fields_plot["v"],
        sx_plot=fields_plot["sx"],
        sy_plot=fields_plot["sy"],
        sxy_plot=fields_plot["sxy"],
        von_mises_plot=fields_plot["von_mises"],
        gamma_plot=fields_plot["gamma"],
        crack_points=np.asarray(crack_points, dtype=np.float64),
        crack_curve_points=np.asarray(model.crack_geometry.curve_points, dtype=np.float64),
    )

    np.savez(
        os.path.join(step_dir, f"fracture_tip_step_{step}.npz"),
        crack_tip=crack_tip_xy,
        T_tip=T_tip,
        Gc_tip=Gc_tip,
        J_tip=J_tip,
        J_over_Gc=J_tip / max(Gc_tip, 1.0e-30),
        K1_tip=K1_tip,
        K2_tip=K2_tip,
        crack_points=np.asarray(crack_points, dtype=np.float64),
        crack_curve_points=np.asarray(model.crack_geometry.curve_points, dtype=np.float64),
    )
    save_crack_path(crack_points, step_dir, step, crack_geometry=model.crack_geometry)

    tip_history_file = os.path.join(results_dir, "fracture_tip_history.csv")
    if step == 0 or not os.path.exists(tip_history_file):
        with open(tip_history_file, "w", encoding="utf-8") as f:
            # J_straight and theta_deg are the last columns, as in sen_shear_fgm.py.
            f.write("step,v_top_mm,T_amp_C,T_tip_C,Gc_tip,J_tip,J_over_Gc,K1_tip,K2_tip,"
                    "crack_length,tip_x,tip_y,J_straight,theta_deg\n")
    with open(tip_history_file, "a", encoding="utf-8") as f:
        f.write(
            f"{step},{v_delta_phys:.12e},{T_delta_phys:.12e},{T_tip:.12e},{Gc_tip:.12e},"
            f"{J_tip:.12e},{J_tip / max(Gc_tip, 1.0e-30):.12e},{K1_tip:.12e},{K2_tip:.12e},"
            f"{total_crack_length(crack_points, model=model):.12e},{crack_tip_xy[0]:.12e},{crack_tip_xy[1]:.12e},"
            f"{J_straight_tip:.12e},{np.rad2deg(theta_tip):.12e}\n"
        )

    mesh_meta = {
        "benchmark": f"notched_cruciform_type_{load_type_label}",
        "geometry_mm": {"arm_half_width": arm_half_width, "half_size": half_size, "notch_length": notch_length},
        "E": model.E,
        "nu": model.nu,
        "alpha": model.alpha,
        "Gc0": Gc0,
        "plane_type": model.plane_type,
        "U_ref": model.U_ref,
        "DeltaT_ref": model.DeltaT_ref,
        "crack_increment": crack_increment,
        "crack_embedding_decay_ell": crack_embedding_decay_ell,
        "tip_enrichment_decay_factor": model.tip_enrichment_decay_factor,
        "crack_points": np.asarray(crack_points, dtype=np.float64).tolist(),
        "crack_path": np.asarray(model.crack_geometry.curve_points, dtype=np.float64).tolist(),
        "J_tip": J_tip,
        "J_over_Gc": J_tip / max(Gc_tip, 1.0e-30),
        "K1_tip": K1_tip,
        "K2_tip": K2_tip,
    }
    model.save_model(step, load_history, Temp_n_f, X_f_T, vertex, parent_indices, step_dir, mesh_policy=mesh_policy, mesh_meta=mesh_meta)

    save_convergence_plot(
        model.thermal_adam_loss_history,
        model.thermal_lbfgs_loss_history,
        os.path.join(step_dir, f"thermal_convergence_{step}.png"),
        f"Thermal convergence, step {step}",
    )
    save_convergence_plot(
        model.elastic_adam_loss_history,
        model.elastic_lbfgs_loss_history,
        os.path.join(step_dir, f"elastic_convergence_{step}.png"),
        f"Elastic convergence, step {step}",
    )

    plot_field_smooth(X_plot, fields_display["T"], domainEndsX, domainEndsY,
                      title=f"Temperature - Step {step}", label="Temperature (C)",
                      filename=f"temperature_{step}", save_dir=step_dir, cmap="coolwarm", show=True)
    plot_field_smooth(X_plot, fields_display["gamma"], domainEndsX, domainEndsY,
                      title=f"Crack embedding function, step {step}", label="Gamma",
                      filename=f"crack_gamma_{step}", save_dir=step_dir, cmap="magma",
                      vmin=-1.0, vmax=1.0, show=True, levels=200)
    plot_field_smooth(X_plot, fields_display["u"], domainEndsX, domainEndsY,
                      title=f"x-displacement - Step {step}", label="u (mm)",
                      filename=f"u_displacement_{step}", save_dir=step_dir, cmap="viridis", show=True)
    plot_field_smooth(X_plot, fields_display["v"], domainEndsX, domainEndsY,
                      title=f"y-displacement - Step {step}", label="v (mm)",
                      filename=f"v_displacement_{step}", save_dir=step_dir, cmap="viridis", show=True)
    plot_field_smooth(X_plot, fields_display["von_mises"], domainEndsX, domainEndsY,
                      title=f"Von Mises stress - Step {step}", label="Von Mises",
                      filename=f"von_mises_{step}", save_dir=step_dir, cmap="viridis",
                      show=True, levels=300)

    print(
        f"step={step}, v_top={v_delta_phys:.6e} mm, T_amp={T_delta_phys:.3f} C, "
        f"G_eff={J_tip:.4e}, G_eff/Gc={J_tip / max(Gc_tip, 1.0e-30):.3f}, "
        f"tip=({crack_tip_xy[0]:.3f},{crack_tip_xy[1]:.3f}), "
        f"Lcrack={total_crack_length(crack_points, model=model):.3f} mm"
    )

print(f"Total simulation time: {time.time() - start_time:.2f} seconds")
