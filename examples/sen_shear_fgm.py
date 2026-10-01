#!/usr/bin/env python
"""
Single-edge notched shear test of a functionally graded plate under thermal load.

This is the example "Single-edge notched shear test with functionally graded
material" of the paper.  The test is the thermo-elastic shear test of
Tangella et al. (2022) in the graded form of their released code, from which
the graded material properties are taken.

Geometry.  Unit square [0, 1] x [0, 1] mm in plane strain; energies and forces
are per unit thickness.  A horizontal edge notch runs from (0, 0.5) to
(0.5, 0.5).

Material.  Each property m varies linearly in x, m(x) = m1 + x * (m2 - m1),
between its value m1 at the left edge (x = 0) and m2 at the right edge
(x = 1):

    Young's modulus E                   340 GPa  -> 450 GPa
    Poisson's ratio nu                  0.22 (uniform)
    thermal expansion coefficient alpha 8.0e-6 1/K (uniform)
    thermal conductivity k              300 -> 350 W/(m K)
    density rho                         2450 -> 4000 kg/m^3
    specific heat capacity c            0.775 J/(kg K) (uniform)
    critical energy release rate Gc     42.47 -> 120 J/m^2
    reference temperature T_ref         300 K

The temperature-dependent fracture energy at the tip is the graded value
Gc(x_tip) times the factor 1 - b1 * s + b2 * s^2 of Ruan et al. (2023), with
s = (T_tip - T_ref) / T_max, b1 = 1.80, b2 = 1.10 and T_max = 1000 K, so the
spatial and the thermal dependence act multiplicatively.

Mechanical boundary conditions, imposed exactly through the form of the
displacement approximation:
    y = 0: u = 0,       v = 0
    y = 1: u = delta_u, v = 0
The two vertical edges are traction free.

Thermal boundary conditions: T = T_ref on y = 0 and T = T_ref + Delta T on
y = 1, both imposed exactly; the vertical edges are insulated, and the crack
faces are adiabatic (a weak penalty on the normal heat flux).  Heat conduction
is transient, with a time step of 2.5e-8 s per load step.

Loading.  141 load steps of 2.5e-5 mm of horizontal top-edge displacement, up
to 3.5e-3 mm.  The temperature difference Delta T is applied in equal
increments over the first 12 load steps (3.0e-4 mm of applied displacement)
and held from there on.  The paper reports Delta T = +30 K (heating), -30 K
(cooling) and 0 K.  At 0 K the temperature stays uniform and the thermal field
is solved at load step 0 only.

Each load step is a staggered thermal then mechanical solve, followed by the
crack growth check: K_I and K_II from the interaction integral (with a
crack-opening displacement fit as fallback), the energy release rate of the
kink G(theta_c) compared with Gc at the tip temperature, and, while
G(theta_c) >= Gc, a crack increment of 0.025 mm in the maximum hoop stress
direction, after which both fields are retrained at the same load step.  The
specimen counts as separated once the tip is within 0.03 mm of the right or
the bottom edge, and the reaction force is zero from then on.

Command line:
    --dT      total temperature change in K: 30 (default), -30 or 0
    --seed    seed of the network initialization (default 2025)
    --resume  continue from the latest checkpoint in the output folder
    --out     output folder (default runs/sen_shear_fgm_dT_<dT>)

Output, in results/ inside the output folder:
    fdGraph.npy, FD.png             force-displacement curve
    FD_compare.png                  the same against a reference curve, if a
                                    two-column file (displacement, force) is
                                    found: FEM.txt or fem_FD.txt next to this
                                    script, or FEM.txt in the output folder
                                    or in results/
    fracture_tip_history.csv        one row per load step: tip position and
                                    temperature, Gc, K_I, K_II, G(theta_c),
                                    crack length and reaction force
    crack_propagation_history.csv   one row per criterion evaluation and one
                                    per crack extension
    step_<n>/                       checkpoint_step_<n>.pt, fields_step_<n>.npz,
                                    fracture_tip_step_<n>.npz, the crack path,
                                    the extraction validation plots, field
                                    plots and training loss histories

    python examples/sen_shear_fgm.py --dT 30     # heated by 30 K
    python examples/sen_shear_fgm.py --dT -30    # cooled by 30 K
    python examples/sen_shear_fgm.py --dT 0      # isothermal

Author: Han Zhang (han.zhang7@unsw.edu.au)
"""

import os
import argparse
import time
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
    load_curve_txt,
    mark_elements_for_refinement,
    plot_field,
    plot_force_disp,
    plot_force_disp_compare,
    plot_mesh,
    set_seed,
    update_adaptive_mesh,
)

from tmdem.sif_estimation import (
    SIFEstimator,
    effective_energy_release_rate,
    local_material_at_tip,
    make_rect_boundary_distance,
    crack_reached_boundary_rect,
    get_active_crack_geometry,
    last_tangent_normal,
    plane_strain_flag,
    propose_crack_extension_rect,
    sample_crack_tip_temperature,
    set_crack_state,
    total_crack_length,
)

from tmdem.utils import (
    find_latest_checkpoint,
    log_resampling_change,
    make_prediction_grid,
    mask_crack_band_for_plot,
    restore_from_checkpoint,
    save_crack_path,
)

plt.close("all")

# -----------------------------------------------------------------------------
# Command line: load case, seed and output folder.  The defaults are the
# settings of the runs reported in the paper.
# -----------------------------------------------------------------------------
cli = argparse.ArgumentParser(description='Single-edge notched shear test of a functionally graded plate under thermal load.')
cli.add_argument("--dT", type=float, default=30.0,
                 help="total temperature change in K: 30 (heating, default), -30 (cooling) or 0")
cli.add_argument("--seed", type=int, default=2025,
                 help="seed of the network initialization (default 2025)")
cli.add_argument("--resume", action="store_true",
                 help="continue from the latest checkpoint in the output folder")
cli.add_argument("--out", default=None,
                 help="output folder (default runs/sen_shear_fgm_dT_<dT>)")
cli_args = cli.parse_args()
out_dir = cli_args.out or os.path.join("runs", f"sen_shear_fgm_dT_{cli_args.dT:g}")
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

# Resume from the latest completed checkpoint when enabled (--resume).
# A checkpoint is written at every load step, so an interrupted run can be
# continued in the same output folder.
resume_from_checkpoint = cli_args.resume
resume_checkpoint_path = "auto"   # "auto" or an explicit checkpoint path
strict_resume = False             # passed as strict to load_state_dict; False tolerates missing or extra keys

start_time = time.time()

# -----------------------------------------------------------------------------
# Temperature-dependent fracture energy
# -----------------------------------------------------------------------------
# Gc(T) of Ruan et al. (2023), evaluated at the tip temperature T_tip and used
# in the propagation criterion G(theta_c) >= Gc(T_tip).  Gc0 (N/mm) is the
# value used without grading; with grading the graded Gc(x) takes its place.
use_temperature_dependent_Gc = True

Gc0 = 0.04247
Gc_b1 = 1.80
Gc_b2 = 1.10
Gc_Tref = 300.0      # K
Gc_Tmax = 1000.0     # K
Gc_min_factor = 0.03 # lower bound on the temperature factor, keeps Gc positive

# -----------------------------------------------------------------------------
# Material and model parameters
# -----------------------------------------------------------------------------
model_params = {
    # mm-N-s-K unit system.
    # Properties graded linearly in x.  On the domain [0, 1] x [0, 1] the
    # interpolation is m(x) = m1 + x * (m2 - m1), with m1 at x = 0 and m2 at
    # x = 1.
    "use_fgm": True,

    # The scalar values are those the shared library reads, and the values
    # used without grading.  The subclass below uses the *1 and *2 values in
    # the energy, stress, post-processing and SIF calculations.
    "E": 340.0e3,        # N/mm^2 = MPa
    "E1": 340.0e3,       # 340 GPa
    "E2": 450.0e3,       # 450 GPa

    "nu": 0.22,
    "nu1": 0.22,
    "nu2": 0.22,

    "alpha": 8.0e-6,     # 1/K
    "alpha1": 8.0e-6,
    "alpha2": 8.0e-6,
    # In plane strain the thermal strain of the in-plane constitutive law uses
    # the effective coefficient (1 + nu) * alpha = 9.76e-6 1/K, the exact
    # plane-strain reduction, as in the phase-field reference
    # (fem/pf_fem.py, --alpha_mode ps3d).
    "plane_strain_thermal_alpha_correction": True,

    # Energies and forces are per unit thickness.
    "k": 300.0,          # W/(m K) = mJ/(s mm K)
    "k1": 300.0,
    "k2": 350.0,

    "rho": 2.45e-6,      # kg/mm^3
    "rho1": 2.45e-6,
    "rho2": 4.00e-6,

    "c": 775.0,          # mJ/(kg K)
    "c1": 775.0,
    "c2": 775.0,

    # Critical energy release rate in N/mm at x = 0 and x = 1.
    # 42.47 J/m^2 = 0.04247 N/mm.
    "Gc1": 0.04247,
    "Gc2": 0.120,

    "T_ref": 300.0,      # K
    "dt": 2.5e-8,          # s
    "plane_type": "plane_strain",

    # Nondimensional load scales.
    "U_ref": 1.0e-4,        # mm
    "DeltaT_ref": 100.0,    # K

    # Unit square domain.
    "lb": np.array([0.0, 0.0]),
    "ub": np.array([1.0, 1.0]),

    # Crack embedding: both networks receive [x, y, Gamma].  The optional
    # explicit jump terms (Gamma times an additional network output) are off.
    "use_thermal_jump": False,
    "use_explicit_mechanical_jump": False,

    # Adiabatic crack faces, as a weak penalty on the normal heat flux at
    # points sampled on both faces.
    "adiabatic_crack_weight": 1.0e-6,
    "adiabatic_crack_samples": 150,
    "crack_face_offset": 1.0e-4,

    # Smoothed conductivity band is off by default.
    "use_crack_conductivity_degradation": False,
    "crack_band_width": 0.005,
    "crack_min_conductivity_ratio": 1.0e-6,

    # Integration points densified near the tip (see
    # generate_integration_points): background elements whose centre lies
    # within tip_densify_radius of the tip receive tip_densify_factor times as
    # many Monte Carlo points, on the same fixed 100 x 100 mesh and with
    # weights area/n.
    "tip_densify_radius": 0.08,
    "tip_densify_factor": 8,

    # Williams near-tip enrichment.
    "use_tip_enrichment": True,
    "tip_enrichment_q": 1.0,
    "tip_enrichment_decay_factor": 5.0,
    "tip_enrichment_u_scale": 1.0,
}

crack_points = [[0.0, 0.5], [0.5, 0.5]]
crack_geometry = None
# Initial notch length; resistance_curve_factor measures the crack extension from it.
initial_crack_length = float(np.sum(np.linalg.norm(np.diff(np.asarray(crack_points, dtype=np.float64), axis=0), axis=1)))


# -----------------------------------------------------------------------------
# Mesh and integration settings
# -----------------------------------------------------------------------------
# Fixed background mesh with Monte Carlo integration.
mesh_policy = "fixed"  # "fixed" or "adaptive"
domainEndsX = np.array([0.0, 1.0])
domainEndsY = np.array([0.0, 1.0])

# 100 x 100 background elements of size h = 0.010 mm with 9 Monte Carlo
# integration points each, 90000 in total, as in the tension example.
nx_elem = 100
ny_elem = 100

# Thermal integration uses Monte Carlo (MC) points with resampling.
thermal_num_integration_points = 9
thermal_integration_method = "mc"

# Elastic integration can use MC or Gaussian points.
elastic_integration_method = "mc"
elastic_num_integration_points = 9       # used only when elastic_integration_method == "mc"
elastic_gauss_order = 3                  # gives elastic_gauss_order^2 points per element

# Adaptive refinement is disabled unless mesh_policy == "adaptive".
enable_refine_elastic = mesh_policy.lower() == "adaptive"
error_percentile = 10
max_refinement_level = 2
num_refine_cycles_step0 = 0 if mesh_policy.lower() == "fixed" else 1
num_refine_cycles_following = 0 if mesh_policy.lower() == "fixed" else 1

# -----------------------------------------------------------------------------
# Loading schedule
# -----------------------------------------------------------------------------
# The temperature difference is ramped against the applied displacement, not
# against the step index, so refining the load step leaves the physical
# schedule unchanged.  With dt = 2.5e-8 s per load step of 2.5e-5 mm the
# displacement rate is 1 m/s; the full difference is reached at 3.0e-4 mm
# (12 load steps, 3e-7 s) and held from there on.
T_total_K = cli_args.dT          # total temperature change; sign selects heating or cooling
T_ramp_disp = 3.0e-4     # mm of applied displacement over which it is reached

deltaU_phys = 2.5e-5     # mm per horizontal mechanical loading step

# With no thermal loading the temperature field is uniform: it is solved
# once at step 0 and skipped afterwards.
skip_thermal_after_step0 = (T_total_K == 0.0)

# Load steps up to an applied displacement of 3.5e-3 mm (141 load steps),
# beyond the displacement at which the specimen separates.
numSteps = int(round(3.5e-3 / deltaU_phys)) + 1

# Training iterations.
thermal_num_Adam_step0 = 3000
thermal_num_LBFGS_step0 = 3000
thermal_num_Adam_step1 = 5000
thermal_num_LBFGS_step1 = 0
thermal_num_Adam = 3000
thermal_num_LBFGS = 0

elastic_num_Adam_step0 = 3000
elastic_num_LBFGS_step0 = 1500
elastic_num_Adam_step1 = 5000
elastic_num_LBFGS_step1 = 0
elastic_num_Adam = 3000
# The per-step elastic L-BFGS stays at 0.  L-BFGS runs on a fixed Monte
# Carlo point set, between whose points the crack-embedded elastic field can
# form spurious compliant layers; Adam with periodic resampling of the
# integration points avoids this.
elastic_num_LBFGS = 0

lr_Adam = 1.0e-3
lr_lbfgs = 0.1
num_log = 100

# Resampling controls.
enable_resample_thermal = True
enable_resample_elastic = True
resample_frequency_epochs_thermal = 500
resample_frequency_epochs_elastic = 500

# -----------------------------------------------------------------------------
# Crack propagation
# -----------------------------------------------------------------------------
enable_crack_propagation = True
propagation_mode = "max_hoop"      # "straight" or "max_hoop"
# Crack increment Delta a per propagation event.  The interaction contour
# radius of 0.0225 mm = 0.9 * Delta a fits inside the straight segment left
# by one extension (see sif_limit_by_last_segment).  SIFEstimator also uses
# 0.5 * Delta a as the lower bound of the COD sampling window.
crack_increment = 0.025             # mm per propagation event
max_propagations_per_step = 50
min_G_over_Gc_to_propagate = 1.0   # chi_c
# Grow the crack only on an admissible extraction, one whose radius sweep
# passes (SIFEstimator.extraction_is_validated, which lets a limited number
# of clipped sweeps through directly after a kink).
require_validated_extraction = True
max_kink_angle_deg = 80.0

# Optional crack growth resistance curve (see resistance_curve_factor), off.
# When enabled, the resistance is flat (R = 1) up to r_curve_onset of crack
# extension and then rises linearly to r_curve_eta over r_curve_length; it
# multiplies Gc at the tip.  With use_r_curve = False the factor is 1 and the
# three parameters below have no effect.
use_r_curve = False
r_curve_eta = 1.5        # R at saturation / R(0), used only if use_r_curve
r_curve_onset = 0.50     # mm of extension at R = 1 before the rise
r_curve_length = 0.05    # mm of extension over which R then rises to eta
# Kink angle without relaxation: the crack turns by the full maximum hoop
# stress angle.  The deadband suppresses turns driven by noise in K_II: there
# is no kink while |K_II| / |K_I| is below k_ratio_deadband.
kink_angle_relaxation = 1.0
k_ratio_deadband = 0.05

# Boundaries at which the specimen counts as separated (see ligament_margin);
# the reaction force is zero from then on.
zero_force_boundaries = ("right", "bottom")

# COD fallback sampling window.
sif_cod_num_points = 15
# Two-sided face-sampling offset: float32 noise << offset << sif_r_min, and
# crack_embedding_decay_alpha * offset << 1, so that the embedded jump is not
# attenuated.
sif_cod_offset = 5.0e-4
# Extraction length scales, written in units of the background element size h
# rather than of the crack increment, so that changing the crack increment
# does not change the extraction.
background_cell_size = (domainEndsX[1] - domainEndsX[0]) / nx_elem   # 0.010 mm
sif_r_min = 1.25 * background_cell_size    # 0.0125 mm, start of the COD window
sif_r_max = 6.25 * background_cell_size    # 0.0625 mm, end of the COD window
# Keep the interaction contour and the COD window inside the last straight
# segment, so that neither encloses the corner of a fresh kink.
sif_limit_by_last_segment = True

# Interaction integral first, COD fallback second.
sif_estimator = "interaction_with_cod_fallback"  # "interaction_with_cod_fallback" or "cod"
# Radius of the interaction-integral contour, 2.25 background elements.
sif_interaction_radius = 2.25 * background_cell_size    # 0.0225 mm
# Separation margin: the specimen counts as separated once the tip comes
# within ligament_margin of a zero-force boundary.
ligament_margin = 0.03
# Number of points on the circular contour of the contour term (periodic
# trapezoidal rule).
sif_interaction_num_points = 2880

# Print COD fallback sampling range.
print_sif_range_diagnostics = True

# -----------------------------------------------------------------------------
# Crack embedding parameters
# -----------------------------------------------------------------------------
# Normal decay of the crack embedding: the envelope is exp(-alpha * dist^p),
# so with p = 1 its e-folding length is 1/alpha, here two background elements.
crack_embedding_decay_length = 2.0 * background_cell_size    # 0.020 mm
crack_embedding_decay_alpha = 1.0 / crack_embedding_decay_length    # 50.0
crack_embedding_decay_p = 1.0
crack_embedding_endpoint_overlap = 1.25 * background_cell_size    # 0.0125 mm

# Retraining after each crack extension, at the same load step.
prop_thermal_num_Adam = 1000
prop_thermal_num_LBFGS = 0
# Before the next K estimate the elastic field receives the Adam budget of a
# full load step (with resampling); no L-BFGS, see elastic_num_LBFGS.
prop_elastic_num_Adam = 3000
prop_elastic_num_LBFGS = 0

def spatial_Gc_at_x(x):
    """Graded critical energy release rate Gc(x) in N/mm at x in [0, 1].

    Linear between Gc1 at x = 0 and Gc2 at x = 1; Gc0 without grading.
    """
    if not model_params.get("use_fgm", False):
        return float(Gc0)
    xi = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    return model_params["Gc1"] + xi * (model_params["Gc2"] - model_params["Gc1"])


def temperature_dependent_Gc(T_abs, x=None):
    """Temperature-dependent fracture energy Gc(T, x) in N/mm.

    Gc(T, x) = Gc(x) * [1 - b1 * s + b2 * s^2] with s = (T - T_ref) / T_max,
    after Ruan et al. (2023).  Gc(x) is the graded value of spatial_Gc_at_x,
    or Gc0 when x is None; the bracketed factor is bounded below by
    Gc_min_factor.
    """
    T_abs = np.asarray(T_abs, dtype=np.float64)
    theta = (T_abs - Gc_Tref) / Gc_Tmax
    factor = 1.0 - Gc_b1 * theta + Gc_b2 * theta**2
    factor = np.maximum(factor, Gc_min_factor)
    Gc_ref = spatial_Gc_at_x(x) if x is not None else float(Gc0)
    return Gc_ref * factor


def resistance_curve_factor(crack_points, model=None):
    """Optional crack growth resistance factor R(da)/R(0), which multiplies Gc.

    da is the crack extension beyond the initial notch.  Returns 1 when
    use_r_curve is False, as in this example.  Otherwise R = 1 up to
    da = r_curve_onset, followed by a linear rise to r_curve_eta over
    r_curve_length.
    """
    if not use_r_curve:
        return 1.0
    da = max(total_crack_length(crack_points, model=model) - initial_crack_length, 0.0)
    s = (da - r_curve_onset) / max(r_curve_length, 1.0e-12)
    return 1.0 + (r_curve_eta - 1.0) * min(max(s, 0.0), 1.0)


def driving_force(K1, K2, crack_tip_xy):
    """Energy release rate of the kink G(theta_c) and the kink angle theta_c.

    Evaluated with ``effective_energy_release_rate`` and the modulus
    E' = E / (1 - nu^2) in plane strain (E' = E in plane stress) of the graded
    material at the tip.  Unlike the straight-ahead J = (K1^2 + K2^2)/E',
    G(theta_c) carries the K1*K2 cross term and so distinguishes a thermal load
    that opens the crack from one that closes it.  The kink angle uses the same
    cap, deadband and relaxation as the direction search.
    """
    E_tip, nu_tip, _ = local_material_at_tip(model, float(crack_tip_xy[0]))
    E_eff = E_tip / (1.0 - nu_tip ** 2) if plane_strain_flag(model) else E_tip
    return effective_energy_release_rate(
        K1, K2, E_eff,
        max_kink_angle_deg=max_kink_angle_deg,
        k_ratio_deadband=k_ratio_deadband,
        relaxation=kink_angle_relaxation,
    )


def get_tip_fracture_properties(model, crack_points, T_delta_nd):
    """Return the tip position, the tip temperature T_tip and Gc at the tip."""
    geom = get_active_crack_geometry(model, crack_points)
    crack_tip_xy = np.asarray(geom.tip, dtype=np.float64)
    # Tip temperature as the mean over a ring of 24 points around the tip, with a
    # ring radius much smaller than the crack length and the thermal boundary
    # layer.
    T_tip = sample_crack_tip_temperature(model, crack_tip_xy, T_delta_nd,
                                         radius=2.0e-3, n_ring=24)
    Gc_tip = float(temperature_dependent_Gc(T_tip, crack_tip_xy[0])) if use_temperature_dependent_Gc else float(spatial_Gc_at_x(crack_tip_xy[0]))
    # The grading, the temperature dependence and the optional resistance
    # curve act multiplicatively.
    Gc_tip *= resistance_curve_factor(crack_points, model=model)
    return crack_tip_xy, T_tip, Gc_tip


def make_elastic_integration_points(model, vertex, X_f_thermal):
    """Create the integration batch used by the elastic energy solve.

    If elastic_integration_method == "mc", the elastic solve starts from the
    latest thermal Monte Carlo batch, so both fields are sampled at the same
    points until the elastic resampling draws new ones.

    If elastic_integration_method == "gaussian", the elastic solve uses its own
    deterministic Gauss batch, and the thermal and elastic integration points
    differ.
    """
    method = elastic_integration_method.lower()
    if method == "mc":
        return X_f_thermal.copy()

    if method in ["gauss", "gaussian"]:
        x_np, y_np, wt_np, elem_np = model.generate_integration_points(
            vertex,
            elastic_num_integration_points,
            method="gaussian",
            gauss_order=elastic_gauss_order,
        )
        return np.column_stack([x_np, y_np, wt_np, elem_np])

    raise ValueError(
        "elastic_integration_method must be 'mc' or 'gaussian', "
        f"got {elastic_integration_method!r}"
    )


def elastic_resampling_args():
    """Return elastic resampling controls.

    Elastic resampling applies only to Monte Carlo integration; a Gaussian
    batch stays fixed.
    """
    use_mc = elastic_integration_method.lower() == "mc"
    if use_mc and enable_resample_elastic:
        return vertex, elastic_num_integration_points, resample_frequency_epochs_elastic
    return None, elastic_num_integration_points, 0


def append_propagation_history(row):
    path = os.path.join(results_dir, "crack_propagation_history.csv")
    write_header = not os.path.exists(path)
    with open(path, "a", encoding="utf-8") as f:
        if write_header:
            # The J and J_over_Gc columns hold the energy release rate of the
            # kink G(theta_c) from driving_force and its ratio to Gc at the tip.
            f.write("step,prop_iter,u_delta_mm,T_delta_K,K1,K2,J,T_tip_K,Gc_tip,J_over_Gc,beta,theta,tip_x,tip_y,propagated\n")
        f.write(",".join(str(x) for x in row) + "\n")


# -----------------------------------------------------------------------------
# Networks with the crack embedding as input
# -----------------------------------------------------------------------------

def make_crack_embedding(points):
    """Build the crack embedding shared by the displacement and temperature networks."""
    points = clean_crack_points(points)
    if len(points) < 2:
        points = [[0.0, 0.5], [0.5, 0.5]]
    return _build_crack_embedding(
        points,
        decay_alpha=crack_embedding_decay_alpha,
        decay_p=crack_embedding_decay_p,
        endpoint_overlap=crack_embedding_endpoint_overlap,
    )


class ShearTopEdgeThermoSolver2D(ThermoSolver2D):
    """Single-edge notched shear specimen with properties graded linearly in x.

    Overrides the displacement approximation for the shear boundary
    conditions and evaluates every material property pointwise from the
    grading in the energies, stresses, residual and reaction force.
    """

    def __init__(self, params, model_uv=None, model_T=None, crack_geometry=None):
        super().__init__(params, model_uv=model_uv, model_T=model_T, crack_geometry=crack_geometry)
        self.use_fgm = bool(params.get("use_fgm", False))
        self.E1 = float(params.get("E1", self.E))
        self.E2 = float(params.get("E2", self.E))
        self.nu1 = float(params.get("nu1", self.nu))
        self.nu2 = float(params.get("nu2", self.nu))
        self.k1 = float(params.get("k1", self.k))
        self.k2 = float(params.get("k2", self.k))
        self.rho1 = float(params.get("rho1", self.rho))
        self.rho2 = float(params.get("rho2", self.rho))
        self.c1 = float(params.get("c1", self.c))
        self.c2 = float(params.get("c2", self.c))
        self.alpha1 = float(params.get("alpha1", self.alpha))
        self.alpha2 = float(params.get("alpha2", self.alpha))

    @staticmethod
    def _safe_grad(value, wrt):
        grad = torch.autograd.grad(
            value,
            wrt,
            grad_outputs=torch.ones_like(value),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if grad is None:
            grad = torch.zeros_like(wrt)
        return grad

    def _fgm_xi(self, x):
        xi = (x - float(self.lb[0])) / (float(self.ub[0]) - float(self.lb[0]))
        return torch.clamp(xi, 0.0, 1.0)

    def material_fields(self, x):
        """Return E, nu, lambda, mu, k, rho, c and alpha at coordinate x."""
        if self.use_fgm:
            xi = self._fgm_xi(x)
            E = self.E1 + xi * (self.E2 - self.E1)
            nu = self.nu1 + xi * (self.nu2 - self.nu1)
            k = self.k1 + xi * (self.k2 - self.k1)
            rho = self.rho1 + xi * (self.rho2 - self.rho1)
            c = self.c1 + xi * (self.c2 - self.c1)
            alpha = self.alpha1 + xi * (self.alpha2 - self.alpha1)
        else:
            E = torch.ones_like(x) * self.E
            nu = torch.ones_like(x) * self.nu
            k = torch.ones_like(x) * self.k
            rho = torch.ones_like(x) * self.rho
            c = torch.ones_like(x) * self.c
            alpha = torch.ones_like(x) * self.alpha

        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        mu = E / (2.0 * (1.0 + nu))
        return E, nu, lam, mu, k, rho, c, alpha

    def material_at_x_np(self, x):
        """Return local E, nu and mu for scalar/numpy post-processing."""
        if self.use_fgm:
            xi = np.clip((float(x) - float(self.lb[0])) / (float(self.ub[0]) - float(self.lb[0])), 0.0, 1.0)
            E = self.E1 + xi * (self.E2 - self.E1)
            nu = self.nu1 + xi * (self.nu2 - self.nu1)
        else:
            E = self.E
            nu = self.nu
        mu = E / (2.0 * (1.0 + nu))
        return E, nu, mu

    def net_uv(self, x: torch.Tensor, y: torch.Tensor, v_delta_nd: torch.Tensor):
        """Physical displacement with the shear boundary conditions imposed exactly.

        u = v = 0 on y = 0, and u = delta_u, v = 0 on y = 1.  The network part
        and the asymptotic enrichment at the tip (Williams terms) are both
        multiplied by y * (1 - y), which vanishes on the two edges.
        """
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

        # Exact boundary conditions on the bottom and top edges.
        vertical_lift = y * (1.0 - y)

        u_nd = y * v_delta_nd \
             + vertical_lift * u_hat \
             + vertical_lift * (u_tip / self.U_ref)

        v_nd = vertical_lift * v_hat \
             + vertical_lift * (v_tip / self.U_ref)

        return self.U_ref * u_nd, self.U_ref * v_nd

    def thermoelastic_strain_stress(self, x, y, T_delta_nd, v_delta_nd, T_precomputed=None):
        """Return displacement, temperature, strains and stresses of the graded material."""
        u, v = self.net_uv(x, y, v_delta_nd)
        T = self.net_thermal(x, y, T_delta_nd) if T_precomputed is None else T_precomputed

        ux = self._safe_grad(u, x)
        uy = self._safe_grad(u, y)
        vx = self._safe_grad(v, x)
        vy = self._safe_grad(v, y)

        E, nu, lam, mu, k, rho, c, alpha = self.material_fields(x)
        eps_th = self.thermal_strain_coefficient(alpha=alpha, nu=nu) * (T - self.T_ref)

        exx_e = ux - eps_th
        eyy_e = vy - eps_th
        gamma_xy = uy + vx
        tr_e = exx_e + eyy_e

        sx = lam * tr_e + 2.0 * mu * exx_e
        sy = lam * tr_e + 2.0 * mu * eyy_e
        sxy = mu * gamma_xy

        return {
            "u": u,
            "v": v,
            "T": T,
            "ux": ux,
            "uy": uy,
            "vx": vx,
            "vy": vy,
            "exx_e": exx_e,
            "eyy_e": eyy_e,
            "gamma_xy": gamma_xy,
            "sx": sx,
            "sy": sy,
            "sxy": sxy,
            "E": E,
            "nu": nu,
            "lam": lam,
            "mu": mu,
            "k": k,
            "rho": rho,
            "c": c,
            "alpha": alpha,
        }

    def elastic_energy_density(self, x, y, T_delta_nd, v_delta_nd, T_precomputed=None):
        """Strain energy density with the local graded material."""
        f = self.thermoelastic_strain_stress(x, y, T_delta_nd, v_delta_nd, T_precomputed=T_precomputed)
        tr_e = f["exx_e"] + f["eyy_e"]
        return 0.5 * f["lam"] * tr_e ** 2 + f["mu"] * (
            f["exx_e"] ** 2 + f["eyy_e"] ** 2 + 0.5 * f["gamma_xy"] ** 2
        )

    def thermal_energy_density(self, x, y, T_delta_nd, Temp_n=None):
        """Transient heat-conduction energy density with the local graded k, rho and c.

        Overrides the method of ThermoSolver2D that train_thermal calls, so the
        thermal solve uses the graded properties.  With Temp_n, the temperature
        of the previous load step, the backward-Euler term
        0.5 * rho * c / dt * (T - Temp_n)^2 is added.
        """
        T = self.net_thermal(x, y, T_delta_nd)
        Tx = self._safe_grad(T, x)
        Ty = self._safe_grad(T, y)
        _, _, _, _, k, rho, c, _ = self.material_fields(x)
        energy = 0.5 * k * (Tx ** 2 + Ty ** 2)
        if Temp_n is not None:
            if not torch.is_tensor(Temp_n):
                Temp_n = torch.tensor(Temp_n, dtype=torch.float32, device=self.device)
            energy = energy + 0.5 * rho * c / self.dt * (T - Temp_n) ** 2
        return energy

    def heat_energy_density(self, x, y, T_delta_nd, Temp_n=None):
        """Alias of thermal_energy_density."""
        return self.thermal_energy_density(x, y, T_delta_nd, Temp_n)

    def compute_elastic_residual(self, x, y, T_delta_nd, v_delta_nd):
        """Strong-form equilibrium residual of the graded thermoelastic stress."""
        f = self.thermoelastic_strain_stress(x, y, T_delta_nd, v_delta_nd)
        sx_x = self._safe_grad(f["sx"], x)
        sxy_y = self._safe_grad(f["sxy"], y)
        sxy_x = self._safe_grad(f["sxy"], x)
        sy_y = self._safe_grad(f["sy"], y)
        rx = sx_x + sxy_y
        ry = sxy_x + sy_y
        return torch.sqrt(rx ** 2 + ry ** 2 + 1.0e-30)

    def predict_fields(self, X, T_delta_nd, v_delta_nd):
        """Predict fields and stresses with the local graded material."""
        X = np.asarray(X, dtype=np.float32)
        x = torch.tensor(X[:, 0:1], dtype=torch.float32, device=self.device, requires_grad=True)
        y = torch.tensor(X[:, 1:2], dtype=torch.float32, device=self.device, requires_grad=True)
        T_delta = torch.tensor([[T_delta_nd]], dtype=torch.float32, device=self.device)
        u_delta = torch.tensor([[v_delta_nd]], dtype=torch.float32, device=self.device)

        f = self.thermoelastic_strain_stress(x, y, T_delta, u_delta)
        sx = f["sx"]
        sy = f["sy"]
        sxy = f["sxy"]
        von_mises = torch.sqrt(sx ** 2 - sx * sy + sy ** 2 + 3.0 * sxy ** 2 + 1.0e-30)

        xy = self.xy_tensor(x, y)
        try:
            gamma = self.model_uv.extendAxis.getGamma(xy)
            if gamma.dim() == 1:
                gamma = gamma.unsqueeze(-1)
            gamma = gamma[:, 0:1]
        except Exception:
            gamma = torch.zeros_like(x)

        out = {
            "T": f["T"],
            "u": f["u"],
            "v": f["v"],
            "sx": sx,
            "sy": sy,
            "sxy": sxy,
            "von_mises": von_mises,
            "gamma": gamma,
            "E": f["E"],
        }
        return {key: val.detach().cpu().numpy() for key, val in out.items()}

    def reaction_force(self, X_f: np.ndarray, T_delta_nd: float, v_delta_nd: float) -> float:
        """Horizontal reaction dPi/d(delta_u) for the imposed top-edge shear displacement."""
        x = self.variable(X_f[:, 0:1], True)
        y = self.variable(X_f[:, 1:2], True)
        wt = self.variable(X_f[:, 2:3], False)
        T_delta = torch.tensor([[T_delta_nd]], dtype=torch.float32, device=self.device)
        u_delta = torch.tensor([[v_delta_nd]], dtype=torch.float32, device=self.device, requires_grad=True)
        energy_e = self.elastic_energy_density(x, y, T_delta, u_delta)
        Pi = torch.sum(energy_e * wt)
        dPi_dudelta_nd = torch.autograd.grad(Pi, u_delta, create_graph=False, retain_graph=False)[0]
        return float(dPi_dudelta_nd.detach().cpu().item() / self.U_ref)


# -----------------------------------------------------------------------------
# Build model and integration mesh
# -----------------------------------------------------------------------------
crack_embedding = make_crack_embedding(crack_points)
crack_geometry = CrackGeometry.from_embedding(crack_embedding, crack_points)

# Mechanical network: NN_uv(x, y, Gamma) -> [u_hat, v_hat].
# The asymptotic enrichment at the tip (Williams terms) is added in net_uv().
uv_core = AxisScaleNet(
    MLP([3, 50, 50, 50, 2], activation=nn.Tanh),
    lb=model_params["lb"],
    ub=model_params["ub"],
)
model_uv = extendAxisNet(uv_core, crack_embedding)

# Thermal network: NN_T(x, y, Gamma) -> T_hat.
thermal_core = ThermalNet([3, 50, 50, 50, 1], lb=model_params["lb"], ub=model_params["ub"], omega_0=30.0)
model_T = extendAxisNet(thermal_core, crack_embedding)

model = ShearTopEdgeThermoSolver2D(model_params, model_uv=model_uv, model_T=model_T, crack_geometry=crack_geometry)

# K extraction shared by the examples: interaction integral with guards, COD
# fallback, cross-check and per-step validation plots (see tmdem/sif_estimation.py).
sif = SIFEstimator(
    model,
    tip_boundary_distance=make_rect_boundary_distance(domainEndsX, domainEndsY),
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
    j_label="J_mech",
)


vertex = get_uniform_mesh(nx_elem, ny_elem, domainEndsX, domainEndsY)
parent_indices = np.full(len(vertex), -1, dtype=np.int64)

# Thermal integration batch.  This is always MC and may be resampled during
# thermal training.  Temp_n_f is carried only on this thermal batch.
x_np, y_np, wt_np, elem_np = model.generate_integration_points(
    vertex,
    thermal_num_integration_points,
    method=thermal_integration_method,
)
X_f_T = np.column_stack([x_np, y_np, wt_np, elem_np])
Temp_n_f = model.T_ref * np.ones((X_f_T.shape[0], 1), dtype=np.float32)

# Elastic integration batch.  For MC it starts as the current thermal batch; for
# Gaussian it is generated independently.
X_f_U = make_elastic_integration_points(model, vertex, X_f_T)


# Derived nondimensional increments and storage arrays.
deltaU = deltaU_phys / model.U_ref


def thermal_ramp_nd(step):
    """Nondimensional temperature increment at a given load step.

    Written against the applied displacement rather than the step index, so
    refining the step size leaves the physical schedule untouched.
    """
    u = step * deltaU_phys
    return (T_total_K / model.DeltaT_ref) * min(u / max(T_ramp_disp, 1.0e-30), 1.0)
fdGraph = np.full((numSteps, 2), np.nan, dtype=np.float64)

# -----------------------------------------------------------------------------
# Optional reference force-displacement curve (two columns: displacement, force)
# -----------------------------------------------------------------------------
fem_curves = {}
for candidate in [
    os.path.join(os.path.dirname(__file__), "FEM.txt"),
    os.path.join(os.path.dirname(__file__), "fem_FD.txt"),
    os.path.join(results_dir, "FEM.txt"),
    "FEM.txt",
]:
    try:
        disp, force = load_curve_txt(candidate)
        fem_curves["FEM"] = (disp, force)
        print(f"Loaded FEM reference curve: {candidate}")
        break
    except Exception:
        pass
if not fem_curves:
    print("[WARN] FEM compare disabled: no FEM.txt/fem_FD.txt found.")


# -----------------------------------------------------------------------------
# Resume from previous run
# -----------------------------------------------------------------------------
start_step = 0
if resume_from_checkpoint:
    if resume_checkpoint_path == "auto":
        _, checkpoint_path = find_latest_checkpoint(results_dir)
    else:
        checkpoint_path = resume_checkpoint_path

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        start_step, fdGraph, Temp_n_f, X_f_T, vertex, parent_indices, ckpt_crack_points = restore_from_checkpoint(
            model, checkpoint_path, fdGraph, device, strict=strict_resume
        )
        X_f_U = make_elastic_integration_points(model, vertex, X_f_T)
        if ckpt_crack_points is not None:
            crack_points = ckpt_crack_points
            crack_points, crack_embedding = set_crack_state(model, crack_points, make_crack_embedding)
        if start_step >= numSteps:
            print("[resume] All requested steps are already completed.")
    else:
        print("[resume] No checkpoint found. Starting from step 0.")

# Plot the initial mesh only for a fresh run.
if start_step == 0:
    plot_mesh(vertex, filename="Initial_Mesh", save_dir=results_dir, show=True)
else:
    print("[resume] Skip initial mesh plot.")

# -----------------------------------------------------------------------------
# Main simulation loop
# -----------------------------------------------------------------------------
for step in range(start_step, numSteps):
    step_dir = os.path.join(results_dir, f"step_{step}")
    os.makedirs(step_dir, exist_ok=True)

    # v_delta_nd is the name of the load parameter in the tmdem.solver
    # interfaces.  In this shear test it is the nondimensional horizontal
    # displacement of the top edge.
    v_delta_nd = deltaU * step
    T_delta_nd = thermal_ramp_nd(step)
    T_delta_nd_prev = thermal_ramp_nd(step - 1) if step >= 1 else 0.0

    u_delta_phys = v_delta_nd * model.U_ref
    T_delta_phys = T_delta_nd * model.DeltaT_ref

    print(f"\n=== Simulation Step {step} ===")
    print(f"Applied top-edge horizontal displacement: {u_delta_phys:.6e} mm | Applied top temperature increment: {T_delta_phys:.2f} K")
    print(
        f"Number of elements: {len(vertex)} | "
        f"Thermal integration points: {X_f_T.shape[0]} | "
        f"Elastic integration points: {X_f_U.shape[0]} ({elastic_integration_method})"
    )

    if step >= 1:
        model.snapshot_thermal_net()

    if step == 0:
        n_adam_T = thermal_num_Adam_step0
        n_lbfgs_T = thermal_num_LBFGS_step0
        n_adam_U = elastic_num_Adam_step0
        n_lbfgs_U = elastic_num_LBFGS_step0
        num_refine_cycles = num_refine_cycles_step0
    elif step == 1:
        n_adam_T = thermal_num_Adam_step1
        n_lbfgs_T = thermal_num_LBFGS_step1
        n_adam_U = elastic_num_Adam_step1
        n_lbfgs_U = elastic_num_LBFGS_step1
        num_refine_cycles = num_refine_cycles_following
    else:
        n_adam_T = thermal_num_Adam
        n_lbfgs_T = thermal_num_LBFGS
        n_adam_U = elastic_num_Adam
        n_lbfgs_U = elastic_num_LBFGS
        num_refine_cycles = num_refine_cycles_following

    if skip_thermal_after_step0 and step > 0:
        # Isothermal case: the temperature stays at T_0 everywhere, so after
        # the step-0 solve there is nothing left for the thermal network to
        # learn.
        n_adam_T = 0
        n_lbfgs_T = 0

    # ---- Thermal solve: Monte Carlo integration with resampling ----
    print("Initial thermal training")
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
    print(f"Thermal training time: {time.time() - t0:.4f} sec")
    log_resampling_change("thermal", X_f_T_before, X_f_T)

    # ---- Elastic solve with current thermal field ----
    # For MC elastic integration, start from the latest thermal MC batch.  For
    # Gaussian elastic integration, use an independent deterministic Gauss batch.
    X_f_U = make_elastic_integration_points(model, vertex, X_f_T)
    elastic_vertex, elastic_npts, elastic_resample_every = elastic_resampling_args()

    print("Initial elastic training")
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
    print(f"Elastic training time: {time.time() - t0:.4f} sec")
    log_resampling_change("elastic", X_f_U_before, X_f_U)

    # ---- Optional adaptive refinement cycle ----
    if mesh_policy.lower() == "adaptive" and enable_refine_elastic and num_refine_cycles > 0:
        for cycle in range(num_refine_cycles):
            print(f"Refinement cycle {cycle + 1}/{num_refine_cycles}")
            x_res = torch.tensor(X_f_U[:, 0:1], dtype=torch.float32, device=model.device, requires_grad=True)
            y_res = torch.tensor(X_f_U[:, 1:2], dtype=torch.float32, device=model.device, requires_grad=True)
            T_d = torch.tensor([[T_delta_nd]], dtype=torch.float32, device=model.device)
            v_d = torch.tensor([[v_delta_nd]], dtype=torch.float32, device=model.device)
            residual = model.compute_elastic_residual(x_res, y_res, T_d, v_d).detach().cpu().numpy()
            fields_tmp = model.predict_fields(X_f_U[:, :2], T_delta_nd, v_delta_nd)
            refined_elements = mark_elements_for_refinement(
                X_f_U, X_f_U[:, 3], vertex, residual, fields_tmp.get("gamma"),
                error_percentile=error_percentile,
                max_level=max_refinement_level,
            )
            if not refined_elements:
                break
            vertex, parent_indices = update_adaptive_mesh(nx_elem, ny_elem, domainEndsX, domainEndsY, refined_elements)
            x_np, y_np, wt_np, elem_np = model.generate_integration_points(
                vertex,
                thermal_num_integration_points,
                method="mc",
            )
            X_f_T = np.column_stack([x_np, y_np, wt_np, elem_np])
            Temp_n_f = model.eval_prev_thermal(X_f_T[:, :2], T_delta_nd_prev)
            X_f_U = make_elastic_integration_points(model, vertex, X_f_T)
            plot_mesh(vertex, filename=f"Refined_Mesh_Step_{step}_Cycle_{cycle + 1}", save_dir=step_dir, show=True)
            X_f_U = model.train_elastic(
                X_f_U, T_delta_nd, v_delta_nd,
                n_adam_U, lr_Adam, n_lbfgs_U, lr_lbfgs, num_log,
                vertex=None, num_integration_points=elastic_num_integration_points, resample_every_iters=0,
            )

    # ---- Crack propagation at the current load step ----
    prop_iter = 0
    while enable_crack_propagation and step > 0 and prop_iter < max_propagations_per_step:
        if crack_reached_boundary_rect(crack_points, domainEndsX, domainEndsY, zero_force_boundaries, model=model, tol=ligament_margin):
            print("[propagation] Crack already reaches a stopping boundary. Skip SIF and further propagation.")
            break

        K1, K2, J, beta_tip = sif.estimate(crack_points, T_delta_nd, v_delta_nd)
        crack_tip_xy, T_tip_prop, Gc_tip_prop = get_tip_fracture_properties(model, crack_points, T_delta_nd)
        G_drive, theta_drive = driving_force(K1, K2, crack_tip_xy)
        ratio = G_drive / max(Gc_tip_prop, 1.0e-30)
        if not np.isfinite(K1) or not np.isfinite(K2) or not np.isfinite(G_drive) or not np.isfinite(Gc_tip_prop) or not np.isfinite(ratio):
            print(
                "[propagation] Non-finite SIF/G/Gc detected. "
                "Stop propagation at this step to avoid corrupting the networks."
            )
            break
        print(
            f"[prop-check step {step} iter {prop_iter}] "
            f"K1={K1:.4e}, K2={K2:.4e}, J_straight={J:.4e}, "
            f"G_eff={G_drive:.4e} (theta={np.rad2deg(theta_drive):+.2f} deg), "
            f"Gc(Ttip)={Gc_tip_prop:.4e}, G_eff/Gc={ratio:.3f}, "
            f"crack_len={total_crack_length(crack_points, model=model):.4f} mm"
        )

        append_propagation_history([
            step, prop_iter, f"{u_delta_phys:.12e}", f"{T_delta_phys:.12e}",
            f"{K1:.12e}", f"{K2:.12e}", f"{G_drive:.12e}",
            f"{T_tip_prop:.12e}", f"{Gc_tip_prop:.12e}", f"{ratio:.12e}",
            f"{beta_tip:.12e}", 0.0,
            f"{crack_tip_xy[0]:.12e}", f"{crack_tip_xy[1]:.12e}", 0,
        ])

        # The estimate is logged above before this check, so rejected
        # extractions also appear in the history.
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

        new_tip, beta_new, theta_kink = propose_crack_extension_rect(
            model, crack_points, K1, K2, domainEndsX, domainEndsY,
            crack_increment=crack_increment,
            propagation_mode=propagation_mode,
            max_kink_angle_deg=max_kink_angle_deg,
            k_ratio_deadband=k_ratio_deadband,
            kink_angle_relaxation=kink_angle_relaxation,
        )
        if new_tip is None:
            print("[propagation] Crack tip reached the boundary or proposed extension is too small. Stop propagation.")
            break

        crack_points.append(new_tip)
        crack_points, crack_embedding = set_crack_state(model, crack_points, make_crack_embedding)
        prop_iter += 1
        print(
            f"[propagation] Extended crack to {new_tip}, "
            f"theta={np.rad2deg(theta_kink):.2f} deg, total points={len(crack_points)}"
        )
        save_crack_path(crack_points, step_dir, step, domainEndsX, domainEndsY, crack_geometry=model.crack_geometry, show=True)
        append_propagation_history([
            step, prop_iter, f"{u_delta_phys:.12e}", f"{T_delta_phys:.12e}",
            f"{K1:.12e}", f"{K2:.12e}", f"{G_drive:.12e}",
            f"{T_tip_prop:.12e}", f"{Gc_tip_prop:.12e}", f"{ratio:.12e}",
            f"{beta_new:.12e}", f"{theta_kink:.12e}",
            f"{new_tip[0]:.12e}", f"{new_tip[1]:.12e}", 1,
        ])

        if crack_reached_boundary_rect(crack_points, domainEndsX, domainEndsY, zero_force_boundaries, model=model, tol=ligament_margin):
            print(
                "[propagation] Crack reached a stopping boundary. "
                "Skip post-extension re-training and keep force = 0 from this point."
            )
            break

        # Re-train both fields on the same load step after changing the crack.
        # The previous-step temperature Temp_n_f is kept as the transient state.
        n_adam_T_prop = 0 if skip_thermal_after_step0 else prop_thermal_num_Adam
        n_lbfgs_T_prop = 0 if skip_thermal_after_step0 else prop_thermal_num_LBFGS
        print("[propagation] Re-training thermal field after crack extension")
        X_f_T_before = X_f_T.copy()
        X_f_T, Temp_n_f = model.train_thermal(
            X_f=X_f_T,
            T_delta_nd=T_delta_nd,
            Temp_n_f=Temp_n_f,
            nIter_Adam=n_adam_T_prop,
            lr_Adam=lr_Adam,
            nIter_LBFGS=n_lbfgs_T_prop,
            lr_LBFGS=lr_lbfgs,
            nIter_log=num_log,
            step=step,
            vertex=vertex,
            num_integration_points=thermal_num_integration_points,
            resample_every_iters=(resample_frequency_epochs_thermal if enable_resample_thermal else 0),
            T_delta_nd_prev=T_delta_nd_prev,
        )
        log_resampling_change("thermal-propagation", X_f_T_before, X_f_T)

        print("[propagation] Re-training elastic field after crack extension")
        X_f_U = make_elastic_integration_points(model, vertex, X_f_T)
        elastic_vertex, elastic_npts, elastic_resample_every = elastic_resampling_args()
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

    if enable_crack_propagation and prop_iter >= max_propagations_per_step:
        print(f"[propagation] Reached max_propagations_per_step={max_propagations_per_step} at step {step}.")

    # ---- Final predictions and saving ----
    # Keep the thermal state on the Monte Carlo integration points, because the
    # next transient heat step needs T_n at the current (possibly resampled)
    # integration points.
    T_pred = model.predict_thermal(X_f_T[:, :2], T_delta_nd)
    Temp_n_f = T_pred.reshape(-1, 1)
    fields = model.predict_fields(X_f_U[:, :2], T_delta_nd, v_delta_nd)

    # Use a deterministic post-processing grid for plots.  The fields are
    # discontinuous across the crack, so interpolating them from the Monte
    # Carlo points with griddata would create artificial mosaic patterns near
    # the crack.
    X_plot = make_prediction_grid(domainEndsX, domainEndsY, n=501)
    fields_plot = model.predict_fields(X_plot, T_delta_nd, v_delta_nd)
    fields_plot_display = dict(fields_plot)
    # A crack mask for the plotted discontinuous fields, used in
    # post-processing only.  It removes the jagged edge produced where the
    # sharp jump cuts across rectangular plotting pixels, most visibly along
    # kinked cracks.  The width falls back to 0.01 if the model has no
    # crack_band_width.
    crack_plot_width = getattr(model, "crack_band_width", 0.01)
    display_crack_width = max(0.35 * crack_plot_width, 2.5e-3)
    display_tip_radius = max(0.75 * crack_plot_width, 5.0e-3)
    for key in ["T", "u", "v", "sx", "sy", "sxy", "von_mises"]:
        fields_plot_display[key] = mask_crack_band_for_plot(
            X_plot, fields_plot[key], model.crack_geometry,
            width=display_crack_width,
            tip_radius=display_tip_radius,
        )

    np.savez(
        os.path.join(step_dir, f"fields_step_{step}.npz"),
        X_f=X_f_U,      # alias of X_f_U, the elastic integration batch
        X_f_T=X_f_T,
        X_f_U=X_f_U,
        X_plot=X_plot,
        T=fields["T"],
        u=fields["u"],
        v=fields["v"],
        sx=fields["sx"],
        sy=fields["sy"],
        sxy=fields["sxy"],
        von_mises=fields["von_mises"],
        gamma=fields["gamma"],
        T_smooth_contrib_K=fields.get("T_smooth_contrib_K"),
        T_jump_contrib_K=fields.get("T_jump_contrib_K"),
        T_plot=fields_plot["T"],
        u_plot=fields_plot["u"],
        v_plot=fields_plot["v"],
        sx_plot=fields_plot["sx"],
        sy_plot=fields_plot["sy"],
        sxy_plot=fields_plot["sxy"],
        von_mises_plot=fields_plot["von_mises"],
        gamma_plot=fields_plot["gamma"],
        T_smooth_contrib_K_plot=fields_plot.get("T_smooth_contrib_K"),
        T_jump_contrib_K_plot=fields_plot.get("T_jump_contrib_K"),
        crack_points=np.asarray(crack_points, dtype=np.float64),
        crack_path=np.asarray(model.crack_geometry.curve_points, dtype=np.float64),
    )

    # Crack-tip quantities at the end of the load step, after any propagation.
    crack_tip_xy = np.asarray(get_active_crack_geometry(model, crack_points).tip, dtype=np.float64)
    if crack_reached_boundary_rect(crack_points, domainEndsX, domainEndsY, zero_force_boundaries, model=model, tol=ligament_margin):
        # Once the specimen has separated, the crack-tip quantities of linear
        # elastic fracture mechanics are not defined, so K and G are set to
        # zero and only the tip temperature and Gc are recorded.
        T_tip = sample_crack_tip_temperature(model, crack_tip_xy, T_delta_nd, radius=2.0e-3, n_ring=24)
        if not np.isfinite(T_tip):
            # Fallback: the linear profile T_ref + Delta T * y of the boundary conditions.
            T_tip = float(model.T_ref + model.DeltaT_ref * T_delta_nd * np.clip(crack_tip_xy[1], 0.0, 1.0))
        Gc_tip = float(temperature_dependent_Gc(T_tip, crack_tip_xy[0])) if use_temperature_dependent_Gc else float(spatial_Gc_at_x(crack_tip_xy[0]))
        K1_tip, K2_tip, J_tip, beta_tip = 0.0, 0.0, 0.0, last_tangent_normal(crack_points, model=model)[2]
        G_tip, theta_tip = 0.0, 0.0
    else:
        crack_tip_xy, T_tip, Gc_tip = get_tip_fracture_properties(model, crack_points, T_delta_nd)
        K1_tip, K2_tip, J_tip, beta_tip = sif.estimate(crack_points, T_delta_nd, v_delta_nd)
        G_tip, theta_tip = driving_force(K1_tip, K2_tip, crack_tip_xy)
        sif.save_validation(step_dir, step, crack_points, T_delta_nd, v_delta_nd)
    np.savez(
        os.path.join(step_dir, f"fracture_tip_step_{step}.npz"),
        crack_tip=crack_tip_xy,
        T_tip=T_tip,
        Gc_tip=Gc_tip,
        Gc0=Gc0,
        J_tip=J_tip,
        G_tip=G_tip,
        theta_tip=theta_tip,
        K1_tip=K1_tip,
        K2_tip=K2_tip,
        J_over_Gc=G_tip / max(Gc_tip, 1.0e-30),
        J_straight_over_Gc=J_tip / max(Gc_tip, 1.0e-30),
        crack_points=np.asarray(crack_points, dtype=np.float64),
        use_temperature_dependent_Gc=use_temperature_dependent_Gc,
    )
    save_crack_path(crack_points, step_dir, step, domainEndsX, domainEndsY, crack_geometry=model.crack_geometry, show=True)

    if crack_reached_boundary_rect(crack_points, domainEndsX, domainEndsY, zero_force_boundaries, model=model, tol=ligament_margin):
        # After separation the remaining energy derivative is numerical, so the
        # force is set to zero.
        force = 0.0
    else:
        force = model.reaction_force(X_f_U, T_delta_nd, v_delta_nd)

    # One row per load step: everything needed to replot the SIF, energy
    # release rate and force-displacement curves against the applied
    # displacement without loading checkpoints.
    tip_history_file = os.path.join(results_dir, "fracture_tip_history.csv")
    if step == 0 or not os.path.exists(tip_history_file):
        with open(tip_history_file, "w", encoding="utf-8") as f:
            # J_tip and J_over_Gc hold the energy release rate of the kink
            # G(theta_c) and its ratio to Gc; J_straight is the straight-ahead
            # value (K1^2 + K2^2)/E'; R_factor is the resistance curve factor
            # (1 when use_r_curve is False).
            f.write("step,u_delta_mm,T_delta_K,T_tip_K,Gc_tip,J_tip,J_over_Gc,K1_tip,K2_tip,"
                    "crack_length,tip_x,tip_y,force_N,R_factor,J_straight,theta_deg\n")
    with open(tip_history_file, "a", encoding="utf-8") as f:
        f.write(
            f"{step},{u_delta_phys:.12e},{T_delta_phys:.12e},"
            f"{T_tip:.12e},{Gc_tip:.12e},{G_tip:.12e},{G_tip / max(Gc_tip, 1.0e-30):.12e},"
            f"{K1_tip:.12e},{K2_tip:.12e},{total_crack_length(crack_points, model=model):.12e},"
            f"{crack_tip_xy[0]:.12e},{crack_tip_xy[1]:.12e},{force:.12e},"
            f"{resistance_curve_factor(crack_points, model=model):.12e},"
            f"{J_tip:.12e},{np.rad2deg(theta_tip):.12e}\n"
        )

    # Save a restart checkpoint before plotting.  This prevents losing a whole
    # trained step if a post-processing/plotting error occurs.
    fdGraph[step, 0] = u_delta_phys
    fdGraph[step, 1] = force
    np.save(os.path.join(results_dir, "fdGraph.npy"), fdGraph)
    print(
        f"Displacement = {u_delta_phys:.6e} mm, Force = {force:.6f}, "
        f"T_tip = {T_tip:.2f} K, Gc_tip = {Gc_tip:.6e}"
    )

    mesh_meta = {
        "nx_elem": nx_elem,
        "ny_elem": ny_elem,
        "Gc0": Gc0,
        "use_temperature_dependent_Gc": use_temperature_dependent_Gc,
        "use_r_curve": use_r_curve,
        "r_curve_eta": r_curve_eta,
        "r_curve_onset": r_curve_onset,
        "r_curve_length": r_curve_length,
        "r_curve_factor": resistance_curve_factor(crack_points, model=model),
        "Gc_tip": Gc_tip,
        "T_tip": T_tip,
        "J_tip": J_tip,
        "G_tip": G_tip,
        "theta_tip_deg": float(np.rad2deg(theta_tip)),
        "K1_tip": K1_tip,
        "K2_tip": K2_tip,
        "K_I_trainable": model.trainable_sif_values()[0],
        "K_II_trainable": model.trainable_sif_values()[1],
        "use_thermal_jump": getattr(model, "use_thermal_jump", False),
        "use_explicit_mechanical_jump": getattr(model, "use_explicit_mechanical_jump", False),
        "J_over_Gc": G_tip / max(Gc_tip, 1.0e-30),
        "chi_c": min_G_over_Gc_to_propagate,
        "crack_points": np.asarray(crack_points, dtype=np.float64).tolist(),
        "crack_path": np.asarray(model.crack_geometry.curve_points, dtype=np.float64).tolist(),
        "crack_length": total_crack_length(crack_points, model=model),
        "thermal_num_integration_points": thermal_num_integration_points,
        "elastic_integration_method": elastic_integration_method,
        "elastic_num_integration_points": elastic_num_integration_points,
        "elastic_gauss_order": elastic_gauss_order,
    }
    model.save_model(step, fdGraph, Temp_n_f, X_f_T, vertex, parent_indices, step_dir,
                     mesh_policy=mesh_policy, mesh_meta=mesh_meta)

    # Training loss histories of the thermal and the elastic solve.
    plt.figure(figsize=(10, 6))
    plt.semilogy(model.thermal_adam_loss_history, label="Adam")
    if model.thermal_lbfgs_loss_history:
        offset = len(model.thermal_adam_loss_history)
        plt.semilogy(range(offset, offset + len(model.thermal_lbfgs_loss_history)), model.thermal_lbfgs_loss_history, label="LBFGS")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig(os.path.join(step_dir, f"thermal_convergence_{step}.png"), dpi=300, bbox_inches="tight")
    plt.show(); plt.close()

    plt.figure(figsize=(10, 6))
    plt.semilogy(model.elastic_adam_loss_history, label="Adam")
    if model.elastic_lbfgs_loss_history:
        offset = len(model.elastic_adam_loss_history)
        plt.semilogy(range(offset, offset + len(model.elastic_lbfgs_loss_history)), model.elastic_lbfgs_loss_history, label="LBFGS")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig(os.path.join(step_dir, f"elastic_convergence_{step}.png"), dpi=300, bbox_inches="tight")
    plt.show(); plt.close()

    plot_field(X_plot, fields_plot_display["T"], domainEndsX, domainEndsY,
               title=f"Temperature - Step {step}", label="Temperature (K)",
               filename=f"temperature_{step}", save_dir=step_dir, cmap="coolwarm", show=True)
    plot_field(X_plot, fields_plot["gamma"], domainEndsX, domainEndsY,
               title=f"Crack embedding function, step {step}", label="Gamma", filename=f"crack_gamma_{step}",
               save_dir=step_dir, cmap="magma", vmin=-1.0, vmax=1.0, show=True)
    plot_field(X_plot, fields_plot_display["u"], domainEndsX, domainEndsY,
               title=f"x-Displacement - Step {step}", label="u (mm)",
               filename=f"u_displacement_{step}", save_dir=step_dir, cmap="viridis", show=True)
    plot_field(X_plot, fields_plot_display["v"], domainEndsX, domainEndsY,
               title=f"y-Displacement - Step {step}", label="v (mm)",
               filename=f"v_displacement_{step}", save_dir=step_dir, cmap="viridis", show=True)
    plot_field(X_plot, fields_plot_display["von_mises"], domainEndsX, domainEndsY,
               title=f"Von Mises Stress - Step {step}", label="Von Mises (MPa)",
               filename=f"von_mises_{step}", save_dir=step_dir, cmap="viridis", show=True)

    plot_force_disp(fdGraph[:step + 1], filename=f"FD_step_{step}", outdir=step_dir, show=False)
    plot_force_disp(fdGraph[:step + 1], filename="FD", outdir=results_dir, show=True)
    if fem_curves:
        plot_force_disp_compare(fdGraph[:step + 1], fem_curves, os.path.join(step_dir, "FD_compare.png"), show=False)
        plot_force_disp_compare(fdGraph[:step + 1], fem_curves, os.path.join(results_dir, "FD_compare.png"), show=True)

print(f"Total simulation time: {time.time() - start_time:.2f} seconds")
