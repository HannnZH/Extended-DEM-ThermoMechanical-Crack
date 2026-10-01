#!/usr/bin/env python
"""
Single-edge notched tension test under thermal load.

The thermoelastic single-edge notched benchmark of Tangella et al. (2022). The
crack is a sharp crack carried by the crack embedding; it grows along the
maximum hoop stress direction when the energy release rate of the kink
G(theta_c) reaches the temperature-dependent fracture energy Gc(T_tip) at the
tip.

Units are mm, N, s and K, and energies and forces are per 1 mm of thickness.

Geometry
    Unit square [0, 1] x [0, 1] with a horizontal edge crack from (0, 0.5) to
    (0.5, 0.5).

Material
    Homogeneous, plane strain: E = 340 GPa, nu = 0.22, alpha = 8.0e-6 1/K,
    k = 300 W/(m K), rho = 2450 kg/m^3, c = 0.775 J/(kg K), reference
    temperature T_ref = 300 K.  The fracture energy follows the law of Ruan
    et al. (2023),
        Gc(T) = Gc0 [1 - b1 (T - T_ref)/T_max + b2 ((T - T_ref)/T_max)^2],
    with Gc0 = 42.47 J/m^2, b1 = 1.80, b2 = 1.10 and T_max = 1000 K,
    evaluated at the tip temperature.

Boundary conditions
    Thermal: transient conduction starting from T_ref everywhere.  The bottom
    edge y = 0 is held at T_ref and the top edge y = 1 at T_ref + DeltaT; the
    vertical edges and the crack faces are adiabatic.
    Mechanical: the bottom edge is fixed, u = v = 0; the top edge has u = 0
    and v = delta_v; the vertical edges and the crack faces are traction free.

Loading
    106 load steps, step 0 unloaded.  delta_v grows by 1e-5 mm per load step.
    DeltaT grows by dT per load step over the first 20 steps and is held from
    step 20 on.  The three thermal cases of the paper are dT = +2.5, -2.5 and
    0 K per load step, that is DeltaT = +50, -50 and 0 K.

Solution
    Each load step is a staggered solve, thermal then mechanical.  The crack
    may then grow by crack increments of 0.025 mm, up to 35 within the load
    step, as long as the extraction is admissible and
    G(theta_c) / Gc(T_tip) >= 1; both fields are retrained after every crack
    increment.  The SIFs come from the interaction integral, with the
    crack-opening displacement (COD) as fallback.  The reaction force is the
    derivative of the potential energy with respect to delta_v.  Once the tip
    reaches the right edge the specimen is separated, growth stops and the
    force is zero.

Command line
    --dT      temperature change per load step in K (default 2.5)
    --seed    seed of the network initialization (default 2025)
    --resume  continue from the latest checkpoint in the output folder
    --out     output folder (default runs/sen_tension_dT_<dT>)

Output, in results/ inside the output folder
    crack_propagation_history.csv   one row per extraction and one per crack
                                    increment
    fracture_tip_history.csv        one row per load step: tip temperature,
                                    Gc, J, K1, K2, crack length, tip position
                                    and force
    fdGraph.npy, FD.png             force-displacement curve
    FD_compare.png                  the same against a reference curve, when a
                                    file FEM.txt or fem_FD.txt (columns
                                    displacement and force) is found next to
                                    this script, or FEM.txt in results/ or in
                                    the output folder
    step_<n>/                       per load step: the fields at the
                                    integration points and on a 501 x 501
                                    plotting grid, the tip quantities, a
                                    checkpoint, the crack path, the radius
                                    sweep of the extraction, the loss
                                    histories and field plots

    python examples/sen_tension.py --dT 2.5     # heated to +50 K
    python examples/sen_tension.py --dT -2.5    # cooled to -50 K
    python examples/sen_tension.py --dT 0       # isothermal

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
    crack_reached_boundary_rect,
    effective_energy_release_rate,
    extraction_is_plausible,
    get_active_crack_geometry,
    last_tangent_normal,
    local_material_at_tip,
    make_rect_boundary_distance,
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
cli = argparse.ArgumentParser(description='Single-edge notched tension test under thermal load.')
cli.add_argument("--dT", type=float, default=2.5,
                 help="temperature change per load step in K, applied over the first 20 steps: "
                      "2.5 (heating to +50 K, default), -2.5 (cooling to -50 K) or 0")
cli.add_argument("--seed", type=int, default=2025,
                 help="seed of the network initialization (default 2025)")
cli.add_argument("--resume", action="store_true",
                 help="continue from the latest checkpoint in the output folder")
cli.add_argument("--out", default=None,
                 help="output folder (default runs/sen_tension_dT_<dT>)")
cli_args = cli.parse_args()
out_dir = cli_args.out or os.path.join("runs", f"sen_tension_dT_{cli_args.dT:g}")
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

# Resume from the latest completed checkpoint when enabled.
resume_from_checkpoint = cli_args.resume
resume_checkpoint_path = "auto"   # "auto" or an explicit checkpoint path
strict_resume = False             # passed as strict to load_state_dict; False tolerates missing or extra keys

# The propagation log is appended to, so a fresh run removes the log of an
# earlier run in the same folder.  The per-step history is truncated at
# step 0.
_prop_log_path = os.path.join(results_dir, "crack_propagation_history.csv")
if not resume_from_checkpoint and os.path.exists(_prop_log_path):
    os.remove(_prop_log_path)

start_time = time.time()

# -----------------------------------------------------------------------------
# Temperature-dependent fracture energy
# -----------------------------------------------------------------------------
# Gc(T) of Ruan et al. (2023), see temperature_dependent_Gc, evaluated at the
# tip temperature in the propagation criterion G(theta_c) / Gc(T_tip) >= chi_c.
# Gc0 is the fracture energy at T_ref in N/mm (42.47 J/m^2).
use_temperature_dependent_Gc = True

Gc0 = 0.04247
Gc_b1 = 1.80
Gc_b2 = 1.10
Gc_Tref = 300.0      # K
Gc_Tmax = 1000.0     # K, reference maximum temperature T_max of Gc(T)
Gc_min_factor = 0.03 # lower clamp on the factor of Gc0, keeps Gc(T) positive

# -----------------------------------------------------------------------------
# Material and model parameters
# -----------------------------------------------------------------------------
model_params = {
    # mm-N-s-K unit system.
    "E": 340.0e3,        # N/mm^2
    "nu": 0.22,
    "alpha": 8.0e-6,    # 1/K
    # Plane strain: the thermal strain in the reduced constitutive law carries
    # the effective coefficient (1 + nu) * alpha, as in the phase-field
    # reference (fem/pf_fem.py, alpha_mode ps3d).
    "plane_strain_thermal_alpha_correction": True,
    # 2-D energies and forces per unit thickness.
    "k": 300.0,         # W/(m K) = mJ/(s mm K)
    "rho": 2.45e-6,     # kg/mm^3
    "c": 775.0,         # mJ/(kg K)
    "T_ref": 300.0,     # K
    "dt": 1e-8,         # s, time step of the transient conduction
    "plane_type": "plane_strain",

    # Nondimensional load scales.
    "U_ref": 1.0e-4,        # mm
    "DeltaT_ref": 100.0,    # K

    # Unit square domain; the initial edge crack is crack_points below.
    "lb": np.array([0.0, 0.0]),
    "ub": np.array([1.0, 1.0]),

    # Crack embedding as network input: the networks receive [x, y, Gamma].
    # The explicit jump branches are off.
    "use_thermal_jump": False,
    "use_explicit_mechanical_jump": False,

    # Adiabatic crack faces, imposed as a soft penalty.
    "adiabatic_crack_weight": 1.0e-6,
    "adiabatic_crack_samples": 150,
    "crack_face_offset": 1.0e-4,

    # Smoothed conductivity band along the crack, off.
    "use_crack_conductivity_degradation": False,
    "crack_band_width": 0.005,
    "crack_min_conductivity_ratio": 1.0e-6,

    # Integration points densified near the tip (see
    # generate_integration_points): background elements within the radius get
    # factor times as many Monte Carlo samples, on the same fixed mesh and with
    # weights area/n.
    "tip_densify_radius": 0.08,
    "tip_densify_factor": 8,

    # Asymptotic enrichment of the displacement at the tip (Williams terms).
    "use_tip_enrichment": True,
    "tip_enrichment_q": 1.0,
    "tip_enrichment_decay_factor": 5.0,
    "tip_enrichment_u_scale": 1.0,
}

crack_points = [[0.0, 0.5], [0.5, 0.5]]
crack_geometry = None


# -----------------------------------------------------------------------------
# Mesh and integration settings
# -----------------------------------------------------------------------------
# A fixed mesh of 100 x 100 background elements with Monte Carlo integration.
mesh_policy = "fixed"  # "fixed" or "adaptive"
domainEndsX = np.array([0.0, 1.0])
domainEndsY = np.array([0.0, 1.0])

nx_elem = 100
ny_elem = 100

# Thermal integration: Monte Carlo points per element, redrawn during training.
thermal_num_integration_points = 9
thermal_integration_method = "mc"

# Elastic integration: Monte Carlo or Gauss points.
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
deltaT_phys = cli_args.dT         # K per load step, over the temperature ramp
deltaV_phys = 1e-5    # mm per load step


num_temp_steps = 21     # DeltaT = deltaT_phys * min(step, num_temp_steps - 1)
numSteps = 106           # includes step 0, where both loads are zero

# Training iterations: load step 0, load step 1 and every later load step.
thermal_num_Adam_step0 = 5000
thermal_num_LBFGS_step0 = 2500
thermal_num_Adam_step1 = 3000
thermal_num_LBFGS_step1 = 0
thermal_num_Adam = 3000
thermal_num_LBFGS = 0

elastic_num_Adam_step0 = 7000
elastic_num_LBFGS_step0 = 3000
elastic_num_Adam_step1 = 3000
elastic_num_LBFGS_step1 = 0
elastic_num_Adam = 3000
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
crack_increment = 0.025              # mm per crack increment
max_propagations_per_step = 35
# Threshold chi_c of the propagation criterion G(theta_c) / Gc(T_tip) >= chi_c,
# 1.0 in all examples.
min_G_over_Gc_to_propagate = 1.0
# Grow the crack only on an admissible extraction, one whose radius sweep
# passes.  The contour radius is a numerical choice, so a K that depends on it
# is not a material quantity.  Setting this False, or passing
# use_plateau_estimate=False to SIFEstimator, removes the check.
require_validated_extraction = True
max_kink_angle_deg = 60.0
# Force is zero after a stopping boundary is reached.
zero_force_boundaries = ("right",)

# Extraction length scales, written as multiples of the background element
# size h rather than of the crack increment.
background_cell_size = (domainEndsX[1] - domainEndsX[0]) / nx_elem   # h = 0.010 mm

# COD fallback sampling window.
sif_cod_num_points = 10
# Two-sided face-sampling offset: float32 noise << offset << sif_r_min, and
# crack_embedding_decay_alpha * offset << 1, so that the embedded jump is not
# attenuated.
sif_cod_offset = 5.0e-4
sif_r_min = 1.25 * background_cell_size    # 0.0125 mm
sif_r_max = 6.25 * background_cell_size    # 0.0625 mm
sif_limit_by_last_segment = True

# Interaction integral first, COD fallback second.
sif_estimator = "interaction_with_cod_fallback"  # "interaction_with_cod_fallback" or "cod"
# 2.25 h = 0.0225 mm, 0.9 of the crack increment.  SIFEstimator.contour_radius
# clips the radius at run time to 0.9 of the straight stretch that ends at
# the tip, which after a kink is one crack increment long.
sif_interaction_radius = 2.25 * background_cell_size   # 0.0225 mm
# Number of points of the interaction integral (see SIFEstimator).
sif_interaction_num_points = 2880

# Print COD fallback sampling range.
print_sif_range_diagnostics = True

# -----------------------------------------------------------------------------
# Crack embedding parameters
# -----------------------------------------------------------------------------
# The normal decay of the crack embedding is exp(-alpha * dist^p), so with
# p = 1 its e-folding length is 1/alpha.  It is set here as a length, five
# background elements.
crack_embedding_decay_length = 5.0 * background_cell_size    # 0.050 mm
crack_embedding_decay_alpha = 1.0 / crack_embedding_decay_length    # 20.0
crack_embedding_decay_p = 1.0
crack_embedding_endpoint_overlap = 1.25 * background_cell_size    # 0.0125 mm

# Retraining after each crack increment, within the same load step.
prop_thermal_num_Adam = 1000
prop_thermal_num_LBFGS = 0
# 3000, as in the shear example: the mechanical field re-converges after
# each crack increment before the next extraction.
prop_elastic_num_Adam = 3000
prop_elastic_num_LBFGS = 0

def temperature_dependent_Gc(T_abs):
    """Temperature-dependent fracture energy (Ruan et al. 2023).

    Gc(T) = Gc0 * [1 - b1 * (T - Tref) / Tmax
                     + b2 * ((T - Tref) / Tmax)^2]

    T_abs must be absolute temperature in K.  The bracketed factor is clamped
    from below at Gc_min_factor.
    """
    T_abs = np.asarray(T_abs, dtype=np.float64)
    theta = (T_abs - Gc_Tref) / Gc_Tmax
    factor = 1.0 - Gc_b1 * theta + Gc_b2 * theta**2
    factor = np.maximum(factor, Gc_min_factor)
    return Gc0 * factor


def driving_force(K1, K2, crack_tip_xy):
    """Energy release rate of the kink G(theta_c), and the kink angle theta_c.

    The same criterion as in the shear example.  E' is taken from the
    material at the tip, E' = E / (1 - nu^2) in plane strain.  This test is
    predominantly mode I, so theta_c stays close to zero and G(theta_c) close
    to the straight-ahead value (K1^2 + K2^2)/E'; with K2 = 0 the two are
    equal.  The kinked form is used so that any mode mixity that does appear
    is treated as in the shear example.
    """
    E_tip, nu_tip, _ = local_material_at_tip(model, float(crack_tip_xy[0]))
    E_eff = E_tip / (1.0 - nu_tip ** 2) if plane_strain_flag(model) else E_tip
    return effective_energy_release_rate(
        K1, K2, E_eff, max_kink_angle_deg=max_kink_angle_deg)


def get_tip_fracture_properties(model, crack_points, T_delta_nd):
    """Return the tip position, the tip temperature T_tip and Gc(T_tip)."""
    geom = get_active_crack_geometry(model, crack_points)
    crack_tip_xy = np.asarray(geom.tip, dtype=np.float64)
    # T_tip is the mean over a ring of 24 points around the tip, which is more
    # stable than the value at the tip itself.  The ring radius is kept much
    # smaller than the crack length and the thermal boundary layer.
    T_tip = sample_crack_tip_temperature(model, crack_tip_xy, T_delta_nd,
                                         radius=2.0e-3, n_ring=24)
    Gc_tip = float(temperature_dependent_Gc(T_tip)) if use_temperature_dependent_Gc else float(Gc0)
    return crack_tip_xy, T_tip, Gc_tip


def make_elastic_integration_points(model, vertex, X_f_thermal):
    """Create the integration batch used by the elastic energy solve.

    If elastic_integration_method == "mc", the elastic solve starts from the
    latest thermal MC batch.  This keeps the two fields sampled at the same
    points unless elastic resampling is explicitly enabled.

    If elastic_integration_method == "gaussian", the elastic solve uses its own
    deterministic Gauss batch.  In that case thermal and elastic integration
    points are intentionally different.
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

    Elastic resampling is only meaningful for MC integration.  For Gaussian
    integration the batch is fixed and deterministic.
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
            f.write("step,prop_iter,v_delta_mm,T_delta_K,K1,K2,J,T_tip_K,Gc_tip,J_over_Gc,beta,theta,tip_x,tip_y,propagated\n")
        f.write(",".join(str(x) for x in row) + "\n")


# -----------------------------------------------------------------------------
# Networks with the crack embedding as input
# -----------------------------------------------------------------------------

def make_crack_embedding(points):
    """Build the shared crack embedding."""
    points = clean_crack_points(points)
    if len(points) < 2:
        points = [[0.0, 0.5], [0.5, 0.5]]
    return _build_crack_embedding(
        points,
        decay_alpha=crack_embedding_decay_alpha,
        decay_p=crack_embedding_decay_p,
        endpoint_overlap=crack_embedding_endpoint_overlap,
    )


# -----------------------------------------------------------------------------
# Build model and integration mesh
# -----------------------------------------------------------------------------
crack_embedding = make_crack_embedding(crack_points)
crack_geometry = CrackGeometry.from_embedding(crack_embedding, crack_points)

# Mechanical network: NN_uv(x, y, Gamma) -> [u_hat, v_hat].
# The asymptotic enrichment (Williams terms) is added in net_uv().
uv_core = AxisScaleNet(
    MLP([3, 50, 50, 50, 2], activation=nn.Tanh),
    lb=model_params["lb"],
    ub=model_params["ub"],
)
model_uv = extendAxisNet(uv_core, crack_embedding)

# Thermal network: NN_T(x, y, Gamma) -> T_hat.
thermal_core = ThermalNet([3, 50, 50, 50, 1], lb=model_params["lb"], ub=model_params["ub"], omega_0=30.0)
model_T = extendAxisNet(thermal_core, crack_embedding)

model = ThermoSolver2D(model_params, model_uv=model_uv, model_T=model_T, crack_geometry=crack_geometry)

# Shared K/J extraction: interaction integral with guards, COD fallback,
# cross-check and per-step validation plots (see tmdem/sif_estimation.py).
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
deltaT = deltaT_phys / model.DeltaT_ref
deltaV = deltaV_phys / model.U_ref
fdGraph = np.full((numSteps, 2), np.nan, dtype=np.float64)

# -----------------------------------------------------------------------------
# Optional reference force-displacement curve (FEM)
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

    v_delta_nd = deltaV * step
    T_delta_nd = deltaT * min(step, num_temp_steps - 1)
    T_delta_nd_prev = deltaT * min(step - 1, num_temp_steps - 1) if step >= 1 else 0.0

    v_delta_phys = v_delta_nd * model.U_ref
    T_delta_phys = T_delta_nd * model.DeltaT_ref

    print(f"\n=== Simulation Step {step} ===")
    print(f"Applied displacement: {v_delta_phys:.6e} mm | Applied top temperature increment: {T_delta_phys:.2f} K")
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

    # ---- Thermal solve: Monte Carlo points, redrawn during training ----
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

    # ---- Crack propagation within the current load step ----
    prop_iter = 0
    prev_ratio = None
    while enable_crack_propagation and step > 0 and prop_iter < max_propagations_per_step:
        if crack_reached_boundary_rect(crack_points, domainEndsX, domainEndsY, zero_force_boundaries, model=model):
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
        ok, why = extraction_is_plausible(ratio, prev_ratio)
        if not ok:
            print(f"[propagation] Rejected extraction at step {step} iter {prop_iter}: {why}. "
                  f"K1={K1:.4e}, K2={K2:.4e}. Stop propagation at this step.")
            break
        prev_ratio = ratio
        print(
            f"[prop-check step {step} iter {prop_iter}] "
            f"K1={K1:.4e}, K2={K2:.4e}, J_straight={J:.4e}, "
            f"G_eff={G_drive:.4e} (theta={np.rad2deg(theta_drive):+.2f} deg), "
            f"Gc(Ttip)={Gc_tip_prop:.4e}, G_eff/Gc={ratio:.3f}, "
            f"crack_len={total_crack_length(crack_points, model=model):.4f} mm"
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

        new_tip, beta_new, theta_kink = propose_crack_extension_rect(
            model, crack_points, K1, K2, domainEndsX, domainEndsY,
            crack_increment=crack_increment,
            propagation_mode=propagation_mode,
            max_kink_angle_deg=max_kink_angle_deg,
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
            step, prop_iter, f"{v_delta_phys:.12e}", f"{T_delta_phys:.12e}",
            f"{K1:.12e}", f"{K2:.12e}", f"{J:.12e}",
            f"{T_tip_prop:.12e}", f"{Gc_tip_prop:.12e}", f"{ratio:.12e}",
            f"{beta_new:.12e}", f"{theta_kink:.12e}",
            f"{new_tip[0]:.12e}", f"{new_tip[1]:.12e}", 1,
        ])

        if crack_reached_boundary_rect(crack_points, domainEndsX, domainEndsY, zero_force_boundaries, model=model):
            print(
                "[propagation] Crack reached a stopping boundary. "
                "Skip post-extension re-training and keep force = 0 from this point."
            )
            break

        # Re-train both fields on the same load step after changing the crack.
        # The previous-step temperature Temp_n_f is kept as the transient state.
        print("[propagation] Re-training thermal field after crack extension")
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
    # Keep the thermal state on the MC integration points because the next
    # transient heat step needs T_n at the new or redrawn integration points.
    T_pred = model.predict_thermal(X_f_T[:, :2], T_delta_nd)
    Temp_n_f = T_pred.reshape(-1, 1)
    fields = model.predict_fields(X_f_U[:, :2], T_delta_nd, v_delta_nd)

    # Use a deterministic post-processing grid for plots.  The fields are
    # discontinuous across the crack, so plotting from MC points with griddata
    # can create artificial mosaic patterns near the crack.
    X_plot = make_prediction_grid(domainEndsX, domainEndsY, n=501)
    fields_plot = model.predict_fields(X_plot, T_delta_nd, v_delta_nd)
    fields_plot_display = dict(fields_plot)
    # Plot-only mask of a narrow band along the crack for the discontinuous
    # fields.  It removes the jagged edge produced where the sharp jump across
    # the crack cuts through rectangular plotting pixels, in particular on
    # kinked cracks.  The band width falls back to 0.01 when the model has no
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
        X_f=X_f_U,      # same as X_f_U, the elastic integration batch
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
        u_tip_raw=fields.get("u_tip_raw"),
        v_tip_raw=fields.get("v_tip_raw"),
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
        u_tip_raw_plot=fields_plot.get("u_tip_raw"),
        v_tip_raw_plot=fields_plot.get("v_tip_raw"),
        T_smooth_contrib_K_plot=fields_plot.get("T_smooth_contrib_K"),
        T_jump_contrib_K_plot=fields_plot.get("T_jump_contrib_K"),
        crack_points=np.asarray(crack_points, dtype=np.float64),
        crack_curve_points=np.asarray(model.crack_geometry.curve_points, dtype=np.float64),
    )

    # Tip quantities at the end of the load step, after any propagation.
    crack_tip_xy = np.asarray(model.crack_geometry.tip, dtype=np.float64)
    if crack_reached_boundary_rect(crack_points, domainEndsX, domainEndsY, zero_force_boundaries, model=model):
        # Once the crack reaches a stopping boundary the specimen is separated
        # and the LEFM tip quantities are no longer meaningful, so the SIFs
        # are not evaluated.
        T_tip = sample_crack_tip_temperature(model, crack_tip_xy, T_delta_nd, radius=2.0e-3, n_ring=24)
        if not np.isfinite(T_tip):
            # Fall back to the linear lifting temperature T_ref + DeltaT * y.
            T_tip = float(model.T_ref + model.DeltaT_ref * T_delta_nd * np.clip(crack_tip_xy[1], 0.0, 1.0))
        Gc_tip = float(temperature_dependent_Gc(T_tip)) if use_temperature_dependent_Gc else float(Gc0)
        K1_tip, K2_tip, J_tip, beta_tip = 0.0, 0.0, 0.0, last_tangent_normal(crack_points, model=model)[2]
    else:
        crack_tip_xy, T_tip, Gc_tip = get_tip_fracture_properties(model, crack_points, T_delta_nd)
        K1_tip, K2_tip, J_tip, beta_tip = sif.estimate(crack_points, T_delta_nd, v_delta_nd)
        sif.save_validation(step_dir, step, crack_points, T_delta_nd, v_delta_nd)
    np.savez(
        os.path.join(step_dir, f"fracture_tip_step_{step}.npz"),
        crack_tip=crack_tip_xy,
        T_tip=T_tip,
        Gc_tip=Gc_tip,
        Gc0=Gc0,
        J_tip=J_tip,
        K1_tip=K1_tip,
        K2_tip=K2_tip,
        J_over_Gc=J_tip / max(Gc_tip, 1.0e-30),
        crack_points=np.asarray(crack_points, dtype=np.float64),
        crack_curve_points=np.asarray(model.crack_geometry.curve_points, dtype=np.float64),
        use_temperature_dependent_Gc=use_temperature_dependent_Gc,
    )
    save_crack_path(crack_points, step_dir, step, domainEndsX, domainEndsY, crack_geometry=model.crack_geometry, show=True)

    if crack_reached_boundary_rect(crack_points, domainEndsX, domainEndsY, zero_force_boundaries, model=model):
        # After separation the remaining energy derivative is numerical noise.
        force = 0.0
    else:
        force = model.reaction_force(X_f_U, T_delta_nd, v_delta_nd)

    # One row per load step: everything needed to replot the SIF/J and FD
    # curves against the applied displacement without loading checkpoints.
    tip_history_file = os.path.join(results_dir, "fracture_tip_history.csv")
    if step == 0 or not os.path.exists(tip_history_file):
        with open(tip_history_file, "w", encoding="utf-8") as f:
            f.write("step,v_delta_mm,T_delta_K,T_tip_K,Gc_tip,J_tip,J_over_Gc,K1_tip,K2_tip,crack_length,tip_x,tip_y,force_N\n")
    with open(tip_history_file, "a", encoding="utf-8") as f:
        f.write(
            f"{step},{v_delta_phys:.12e},{T_delta_phys:.12e},"
            f"{T_tip:.12e},{Gc_tip:.12e},{J_tip:.12e},{J_tip / max(Gc_tip, 1.0e-30):.12e},"
            f"{K1_tip:.12e},{K2_tip:.12e},{total_crack_length(crack_points, model=model):.12e},"
            f"{crack_tip_xy[0]:.12e},{crack_tip_xy[1]:.12e},{force:.12e}\n"
        )

    # Save a restart checkpoint before plotting.  This prevents losing a whole
    # trained step if a post-processing/plotting error occurs.
    fdGraph[step, 0] = v_delta_phys
    fdGraph[step, 1] = force
    np.save(os.path.join(results_dir, "fdGraph.npy"), fdGraph)
    print(
        f"Displacement = {v_delta_phys:.6e} mm, Force = {force:.6f}, "
        f"T_tip = {T_tip:.2f} K, Gc_tip = {Gc_tip:.6e}"
    )

    mesh_meta = {
        "nx_elem": nx_elem,
        "ny_elem": ny_elem,
        "Gc0": Gc0,
        "use_temperature_dependent_Gc": use_temperature_dependent_Gc,
        "Gc_tip": Gc_tip,
        "T_tip": T_tip,
        "J_tip": J_tip,
        "K1_tip": K1_tip,
        "K2_tip": K2_tip,
        "K_I_trainable": model.trainable_sif_values()[0],
        "K_II_trainable": model.trainable_sif_values()[1],
        "use_thermal_jump": getattr(model, "use_thermal_jump", False),
        "use_explicit_mechanical_jump": getattr(model, "use_explicit_mechanical_jump", False),
        "J_over_Gc": J_tip / max(Gc_tip, 1.0e-30),
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

    # Loss histories of the thermal and mechanical training.
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
    u_tip_raw_diag = fields_plot.get("u_tip_raw")
    v_tip_raw_diag = fields_plot.get("v_tip_raw")
    if u_tip_raw_diag is not None:
        plot_field(X_plot, u_tip_raw_diag, domainEndsX, domainEndsY,
                   title=f"Tip enrichment u - Step {step}", label="u_tip (mm)",
                   filename=f"u_tip_raw_{step}", save_dir=step_dir, cmap="viridis", show=True)
    if v_tip_raw_diag is not None:
        plot_field(X_plot, v_tip_raw_diag, domainEndsX, domainEndsY,
                   title=f"Tip enrichment v - Step {step}", label="v_tip (mm)",
                   filename=f"v_tip_raw_{step}", save_dir=step_dir, cmap="viridis", show=True)
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
