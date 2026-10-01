#!/usr/bin/env python
"""
Stationary thermal edge crack in a strip.

Verification of the SIF extraction on a crack that does not move, against the
published value for the edge-cracked strip under thermal load of

    Wang, Tanaka, Oterkus and Oterkus (2023), Contin. Mech. Thermodyn. 35,
    1021-1040, Section 5.2 and Fig. 9.

Units are mm and N; temperatures are changes relative to the stress-free
state, in K.

Geometry
    Strip of width L = 0.5 mm (x in [-0.25, 0.25]) and height W = 2 mm
    (y in [-1, 1]), with a horizontal edge crack of length a = 0.25 mm along
    y = 0 from the cold face, so that the tip sits at the centre (0, 0) and
    a/L = 0.5.

Material
    Homogeneous, plane strain: E = 218.4 GPa, nu = 0.25, alpha = 1.67e-5 1/K.

Boundary conditions
    Thermal: steady conduction (rho = 0), temperature change -T0 on the face
    x = -L/2 and +T0 on the face x = +L/2, with T0 = 1 K; the top and bottom
    edges and the crack faces are adiabatic.
    Mechanical: rollers on the top and bottom edges (v = 0, u free); the two
    vertical faces and the crack faces are traction free.  The rollers remove
    the rotation and the y translation, and the x translation is removed by
    subtracting u at one point on the top edge, off the crack line, which
    changes no strain and no K.

The rollers prevent the cooled face from contracting, so sigma_yy goes into
tension and the horizontal crack opens in mode I.

Reference value
    K_I* = K_I (1 - nu) / (E alpha T0 sqrt(pi a)) = K_I / K0 = 0.500, from
    Table 3 of the reference, which lists K* together with its relative
    error.

Solution
    One thermal solve, then one mechanical solve, each with Adam followed by
    L-BFGS on a fixed draw of Monte Carlo integration points.  The SIFs are
    extracted by the interaction integral with the thermal area term of Wilson
    and Yu, and read off a radius sweep over five contour radii from 0.5 to 2
    times the nominal radius.

Command line
    --nx    background elements across the width (default 96); the paper uses
            48, 72 and 96
    --out   output folder (default runs/stationary_crack)

Environment variables
    NY      background elements along the height (default 4 * nx, square
            elements)
    NPTS    integration points per background element (default 9)
    ADAM    Adam iterations of each solve (default 8000)
    LBFGS   L-BFGS iterations of each solve (default 2000)
    RSIF    nominal interaction contour radius in mm (default 3.2 h)
    ENRICH  decay factor of the asymptotic enrichment,
            tip_enrichment_decay_factor (default 5.0)
    KSEED   initial value of the trainable K_I amplitude of the enrichment
            (default 0.0)
    DEV     "cpu" to run on the CPU (default: CUDA when available)

Output
    Printed: the trained enrichment amplitudes, K1, K2, K1/K0 against the
    reference, and the verdict of the radius sweep with its exponent p.
    Written to the output folder: radius_sweep_nx<nx>.csv, or
    radius_sweep_nx<nx>_kseed<KSEED>.csv when KSEED is set, with the columns
    radius_mm, K1, K2 and K1_over_K0 for every radius of the sweep.

    python examples/stationary_crack.py --nx 48
    python examples/stationary_crack.py --nx 72
    python examples/stationary_crack.py --nx 96

Author: Han Zhang (han.zhang7@unsw.edu.au)
"""

import os
import argparse
import time

import numpy as np
import torch
import torch.nn as nn

import _entry  # noqa: F401  (puts the repository root on the import path)

from tmdem.crack_embedding import extendAxisNet, make_crack_embedding as _build
from tmdem.solver import (AxisScaleNet, CrackGeometry, MLP, ThermalNet,
                          ThermoSolver2D, get_uniform_mesh, set_seed)
from tmdem.sif_estimation import SIFEstimator

cli = argparse.ArgumentParser(description="Stationary thermal edge crack in a strip (Wang et al. 2023).")
cli.add_argument("--nx", type=int, default=96,
                 help="background elements across the width; the paper uses 48, 72 and 96 (default)")
cli.add_argument("--out", default=os.path.join("runs", "stationary_crack"),
                 help="output folder (default runs/stationary_crack)")
cli_args = cli.parse_args()
os.makedirs(cli_args.out, exist_ok=True)
os.chdir(cli_args.out)


# -----------------------------------------------------------------------------
# Geometry and material of Wang et al. (2023), Fig. 9 and Table 2, in mm and N
# -----------------------------------------------------------------------------
HALF_L = 0.25           # half the 0.5 mm across the gradient, x in [-0.25, 0.25]
HALF_W = 1.0            # half the 2 mm height, y in [-1, 1]
A_CRACK = 0.25          # from the cold edge to the centre, so the tip is at x=0
E = 218.4e3             # N/mm^2, that is 218.4 GPa
NU = 0.25
ALPHA = 1.67e-5         # 1/K
T0 = 1.0                # K on the hot face, minus that on the cold one

# Normalization of their eq. (37), K0 = E alpha T0 sqrt(pi a) / (1 - nu), with
# the reference value K_I / K0 from their Table 3.
K0 = E * ALPHA * T0 * np.sqrt(np.pi * A_CRACK) / (1.0 - NU)
REFERENCE = 0.500

nx_elem = cli_args.nx
ny_elem = int(os.environ.get("NY", str(4 * nx_elem)))
num_int_pts = int(os.environ.get("NPTS", "9"))
n_adam = int(os.environ.get("ADAM", "8000"))
n_lbfgs = int(os.environ.get("LBFGS", "2000"))
device_name = os.environ.get("DEV", "cuda" if torch.cuda.is_available() else "cpu")

domainEndsX = np.array([-HALF_L, HALF_L])
domainEndsY = np.array([-HALF_W, HALF_W])
crack_points = [[-HALF_L, 0.0], [0.0, 0.0]]     # mouth first, tip last
background_cell = 2.0 * HALF_L / nx_elem
sif_radius = float(os.environ.get("RSIF", 3.2 * background_cell))

model_params = {
    "E": E, "nu": NU, "alpha": ALPHA,
    # Plane strain carries (1 + nu) alpha in the reduced constitutive law.  The
    # flag that applies it defaults to False, so it is set together with
    # plane_type.
    "plane_strain_thermal_alpha_correction": True,
    "plane_type": "plane_strain",
    # rho = 0 drops the storage term, so the conduction problem is steady, as in
    # the reference.
    "k": 1.0, "rho": 0.0, "c": 1.0,
    "T_ref": 0.0, "dt": 1.0,
    "U_ref": ALPHA * T0 * HALF_L,      # free thermal stretch over a half width
    "DeltaT_ref": T0,
    "lb": np.array([-HALF_L, -HALF_W]), "ub": np.array([HALF_L, HALF_W]),
    "use_thermal_jump": False,
    "use_explicit_mechanical_jump": False,
    # The temperature gradient runs across the crack, so the penalty on the
    # normal heat flux at the adiabatic crack faces is set well above the default
    # 1e-6.
    "adiabatic_crack_weight": 1.0e-2,
    "adiabatic_crack_samples": 200,
    "crack_face_offset": 1.0e-4,
    "use_crack_conductivity_degradation": False,
    "use_tip_enrichment": True,
    "tip_enrichment_decay_factor": float(os.environ.get("ENRICH", "5.0")),
    # Initial value of the trainable K_I amplitude of the enrichment.  The whole
    # thermal load is applied in one solve, and a nonzero seed only breaks the
    # symmetry.
    "K_I_init": float(os.environ.get("KSEED", "0.0")),
}


class EdgeStrip(ThermoSolver2D):
    """Homogeneous strip, gradient across the width, rollers top and bottom."""

    # -- temperature ------------------------------------------------------
    def net_thermal(self, x, y, T_delta_nd):
        """theta = -T0 at x = -L/2 and +T0 at x = +L/2, top and bottom natural.

        Without the crack the exact solution is the linear ramp below.  The
        adiabatic crack faces block the flux, so the field is the ramp plus a
        correction that the network supplies through the gate 1 - X^2, which
        vanishes on the two Dirichlet edges.
        """
        xy = self.xy_tensor(x, y)
        T_hat = self.model_T(xy)[:, 0:1]
        X = x / float(HALF_L)
        theta_nd = T_delta_nd * X + torch.clamp(1.0 - X ** 2, min=0.0) * T_hat
        return self.T_ref + self.DeltaT_ref * theta_nd

    # -- displacement -----------------------------------------------------
    def raw_uv_nd(self, xy):
        """u and v before the gauge, with the enrichment already gated."""
        x, y = xy[:, 0:1], xy[:, 1:2]
        uv = self.model_uv(xy)
        u_tip, v_tip = self.crack_tip_asymptotic_displacement(x, y)
        Y = y / float(HALF_W)
        gate = torch.clamp(1.0 - Y ** 2, min=0.0)
        # One scalar gate for the enrichment, not one per component: the
        # Williams field is a vector and scaling u and v differently distorts
        # the mode shape, which the shared amplitudes cannot absorb.
        u_nd = uv[:, 0:1] + gate * (u_tip / self.U_ref)
        v_nd = gate * (uv[:, 1:2] + v_tip / self.U_ref)
        return u_nd, v_nd

    def net_uv(self, x, y, v_delta_nd):
        """v = 0 on the top and bottom edges; u free, one translation gauged.

        The rollers remove the rotation and the y translation between them, so
        the only rigid mode left is the x translation.  Subtracting u at a
        single point removes it and changes no strain.
        """
        xy = self.xy_tensor(x, y)
        u_nd, v_nd = self.raw_uv_nd(xy)
        p = torch.tensor([[0.0, HALF_W]], dtype=xy.dtype, device=xy.device)
        u_p, _ = self.raw_uv_nd(p)
        return self.U_ref * (u_nd - u_p), self.U_ref * v_nd


def main():
    set_seed(2025)
    if device_name == "cpu":
        _real = torch.cuda.is_available
        torch.cuda.is_available = lambda: False
    emb = _build(crack_points, decay_alpha=1.0 / (5.0 * background_cell),
                 decay_p=1.0)
    geom = CrackGeometry.from_embedding(emb, crack_points)
    uv = extendAxisNet(AxisScaleNet(MLP([3, 50, 50, 50, 2], activation=nn.Tanh),
                                    lb=model_params["lb"],
                                    ub=model_params["ub"]), emb)
    th = extendAxisNet(ThermalNet([3, 50, 50, 50, 1], lb=model_params["lb"],
                                  ub=model_params["ub"], omega_0=30.0), emb)
    model = EdgeStrip(model_params, model_uv=uv, model_T=th, crack_geometry=geom)
    if device_name == "cpu":
        torch.cuda.is_available = _real
    print(f"device {model.device}")
    print(f"strip {2*HALF_L} x {2*HALF_W} mm, edge crack a = {A_CRACK} from the "
          f"cold edge, tip at (0, 0), a/L = {A_CRACK/(2*HALF_L):.2f}")
    print(f"E {E:.4g} N/mm^2, nu {NU}, alpha {ALPHA:.3e} /C, T0 = {T0} C, "
          f"plane strain")
    print(f"mesh {nx_elem} x {ny_elem}, cell {background_cell:.5f} x "
          f"{2*HALF_W/ny_elem:.5f} mm, contour {sif_radius:.5f} mm "
          f"({sif_radius/background_cell:.2f} cells, {sif_radius/A_CRACK:.3f} a)")
    print(f"K0 = {K0:.6f} N/mm^1.5, so the reference K_I is "
          f"{REFERENCE*K0:.6f}\n")

    vertex = get_uniform_mesh(nx_elem, ny_elem, domainEndsX, domainEndsY)
    xn, yn, wn, en = model.generate_integration_points(vertex, num_int_pts,
                                                       method="mc")
    X_f = np.column_stack([xn, yn, wn, en])
    print(f"{X_f.shape[0]} integration points")

    def tip_boundary_distance(tip):
        tip = np.asarray(tip, dtype=np.float64).ravel()
        return float(min(HALF_L - tip[0], HALF_L + tip[0],
                         HALF_W - tip[1], HALF_W + tip[1]))

    sif = SIFEstimator(
        model, tip_boundary_distance=tip_boundary_distance,
        interaction_radius=sif_radius, interaction_num_points=2880,
        cod_r_min=1.5 * background_cell, cod_r_max=6.0 * background_cell,
        cod_offset=1.0e-4, estimator="interaction_with_cod_fallback",
        limit_r_by_last_segment=False, use_thermal_domain_term=True)

    t0 = time.time()
    # Temp_n_f is the previous field a transient step would march from.  With
    # rho = 0 it drops out of the functional entirely and only has to have the
    # right shape; it is set to T_ref.
    Temp_n = np.full(X_f.shape[0], model_params["T_ref"], dtype=np.float64)
    model.train_thermal(X_f=X_f, T_delta_nd=1.0, Temp_n_f=Temp_n,
                        nIter_Adam=n_adam, lr_Adam=1.0e-3,
                        nIter_LBFGS=n_lbfgs, lr_LBFGS=0.1, nIter_log=500,
                        step=0, vertex=vertex,
                        num_integration_points=num_int_pts,
                        resample_every_iters=0)
    print(f"thermal solve {time.time() - t0:.1f} s")
    t0 = time.time()
    model.train_elastic(X_f=X_f, T_delta_nd=1.0, v_delta_nd=0.0,
                        nIter_Adam=n_adam, lr_Adam=1.0e-3,
                        nIter_LBFGS=n_lbfgs, lr_LBFGS=0.1, nIter_log=500,
                        vertex=vertex, num_integration_points=num_int_pts,
                        resample_every_iters=0)
    print(f"elastic solve {time.time() - t0:.1f} s")

    kI, kII = model.trainable_sif_values()
    print(f"\nenrichment amplitude after training: K_I {kI:.6e}  K_II {kII:.6e}")
    print(f"   for scale, the reference K_I is {REFERENCE*K0:.6e}")

    K1, K2, J, beta = sif.estimate(crack_points, 1.0, 0.0)
    v = sif.last_plateau or {}
    print(f"\nK1 = {K1:.6e}   K2 = {K2:.6e}   K2/K1 = {K2/K1:.4f}")
    print(f"K1/K0 = {K1/K0:.5f}   against the reference {REFERENCE:.3f}, "
          f"{100*(K1/K0 - REFERENCE)/REFERENCE:+.2f}%")
    print(f"radius sweep: {v.get('status', 'n/a')}, exponent p = "
          f"{v.get('p', float('nan')):.4f} over {v.get('n', 0)} radii")

    # The whole radius sweep is written, with the radii and their K taken from
    # sif.last_plateau.
    radii, k1s, k2s = v.get("radii"), v.get("K1"), v.get("K2")
    if radii and k1s:
        name = "radius_sweep_nx%s%s.csv" % (
            nx_elem, "" if not os.environ.get("KSEED") else
            "_kseed" + os.environ["KSEED"])
        with open(name, "w", encoding="utf-8") as f:
            f.write("radius_mm,K1,K2,K1_over_K0\n")
            for r, a, b in zip(radii, k1s, k2s):
                f.write("%.9e,%.9e,%.9e,%.9e\n" % (r, a, b, a / K0))
        print("wrote %s (%d radii, %.4f to %.4f mm)"
              % (name, len(radii), min(radii), max(radii)))


if __name__ == "__main__":
    main()
