"""Crack-tip post-processing shared by the numerical examples.

Each example builds one SIFEstimator and keeps only its example-specific
pieces: domain geometry, fracture energy laws, boundary conditions and
propagation policy.  This module provides:

  * The interaction integral for K_I and K_II.  Its contour term (M_integral)
    is built from elastic strains and thermoelastic stresses through
    ThermoSIFAdapter, which removes the thermal strain; the thermal area term
    of Wilson and Yu (thermal_interaction_term, SIFEstimator.thermal_domain_dK)
    supplies the part of the thermal contribution that the contour term
    cannot carry.
  * The radius sweep over contour radii and its admissibility check
    (SIFEstimator.radius_sweep, sweep_verdict, extraction_is_validated).
  * The crack-opening (COD) fallback.  The interaction contour must stay
    inside the domain (the network fields have no meaning outside it); when
    the tip is too close to a boundary the estimator falls back to two-sided
    COD sampling, which only needs points behind the tip.
  * The maximum hoop stress kink angle, the SIFs of the kinked tip and the
    energy release rate of the kink used by the growth criterion.

Author: Han Zhang (han.zhang7@unsw.edu.au)
"""

import os
from typing import Callable, Optional

import numpy as np
import torch

from .solver import (
    CrackGeometry,
    cod_face_jump_sif,
    plot_cod_extrapolation,
    plot_sif_radius_sweep,
)
from .crack_embedding import clean_crack_points


# Admissibility threshold of the radius sweep, as an exponent p of K ~ r^p so
# that sweeps of different widths are judged alike: the exponent at which K
# drifts by 0.15 over the full factor-of-four sweep, 1 - 4^(-p) = 0.15.
_PATH_DEPENDENT_P = -float(np.log(1.0 - 0.15)) / float(np.log(4.0))   # 0.1172

# A sweep narrower than this span was cut short by the contour guard (domain
# boundary or kink).  The nominal sweep spans a factor of four, and a
# threshold of three leaves room for one radius dropped at its edge.
_CLIPPED_SPAN = 3.0

# Number of consecutive crack increments that may be taken on a clipped sweep
# that did not pass.  The count is reset by a full-width sweep that passes,
# and zero disables the release.  Directly after a kink the contour is
# limited to the straight run behind the tip, so the sweep is clipped until
# the crack has grown a straight run again.  Releases are counted in
# SIFEstimator.clipped_releases.
_CLIPPED_SWEEP_GRACE = 2


# ---------------------------------------------------------------------------
# Small geometry/material helpers shared by all examples
# ---------------------------------------------------------------------------
def plane_strain_flag(model) -> bool:
    return str(getattr(model, "plane_type", "plane_strain")).lower() == "plane_strain"


def J_from_K(K1, K2, E, nu, plane_strain=True):
    """LEFM energy release rate from SIFs."""
    if plane_strain:
        return (1.0 - nu ** 2) / E * (K1 ** 2 + K2 ** 2)
    return (K1 ** 2 + K2 ** 2) / E


def local_material_at_tip(model, x):
    """Local E, nu, mu at coordinate x (graded materials via material_at_x_np)."""
    if hasattr(model, "material_at_x_np"):
        E, nu, mu = model.material_at_x_np(x)
        return float(E), float(nu), float(mu)
    E = float(model.E)
    nu = float(model.nu)
    mu = float(getattr(model, "mu", E / (2.0 * (1.0 + nu))))
    return E, nu, mu


def get_active_crack_geometry(model, crack_points=None, embedding_factory=None):
    """Return the active crack geometry, rebuilding from points if needed."""
    geom = getattr(model, "crack_geometry", None)
    if geom is not None and hasattr(geom, "tip"):
        return geom
    if crack_points is None or embedding_factory is None:
        raise ValueError("model has no crack geometry and no fallback points/factory given")
    emb = embedding_factory(crack_points)
    return CrackGeometry.from_embedding(emb, crack_points)


def total_crack_length(points, model=None):
    """Length of the active crack curve when model is available, else polyline."""
    if model is not None:
        try:
            return float(model.crack_geometry.total_length)
        except Exception:
            pass
    pts = [np.asarray(p, dtype=np.float64) for p in points]
    return float(sum(np.linalg.norm(p1 - p0) for p0, p1 in zip(pts[:-1], pts[1:])))


def last_tangent_normal(points, model=None):
    """Tangent, normal and angle beta of the crack at the active tip.

    Taken from the model's crack geometry when available, else from the last
    segment of the polyline.
    """
    if model is not None:
        try:
            tangent = np.asarray(model.crack_geometry.last_tangent, dtype=np.float64)
            normal = np.asarray(model.crack_geometry.last_normal, dtype=np.float64)
            beta = float(model.crack_geometry.last_beta)
            return tangent, normal, beta
        except Exception:
            pass

    p0 = np.asarray(points[-2], dtype=np.float64)
    p1 = np.asarray(points[-1], dtype=np.float64)
    tangent = p1 - p0
    nrm = np.linalg.norm(tangent)
    if nrm < 1.0e-14:
        tangent = np.array([1.0, 0.0], dtype=np.float64)
    else:
        tangent = tangent / nrm
    normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
    beta = float(np.arctan2(tangent[1], tangent[0]))
    return tangent, normal, beta


def set_crack_state(model, points, embedding_factory):
    """Install a new crack polyline on both networks and the model geometry."""
    points = clean_crack_points([list(map(float, p)) for p in points])
    embedding = embedding_factory(points)
    if hasattr(model.model_uv, "set_extend_axis"):
        model.model_uv.set_extend_axis(embedding)
    if hasattr(model.model_T, "set_extend_axis"):
        model.model_T.set_extend_axis(embedding)
    model.crack_geometry = CrackGeometry.from_embedding(embedding, points)
    return points, embedding


def sample_crack_tip_temperature(model, crack_tip_xy, T_delta_nd, radius=0.0, n_ring=16, keep_fn=None):
    """Absolute tip temperature T_tip, optionally averaged over a ring.

    With ``radius`` > 0 the temperature is averaged over ``n_ring`` points on
    a ring of that radius around the tip; otherwise it is taken at the tip.
    ``keep_fn(points) -> bool mask`` filters ring points to the domain; the
    default keeps points inside the [lb, ub] bounding box.
    """
    tip = np.asarray(crack_tip_xy, dtype=np.float64).reshape(1, 2)

    if radius is None or radius <= 0.0:
        X_tip = tip.astype(np.float32)
    else:
        theta = np.linspace(0.0, 2.0 * np.pi, n_ring, endpoint=False)
        ring = tip + radius * np.column_stack([np.cos(theta), np.sin(theta)])
        if keep_fn is None:
            lb = np.asarray(model.lb, dtype=np.float64)
            ub = np.asarray(model.ub, dtype=np.float64)
            keep = np.all((ring >= lb) & (ring <= ub), axis=1)
        else:
            keep = np.asarray(keep_fn(ring), dtype=bool)
        X_tip = ring[keep].astype(np.float32)
        if X_tip.shape[0] == 0:
            X_tip = tip.astype(np.float32)

    T_tip = model.predict_thermal(X_tip, T_delta_nd)
    return float(np.mean(np.asarray(T_tip).reshape(-1)))


def max_hoop_kink_angle(K1, K2, max_kink_angle_deg=80.0, k_ratio_deadband=0.0, relaxation=1.0):
    """Maximum hoop stress kink angle (Erdogan and Sih) in the local tip frame.

    The angle is clipped to +-``max_kink_angle_deg``.  ``k_ratio_deadband``
    suppresses the kink when |K2|/|K1| is below the given ratio (noise
    deadband) and ``relaxation`` scales the clipped angle; both default to the
    plain criterion.
    """
    if not np.isfinite(K1) or not np.isfinite(K2):
        return 0.0
    if abs(K2) < 1.0e-12:
        return 0.0
    if k_ratio_deadband > 0.0 and abs(K2) / (abs(K1) + 1.0e-12) < k_ratio_deadband:
        return 0.0
    disc = np.sqrt(K1 ** 2 + 8.0 * K2 ** 2)
    theta = 2.0 * np.arctan((K1 - disc) / (4.0 * K2))
    max_angle = np.deg2rad(max_kink_angle_deg)
    theta = float(np.clip(theta, -max_angle, max_angle))
    return float(relaxation * theta)


def extraction_is_plausible(ratio, prev_ratio=None, *, ratio_cap=5.0, jump_cap=2.0):
    """Guard against acting on an extraction from a field that has not
    re-converged.

    Several crack increments may be taken in one load step, each followed by a
    partial retraining of the fields.  A field that has not re-converged gives
    finite but wrong SIFs, and the growth direction is taken from the same
    SIFs.  ``ratio`` is the driving force ratio of the growth criterion: a value
    above ``ratio_cap`` is rejected, and so is a jump by more than ``jump_cap``
    relative to the previous extraction in the same load step.

    Returns ``(ok, reason)``; ``reason`` is empty when the value is accepted.
    """
    if not np.isfinite(ratio):
        return False, "non-finite driving force"
    if ratio > ratio_cap:
        return False, (f"driving force ratio {ratio:.3g} exceeds the plausibility "
                       f"cap {ratio_cap:.3g}; the field has not re-converged")
    if prev_ratio is not None and np.isfinite(prev_ratio) and prev_ratio > 0.0:
        if ratio > jump_cap * prev_ratio:
            return False, (f"driving force ratio jumped {ratio / prev_ratio:.2g}x "
                           f"({prev_ratio:.3g} -> {ratio:.3g}) within one load step")
    return True, ""


def kink_local_sifs(K1, K2, theta):
    """Local SIFs of an infinitesimal kink at ``theta`` (first order).

    First-order kinked-tip SIFs in the sense of Cotterell and Rice, with the
    parent-tip K1, K2 as input.  Pure mode II kinking at -70.53 deg gives
    k1 = (2/sqrt(3)) K2 and k2 = 0, hence an energy release rate 4/3 times the
    straight-ahead value.
    """
    t = 0.5 * float(theta)
    c, s = np.cos(t), np.sin(t)
    k1 = c ** 3 * K1 - 3.0 * c ** 2 * s * K2
    k2 = c ** 2 * s * K1 + c * (1.0 - 3.0 * s ** 2) * K2
    return float(k1), float(k2)


def effective_energy_release_rate(K1, K2, E_eff, theta=None, **kink_angle_kwargs):
    """Driving force evaluated on the kinked configuration.

        G(theta) = [k1(theta)^2 + k2(theta)^2] / E_eff

    with k1, k2 the SIFs at the tip of an infinitesimal kink at ``theta``
    (``kink_local_sifs``), after Nuismer (1975) and Cotterell and Rice (1980).
    In pure mode I (theta = 0) it equals the straight-ahead rate K1^2/E_eff;
    in pure mode II the maximum hoop stress angle of 70.53 deg gives
    (4/3) K2^2/E_eff.  Unlike the straight-ahead form (K1^2 + K2^2)/E_eff,
    which is even in both SIFs, the kinked form carries the product K1*K2 and
    so distinguishes a thermal state that opens the crack from one that closes
    it whenever K2 is nonzero.

    Returns ``(G, theta)``.
    """
    if theta is None:
        theta = max_hoop_kink_angle(K1, K2, **kink_angle_kwargs)
    k1, k2 = kink_local_sifs(K1, K2, theta)
    return float((k1 ** 2 + k2 ** 2) / E_eff), float(theta)


def propose_crack_extension_rect(model, points, K1, K2, domain_x, domain_y, *,
                                 crack_increment, propagation_mode="max_hoop",
                                 max_kink_angle_deg=80.0,
                                 k_ratio_deadband=0.0, kink_angle_relaxation=1.0):
    """Propose a new control point inside a rectangular domain.

    The crack increment is taken in the maximum hoop stress direction, or
    straight ahead when ``propagation_mode == "straight"``.  Near a boundary
    it is shortened so that the new tip lands on the boundary.

    Returns ``(new_tip, beta_new, theta)``; ``new_tip`` is None when the tip
    cannot move.
    """
    tangent, normal, beta = last_tangent_normal(points, model=model)
    theta = 0.0 if propagation_mode == "straight" else max_hoop_kink_angle(
        K1, K2, max_kink_angle_deg=max_kink_angle_deg,
        k_ratio_deadband=k_ratio_deadband, relaxation=kink_angle_relaxation)
    beta_new = beta + theta
    direction = np.array([np.cos(beta_new), np.sin(beta_new)], dtype=np.float64)
    tip = np.asarray(model.crack_geometry.tip, dtype=np.float64)

    scale = 1.0
    hit_boundary = False
    for i, bounds in enumerate([domain_x, domain_y]):
        lo, hi = float(bounds[0]), float(bounds[1])
        if abs(direction[i]) <= 1.0e-14:
            continue
        if tip[i] + crack_increment * direction[i] < lo:
            cand = (lo - tip[i]) / (crack_increment * direction[i])
            if cand < scale:
                scale = cand
                hit_boundary = True
        elif tip[i] + crack_increment * direction[i] > hi:
            cand = (hi - tip[i]) / (crack_increment * direction[i])
            if cand < scale:
                scale = cand
                hit_boundary = True

    scale = max(0.0, min(1.0, scale))
    new_tip = tip + scale * crack_increment * direction

    if np.linalg.norm(new_tip - tip) < 1.0e-12:
        return None, beta_new, theta
    if hit_boundary:
        new_tip[0] = min(max(new_tip[0], float(domain_x[0])), float(domain_x[1]))
        new_tip[1] = min(max(new_tip[1], float(domain_y[0])), float(domain_y[1]))

    return new_tip.tolist(), beta_new, theta


def make_rect_boundary_distance(domain_x, domain_y):
    """Return tip_boundary_distance(tip) for a rectangular domain."""
    x0, x1 = float(domain_x[0]), float(domain_x[1])
    y0, y1 = float(domain_y[0]), float(domain_y[1])

    def tip_boundary_distance(tip):
        tip = np.asarray(tip, dtype=np.float64)
        return float(min(tip[0] - x0, x1 - tip[0], tip[1] - y0, y1 - tip[1]))

    return tip_boundary_distance


def crack_reached_boundary_rect(points, domain_x, domain_y, boundaries=("right",), model=None, tol=1.0e-10):
    """True when the active tip reaches a selected rectangular boundary.

    ``tol`` doubles as a ligament margin: the tip counts as having reached the
    boundary once it comes within ``tol`` of it.  Passing the interaction
    contour radius stops the analysis while the contour still fits inside the
    material.
    """
    if model is not None:
        try:
            tip = np.asarray(model.crack_geometry.tip, dtype=np.float64)
        except Exception:
            tip = np.asarray(points[-1], dtype=np.float64)
    else:
        tip = np.asarray(points[-1], dtype=np.float64)

    x0, x1 = float(domain_x[0]), float(domain_x[1])
    y0, y1 = float(domain_y[0]), float(domain_y[1])
    checks = {
        "left": tip[0] <= x0 + tol,
        "right": tip[0] >= x1 - tol,
        "bottom": tip[1] <= y0 + tol,
        "top": tip[1] >= y1 - tol,
    }
    return any(checks.get(side, False) for side in boundaries)


# ---------------------------------------------------------------------------
# Contour term of the interaction integral
# ---------------------------------------------------------------------------
class M_integral(object):
    """Contour term of the interaction integral for mixed-mode K1/K2 extraction.

    The auxiliary fields are the unit-K Williams near-tip fields.  ``model``
    must expose the adapter interface: pred_uv(xy), compute_Strain(u, v, xy)
    returning engineering shear, and constitutive(exx, eyy, exy).  See
    ThermoSIFAdapter below.  The thermal area term is added separately by
    SIFEstimator.thermal_domain_dK.
    """

    def __init__(self, model, device: torch.device):
        self.model = model
        self.device = device
    def _rot_mat(self, beta: float, device):
        # float(beta): an np.float64 would give a float64 tensor, which cannot
        # be multiplied with the float32 field tensors
        c = torch.cos(torch.tensor(float(beta), device=device))
        s = torch.sin(torch.tensor(float(beta), device=device))
        R = torch.stack([torch.stack([c,  s], -1),
                        torch.stack([-s, c], -1)], -2)  # 2x2
        return R

    def _strain_vec_to_tensor(self, exx, eyy, exy):
        # exy is the tensor shear strain eps_xy (not the engineering strain)
        # Returns the 2x2 strain tensor
        e12 = exy
        e21 = exy
        E = torch.stack([
            torch.stack([exx, e12], -1),
            torch.stack([e21, eyy], -1)
        ], -2)
        return E

    def _tensor_to_strain_vec(self, E):
        # 2x2 -> (exx, eyy, exy)
        exx = E[...,0,0]
        eyy = E[...,1,1]
        exy = 0.5*(E[...,0,1] + E[...,1,0])  # tensor shear
        return exx, eyy, exy

    def _rotate_tensor(self, T, beta, device):
        R = self._rot_mat(beta, device)
        return R @ T @ R.transpose(-1, -2)

    def _rotate_vector(self, v, beta, device):
        R = self._rot_mat(beta, device)
        return (R @ v.unsqueeze(-1)).squeeze(-1)

    def _aux_stress_local_unitKI(self, r, th):
        # Williams near-tip field for unit K_I = 1 (in MPa*sqrt(mm))
        # Returns sigma_xx, sigma_yy, sigma_xy in the local Cartesian frame,
        # x1 along the crack
        root = torch.sqrt(2*np.pi*r).clamp(min=1e-12)
        c = torch.cos(th/2.0); s = torch.sin(th/2.0)
        c3 = torch.cos(3*th/2.0); s3 = torch.sin(3*th/2.0)

        sxx = (c*(1 - s*s3)) / root
        syy = (c*(1 + s*s3)) / root
        sxy = (s*c*c3) / root
        return sxx, syy, sxy

    def _aux_stress_local_unitKII(self, r, th):
        # Same for unit K_II = 1
        root = torch.sqrt(2*np.pi*r).clamp(min=1e-12)
        c = torch.cos(th/2.0); s = torch.sin(th/2.0)
        c3 = torch.cos(3*th/2.0); s3 = torch.sin(3*th/2.0)

        sxx = -(s*(2 + c*c3)) / root
        syy =  (s*c*c3) / root
        sxy =   c*(1 - s*s3)  / root
        return sxx, syy, sxy

    def aux_u_col1_I(self, r, theta, E, nu, plane_strain=True):
        """
        x1 derivatives u_{i,1}^a (i = 1, 2) of the unit-K_I auxiliary displacement.
        r, theta: tensors of shape (N,), polar coordinates in the local tip frame
        Returns: (N, 2) = [u_{1,1}^a, u_{2,1}^a]
        """
        mu = E / (2*(1+nu))
        if plane_strain:
            kappa = 3 - 4*nu
        else:
            kappa = (3 - nu) / (1 + nu)

        fac = 1.0 / (2*mu) * torch.sqrt(r/(2*np.pi))

        # auxiliary displacement (K_I = 1)
        u1 = fac * torch.cos(theta/2) * (kappa - 1 + 2*torch.sin(theta/2)**2)
        u2 = fac * torch.sin(theta/2) * (kappa + 1 - 2*torch.cos(theta/2)**2)

        # derivatives with respect to r and theta
        du1_dr = (1/(4*mu)) * (1/torch.sqrt(2*np.pi*r)) * torch.cos(theta/2) * (kappa - 1 + 2*torch.sin(theta/2)**2)
        du2_dr = (1/(4*mu)) * (1/torch.sqrt(2*np.pi*r)) * torch.sin(theta/2) * (kappa + 1 - 2*torch.cos(theta/2)**2)

        du1_dtheta = fac * (torch.sin(theta)*torch.cos(theta/2) + (kappa-torch.cos(theta))*(-1)*(torch.sin(theta/2))/2)
        du2_dtheta = fac * (torch.sin(theta)*torch.sin(theta/2) + (kappa - torch.cos(theta))*torch.cos(theta/2)/2)

        # chain rule to the x1 derivative
        cosT, sinT = torch.cos(theta), torch.sin(theta)
        u1x = cosT*du1_dr - (sinT/r)*du1_dtheta
        u2x = cosT*du2_dr - (sinT/r)*du2_dtheta

        return torch.stack([u1x, u2x], -1)


    def aux_u_col1_II(self, r, theta, E, nu, plane_strain=True):
        """
        x1 derivatives u_{i,1}^a (i = 1, 2) of the unit-K_II auxiliary displacement.
        Same arguments and return shape as aux_u_col1_I.
        """
        mu = E / (2*(1+nu))
        if plane_strain:
            kappa = 3 - 4*nu
        else:
            kappa = (3 - nu) / (1 + nu)

        fac = 1.0 / (2*mu) * torch.sqrt(r/(2*np.pi))

        # auxiliary displacement (K_II = 1)
        u1 = fac * 0.5*((2*kappa+3)*torch.sin(theta/2)+torch.sin(theta*1.5))
        u2 = fac * 0.5*(-(2*kappa-3)*torch.cos(theta/2)-torch.cos(theta*1.5))
        # derivatives with respect to r and theta
        du1_dr = (1/(4*mu)) * (1/torch.sqrt(2*np.pi*r)) * 0.5*((2*kappa+3)*torch.sin(theta/2)+torch.sin(theta*1.5))
        du2_dr = (1/(4*mu)) * (1/torch.sqrt(2*np.pi*r)) * 0.5*(-(2*kappa-3)*torch.cos(theta/2)-torch.cos(theta*1.5))

        du1_dtheta = fac * 0.5 * ((2*kappa+3)*torch.cos(theta/2)*0.5 + (torch.cos(theta*1.5)*1.5))
        du2_dtheta = fac * 0.5 * ((2*kappa-3)*torch.sin(theta/2)*0.5 + (torch.sin(theta*1.5)*1.5))

        # chain rule to the x1 derivative
        cosT, sinT = torch.cos(theta), torch.sin(theta)
        u1x = cosT*du1_dr - (sinT/r)*du1_dtheta
        u2x = cosT*du2_dr - (sinT/r)*du2_dtheta

        return torch.stack([u1x, u2x], -1)



    def _compliance_eps_from_sigma(self, sxx, syy, sxy, E, nu, plane_strain: bool):
        # Strain from stress through the compliance (tensor shear exy)
        if plane_strain:
            # plane strain, eps = S_ps * sigma with E' = E/(1 - nu^2):
            # exx = sxx/E' - nu (1 + nu)/E * syy
            # eyy = syy/E' - nu (1 + nu)/E * sxx
            # gamma_xy = 2*exy = tau_xy / G, G = E/(2(1+nu))
            Eprime = E/(1-nu**2)
            exx = sxx/Eprime - (nu/Eprime)/(1-nu) * syy
            eyy = syy/Eprime - (nu/Eprime)/(1-nu) * sxx
            exy = (1.0/(2.0*E/(2*(1+nu)))) * sxy  # exy = tau/(2G), G = E/(2(1+nu))
        else:
            # plane stress
            exx = (1.0/E)*(sxx - nu*syy)
            eyy = (1.0/E)*(syy - nu*sxx)
            exy = (1.0/(2.0*E/(2*(1+nu)))) * sxy  # as above: exy = tau/(2G)

        return exx, eyy, exy

    def compute_K_via_interaction_integral(self,
                                        crack_tip_xy,      # (2,)
                                        radius: float,
                                        beta: float,
                                        E: float, nu: float,
                                        plane_strain: bool = True,
                                        num_points: int = 720,
                                        device=None):
        """
        K1, K2 from the contour term of the interaction integral.
        - Unit auxiliary fields, KI_aux = 1 and KII_aux = 1
        - Line integral over a circle of radius ``radius`` about the tip:
          dGamma = R dtheta
        - ``beta`` is the crack angle at the tip; the fields are rotated into
          the local tip frame, x1 along the crack
        Returns (KI, KII, M_I, M_II) as tensors.
        """
        if device is None:
            device = next(self.model.parameters()).device if hasattr(self.model, "parameters") else torch.device("cpu")
        crack_tip_xy = np.array(crack_tip_xy)
        crack_tip_xy = crack_tip_xy.astype(np.float32)
        # 1) circular contour (global frame)
        # The periodic trapezoidal rule needs num_points distinct angles:
        # linspace(0, 2pi, N) would count theta = 0 and theta = 2pi twice,
        # whereas dtheta = 2pi/N assumes N equally spaced samples
        theta = torch.arange(num_points, device=device, dtype=torch.float32) * (2.0 * np.pi / num_points)
        cx = torch.tensor(crack_tip_xy[0], device=device)
        cy = torch.tensor(crack_tip_xy[1], device=device)
        xg = cx + radius * torch.cos(theta)
        yg = cy + radius * torch.sin(theta)
        contour_xy = torch.stack([xg, yg], -1).requires_grad_(True)

        # outward normal (global frame)
        normals_g = torch.stack([torch.cos(theta), torch.sin(theta)], -1)  # outward normal of the circle

        # 2) actual fields of the model (global -> local)
        u, v = self.model.pred_uv(contour_xy)
        eXX, eYY, eXY = self.model.compute_Strain(u, v, contour_xy)  # eXY is engineering shear strain
        sx, sy, sxy = self.model.constitutive(eXX, eYY, eXY)

        # assemble the sigma and eps tensors (global frame)
        sig_g = torch.stack([torch.stack([sx,  sxy], -1),
                            torch.stack([sxy, sy ], -1)], -2)  # (N,2,2)
        eps_g = self._strain_vec_to_tensor(eXX, eYY, eXY/2)            # (N,2,2); engineering shear halved to tensor shear

        # rotate to the local frame
        sig_l = self._rotate_tensor(sig_g, beta, device)             # (N,2,2)
        eps_l = self._rotate_tensor(eps_g, beta, device)             # (N,2,2)
        n_l   = self._rotate_vector(normals_g, beta, device)         # (N,2)

        # 3) local coordinates relative to the tip, and polar coordinates
        # global points -> local points
        pts_local = self._rotate_vector(contour_xy - torch.tensor(crack_tip_xy, device=device), beta, device)
        x1 = pts_local[:,0]; x2 = pts_local[:,1]
        r  = torch.sqrt(torch.clamp(x1**2 + x2**2, min=1e-18))
        th = torch.atan2(x2, x1)

        # 4) unit auxiliary fields: stress (local frame) and the auxiliary
        # strain obtained from it through the compliance
        sxx_I, syy_I, sxy_I = self._aux_stress_local_unitKI(r, th)
        sxx_II, syy_II, sxy_II = self._aux_stress_local_unitKII(r, th)

        exx_I, eyy_I, exy_I = self._compliance_eps_from_sigma(sxx_I, syy_I, sxy_I, E, nu, plane_strain)
        exx_II, eyy_II, exy_II = self._compliance_eps_from_sigma(sxx_II, syy_II, sxy_II, E, nu, plane_strain)

        sig_aux_I = torch.stack([torch.stack([sxx_I,  sxy_I], -1),
                                torch.stack([sxy_I, syy_I ], -1)], -2)
        sig_aux_II = torch.stack([torch.stack([sxx_II,  sxy_II], -1),
                                torch.stack([sxy_II, syy_II ], -1)], -2)

        eps_aux_I = self._strain_vec_to_tensor(exx_I, eyy_I, exy_I)
        eps_aux_II = self._strain_vec_to_tensor(exx_II, eyy_II, exy_II)

        # 5) interaction strain energy density
        # W_mix = 0.5 (sigma : eps_aux + sigma_aux : eps)
        def double_contraction(A, B):
            return (A*B).sum(dim=(-1,-2))

        Wmix_I  = 0.5*(double_contraction(sig_l, eps_aux_I)  + double_contraction(sig_aux_I,  eps_l))
        Wmix_II = 0.5*(double_contraction(sig_l, eps_aux_II) + double_contraction(sig_aux_II, eps_l))

        # 6) tractions t = sigma n of the actual and auxiliary fields
        t_l       = sig_l @ n_l.unsqueeze(-1)         # (N,2,1)
        t_l_I_aux = sig_aux_I @ n_l.unsqueeze(-1)
        t_l_II_aux= sig_aux_II @ n_l.unsqueeze(-1)

        # 7) displacement gradient grad u, rotated to the local frame
        # F_g = [ [du1/dx, du1/dy],
        #         [du2/dx, du2/dy] ]
        du_dx = torch.autograd.grad(u.sum(), contour_xy, create_graph=True)[0]  # (N,2)
        dv_dx = torch.autograd.grad(v.sum(), contour_xy, create_graph=True)[0]  # (N,2)

        F_g = torch.stack([torch.stack([du_dx[:,0], du_dx[:,1]], -1),
                        torch.stack([dv_dx[:,0], dv_dx[:,1]], -1)], -2)  # (N,2,2)

        # rotate to the local frame
        F_l = self._rotate_tensor(F_g, beta, device)  # (N,2,2)

        # du_i/dx1 = F_l[:, i, 0]
        u1x = F_l[:,0,0]   # du1/dx1
        u2x = F_l[:,1,0]   # du2/dx1
        u_col1 = torch.stack([u1x, u2x], -1)  # (N,2)

        # the same column of the auxiliary displacement gradient
        u_col1_I  = self.aux_u_col1_I(r, th, E, nu, plane_strain)
        u_col1_II = self.aux_u_col1_II(r, th, E, nu, plane_strain)

        # 8) integrand of the interaction integral:
        # Wmix*n1 - t . u_aux_col1 - t_aux . u_col1
        n1 = n_l[:,0]
        integrand_I  = Wmix_I  * n1 - (t_l.squeeze(-1)*u_col1_I).sum(dim=-1)  - (t_l_I_aux.squeeze(-1)*u_col1).sum(dim=-1)
        integrand_II = Wmix_II * n1 - (t_l.squeeze(-1)*u_col1_II).sum(dim=-1) - (t_l_II_aux.squeeze(-1)*u_col1).sum(dim=-1)

        # 9) line integral over equally spaced angles: sum of the integrand
        # times dGamma, dGamma = R dtheta
        dtheta = 2*np.pi / num_points
        M_I  = torch.sum(integrand_I)  * radius * dtheta
        M_II = torch.sum(integrand_II) * radius * dtheta

        # 10) M -> K: K = E' M / 2, E' = E/(1 - nu^2) in plane strain, E in plane stress
        Eprime = E/(1-nu**2) if plane_strain else E
        KI  = Eprime * M_I/2
        KII = Eprime * M_II/2
        return KI, KII, M_I, M_II


def thermal_interaction_term(dT_dx1_fn, coeff, r_in, r_out, n_r=80, n_theta=360):
    """Thermal area term of the interaction integral (Wilson and Yu).

    ThermoSIFAdapter removes the thermal strain before the contour term is
    evaluated, so the contour term alone gives the mechanical part of the
    energy release rate.  The thermal part is the area term of Wilson and Yu,
    an area integral that has no contour form and is therefore evaluated
    separately here.

    For an isotropic thermal strain eps_th = coeff * (T - T_ref) the term is

        M_therm = INT_A  sigma^aux_kk * coeff * dT/dx1 * q  dA

    with the auxiliary stress taken from the same unit-K Williams fields the
    contour term uses.  Its in-plane trace is

        sigma^aux_kk = 2 cos(theta/2) / sqrt(2 pi r)      (unit K_I)
                     = -2 sin(theta/2) / sqrt(2 pi r)     (unit K_II)

    which is integrable against the r dr d(theta) measure despite the
    singularity, so the disc is swept from the tip outward with a midpoint
    rule; no core needs excluding.

    ``q`` is the usual weight: one inside ``r_in``, falling linearly to zero at
    ``r_out``; ``r_in=None`` makes it one over the whole disc.
    ``dT_dx1_fn(r, theta) -> array`` supplies the temperature gradient along
    the crack in the local tip frame, so that the quadrature does not depend
    on how the field was produced.
    SIFEstimator.thermal_domain_dK passes the gradient of the thermal strain
    itself, with ``coeff = 1``.

    Returns ``(M_therm_I, M_therm_II)``, on the same footing as the M values
    returned by the contour term.
    """
    r_out = float(r_out)
    # r_in None makes q one over the whole disc, the weight that corrects a
    # contour value at r_out; a ramp to zero at r_out is the weight of a
    # domain-form evaluation.
    if r_in is None:
        r_in = r_out
    r_in = float(r_in)
    if not (0.0 < r_in <= r_out):
        raise ValueError(f"need 0 < r_in <= r_out, got {r_in} and {r_out}")

    # Midpoint in both directions; theta spans the full disc, the crack faces
    # sitting at the two ends of the range.
    dr = r_out / n_r
    dth = 2.0 * np.pi / n_theta
    r = (np.arange(n_r) + 0.5) * dr
    th = -np.pi + (np.arange(n_theta) + 0.5) * dth
    R, TH = np.meshgrid(r, th, indexing="ij")

    if r_in >= r_out:
        q = np.ones_like(R)          # no ramp: q is one out to r_out
    else:
        q = np.clip((r_out - R) / (r_out - r_in), 0.0, 1.0)

    root = np.sqrt(2.0 * np.pi * R)
    trace_I = 2.0 * np.cos(TH / 2.0) / root
    trace_II = -2.0 * np.sin(TH / 2.0) / root

    grad = np.asarray(dT_dx1_fn(R, TH), dtype=np.float64).reshape(R.shape)
    common = float(coeff) * grad * q * R * dr * dth
    return float(np.sum(trace_I * common)), float(np.sum(trace_II * common))


class ThermoSIFAdapter:
    """Adapter exposing the interface expected by M_integral.

    compute_Strain subtracts the isotropic thermal strain, so the contour term
    of the interaction integral is built from elastic strains and
    thermoelastic stresses; the shear component is unaffected because the
    thermal strain is isotropic.  Graded material properties are taken from
    model.material_fields when present.
    """

    def __init__(self, model, T_delta_nd, v_delta_nd):
        self.model = model
        self.device = model.device
        self.T_delta_t = torch.tensor([[T_delta_nd]], dtype=torch.float32, device=model.device)
        self.v_delta_t = torch.tensor([[v_delta_nd]], dtype=torch.float32, device=model.device)
        self._xy = None

    def pred_uv(self, xy):
        self._xy = xy
        u, v = self.model.net_uv(xy[:, 0:1], xy[:, 1:2], self.v_delta_t)
        return u.squeeze(-1), v.squeeze(-1)

    def thermal_strain(self, xy):
        T = self.model.net_thermal(xy[:, 0:1], xy[:, 1:2], self.T_delta_t).squeeze(-1)
        if hasattr(self.model, "material_fields"):
            _, nu, _, _, _, _, _, alpha = self.model.material_fields(xy[:, 0:1])
            coeff = self.model.thermal_strain_coefficient(alpha=alpha.squeeze(-1), nu=nu.squeeze(-1))
            return coeff * (T - self.model.T_ref)
        return self.model.thermal_strain_coefficient() * (T - self.model.T_ref)

    def compute_Strain(self, u, v, xy):
        self._xy = xy
        du_dxy = torch.autograd.grad(u.sum(), xy, create_graph=True)[0]
        dv_dxy = torch.autograd.grad(v.sum(), xy, create_graph=True)[0]

        eXX_total = du_dxy[:, 0]
        eYY_total = dv_dxy[:, 1]
        eXY_eng = du_dxy[:, 1] + dv_dxy[:, 0]

        eps_th = self.thermal_strain(xy)
        return eXX_total - eps_th, eYY_total - eps_th, eXY_eng

    def constitutive(self, eXX, eYY, eXY):
        tr_e = eXX + eYY
        if hasattr(self.model, "material_fields") and self._xy is not None:
            _, _, lam, mu, _, _, _, _ = self.model.material_fields(self._xy[:, 0:1])
            lam = lam.squeeze(-1)
            mu = mu.squeeze(-1)
        else:
            lam = self.model.lamda
            mu = self.model.mu
        sx = lam * tr_e + 2.0 * mu * eXX
        sy = lam * tr_e + 2.0 * mu * eYY
        sxy = mu * eXY
        return sx, sy, sxy

    def parameters(self):
        return self.model.mechanical_parameters()


class SIFEstimator:
    """K/J extraction with guards, fallback, cross-check and validation plots.

    One instance per example run.  ``tip_boundary_distance(tip) -> float``
    supplies the domain geometry (rectangle, cruciform, ...); everything else
    is configuration set by the example script.
    """

    def __init__(
        self,
        model,
        *,
        tip_boundary_distance: Callable[[np.ndarray], float],
        interaction_radius: float,
        # Angular samples on the contour.  The integrand jumps across the crack
        # face, so the periodic trapezoidal rule converges only to first order in
        # this number, and the grid is fine.
        interaction_num_points: int = 2880,
        cod_r_min: float,
        cod_r_max: float,
        cod_num_points: int = 15,
        cod_offset: float = 5.0e-4,
        crack_increment: float = 0.0,
        limit_r_by_last_segment: bool = False,
        estimator: str = "interaction_with_cod_fallback",
        print_range_diagnostics: bool = True,
        j_label: str = "J",
        use_plateau_estimate: bool = True,
        plateau_radius_factors=(0.5, 0.75, 1.0, 1.5, 2.0),
        # Thermal area term of Wilson and Yu (see thermal_domain_dK), zero without a
        # thermal load and when the thermal strain does not vary along the crack.
        # The two numbers below are the radial and angular midpoint samples of its
        # quadrature.
        use_thermal_domain_term: bool = True,
        thermal_domain_n_r: int = 80,
        thermal_domain_n_theta: int = 360,
    ):
        self.model = model
        self.tip_boundary_distance = tip_boundary_distance
        self.interaction_radius = float(interaction_radius)
        self.interaction_num_points = int(interaction_num_points)
        self.cod_r_min = float(cod_r_min)
        self.cod_r_max = float(cod_r_max)
        self.cod_num_points = int(cod_num_points)
        self.cod_offset = float(cod_offset)
        self.crack_increment = float(crack_increment)
        self.limit_r_by_last_segment = bool(limit_r_by_last_segment)
        self.estimator = str(estimator)
        self.print_range_diagnostics = bool(print_range_diagnostics)
        self.j_label = str(j_label)
        # Read K off the whole radius sweep rather than off one chosen radius
        # (see by_plateau).  When switched off, K is taken at the single radius
        # from contour_radius and no admissibility verdict is produced.
        self.use_plateau_estimate = bool(use_plateau_estimate)
        self.plateau_radius_factors = tuple(float(f) for f in plateau_radius_factors)
        self.use_thermal_domain_term = bool(use_thermal_domain_term)
        self.thermal_domain_n_r = int(thermal_domain_n_r)
        self.thermal_domain_n_theta = int(thermal_domain_n_theta)
        # Verdict of the sweep behind the most recent estimate, which the examples
        # read before a crack increment; None until the first estimate and while
        # the plateau estimate is off, and never validated for the COD paths.
        self.last_plateau = None
        # Crack increments taken on a clipped sweep that did not pass: the running
        # total and the consecutive count (see _CLIPPED_SWEEP_GRACE).
        self.clipped_releases = 0
        self._clipped_used = 0
        # Cache for the radius sweep, which estimate() and save_validation() evaluate
        # on the same radii.  Keyed on the identity of the crack geometry object,
        # since set_crack_state installs a new CrackGeometry on every crack
        # increment; solver.py caches its crack-face samples the same way.
        self._sweep_cache = None

    # -- geometry ----------------------------------------------------------
    def geometry(self, points):
        return get_active_crack_geometry(self.model, points)

    def straight_run_length(self, points, angle_tol_deg=5.0):
        """Length of the straight stretch of crack path ending at the tip.

        Walks back from the tip while each segment keeps the direction of the
        last one to within ``angle_tol_deg``, and returns the accumulated
        length (inf when the path has no kink at all).
        """
        pts = np.asarray(points, dtype=np.float64)
        if len(pts) < 3:
            return np.inf
        seg = np.diff(pts, axis=0)
        lens = np.linalg.norm(seg, axis=1)
        keep = lens > 1.0e-12
        seg, lens = seg[keep], lens[keep]
        if len(seg) < 2:
            return np.inf
        tan = seg / lens[:, None]
        cos_tol = np.cos(np.deg2rad(angle_tol_deg))
        total = lens[-1]
        for j in range(len(seg) - 2, -1, -1):
            if float(np.dot(tan[j], tan[-1])) < cos_tol:
                return total
            total += lens[j]
        return np.inf

    def contour_radius(self, tip, points=None):
        """Interaction contour radius, clipped by the domain and by kinks.

        The Williams auxiliary field assumes a straight crack through the
        contour, so a contour that encloses a kink returns a meaningless K.
        The radius is therefore limited to the straight stretch ending at the
        tip whenever ``limit_r_by_last_segment`` is set.
        """
        dist = float(self.tip_boundary_distance(np.asarray(tip, dtype=np.float64)))
        radius = min(self.interaction_radius, 0.9 * dist)
        limit_reason = "tip near boundary"
        if self.limit_r_by_last_segment and points is not None:
            straight = self.straight_run_length(points)
            if np.isfinite(straight) and 0.9 * straight < radius:
                radius = 0.9 * straight
                limit_reason = "kink inside contour"
        if radius < 0.5 * self.interaction_radius:
            raise ValueError(
                f"interaction contour does not fit ({limit_reason}): "
                f"tip-boundary distance {dist:.3e}, usable radius {radius:.3e}, "
                f"requested {self.interaction_radius:.3e}"
            )
        if radius < self.interaction_radius:
            print(f"[SIF] interaction contour shrunk to r={radius:.3e} ({limit_reason})")
        return radius

    # -- interaction integral -----------------------------------------------
    def thermal_domain_dK(self, geom, T_delta_nd, radius, E_tip, nu_tip):
        """Thermal area term of Wilson and Yu, as an increment on the contour K.

        The contour term of the interaction integral is path independent only
        without a thermal eigenstrain.  With one, two contours differ by

            M(G2) - M(G1) = - INT_A sigma^aux_kk d(alpha dT)/dx1 dA

        so the value read on a contour of radius r differs from the near-tip
        one by the area integral inside it, which this returns.  Without it the contour value depends on the contour radius
        wherever the thermal strain varies along the crack.

        The weight is q = 1 over the whole disc, not the ramp a domain-form
        evaluation would use: the difference being corrected is the one
        between a vanishing contour and this one.

        The gradient taken is that of the thermal strain, not of the
        temperature, so a graded expansion coefficient is differentiated
        rather than treated as a constant.

        Returns ``(dK1, dK2)``, zero when the term is switched off or there is
        no thermal load.
        """
        if not self.use_thermal_domain_term or float(T_delta_nd) == 0.0:
            return 0.0, 0.0
        tip = np.asarray(geom.tip, dtype=np.float64)
        beta = float(geom.last_beta)
        c, s = float(np.cos(beta)), float(np.sin(beta))
        adapter = ThermoSIFAdapter(self.model, T_delta_nd, 0.0)
        device = self.model.device

        def d_eps_th_dx1(R, TH):
            x1, x2 = R * np.cos(TH), R * np.sin(TH)
            gx = tip[0] + c * x1 - s * x2
            gy = tip[1] + s * x1 + c * x2
            xy = torch.tensor(np.column_stack([gx.ravel(), gy.ravel()]),
                              dtype=torch.float32, device=device,
                              requires_grad=True)
            eps = adapter.thermal_strain(xy)
            g, = torch.autograd.grad(eps.sum(), xy)
            return (c * g[:, 0] + s * g[:, 1]).detach().cpu().numpy().reshape(R.shape)

        # The coefficient is already inside thermal_strain, so it is one here.
        mI, mII = thermal_interaction_term(
            d_eps_th_dx1, 1.0, None, float(radius),
            n_r=self.thermal_domain_n_r, n_theta=self.thermal_domain_n_theta)
        Eprime = E_tip / (1.0 - nu_tip ** 2) if plane_strain_flag(self.model) else E_tip
        return 0.5 * Eprime * mI, 0.5 * Eprime * mII

    def by_interaction(self, points, T_delta_nd, v_delta_nd, radius=None):
        geom = self.geometry(points)
        beta = float(geom.last_beta)
        tip = np.asarray(geom.tip, dtype=np.float64)
        E_tip, nu_tip, _ = local_material_at_tip(self.model, tip[0])
        rad = self.contour_radius(tip, points) if radius is None else float(radius)

        adapter = ThermoSIFAdapter(self.model, T_delta_nd, v_delta_nd)
        mtool = M_integral(adapter, self.model.device)
        K1_t, K2_t, _, _ = mtool.compute_K_via_interaction_integral(
            crack_tip_xy=tip,
            radius=rad,
            beta=beta,
            E=E_tip,
            nu=nu_tip,
            plane_strain=plane_strain_flag(self.model),
            num_points=self.interaction_num_points,
            device=self.model.device,
        )
        K1 = float(K1_t.detach().cpu().numpy())
        K2 = float(K2_t.detach().cpu().numpy())
        if not np.isfinite(K1) or not np.isfinite(K2):
            raise ValueError("non-finite K from interaction integral")
        # The area term is added here rather than in the sweep, so the plateau
        # estimate, the validation figure, the driving force and the COD cross-check
        # all use the corrected value.
        dK1, dK2 = self.thermal_domain_dK(geom, T_delta_nd, rad, E_tip, nu_tip)
        K1 += dK1
        K2 += dK2
        J = float(J_from_K(K1, K2, E_tip, nu_tip, plane_strain_flag(self.model)))
        return K1, K2, J, beta

    # -- COD ---------------------------------------------------------------
    def cod_sampling_range(self, points):
        """Effective COD sampling range behind the active crack tip."""
        geom = self.geometry(points)
        total_len = max(total_crack_length(points, model=self.model), 1.0e-12)
        last_len = float(getattr(geom, "last_segment_length", total_len))

        r_candidates = [self.cod_r_max, 0.80 * total_len]
        if self.limit_r_by_last_segment:
            r_candidates.append(0.75 * max(last_len, self.cod_r_min * 2.0))
        r_max_eff = min(r_candidates)

        r_min_eff = self.cod_r_min
        if self.crack_increment > 0.0:
            r_min_eff = max(r_min_eff, 0.5 * self.crack_increment)
        if r_max_eff <= r_min_eff * 1.5:
            r_max_eff = 1.5 * r_min_eff
        return r_min_eff, r_max_eff, total_len, last_len, r_candidates

    def print_cod_diagnostics(self, points, label="COD"):
        if not self.print_range_diagnostics:
            return
        r_min_eff, r_max_eff, total_len, last_len, _ = self.cod_sampling_range(points)
        n = self.cod_num_points
        dr = (r_max_eff - r_min_eff) / max(n - 1, 1)
        da = self.crack_increment

        warnings = []
        if n < 7:
            warnings.append("n<7")
        if r_min_eff <= 0.0 or r_max_eff <= r_min_eff:
            warnings.append("bad-range")
        if r_max_eff > 0.80 * total_len + 1.0e-12:
            warnings.append("r_max>0.8L")
        if self.limit_r_by_last_segment and r_max_eff > 0.75 * max(last_len, self.cod_r_min * 2.0) + 1.0e-12:
            warnings.append("narrow/local-window")
        if da > 0.0 and r_min_eff < 0.5 * da:
            warnings.append("r_min<0.5da")
        if da > 0.0 and r_max_eff > 3.0 * da:
            warnings.append("r_max>3da")
        if da > 0.0 and (r_max_eff - r_min_eff) < 0.75 * da:
            warnings.append("window<0.75da")

        status = "OK" if not warnings else "check " + ",".join(warnings)
        print(
            f"[SIF-range {label}] COD requested r=[{self.cod_r_min:.3e},{self.cod_r_max:.3e}], "
            f"effective r=[{r_min_eff:.3e},{r_max_eff:.3e}], n={n}, dr={dr:.3e}, "
            f"crack_len={total_len:.3e}, last_seg={last_len:.3e}, "
            f"r/crack_increment=[{r_min_eff / max(da, 1.0e-30):.2f},"
            f"{r_max_eff / max(da, 1.0e-30):.2f}], status={status}"
        )

    def by_cod(self, points, T_delta_nd, v_delta_nd, print_diagnostics=True):
        """Two-sided face-sampling displacement extrapolation (fallback)."""
        geom = self.geometry(points)
        beta = float(geom.last_beta)

        r_min, r_max, _, _, _ = self.cod_sampling_range(points)
        if print_diagnostics:
            self.print_cod_diagnostics(points, label="COD")
        r_vals = np.linspace(r_min, r_max, self.cod_num_points)

        E_tip, nu_tip, mu_tip = local_material_at_tip(self.model, geom.tip[0])
        K1, K2, K1_samples, K2_samples, attenuation = cod_face_jump_sif(
            self.model, geom, v_delta_nd, r_vals, self.cod_offset,
            E_tip, nu_tip, mu_tip, plane_strain=plane_strain_flag(self.model),
        )
        J = float(J_from_K(K1, K2, E_tip, nu_tip, plane_strain_flag(self.model)))

        if print_diagnostics:
            att_txt = "n/a" if attenuation is None else f"{attenuation:.3f}"
            print(
                f"[SIF-COD] method=face-offset, offset={self.cod_offset:.1e}, "
                f"embed_attenuation={att_txt}, r=[{r_vals[0]:.3e},{r_vals[-1]:.3e}], "
                f"K1_sample=[{np.min(K1_samples):.3e},{np.max(K1_samples):.3e}], "
                f"K2_sample=[{np.min(K2_samples):.3e},{np.max(K2_samples):.3e}]"
            )
        return K1, K2, J, beta

    # -- combined entry point ------------------------------------------------
    def _crosscheck(self, points, T_delta_nd, v_delta_nd, K1_int, K2_int):
        """Diagnostic-only COD cross-check of the interaction K; never raises."""
        try:
            K1_cod, K2_cod, _, _ = self.by_cod(points, T_delta_nd, v_delta_nd, print_diagnostics=False)
        except Exception as exc:
            print(f"[SIF-xcheck] COD cross-check unavailable: {exc}")
            return
        scale = max(abs(K1_int), abs(K2_int), 1.0e-30)
        dev = max(abs(K1_cod - K1_int), abs(K2_cod - K2_int)) / scale
        status = "OK" if dev < 0.25 else "MISMATCH (check contour radius / near-tip resolution)"
        print(
            f"[SIF-xcheck] interaction K=({K1_int:.4e},{K2_int:.4e}) "
            f"vs COD K=({K1_cod:.4e},{K2_cod:.4e}), rel_dev={dev:.2f} -> {status}"
        )

    # -- radius sweep, shared by the estimate and the validation figure -------
    def radius_sweep(self, points, T_delta_nd, v_delta_nd):
        """K over a range of contour radii, dropping the ones that do not fit.

        A radius is dropped when it would leave the domain or enclose the last
        kink, since the Williams auxiliary field assumes a straight crack through
        the contour.  The sweep is then narrower than requested, and the verdict
        takes its width into account.

        Returns ``(radii, K1_list, K2_list)`` for the retained radii, each K
        including the thermal area term.
        """
        geom = self.geometry(points)
        # Same crack, load and radii give the same sweep.  The geometry is compared
        # by identity, since a new one is installed on every crack increment.
        cache = self._sweep_cache
        if cache is not None:
            c_geom, c_T, c_v, c_result = cache
            if c_geom is geom and c_T == float(T_delta_nd) and c_v == float(v_delta_nd):
                # Fresh lists, as a tuple of three like the computed path
                # returns, so the cached and computed results unpack alike.
                return tuple(list(x) for x in c_result)

        dist = float(self.tip_boundary_distance(np.asarray(geom.tip, dtype=np.float64)))
        r_kink = np.inf
        if self.limit_r_by_last_segment:
            straight = self.straight_run_length(points)
            if np.isfinite(straight):
                r_kink = 0.9 * straight

        radii, K1_list, K2_list = [], [], []
        for factor in self.plateau_radius_factors:
            rad = factor * self.interaction_radius
            if rad > 0.9 * dist or rad > r_kink:
                continue
            try:
                K1_r, K2_r, _, _ = self.by_interaction(points, T_delta_nd, v_delta_nd, radius=rad)
            except Exception:
                continue
            radii.append(rad)
            K1_list.append(K1_r)
            K2_list.append(K2_r)

        self._sweep_cache = (geom, float(T_delta_nd), float(v_delta_nd),
                             (tuple(radii), tuple(K1_list), tuple(K2_list)))
        return radii, K1_list, K2_list

    @staticmethod
    def sweep_verdict(radii, K1_list, K2_list):
        """Judge a radius sweep on its scaling exponent rather than on the raw drift.

        For K ~ r^p the drift is 1 - (r_max/r_min)^(-p) and depends on the width
        of the sweep, so it is converted to the exponent p over the span actually
        swept.  The extraction is admissible when p < _PATH_DEPENDENT_P.  Returns a
        dict; ``validated`` is the flag to act on, and ``clipped`` marks a sweep
        narrower than _CLIPPED_SPAN.
        """
        out = {"n": len(radii), "radii": list(radii), "K1": list(K1_list), "K2": list(K2_list),
               "drift": float("nan"), "span": float("nan"), "p": float("nan"),
               "validated": False, "clipped": False, "status": ""}
        if len(radii) < 2:
            out["status"] = "NOT VALIDATED (fewer than two radii fit)"
            out["clipped"] = True
            return out

        scale = max(max(abs(k) for k in K1_list), max(abs(k) for k in K2_list), 1.0e-30)
        out["drift"] = drift = float(max(np.ptp(K1_list), np.ptp(K2_list)) / scale)
        out["span"] = span = float(max(radii) / max(min(radii), 1.0e-30))
        # Whether the contour guard cut the sweep short of the requested radii.
        out["clipped"] = bool(span < _CLIPPED_SPAN)

        if span <= 1.0:
            # Every retained radius was the same, so nothing was swept.
            out["status"] = "NOT VALIDATED (sweep collapsed to one radius)"
        elif drift >= 1.0:
            # K changed sign across the sweep, so no exponent exists.
            out["status"] = "PATH-DEPENDENT (K changes sign across the sweep)"
        else:
            out["p"] = p = float(-np.log(1.0 - drift) / np.log(span))
            out["validated"] = bool(p < _PATH_DEPENDENT_P)
            out["status"] = "OK" if out["validated"] else "PATH-DEPENDENT (do not trust K this step)"
        return out

    def by_plateau(self, points, T_delta_nd, v_delta_nd):
        """Interaction K read off the whole sweep instead of one chosen radius.

        The median over the sweep does not depend on the choice of a single radius,
        and the growth criterion uses the same quantity that the admissibility
        check and the validation plots judge.  The median is component-wise; for a
        monotone K(r) both components take it at the same radius, so the mode
        mixity K2/K1 is preserved.

        Sets ``self.last_plateau`` for the caller to act on and returns the same
        tuple as the other estimators.
        """
        radii, K1_list, K2_list = self.radius_sweep(points, T_delta_nd, v_delta_nd)
        verdict = self.sweep_verdict(radii, K1_list, K2_list)
        self.last_plateau = verdict
        if not radii:
            raise ValueError("no contour radius in the sweep fits at this tip")

        geom = self.geometry(points)
        beta = float(geom.last_beta)
        K1 = float(np.median(K1_list))
        K2 = float(np.median(K2_list))
        E_tip, nu_tip, _ = local_material_at_tip(self.model, np.asarray(geom.tip, dtype=np.float64)[0])
        J = float(J_from_K(K1, K2, E_tip, nu_tip, plane_strain_flag(self.model)))
        return K1, K2, J, beta

    _COD_NOT_SWEPT = {"n": 0, "radii": [], "K1": [], "K2": [], "drift": float("nan"),
                      "span": float("nan"), "p": float("nan"), "validated": False,
                      "status": "NOT VALIDATED (crack-opening estimate, no sweep)"}

    def estimate(self, points, T_delta_nd, v_delta_nd):
        """Interaction integral first, COD fallback second.

        With ``estimator == "cod"`` only the COD estimate is used.  Sets
        ``self.last_plateau`` and returns ``(K1, K2, J, beta)``.
        """
        if self.estimator == "cod":
            self.last_plateau = dict(self._COD_NOT_SWEPT)
            K1, K2, J, beta = self.by_cod(points, T_delta_nd, v_delta_nd, print_diagnostics=True)
            print(f"[SIF] method=COD, K1={K1:.4e}, K2={K2:.4e}, {self.j_label}={J:.4e}")
            return K1, K2, J, beta

        self.print_cod_diagnostics(points, label="precheck")
        try:
            if self.use_plateau_estimate:
                K1, K2, J, beta = self.by_plateau(points, T_delta_nd, v_delta_nd)
                v = self.last_plateau
                print(f"[SIF] method=interaction-plateau, n={v['n']}, span={v['span']:.2f}x, "
                      f"p={v['p']:.3f} -> {v['status']}, "
                      f"K1={K1:.4e}, K2={K2:.4e}, {self.j_label}={J:.4e}")
            else:
                self.last_plateau = None
                K1, K2, J, beta = self.by_interaction(points, T_delta_nd, v_delta_nd)
                print(f"[SIF] method=interaction, K1={K1:.4e}, K2={K2:.4e}, {self.j_label}={J:.4e}")
            self._crosscheck(points, T_delta_nd, v_delta_nd, K1, K2)
            return K1, K2, J, beta
        except Exception as exc:
            print(f"[SIF] interaction failed, using COD fallback. Reason: {exc}")
            self.last_plateau = dict(self._COD_NOT_SWEPT)
            K1, K2, J, beta = self.by_cod(points, T_delta_nd, v_delta_nd, print_diagnostics=True)
            print(f"[SIF] method=COD-fallback, K1={K1:.4e}, K2={K2:.4e}, {self.j_label}={J:.4e}")
            return K1, K2, J, beta

    def extraction_is_validated(self):
        """Whether the sweep behind the last estimate passed the admissibility check.

        Separate from ``extraction_is_plausible``, which catches a field that has
        not re-converged after a crack increment; this check rejects a K that
        scales with the contour radius.  Returns ``(ok, reason)``; ``reason`` is
        empty when the value is accepted, and with the plateau estimate off
        everything is accepted.

        A clipped sweep that fails is let through up to ``_CLIPPED_SWEEP_GRACE``
        consecutive times, with a non-empty reason so that the caller can report
        it, and counted in ``clipped_releases``.
        """
        v = self.last_plateau
        if v is None:
            return True, ""
        if v["validated"]:
            # Only a full-width sweep that passes resets the consecutive count.
            if not v.get("clipped", False):
                self._clipped_used = 0
            return True, ""

        detail = (f"{v['status']} (n={v['n']}, span={v['span']:.2f}x, p={v['p']:.3f})")
        if v.get("clipped", False) and self._clipped_used < _CLIPPED_SWEEP_GRACE:
            self._clipped_used += 1
            self.clipped_releases += 1
            return True, (f"K not validated but the sweep was clipped to "
                          f"{v['span']:.2f}x; releasing {self._clipped_used} of "
                          f"{_CLIPPED_SWEEP_GRACE} so the straight run can recover: {detail}")
        return False, f"radius sweep did not validate K: {detail}"

    # -- per-step validation figures ------------------------------------------
    def save_validation(self, step_dir, step, points, T_delta_nd, v_delta_nd):
        """Radius-sweep path-independence and COD extrapolation plots.

        Flat sweep curves and a clean linear COD extrapolation are the
        acceptance checks for the extracted K.  Never raises.
        """
        try:
            geom = self.geometry(points)
            E_tip, nu_tip, mu_tip = local_material_at_tip(self.model, geom.tip[0])

            r_min, r_max, _, _, _ = self.cod_sampling_range(points)
            r_vals = np.linspace(r_min, r_max, self.cod_num_points)
            K1_cod, K2_cod, K1_s, K2_s, _ = cod_face_jump_sif(
                self.model, geom, v_delta_nd, r_vals, self.cod_offset,
                E_tip, nu_tip, mu_tip, plane_strain=plane_strain_flag(self.model),
            )
            plot_cod_extrapolation(step_dir, f"step_{step}", r_vals, K1_s, K2_s, K1_cod, K2_cod)

            # The sweep and its verdict come from the same functions as the estimate.
            radii, K1_list, K2_list = self.radius_sweep(points, T_delta_nd, v_delta_nd)
            if len(radii) >= 2:
                plot_sif_radius_sweep(
                    step_dir, f"step_{step}", radii, K1_list, K2_list,
                    K1_ref=K1_cod, K2_ref=K2_cod,
                )
                v = self.sweep_verdict(radii, K1_list, K2_list)
                p_str = "n/a" if not np.isfinite(v["p"]) else f"{v['p']:.3f}"
                print(f"[SIF-validate step {step}] radius-sweep drift={v['drift']:.2f} "
                      f"over r={radii} (span {v['span']:.2f}x) -> p={p_str} -> {v['status']}")
        except Exception as exc:
            print(f"[SIF-validate step {step}] skipped: {exc}")
