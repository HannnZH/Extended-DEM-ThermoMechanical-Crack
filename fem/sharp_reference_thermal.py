r"""The sharp-crack reference carried through the thermal load cases.

Extends sharp_reference.py from the isothermal problem to the three thermal
cases of the graded shear test, DT = -30, 0 and +30 K, by adding the transient
conduction the specimen runs under and the thermal strain it drives.  The crack
stands at its initial length throughout, which is all that is needed: the
specimen is elastic up to initiation, so a stationary solution fixes the whole
loading ramp and the load at which the criterion is first met.

Why the conduction has to be transient.  The boundary temperature reaches its
full value after T_RAMP_DISP / DU = 12 load steps, but the tip temperature
keeps rising for many load steps after that, so a steady solve would put the
tip temperature, and with it Gc(T_tip), in the wrong place.

Two properties of this benchmark are matched here:

  the adiabatic faces.  All three solutions intend them.  The phase field
  degrades the conductivity by (1-p)^2 + 1e-6, the proposed method imposes them
  weakly with a small flux consistency term, and a seam carries them exactly
  as the natural boundary condition.

  the specific heat.  The test of Tangella et al. carries Cp = 0.775 J/(kg K),
  and so do the phase-field reference and the proposed method.  The
  diffusivity is k/(rho Cp) = 1.58e5 mm^2/s at x = 0, as in pf_fem.py.

The driving force is the derivative of the potential energy in crack length,
as in the isothermal case, with the temperature field held fixed at the value
the current crack length produced.  Holding it fixed is what the interaction
integral with the thermal area term of Wilson and Yu also does, and it makes
the difference one-sided: the field of the shorter crack is single valued
across the segment about to be opened, so transferring it to the longer crack
is unambiguous, whereas going the other way is not.

    python fem/sharp_reference_thermal.py

Needs scikit-fem, and sharp_reference.py in the same directory.

Author: Han Zhang (han.zhang7@unsw.edu.au)
"""
import numpy as np
from skfem import (Basis, ElementVector, ElementTriP1, ElementTriP2,
                   BilinearForm, LinearForm, asm, condense, solve)
from skfem.helpers import dot, grad, sym_grad, trace
from scipy.sparse.linalg import splu

import sharp_reference as S
from sharp_reference import L, A0, E1, E2, NU, GC1, GC2, slit_mesh, stiffness

# ------------------------------------------------- thermal material, mm-N-s-K
KC1, KC2 = 300.0, 350.0          # mJ/(s mm K), conductivity at x = 0 and L
RHO1, RHO2 = 2.45e-6, 4.00e-6    # kg/mm^3
CP = 775.0                       # mJ/(kg K), i.e. 0.775 J/(kg K)
DT = 2.5e-8                      # s, the load step's time increment
T_REF = 300.0                    # K
T_RAMP_DISP = 3.0e-4             # mm of applied displacement to reach full dT
ALPHA = 8.0e-6                   # 1/K
DU = 2.5e-5                      # mm, applied displacement per load step
NSTEP = 141
INTORDER = 4                     # shared quadrature order for the two bases

# Plane strain effective coefficient, as the solver's
# plane_strain_thermal_alpha_correction sets it.
ALPHA_EFF = (1.0 + NU) * ALPHA

# Temperature dependent toughness, as examples/sen_shear_fgm.py sets it:
# Gc(x) * (1 - b1 theta + b2 theta^2), theta = (T - 300) / 1000, clamped.
GC_TMAX = 1000.0
GC_B1, GC_B2 = 1.80, 1.10
GC_MIN = 0.03


def gc_at(x, T):
    gc_ref = GC1 + (x / L) * (GC2 - GC1)
    th = (T - T_REF) / GC_TMAX
    return gc_ref * max(1.0 - GC_B1 * th + GC_B2 * th ** 2, GC_MIN)


def mts_angle(k1, k2):
    """Maximum hoop stress kink angle."""
    if abs(k2) < 1.0e-14:
        return 0.0
    s = 1.0 if k2 > 0 else -1.0
    return 2.0 * np.arctan((k1 - s * np.sqrt(k1 ** 2 + 8.0 * k2 ** 2))
                           / (4.0 * k2))


def kinked(k1, k2, th):
    """Factors of an infinitesimal kink at angle ``th`` (Cotterell and Rice).

    Their squared sum over E' is the energy release rate of the kink
    G(theta_c) that the criterion of the proposed method uses.
    """
    c1, c3 = np.cos(0.5 * th), np.cos(1.5 * th)
    s1, s3 = np.sin(0.5 * th), np.sin(1.5 * th)
    kI = (3.0 * c1 + c3) / 4.0 * k1 - 3.0 * (s1 + s3) / 4.0 * k2
    kII = (s1 + s3) / 4.0 * k1 + (c1 + 3.0 * c3) / 4.0 * k2
    return kI, kII


@BilinearForm
def th_stiff(u, v, w):
    k = KC1 + (w.x[0] / L) * (KC2 - KC1)
    return k * dot(grad(u), grad(v))


@BilinearForm
def th_mass(u, v, w):
    rho = RHO1 + (w.x[0] / L) * (RHO2 - RHO1)
    return rho * CP * u * v


@LinearForm
def th_load(v, w):
    """Work of the thermal stress, sigma_th : eps(v) with sigma_th = 2(lam+mu) beta I."""
    E = E1 + (w.x[0] / L) * (E2 - E1)
    lam = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
    mu = E / (2.0 * (1.0 + NU))
    beta = ALPHA_EFF * (w["T"] - T_REF)
    return 2.0 * (lam + mu) * beta * trace(sym_grad(v))


class Case(object):
    """One crack length: the meshes, the operators and the marching state."""

    def __init__(self, n, a, elem_u, elem_t):
        self.m, self.pairs = slit_mesh(n, a)
        self.a = a
        # The thermal load form integrates a P1 temperature against P2 test
        # strains, so the two bases have to be quadrature compatible; skfem
        # picks the order per element by default and they would not match.
        self.bu = Basis(self.m, ElementVector(elem_u), intorder=INTORDER)
        self.bti = Basis(self.m, elem_t, intorder=INTORDER)
        self.bt = Basis(self.m, elem_t)
        # The temperature is carried on the mesh nodes so that transfer()
        # between two crack lengths is a lookup by position.  A quadratic
        # temperature would add edge degrees of freedom that have no such
        # correspondence.  Linear also matches the CG1 temperature of the
        # implementation of Tangella et al.
        assert self.bt.N == self.m.p.shape[1], \
            "the temperature element must be nodal (P1)"
        self.K = asm(stiffness, self.bu)
        self.A = asm(th_stiff, self.bt)
        self.M = asm(th_mass, self.bt)
        self.bot_u = self.bu.get_dofs(lambda x: np.isclose(x[1], 0.0))
        self.top_u = self.bu.get_dofs(lambda x: np.isclose(x[1], L))
        self.top_ux = self.top_u.all("u^1")
        self.Du = np.concatenate([self.bot_u.all(), self.top_u.all()])
        self.bot_t = self.bt.get_dofs(lambda x: np.isclose(x[1], 0.0)).all()
        self.top_t = self.bt.get_dofs(lambda x: np.isclose(x[1], L)).all()
        self.T = np.full(self.bt.N, T_REF)
        self._lu_u = None
        self._lu_t = None

    def thermal_step(self, dT_top):
        """One backward Euler step of the conduction problem.

        The crack does not move, so both operators are constant over the whole
        load history and are factorized once.
        """
        rhs = (self.M / DT) @ self.T
        x = np.zeros(self.bt.N)
        x[self.bot_t] = T_REF
        x[self.top_t] = T_REF + dT_top
        D = np.concatenate([self.bot_t, self.top_t])
        Ac, bc, xc, I = condense(self.M / DT + self.A, rhs, x=x, D=D)
        if self._lu_t is None:
            self._lu_t = splu(Ac.tocsc())
        T = xc.copy()
        T[I] = self._lu_t.solve(bc)
        self.T = T
        return self.T

    def mechanics(self, delta, T):
        """Solve at applied displacement ``delta`` with temperature ``T``.

        Returns the reaction force and the potential energy.
        """
        f = asm(th_load, self.bu, T=self.bti.interpolate(T))
        x = self.bu.zeros()
        x[self.top_ux] = delta
        Kc, fc, xc, I = condense(self.K, f, x=x, D=self.Du)
        if self._lu_u is None:
            self._lu_u = splu(Kc.tocsc())
        u = xc.copy()
        u[I] = self._lu_u.solve(fc)
        F = float((self.K @ u - f)[self.top_ux].sum())
        pi = 0.5 * float(u @ (self.K @ u)) - float(f @ u)
        return F, pi, u

    def mixity(self, u, rmin=2, rmax=12):
        """K_I / K_II from the ratio of the two crack opening components.

        The factors themselves are not taken from the opening: K(r) does not
        reach a plateau and the extrapolated value depends on the fitting
        window.  Their ratio is stable, because both components drift the
        same way with the window and the drift divides out.  So the magnitude
        is taken from the energy release rate and only the split from here,
        each component fitted against sqrt(r) between ``rmin`` and ``rmax``
        element sizes behind the tip.
        """
        ux = u[self.bu.nodal_dofs[0]]
        uy = u[self.bu.nodal_dofs[1]]
        r = self.a - self.m.p[0, self.pairs[:, 0]]
        d_t = ux[self.pairs[:, 1]] - ux[self.pairs[:, 0]]
        d_n = uy[self.pairs[:, 1]] - uy[self.pairs[:, 0]]
        g = r > 0
        r, d_t, d_n = r[g], d_t[g], d_n[g]
        o = np.argsort(r)
        r, d_t, d_n = r[o], d_t[o], d_n[o]
        h = np.min(np.diff(r)) if len(r) > 1 else r[0]
        sel = (r >= rmin * h) & (r <= rmax * h)
        an = np.polyfit(r[sel], d_n[sel] / np.sqrt(r[sel]), 1)[1]
        at = np.polyfit(r[sel], d_t[sel] / np.sqrt(r[sel]), 1)[1]
        return an / at

    def tip_temperature(self, T):
        """Temperature at the tip node, which is the single unsplit one."""
        p = self.m.p
        i = np.argmin((p[0] - self.a) ** 2 + (p[1] - S.YC) ** 2)
        return float(T[i])


def transfer(short, long_, T):
    """Carry the temperature of the shorter crack onto the longer mesh.

    Every node of the longer mesh sits at a position the shorter mesh also
    has.  Across the segment that the longer crack opens the shorter field is
    single valued, so both new faces take that one value; elsewhere the split
    already existed and the two copies keep their own.
    """
    ps, pl = short.m.p, long_.m.p
    key = {}
    for j in range(ps.shape[1]):
        key.setdefault((round(ps[0, j], 12), round(ps[1, j], 12)), []).append(j)
    out = np.empty(pl.shape[1])
    seen = {}
    for j in range(pl.shape[1]):
        k = (round(pl[0, j], 12), round(pl[1, j], 12))
        cand = key[k]
        if len(cand) == 1:
            out[j] = T[cand[0]]
        else:
            # Both meshes split here; take the copies in the same order.
            i = seen.get(k, 0)
            out[j] = T[cand[min(i, len(cand) - 1)]]
            seen[k] = i + 1
    return out


def march(n, dT_max, da, elem_u=ElementTriP2(), elem_t=ElementTriP1(),
          nstep=NSTEP, verbose=True):
    """March the load history and return the ramp and the initiation point."""
    # The difference has to be one-sided (see the module docstring), so it is
    # first order in da.  Two extensions, da and da/2, combined by Richardson
    # extrapolation remove the first-order error.
    base = Case(n, A0, elem_u, elem_t)
    ext = Case(n, A0 + da, elem_u, elem_t)
    ext_h = Case(n, A0 + 0.5 * da, elem_u, elem_t)

    ep_tip = S.EP_TIP
    rows = []
    init = None
    for s in range(1, nstep + 1):
        delta = s * DU
        dT = dT_max * min(delta / T_RAMP_DISP, 1.0)
        T = base.thermal_step(dT).copy()
        # The extended meshes are marched on their own so that their states
        # match; only the frozen field of the base mesh enters the difference.
        ext.thermal_step(dT)
        ext_h.thermal_step(dT)

        F, pi, u = base.mechanics(delta, T)
        _, pi_1, _ = ext.mechanics(delta, transfer(base, ext, T))
        _, pi_2, _ = ext_h.mechanics(delta, transfer(base, ext_h, T))
        g1 = -(pi_1 - pi) / da
        g2 = -(pi_2 - pi) / (0.5 * da)
        G = 2.0 * g2 - g1

        # Magnitude from the energy release rate, split from the opening, then
        # the energy release rate of the kink.  The thermal load adds a mode I
        # component at the tip, so the pure mode II value G(theta_c) = 4/3 G
        # does not apply and is not used.
        kmag = np.sqrt(max(ep_tip * G, 0.0))
        mx = base.mixity(u)
        k2 = kmag / np.sqrt(1.0 + mx ** 2)
        k1 = mx * k2
        th_c = mts_angle(k1, k2)
        kI, kII = kinked(k1, k2, th_c)
        g_kink = (kI ** 2 + kII ** 2) / ep_tip

        T_tip = base.tip_temperature(T)
        gc = gc_at(A0, T_tip)
        ratio = g_kink / gc
        rows.append((s, delta, dT, F, T_tip, g_kink, gc, ratio, k1, k2,
                     np.degrees(th_c)))
        if init is None and ratio >= 1.0:
            init = rows[-1]
        if verbose and (s % 10 == 0 or s <= 2):
            print("  step %3d  d=%.3e  dT=%6.2f  F=%8.3f  T_tip=%8.4f  "
                  "G=%.4e  G(th)/Gc=%.4f" % (s, delta, dT, F, T_tip, G, ratio),
                  flush=True)
        if init is not None and s > init[0] + 2:
            break
    return np.array(rows), init


def main():
    print("sharp-crack SEN shear reference, transient thermal, plane strain")
    print("diffusivity k/(rho Cp) = %.3e mm^2/s   dt = %.1e s\n"
          % (KC1 / (RHO1 * CP), DT))

    n, da = 80, 0.025
    print("check: the isothermal case through the thermal code path")
    rows, init = march(n, 0.0, da, nstep=45, verbose=False)
    s5 = rows[np.argmin(np.abs(rows[:, 1] - 1.0e-3))]
    print("  at delta = 1e-3 mm:  G(theta_c) = %.6e N/mm   k = %.1f N/mm"
          % (s5[5], s5[3] / s5[1]))
    print("  compare k, and G(theta_c) = 4/3 G in pure mode II, with "
          "sharp_reference.py")
    print("  T stays at %.4f K\n" % rows[-1, 4])

    for dT in (-30.0, 0.0, 30.0):
        print("dT = %+.0f K" % dT)
        rows, init = march(n, dT, da)
        if init is None:
            print("  no initiation within %d steps\n" % NSTEP)
            continue
        s, delta, dTn, F, T_tip, G, gc, ratio = init[:8]
        pk = rows[rows[:, 0] < s]
        print("  initiation at step %d, delta = %.4e mm, F = %.2f N" %
              (s, delta, F))
        print("  T_tip = %.4f K, Gc = %.5f N/mm, peak on the ramp = %.2f N\n"
              % (T_tip, gc, pk[-1, 3] if len(pk) else F))


if __name__ == "__main__":
    main()
