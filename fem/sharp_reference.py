r"""Sharp-crack finite element reference for the graded shear specimen.

This script solves the boundary value problem of the graded shear test by
standard finite elements on a mesh with the crack cut into it as a seam.  No
phase field, no damage, no propagation, no criterion.  It gives

    stiffness  -- the elastic stiffness of the specimen, and
    factors    -- K_I and K_II at a given applied displacement.

Both are needed only at the initial crack length, because the peak load of a
Griffith crack is decided before the crack has moved: the specimen is elastic
up to initiation, so a stationary solution fixes the whole ramp and the load at
which the criterion is first met.

The seam is made by duplicating the nodes on the notch line and giving the
elements above the notch the duplicates, so the two faces share no degrees of
freedom.  The node at the tip stays single, which is what makes the cut a crack
and not a full separation.

The driving force is taken from the compliance rather than from any near-tip
formula.  Under displacement control the potential energy is k(a) delta^2 / 2,
so differencing the stiffness in crack length gives G directly: no asymptotic
field, no auxiliary field, no fitting window, and none of the correction the
interaction integral needs in a graded material, where the usual auxiliary
fields do not satisfy equilibrium.  main() prints its convergence under mesh
refinement, with a Richardson extrapolation, and its dependence on the
differencing step.

The factors read off the crack face opening are computed below as a
cross-check only: on this problem K(r) built from the opening does not reach a
plateau, so the value extrapolated to the tip depends on the choice of fitting
window.

The compliance route gives the magnitude and not the mode split, which is
enough here because the isothermal specimen is loaded in nearly pure mode II.
Under thermal load sharp_reference_thermal.py takes the split from the ratio of
the normal to the tangential face opening.  A kinked crack would need the
interaction integral with the non-equilibrium term of Kim and Paulino.

Geometry, material and boundary conditions follow examples/sen_shear_fgm.py
and the graded shear test of Tangella et al. (2022): a unit square in mm-N, an
edge notch to the centre, the lower edge clamped, the upper edge given a
horizontal displacement with its vertical component held.

    python fem/sharp_reference.py

Needs scikit-fem; it does not use the neural network solver.

Author: Han Zhang (han.zhang7@unsw.edu.au)
"""
import numpy as np
from skfem import (MeshTri, Basis, ElementVector, ElementTriP1, ElementTriP2,
                   BilinearForm, asm, condense, solve)
from skfem.helpers import ddot, sym_grad, trace

# ------------------------------------------------------------------ problem
L = 1.0                         # mm, unit square
A0 = 0.5                        # mm, notch from the left edge to the centre
YC = 0.5                        # mm, the notch line
E1, E2 = 340.0e3, 450.0e3       # MPa at x = 0 and x = L
NU = 0.22
GC1, GC2 = 0.04247, 0.120       # N/mm at x = 0 and x = L
DELTA = 1.0e-3                  # mm; the response is linear, so any value
DA = 0.025                      # mm, the crack length step G is differenced on

E_TIP = E1 + (A0 / L) * (E2 - E1)
EP_TIP = E_TIP / (1.0 - NU ** 2)          # plane strain E'
MU_TIP = E_TIP / (2.0 * (1.0 + NU))
KAPPA = 3.0 - 4.0 * NU                    # plane strain
GC_TIP = GC1 + (A0 / L) * (GC2 - GC1)


def slit_mesh(n, a=A0):
    """Structured triangles on the unit square, notch of length ``a`` cut in.

    Returns the mesh and the pairing of the doubled nodes, lower copy first,
    which is what the crack opening is read from.  ``n`` must put both the
    notch line and the tip on the grid.
    """
    assert n % 2 == 0, "n must be even for the notch line to lie on the grid"
    itip = int(round(a / L * n))
    assert abs(itip * L / n - a) < 1e-12, "the tip must fall on a grid node"

    xs = np.linspace(0.0, L, n + 1)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    p = np.vstack([X.ravel(order="F"), Y.ravel(order="F")])

    def nid(i, j):
        return j * (n + 1) + i

    tris = []
    for j in range(n):
        for i in range(n):
            v0, v1 = nid(i, j), nid(i + 1, j)
            v2, v3 = nid(i + 1, j + 1), nid(i, j + 1)
            tris.append([v0, v1, v2])
            tris.append([v0, v2, v3])
    t = np.array(tris, dtype=np.int64).T

    jc = n // 2
    seam = {nid(i, jc): k for k, i in enumerate(range(0, itip))}
    dup = np.array([p[:, v] for v in seam], dtype=np.float64).T
    p = np.hstack([p, dup])
    nold = (n + 1) ** 2

    cy = p[1, t].mean(axis=0)
    for col in np.nonzero(cy > YC)[0]:
        for r in range(3):
            v = t[r, col]
            if v in seam:
                t[r, col] = nold + seam[v]

    pairs = np.array([[v, nold + k] for v, k in seam.items()], dtype=np.int64)
    return MeshTri(p, t), pairs


@BilinearForm
def stiffness(u, v, w):
    E = E1 + (w.x[0] / L) * (E2 - E1)
    lam = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))      # plane strain
    mu = E / (2.0 * (1.0 + NU))
    eu, ev = sym_grad(u), sym_grad(v)
    return 2.0 * mu * ddot(eu, ev) + lam * trace(eu) * trace(ev)


def solve_one(n, elem, a=A0):
    """Solve the elastic problem; return stiffness, mesh, basis, solution."""
    m, pairs = slit_mesh(n, a)
    basis = Basis(m, ElementVector(elem))
    K = asm(stiffness, basis)

    bot = basis.get_dofs(lambda x: np.isclose(x[1], 0.0))
    top = basis.get_dofs(lambda x: np.isclose(x[1], L))
    top_ux = top.all("u^1")

    x = basis.zeros()
    x[top_ux] = DELTA
    u = solve(*condense(K, basis.zeros(), x=x,
                        D=np.concatenate([bot.all(), top.all()])))
    k = float((K @ u)[top_ux].sum()) / DELTA
    return k, m, basis, u, pairs


def factors_from_opening(m, basis, u, pairs, a=A0, rmin=2, rmax=12, n=None):
    """K_I and K_II from the crack face opening, extrapolated to the tip.

    Behind the tip the two faces separate by

        [[u]] = (kappa + 1) / mu * K * sqrt(r / (2 pi)),

    normal component carrying K_I and tangential K_II.  The separation is the
    difference between the two faces, at theta = +pi and -pi, which is twice
    the displacement of either one; the one-sided form would put K out by a
    factor of two and G by four.

    K(r) built from the opening would be constant in the asymptotic region.
    On this problem it is not (see the module docstring), so the intercept
    returned here depends on the fitting window and serves as a cross-check
    only.
    """
    ux = u[basis.nodal_dofs[0]]
    uy = u[basis.nodal_dofs[1]]
    xs = m.p[0, pairs[:, 0]]
    r = a - xs                                    # distance behind the tip
    d_t = ux[pairs[:, 1]] - ux[pairs[:, 0]]       # tangential, mode II
    d_n = uy[pairs[:, 1]] - uy[pairs[:, 0]]       # normal, mode I

    c = (MU_TIP / (KAPPA + 1.0)) * np.sqrt(2.0 * np.pi)
    good = r > 0
    r, d_t, d_n = r[good], d_t[good], d_n[good]
    o = np.argsort(r)
    r, d_t, d_n = r[o], d_t[o], d_n[o]

    h = L / n
    sel = (r >= rmin * h) & (r <= rmax * h)
    out = []
    for d in (d_n, d_t):
        kk = c * d[sel] / np.sqrt(r[sel])
        out.append(float(np.polyfit(r[sel], kk, 1)[1]))   # intercept at r = 0
    return out[0], out[1], int(sel.sum())


def g_from_compliance(n, elem, da):
    """Energy release rate by differencing the stiffness in crack length.

    Under displacement control the potential energy is k(a) delta^2 / 2, so
    G = -dk/da * delta^2 / 2.  A central difference, and no asymptotics.
    """
    kp = solve_one(n, elem, a=A0 + da)[0]
    km = solve_one(n, elem, a=A0 - da)[0]
    return -0.5 * DELTA ** 2 * (kp - km) / (2.0 * da)


def main():
    elem = ElementTriP2()
    print("sharp-crack SEN shear, isothermal, plane strain, FGM 340->450 GPa")
    print("E'(tip) = %.0f MPa   Gc(tip) = %.5f N/mm   delta = %.1e mm\n"
          % (EP_TIP, GC_TIP, DELTA))

    def rich(v):
        v = np.array(v, dtype=float)
        r = (v[0] - v[1]) / (v[1] - v[2])
        return v[-1] - (v[-2] - v[-1]) / (r - 1.0), r

    ns = (40, 80, 160)
    print("%-6s %10s %12s %14s" % ("n", "nodes", "k (N/mm)", "G (N/mm)"))
    ks, gs = [], []
    for n in ns:
        k, m, basis, u, pairs = solve_one(n, elem)
        g = g_from_compliance(n, elem, da=DA)
        ks.append(k)
        gs.append(g)
        print("%-6d %10d %12.1f %14.6e" % (n, m.p.shape[1], k, g))
    k_inf, rk = rich(ks)
    g_inf, rg = rich(gs)
    print("%-6s %10s %12.1f %14.6e   <- Richardson (orders %.2f, %.2f)"
          % ("ext", "", k_inf, g_inf, np.log2(rk), np.log2(rg)))

    print("\ndifferencing step, at n = %d:" % ns[-1])
    for f in (2, 4, 8):
        da = f * L / ns[-1]
        print("  da = %.5f mm   G = %.6e N/mm"
              % (da, g_from_compliance(ns[-1], elem, da=da)))

    print("\ncross-check, factors from the crack opening at n = %d" % ns[-1])
    print("  (depends on the fitting window, see the docstring)")
    _, m, basis, u, pairs = solve_one(ns[-1], elem)
    for lo, hi in ((1, 6), (2, 12), (4, 30), (8, 40)):
        kI, kII, npts = factors_from_opening(m, basis, u, pairs, rmin=lo,
                                             rmax=hi, n=ns[-1])
        print("  window %2d-%2dh (%2d pts)   K_I = %6.3f   K_II = %8.3f"
              % (lo, hi, npts, kI, kII))

    print("\n" + "=" * 66)
    kmag = np.sqrt(EP_TIP * g_inf)
    print("driving force at delta = %.1e mm, from the compliance" % DELTA)
    print("  G = %.6e N/mm    |K| = sqrt(E' G) = %.2f MPa sqrt(mm)"
          % (g_inf, kmag))
    print("  stiffness k = %.0f N/mm" % k_inf)


if __name__ == "__main__":
    main()
