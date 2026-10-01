r"""The sharp-crack finite element reference, checked against a published value.

The graded shear test is compared with a finite element solution that carries
the crack as a seam in the mesh (sharp_reference.py and
sharp_reference_thermal.py).  That implementation was written for this study,
so before it serves as a reference it has to reproduce a published number.

This is that check.  The configuration is the thermal edge crack of Wang et
al. (2023), the stationary benchmark of the proposed method
(examples/stationary_crack.py): a strip 0.5 mm across and 2 mm tall, held at
-1 K on one face and +1 K on the other, rollers (v = 0, u free) on the top and
bottom edges, and an adiabatic edge crack of length 0.25 mm running from the
cold face to the centre line.  Plane strain, E = 218.4 GPa, nu = 0.25,
alpha = 1.67e-5 /K.  Normalized by K_0 = E alpha T_0 sqrt(pi a) / (1 - nu),
Wang et al. report K_I/K_0 = 0.500.

The driving force is taken the same way as in the shear reference: from the
derivative of the potential energy with respect to crack length, with the
temperature field held fixed.  The loading is pure mode I here, so
K_I = sqrt(E' G) needs no mode split.

    python fem/wang_sharp_fe.py

Needs scikit-fem.

Author: Han Zhang (han.zhang7@unsw.edu.au)
"""
import numpy as np
from skfem import (MeshTri, Basis, ElementVector, ElementTriP1, ElementTriP2,
                   BilinearForm, LinearForm, asm, condense, solve)
from skfem.helpers import dot, grad, sym_grad, trace
from scipy.sparse.linalg import splu

# ------------------------------------------------------------- the problem
HALF_L = 0.25        # mm, x in [-0.25, 0.25], so the strip is 0.5 across
HALF_W = 1.0         # mm, y in [-1, 1], so it is 2 tall
A_CRACK = 0.25       # mm, from the cold face at x = -0.25 to the centre line
E = 218.4e3          # MPa
NU = 0.25
ALPHA = 1.67e-5      # 1/K
T0 = 1.0             # K on each face, cold at x = -HALF_L
EP = E / (1.0 - NU ** 2)                  # plane strain
ALPHA_EFF = (1.0 + NU) * ALPHA            # plane strain effective coefficient
K0 = E * ALPHA * T0 * np.sqrt(np.pi * A_CRACK) / (1.0 - NU)
REFERENCE = 0.500

LAM = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
MU = E / (2.0 * (1.0 + NU))
INTORDER = 4


def slit_mesh(nx, ny, a):
    """Rectangle with the edge crack cut in as a seam.

    The crack lies on y = 0 from the left edge to x = -HALF_L + a, and the node
    at the tip is left single, which is what makes the cut a crack.
    """
    assert ny % 2 == 0, "y = 0 must be a grid line"
    xs = np.linspace(-HALF_L, HALF_L, nx + 1)
    ys = np.linspace(-HALF_W, HALF_W, ny + 1)
    itip = int(round((a / (2 * HALF_L)) * nx))
    assert abs(xs[itip] - (-HALF_L + a)) < 1e-12, "the tip must land on a node"

    X, Y = np.meshgrid(xs, ys, indexing="ij")
    p = np.vstack([X.ravel(order="F"), Y.ravel(order="F")])

    def nid(i, j):
        return j * (nx + 1) + i

    t = []
    for j in range(ny):
        for i in range(nx):
            v0, v1 = nid(i, j), nid(i + 1, j)
            v2, v3 = nid(i + 1, j + 1), nid(i, j + 1)
            t.append([v0, v1, v2])
            t.append([v0, v2, v3])
    t = np.array(t, dtype=np.int64).T

    jc = ny // 2
    seam = {nid(i, jc): k for k, i in enumerate(range(0, itip))}
    dup = np.array([p[:, v] for v in seam], dtype=np.float64).T
    p = np.hstack([p, dup])
    nold = (nx + 1) * (ny + 1)
    cy = p[1, t].mean(axis=0)
    for col in np.nonzero(cy > 0.0)[0]:
        for r in range(3):
            if t[r, col] in seam:
                t[r, col] = nold + seam[t[r, col]]
    return MeshTri(p, t)


@BilinearForm
def elasticity(u, v, w):
    eu, ev = sym_grad(u), sym_grad(v)
    return (2.0 * MU * (eu[0, 0] * ev[0, 0] + eu[1, 1] * ev[1, 1]
                        + 2.0 * eu[0, 1] * ev[0, 1])
            + LAM * trace(eu) * trace(ev))


@BilinearForm
def conduction(u, v, w):
    return dot(grad(u), grad(v))


@LinearForm
def thermal_load(v, w):
    beta = ALPHA_EFF * w["T"]
    return 2.0 * (LAM + MU) * beta * trace(sym_grad(v))


def solve_case(nx, ny, a):
    m = slit_mesh(nx, ny, a)
    bu = Basis(m, ElementVector(ElementTriP2()), intorder=INTORDER)
    bt = Basis(m, ElementTriP1())
    bti = Basis(m, ElementTriP1(), intorder=INTORDER)

    # Steady conduction, the faces held and the crack adiabatic by construction.
    A = asm(conduction, bt)
    left = bt.get_dofs(lambda x: np.isclose(x[0], -HALF_L)).all()
    right = bt.get_dofs(lambda x: np.isclose(x[0], HALF_L)).all()
    x = np.zeros(bt.N)
    x[left] = -T0
    x[right] = T0
    T = solve(*condense(A, np.zeros(bt.N), x=x, D=np.concatenate([left, right])))

    # Rollers on top and bottom: v held, u free.  That leaves the strip free to
    # translate in x, so one node's u is pinned to make the system solvable;
    # a rigid translation carries no energy and no driving force.
    K = asm(elasticity, bu)
    f = asm(thermal_load, bu, T=bti.interpolate(T))
    top = bu.get_dofs(lambda x: np.isclose(x[1], HALF_W))
    bot = bu.get_dofs(lambda x: np.isclose(x[1], -HALF_W))
    D = np.concatenate([top.all("u^2"), bot.all("u^2"),
                        bu.get_dofs(lambda x: (np.isclose(x[1], HALF_W)
                                               & np.isclose(x[0], HALF_L))
                                    ).all("u^1")])
    u = solve(*condense(K, f, x=bu.zeros(), D=D))
    pi = 0.5 * float(u @ (K @ u)) - float(f @ u)
    return pi, T, m


def main():
    print("Wang thermal edge crack, sharp crack finite elements")
    print("K_0 = %.6f N/mm^1.5, so the published K_I is %.6f\n"
          % (K0, REFERENCE * K0))
    print("%6s %8s %12s %14s %10s %9s"
          % ("nx", "nodes", "da (mm)", "G (N/mm)", "K_I", "K_I/K_0"))

    rows = []
    for nx in (32, 64, 128):
        ny = 4 * nx
        da = 2.0 * HALF_L / nx          # one element of crack extension
        pi0, _, m = solve_case(nx, ny, A_CRACK)
        pi1, _, _ = solve_case(nx, ny, A_CRACK + da)
        G = -(pi1 - pi0) / da
        kI = np.sqrt(max(EP * G, 0.0))
        rows.append((nx, G, kI))
        print("%6d %8d %12.5f %14.6e %10.5f %9.5f"
              % (nx, m.p.shape[1], da, G, kI, kI / K0))

    g = np.array([r[1] for r in rows])
    r = (g[0] - g[1]) / (g[1] - g[2])
    ginf = g[-1] - (g[-2] - g[-1]) / (r - 1.0)
    kinf = np.sqrt(EP * ginf)
    print("%6s %8s %12s %14.6e %10.5f %9.5f   <- Richardson (ratio %.2f)"
          % ("ext", "", "", ginf, kinf, kinf / K0, r))
    print("\npublished %.3f, computed %.5f, %+.2f per cent"
          % (REFERENCE, kinf / K0, 100.0 * (kinf / K0 / REFERENCE - 1.0)))


if __name__ == "__main__":
    main()
