"""Staggered phase-field finite element solver (scikit-fem), the phase-field reference of the
single-edge notched tension and graded shear tests.

The solver is the phase-field reference of Zhang et al. (2026) for the isothermal elastic case,
extended here to thermoelasticity. It needs scikit-fem only. Units are mm-N-s-K: E in
MPa, Gc in N/mm (= mJ/mm^2), conductivity in mJ/(s mm K) = W/(m K). The code is split into
elasticity, scalar-field and staggered-driver parts: the temperature is one more scalar diffusion
assembled by the same machinery, and the thermoelastic coupling adds a thermal-strain term to the
right-hand side of the elasticity system.

Examples (EXAMPLES below): "tension" and "shear" are the single-edge notched tests; "bifurcation"
(shear with the iso split) and "coalescence" (three en-echelon cracks in tension) are the
isothermal examples of Zhang et al. (2026).

Model: plane strain, AT2 (w=phi^2, c_w=2), degradation g=(1-phi)^2+g_floor. Splits:
  iso    -- no split: u degrades the full isotropic energy, and the full energy drives phi
            (the crack branches in the bifurcation example).
  hybrid -- Ambati et al. (2015): u degrades the full energy (same u-solve as iso), and the
            spectral tension energy (Miehe et al.) drives phi, giving a single crack band; the
            formulation of Tangella et al. (2022).
Irreversibility through the history field H = max_tau psi+ (Miehe et al., 2010). The pre-crack is
a phi=1 Dirichlet condition on the nodes within 1.5 h of the notch segment(s). Displacement and
phase field are staggered within each load step. Run:
  python fem/pf_fem.py --example bifurcation --split iso --l 0.01 --refine 8 --out runs_fem/bifurc

Thermoelastic loading. dT_final=0 is the isothermal elastic solver, and the thermal operators are
not built. dT_final!=0 switches on the thermal load of the single-edge notched tests:
  - Temperature: one more scalar field, advanced by one backward Euler step per load step on the
    physical clock: (M/dt + Kc(phi)) T_n = (M/dt) T_{n-1}. k is degraded by (1-phi)^2, so that
    the crack faces are adiabatic (as in Tangella et al.), with the phase field of the previous
    load step (one-step lag, staggered). BCs: top edge Dirichlet T_top(t)=T0+dT, ramped at
    ramp_rate to |dT| and then held; bottom edge fixed at T0; vertical edges insulated (natural).
  - Clock: t = time_per_delta*delta (default 1e-3 s/mm) and ramp_rate (default 1e8 K/s), the
    loading history of the proposed method.
  - Coupling: elastic strain eps_e = eps - alpha_eff (T-T0) I. alpha_mode ps3d = plane strain
    with the out-of-plane thermal strain, alpha_eff=(1+nu)alpha, which also enters the driving
    energy as an out-of-plane channel | 2d = in-plane thermal strain with alpha. Thermal load
    f_th = int g(phi) [C:eps_th]:eps(v) on the elasticity RHS; psi+ and the reaction use eps_e.
  - Per load step the thermal solve comes first and is then held fixed while u and phi are
    staggered.
  Run: python fem/pf_fem.py --example shear --dT_final 30 --l 0.01 --refine 8 --out runs_fem/shear_dT30
FGM (--fgm; Table 2 of Tangella et al., linear in x): E 340->450 GPa, Gc 42.47->120 J/m^2,
rho 2450->4000 kg/m^3, k 300->350 W/(m K); nu, Cp, alpha, T0 constant. Independent of the thermal
load and combinable with it (graded thermal shear test: --example shear --fgm --dT_final ...).
--asm slow supports the homogeneous isothermal path only; the thermal load and FGM need --asm fast.

Author: Han Zhang (han.zhang7@unsw.edu.au)
"""
import argparse
import math
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, diags as sp_diags
from skfem import (MeshQuad, Basis, ElementVector, ElementQuad1, BilinearForm,
                   LinearForm, asm, solve, condense)
from skfem.helpers import grad, sym_grad, ddot, dot, div


# ---- linear solver: MKL PARDISO through pypardiso when it is installed (multithreaded, follows
# MKL_NUM_THREADS), otherwise scipy's serial SuperLU. PARDISO keeps refine=9 (512 x 512 elements,
# about 0.5M displacement dofs) over hundreds of load steps affordable. The systems (degraded
# elasticity, AT2 phase field) are symmetric, which suits PARDISO.
def _make_linsolve():
    try:
        import pypardiso
        def _f(A, b):
            return pypardiso.spsolve(A.tocsr(), b)
        _f.name = "pardiso"
        return _f
    except Exception:
        from skfem.utils import solver_direct_scipy
        f = solver_direct_scipy()
        try:
            f.name = "scipy"
        except Exception:
            pass
        return f


LINSOLVE = _make_linsolve()


# ------------------------------- config -------------------------------
SEN_NOTCH = [(0.0, 0.5, 0.5, 0.5)]                         # SEN: left-half horizontal notch
COAL_CRACKS = [(0.25, 0.35, 0.35, 0.45),                  # en-echelon cracks (Manav et al., 2024)
               (0.45, 0.45, 0.55, 0.55),
               (0.65, 0.55, 0.75, 0.65)]

EXAMPLES = {
    # example      loading    split(default)  cracks        delta_max
    "shear":       ("shear",   "hybrid", SEN_NOTCH,   2.7e-3),
    "tension":     ("tension", "hybrid", SEN_NOTCH,   8e-4),
    "bifurcation": ("shear",   "iso",    SEN_NOTCH,   2.7e-3),
    "coalescence": ("tension", "hybrid", COAL_CRACKS, 2.7e-3),
}


# FGM gradation endpoints (Table 2 of Tangella et al., linear in x: m(x)=m1 + x (m2-m1); x=0
# keeps the base homogeneous values). E/Gc/rho/k vary; nu, Cp, alpha, T0 are constant.
FGM_E2 = 450e3          # MPa
FGM_GC2 = 0.120         # mJ/mm^2
FGM_K2 = 350.0          # mJ/(s mm K) = W/(m K)
FGM_RHO2 = 4000.0       # kg/m^3


class Cfg:
    def __init__(self, a):
        self.E, self.nu, self.Gc, self.l = a.E, a.nu, a.Gc, a.l
        self.lam = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))   # plane strain
        self.mu = self.E / (2 * (1 + self.nu))
        self.g_floor = 1e-6
        loading, split, cracks, dmax = EXAMPLES[a.example]
        self.example = a.example
        self.loading = a.loading or loading
        self.split = a.split or split
        self.cracks = cracks
        self.delta_max = a.delta_max if a.delta_max > 0 else dmax
        self.refine = a.refine
        self.nsteps = a.nsteps
        self.stag_tol, self.stag_max = a.stag_tol, a.stag_max
        self.asm = a.asm
        self.out = a.out
        # ---- thermoelastic loading (dT_final=0 => isothermal elastic) ----
        self.dT_final = a.dT_final                # top-edge total dT, K (+-30/+-50); 0 = isothermal
        self.thermo = abs(a.dT_final) > 1e-12
        self.alpha = a.alpha                      # thermal expansion /K
        self.T0 = a.T0                            # reference (stress-free) temperature
        self.kcond = a.kcond                      # conductivity W/(mK) == mJ/(s mm K)
        self.rho_cp = a.rho * a.cp * 1e-6         # kg/m^3 * J/(kgK) -> mJ/(mm^3 K)
        self.kdeg = bool(a.kdeg)                  # degrade k by (1-phi)^2 (adiabatic crack faces)
        self.alpha_mode = a.alpha_mode            # ps3d (plane strain, factor (1+nu)) | 2d
        self.ramp_rate = a.ramp_rate              # top-edge K/s until |dT_final| (default 1e8)
        self.time_per_delta = a.time_per_delta    # s/mm clock (1e-3 => dt=1e-8 s at a 1e-5 mm step)
        self.heat_lump = a.heat_mass == "lumped"  # lumped heat capacity => monotone BE (default)
        # plane strain carries the out-of-plane thermal strain => (1+nu) factor
        self.alpha_eff = self.alpha * (1.0 + self.nu) if a.alpha_mode == "ps3d" else self.alpha
        # ---- FGM material gradation (fgm=True => linear-in-x fields; else homogeneous) ----
        self.fgm = bool(a.fgm)
        self.E2, self.Gc2, self.kcond2 = FGM_E2, FGM_GC2, FGM_K2
        self.rho_cp2 = FGM_RHO2 * a.cp * 1e-6


# ------------------------- geometry / notch ---------------------------
def dist_to_segments(px, py, cracks):
    """min point-to-segment distance over all crack segments (vectorized over points)."""
    d = np.full(px.shape, np.inf)
    for ax, ay, bx, by in cracks:
        vx, vy = bx - ax, by - ay
        L2 = vx * vx + vy * vy + 1e-30
        t = np.clip(((px - ax) * vx + (py - ay) * vy) / L2, 0.0, 1.0)
        cx, cy = ax + t * vx, ay + t * vy
        d = np.minimum(d, np.sqrt((px - cx) ** 2 + (py - cy) ** 2))
    return d


def notch_dofs(ip, cfg, band):
    """phi=1 Dirichlet dofs: scalar dofs whose node lies within `band` of any crack segment."""
    return ip.get_dofs(lambda x: dist_to_segments(x[0], x[1], cfg.cracks) < band)


# ------------------------ material fields (homogeneous | FGM) ---------------------
def material_fields(xq, cfg):
    """(lam, mu, Gc, k, rho_cp) at quad points. Homogeneous => the scalar cfg values;
       FGM => linear-in-x arrays shaped like xq (Table 2 of Tangella et al.).
       xq = physical x at the shared quad points."""
    if not cfg.fgm:
        return cfg.lam, cfg.mu, cfg.Gc, cfg.kcond, cfg.rho_cp
    w = np.clip(xq, 0.0, 1.0)
    E = cfg.E + (cfg.E2 - cfg.E) * w
    lam = E * cfg.nu / ((1 + cfg.nu) * (1 - 2 * cfg.nu))
    mu = E / (2 * (1 + cfg.nu))
    Gc = cfg.Gc + (cfg.Gc2 - cfg.Gc) * w
    k = cfg.kcond + (cfg.kcond2 - cfg.kcond) * w
    rho_cp = cfg.rho_cp + (cfg.rho_cp2 - cfg.rho_cp) * w
    return lam, mu, Gc, k, rho_cp


# ------------------------------- thermal clock / BC -------------------------------
def T_top_of_t(cfg, t):
    """Top-edge Dirichlet value at time t: ramp at ramp_rate to T0+dT_final, then hold
       (Tangella et al.), on the clock t = time_per_delta * delta of the proposed method."""
    dT = math.copysign(min(cfg.ramp_rate * t, abs(cfg.dT_final)), cfg.dT_final)
    return cfg.T0 + dT


def heat_bc_dofs(ip):
    """top (ramped) + bottom (fixed T0) Dirichlet; left/right insulated = natural BC."""
    bot = ip.get_dofs(lambda x: x[1] < 1e-9)
    top = ip.get_dofs(lambda x: x[1] > 1 - 1e-9)
    return bot.all(), top.all()


# ------------------------------- forms --------------------------------
def make_forms(cfg):
    lam, mu, gf, Gc, l = cfg.lam, cfg.mu, cfg.g_floor, cfg.Gc, cfg.l

    @BilinearForm
    def Kdeg(u, v, w):                                     # degraded elasticity (full energy)
        g = (1.0 - w["phi"]) ** 2 + gf
        return g * (lam * div(u) * div(v) + 2.0 * mu * ddot(sym_grad(u), sym_grad(v)))

    @BilinearForm
    def Aphi(p, q, w):                                     # AT2 phi-system LHS (scalar grad: dot)
        return Gc * l * dot(grad(p), grad(q)) + (Gc / l + 2.0 * w["H"]) * p * q

    @LinearForm
    def bphi(q, w):                                        # AT2 phi-system RHS
        return 2.0 * w["H"] * q

    return Kdeg, Aphi, bphi


# ---- fast reassembly (cfg.asm == "fast") --------------------------------------------------------
# The staggered loop reassembles K(phi) and A(H) every iteration, and on fine meshes skfem's asm of
# the elasticity operator dominates the cost of an iteration. The geometry and material part of the
# integrand (B^T C B x |detJ|) is constant; only g(phi) and H change. The per-(elem,qp) kernels are
# therefore computed once, and each iteration is a weighted sum and a scatter into a fixed sparsity
# pattern. The result equals asm(Kdeg)/asm(Aphi)/asm(bphi) to floating-point precision. The thermal
# operators (FastHeat, FastThermalRHS) use the same "assemble once, rescale" pattern.
def _coo_template(basis, NB, NE):
    """Fixed (row,col) scatter for a basis: returns (csr template, inverse index into the unique
    entries, n_unique) so per-iteration assembly is bincount(inv, data) into a prebuilt sparsity."""
    ed, N = basis.element_dofs, basis.N
    rows = np.broadcast_to(ed[:, None, :], (NB, NB, NE)).ravel()
    cols = np.broadcast_to(ed[None, :, :], (NB, NB, NE)).ravel()
    keys = rows.astype(np.int64) * N + cols.astype(np.int64)
    uniq, inv = np.unique(keys, return_inverse=True)
    templ = coo_matrix((np.zeros(len(uniq)), (uniq // N, uniq % N)), shape=(N, N)).tocsr()
    return templ, inv, len(uniq)


class FastKdeg:
    """K(phi) = int g(phi) [lam div u div v + 2mu eps(u):eps(v)]. The bracket x |detJ| is precomputed
    per (elem, qp); __call__(phi_qp) weights it by g(phi_qp) = (1-phi_qp)^2 + g_floor and scatters.
    lam, mu may be scalars (homogeneous) or (NE,NQ) fields (FGM)."""
    def __init__(self, iu, cfg, lam, mu):
        self.gf = cfg.g_floor
        NB, NE = iu.Nbfun, iu.nelems
        dx = iu.dx
        grads = [np.asarray(iu.basis[i][0].grad) for i in range(NB)]               # each (2,2,NE,NQ)
        div_i = np.stack([g[0, 0] + g[1, 1] for g in grads], 0)                    # (NB,NE,NQ)
        eps_i = np.stack([0.5 * (g + g.transpose(1, 0, 2, 3)) for g in grads], 0)  # (NB,2,2,NE,NQ)
        divdiv = np.einsum('ieq,jeq->ijeq', div_i, div_i)
        epseps = np.einsum('iabeq,jabeq->ijeq', eps_i, eps_i)
        self.Vqp = dx[None, None] * (lam * divdiv + 2.0 * mu * epseps)             # (NB,NB,NE,NQ)
        self._T, self._inv, self._n = _coo_template(iu, NB, NE)

    def __call__(self, phi_qp):
        g = (1.0 - phi_qp) ** 2 + self.gf
        data = np.einsum('eq,ijeq->ije', g, self.Vqp).ravel()
        K = self._T.copy()
        K.data[:] = np.bincount(self._inv, weights=data, minlength=self._n)
        return K


class FastPhi:
    """AT2 phi-system. A(H) = Gc*l (grad,grad) + (Gc/l + 2H)(.,.): stiffness + Gc/l mass are constant
    (precomputed into _const); only the 2H mass and RHS b(H) = int 2H q change with the history field.
    Gc may be a scalar (homogeneous) or an (NE,NQ) field (FGM)."""
    def __init__(self, ip, cfg, Gc):
        l = cfg.l
        NB, NE = ip.Nbfun, ip.nelems
        dx = ip.dx
        p = np.stack([np.asarray(ip.basis[i][0].value) for i in range(NB)], 0)    # (NB,NE,NQ)
        gp = np.stack([np.asarray(ip.basis[i][0].grad) for i in range(NB)], 0)    # (NB,2,NE,NQ)
        self.Wqp = dx[None, None] * np.einsum('ieq,jeq->ijeq', p, p)              # 2H mass kernel
        lap = np.einsum('ideq,jdeq->ijeq', gp, gp)
        Cqp = dx[None, None] * (Gc * l * lap + (Gc / l) * np.einsum('ieq,jeq->ijeq', p, p))
        self._T, self._inv, self._n = _coo_template(ip, NB, NE)
        self._const = np.einsum('ijeq->ije', Cqp).ravel()
        self._Bqp = dx[None] * p                                                  # (NB,NE,NQ) RHS kernel
        self._ed, self._N = ip.element_dofs, ip.N

    def A(self, H):
        data = self._const + np.einsum('eq,ijeq->ije', 2.0 * H, self.Wqp).ravel()
        A = self._T.copy()
        A.data[:] = np.bincount(self._inv, weights=data, minlength=self._n)
        return A

    def b(self, H):
        be = np.einsum('eq,ieq->ie', 2.0 * H, self._Bqp)
        b = np.zeros(self._N)
        np.add.at(b, self._ed, be)
        return b


class FastHeat:
    """Transient backward-Euler heat: (M/dt + Kc(phi)) T = (M/dt) T_prev, staggered
    (Kc uses the previous step's phi). M = int rho_cp p q is constant; Kc = int k_d |grad|^2
    with k_d = ((1-phi)^2+g_floor) k is reassembled per step (same assemble-once, rescale
    pattern as FastKdeg). k, rho_cp may be scalars or (NE,NQ) FGM fields."""
    def __init__(self, ip, cfg, k, rho_cp):
        self.gf, self.kdeg, self.lump = cfg.g_floor, cfg.kdeg, cfg.heat_lump
        NB, NE = ip.Nbfun, ip.nelems
        dx = ip.dx
        p = np.stack([np.asarray(ip.basis[i][0].value) for i in range(NB)], 0)     # (NB,NE,NQ)
        gp = np.stack([np.asarray(ip.basis[i][0].grad) for i in range(NB)], 0)     # (NB,2,NE,NQ)
        massqp = dx[None, None] * rho_cp * np.einsum('ieq,jeq->ijeq', p, p)
        self.condqp = dx[None, None] * k * np.einsum('ideq,jdeq->ijeq', gp, gp)    # (NB,NB,NE,NQ)
        self._T, self._inv, self._n = _coo_template(ip, NB, NE)
        self._massdata = np.bincount(self._inv,
                                     weights=np.einsum('ijeq->ije', massqp).ravel(),
                                     minlength=self._n)
        self.M = self._T.copy(); self.M.data[:] = self._massdata
        # lumped (row-sum) heat capacity => backward-Euler is an M-matrix => monotone, no
        # under/overshoot of the [T0, T0+dT] band (default; consistent mass overshoots at a
        # steep front). row sum = int rho_cp p_i since sum_j p_j = 1.
        self.mlump = np.asarray(self.M.sum(axis=1)).ravel()

    def A(self, phi_qp, dt):
        g = (1.0 - phi_qp) ** 2 + self.gf if self.kdeg else np.ones_like(phi_qp)
        cond = np.bincount(self._inv,
                           weights=np.einsum('eq,ijeq->ije', g, self.condqp).ravel(),
                           minlength=self._n)
        if self.lump:
            A = self._T.copy(); A.data[:] = cond
            return (A + sp_diags(self.mlump / dt)).tocsr()
        A = self._T.copy(); A.data[:] = self._massdata / dt + cond
        return A

    def b(self, T_prev, dt):
        if self.lump:
            return (self.mlump / dt) * T_prev
        return self.M.dot(T_prev) / dt


class FastThermalRHS:
    """Thermoelastic coupling force f_th(v) = int g(phi) [C:eps_th] : eps(v), with the thermal
    eigenstrain eps_th = c I (c = alpha_eff (T-T0)). For isotropic C: (C:eps_th):eps(v) =
    2(lam+mu) c div(v). Precompute 2(lam+mu) dx div_i once; __call__ weights by c(T)*g(phi)."""
    def __init__(self, iu, cfg, lam, mu):
        NB, NE = iu.Nbfun, iu.nelems
        dx = iu.dx
        grads = [np.asarray(iu.basis[i][0].grad) for i in range(NB)]
        div_i = np.stack([g[0, 0] + g[1, 1] for g in grads], 0)                    # (NB,NE,NQ)
        self.Kern = (2.0 * (lam + mu)) * dx[None] * div_i                          # (NB,NE,NQ)
        self._ed, self._N = iu.element_dofs, iu.N

    def __call__(self, c_qp, g_qp):
        w = c_qp * g_qp                                                            # (NE,NQ)
        fe = np.einsum('eq,ieq->ie', w, self.Kern)
        f = np.zeros(self._N)
        np.add.at(f, self._ed, fe)
        return f


def psi_plus(iu, u, cfg, lam, mu, c_qp=0.0):
    """Tensile driving energy density at quad points on the elastic strain eps_e = eps - eps_th
       (eps_th = c I; ezz_e = -c for plane strain (ps3d), 0 for 2d). c_qp=0 => isothermal.
       iso = full isotropic energy; hybrid = spectral tension part (Miehe et al.).
       lam, mu scalars (homogeneous) or (NE,NQ) fields (FGM)."""
    du = iu.interpolate(u).grad
    exx = du[0, 0] - c_qp                                                          # elastic strain
    eyy = du[1, 1] - c_qp
    exy = 0.5 * (du[0, 1] + du[1, 0])
    ezz = -c_qp if cfg.alpha_mode == "ps3d" else 0.0 * exx                         # out-of-plane
    tr = exx + eyy + ezz
    if cfg.split == "iso":
        return 0.5 * lam * tr ** 2 + mu * (exx ** 2 + eyy ** 2 + ezz ** 2 + 2.0 * exy ** 2)
    # hybrid: spectral tension. in-plane eigenvalues + the out-of-plane channel
    rad = np.sqrt(((exx - eyy) * 0.5) ** 2 + exy ** 2)
    e1, e2 = 0.5 * (exx + eyy) + rad, 0.5 * (exx + eyy) - rad
    pos = lambda z: np.maximum(z, 0.0)
    return 0.5 * lam * pos(tr) ** 2 + mu * (pos(e1) ** 2 + pos(e2) ** 2 + pos(ezz) ** 2)


# ------------------------------- driver -------------------------------
def bc_dofs(iu, cfg):
    bot = iu.get_dofs(lambda x: x[1] < 1e-9)
    top = iu.get_dofs(lambda x: x[1] > 1 - 1e-9)
    return bot, top


def apply_load(iu, bot, top, delta, cfg):
    """returns (x prescribed vector, D constrained dofs, loaded-dof indices for the reaction)."""
    x = iu.zeros()
    if cfg.loading == "shear":                            # bottom clamped, top u=delta v=0
        x[top.all("u^1")] = delta
        D = np.concatenate([bot.all(), top.all("u^1"), top.all("u^2")])
        loaded = top.all("u^1")
    else:                                                 # tension: bottom clamped, top v=delta u=0
        x[top.all("u^2")] = delta
        D = np.concatenate([bot.all(), top.all("u^2"), top.all("u^1")])
        loaded = top.all("u^2")
    return x, D, loaded


def run(cfg):
    os.makedirs(cfg.out, exist_ok=True)
    m = MeshQuad().refined(cfg.refine)
    h = 1.0 / (2 ** cfg.refine)
    iu = Basis(m, ElementVector(ElementQuad1()), intorder=2)
    ip = Basis(m, ElementQuad1(), intorder=2)
    xq = ip.global_coordinates().value[0]                 # physical x at the shared quad points
    lam, mu, Gc, kcond, rho_cp = material_fields(xq, cfg)  # scalars (homogeneous) or (NE,NQ) fields (FGM)
    if cfg.asm == "fast":                                 # precomputed-kernel reassembly (default)
        fk, fp = FastKdeg(iu, cfg, lam, mu), FastPhi(ip, cfg, Gc)
        asmK = lambda phi: fk(ip.interpolate(phi).value)
        asmA, asmb = fp.A, fp.b
    else:                                                 # slow = skfem asm every iteration, as a check
        if cfg.thermo or cfg.fgm:
            raise SystemExit("--asm slow supports the homogeneous elastic path only; "
                             "use --asm fast for thermo / FGM.")
        Kdeg, Aphi, bphi = make_forms(cfg)
        asmK = lambda phi: asm(Kdeg, iu, phi=ip.interpolate(phi))
        asmA = lambda H: asm(Aphi, ip, H=H)
        asmb = lambda H: asm(bphi, ip, H=H)
    notch = notch_dofs(ip, cfg, band=1.5 * h)
    bot, top = bc_dofs(iu, cfg)
    # thermal operators (only built when dT_final != 0)
    heat = FastHeat(ip, cfg, kcond, rho_cp) if cfg.thermo else None
    fth = FastThermalRHS(iu, cfg, lam, mu) if cfg.thermo else None
    hbot, htop = heat_bc_dofs(ip) if cfg.thermo else (None, None)
    T = ip.zeros() + cfg.T0 if cfg.thermo else None       # uniform initial temperature T0
    gf = cfg.g_floor
    print(f"[pf_fem] {cfg.example}: split={cfg.split} loading={cfg.loading} l={cfg.l} "
          f"h={h:.4f} (h/l={h/cfg.l:.2f})  u-dofs={iu.N} phi-dofs={ip.N} notch-dofs={len(notch.all())} "
          f"solver={getattr(LINSOLVE, 'name', '?')} asm={cfg.asm}")
    if cfg.thermo:
        print(f"[pf_fem] THERMO: dT_final={cfg.dT_final:+.0f}K  alpha_mode={cfg.alpha_mode} "
              f"(alpha_eff={cfg.alpha_eff:.3e})  kdeg={cfg.kdeg}  ramp={cfg.ramp_rate:.1e}K/s "
              f"clock={cfg.time_per_delta:.1e}s/mm  kappa={cfg.kcond/cfg.rho_cp:.3e}mm2/s")
    if cfg.fgm:
        print(f"[pf_fem] FGM: E {cfg.E:.0f}->{cfg.E2:.0f}  Gc {cfg.Gc:.4f}->{cfg.Gc2:.3f}  "
              f"k {cfg.kcond:.0f}->{cfg.kcond2:.0f}  rho_cp {cfg.rho_cp:.3e}->{cfg.rho_cp2:.3e}")

    phi = ip.zeros(); phi[notch.all()] = 1.0
    H = np.zeros((m.nelements, 4))
    FD, TH = [], []
    deltas = np.linspace(0.0, cfg.delta_max, cfg.nsteps + 1)[1:]
    t_prev = 0.0
    for k, delta in enumerate(deltas):
        x, D, loaded = apply_load(iu, bot, top, delta, cfg)
        # ---- thermal solve first, then held fixed: transient BE with the previous step's phi ----
        c_qp = 0.0
        if cfg.thermo:
            t_now = cfg.time_per_delta * delta
            dt = max(t_now - t_prev, 1e-30)
            T_top = T_top_of_t(cfg, t_now)
            xT = ip.zeros(); xT[hbot] = cfg.T0; xT[htop] = T_top
            phi_qp = ip.interpolate(phi).value            # one-step lag in phi (staggered)
            T = solve(*condense(heat.A(phi_qp, dt), heat.b(T, dt),
                                 x=xT, D=np.concatenate([hbot, htop])), solver=LINSOLVE)
            c_qp = cfg.alpha_eff * (ip.interpolate(T).value - cfg.T0)
            t_prev = t_now
        # ---- displacement and phase field, staggered, on the fixed T ----
        f_ext = iu.zeros()
        for it in range(cfg.stag_max):
            Km = asmK(phi)
            if cfg.thermo:
                g_qp = (1.0 - ip.interpolate(phi).value) ** 2 + gf
                f_ext = fth(c_qp, g_qp)                    # thermoelastic coupling force
            u = solve(*condense(Km, f_ext, x=x, D=D), solver=LINSOLVE)
            H = np.maximum(H, psi_plus(iu, u, cfg, lam, mu, c_qp))   # irreversible history
            phi_new = solve(*condense(asmA(H), asmb(H),
                                      x=ip.ones(), D=notch.all()), solver=LINSOLVE)
            dchg = np.linalg.norm(phi_new - phi) / (np.linalg.norm(phi_new) + 1e-30)
            phi = phi_new
            if dchg < cfg.stag_tol:
                break
        if cfg.thermo:                                    # reaction on the constrained edge
            g_qp = (1.0 - ip.interpolate(phi).value) ** 2 + gf
            f_ext = fth(c_qp, g_qp)
        F = (asmK(phi) @ u - f_ext)[loaded].sum()
        FD.append([delta, F])
        if cfg.thermo:
            TH.append([delta, cfg.time_per_delta * delta, T_top_of_t(cfg, cfg.time_per_delta * delta),
                       float(T.min()), float(T.max())])
        if k % max(1, cfg.nsteps // 20) == 0 or k == len(deltas) - 1:
            extra = f"  T_top={T_top_of_t(cfg, cfg.time_per_delta*delta):.1f}K" if cfg.thermo else ""
            print(f"  step {k:3d}/{len(deltas)} delta={delta:.3e}  F={F:8.3f}  "
                  f"stag_it={it}  phi_max={phi.max():.3f}  cracked={(phi>0.9).sum()}{extra}")
    FD = np.array(FD)
    np.savetxt(os.path.join(cfg.out, "FD.txt"), FD, header="delta(mm)  F(N)")
    _plot_phi(m, ip, phi, cfg, os.path.join(cfg.out, "phi_final.png"))
    _plot_FD(FD, cfg, os.path.join(cfg.out, "FD.png"))
    np.save(os.path.join(cfg.out, "phi_final.npy"), phi)
    np.save(os.path.join(cfg.out, "nodes.npy"), m.p)
    if cfg.thermo:
        np.savetxt(os.path.join(cfg.out, "THERMO.txt"), np.array(TH),
                   header="delta(mm)  t(s)  T_top(K)  T_min(K)  T_max(K)")
        _plot_T(m, ip, T, cfg, os.path.join(cfg.out, "T_final.png"))
        np.save(os.path.join(cfg.out, "T_final.npy"), T)
    print(f"[pf_fem] done -> {cfg.out}")
    return FD, phi, m


def _plot_phi(m, ip, phi, cfg, path):
    fig, ax = plt.subplots(figsize=(6, 5.4))
    tp = ax.tripcolor(m.p[0], m.p[1], phi, shading="gouraud", cmap="magma", vmin=0, vmax=1)
    fig.colorbar(tp, label="phi")
    ax.set_aspect("equal"); ax.set_title(f"FEM phi  {cfg.example} ({cfg.split}, l={cfg.l})")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def _plot_FD(FD, cfg, path):
    fig, ax = plt.subplots(figsize=(6, 4.6))
    ax.plot(FD[:, 0], FD[:, 1], "-", lw=1.6)
    ax.set_xlabel("displacement delta (mm)"); ax.set_ylabel("reaction F (N)")
    ax.set_title(f"FEM F-delta  {cfg.example} ({cfg.split}, l={cfg.l})")
    ax.grid(True, alpha=0.3); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def _plot_T(m, ip, T, cfg, path):
    # coolwarm anchored to the case range (heating: T0=blue -> T0+dT=reddest; cooling reversed)
    lo, hi = min(cfg.T0, cfg.T0 + cfg.dT_final), max(cfg.T0, cfg.T0 + cfg.dT_final)
    fig, ax = plt.subplots(figsize=(6, 5.4))
    tp = ax.tripcolor(m.p[0], m.p[1], T, shading="gouraud", cmap="coolwarm", vmin=lo, vmax=hi)
    fig.colorbar(tp, label="T (K)")
    ax.set_aspect("equal"); ax.set_title(f"FEM T  {cfg.example} (dT={cfg.dT_final:+.0f}K)")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--example", default="bifurcation", choices=list(EXAMPLES))
    ap.add_argument("--split", default="", help="iso|hybrid (default per example)")
    ap.add_argument("--loading", default="", help="shear|tension (default per example)")
    ap.add_argument("--E", type=float, default=340e3)
    ap.add_argument("--nu", type=float, default=0.22)
    ap.add_argument("--Gc", type=float, default=0.04247)
    ap.add_argument("--l", type=float, default=0.01)
    ap.add_argument("--refine", type=int, default=8, help="MeshQuad refinements (8 => 256x256)")
    ap.add_argument("--nsteps", type=int, default=120)
    ap.add_argument("--delta_max", type=float, default=0.0, help="0 => per-example default")
    ap.add_argument("--stag_tol", type=float, default=1e-3)
    ap.add_argument("--stag_max", type=int, default=50)
    ap.add_argument("--asm", default="fast", choices=["fast", "slow"],
                    help="fast = precomputed-kernel reassembly (default); "
                         "slow = skfem asm each iteration (reference path; homogeneous elastic only)")
    # ---- thermoelastic loading; dT_final=0 => isothermal elastic (default) ----
    ap.add_argument("--dT_final", type=float, default=0.0,
                    help="top-edge total dT (+-30/+-50 K); 0 = isothermal elastic")
    ap.add_argument("--alpha", type=float, default=8e-6, help="thermal expansion /K")
    ap.add_argument("--T0", type=float, default=300.0, help="reference (stress-free) temperature K")
    ap.add_argument("--kcond", type=float, default=300.0, help="conductivity W/(mK)")
    ap.add_argument("--rho", type=float, default=2450.0, help="density kg/m^3")
    ap.add_argument("--cp", type=float, default=0.775, help="specific heat J/(kgK)")
    ap.add_argument("--kdeg", type=int, default=1, help="degrade k by (1-phi)^2 (adiabatic crack faces)")
    ap.add_argument("--alpha_mode", default="ps3d", choices=["ps3d", "2d"],
                    help="ps3d = plane strain with the (1+nu) alpha thermal strain | 2d = in-plane alpha")
    ap.add_argument("--ramp_rate", type=float, default=1e8, help="top-edge temperature ramp in K/s (default 1e8)")
    ap.add_argument("--time_per_delta", type=float, default=1e-3,
                    help="clock in s per mm of applied displacement (1e-3 gives dt = 1e-8 s at 1e-5 mm per step)")
    ap.add_argument("--heat_mass", default="lumped", choices=["lumped", "consistent"],
                    help="lumped (default) = monotone BE, T stays in [T0,T0+dT]; consistent "
                         "overshoots at a steep front")
    # ---- FGM material gradation (Table 2 of Tangella et al., linear in x) ----
    ap.add_argument("--fgm", action="store_true", help="graded E/Gc/rho/k (else homogeneous)")
    ap.add_argument("--out", default="runs_fem/pf")
    run(Cfg(ap.parse_args()))


if __name__ == "__main__":
    main()
