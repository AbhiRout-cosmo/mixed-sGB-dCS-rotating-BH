"""
stage2_source.py
================

Assemble the Stage-II source (report Eq. 150)

    S_{mu nu} = 1/2 (T^(phi)(1,1) + T^(theta)(1,1))
                - H^(1,1) - C^(1,1) - 2 G^(2)[h^(1,0), h^(0,1)]

and solve G^(1)[h^(1,1)] = S for the mixed metric deformation.

All inputs come from the validated lower-order machinery:
  pure sectors  h^(1,0), h^(0,1), phi^(1/2,0), theta^(0,1/2)   (data)
  mixed scalars phi^(1/2,1), theta^(1,1/2)                     (stage1_solver)
"""

from __future__ import annotations

import numpy as np

from kerr_background import KerrBackground, _LC
from perturbation import (assemble_h, covariant_derivs_h, linearized_riemann,
                          dGamma)
from correction_tensors import covariant_hessian, H_sGB_leading, C_dCS_leading
from einstein_operator import G1_lin
from stage2_solver import build_operator, solve_mixed_metric, COMPS


# ---------------------------------------------------------------------- #
# perturbations of connection / Hessian / (dual) Riemann
# ---------------------------------------------------------------------- #
def delta_gamma(bg, Dh):
    """dGamma^lambda_{rho sig} = 1/2 g^{l k}(nabla_rho h_{k sig}+nabla_sig h_{k rho}-nabla_k h_{rho sig})."""
    return 0.5 * np.einsum("...lk,...rks->...lrs", bg.g_uu,
                           np.einsum("...rks->...rks", Dh)
                           + np.einsum("...skr->...rks", Dh)
                           - np.einsum("...krs->...rks", Dh))


def delta_hessian_scalar(bg, dGam, coeffs):
    """(delta nabla_rho nabla_sig f)^(pert) = - dGamma^lambda_{rho sig} d_lambda f."""
    g = bg.grid
    grad = np.zeros(bg.shape + (4,))
    grad[..., 1] = g.d_dr(coeffs); grad[..., 2] = g.d_dchi(coeffs)
    return -np.einsum("...lrs,...l->...rs", dGam, grad)


from functools import lru_cache
from kerr_background import _symbolic_kerr


def _dual_lo(bg):
    """Background *R_{a b c d} (all lower)."""
    return np.einsum("...ai,...bj,...ijcd->...abcd",
                     bg.g_dd, bg.g_dd, bg.dualRiemann_uudd)


@lru_cache(maxsize=None)
def _bg_d2g_cache(spin, Nz, Nchi):
    return None  # placeholder to key the cache; array built in _bg_d2g


def _bg_d2g(bg):
    """Background second metric derivatives d_c d_d g_{mu nu} on the grid."""
    _, _, d2gL = _symbolic_kerr()
    d2 = np.zeros(bg.shape + (4, 4, 4, 4))
    def ev(L):
        o = np.zeros(bg.shape + (4, 4))
        with np.errstate(all="ignore"):
            for i in range(4):
                for j in range(4):
                    o[..., i, j] = np.broadcast_to(
                        np.asarray(L[i][j](bg.R, bg.CHI, bg.a), float), bg.shape)
        return o
    d2[..., 1, 1] = ev(d2gL[(1, 1)]); d2[..., 2, 2] = ev(d2gL[(2, 2)])
    d2[..., 1, 2] = d2[..., 2, 1] = ev(d2gL[(1, 2)])
    return d2


def _geom(bg, gd, dg, d2g):
    """Riemann_{abcd} and dual *R_{abcd} for a metric (gd, dg, d2g) on bg's grid."""
    good = bg.interior
    gi = np.full_like(gd, np.nan)
    gi[good] = np.linalg.inv(gd[good])
    dgi = -np.einsum("...ma,...cab,...bn->...cmn", gi, dg, gi)
    Gam = 0.5 * (np.einsum("...ls,...mns->...lmn", gi, dg)
                 + np.einsum("...ls,...nms->...lmn", gi, dg)
                 - np.einsum("...ls,...smn->...lmn", gi, dg))
    dGam = 0.5 * (np.einsum("...cls,...mns->...clmn", dgi, dg)
                  + np.einsum("...cls,...nms->...clmn", dgi, dg)
                  - np.einsum("...cls,...smn->...clmn", dgi, dg)
                  + np.einsum("...ls,...cmns->...clmn", gi, d2g)
                  + np.einsum("...ls,...cnms->...clmn", gi, d2g)
                  - np.einsum("...ls,...csmn->...clmn", gi, d2g))
    Ru = (np.einsum("...mrns->...rsmn", dGam) - np.einsum("...nrms->...rsmn", dGam)
          + np.einsum("...rml,...lns->...rsmn", Gam, Gam)
          - np.einsum("...rnl,...lms->...rsmn", Gam, Gam))
    Rd = np.einsum("...ar,...rbcd->...abcd", gd, Ru)
    sg = np.sqrt(np.abs(np.linalg.det(np.where(np.isnan(gd), 0.0, gd))))
    with np.errstate(divide="ignore", invalid="ignore"):
        eu = (1.0 / sg)[..., None, None, None, None] * _LC
    euudd = np.einsum("...abef,...ce,...df->...abcd", eu, gd, gd)
    Ruudd = np.einsum("...ai,...bj,...ijcd->...abcd", gi, gi, Rd)
    duudd = 0.5 * np.einsum("...abef,...efcd->...abcd", euudd, Ruudd)
    dual = np.einsum("...ai,...bj,...ijcd->...abcd", gd, gd, duudd)
    return Rd, dual


def delta_geom(bg, h, dh, d2h, eps=1e-6):
    """True first-order (dR_{abcd}, d*R_{abcd}) from perturbation h via eps-FD."""
    dg0 = bg.dg_ddd
    d2g0 = _bg_d2g(bg)
    with np.errstate(divide="ignore", invalid="ignore"):
        Rp, Dp = _geom(bg, bg.g_dd + eps * h, dg0 + eps * dh, d2g0 + eps * d2h)
        Rm, Dm = _geom(bg, bg.g_dd - eps * h, dg0 - eps * dh, d2g0 - eps * d2h)
    return (Rp - Rm) / (2 * eps), (Dp - Dm) / (2 * eps)


# ---------------------------------------------------------------------- #
# mixed correction tensors H^(1,1), C^(1,1)  (Eqs. 84, 93)
# ---------------------------------------------------------------------- #
def H_mixed(bg, phi10, phi11, h01, dh01, d2h01):
    """H^(1,1)_{mu nu} = curvature(phi11) + source(phi10, h01)."""
    Hcurv = H_sGB_leading(bg, phi11)                                     # 4 R nabla nabla phi11
    dR, _ = delta_geom(bg, h01, dh01, d2h01)                            # true dR_{abcd}
    hess = covariant_hessian(bg, phi10)
    hess_uu = np.einsum("...ma,...nb,...ab->...mn", bg.g_uu, bg.g_uu, hess)
    term_dR = 4.0 * np.einsum("...mrns,...rs->...mn", dR, hess_uu)
    Dh01, _ = covariant_derivs_h(bg, h01, dh01, d2h01)
    dGam = delta_gamma(bg, Dh01)
    dhess = delta_hessian_scalar(bg, dGam, phi10)
    dhess_uu = np.einsum("...ma,...nb,...ab->...mn", bg.g_uu, bg.g_uu, dhess)
    term_dH = 4.0 * np.einsum("...mrns,...rs->...mn", bg.Riemann_dddd, dhess_uu)
    return Hcurv + term_dR + term_dH


def C_mixed(bg, th10, th11, h10, dh10, d2h10):
    """C^(1,1)_{mu nu} = curvature(theta11) + source(theta10, h10)."""
    Ccurv = C_dCS_leading(bg, th11)
    dualR = _dual_lo(bg)
    # source term A: (delta nabla nabla theta10) contracted with *R^(0)
    Dh10, DDh10 = covariant_derivs_h(bg, h10, dh10, d2h10)
    dGam = delta_gamma(bg, Dh10)
    dhess = delta_hessian_scalar(bg, dGam, th10)
    dhess_uu = np.einsum("...ma,...nb,...ab->...mn", bg.g_uu, bg.g_uu, dhess)
    Csrc_A = -4.0 * 0.5 * (
        np.einsum("...ad,...amnd->...mn", dhess_uu, dualR)
        + np.swapaxes(np.einsum("...ad,...amnd->...mn", dhess_uu, dualR), -1, -2))
    # source term B: (nabla nabla theta10) contracted with delta(*R)^(1,0)
    hess = covariant_hessian(bg, th10)
    hess_uu = np.einsum("...ma,...nb,...ab->...mn", bg.g_uu, bg.g_uu, hess)
    _, ddual = delta_geom(bg, h10, dh10, d2h10)                         # true d*R_{abcd}
    Csrc_B = -4.0 * 0.5 * (
        np.einsum("...ad,...amnd->...mn", hess_uu, ddual)
        + np.swapaxes(np.einsum("...ad,...amnd->...mn", hess_uu, ddual), -1, -2))
    return Ccurv + Csrc_A + Csrc_B


# ---------------------------------------------------------------------- #
# second-order Einstein operator  G^(2)[h]  (Eqs. 74-76)
# ---------------------------------------------------------------------- #
def _R1(bg, h, dh, d2h):
    guu = bg.g_uu
    _, DDh = covariant_derivs_h(bg, h, dh, d2h)
    t1 = np.einsum("...ab,...bman->...mn", guu, DDh)
    t2 = np.einsum("...ab,...bnam->...mn", guu, DDh)
    box_h = np.einsum("...ab,...abmn->...mn", guu, DDh)
    hess_tr = np.einsum("...ab,...mnab->...mn", guu, DDh)
    R1_dd = 0.5 * (t1 + t2 - box_h - hess_tr)
    R1 = -np.einsum("...mn,...mn->...", guu, hess_tr) \
        + np.einsum("...ma,...nb,...mnab->...", guu, guu, DDh)
    return R1_dd, R1


def R2_single(bg, h, dh, d2h):
    guu = bg.g_uu
    Dh, DDh = covariant_derivs_h(bg, h, dh, d2h)
    huu = np.einsum("...ma,...nb,...ab->...mn", guu, guu, h)
    Dh_uu = np.einsum("...rab,...ac,...bd->...rcd", Dh, guu, guu)          # nabla_r h^{cd}

    T1 = 0.5 * np.einsum("...mab,...nab->...mn", Dh, Dh_uu)
    T2 = (np.einsum("...ab,...mnab->...mn", huu, DDh)
          + np.einsum("...ab,...abmn->...mn", huu, DDh)
          - np.einsum("...ab,...amnb->...mn", huu, DDh)
          - np.einsum("...ab,...anmb->...mn", huu, DDh))
    M1 = np.einsum("...rcn,...ac->...ran", Dh, guu)     # nabla_beta h^alpha_nu  [beta,alpha,nu]
    M2 = np.einsum("...dam,...bd->...bam", Dh, guu)     # nabla^beta h_{alpha mu}[beta,alpha,mu]
    M3 = np.einsum("...adm,...bd->...abm", Dh, guu)     # nabla_alpha h^beta_mu [alpha,beta,mu]
    T3 = (np.einsum("...ban,...bam->...mn", M1, M2)
          - np.einsum("...ban,...abm->...mn", M1, M3))
    Fu = (np.einsum("...bcd,...ac,...bd->...a", Dh, guu, guu)              # nabla_b h^{ab}
          - 0.5 * np.einsum("...ac,...ef,...cef->...a", guu, guu, Dh))    # 1/2 nabla^a h
    bracket = (np.einsum("...man->...amn", Dh)
               + np.einsum("...nam->...amn", Dh)
               - np.einsum("...amn->...amn", Dh))
    T4 = -np.einsum("...a,...amn->...mn", Fu, bracket)

    R2 = 0.5 * (T1 + T2 + T3 + T4)
    return 0.5 * (R2 + np.swapaxes(R2, -1, -2))          # R^(2) symmetric


def G2_single(bg, h, dh, d2h):
    R2 = R2_single(bg, h, dh, d2h)
    R1_dd, R1 = _R1(bg, h, dh, d2h)
    guu, gdd = bg.g_uu, bg.g_dd
    R2tr = np.einsum("...ab,...ab->...", guu, R2)
    hR1 = np.einsum("...ab,...ab->...", np.einsum("...ma,...nb,...ab->...mn", guu, guu, h), R1_dd)
    return (R2 - 0.5 * gdd * R2tr[..., None, None]
            + 0.5 * gdd * hR1[..., None, None] - 0.5 * h * R1[..., None, None])


def G2_bilinear(bg, Ha, Hb):
    """G^(2)[h_a, h_b] by polarization (Eq. 76).  Ha, Hb are (h,dh,d2h) tuples."""
    with np.errstate(invalid="ignore"):
        hab = tuple(a + b for a, b in zip(Ha, Hb))
    return 0.5 * (G2_single(bg, *hab) - G2_single(bg, *Ha) - G2_single(bg, *Hb))


# ---------------------------------------------------------------------- #
# mixed stress-energy (Eqs. 108, 109)
# ---------------------------------------------------------------------- #
def stress_cross(bg, cA, cB):
    g = bg.grid
    gA = np.zeros(bg.shape + (4,)); gA[..., 1] = g.d_dr(cA); gA[..., 2] = g.d_dchi(cA)
    gB = np.zeros(bg.shape + (4,)); gB[..., 1] = g.d_dr(cB); gB[..., 2] = g.d_dchi(cB)
    dot = np.einsum("...mn,...m,...n->...", bg.g_uu, gA, gB)
    return (np.einsum("...m,...n->...mn", gA, gB)
            + np.einsum("...m,...n->...mn", gB, gA)
            - bg.g_dd * dot[..., None, None])


# ---------------------------------------------------------------------- #
# full source + solve
# ---------------------------------------------------------------------- #
def build_source(bg, phi10, th10, phi11, th11, Hsg, Hcs):
    h10, dh10, d2h10 = assemble_h(bg, Hsg)
    h01, dh01, d2h01 = assemble_h(bg, Hcs)

    T = 0.5 * (stress_cross(bg, phi10, phi11) + stress_cross(bg, th10, th11))
    H11 = H_mixed(bg, phi10, phi11, h01, dh01, d2h01)
    C11 = C_mixed(bg, th10, th11, h10, dh10, d2h10)
    G2 = G2_bilinear(bg, (h10, dh10, d2h10), (h01, dh01, d2h01))
    return T - H11 - C11 - 2.0 * G2


def _selftest(path="data_a_50_0.9.json"):
    import os
    from spectral_grid import SpectralGrid
    from stage1_solver import solve_mixed_scalars

    print("=" * 68)
    print("stage2_source  full mixed-metric solve")
    print("=" * 68)
    if not os.path.exists(path):
        print(f"[data] '{path}' not found; exiting."); return

    g, raw = SpectralGrid.from_file(path)
    bg = KerrBackground(g); intr = bg.interior
    Hsg = {i: g.fit(g.field_values(raw, f"sGBH{i}"), "even") for i in (1, 2, 3, 4)}
    Hcs = {i: g.fit(g.field_values(raw, f"dCSH{i}"), "even") for i in (1, 2, 3, 4)}
    phi10 = g.fit(g.field_values(raw, "sGBPhi"), "even")
    th10 = g.fit(g.field_values(raw, "dCSPhi"), "odd")

    print("\n[1] solving Stage-I mixed scalars ...")
    phi11, th11 = solve_mixed_scalars(bg, phi10, th10, Hsg, Hcs)

    print("[2] assembling Stage-II source ...")
    S = build_source(bg, phi10, th10, phi11, th11, Hsg, Hcs)
    sym = np.max(np.abs((S - np.swapaxes(S, -1, -2))[intr]))
    bulk = np.zeros(bg.shape, bool); bulk[5:-5, 5:-5] = True
    print(f"    source symmetric : {sym:.2e}   max|S|={np.max(np.abs(S[intr])):.2e}"
          f"   bulk-max|S|={np.max(np.abs(S[bulk])):.2e}")

    print("[3] building G^(1) operator (slow) ...")
    A, cols = build_operator(bg, trunc=3)
    Hsol = solve_mixed_metric(bg, S, A_cols=(A, cols))

    res = np.max(np.abs((G1_lin(bg, *assemble_h(bg, Hsol)) - S)[intr]))
    sc = np.max(np.abs(S[intr]))
    Rt = G1_lin(bg, *assemble_h(bg, Hsol)) - S
    # even (physical, gauge-invariant) sector vs the odd rchi gauge component
    ev = max(np.max(np.abs(Rt[..., a, b][bulk])) for (a, b) in COMPS)
    evsc = max(np.max(np.abs(S[..., a, b][bulk])) for (a, b) in COMPS)
    rchi = np.max(np.abs(Rt[..., 1, 2][bulk]))
    print(f"\n[4] full-tensor residual        : {res:.2e}   rel {res/sc:.2e}")
    print(f"    EVEN-sector residual (phys) : {ev:.2e}   rel {ev/evsc:.2e}")
    print(f"    rchi component (gauge)      : {rchi:.2e}   [circular ansatz omits h_rchi]")
    for i in (1, 2, 3, 4):
        print(f"    max|H_{i}^(1,1)| = {np.max(np.abs(g.to_grid(Hsol[i])[intr])):.3e}")
    print()


if __name__ == "__main__":
    import sys
    _selftest(sys.argv[1] if len(sys.argv) > 1 else "data_a_50_0.99.json")
