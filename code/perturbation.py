"""
perturbation.py
===============

Metric-deformation machinery for the mixed-order (Stage I) scalar sources.

Given the sector deformation functions H_i(r, chi) (spectral coefficients,
even-l), this module:

  * assembles h_{mu nu} via the report ansatz Eq. (35) and its first/second
    COORDINATE derivatives (analytic prefactor * spectral H_i, product rule);
  * forms first/second COVARIANT derivatives of h on the Kerr background;
  * builds the linearized Riemann dR_{mu nu rho sig} (Eq. 127);
  * builds the four mixed-scalar source pieces:
        G^{(0,1)}     Eq. (142)/(116)      -> dilaton cross-source
        box^{(0,1)}   Eq. (143)            -> perturbed d'Alembertian
        (*RR)^{(1,0)} Eq. (146)            -> axion cross-source
        box^{(1,0)}   Eq. (145)  (same operator as box^{(0,1)})

Conventions follow kerr_background: COORD=(t,r,chi,phi), M=1, sig(-,+,+,+),
Riemann_dddd[...,a,b,c,d]=R_{abcd}, dualRiemann_uudd[...,a,b,c,d]=*R^{ab}{}_{cd}.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Tuple

import numpy as np
import sympy as sp

from kerr_background import KerrBackground, _symbolic_kerr
from correction_tensors import covariant_hessian


# ---------------------------------------------------------------------- #
# ansatz prefactors P_{mu nu}(r, chi, a): h_{mu nu} = P_{mu nu} * H_k
#   (t,t)->H1  (t,phi)->H2  (r,r)&(chi,chi)->H3  (phi,phi)->H4   [Eq. 35, M=1]
# ---------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def _prefactor_forms():
    r, chi, a = sp.symbols("r chi a", real=True)
    Sig = r**2 + a**2 * chi**2
    Delta = r**2 - 2 * r + a**2
    omc = 1 - chi**2

    P = {
        (0, 0): sp.Integer(1),                                   # h_tt      = H1
        (0, 3): -2 * a * r * omc / Sig,                          # h_tphi    = H2
        (1, 1): Sig / Delta,                                     # h_rr      = H3
        (2, 2): Sig / omc,                                       # h_chichi  = H3
        (3, 3): omc * (r**2 + a**2 + 2 * a**2 * r * omc / Sig),  # h_phiphi  = H4
        (1, 2): sp.Integer(1),                                   # h_rchi    = H5 (odd chi)
    }
    Hidx = {(0, 0): 1, (0, 3): 2, (1, 1): 3, (2, 2): 3, (3, 3): 4, (1, 2): 5}

    def lam(expr):
        return {
            "P":  sp.lambdify((r, chi, a), expr, "numpy"),
            "r":  sp.lambdify((r, chi, a), sp.diff(expr, r), "numpy"),
            "c":  sp.lambdify((r, chi, a), sp.diff(expr, chi), "numpy"),
            "rr": sp.lambdify((r, chi, a), sp.diff(expr, r, r), "numpy"),
            "cc": sp.lambdify((r, chi, a), sp.diff(expr, chi, chi), "numpy"),
            "rc": sp.lambdify((r, chi, a), sp.diff(expr, r, chi), "numpy"),
        }

    return {k: lam(v) for k, v in P.items()}, Hidx


def _eval(fn, R, CHI, a):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.broadcast_to(np.asarray(fn(R, CHI, a), float), R.shape)


# ---------------------------------------------------------------------- #
# h assembly with coordinate derivatives
# ---------------------------------------------------------------------- #
def assemble_h(bg: KerrBackground, Hcoeffs: Dict[int, np.ndarray]
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (h_dd, dh, d2h):
        h_dd : (...,4,4)          h_{mu nu}
        dh   : (...,4,4,4)        d_c h_{mu nu}      [c = deriv dir]
        d2h  : (...,4,4,4,4)      d_c d_d h_{mu nu}
    Hcoeffs maps i in {1,2,3,4} -> spectral coefficients of H_i (even-l).
    """
    g = bg.grid
    R, CHI, a = bg.R, bg.CHI, bg.a
    forms, Hidx = _prefactor_forms()

    sh = bg.shape
    h = np.zeros(sh + (4, 4))
    dh = np.zeros(sh + (4, 4, 4))       # [...,c,mu,nu]
    d2h = np.zeros(sh + (4, 4, 4, 4))   # [...,c,d,mu,nu]

    # spectral H_i values + derivatives
    Hval, Hr, Hc, Hrr, Hcc, Hrc = {}, {}, {}, {}, {}, {}
    for i, c in Hcoeffs.items():
        Hval[i] = g.to_grid(c)
        Hr[i] = g.d_dr(c);   Hc[i] = g.d_dchi(c)
        Hrr[i] = g.d2_dr2(c); Hcc[i] = g.d2_dchi2(c)
        Hrc[i] = g._dz_dr[:, None] * g.d_dz_dchi(c)

    for (i, j), fset in forms.items():
        k = Hidx[(i, j)]
        if k not in Hval:                               # component not supplied -> skip
            continue
        P = _eval(fset["P"], R, CHI, a)
        Pr = _eval(fset["r"], R, CHI, a);  Pc = _eval(fset["c"], R, CHI, a)
        Prr = _eval(fset["rr"], R, CHI, a); Pcc = _eval(fset["cc"], R, CHI, a)
        Prc = _eval(fset["rc"], R, CHI, a)
        Hv = Hval[k]

        with np.errstate(divide="ignore", invalid="ignore"):
            val = P * Hv
            v_r = Pr * Hv + P * Hr[k]
            v_c = Pc * Hv + P * Hc[k]
            v_rr = Prr * Hv + 2 * Pr * Hr[k] + P * Hrr[k]
            v_cc = Pcc * Hv + 2 * Pc * Hc[k] + P * Hcc[k]
            v_rc = Prc * Hv + Pr * Hc[k] + Pc * Hr[k] + P * Hrc[k]

        for (m, n) in {(i, j), (j, i)}:                 # symmetric placement
            h[..., m, n] = val
            dh[..., 1, m, n] = v_r;  dh[..., 2, m, n] = v_c
            d2h[..., 1, 1, m, n] = v_rr
            d2h[..., 2, 2, m, n] = v_cc
            d2h[..., 1, 2, m, n] = d2h[..., 2, 1, m, n] = v_rc
    return h, dh, d2h


# ---------------------------------------------------------------------- #
# derivative of the Christoffel connection  d_c Gamma^l_{mn}
# ---------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def _d2g_forms():
    _, _, d2gL = _symbolic_kerr()
    return d2gL


def dGamma(bg: KerrBackground) -> np.ndarray:
    """d_c Gamma^l_{mn}, shape (...,4,4,4,4) = [...,c,l,m,n]  (c=r,chi nonzero)."""
    ginv = bg.g_uu
    dgc = bg.dg_ddd                                      # [...,c,mu,nu]
    d2gL = _d2g_forms()
    R, CHI, a = bg.R, bg.CHI, bg.a

    d2gc = np.zeros(bg.shape + (4, 4, 4, 4))            # [...,c,d,mu,nu]
    for c, d, L in ((1, 1, d2gL[(1, 1)]), (2, 2, d2gL[(2, 2)]),
                    (1, 2, d2gL[(1, 2)])):
        M = np.zeros(bg.shape + (4, 4))
        with np.errstate(divide="ignore", invalid="ignore"):
            for ii in range(4):
                for jj in range(4):
                    M[..., ii, jj] = np.broadcast_to(
                        np.asarray(L[ii][jj](R, CHI, a), float), bg.shape)
        d2gc[..., c, d, :, :] = M
        if c != d:
            d2gc[..., d, c, :, :] = M

    dginv = -np.einsum("...ma,...cab,...bn->...cmn", ginv, dgc, ginv)
    return 0.5 * (
        np.einsum("...cls,...mns->...clmn", dginv, dgc)
        + np.einsum("...cls,...nms->...clmn", dginv, dgc)
        - np.einsum("...cls,...smn->...clmn", dginv, dgc)
        + np.einsum("...ls,...cmns->...clmn", ginv, d2gc)
        + np.einsum("...ls,...cnms->...clmn", ginv, d2gc)
        - np.einsum("...ls,...csmn->...clmn", ginv, d2gc)
    )


# ---------------------------------------------------------------------- #
# covariant derivatives of h  ->  Dh_{rho mu sig},  DDh_{rho nu mu sig}
#   Dh[rho,mu,sig]      = nabla_rho h_{mu sig}
#   DDh[rho,nu,mu,sig]  = nabla_rho nabla_nu h_{mu sig}
# ---------------------------------------------------------------------- #
def covariant_derivs_h(bg, h, dh, d2h):
    G = bg.Gamma
    dG = dGamma(bg)

    Dh = (dh
          - np.einsum("...lrm,...ls->...rms", G, h)
          - np.einsum("...lrs,...ml->...rms", G, h))

    dDh = (d2h
           - np.einsum("...rlnm,...ls->...rnms", dG, h)
           - np.einsum("...lnm,...rls->...rnms", G, dh)
           - np.einsum("...rlns,...ml->...rnms", dG, h)
           - np.einsum("...lns,...rml->...rnms", G, dh))

    DDh = (dDh
           - np.einsum("...lrn,...lms->...rnms", G, Dh)
           - np.einsum("...lrm,...nls->...rnms", G, Dh)
           - np.einsum("...lrs,...nml->...rnms", G, Dh))
    return Dh, DDh


def linearized_riemann(DDh: np.ndarray) -> np.ndarray:
    """dR_{mu nu rho sig} = 1/2(nabla_rho nabla_nu h_{mu sig} + nabla_sig nabla_mu h_{nu rho}
                                -nabla_sig nabla_nu h_{mu rho} - nabla_rho nabla_mu h_{nu sig})
       (Eq. 127).  DDh axes = [rho, nu, mu, sig]."""
    return 0.5 * (np.einsum("...RNMS->...MNRS", DDh)
                  + np.einsum("...SMNR->...MNRS", DDh)
                  - np.einsum("...SNMR->...MNRS", DDh)
                  - np.einsum("...RMNS->...MNRS", DDh))


# ---------------------------------------------------------------------- #
# mixed-scalar source pieces
# ---------------------------------------------------------------------- #
def _riem_uuuu(bg):
    return np.einsum("...ai,...bj,...ck,...dl,...ijkl->...abcd",
                     bg.g_uu, bg.g_uu, bg.g_uu, bg.g_uu, bg.Riemann_dddd)


def _dual_uuuu(bg):
    return np.einsum("...ck,...dl,...abcd->...abkl",
                     bg.g_uu, bg.g_uu, bg.dualRiemann_uudd)


def gb_invariant_pert(bg, h, dh, d2h):
    """G^{(0,1)} = 4 R^{mu nu rho sig} nabla_rho nabla_nu h_{mu sig}  (Eq. 116)."""
    _, DDh = covariant_derivs_h(bg, h, dh, d2h)
    Ruuuu = _riem_uuuu(bg)
    return 4.0 * np.einsum("...mnrs,...rnms->...", Ruuuu, DDh)


def pontryagin_pert(bg, h, dh, d2h):
    """(*RR)^{(1,0)} = -1/2 h (*RR)^{(0)} + 4 *R^{mu nu rho sig} nabla_rho nabla_nu h_{mu sig}
       (Eq. 146)."""
    _, DDh = covariant_derivs_h(bg, h, dh, d2h)
    htr = np.einsum("...mn,...mn->...", bg.g_uu, h)
    dRuuuu = _dual_uuuu(bg)
    return -0.5 * htr * bg.pontryagin \
        + 4.0 * np.einsum("...mnrs,...rnms->...", dRuuuu, DDh)


def box_pert(bg, h, dh, field_coeffs):
    """
    Perturbed d'Alembertian acting on a scalar (Eqs. 143 / 145):
        box^{(1)} f = -h^{mu nu} nabla_mu nabla_nu f
                      - (nabla_mu h^{mu nu} - 1/2 nabla^nu h) nabla_nu f
    Works for either sector (pass that sector's h, dh).
    """
    Dh, _ = covariant_derivs_h(bg, h, dh, np.zeros(bg.shape + (4, 4, 4, 4)))
    huu = np.einsum("...ma,...nb,...ab->...mn", bg.g_uu, bg.g_uu, h)
    hess = covariant_hessian(bg, field_coeffs)
    div = np.einsum("...ma,...nb,...mab->...n", bg.g_uu, bg.g_uu, Dh)  # nabla_mu h^{mu nu}
    gradh_d = np.einsum("...ab,...nab->...n", bg.g_uu, Dh)             # nabla_nu h
    gradh_u = np.einsum("...nm,...m->...n", bg.g_uu, gradh_d)

    gf = bg.grid.d_dr(field_coeffs), bg.grid.d_dchi(field_coeffs)
    gradf_d = np.zeros(bg.shape + (4,))
    gradf_d[..., 1] = gf[0]; gradf_d[..., 2] = gf[1]

    term1 = -np.einsum("...mn,...mn->...", huu, hess)
    term2 = -np.einsum("...n,...n->...", div - 0.5 * gradh_u, gradf_d)
    return term1 + term2


# ====================================================================== #
# self-test
# ====================================================================== #
def _selftest(path: str = "data_a_50_0.99.json") -> None:
    import os
    from spectral_grid import SpectralGrid

    print("=" * 68)
    print("perturbation self-test")
    print("=" * 68)

    if not os.path.exists(path):
        print(f"[data] '{path}' not found; cannot run.  Exiting.")
        return

    g, raw = SpectralGrid.from_file(path)
    bg = KerrBackground(g)
    intr = bg.interior

    Hsg = {i: g.fit(g.field_values(raw, f"sGBH{i}"), parity="even") for i in (1, 2, 3, 4)}
    Hcs = {i: g.fit(g.field_values(raw, f"dCSH{i}"), parity="even") for i in (1, 2, 3, 4)}
    phi = g.fit(g.field_values(raw, "sGBPhi"), parity="even")
    th = g.fit(g.field_values(raw, "dCSPhi"), parity="odd")

    h_sg, dh_sg, d2h_sg = assemble_h(bg, Hsg)
    h_cs, dh_cs, d2h_cs = assemble_h(bg, Hcs)

    def rel(a, b):
        s = np.max(np.abs(a[intr]))
        return np.max(np.abs((a - b)[intr])) / max(s, 1e-300), s

    # (1) G^(0,1): Eq.116 vs Eq.115 (via linearized Riemann)
    _, DDh_cs = covariant_derivs_h(bg, h_cs, dh_cs, d2h_cs)
    dR_cs = linearized_riemann(DDh_cs)
    Ruuuu = _riem_uuuu(bg)
    G01_a = gb_invariant_pert(bg, h_cs, dh_cs, d2h_cs)
    G01_b = 2.0 * np.einsum("...mnrs,...mnrs->...", Ruuuu, dR_cs)
    r1, s1 = rel(G01_a, G01_b)
    print(f"\n[1] G^(0,1)  Eq116 vs Eq115     rel {r1:.2e}   scale {s1:.2e}")

    # (2) (*RR)^(1,0): Eq.146 vs Eq.125
    _, DDh_sg = covariant_derivs_h(bg, h_sg, dh_sg, d2h_sg)
    dR_sg = linearized_riemann(DDh_sg)
    dRuuuu = _dual_uuuu(bg)
    htr = np.einsum("...mn,...mn->...", bg.g_uu, h_sg)
    pRR_a = pontryagin_pert(bg, h_sg, dh_sg, d2h_sg)
    pRR_b = -0.5 * htr * bg.pontryagin + 2.0 * np.einsum("...mnrs,...mnrs->...", dRuuuu, dR_sg)
    r2, s2 = rel(pRR_a, pRR_b)
    print(f"[2] (*RR)^(1,0) Eq146 vs Eq125  rel {r2:.2e}   scale {s2:.2e}")

    # (3) linearized Riemann pair symmetries
    asym12 = np.max(np.abs((dR_cs + np.swapaxes(dR_cs, -4, -3))[intr]))
    asym34 = np.max(np.abs((dR_cs + np.swapaxes(dR_cs, -2, -1))[intr]))
    print(f"[3] dR antisym(12)/(34)         {asym12:.2e} / {asym34:.2e}   "
          f"(scale {np.max(np.abs(dR_cs[intr])):.1e})")

    # (4) box_pert consistency: box^(0,1)[h=g] f = -box0 f
    #     (h=g with its true coordinate derivatives -> nabla g = 0, so only
    #      -g^{mu nu} nabla_mu nabla_nu f = -box0 f survives)
    box_g = box_pert(bg, bg.g_dd, bg.dg_ddd, phi)
    b0 = bg.box0(phi)
    r4 = np.max(np.abs((box_g + b0)[intr])) / max(np.max(np.abs(b0[intr])), 1e-300)
    print(f"[4] box_pert[g]phi == -box0 phi  rel {r4:.2e}")

    # actual mixed sources (report Stage-I RHS)
    box01_phi = box_pert(bg, h_cs, dh_cs, phi)
    box10_th = box_pert(bg, h_sg, dh_sg, th)
    print(f"\n    mixed dilaton source  |box^(0,1)phi + G^(0,1)|_inf : "
          f"{np.max(np.abs((box01_phi + G01_a)[intr])):.3e}")
    print(f"    mixed axion   source  |box^(1,0)th + (*RR)^(1,0)|_inf: "
          f"{np.max(np.abs((box10_th + pRR_a)[intr])):.3e}")

    assert r1 < 1e-3 and r2 < 1e-3 and r4 < 1e-3
    assert max(asym12, asym34) < 1e-8 * np.max(np.abs(dR_cs[intr]))
    print("\nself-test PASSED\n")


if __name__ == "__main__":
    import sys
    _selftest(sys.argv[1] if len(sys.argv) > 1 else "data_a_50_0.99.json")
