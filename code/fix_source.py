"""
fix_source.py -- complete mixed-order source and second-order peeling conditions.

Three corrections to the O(zeta_GB zeta_CS) pipeline (2026-10-05):

 1. Terms omitted by perturbation.gb_invariant_pert and stage2_source.{H_mixed, C_mixed, stress_cross}:
      Stage I   G^(0,1)  +=  -1/2 h GB^(0)
      H^(1,1)   +=  index-raising terms  +  all Ricci-dependent terms of the Gauss-Bonnet tensor
      C^(1,1)   +=  index-raising terms  +  the grad(delta Ricci) term
      T^(1,1)   +=  metric-perturbation part of T[phi0], T[theta0]
 2. The Ricci-dependent terms of H carry the signs of  H = -4 (**R)_{mrns} nabla^r nabla^s phi,
      -2 R f_mn + 2 g R box f - 4 R_mn box f + 4 R_m^r f_rn + 4 R_n^r f_rm - 4 g R^rs f_rs,
    which is the opposite of manuscript Eq. (A6).
 3. Peeling conditions: the pure-sector functions do not vanish at infinity (c = H3(inf) != 0), so keeping
    the ADM mass and angular momentum fixed at order zeta_GB*zeta_CS needs inhomogeneous right-hand sides
    [0, c10*c01, -c10*c01/2, -c10*c01/2] in the four conditions of Lam et al. Eq. (22).
    With zero right-hand sides the solution has dM = c10*c01/4 and dJ = -3 a c10*c01/4.

Checks passed with all three in place (a = 0.3, 0.5; 20x20 to 30x30):
    all six field-equation components, including r-chi, to ~1e-9 (was 3e-2 and 0.3);
    chi-spread of Omega_H^(1,1) ~1e-9 (was 2e-2);
    mixed Wald entropy equals the first-law (Smarr) value to 2e-4, the same level as the pure sectors.
"""
from __future__ import annotations
import numpy as np

from kerr_background import KerrBackground, _LC, _symbolic_kerr
from spectral_grid import _cheb_basis, _leg_basis
from perturbation import (assemble_h, gb_invariant_pert, pontryagin_pert, box_pert,
                          covariant_derivs_h)
from correction_tensors import covariant_hessian, H_sGB_leading, C_dCS_leading
from stage1_solver import solve_box0
from stage2_solver import COMPS
from stage2_source import (H_mixed, C_mixed, stress_cross, G2_bilinear, _R1, _dual_lo)


# ---------------------------------------------------------------------- #
# Stage I
# ---------------------------------------------------------------------- #
def gb_invariant_pert_full(bg, h, dh, d2h):
    """G^(1) = -1/2 h GB^(0) + 4 R^{mnrs} nabla_r nabla_n h_ms   (Ricci-flat background, D=4)."""
    htr = np.einsum("...mn,...mn->...", bg.g_uu, h)
    return gb_invariant_pert(bg, h, dh, d2h) - 0.5 * htr * bg.gb_scalar


def solve_mixed_scalars_full(bg, phi, theta, Hsg, Hcs):
    h_sg, dh_sg, d2h_sg = assemble_h(bg, Hsg)
    h_cs, dh_cs, d2h_cs = assemble_h(bg, Hcs)
    s_phi = -(box_pert(bg, h_cs, dh_cs, phi) + gb_invariant_pert_full(bg, h_cs, dh_cs, d2h_cs))
    s_th = -(box_pert(bg, h_sg, dh_sg, theta) + pontryagin_pert(bg, h_sg, dh_sg, d2h_sg))
    return solve_box0(bg, s_phi, "even"), solve_box0(bg, s_th, "odd")


# ---------------------------------------------------------------------- #
# Stage II pieces
# ---------------------------------------------------------------------- #
def _grad(bg, c):
    g = bg.grid
    out = np.zeros(bg.shape + (4,)); out[..., 1] = g.d_dr(c); out[..., 2] = g.d_dchi(c)
    return out


def T_metric_part(bg, c, h):
    """d/d(eps) of T[f](g0+eps h) at fixed f:  -1/2 h_mn (grad f)^2 + 1/2 g_mn h^{rs} f_r f_s."""
    gr = _grad(bg, c)
    huu = np.einsum("...ma,...nb,...ab->...mn", bg.g_uu, bg.g_uu, h)
    sq = np.einsum("...mn,...m,...n->...", bg.g_uu, gr, gr)
    hh = np.einsum("...mn,...m,...n->...", huu, gr, gr)
    return -0.5 * h * sq[..., None, None] + 0.5 * bg.g_dd * hh[..., None, None]


def H_extra(bg, phi10, h, dh, d2h, ricci_sign=+1.0):
    """Terms of the source part of H^(1,1) that H_mixed omits (raising + Ricci)."""
    guu, gdd = bg.g_uu, bg.g_dd
    huu = np.einsum("...ma,...nb,...ab->...mn", guu, guu, h)
    hess = covariant_hessian(bg, phi10)
    hess_ud = np.einsum("...ab,...bs->...as", hess, guu)
    raise_fix = -4 * (np.einsum("...mrns,...ra,...as->...mn", bg.Riemann_dddd, huu, hess_ud)
                      + np.einsum("...mrns,...sa,...ar->...mn", bg.Riemann_dddd, huu, hess_ud))
    dRic, dRs = _R1(bg, h, dh, d2h)                       # linearized Ricci tensor and scalar
    box0 = np.einsum("...mn,...mn->...", guu, hess)
    dRic_ud = np.einsum("...ma,...ar->...mr", dRic, guu)
    dRic_uu = np.einsum("...ra,...sb,...ab->...rs", guu, guu, dRic)
    # Ricci-dependent terms of H = -4 (**R)_{m r n s} nabla^r nabla^s phi:
    #   -2 R f_mn + 2 g_mn R box f - 4 R_mn box f + 4 R_m^r f_rn + 4 R_n^r f_rm - 4 g_mn R^rs f_rs
    # (trace = -4 G^{rs} f_rs).  NOTE: manuscript Eq. (A6) has the opposite sign on all six.
    ricci_fix = ricci_sign * (-2 * dRs[..., None, None] * hess + 2 * gdd * (dRs * box0)[..., None, None]
                 - 4 * dRic * box0[..., None, None]
                 + 4 * np.einsum("...mr,...rn->...mn", dRic_ud, hess)
                 + 4 * np.einsum("...nr,...rm->...mn", dRic_ud, hess)
                 - 4 * gdd * np.einsum("...rs,...rs->...", dRic_uu, hess)[..., None, None])
    return raise_fix + ricci_fix


def C_raise_extra(bg, th10, h):
    guu = bg.g_uu
    huu = np.einsum("...ma,...nb,...ab->...mn", guu, guu, h)
    hess = covariant_hessian(bg, th10)
    dual = _dual_lo(bg)
    hu = np.einsum("...bs,...sd->...bd", hess, guu)        # nabla_b nabla^d theta
    X = (-np.einsum("...ab,...bd,...amnd->...mn", huu, hu, dual)
         - np.einsum("...db,...ba,...amnd->...mn", huu, hu, dual))
    return -4 * 0.5 * (X + np.swapaxes(X, -1, -2))


# ---- on-shell first-order Ricci of the sGB sector, evaluated at arbitrary points ---- #
def _kerr_at(R, CHI, a):
    """Kerr g, g^-1, Gamma, Riemann_dddd at arbitrary (R, CHI) arrays (same algebra as KerrBackground)."""
    gL, dgL, d2gL = _symbolic_kerr()
    sh = R.shape
    def ev(L):
        o = np.zeros(sh + (4, 4))
        for i in range(4):
            for j in range(4):
                o[..., i, j] = np.broadcast_to(np.asarray(L[i][j](R, CHI, a), float), sh)
        return o
    g = ev(gL)
    dg = np.zeros(sh + (4, 4, 4)); dg[..., 1, :, :] = ev(dgL[1]); dg[..., 2, :, :] = ev(dgL[2])
    d2g = np.zeros(sh + (4, 4, 4, 4))
    d2g[..., 1, 1, :, :] = ev(d2gL[(1, 1)]); d2g[..., 2, 2, :, :] = ev(d2gL[(2, 2)])
    d2g[..., 1, 2, :, :] = ev(d2gL[(1, 2)]); d2g[..., 2, 1, :, :] = d2g[..., 1, 2, :, :]
    gi = np.linalg.inv(g)
    dgi = -np.einsum("...ma,...cab,...bn->...cmn", gi, dg, gi)
    Gam = 0.5 * (np.einsum("...ls,...mns->...lmn", gi, dg) + np.einsum("...ls,...nms->...lmn", gi, dg)
                 - np.einsum("...ls,...smn->...lmn", gi, dg))
    dGam = 0.5 * (np.einsum("...cls,...mns->...clmn", dgi, dg) + np.einsum("...cls,...nms->...clmn", dgi, dg)
                  - np.einsum("...cls,...smn->...clmn", dgi, dg) + np.einsum("...ls,...cmns->...clmn", gi, d2g)
                  + np.einsum("...ls,...cnms->...clmn", gi, d2g) - np.einsum("...ls,...csmn->...clmn", gi, d2g))
    Ru = (np.einsum("...mrns->...rsmn", dGam) - np.einsum("...nrms->...rsmn", dGam)
          + np.einsum("...rml,...lns->...rsmn", Gam, Gam) - np.einsum("...rnl,...lms->...rsmn", Gam, Gam))
    Rd = np.einsum("...ar,...rbcd->...abcd", g, Ru)
    return g, gi, Gam, Rd


def _field_at(grid, C, z, chi):
    """f, f_r, f_c, f_rr, f_cc, f_rc on the tensor grid (z, chi), arbitrary points."""
    Tz = _cheb_basis(z, grid.Nz); Pl = _leg_basis(chi, grid.Nchi)
    Dz, Dc = grid._Dz, grid._Dchi
    ev = lambda M: Tz @ M @ Pl.T
    zr = (-(1 + z) ** 2 / (2 * grid.r_plus))[:, None]
    zrr = ((1 + z) ** 3 / (2 * grid.r_plus ** 2))[:, None]
    f = ev(C); fz = ev(Dz @ C); fzz = ev(Dz @ Dz @ C)
    fc = ev(C @ Dc.T); fcc = ev(C @ Dc.T @ Dc.T); fzc = ev(Dz @ C @ Dc.T)
    return f, zr * fz, fc, zr ** 2 * fzz + zrr * fz, fcc, zr * fzc


def _ricci10_onshell_at(grid, phi10, z, chi):
    """delta R^(1,0)_{mu nu} = 1/2 d_mu phi d_nu phi - H^(1,0)_{mu nu}, at arbitrary points; also Gamma."""
    r = 2 * grid.r_plus / (1 + z)
    R, CH = np.meshgrid(r, chi, indexing="ij")
    g, gi, Gam, Rd = _kerr_at(R, CH, grid.spin)
    f, fr, fc, frr, fcc, frc = _field_at(grid, phi10, z, chi)
    grad = np.zeros(R.shape + (4,)); grad[..., 1] = fr; grad[..., 2] = fc
    d2 = np.zeros(R.shape + (4, 4)); d2[..., 1, 1] = frr; d2[..., 2, 2] = fcc; d2[..., 1, 2] = d2[..., 2, 1] = frc
    hess = d2 - np.einsum("...lmn,...l->...mn", Gam, grad)
    hess_uu = np.einsum("...ma,...nb,...ab->...mn", gi, gi, hess)
    H = 4.0 * np.einsum("...mrns,...rs->...mn", Rd, hess_uu)
    return 0.5 * np.einsum("...m,...n->...mn", grad, grad) - H, Gam


def C_gradricci(bg, th10, phi10, delta=1e-5, sign=+1.0):
    """
    -4 sym_{mu nu}[ (d_sigma theta) eps^{sigma delta alpha}{}_mu  nabla_alpha dR_{nu delta} ],
    dR = first-order sGB Ricci tensor (on shell).  Interior nodes only.
    """
    g = bg.grid
    zi = g.z[1:-1]; ci = g.chi[1:-1]                      # interior nodes
    ri = 2 * g.r_plus / (1 + zi)
    X0, Gam = _ricci10_onshell_at(g, phi10, zi, ci)
    zp = 2 * g.r_plus / (ri + delta) - 1; zm = 2 * g.r_plus / (ri - delta) - 1
    dXr = (_ricci10_onshell_at(g, phi10, zp, ci)[0] - _ricci10_onshell_at(g, phi10, zm, ci)[0]) / (2 * delta)
    dXc = (_ricci10_onshell_at(g, phi10, zi, ci + delta)[0] - _ricci10_onshell_at(g, phi10, zi, ci - delta)[0]) / (2 * delta)
    dX = np.zeros(X0.shape[:2] + (4, 4, 4)); dX[..., 1, :, :] = dXr; dX[..., 2, :, :] = dXc
    DX = (dX - np.einsum("...lan,...ld->...and", Gam, X0)        # nabla_a X_{n d}
          - np.einsum("...lad,...nl->...and", Gam, X0))
    out = np.full(bg.shape + (4, 4), np.nan)
    gth = _grad(bg, th10)[1:-1, 1:-1]
    gdd = bg.g_dd[1:-1, 1:-1]
    eps_u = (1.0 / bg.sqrt_mg[1:-1, 1:-1])[..., None, None, None, None] * _LC
    eps_m = np.einsum("...sdal,...lm->...sdam", eps_u, gdd)       # eps^{s d a}{}_m
    Y = np.einsum("...s,...sdam,...and->...mn", gth, eps_m, DX)
    out[1:-1, 1:-1] = sign * (-4.0) * 0.5 * (Y + np.swapaxes(Y, -1, -2))
    return out


# ---------------------------------------------------------------------- #
def build_source_full(bg, phi10, th10, phi11, th11, Hsg, Hcs, gradricci_sign=+1.0, parts=False, ricci_sign=+1.0):
    h10, dh10, d2h10 = assemble_h(bg, Hsg)
    h01, dh01, d2h01 = assemble_h(bg, Hcs)
    T = 0.5 * (stress_cross(bg, phi10, phi11) + T_metric_part(bg, phi10, h01)
               + stress_cross(bg, th10, th11) + T_metric_part(bg, th10, h10))
    H11 = H_mixed(bg, phi10, phi11, h01, dh01, d2h01) + H_extra(bg, phi10, h01, dh01, d2h01, ricci_sign=ricci_sign)
    C11 = C_mixed(bg, th10, th11, h10, dh10, d2h10) + C_raise_extra(bg, th10, h10)
    if gradricci_sign != 0.0:
        C11 = C11 + C_gradricci(bg, th10, phi10, sign=gradricci_sign)
    G2 = G2_bilinear(bg, (h10, dh10, d2h10), (h01, dh01, d2h01))
    S = T - H11 - C11 - 2.0 * G2
    return (S, dict(T=T, H=H11, C=C11, G2=G2)) if parts else S


# ---------------------------------------------------------------------- #
# second-order peeling conditions, solve, and ADM diagnostics
# ---------------------------------------------------------------------- #
def peeling_rhs(bg, H10, H01):
    """Inhomogeneous terms of the peeling conditions at order zeta_GB*zeta_CS that keep the ADM mass and
    angular momentum fixed.  Exact conditions (M = 1), with c = H3(inf) = H4(inf) and H^(1) the 1/r coefficient:
        H1(inf) = 0,   (1 + H2(inf)) sqrt(1+c) = 1,   (2 + H3^(1)/(1+c)) sqrt(1+c) = 2.
    Their zeta_GB*zeta_CS coefficients are the linear conditions of Lam et al. with right-hand sides
        [0, c10*c01, -c10*c01/2, -c10*c01/2]."""
    g = bg.grid
    c10 = float(np.mean(g.to_grid(H10[3])[-1, :])); c01 = float(np.mean(g.to_grid(H01[3])[-1, :]))
    return [0.0, c10 * c01, -0.5 * c10 * c01, -0.5 * c10 * c01], c10, c01

def solve_metric_bc(bg, S, A, cols, bc):
    """Same equilibrated least-squares solve as stage2_solver.solve_mixed_metric, with BC right-hand sides."""
    g = bg.grid; intr = bg.interior; nb = g.Nchi + 1
    S = np.where(np.isnan(S), 0.0, S)
    b = np.concatenate([np.concatenate([S[..., a, c][intr] for (a, c) in COMPS])] + [np.full(nb, v) for v in bc])
    rw = 1.0 / np.maximum(np.max(np.abs(A), axis=1), 1e-300)
    Aw = A * rw[:, None]; bw = b * rw
    cw = 1.0 / np.maximum(np.linalg.norm(Aw, axis=0), 1e-300)
    sol, *_ = np.linalg.lstsq(Aw * cw[None, :], bw, rcond=None)
    sol = sol * cw
    Hc = {i: np.zeros((g.Nz + 1, g.Nchi + 1)) for i in (1, 2, 3, 4)}
    for (i, n, l), v in zip(cols, sol):
        Hc[i][n, l] = v
    return Hc

def adm_shifts(bg, H10, H01, H11):
    """delta M and delta J/(a) at order zeta_GB*zeta_CS implied by the asymptotics of the solution (M = 1)."""
    g = bg.grid; rp = g.r_plus
    v = lambda H, i: float(np.mean(g.to_grid(H[i])[-1, :])); d = lambda H, i: float(np.mean(2 * rp * g.d_dz(H[i])[-1, :]))
    c10, c01, c11 = v(H10, 3), v(H01, 3), v(H11, 3)
    dM_rr = 0.5 * (c11 + d(H11, 3) + 0.5 * c10 * c01)
    dM_tt = 0.5 * d(H11, 1) + 0.5 * c11 + 0.25 * (d(H10, 1) * c01 + d(H01, 1) * c10) - 0.25 * c10 * c01
    dJ = v(H11, 2) + 0.5 * c11 + 0.5 * (v(H10, 2) * c01 + v(H01, 2) * c10) - 0.25 * c10 * c01
    return dM_rr, dM_tt, dJ
