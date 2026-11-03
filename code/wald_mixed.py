"""
wald_mixed.py  -- mixed-order Wald entropy S^(1,1)  (Stage 3)
============================================================

From the action  S = (1/16pi) int [R - 1/2(grad phi)^2 - 1/2(grad theta)^2
                                    + alpha_GB phi G + alpha_CS theta *RR],
the Wald entropy is (derived, c_JM = -4 from Gauss-Codazzi):

    S = A/4 + (zeta_GB/2) I_sGB - (zeta_CS/4) I_dCS,
    I_sGB = oint phi~ R^(2) sqrt(h),   I_dCS = oint theta~ *R^{abcd} eps eps sqrt(h),

with zeta = alpha^2 and phi~, theta~ the source-normalized scalars.  Hence the
mixed-order piece:

    S^(1,1) = A^(1,1)/4 + (1/2) dI_sGB/dzeta_CS - (1/4) dI_dCS/dzeta_GB,

where the CS-derivative of I_sGB uses (h^(0,1), phi11) and the GB-derivative of
I_dCS uses (h^(1,0), theta11) -- the cached mixed metric + mixed scalars.

First-law check (across spin, fixed M): 0 = T dS/da + Omega_H dJ/da at each order,
so at O(zeta_GB zeta_CS):  dS^(1,1)/da  should equal  -(1/T^(0))[Omega^(1,1)
  + (T Omega derivative cross terms)].  We report the mixed Smarr residual
  M^(1,1) = 0 =? (1/4pi)[kappa S-pieces] + 2 J Omega^(1,1)  as the closure test.
"""

from __future__ import annotations
import numpy as np
from numpy.polynomial import chebyshev as C

from perturbation import assemble_h
from stage2_source import _geom, _bg_d2g
from wald_sgb import horizon_ricci, sgb_entropy_integral
from wald_dcs import _binormal_W


# ---- I_dCS on an arbitrary metric (perturbed *R via _geom) ------------ #
def dcs_integral_full(bg, gd, dg, d2g, theta_coef, nfit=6):
    g = bg.grid; z = g.z; chi = g.chi
    good = bg.interior
    guu = np.full_like(gd, np.nan)
    guu[good] = np.linalg.inv(gd[good])
    _, dual_dddd = _geom(bg, gd, dg, d2g)
    with np.errstate(all="ignore"):
        dual_uudd = np.einsum("...ai,...bj,...ijcd->...abcd", guu, guu, dual_dddd)
        # binormal from perturbed metric
        Om = -gd[..., 0, 3] / gd[..., 3, 3]
    Om_eq = np.array([np.interp(0, chi, Om[r, :]) for r in range(1, nfit + 1)])
    OmH = np.polyval(np.polyfit(z[1:nfit + 1], Om_eq, nfit - 1), 1.0)
    dxi = dg[..., :, :, 0] + OmH * dg[..., :, :, 3]
    K = 0.5 * (dxi - np.swapaxes(dxi, -1, -2))
    with np.errstate(all="ignore"):
        Kup = np.einsum("...ac,...bd,...cd->...ab", guu, guu, K)
        kap2 = -0.5 * np.einsum("...ab,...ab->...", K, Kup)
        eps = K / np.sqrt(kap2)[..., None, None]
        epsu = np.einsum("...ac,...bd,...cd->...ab", guu, guu, eps)
        W = np.einsum("...abef,...ab,...ef->...", dual_uudd, eps, epsu)
    # extrapolate W to horizon
    Wh = np.empty(len(chi))
    for j in range(len(chi)):
        col = W[1:nfit + 1, j]
        Wh[j] = (np.polyval(np.polyfit(z[1:nfit + 1], col, nfit - 1), 1.0)
                 if np.all(np.isfinite(col)) else np.nan)
    # sqrt(h) from this metric
    xf, _, sh_f, _ = horizon_ricci(bg, gd)
    th_h = g.to_grid(theta_coef)[0, :]
    ok = np.isfinite(Wh)
    c_W = C.chebfit(chi[ok], Wh[ok], min(np.sum(ok) - 1, g.Nchi, 20))
    c_th = C.chebfit(chi, th_h, min(g.Nchi, 20))
    Wf = C.chebval(xf, c_W); thf = C.chebval(xf, c_th)
    c_int = C.chebfit(xf, thf * Wf * sh_f, 26)
    return 2 * np.pi * (C.chebval(1.0, C.chebint(c_int)) - C.chebval(-1.0, C.chebint(c_int)))


def entropy_mixed(bg, H10, H01, H11, phi0, theta0, phi11, theta11, eps=1e-3):
    """Return A^(1,1)/4, dI_sGB/dzeta_CS, dI_dCS/dzeta_GB, and S^(1,1)."""
    from horizon_area import area_shifts
    g = bg.grid
    h10, dh10, d2h10 = assemble_h(bg, H10)
    h01, _, _ = assemble_h(bg, H01)
    d2g0 = _bg_d2g(bg)

    # A^(1,1)/4
    _, _, _, A11 = area_shifts(bg, H10, H01, H11, eps)
    A_term = A11 / 4.0

    # dI_sGB/dzeta_CS : perturb metric by h01 and scalar by phi11 (CS direction)
    def phic(s):  # source-normalized sGB scalar at CS-coupling s
        return {k: phi0 + s * phi11 for k in [0]}[0]
    with np.errstate(invalid="ignore"):
        IsGB_p = sgb_entropy_integral(bg, bg.g_dd + eps * h01, phi0 + eps * phi11)
        IsGB_m = sgb_entropy_integral(bg, bg.g_dd - eps * h01, phi0 - eps * phi11)
        IdCS_p = dcs_integral_full(bg, bg.g_dd + eps * h10, bg.dg_ddd + eps * dh10,
                               d2g0 + eps * d2h10, theta0 + eps * theta11)
        IdCS_m = dcs_integral_full(bg, bg.g_dd - eps * h10, bg.dg_ddd - eps * dh10,
                               d2g0 - eps * d2h10, theta0 - eps * theta11)
    dIsGB = (IsGB_p - IsGB_m) / (2 * eps)
    dIdCS = (IdCS_p - IdCS_m) / (2 * eps)

    S11 = A_term + 0.5 * dIsGB - 0.25 * dIdCS
    return A_term, 0.5 * dIsGB, -0.25 * dIdCS, S11


def _sweep(mixed_paths):
    import os
    from spectral_grid import SpectralGrid
    from mixed_cache import load_mixed
    print(f"{'a':>5} {'A11/4':>12} {'sGB-term':>12} {'dCS-term':>12} {'S^(1,1)':>13}")
    for p in mixed_paths:
        datap = os.path.basename(p).replace("mixed_a_", "data_a_")
        if not (os.path.exists(p) and os.path.exists(datap)):
            print(f"[skip] need {p} and {datap}"); continue
        bg, H10, H01, H11, phi11, theta11 = load_mixed(p)
        g, raw = SpectralGrid.from_file(datap)
        phi0 = bg.grid.fit(bg.grid.field_values(raw, "sGBPhi"), "even")
        theta0 = bg.grid.fit(bg.grid.field_values(raw, "dCSPhi"), "odd")
        At, sg, dc, S11 = entropy_mixed(bg, H10, H01, H11, phi0, theta0, phi11, theta11)
        print(f"{bg.a:>5.2f} {At:>12.4e} {sg:>12.4e} {dc:>12.4e} {S11:>13.5e}")


if __name__ == "__main__":
    import sys
    paths = sys.argv[1:] or [f"mixed_a_0.{s}.json" for s in
                             ["1", "2", "3", "4", "5", "6", "7", "8", "9"]]
    _sweep(paths)


def firstlaw_rhs(bg, H10, H01, H11, eps=1e-3):
    """[-2 pi Omega_H / kappa]^(1,1) -- the RHS the first law says dS^(1,1)/da equals."""
    from kappa_mixed import kappa_of_metric
    h10, dh10, _ = assemble_h(bg, H10)
    h01, dh01, _ = assemble_h(bg, H01)
    h11, dh11, _ = assemble_h(bg, H11)
    chi = bg.grid.chi; z = bg.grid.z

    def f(ca, cb):
        gd = bg.g_dd + ca * h10 + cb * h01 + ca * cb * h11
        dg = bg.dg_ddd + ca * dh10 + cb * dh01 + ca * cb * dh11
        with np.errstate(all="ignore"):
            Om = -gd[..., 0, 3] / gd[..., 3, 3]
        Om_eq = np.array([np.interp(0, chi, Om[r, :]) for r in range(1, 7)])
        OmH = np.polyval(np.polyfit(z[1:7], Om_eq, 5), 1.0)
        kap = float(np.interp(0, chi, kappa_of_metric(bg, gd, dg)))
        return -2 * np.pi * OmH / kap

    fpp, fpm = f(+eps, +eps), f(+eps, -eps)
    fmp, fmm = f(-eps, +eps), f(-eps, -eps)
    return (fpp - fpm - fmp + fmm) / (4 * eps * eps)


def firstlaw_sweep(mixed_paths):
    """Print a, S^(1,1), RHS^(1,1) per spin. Then check dS^(1,1)/da vs RHS^(1,1)
    by CENTRAL differences on the (fine, evenly-spaced) sweep -- the real test."""
    import os
    from spectral_grid import SpectralGrid
    from mixed_cache import load_mixed
    A = []
    print(f"{'a':>5} {'S^(1,1)':>13} {'RHS^(1,1)':>13}")
    for p in mixed_paths:
        datap = p.replace("mixed_a_", "data_a_")
        if not (os.path.exists(p) and os.path.exists(datap)):
            print(f"[skip] {p} exists={os.path.exists(p)}  {datap} exists={os.path.exists(datap)}")
            continue
            print(f"[skip] {p}"); continue
        bg, H10, H01, H11, phi11, theta11 = load_mixed(p)
        g, raw = SpectralGrid.from_file(datap)
        phi0 = bg.grid.fit(bg.grid.field_values(raw, "sGBPhi"), "even")
        theta0 = bg.grid.fit(bg.grid.field_values(raw, "dCSPhi"), "odd")
        _, _, _, S11 = entropy_mixed(bg, H10, H01, H11, phi0, theta0, phi11, theta11)
        rhs = firstlaw_rhs(bg, H10, H01, H11)
        A.append((bg.a, S11, rhs))
        print(f"{bg.a:>5.2f} {S11:>13.5e} {rhs:>13.5e}", flush=True)
    A.sort()
    if len(A) >= 3:
        a = np.array([r[0] for r in A]); S = np.array([r[1] for r in A])
        rhs = np.array([r[2] for r in A])
        print("\n--- first-law closure (central differences) ---")
        print(f"{'a':>5} {'dS/da':>12} {'RHS':>12} {'ratio':>8}")
        for i in range(1, len(a) - 1):
            dSda = (S[i + 1] - S[i - 1]) / (a[i + 1] - a[i - 1])
            print(f"{a[i]:>5.2f} {dSda:>12.3f} {rhs[i]:>12.3f} {dSda/rhs[i]:>8.3f}")


if __name__ == "__main__":
    firstlaw_sweep([f"mixed_a_0.{s}.json" for s in "123456789"])
