"""
wald_dcs.py  -- dCS-sector Wald entropy (Stage 2)
=================================================

For dynamical Chern-Simons gravity the Wald entropy carries the extra term
    S_dCS = A/4G  - 2 pi alpha  oint_H  theta_dCS  *R^{abcd} eps_ab eps_cd  sqrt(h),
with *R^{abcd} the dual Riemann tensor and eps_ab the horizon binormal
(eps_ab eps^ab = -2).

Binormal (regular form):  eps_ab = K_ab / kappa,
    K_ab = 1/2 (d_a xi_b - d_b xi_a),   xi = d_t + Omega_H d_phi,
    kappa^2 = -1/2 K_ab K^ab.
Then automatically eps_ab eps^ab = -2.  We evaluate the SCALAR
    W = *R^{abcd} eps_ab eps_cd
on interior rows (all regular, non-degenerate) and extrapolate to the horizon
z=+1 per chi -- the same strategy validated for the surface gravity.

Validation (no external gate for dCS):
  * I_dCS -> 0 as a -> 0  (theta_dCS ~ a).
  * smooth, monotone a-scaling.
  * ultimate check is first-law closure (Stage 3).
"""

from __future__ import annotations
import numpy as np
from numpy.polynomial import chebyshev as C
from wald_sgb import horizon_ricci      # reuse validated sqrt(h)


def _binormal_W(bg, gdd, dg, guu, nfit=6):
    """W(chi) = *R^{abcd} eps_ab eps_cd at the horizon (extrapolated), per chi."""
    g = bg.grid; z = g.z; chi = g.chi
    # horizon angular velocity (chi-independent): extrapolate equator to z=1
    with np.errstate(all="ignore"):
        Om = -gdd[..., 0, 3] / gdd[..., 3, 3]
    Om_eq = np.array([np.interp(0, chi, Om[r, :]) for r in range(1, nfit + 1)])
    OmH = np.polyval(np.polyfit(z[1:nfit + 1], Om_eq, nfit - 1), 1.0)

    # xi_b = g_bt + OmH g_bphi ; d_a xi_b = dg[..,a,b,t] + OmH dg[..,a,b,phi]
    dxi = dg[..., :, :, 0] + OmH * dg[..., :, :, 3]        # [...,a,b]
    K = 0.5 * (dxi - np.swapaxes(dxi, -1, -2))             # K_ab (antisym)
    with np.errstate(all="ignore"):
        Kup = np.einsum("...ac,...bd,...cd->...ab", guu, guu, K)   # K^{ab}
        kap2 = -0.5 * np.einsum("...ab,...ab->...", K, Kup)        # kappa^2
        eps_dn = K / np.sqrt(kap2)[..., None, None]               # eps_ab = K_ab/kappa
        eps_up = np.einsum("...ac,...bd,...cd->...ab", guu, guu, eps_dn)  # eps^{ab}
        # W = *R^{ab}_{ef} eps_ab eps^{ef}
        W = np.einsum("...abef,...ab,...ef->...", bg.dualRiemann_uudd, eps_dn, eps_up)

    # extrapolate W(z,chi) to z=+1 per chi from interior rows
    out = np.empty(len(chi))
    for j in range(len(chi)):
        col = W[1:nfit + 1, j]
        if np.all(np.isfinite(col)):
            out[j] = np.polyval(np.polyfit(z[1:nfit + 1], col, nfit - 1), 1.0)
        else:
            out[j] = np.nan
    return out


def dcs_entropy_integral(bg, gdd, dg, guu, theta_coef):
    """oint theta_dCS W sqrt(h) dchi dphi  (= 2 pi * integral over chi)."""
    g = bg.grid
    xf, _, sh_f, _ = horizon_ricci(bg, gdd)
    W = _binormal_W(bg, gdd, dg, guu)
    th_h = g.to_grid(theta_coef)[0, :]
    chi = g.chi
    ok = np.isfinite(W)
    c_W = C.chebfit(chi[ok], W[ok], min(np.sum(ok) - 1, g.Nchi, 20))
    c_th = C.chebfit(chi, th_h, min(g.Nchi, 20))
    Wf = C.chebval(xf, c_W); thf = C.chebval(xf, c_th)
    c_int = C.chebfit(xf, thf * Wf * sh_f, 26)
    I = C.chebval(1.0, C.chebint(c_int)) - C.chebval(-1.0, C.chebint(c_int))
    return 2.0 * np.pi * I


# ====================================================================== #
def _sweep(mixed_paths):
    import os
    from spectral_grid import SpectralGrid
    from kerr_background import KerrBackground
    from mixed_cache import load_mixed
    print(f"{'a':>5} {'I_dCS':>14} {'theta_dCS_off':>13}   (I_dCS -> 0 as a->0)")
    for p in mixed_paths:
        datap = os.path.basename(p).replace("mixed_a_", "data_a_")
        if not (os.path.exists(p) and os.path.exists(datap)):
            print(f"[skip] need {p} and {datap}"); continue
        bg, H10, H01, H11, _, _ = load_mixed(p)
        g, raw = SpectralGrid.from_file(datap)
        th = bg.grid.fit(bg.grid.field_values(raw, "dCSPhi"), "odd")
        I = dcs_entropy_integral(bg, bg.g_dd, bg.dg_ddd, bg.g_uu, th)
        # dCS scalar is odd in chi -> sample off-equator
        th_off = float(np.interp(0.5, bg.grid.chi, bg.grid.to_grid(th)[0, :]))
        print(f"{bg.a:>5.2f} {I:>14.6e} {th_off:>13.5f}")


if __name__ == "__main__":
    import sys
    paths = sys.argv[1:] or [f"mixed_a_0.{s}.json" for s in
                             ["1", "2", "3", "4", "5", "6", "7", "8", "9"]]
    _sweep(paths)
