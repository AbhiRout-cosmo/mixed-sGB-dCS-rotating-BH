"""
wald_sgb.py  -- sGB-sector Wald entropy (Stage 1 of the entropy calculation)
============================================================================

For scalar-Gauss-Bonnet gravity the Wald entropy is
    S_sGB = A/4G + c_GB * oint_H  theta_sGB * R^(2) sqrt(h) dchi dphi,
where R^(2) is the INTRINSIC Ricci scalar of the horizon 2-surface and
sqrt(h) its area element.

Validation gate (coupling-independent): by Gauss-Bonnet,
    oint_H R^(2) sqrt(h) dchi dphi = 8 pi   (topological, 2-sphere),
i.e.  integral_{-1}^{+1} R^(2) sqrt(h) dchi = 4.
This validates the R^(2) + sqrt(h) machinery with NO reference to the coupling.

Horizon 2-metric (diagonal, circular):  h_chichi = g_chichi|r+, h_phiphi = g_phiphi|r+.
For a diagonal 2-metric depending only on chi, the Gaussian curvature is
    K = -1/(2 sqrt(h)) d/dchi ( (d h_phiphi/dchi) / sqrt(h) ),   R^(2) = 2K,
using sqrt(h) = sqrt(h_chichi h_phiphi) (so h_chichi enters only through sqrt(h),
avoiding its 1/(1-chi^2) pole).
"""

from __future__ import annotations
import numpy as np
from numpy.polynomial import chebyshev as C


def _even_fit(x, y, deg):
    """Chebyshev fit; y is even in chi so we still fit full and rely on symmetry."""
    return C.chebfit(x, y, deg)


def horizon_ricci(bg, gdd, nfine=4000, deg=None):
    """Return (chi_fine, R2, sqrt_h) on a fine grid, plus the Gauss-Bonnet integral."""
    g = bg.grid
    chi = g.chi
    with np.errstate(invalid="ignore", divide="ignore"):
        prod = gdd[0, :, 2, 2] * gdd[0, :, 3, 3]           # h_chichi * h_phiphi
        sh = np.sqrt(np.abs(prod))
    hpp = gdd[0, :, 3, 3]                                   # h_phiphi (regular)
    ok = np.isfinite(sh) & np.isfinite(hpp)
    ci, shi, hi = chi[ok], sh[ok], hpp[ok]
    order = np.argsort(ci)
    ci, shi, hi = ci[order], shi[order], hi[order]
    deg = deg or min(len(ci) - 1, g.Nchi, 24)
    c_sh = _even_fit(ci, shi, deg)
    c_hpp = _even_fit(ci, hi, deg)

    xf = np.linspace(-0.999, 0.999, nfine)
    sh_f = C.chebval(xf, c_sh)
    dhpp = C.chebval(xf, C.chebder(c_hpp))                 # d h_phiphi / dchi
    g1 = dhpp / sh_f
    c_g1 = C.chebfit(xf, g1, deg + 2)
    dg1 = C.chebval(xf, C.chebder(c_g1))
    K = -dg1 / (2.0 * sh_f)
    R2 = 2.0 * K
    c_int = C.chebfit(xf, R2 * sh_f, deg + 2)              # integrate exactly
    GB = C.chebval(1.0, C.chebint(c_int)) - C.chebval(-1.0, C.chebint(c_int))
    return xf, R2, sh_f, GB


def sgb_entropy_integral(bg, gdd, theta_coef):
    """oint theta_sGB R^(2) sqrt(h) dchi dphi  (= 2 pi * integral over chi)."""
    xf, R2, sh_f, _ = horizon_ricci(bg, gdd)
    # theta_sGB on the horizon, evaluated on xf
    g = bg.grid
    th_h = g.to_grid(theta_coef)[0, :]                      # horizon row over chi nodes
    c_th = C.chebfit(g.chi, th_h, min(g.Nchi, 24))
    th_f = C.chebval(xf, c_th)
    c_int = C.chebfit(xf, th_f * R2 * sh_f, min(len(xf) - 1, 26))
    I = C.chebval(1.0, C.chebint(c_int)) - C.chebval(-1.0, C.chebint(c_int))
    return 2.0 * np.pi * I


# ====================================================================== #
def _selftest():
    from spectral_grid import SpectralGrid
    from kerr_background import KerrBackground
    print("Gauss-Bonnet gate: oint R^(2) sqrt(h) dchi dphi  should = 8 pi = %.6f" % (8*np.pi))
    for a in (0.1, 0.5, 0.8, 0.9):
        g = SpectralGrid(Nz=40, Nchi=40, spin=a); bg = KerrBackground(g)
        _, _, _, GB = horizon_ricci(bg, bg.g_dd)
        print(f"  a={a}: integral_[-1,1] R^(2) sqrt(h) dchi = {GB:.6f}  "
              f"(target 4)   oint = {2*np.pi*GB:.5f} (target {8*np.pi:.5f})")


def _sweep(mixed_paths):
    """I_sGB = oint theta_sGB R^(2) sqrt(h) on the horizon, per spin.
    Takes mixed_a_0_X.json (grid/spin); reads theta_sGB from data_a_0_X.json.
    a->0 should approach 8 pi * theta_sGB (static Gauss-Bonnet form)."""
    import os
    from spectral_grid import SpectralGrid
    from kerr_background import KerrBackground
    from mixed_cache import load_mixed

    print(f"{'a':>5} {'I_sGB':>14} {'GB-gate':>9} {'theta_eq':>9}  "
          f"{'8pi*theta_eq':>12}")
    for p in mixed_paths:
        datap = os.path.basename(p).replace("mixed_a_", "data_a_")
        if not (os.path.exists(p) and os.path.exists(datap)):
            print(f"[skip] need both {p} and {datap}"); continue
        bg, H10, H01, H11, _, _ = load_mixed(p)
        g, raw = SpectralGrid.from_file(datap)
        th = bg.grid.fit(bg.grid.field_values(raw, "sGBPhi"), "even")
        _, _, _, GB = horizon_ricci(bg, bg.g_dd)
        I = sgb_entropy_integral(bg, bg.g_dd, th)
        th_eq = float(np.interp(0, bg.grid.chi, bg.grid.to_grid(th)[0, :]))
        print(f"{bg.a:>5.2f} {I:>14.6e} {GB:>9.5f} {th_eq:>9.4f}  "
              f"{8*np.pi*th_eq:>12.4f}")


if __name__ == "__main__":
    import sys
    _selftest()
    print()
    paths = sys.argv[1:] or [f"mixed_a_0.{s}.json" for s in
                             ["1", "2", "3", "4", "5", "6", "7", "8", "9"]]
    _sweep(paths)
