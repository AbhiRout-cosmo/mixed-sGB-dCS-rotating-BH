"""
kappa_horizon.py -- surface gravity evaluated on the horizon, with no extrapolation.

For the circular metric with g_{r chi} = 0 and the horizon at r = r_+, where
D = g_tphi^2 - g_tt g_phiphi vanishes, the lapse is N^2 = D / g_phiphi and
    kappa^2 = (1/4) g^{rr} (d_r N^2)^2 / N^2  ->  kappa = (1/2) sqrt( Delta'(r_+) d_r D / [ g_phiphi (1 + H_3) Sigma ] )   at r = r_+ ,
since g^{rr} = Delta / [(1 + H_3) Sigma].  Everything is evaluated on the horizon row of the spectral grid
(H_i and d_r H_i there are exact spectral values), so there is no near-horizon fit.
kappa_mixed.kappa_of_metric instead extrapolates kappa^2 from the six nodes next to the horizon; that
extrapolation is what limits it at high spin.

    kappa_coeffs(bg, H10, H01, H11) -> kappa^(1,0), kappa^(0,1), kappa^(1,1) as arrays over chi
    horizon_defect(bg, H10, H01, H11) -> max |D| / scale on the horizon at each order (should be round-off)

Table for a set of caches, in the same layout as kappa_mixed.py (equator value; chi-spread over nodes [3:-3]):

    python kappa_horizon.py                          all spins a = 0.10 ... 0.90, mixed_a_0.X.json in this folder
    python kappa_horizon.py mixed_a_0.5.json ...     selected files

The coupling step is eps = 1e-4: with 1e-3 the O(eps^2) truncation reaches 1e-4 of kappa^(1,1) at a = 0.9.
"""
import numpy as np


def _kerr_tphi(r, chi, a):
    S = r * r + a * a * chi * chi; s2 = 1 - chi * chi
    return -(1 - 2 * r / S), -2 * a * r * s2 / S, s2 * (r * r + a * a + 2 * a * a * r * s2 / S)


def _pieces(bg, H, ca_cb=None):
    g = bg.grid; rp = g.r_plus; a = bg.a; chi = g.chi
    h = 1e-30                                              # complex step: exact r-derivatives of the Kerr functions
    f = _kerr_tphi(rp + 1j * h, chi.astype(complex), a)
    g0 = [x.real for x in f]; d0 = [x.imag / h for x in f]
    val = {i: g.to_grid(H[i])[0, :] for i in (1, 2, 3, 4)}
    der = {i: g.d_dr(H[i])[0, :] for i in (1, 2, 3, 4)}
    return g0, d0, val, der


def _kappa(bg, Hs, ca, cb):
    g = bg.grid; rp = g.r_plus; a = bg.a; chi = g.chi
    H = {i: ca * Hs[0][i] + cb * Hs[1][i] + ca * cb * Hs[2][i] for i in (1, 2, 3, 4)}
    (tt0, tf0, ff0), (dtt0, dtf0, dff0), v, d = _pieces(bg, H)
    tt = tt0 + v[1]; tf = tf0 * (1 + v[2]); ff = ff0 * (1 + v[4])
    dtt = dtt0 + d[1]; dtf = dtf0 * (1 + v[2]) + tf0 * d[2]; dff = dff0 * (1 + v[4]) + ff0 * d[4]
    D = tf * tf - tt * ff
    dD = 2 * tf * dtf - dtt * ff - tt * dff
    Sig = rp * rp + a * a * chi * chi
    with np.errstate(all="ignore"):
        kap = 0.5 * np.sqrt(2 * (rp - 1) * dD / (ff * (1 + v[3]) * Sig))
        defect = D / ff
    return kap, defect


def kappa_coeffs(bg, H10, H01, H11, eps=1e-4):
    Hs = (H10, H01, H11)
    k = {s: _kappa(bg, Hs, s[0] * eps, s[1] * eps)[0] for s in ((1, 1), (1, -1), (-1, 1), (-1, -1))}
    k10 = (k[1, 1] + k[1, -1] - k[-1, 1] - k[-1, -1]) / (4 * eps)
    k01 = (k[1, 1] - k[1, -1] + k[-1, 1] - k[-1, -1]) / (4 * eps)
    k11 = (k[1, 1] - k[1, -1] - k[-1, 1] + k[-1, -1]) / (4 * eps * eps)
    return k10, k01, k11


def horizon_defect(bg, H10, H01, H11, eps=1e-4):
    """Coefficients of N^2 = D/g_phiphi on the horizon at orders (1,0), (0,1), (1,1); zero if the horizon stays at r_+."""
    Hs = (H10, H01, H11)
    d = {s: _kappa(bg, Hs, s[0] * eps, s[1] * eps)[1] for s in ((1, 1), (1, -1), (-1, 1), (-1, -1))}
    d10 = (d[1, 1] + d[1, -1] - d[-1, 1] - d[-1, -1]) / (4 * eps)
    d01 = (d[1, 1] - d[1, -1] + d[-1, 1] - d[-1, -1]) / (4 * eps)
    d11 = (d[1, 1] - d[1, -1] - d[-1, 1] + d[-1, -1]) / (4 * eps * eps)
    return d10, d01, d11


def run_one(path, eps=1e-4):
    """Same return layout as kappa_mixed.run_one: spin, (kappa11, spread), (kappa10, spread), (kappa01, spread)."""
    from mixed_cache import load_mixed
    bg, H10, H01, H11, _, _ = load_mixed(path)
    chi = bg.grid.chi
    k10, k01, k11 = kappa_coeffs(bg, H10, H01, H11, eps)
    inner = slice(3, -3)                       # drop pole columns for the error bar

    def eqspread(v):
        seg = v[inner]; seg = seg[np.isfinite(seg)]
        return float(np.interp(0, chi, v)), float(np.max(seg) - np.min(seg))
    return bg.grid.spin, eqspread(k11), eqspread(k10), eqspread(k01)


if __name__ == "__main__":
    import sys
    paths = sys.argv[1:] or [f"mixed_a_{f'{a:.2f}'.rstrip('0')}.json" for a in np.arange(0.10, 0.901, 0.02)]
    print(f"{'a':>5}  {'kappa^(1,1)':>14} {'chi-spread':>11}  "
          f"{'kappa^(1,0)_sGB':>15} {'kappa^(0,1)_dCS':>15}")
    for p in paths:
        try:
            a, (k11, s11), (k10, _), (k01, _) = run_one(p)
            print(f"{a:>5.2f}  {k11:>+14.6e} {s11:>11.2e}  "
                  f"{k10:>+15.6e} {k01:>+15.6e}")
        except FileNotFoundError:
            print(f"[skip] {p} not found")
