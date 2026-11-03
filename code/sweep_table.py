"""
sweep_table.py -- observables and consistency checks from the corrected mixed caches.

    cd data && python ../code/sweep_table.py            (every mixed_a_*.json with a matching data_a_*.json)
    cd data && python ../code/sweep_table.py mixed_a_0.5.json ...

Writes sweep_corrected.csv and sweep_corrected.json, one row per spin. Needs kappa_horizon.py and lyapunov_fix.py.

Columns (X^(0), X^(1,0), X^(0,1), X^(1,1) are written X0, X10, X01, X11; all at fixed M = 1 and fixed J):
  even_residual_rel, rchi_residual_rel   field-equation residuals of the mixed solution (from mixed_cache_fixed)
  dM_grr, dM_gtt, dJ_over_a              ADM shifts of the mixed solution (zero if the peeling conditions hold)
  c10, c01                               H_3^(1,0), H_3^(0,1) at infinity
  Om*                                    horizon angular velocity; Om11 = Om11_linear + Om11_cross; Om11_spread = chi-spread
  k*                                     surface gravity, evaluated on the horizon; k11_spread = chi-spread
  k*_extrap                              the same from kappa_mixed.py (near-horizon extrapolation), for reference
  A*                                     horizon area
  risco*, rph*, Omc*, lam*  (_pro/_ret)  ISCO radius, photon-orbit radius and frequency, Lyapunov exponent.
                                         Radii are coordinate radii in Lam's r.  Coefficients are Richardson-extrapolated
                                         in the coupling step; *_eps1e-3 is the plain eps = 1e-3 value.
  S10/S01/S11 _wald, _fl                 Wald entropy and the first-law (Smarr) value
  maxH11, phi11_hor_eq, theta11_hor_half size of the mixed fields
"""
import glob, json, re, sys, warnings, csv
import numpy as np
warnings.filterwarnings("ignore")
from spectral_grid import SpectralGrid
from mixed_cache import load_mixed
from mixed_observables import omega_mixed
from kappa_mixed import kappa_mixed
from horizon_area import area_shifts
from isco_photon import EqMetric, r_isco, r_photon, mixed_shift
from qnm_eikonal import _fd, omega_c_scalar
from lyapunov_fix import lyapunov          # 5-point R'' stencil; see lyapunov_fix.py
from wald_sgb import sgb_entropy_integral
from wald_dcs import dcs_entropy_integral
from wald_mixed import entropy_mixed
from kappa_horizon import kappa_coeffs, _kappa


def first_law_entropy(bg, H10, H01, H11, eps=1e-4):
    """s^(1,0), s^(0,1), s^(1,1) from the Smarr relation, with kappa evaluated on the horizon (kappa_horizon.py)."""
    g = bg.grid; chi = g.chi; a, rp = bg.a, g.r_plus
    hz = lambda H, i: float(np.interp(0, chi, g.to_grid(H[i])[0]))
    def f(ca, cb):
        kap = float(np.interp(0, chi, _kappa(bg, (H10, H01, H11), ca, cb)[0]))
        H2 = ca * hz(H10, 2) + cb * hz(H01, 2) + ca * cb * hz(H11, 2)
        H4 = ca * hz(H10, 4) + cb * hz(H01, 4) + ca * cb * hz(H11, 4)
        return np.array([2 * np.pi * a / (2 * rp) * (1 + H2) / (1 + H4) / kap, 2 * np.pi / kap])
    pp, pm, mp, mm = f(eps, eps), f(eps, -eps), f(-eps, eps), f(-eps, -eps)
    d10 = (pp + pm - mp - mm) / (4 * eps); d01 = (pp - pm + mp - mm) / (4 * eps); d11 = (pp - pm - mp + mm) / (4 * eps * eps)
    return (2 * a * d10[0] - d10[1]) / 2, (2 * a * d01[0] - d01[1]) / 2, (2 * a * d11[0] - d11[1]) / 6


def richardson(fn, eps=1e-3):
    """Coupling-expansion coefficients with the O(eps^2) truncation term removed: (4 f(eps) - f(2 eps)) / 3.
    Returns (values, plain eps values) so the size of the correction is visible."""
    v1 = np.array(fn(eps), float); v2 = np.array(fn(2 * eps), float)
    out = (4 * v1 - v2) / 3; out[0] = v1[0]
    return out, v1


def row(path):
    meta = json.load(open(path))
    bg, H10, H01, H11, p11, t11 = load_mixed(path)
    g = bg.grid; chi = g.chi; a, rp = bg.a, g.r_plus
    gd, raw = SpectralGrid.from_file(path.replace("mixed_a_", "data_a_"))
    p10 = g.fit(g.field_values(raw, "sGBPhi"), "even"); t10 = g.fit(g.field_values(raw, "dCSPhi"), "odd")
    eq = lambda f: float(np.interp(0, chi, f))
    sp = lambda f: float(np.max(f[3:-3]) - np.min(f[3:-3]))
    o = {"a": float(a), "N": int(g.Nz)}
    for k in ("even_residual_rel", "rchi_residual_rel", "dM_grr", "dM_gtt", "dJ_over_a", "c10", "c01"):
        o[k] = meta.get(k)
    om = omega_mixed(bg, H10, H01, H11)
    lin = a / (2 * rp) * (g.to_grid(H11[2])[0] - g.to_grid(H11[4])[0])
    o["Om0"] = a / (2 * rp)
    o["Om10"] = eq(a / (2 * rp) * (g.to_grid(H10[2])[0] - g.to_grid(H10[4])[0]))
    o["Om01"] = eq(a / (2 * rp) * (g.to_grid(H01[2])[0] - g.to_grid(H01[4])[0]))
    o["Om11"] = eq(om); o["Om11_spread"] = sp(om); o["Om11_linear"] = eq(lin); o["Om11_cross"] = o["Om11"] - o["Om11_linear"]
    k10, k01, k11 = kappa_coeffs(bg, H10, H01, H11, eps=1e-4)          # on the horizon, no extrapolation
    o.update(k0=float(np.sqrt(1 - a * a) / (2 * rp)), k10=eq(k10), k01=eq(k01), k11=eq(k11), k11_spread=sp(k11))
    e10, e01, e11 = kappa_mixed(bg, H10, H01, H11)                     # original near-horizon extrapolation, for reference
    o.update(k10_extrap=eq(e10), k01_extrap=eq(e01), k11_extrap=eq(e11), k11_extrap_spread=sp(e11))
    A0, A10, A01, A11 = area_shifts(bg, H10, H01, H11)
    o.update(A0=float(A0), A10=float(A10), A01=float(A01), A11=float(A11))
    em = EqMetric(bg, H10, H01, H11)
    for pro, tag in ((True, "pro"), (False, "ret")):
        for nm, fn in (("risco", r_isco), ("rph", r_photon)):
            try:
                v, v1 = richardson(lambda e: mixed_shift(em, fn, pro, eps=e))
            except Exception:
                v = v1 = [np.nan] * 4
            for lab, x in zip(("0", "10", "01", "11"), v): o[f"{nm}{lab}_{tag}"] = float(x)
            o[f"{nm}11_{tag}_eps1e-3"] = float(v1[3])
        for nm, fn in (("Omc", omega_c_scalar), ("lam", lyapunov)):
            try:
                v, v1 = richardson(lambda e: _fd(fn, em, pro, eps=e))
            except Exception:
                v = v1 = [np.nan] * 4
            for lab, x in zip(("0", "10", "01", "11"), v): o[f"{nm}{lab}_{tag}"] = float(x)
            o[f"{nm}11_{tag}_eps1e-3"] = float(v1[3])
    # entropy: Wald vs first law
    o["S10_wald"] = float(A10 / 4 + 0.5 * sgb_entropy_integral(bg, bg.g_dd, p10))
    o["S01_wald"] = float(A01 / 4 - 0.25 * dcs_entropy_integral(bg, bg.g_dd, bg.dg_ddd, bg.g_uu, t10))
    o["S11_wald"] = float(entropy_mixed(bg, H10, H01, H11, p10, t10, p11, t11)[3])
    s10, s01, s11 = first_law_entropy(bg, H10, H01, H11)
    o.update(S10_fl=float(s10), S01_fl=float(s01), S11_fl=float(s11))
    o["maxH11"] = float(max(np.max(np.abs(g.to_grid(H11[i])[1:-1, 1:-1])) for i in (1, 2, 3, 4)))
    o["phi11_hor_eq"] = eq(g.to_grid(p11)[0]); o["theta11_hor_half"] = float(np.interp(0.5, chi, g.to_grid(t11)[0]))
    return o


if __name__ == "__main__":
    files = sys.argv[1:] or [f for f in glob.glob("mixed_a_*.json") if ".old." not in f and "_50_" not in f]
    files = sorted(files, key=lambda p: float(re.search(r"([0-9.]+)\.json$", p).group(1)))
    rows = []
    for p in files:
        try:
            r = row(p); rows.append(r)
            print(f"a={r['a']:.2f}  Om11 {r['Om11']:+.5e}  k11 {r['k11']:+.5e}  A11 {r['A11']:+.5e}  "
                  f"S11 {r['S11_wald']:+.4e}/{r['S11_fl']:+.4e}  res {r['even_residual_rel']:.1e}/{r['rchi_residual_rel']:.1e}", flush=True)
        except Exception as e:
            print("FAILED", p, repr(e), flush=True)
    json.dump(rows, open("sweep_corrected.json", "w"), indent=1)
    keys = list(rows[0].keys())
    with open("sweep_corrected.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print("wrote sweep_corrected.csv / .json:", len(rows), "spins")
