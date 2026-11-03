"""
thermo_check.py -- thermodynamics of the mixed solution: Wald entropy, first-law (Smarr) check, and the
arrays needed to plot them.

    python thermo_check.py                              all spins a = 0.10 ... 0.90 found in this folder
    python thermo_check.py mixed_a_0.3.json ...         selected files

Each mixed_a_X.json needs the matching data_a_X.json (the Wald integrals use the leading scalars).

First-law check (no derivative in spin needed).  With S = M^2 s(a, zeta_GB, zeta_CS) and
dM = T dS + Omega dJ at fixed couplings,
    1/T = 2 s + 2 a Omega/T - 4 sum_i zeta_i ds/dzeta_i        (M = 1, 1/T = 2 pi/kappa),
so   s^(1,0) = (2a[Omega/T]^(1,0) - [1/T]^(1,0))/2,   likewise s^(0,1),
     s^(1,1) = (2a[Omega/T]^(1,1) - [1/T]^(1,1))/6.
These are compared with the Wald entropy  S = A/4 + (zeta_GB/2) I_sGB - (zeta_CS/4) I_dCS.

Output: the comparison table, then every quantity as a numpy array (one entry per spin, ready to paste into
a plotting cell), and the same columns in thermo_data.csv.  Expansion coefficients are written
X0, X10, X01, X11 for X^(0), X^(1,0), X^(0,1), X^(1,1); everything is at fixed M = 1 and fixed J.

    spins                      a
    S0                         Kerr entropy A0/4
    S10_wald, S01_wald         pure-sector Wald entropy
    S11_wald                   mixed Wald entropy = S11_area + S11_sGB + S11_dCS
    S11_area                   A^(1,1)/4
    S11_sGB, S11_dCS           the two scalar-curvature (Wald) terms of S^(1,1)
    S10_fl, S01_fl, S11_fl     the same coefficients from the first law
    rel10, rel01, rel11        |Wald - first law| / |first law|
    T0, T10, T01, T11          temperature kappa/(2 pi), kappa evaluated on the horizon
    Om0, Om10, Om01, Om11      horizon angular velocity
    A0, A10, A01, A11          horizon area
"""
import sys, os, csv, warnings
import numpy as np
warnings.filterwarnings("ignore")
from spectral_grid import SpectralGrid
from mixed_cache import load_mixed
from mixed_observables import omega_pure, omega_mixed
from kappa_horizon import _kappa, kappa_coeffs
from horizon_area import area_shifts
from wald_sgb import sgb_entropy_integral
from wald_dcs import dcs_entropy_integral
from wald_mixed import entropy_mixed

CSV_OUT = "thermo_data.csv"          # set to None to skip writing the file
KEYS = ["spins", "S0", "S10_wald", "S01_wald", "S11_wald", "S11_area", "S11_sGB", "S11_dCS",
        "S10_fl", "S01_fl", "S11_fl", "rel10", "rel01", "rel11",
        "T0", "T10", "T01", "T11", "Om0", "Om10", "Om01", "Om11", "A0", "A10", "A01", "A11"]


def first_law_entropy(bg, H10, H01, H11, eps=1e-4):
    """kappa is evaluated on the horizon (kappa_horizon.py). eps = 1e-4 keeps the truncation of the coupling
    difference below 1e-5 up to a = 0.9; with eps = 1e-3 it reaches 1e-3 there."""
    g = bg.grid; chi = g.chi; a, rp = bg.a, g.r_plus
    hz = lambda H, i: float(np.interp(0, chi, g.to_grid(H[i])[0]))
    def f(ca, cb):
        kap = float(np.interp(0, chi, _kappa(bg, (H10, H01, H11), ca, cb)[0]))
        H2 = ca * hz(H10, 2) + cb * hz(H01, 2) + ca * cb * hz(H11, 2)
        H4 = ca * hz(H10, 4) + cb * hz(H01, 4) + ca * cb * hz(H11, 4)
        return np.array([2 * np.pi * a / (2 * rp) * (1 + H2) / (1 + H4) / kap, 2 * np.pi / kap])          # Omega/T, 1/T
    pp, pm, mp, mm = f(eps, eps), f(eps, -eps), f(-eps, eps), f(-eps, -eps)
    d10 = (pp + pm - mp - mm) / (4 * eps); d01 = (pp - pm + mp - mm) / (4 * eps)
    d11 = (pp - pm - mp + mm) / (4 * eps * eps)
    return (2 * a * d10[0] - d10[1]) / 2, (2 * a * d01[0] - d01[1]) / 2, (2 * a * d11[0] - d11[1]) / 6


def run_one(path):
    """All thermodynamic quantities for one cache, as a dict keyed like KEYS."""
    bg, H10, H01, H11, p11, t11 = load_mixed(path)
    g = bg.grid; chi = g.chi; a, rp = bg.a, g.r_plus
    _, raw = SpectralGrid.from_file(path.replace("mixed_a_", "data_a_").replace(".old.json", ".json"))
    p10 = g.fit(g.field_values(raw, "sGBPhi"), "even")
    t10 = g.fit(g.field_values(raw, "dCSPhi"), "odd")
    eq = lambda v: float(np.interp(0, chi, v))

    A0, A10, A01, A11 = area_shifts(bg, H10, H01, H11)
    S10 = A10 / 4 + 0.5 * sgb_entropy_integral(bg, bg.g_dd, p10)
    S01 = A01 / 4 - 0.25 * dcs_entropy_integral(bg, bg.g_dd, bg.dg_ddd, bg.g_uu, t10)
    S11_area, S11_sGB, S11_dCS, S11 = entropy_mixed(bg, H10, H01, H11, p10, t10, p11, t11)
    s10, s01, s11 = first_law_entropy(bg, H10, H01, H11)
    k10, k01, k11 = kappa_coeffs(bg, H10, H01, H11)
    k0 = np.sqrt(1 - a * a) / (2 * rp)
    tp = 2 * np.pi
    rel = lambda w, f: abs(w - f) / abs(f)
    vals = [a, A0 / 4, S10, S01, S11, S11_area, S11_sGB, S11_dCS, s10, s01, s11,
            rel(S10, s10), rel(S01, s01), rel(S11, s11),
            k0 / tp, eq(k10) / tp, eq(k01) / tp, eq(k11) / tp,
            a / (2 * rp), eq(omega_pure(bg, H10)), eq(omega_pure(bg, H01)), eq(omega_mixed(bg, H10, H01, H11)),
            A0, A10, A01, A11]
    return dict(zip(KEYS, [float(v) for v in vals]))


def print_arrays(rows, per_line=6):
    """Print every column as  name = np.array([...])  for pasting into a plotting cell."""
    print("import numpy as np\n")
    for k in KEYS:
        fmt = (lambda v: f"{v:.2f}") if k == "spins" else (lambda v: f"{v:.2e}") if k.startswith("rel") \
            else (lambda v: f"{v:+.6e}")
        v = [fmt(r[k]) for r in rows]
        n = 10 if k == "spins" else per_line
        body = ",\n    ".join(", ".join(v[i:i + n]) for i in range(0, len(v), n))
        print(f"{k} = np.array([\n    {body}\n])\n")


if __name__ == "__main__":
    paths = sys.argv[1:] or [f"mixed_a_{f'{a:.2f}'.rstrip('0')}.json" for a in np.arange(0.10, 0.901, 0.02)]
    rows = []
    print(f"{'a':>5} {'S10 Wald':>10} {'1st law':>10} {'S01 Wald':>10} {'1st law':>10} "
          f"{'S11 Wald':>11} {'1st law':>11} {'rel diff':>9}")
    for p in paths:
        if not os.path.exists(p):
            print(f"[skip] {p} not found"); continue
        try:
            r = run_one(p)
        except FileNotFoundError as e:
            print(f"[skip] {p}: {e.filename} not found"); continue
        rows.append(r)
        print(f"{r['spins']:>5.2f} {r['S10_wald']:>+10.5f} {r['S10_fl']:>+10.5f} {r['S01_wald']:>+10.5f} "
              f"{r['S01_fl']:>+10.5f} {r['S11_wald']:>+11.5f} {r['S11_fl']:>+11.5f} {r['rel11']:>9.1e}", flush=True)
    if rows:
        print("\n# " + "=" * 70 + " #")
        print("# ARRAYS FOR PLOTTING (one entry per spin, in the order of `spins`)")
        print("# " + "=" * 70 + " #")
        print_arrays(rows)
        if CSV_OUT:
            with open(CSV_OUT, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=["a"] + KEYS[1:]); w.writeheader()
                for r in rows:
                    w.writerow({("a" if k == "spins" else k): r[k] for k in KEYS})
            print(f"# saved {CSV_OUT}  ({len(rows)} spins)")
