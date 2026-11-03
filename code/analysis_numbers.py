"""
analysis_numbers.py -- numbers quoted in Sec. VII (perturbative control, observability, low-spin scaling).

    cd data && python ../code/analysis_numbers.py            -> prints the numbers, writes analysis_numbers.json

1. Perturbative control (Sec. VII E): maximum deformation amplitudes A^(m,n) = max_{r,chi} |H_i^(m,n)| and the
   largest coupling zeta_max (zeta_GB = zeta_CS = zeta) with total fractional deformation
   zeta (A10 + A01) + zeta^2 A11 <= 0.1.
2. Observability (Sec. VII F): Omega_c^(0) and the Richardson-extrapolated Omega_c^(1,1) at a = 0.7, the
   fractional shift |dOmega_c/Omega_c| = zeta_GB zeta_CS |Omega_c^(1,1)/Omega_c^(0)| for the current bounds
   sqrt(alpha_GB) = 1.7 km, sqrt(alpha_CS) = 8.5 km, and the mass M_eft above which the dCS deformation at its
   bound stays below 0.1.
3. Low-spin scaling (Sec. VII G): log-log slopes of |Omega_H^(0,1)|, |A^(0,1)|, |A^(1,1)| from a = 0.1, 0.2, 0.3
   and from all a <= 0.2.
"""
import os, json
import numpy as np
from mixed_cache import load_mixed
from mixed_observables import omega_pure
from horizon_area import area_shifts
from isco_photon import EqMetric
from qnm_eikonal import _fd, omega_c_scalar


def cache(a):
    """mixed_a_0.1.json, mixed_a_0.12.json, ... (the naming written by mixed_cache_fixed.py)."""
    return f"mixed_a_{f'{a:.2f}'.rstrip('0')}.json"


def maxH(bg, Hc):
    return max(np.nanmax(np.abs(bg.grid.to_grid(Hc[i]))) for i in (1, 2, 3, 4))


out = {}

# 1. perturbative control -------------------------------------------------------------------------------
rows = []
for s in np.round(np.arange(0.10, 0.901, 0.02), 2):
    if not os.path.exists(cache(s)):
        continue
    bg, H10, H01, H11, _, _ = load_mixed(cache(s))
    rows.append((bg.a, maxH(bg, H10), maxH(bg, H01), maxH(bg, H11)))
a = np.array([r[0] for r in rows]); A10 = np.array([r[1] for r in rows])
A01 = np.array([r[2] for r in rows]); A11 = np.array([r[3] for r in rows])
thr = 0.1
zmax = (-(A10 + A01) + np.sqrt((A10 + A01)**2 + 4 * A11 * thr)) / (2 * A11)
ratio = A11 / np.maximum(A10, A01)
out["eft_control"] = dict(a=a.tolist(), A10=A10.tolist(), A01=A01.tolist(), A11=A11.tolist(), zeta_max=zmax.tolist())
print("Perturbative control")
print(f"{'a':>5} {'A_sGB':>8} {'A_dCS':>8} {'A_mix':>9} {'zeta_max':>9} {'A_mix/max(pure)':>16}")
for i in range(len(a)):
    if round(a[i] * 100) % 10 == 0:
        print(f"{a[i]:>5.2f} {A10[i]:>8.3f} {A01[i]:>8.3f} {A11[i]:>9.3f} {zmax[i]:>9.4f} {ratio[i]:>16.2f}")
print(f"  A_mix exceeds both pure amplitudes at every spin: {bool(np.all(ratio > 1))};"
      f"  smallest ratio {ratio.min():.2f} at a = {a[np.argmin(ratio)]:.2f}")

# 2. observability --------------------------------------------------------------------------------------
GMsun_km = 1.47662
sqrt_aGB, sqrt_aCS = 1.7, 8.5          # km, current bounds
a_obs = 0.7
bg, H10, H01, H11, _, _ = load_mixed(cache(a_obs))
em = EqMetric(bg, H10, H01, H11)
f1 = np.array(_fd(omega_c_scalar, em, True, eps=1e-3)); f2 = np.array(_fd(omega_c_scalar, em, True, eps=2e-3))
Oc0 = f1[0]; Oc11 = (4 * f1[3] - f2[3]) / 3
A_dCS = maxH(bg, H01)
M_eft = sqrt_aCS / (0.1 / A_dCS)**0.25 / GMsun_km
frac = lambda M: abs((sqrt_aGB / (M * GMsun_km))**4 * (sqrt_aCS / (M * GMsun_km))**4 * Oc11 / Oc0)
out["observability"] = dict(a=a_obs, Omc0=float(Oc0), Omc11=float(Oc11), M_eft=float(M_eft),
                            frac_at_M_eft=float(frac(M_eft)), frac_at_10=float(frac(10.0)), frac_at_30=float(frac(30.0)))
print("\nObservability (prograde photon orbit, a = 0.7)")
print(f"  Omega_c^(0) = {Oc0:.5f}   Omega_c^(1,1) = {Oc11:+.5f}   M_eft = {M_eft:.1f} Msun")
print(f"  |dOmega_c/Omega_c|: {frac(M_eft):.1e} at M_eft, {frac(10.0):.1e} at 10 Msun, {frac(30.0):.1e} at 30 Msun")

# 3. low-spin scaling -----------------------------------------------------------------------------------
av, Ocs, Ad, Am = [], [], [], []
for s in np.round(np.arange(0.10, 0.501, 0.02), 2):
    if not os.path.exists(cache(s)):
        continue
    bg, H10, H01, H11, _, _ = load_mixed(cache(s)); chi = bg.grid.chi
    _, _, a01, a11 = area_shifts(bg, H10, H01, H11)
    av.append(bg.a); Ocs.append(abs(float(np.interp(0, chi, omega_pure(bg, H01))))); Ad.append(abs(a01)); Am.append(abs(a11))
av, Ocs, Ad, Am = map(np.array, (av, Ocs, Ad, Am))
slope = lambda y, idx: float(np.polyfit(np.log(av[idx]), np.log(y[idx]), 1)[0])
pick = np.array([np.argmin(np.abs(av - s)) for s in (0.1, 0.2, 0.3)]); low = av <= 0.201
out["lowspin"] = dict(slopes_0p1_0p2_0p3=dict(Om01=slope(Ocs, pick), A01=slope(Ad, pick), A11=slope(Am, pick)),
                      slopes_a_le_0p2=dict(Om01=slope(Ocs, low), A01=slope(Ad, low), A11=slope(Am, low)))
print("\nLow-spin slopes")
print("  from a = 0.1, 0.2, 0.3:", {k: round(v, 2) for k, v in out["lowspin"]["slopes_0p1_0p2_0p3"].items()})
print("  from all a <= 0.2:     ", {k: round(v, 2) for k, v in out["lowspin"]["slopes_a_le_0p2"].items()})

json.dump(out, open("analysis_numbers.json", "w"), indent=1)
print("\nwrote analysis_numbers.json")
