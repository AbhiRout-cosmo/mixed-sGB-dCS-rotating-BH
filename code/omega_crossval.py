"""
omega_crossval.py -- cross-validation of the closed-form Omega_H^(1,1) (Table III).

    cd data && python ../code/omega_crossval.py [mixed_a_0.1.json mixed_a_0.3.json ...]

Compares eq. (omega11) of the paper (mixed_observables.omega_mixed, equator value) with a direct second
mixed finite difference of Omega_H = -g_tphi/g_phiphi = Omega_H^(0) (1 + H2)/(1 + H4) on the horizon,
built from the fully assembled corrected metric with step eps = 1e-3.
"""
import sys
import numpy as np
from mixed_cache import load_mixed
from mixed_observables import omega_mixed

EPS = 1e-3
paths = sys.argv[1:] or [f"mixed_a_{s}.json" for s in ("0.1", "0.3", "0.5", "0.7", "0.9")]
print(f"{'a':>4} {'closed form':>14} {'direct diff.':>14} {'rel. diff.':>10}")
for p in paths:
    bg, H10, H01, H11, _, _ = load_mixed(p)
    g = bg.grid; chi = g.chi; a, rp = bg.a, g.r_plus
    hz = lambda H, i: float(np.interp(0, chi, g.to_grid(H[i])[0]))
    H = {i: (hz(H10, i), hz(H01, i), hz(H11, i)) for i in (2, 4)}
    def Om(ca, cb):
        h2 = ca * H[2][0] + cb * H[2][1] + ca * cb * H[2][2]
        h4 = ca * H[4][0] + cb * H[4][1] + ca * cb * H[4][2]
        return a / (2 * rp) * (1 + h2) / (1 + h4)
    direct = (Om(EPS, EPS) - Om(EPS, -EPS) - Om(-EPS, EPS) + Om(-EPS, -EPS)) / (4 * EPS * EPS)
    closed = float(np.interp(0, chi, omega_mixed(bg, H10, H01, H11)))
    print(f"{a:>4.1f} {closed:>+14.7f} {direct:>+14.7f} {abs(direct - closed) / abs(closed):>10.1e}")
