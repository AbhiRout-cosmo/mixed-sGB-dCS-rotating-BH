"""
spherical_orbits_sweep.py -- coupling coefficients of the near-spherical photon orbits versus inclination.

    cd data && python ../code/spherical_orbits_sweep.py mixed_a_0.5.json pro   spherical_orbits_pro.json
    cd data && python ../code/spherical_orbits_sweep.py mixed_a_0.5.json retro spherical_orbits_retro.json

For each polar turning angle theta_t the orbit is solved in the metrics with couplings (0, 0), (+-eps, +-eps)
and (+-2 eps, +-2 eps), eps = 1e-3, and the coefficients X^(1,0), X^(0,1), X^(1,1) of
    rt  (radius at the turning latitude),  re  (radius at the equator),  b  (impact parameter)
are taken with the four-point stencil (as for the equatorial orbits), plain and Richardson-extrapolated.
Inclination: i = 90 - theta_t (prograde), 90 + theta_t (retrograde); Delta r = rt - re.
The two branches are independent and can run in parallel (about 10 minutes each on one core).
"""
import sys, json
import numpy as np
from spherical_orbits import Metric, solve, kerr_rs_from_turn

cache, branch, outp = sys.argv[1], sys.argv[2], sys.argv[3]
pro = branch == 'pro'
E1, E2 = 1e-3, 2e-3
combos = [(0, 0)] + [(sa * e, sb * e) for e in (E1, E2) for sa in (1, -1) for sb in (1, -1)]
M = {c: Metric(cache, *c) for c in combos}
TH = [89.5, 89, 88, 86, 82, 75, 65, 55, 45, 35, 25, 15, 8, 4]
out = []
for thd in TH:
    tht = np.radians(thd); rk = kerr_rs_from_turn(M[(0, 0)].a, tht, pro)
    R = {c: solve(M[c], tht, pro, rk) for c in combos}
    row = dict(th_t=thd, inclination=(90 - thd) if pro else (90 + thd), r_kerr=rk)
    for key in ('rt', 're', 'b'):
        def coeffs(e):
            pp, pm, mp, mm = (R[(sa * e, sb * e)][key] for sa, sb in ((1, 1), (1, -1), (-1, 1), (-1, -1)))
            return ((pp + pm - mp - mm) / (4 * e), (pp - pm + mp - mm) / (4 * e), (pp - pm - mp + mm) / (4 * e * e))
        c1, c2 = coeffs(E1), coeffs(E2)
        row[key] = dict(r0=float(R[(0, 0)][key]), c10=float(c1[0]), c01=float(c1[1]), c11=float(c1[2]),
                        c10_rich=float((4 * c1[0] - c2[0]) / 3), c01_rich=float((4 * c1[1] - c2[1]) / 3),
                        c11_rich=float((4 * c1[2] - c2[2]) / 3))
    out.append(row)
    print(f"theta_t = {thd:5.1f}  re = {row['re']['r0']:.6f}  re^(1,1) = {row['re']['c11']:+.5f}  "
          f"Delta r^(1,1) = {row['rt']['c11'] - row['re']['c11']:+.4e}", flush=True)
    json.dump(out, open(outp, 'w'), indent=1)
