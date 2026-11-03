"""
qnm_eikonal.py  -- eikonal (photon-ring) QNM shifts at mixed order
==================================================================

Cardoso-Berti correspondence (large-l):
    omega_QNM ~ Omega_c (l+1/2) - i (n+1/2) |lambda|,
with Omega_c the orbital frequency of the equatorial photon orbit and lambda
its Lyapunov (instability) exponent.  Both come from the equatorial circular
NULL geodesic -- the same machinery as the photon ring, so it inherits the good
(bulk, equatorial) conditioning.

For a null circular orbit (E=1, impact parameter b):
    Omega_c = Omega_+/-(r_ph)                       (orbital frequency)
    R(r)   = (g_phiphi + 2 b g_tphi + b^2 g_tt)/(g_rr W),   W = g_tphi^2 - g_tt g_phiphi
    tdot   = (g_phiphi + b g_tphi)/W
    lambda = sqrt( R''(r_ph) / (2 tdot^2) )         (coordinate-time Lyapunov)

Validation: Schwarzschild (a=0) gives  Omega_c = lambda = 1/(3 sqrt 3) = 0.19245.
Mixed shift Omega_c^(1,1), lambda^(1,1) via FD-in-couplings (prograde/retrograde).

NOTE: eikonal = leading large-l estimate; it does NOT capture the axial-polar
mode coupling that the mixed sGB x dCS term induces (that needs the full
perturbation problem).  Reported as the leading ringdown frequency/damping shift.
"""

from __future__ import annotations
import numpy as np
from isco_photon import EqMetric, _omega, r_photon, mixed_shift


def _grr(em, r, ca, cb):
    H3, _ = em._H(3, r, ca, cb)
    Delta = r**2 - 2 * r + em.a**2
    return (r**2 / Delta) * (1.0 + H3)


def omega_c(em, ca, cb, pro):
    """Orbital frequency of the photon orbit."""
    rph = r_photon(em, ca, cb, pro)
    gtt, gtf, gff, dtt, dtf, dff = em.g(rph, ca, cb)
    return _omega(gtt, gtf, gff, dtt, dtf, dff, pro), rph


def lyapunov(em, ca, cb, pro):
    """Coordinate-time Lyapunov exponent of the photon orbit."""
    Om, rph = omega_c(em, ca, cb, pro)
    gtt, gtf, gff, *_ = em.g(rph, ca, cb)
    W = gtf**2 - gtt * gff
    b = -(gtf + Om * gff) / (gtt + Om * gtf)               # critical impact parameter

    def R(r):
        tt, tf, ff, *_ = em.g(r, ca, cb)
        Wr = tf**2 - tt * ff
        return (ff + 2 * b * tf + b**2 * tt) / (_grr(em, r, ca, cb) * Wr)

    dr = 1e-4
    Rpp = (R(rph + dr) - 2 * R(rph) + R(rph - dr)) / dr**2  # R''(r_ph)
    tdot = (gff + b * gtf) / W
    return np.sqrt(abs(Rpp) / (2 * tdot**2))


# ---- mixed-order shifts ------------------------------------------------ #
def _fd(fn, em, pro, eps=1e-3):
    f = lambda ca, cb: fn(em, ca, cb, pro)
    fpp, fpm = f(+eps, +eps), f(+eps, -eps)
    fmp, fmm = f(-eps, +eps), f(-eps, -eps)
    f0 = f(0.0, 0.0)
    return (f0,
            (fpp + fpm - fmp - fmm) / (4 * eps),           # ^(1,0)
            (fpp - fpm + fmp - fmm) / (4 * eps),           # ^(0,1)
            (fpp - fpm - fmp + fmm) / (4 * eps * eps))     # ^(1,1)


def omega_c_scalar(em, ca, cb, pro):
    return omega_c(em, ca, cb, pro)[0]


# ====================================================================== #
def _selftest():
    from spectral_grid import SpectralGrid
    from kerr_background import KerrBackground
    print("Schwarzschild (a->0) check: Omega_c = lambda = 1/(3 sqrt3) = %.5f" % (1/(3*np.sqrt(3))))
    for a in (0.05, 0.5, 0.9):
        g = SpectralGrid(Nz=30, Nchi=30, spin=a); bg = KerrBackground(g)
        z0 = {i: np.zeros((g.Nz + 1, g.Nchi + 1)) for i in (1, 2, 3, 4)}
        em = EqMetric(bg, z0, z0, z0)
        for pro, tag in ((True, "pro"), (False, "retro")):
            Om, rph = omega_c(em, 0., 0., pro)
            lam = lyapunov(em, 0., 0., pro)
            print(f"  a={a} {tag:5s}: r_ph={rph:.4f}  Omega_c={Om:.5f}  lambda={lam:.5f}")


def _sweep(mixed_paths):
    from mixed_cache import load_mixed
    import os
    print(f"{'a':>5} {'dir':>6} {'Om_c':>9} {'Om_c^(1,1)':>12} {'lambda':>9} {'lam^(1,1)':>12}")
    for p in mixed_paths:
        if not os.path.exists(p):
            print(f"[skip] {p}"); continue
        bg, H10, H01, H11, _, _ = load_mixed(p)
        em = EqMetric(bg, H10, H01, H11)
        for pro, tag in ((True, "pro"), (False, "retro")):
            Om0, _, _, Om11 = _fd(omega_c_scalar, em, pro)
            la0, _, _, la11 = _fd(lyapunov, em, pro)
            print(f"{bg.a:>5.2f} {tag:>6} {Om0:>9.4f} {Om11:>+12.4f} {la0:>9.4f} {la11:>+12.4f}")


if __name__ == "__main__":
    _selftest()
    print()

    # Formats numbers as '0.1', '0.12', ..., '0.9' without trailing zeros
    spins = [f"{a:.2f}".rstrip("0").rstrip(".") for a in np.arange(0.10, 0.91, 0.02)]
    file_list = [f"mixed_a_{s}.json" for s in spins]

    _sweep(file_list)
