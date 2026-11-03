"""
lyapunov_fix.py -- drop-in replacement for qnm_eikonal.lyapunov.

The original takes R''(r_ph) from a 3-point difference with dr = 1e-4. Its round-off error (~1e-8) is then
divided by 4 eps^2 = 4e-6 in the mixed coupling difference, which leaves noise of a few 1e-3 in lambda^(1,1):
comparable to the retrograde value itself. A 5-point stencil with dr = 1e-2 has truncation and round-off
errors below 1e-10.
"""
import numpy as np
from qnm_eikonal import omega_c, _grr


def lyapunov(em, ca, cb, pro, dr=1e-2):
    Om, rph = omega_c(em, ca, cb, pro)
    gtt, gtf, gff, *_ = em.g(rph, ca, cb)
    W = gtf**2 - gtt * gff
    b = -(gtf + Om * gff) / (gtt + Om * gtf)

    def R(r):
        tt, tf, ff, *_ = em.g(r, ca, cb)
        return (ff + 2 * b * tf + b**2 * tt) / (_grr(em, r, ca, cb) * (tf**2 - tt * ff))

    Rpp = (-R(rph + 2 * dr) + 16 * R(rph + dr) - 30 * R(rph) + 16 * R(rph - dr) - R(rph - 2 * dr)) / (12 * dr**2)
    tdot = (gff + b * gtf) / W
    return np.sqrt(abs(Rpp) / (2 * tdot**2))
