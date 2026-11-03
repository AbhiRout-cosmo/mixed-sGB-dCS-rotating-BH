"""
spherical_orbits.py -- near-spherical (non-equatorial) photon orbits in the corrected metric (Sec. V C, App. G).

Metric (M = 1), the ansatz of the paper extended off the equator with chi = cos(theta):
    g_tt   = g_tt^Kerr + H1,          g_tphi = g_tphi^Kerr (1 + H2),
    g_rr   = g_rr^Kerr (1 + H3),      g_thth = Sigma (1 + H3),       g_phph = g_phph^Kerr (1 + H4),
    H_i    = ca H_i^(1,0) + cb H_i^(0,1) + ca cb H_i^(1,1),   ca = zeta_GB, cb = zeta_CS.

The H_i are evaluated from their spectral coefficients on a 400 x 401 (r, theta) grid and represented by
quintic splines, which supply the first derivatives in Hamilton's equations for H = g^{mu nu} p_mu p_nu / 2,
with p_t = -1 and p_phi = b.

There is no Carter constant.  Orbits are found by shooting from the northern turning point
(theta = theta_t, p_r = p_theta = 0, b fixed by the null condition) to the first equator crossing and
requiring p_r = 0 there.  By equatorial reflection and (t, phi) -> (-t, -phi) symmetry this quarter orbit is
periodic in (r, theta).

    solve(Metric(cache, ca, cb), theta_t, prograde, r_guess)  ->  dict(rt, re, b, dphi, sol)
"""
import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from mixed_cache import load_mixed


class Metric:
    def __init__(self, cache, ca=0.0, cb=0.0, rmax=10.0, nr=400, nth=401):
        bg, H10, H01, H11, _, _ = load_mixed(cache)
        self.a = bg.a; g = bg.grid; rp = g.r_plus; self.rp = rp
        r = np.linspace(rp * 1.001, rmax, nr)
        th = np.linspace(0.0, np.pi, nth)
        z = 2 * rp / r - 1.0
        self.S = {}
        for i in (1, 2, 3, 4):
            c = ca * H10[i] + cb * H01[i] + ca * cb * H11[i]
            F = g.evaluate(c, z, np.cos(th))
            self.S[i] = RectBivariateSpline(r, th, F, kx=5, ky=5)

    def H(self, i, r, th):
        s = self.S[i]
        return s(r, th, grid=False), s(r, th, dx=1, grid=False), s(r, th, dy=1, grid=False)

    def metric(self, r, th):
        """Covariant (g_tt, g_tphi, g_phiphi, g_rr, g_thth), each as (value, d/dr, d/dtheta)."""
        a = self.a
        s, c = np.sin(th), np.cos(th)
        Sig = r**2 + a**2 * c**2; Del = r**2 - 2 * r + a**2
        dSig_r, dSig_t = 2 * r, -2 * a**2 * s * c
        # Kerr
        gtt = -(1 - 2 * r / Sig)
        gtt_r = 2 * (Sig - r * dSig_r) / Sig**2
        gtt_t = -2 * r * dSig_t / Sig**2
        gtf = -2 * a * r * s**2 / Sig
        gtf_r = -2 * a * s**2 * (Sig - r * dSig_r) / Sig**2
        gtf_t = -2 * a * r * (2 * s * c * Sig - s**2 * dSig_t) / Sig**2
        A = (r**2 + a**2)**2 - a**2 * Del * s**2
        A_r = 4 * r * (r**2 + a**2) - a**2 * (2 * r - 2) * s**2
        A_t = -a**2 * Del * 2 * s * c
        gff = A * s**2 / Sig
        gff_r = s**2 * (A_r * Sig - A * dSig_r) / Sig**2
        gff_t = ((A_t * s**2 + A * 2 * s * c) * Sig - A * s**2 * dSig_t) / Sig**2
        grr = Sig / Del
        grr_r = (dSig_r * Del - Sig * (2 * r - 2)) / Del**2
        grr_t = dSig_t / Del
        gqq, gqq_r, gqq_t = Sig, dSig_r, dSig_t
        # corrections
        H1, H1r, H1t = self.H(1, r, th)
        H2, H2r, H2t = self.H(2, r, th)
        H3, H3r, H3t = self.H(3, r, th)
        H4, H4r, H4t = self.H(4, r, th)
        return dict(
            tt=(gtt + H1, gtt_r + H1r, gtt_t + H1t),
            tf=(gtf * (1 + H2), gtf_r * (1 + H2) + gtf * H2r, gtf_t * (1 + H2) + gtf * H2t),
            ff=(gff * (1 + H4), gff_r * (1 + H4) + gff * H4r, gff_t * (1 + H4) + gff * H4t),
            rr=(grr * (1 + H3), grr_r * (1 + H3) + grr * H3r, grr_t * (1 + H3) + grr * H3t),
            qq=(gqq * (1 + H3), gqq_r * (1 + H3) + gqq * H3r, gqq_t * (1 + H3) + gqq * H3t))

    def rhs(self, lam, y, b):
        """Hamilton's equations; y = (t, r, theta, phi, p_r, p_theta), p_t = -1, p_phi = b."""
        _, r, th, _, pr, pq = y
        m = self.metric(r, th)
        tt, tf, ff, rr, qq = (m[k] for k in ('tt', 'tf', 'ff', 'rr', 'qq'))
        D = tf[0]**2 - tt[0] * ff[0]
        pt, pf = -1.0, b
        dKs = []                                   # derivatives of the (t, phi) part of g^{mu nu} p_mu p_nu
        for j in (1, 2):
            dD = 2 * tf[0] * tf[j] - tt[j] * ff[0] - tt[0] * ff[j]
            N = ff[0] * pt**2 - 2 * tf[0] * pt * pf + tt[0] * pf**2
            dN = ff[j] * pt**2 - 2 * tf[j] * pt * pf + tt[j] * pf**2
            dKs.append(-(dN * D - N * dD) / D**2)
        dr = pr / rr[0]; dth = pq / qq[0]
        dt = (-ff[0] * pt + tf[0] * pf) / D
        dphi = (tf[0] * pt - tt[0] * pf) / D
        dpr = -0.5 * (dKs[0] - rr[1] / rr[0]**2 * pr**2 - qq[1] / qq[0]**2 * pq**2)
        dpq = -0.5 * (dKs[1] - rr[2] / rr[0]**2 * pr**2 - qq[2] / qq[0]**2 * pq**2)
        return [dt, dr, dth, dphi, dpr, dpq]


def kerr_spherical(a, r):
    """Impact parameter b = L/E and Carter constant q = Q/E^2 of the Kerr spherical photon orbit at radius r."""
    b = -(r**3 - 3 * r**2 + a**2 * r + a**2) / (a * (r - 1))
    q = r**3 * (4 * a**2 - r * (r - 3)**2) / (a**2 * (r - 1)**2)
    return b, q


def kerr_rs_from_turn(a, tht, pro):
    """Radius of the Kerr spherical photon orbit whose polar turning angle is tht (prograde or retrograde branch)."""
    rpro = 2 * (1 + np.cos(2 / 3 * np.arccos(-a))); rret = 2 * (1 + np.cos(2 / 3 * np.arccos(a)))
    def g(r):
        b, q = kerr_spherical(a, r)
        u = np.cos(tht)**2                         # zero of the polar potential: q + a^2 u - b^2 u / (1 - u)
        return q + a**2 * u - b**2 * u / (1 - u)
    rpol = brentq(lambda r: kerr_spherical(a, r)[0], rpro + 1e-6, rret - 1e-6)
    return brentq(g, rpro + 1e-9, rpol) if pro else brentq(g, rpol, rret - 1e-9)


def b_turn(m, r, th, pro):
    """Impact parameter of a photon at rest in (r, theta) at (r, th): root of the null condition."""
    mm = m.metric(r, th)
    tt, tf, ff = mm['tt'][0], mm['tf'][0], mm['ff'][0]
    D = tf**2 - tt * ff
    b1 = (-tf + np.sqrt(D)) / tt; b2 = (-tf - np.sqrt(D)) / tt
    return max(b1, b2) if pro else min(b1, b2)


def arc(m, rt, tht, pro, dense=False):
    """Integrate from the northern turning point (rt, tht) to the first equator crossing."""
    b = b_turn(m, rt, tht, pro)
    y0 = [0.0, rt, tht, 0.0, 0.0, 0.0]
    def eq(l, y, b): return y[2] - np.pi / 2
    eq.terminal = True; eq.direction = 1
    s = solve_ivp(m.rhs, (0, 60), y0, args=(b,), rtol=1e-12, atol=1e-13,
                  method='DOP853', events=[eq], dense_output=dense)
    return s, b


def solve(m, tht, pro, rguess, dr=2e-3):
    """Turning radius rt such that p_r = 0 at the equator.  Returns rt, the equatorial radius re, b, the
    azimuth advanced over the quarter orbit, and the dense solution."""
    f = lambda r: arc(m, r, tht, pro)[0].y_events[0][0][4]
    lo, hi = rguess - dr, rguess + dr
    flo, fhi = f(lo), f(hi); k = 0
    while flo * fhi > 0:
        dr *= 3; lo, hi = rguess - dr, rguess + dr; flo, fhi = f(lo), f(hi); k += 1
        if k > 6:
            raise RuntimeError('no bracket')
    rt = brentq(f, lo, hi, xtol=1e-14, rtol=1e-15)
    s, b = arc(m, rt, tht, pro, dense=True)
    return dict(rt=rt, re=s.y_events[0][0][1], b=b, dphi=s.y_events[0][0][3], sol=s)
