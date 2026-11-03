"""
isco_photon.py
==============

Mixed-order O(zeta_GB zeta_CS) shifts of the equatorial ISCO and photon-ring
(photon-sphere) radii, from the cached mixed metric (mixed_a_0_X.json).

Equatorial circular orbits see only g_tt, g_tphi, g_phiphi.  With chi=0 the
metric functions are (M=1):
    g_tt   = -(1 - 2/r) + H1
    g_tphi = -(2a/r)(1 + H2)
    g_phiphi = (r^2 + a^2 + 2a^2/r)(1 + H4)
where H1,H2,H4 are the (r-dependent) equatorial metric functions of the full
metric  g = g^(0) + a h^(1,0) + b h^(0,1) + a b h^(1,1).

Circular-orbit frequency  Omega_+/-(r) = (-g_tphi' +/- sqrt(g_tphi'^2 - g_tt' g_phiphi'))/g_phiphi'
Photon sphere:  g_tt + 2 Omega g_tphi + Omega^2 g_phiphi = 0.
ISCO        :  dE_circ/dr = 0,  E = -(g_tt + Omega g_tphi)/sqrt(-(g_tt+2Omega g_tphi+Omega^2 g_phiphi)).

The mixed shift r^(1,1) is a 2nd finite difference in the two couplings; the
first differences r^(1,0)/r^(0,1) are printed as a Lam (Fig. 15/16) check.
Everything is equatorial and in the bulk, so it is far better conditioned than
the horizon observables.
"""

from __future__ import annotations
import numpy as np
from numpy.polynomial import chebyshev as C
from numpy.polynomial import legendre as L
from scipy.optimize import brentq


# ---------------------------------------------------------------------- #
def _Pl0(Nchi):
    """P_l(0) for l = 0..Nchi."""
    out = np.zeros(Nchi + 1)
    for l in range(Nchi + 1):
        e = np.zeros(l + 1); e[l] = 1.0
        out[l] = L.legval(0.0, e)
    return out


def _equatorial_cheb(Hc, Pl0):
    """Chebyshev-z coefficients of H_i on the equator (chi=0): a_n = sum_l c_{n,l} P_l(0)."""
    return {i: Hc[i] @ Pl0 for i in Hc}


class EqMetric:
    """Equatorial metric g_tt,g_tphi,g_phiphi and r-derivatives for couplings (ca,cb)."""

    def __init__(self, bg, H10, H01, H11):
        self.a = bg.a; self.rp = bg.grid.r_plus
        Pl0 = _Pl0(bg.grid.Nchi)
        self.e10 = _equatorial_cheb(H10, Pl0)
        self.e01 = _equatorial_cheb(H01, Pl0)
        self.e11 = _equatorial_cheb(H11, Pl0)

    def _H(self, i, r, ca, cb):
        """H_i and dH_i/dr at radius r (equator) for couplings (ca, cb)."""
        z = 2 * self.rp / r - 1.0
        a = ca * self.e10[i] + cb * self.e01[i] + ca * cb * self.e11[i]
        H = C.chebval(z, a)
        dHdz = C.chebval(z, C.chebder(a))
        dzdr = -2 * self.rp / r**2
        return H, dHdz * dzdr

    def g(self, r, ca, cb):
        """Return (g_tt,g_tphi,g_phiphi, g_tt',g_tphi',g_phiphi') at r."""
        a = self.a
        H1, dH1 = self._H(1, r, ca, cb)
        H2, dH2 = self._H(2, r, ca, cb)
        H4, dH4 = self._H(4, r, ca, cb)
        # background pieces (chi=0, M=1)
        gtt0 = -(1 - 2 / r);           dgtt0 = -2 / r**2
        gtf0 = -2 * a / r;             dgtf0 = 2 * a / r**2
        gff0 = r**2 + a**2 + 2 * a**2 / r
        dgff0 = 2 * r - 2 * a**2 / r**2
        gtt = gtt0 + H1
        dgtt = dgtt0 + dH1
        gtf = gtf0 * (1 + H2)
        dgtf = dgtf0 * (1 + H2) + gtf0 * dH2
        gff = gff0 * (1 + H4)
        dgff = dgff0 * (1 + H4) + gff0 * dH4
        return gtt, gtf, gff, dgtt, dgtf, dgff


def _omega(gtt, gtf, gff, dgtt, dgtf, dgff, pro):
    disc = dgtf**2 - dgtt * dgff
    if disc < 0:
        return np.nan
    s = 1.0 if pro else -1.0
    return (-dgtf + s * np.sqrt(disc)) / dgff


def _Efun(em, r, ca, cb, pro):
    gtt, gtf, gff, dtt, dtf, dff = em.g(r, ca, cb)
    Om = _omega(gtt, gtf, gff, dtt, dtf, dff, pro)
    N2 = -(gtt + 2 * Om * gtf + Om**2 * gff)
    if N2 <= 0:
        return np.nan
    return -(gtt + Om * gtf) / np.sqrt(N2)


def _photon_res(em, r, ca, cb, pro):
    gtt, gtf, gff, dtt, dtf, dff = em.g(r, ca, cb)
    Om = _omega(gtt, gtf, gff, dtt, dtf, dff, pro)
    return gtt + 2 * Om * gtf + Om**2 * gff


def r_isco(em, ca, cb, pro, rlo=None, rhi=12.0):
    """ISCO radius: root of dE/dr."""
    rlo = rlo or em.rp + 1e-3
    dr = 1e-5
    dE = lambda r: (_Efun(em, r + dr, ca, cb, pro) - _Efun(em, r - dr, ca, cb, pro)) / (2 * dr)
    rs = np.linspace(rlo + 0.05, rhi, 800)
    vals = np.array([dE(r) for r in rs])
    ok = np.isfinite(vals)
    rs, vals = rs[ok], vals[ok]
    sign = np.where(np.diff(np.sign(vals)) != 0)[0]
    if len(sign) == 0:
        return np.nan
    i = sign[0]                                   # innermost dE/dr=0 -> ISCO
    return brentq(dE, rs[i], rs[i + 1], xtol=1e-10)


def r_photon(em, ca, cb, pro, rlo=None, rhi=8.0):
    """Photon-sphere radius: root of g_tt+2 Omega g_tphi+Omega^2 g_phiphi."""
    rlo = rlo or em.rp + 1e-3
    f = lambda r: _photon_res(em, r, ca, cb, pro)
    rs = np.linspace(rlo + 0.02, rhi, 800)
    vals = np.array([f(r) for r in rs])
    ok = np.isfinite(vals)
    rs, vals = rs[ok], vals[ok]
    sign = np.where(np.diff(np.sign(vals)) != 0)[0]
    if len(sign) == 0:
        return np.nan
    i = sign[0]
    return brentq(f, rs[i], rs[i + 1], xtol=1e-10)


# ---------------------------------------------------------------------- #
def mixed_shift(em, root_fn, pro, eps=1e-3):
    """Return r^(0) and (r^(1,0), r^(0,1), r^(1,1)) for a radius functional."""
    r0 = root_fn(em, 0.0, 0.0, pro)
    rpp = root_fn(em, +eps, +eps, pro); rpm = root_fn(em, +eps, -eps, pro)
    rmp = root_fn(em, -eps, +eps, pro); rmm = root_fn(em, -eps, -eps, pro)
    r10 = (rpp + rpm - rmp - rmm) / (4 * eps)
    r01 = (rpp - rpm + rmp - rmm) / (4 * eps)
    r11 = (rpp - rpm - rmp + rmm) / (4 * eps * eps)
    return r0, r10, r01, r11


def _selftest():
     """Validate the Kerr (uncorrected) ISCO/photon radii against Bardeen."""
     from spectral_grid import SpectralGrid
     from kerr_background import KerrBackground
 
     def kerr_isco(a, pro):
         Z1 = 1 + (1 - a**2)**(1/3) * ((1 + a)**(1/3) + (1 - a)**(1/3))
         Z2 = np.sqrt(3 * a**2 + Z1**2)
         return 3 + Z2 - (1 if pro else -1) * np.sqrt((3 - Z1) * (3 + Z1 + 2 * Z2))
 
     print("Kerr (eps=0) validation:")
     for a in (0.5, 0.8, 0.9):
         g = SpectralGrid(Nz=30, Nchi=30, spin=a); bg = KerrBackground(g)
         z0 = {i: np.zeros((g.Nz + 1, g.Nchi + 1)) for i in (1, 2, 3, 4)}
         em = EqMetric(bg, z0, z0, z0)
         for pro, tag in ((True, "pro"), (False, "retro")):
             ri = r_isco(em, 0.0, 0.0, pro); rp = r_photon(em, 0.0, 0.0, pro)
             print(f"  a={a} {tag:5s}: ISCO={ri:.4f} (Bardeen {kerr_isco(a,pro):.4f})  "
                   f"photon={rp:.4f}")


if __name__ == "__main__":
    from mixed_cache import load_mixed

    # Generates: mixed_a_0.1.json, mixed_a_0.12.json, ..., mixed_a_0.2.json, ..., mixed_a_0.9.json
    paths = [
        f"mixed_a_0.{i // 10}.json" if i % 10 == 0 else f"mixed_a_0.{i}.json"
        for i in range(10, 91, 2)
    ]

    header_fmt = "{:<5} {:<6} {:<12} {:<12} {:<12} {:<12}"
    print("=" * 68)
    print(header_fmt.format("a", "Orbit", "r^(0)", "r^(1,0)", "r^(0,1)", "r^(1,1)"))
    print("=" * 68)

    for p in paths:
        try:
            bg, H10, H01, H11, _, _ = load_mixed(p)
            em = EqMetric(bg, H10, H01, H11)
            spin = bg.a

            for pro, label in [(True, "pro"), (False, "retro")]:
                print(f"\n--- Spin a = {spin:.2f} ({label.upper()}) ---")
                
                # Compute ISCO Roots
                try:
                    r0_i, r10_i, r01_i, r11_i = mixed_shift(em, r_isco, pro=pro)
                    print(header_fmt.format(f"{spin:.2f}", "ISCO", 
                                            f"{r0_i:.5f}", f"{r10_i:+.5e}", 
                                            f"{r01_i:+.5e}", f"{r11_i:+.5e}"))
                except Exception as e:
                    print(f"  [ISCO Calculation Exception: {e}]")

                # Compute Photon Ring Roots
                try:
                    r0_p, r10_p, r01_p, r11_p = mixed_shift(em, r_photon, pro=pro)
                    print(header_fmt.format(f"{spin:.2f}", "Photon", 
                                            f"{r0_p:.5f}", f"{r10_p:+.5e}", 
                                            f"{r01_p:+.5e}", f"{r11_p:+.5e}"))
                except Exception as e:
                    print(f"  [Photon Sphere Calculation Exception: {e}]")
            print("-" * 68)

        except FileNotFoundError:
            print(f"[{p}] skipped: File not found")
        except Exception as file_err:
            print(f"⚠️ Error parsing file structure for {p}: {file_err}")