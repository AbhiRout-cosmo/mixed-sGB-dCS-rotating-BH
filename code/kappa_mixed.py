"""
kappa_mixed.py
==============

Mixed-order surface gravity kappa^(1,1) (leading parity-mixing correction to the
horizon temperature), computed from first principles -- the paper's Eq. (68) is
typo'd and no H_i-level formula exists in Lam's notebook.

Method (validated): surface gravity of a Killing horizon,
    kappa^2 = -1/2 (nabla_a xi_b)(nabla^a xi^b)|_{r+},   xi = d_t + Omega_H d_phi,
with nabla_a xi_b = 1/2(d_a xi_b - d_b xi_a) (Killing), evaluated near the horizon
and extrapolated to z=+1.  Applied to g^(0)+zeta*h it reproduces Lam's pure
kappa^(1,0)/kappa^(0,1) to ~5 digits.  The mixed coefficient is obtained by a
second finite difference in the two couplings of the FULL metric
    g(a,b) = g^(0) + a h^(1,0) + b h^(0,1) + a b h^(1,1).

Reports kappa^(1,1) at the equator (Lam's "true value" convention) with the
chi-spread as error bar; also prints kappa^(1,0)/kappa^(0,1) as a live check.
"""

from __future__ import annotations
import numpy as np

from spectral_grid import SpectralGrid
from kerr_background import KerrBackground
from perturbation import assemble_h
from stage1_solver import solve_mixed_scalars
from stage2_solver import build_operator, solve_mixed_metric
from stage2_source import build_source


def kappa_of_metric(bg, gdd, dg, nfit=6):
    """Surface gravity over chi for a metric (gdd, dg=d_c g).  Returns array(chi)."""
    g = bg.grid; chi = g.chi; z = g.z
    intr = bg.interior
    guu = np.full_like(gdd, np.nan)
    with np.errstate(all="ignore"):
        guu[intr] = np.linalg.inv(gdd[intr])
        Om = -gdd[..., 0, 3] / gdd[..., 3, 3]
    # horizon angular velocity (chi-independent): extrapolate equator column to z=1
    Om_eq = np.array([np.interp(0, chi, Om[r, :]) for r in range(1, nfit + 1)])
    OmH = np.polyval(np.polyfit(z[1:nfit + 1], Om_eq, nfit - 1), 1.0)

    dxi = dg[..., :, :, 0] + OmH * dg[..., :, :, 3]        # d_c xi_b
    K = np.zeros(bg.shape + (4, 4))
    for aa in range(4):
        for bb in range(4):
            da = dxi[..., aa, bb] if aa in (1, 2) else 0.0
            db = dxi[..., bb, aa] if bb in (1, 2) else 0.0
            K[..., aa, bb] = 0.5 * (da - db)
    with np.errstate(all="ignore"):
        kap2 = -0.5 * np.einsum("...ac,...bd,...ab,...cd->...", guu, guu, K, K)
    # extrapolate kappa^2 to horizon (z=1) per chi column
    out = np.empty(len(chi))
    for j in range(len(chi)):
        col = kap2[1:nfit + 1, j]
        out[j] = np.sqrt(abs(np.polyval(np.polyfit(z[1:nfit + 1], col, nfit - 1), 1.0)))
    return out


def _metric(bg, ca, cb, H10, H01, H11):
    """Full metric (gdd, dg) at couplings (ca, cb) incl. the ca*cb h^(1,1) term."""
    h10, dh10, _ = assemble_h(bg, H10)
    h01, dh01, _ = assemble_h(bg, H01)
    h11, dh11, _ = assemble_h(bg, H11)
    with np.errstate(invalid="ignore"):
        gdd = bg.g_dd + ca * h10 + cb * h01 + ca * cb * h11
        dg = bg.dg_ddd + ca * dh10 + cb * dh01 + ca * cb * dh11
    return gdd, dg


def kappa_mixed(bg, H10, H01, H11, eps=1e-3):
    """kappa^(1,0), kappa^(0,1), kappa^(1,1) over chi via finite differences."""
    kpp = kappa_of_metric(bg, *_metric(bg, +eps, +eps, H10, H01, H11))
    kpm = kappa_of_metric(bg, *_metric(bg, +eps, -eps, H10, H01, H11))
    kmp = kappa_of_metric(bg, *_metric(bg, -eps, +eps, H10, H01, H11))
    kmm = kappa_of_metric(bg, *_metric(bg, -eps, -eps, H10, H01, H11))
    k10 = (kpp + kpm - kmp - kmm) / (4 * eps)
    k01 = (kpp - kpm + kmp - kmm) / (4 * eps)
    k11 = (kpp - kpm - kmp + kmm) / (4 * eps * eps)
    return k10, k01, k11


def run_one(path, eps=1e-3):
    from mixed_cache import load_mixed
    bg, H10, H01, H11, p11, t11 = load_mixed(path)
    g = bg.grid
    k10, k01, k11 = kappa_mixed(bg, H10, H01, H11, eps)
    chi = g.chi
    inner = slice(3, -3)                       # drop pole columns for the error bar
    def eqspread(v):
        eq = float(np.interp(0, chi, v))
        seg = v[inner]; seg = seg[np.isfinite(seg)]
        return eq, float(np.max(seg) - np.min(seg))
    return g.spin, eqspread(k11), eqspread(k10), eqspread(k01)


if __name__ == "__main__":
    import sys
    paths = sys.argv[1:] or [f"mixed_a_0.{s}.json" for s in
                             ["1","12","14","16","18", "2","22","24","26","28","3","32","34","36","38", "4","42","44","46","48", "5","52","54","56","58","6","62","64","66","68", "7","72","74","76","78","8","82","84","86","88","9"]]
    print(f"{'a':>5}  {'kappa^(1,1)':>14} {'chi-spread':>11}  "
          f"{'kappa^(1,0)_sGB':>15} {'kappa^(0,1)_dCS':>15}")
    for p in paths:
        try:
            a, (k11, s11), (k10, _), (k01, _) = run_one(p)
            print(f"{a:>5.2f}  {k11:>+14.6e} {s11:>11.2e}  "
                  f"{k10:>+15.6e} {k01:>+15.6e}")
        except FileNotFoundError:
            print(f"[skip] {p} not found")
