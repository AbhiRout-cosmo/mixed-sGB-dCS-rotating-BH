"""
mixed_observables.py
====================

Horizon observables at mixed order O(zeta_GB zeta_CS) from the metric functions.

With the ansatz the (t,phi) and (phi,phi) deformations are multiplicative,
    g_tphi = g^(0)_tphi (1 + zeta H2),   g_phiphi = g^(0)_phiphi (1 + zeta H4),
so the horizon angular velocity Omega_H = |g_tphi|/g_phiphi expands as
Omega^(0) * N/D with N (D) carrying H2 (H4).  Collecting the zeta_GB zeta_CS
term of N/D gives

  Omega^(1,1) = (a/2 r+) [ (H2^(1,1) - H4^(1,1))
                           - H2^(1,0) H4^(0,1) - H2^(0,1) H4^(1,0)
                           + 2 H4^(1,0) H4^(0,1) ]                     (horizon)

The first bracket is the linear response to the new mixed metric h^(1,1); the
remaining products are cross terms from the pure sectors (H^(1,0): sGB,
H^(0,1): dCS).  Omega^(1,1) must be chi-independent (zeroth law) -- the
chi-spread is the built-in validation.

The pure-sector Omega^(1,0)/Omega^(0,1) = (a/2r+)(H2 - H4) is reproduced by
`omega_pure` (validated against Lam et al. to ~1e-7).

M = 1.  Surface gravity kappa^(1,1) requires the corrected Eq.(68) and its own
second-order expansion; it is left out until the pure-sector kappa formula is
fixed.
"""

from __future__ import annotations
import numpy as np


def _hz(bg, Hc, i):
    """Horizon (z=+1) values of H_i over chi."""
    return bg.grid.to_grid(Hc[i])[0, :]


def omega_pure(bg, Hc):
    """Pure-sector Omega^(1) = (a/2r+)(H2 - H4)|_horizon  (validated)."""
    a, rp = bg.a, bg.grid.r_plus
    return a / (2 * rp) * (_hz(bg, Hc, 2) - _hz(bg, Hc, 4))


def omega_mixed(bg, H10, H01, H11):
    """Mixed-order Omega^(1,1) over chi (see module docstring)."""
    a, rp = bg.a, bg.grid.r_plus
    H2_10, H4_10 = _hz(bg, H10, 2), _hz(bg, H10, 4)
    H2_01, H4_01 = _hz(bg, H01, 2), _hz(bg, H01, 4)
    H2_11, H4_11 = _hz(bg, H11, 2), _hz(bg, H11, 4)
    lin = H2_11 - H4_11
    cross = -H2_10 * H4_01 - H2_01 * H4_10 + 2 * H4_10 * H4_01
    return a / (2 * rp) * (lin + cross)


def _eq_spread(vals, chi):
    return float(np.interp(0.0, chi, vals)), float(np.max(vals) - np.min(vals))


# ====================================================================== #
def _selftest(path="data_a_0_5.json"):
    import os
    from spectral_grid import SpectralGrid
    from kerr_background import KerrBackground
    from stage1_solver import solve_mixed_scalars
    from stage2_solver import build_operator, solve_mixed_metric
    from stage2_source import build_source

    print("=" * 68)
    print("mixed_observables: Omega^(1,1) from h^(1,1)")
    print("=" * 68)
    if not os.path.exists(path):
        print(f"[data] '{path}' not found; exiting."); return

    g, raw = SpectralGrid.from_file(path)
    bg = KerrBackground(g)
    H10 = {i: g.fit(g.field_values(raw, f"sGBH{i}"), "even") for i in (1, 2, 3, 4)}
    H01 = {i: g.fit(g.field_values(raw, f"dCSH{i}"), "even") for i in (1, 2, 3, 4)}
    p10 = g.fit(g.field_values(raw, "sGBPhi"), "even")
    t10 = g.fit(g.field_values(raw, "dCSPhi"), "odd")

    print("[1] pure-sector Omega (validation):")
    for nm, Hc in (("sGB", H10), ("dCS", H01)):
        eq, sp = _eq_spread(omega_pure(bg, Hc), g.chi)
        print(f"    Omega^(1)_{nm} = {eq:+.6e}   chi-spread {sp:.1e}")

    print("[2] solving Stage I + II for h^(1,1) ...")
    p11, t11 = solve_mixed_scalars(bg, p10, t10, H10, H01)
    S = build_source(bg, p10, t10, p11, t11, H10, H01)
    A, cols = build_operator(bg, trunc=3)
    H11 = solve_mixed_metric(bg, S, A_cols=(A, cols))

    om11 = omega_mixed(bg, H10, H01, H11)
    eq, sp = _eq_spread(om11, g.chi)
    # decompose to show the cross-term contribution
    a, rp = bg.a, g.r_plus
    lin = a / (2 * rp) * (_hz(bg, H11, 2) - _hz(bg, H11, 4))
    print(f"\n[3] MIXED Omega^(1,1) (a={g.spin}):")
    print(f"    Omega^(1,1)          = {eq:+.6e}   chi-spread {sp:.1e}")
    print(f"    (linear-in-h11 part  = {float(np.interp(0,g.chi,lin)):+.6e},"
          f"  cross-term part = {eq-float(np.interp(0,g.chi,lin)):+.6e})")
    print()


if __name__ == "__main__":
    import sys
    _selftest(sys.argv[1] if len(sys.argv) > 1 else "data_a_0_5.json")
