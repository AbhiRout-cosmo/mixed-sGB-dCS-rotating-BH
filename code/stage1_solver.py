"""
stage1_solver.py
================

Stage I of the mixed sector: solve the two decoupled linear scalar equations
for the cross-sourced fields (report Eqs. 141, 144)

    box^(0) phi^(1/2,1) = -( box^(0,1) phi^(1/2,0) + G^(0,1) )      [even-l]
    box^(0) theta^(1,1/2) = -( box^(1,0) theta^(0,1/2) + (*RR)^(1,0) )  [odd-l]

box^(0) is the regular Kerr d'Alembertian (KerrBackground.box0).  It is built
as a spectral operator matrix acting on parity-restricted Chebyshev(z)xLegendre(chi)
coefficients; the homogeneous null space (box0(const)=0) is fixed by imposing
decay f -> 0 at spatial infinity (z = -1).
"""

from __future__ import annotations

import numpy as np

from kerr_background import KerrBackground
from perturbation import (assemble_h, gb_invariant_pert, pontryagin_pert,
                          box_pert)


def _parity_ls(N, parity):
    return [l for l in range(N + 1)
            if (parity == "none")
            or (parity == "even" and l % 2 == 0)
            or (parity == "odd" and l % 2 == 1)]


def solve_box0(bg: KerrBackground, source_vals: np.ndarray,
               parity: str, bc_inf: np.ndarray | None = None,
               trunc: int = 3) -> np.ndarray:
    """
    Solve box0 f = source_vals (grid values) for a scalar f of given chi-parity.
    Decay values at r=inf (z=-1) are set by `bc_inf` (per-chi; default 0).
    Returns spectral coefficients C[n, l] of f.

    box0 annihilates high-frequency modes on the collocation grid, so the
    solution basis is truncated by `trunc` (top Chebyshev orders and top
    parity-Legendre order) for full column rank; the single remaining null
    mode (box0(const)=0) is fixed by the infinity BC.
    """
    g = bg.grid
    Nz, Nchi = g.Nz, g.Nchi
    ls = _parity_ls(Nchi, parity)
    lmax = max(ls) - 2 * (trunc - 1) if len(ls) > trunc else max(ls)
    cols = [(n, l) for l in ls if l <= lmax for n in range(Nz + 1 - trunc)]
    intr = bg.interior

    Aop = np.zeros((int(intr.sum()), len(cols)))
    Abc = np.zeros((Nchi + 1, len(cols)))
    for j, (n, l) in enumerate(cols):
        e = np.zeros((Nz + 1, Nchi + 1)); e[n, l] = 1.0
        Aop[:, j] = bg.box0(e)[intr]
        Abc[:, j] = g.to_grid(e)[-1, :]              # value at z=-1 (r=inf)

    if bc_inf is None:
        bc_inf = np.zeros(Nchi + 1)
    A = np.vstack([Aop, Abc])
    b = np.concatenate([source_vals[intr], bc_inf])
    csol, *_ = np.linalg.lstsq(A, b, rcond=None)

    C = np.zeros((Nz + 1, Nchi + 1))
    for j, (n, l) in enumerate(cols):
        C[n, l] = csol[j]
    return C


# ---------------------------------------------------------------------- #
# full Stage-I driver: build sources from lower-order data, solve
# ---------------------------------------------------------------------- #
def solve_mixed_scalars(bg, phi, theta, Hsg, Hcs):
    """
    Returns (phi_11, theta_11) : spectral coeffs of phi^(1/2,1) (even) and
    theta^(1,1/2) (odd), the mixed-order scalar fields.
    Inputs are the pure-sector spectral coeffs / H_i coeff dicts.
    """
    h_sg, dh_sg, d2h_sg = assemble_h(bg, Hsg)
    h_cs, dh_cs, d2h_cs = assemble_h(bg, Hcs)

    # dilaton cross-source (Eq. 141): box0 phi11 = -(box^(0,1)phi + G^(0,1))
    s_phi = -(box_pert(bg, h_cs, dh_cs, phi)
              + gb_invariant_pert(bg, h_cs, dh_cs, d2h_cs))
    # axion cross-source (Eq. 144): box0 th11 = -(box^(1,0)th + (*RR)^(1,0))
    s_th = -(box_pert(bg, h_sg, dh_sg, theta)
             + pontryagin_pert(bg, h_sg, dh_sg, d2h_sg))

    phi_11 = solve_box0(bg, s_phi, parity="even")
    theta_11 = solve_box0(bg, s_th, parity="odd")
    return phi_11, theta_11


# ====================================================================== #
# self-test
# ====================================================================== #
def _selftest(path: str = "data_a_50_0.99.json") -> None:
    import os
    from spectral_grid import SpectralGrid

    print("=" * 68)
    print("stage1_solver self-test")
    print("=" * 68)
    if not os.path.exists(path):
        print(f"[data] '{path}' not found; exiting.")
        return

    g, raw = SpectralGrid.from_file(path)
    bg = KerrBackground(g)
    intr = bg.interior

    # (A) manufactured-solution test: pick f_true, s = box0 f_true, recover f_true
    for parity, seed in (("even", 1), ("odd", 2)):
        rng = np.random.default_rng(seed)
        Ct = np.zeros((g.Nz + 1, g.Nchi + 1))
        for l in _parity_ls(g.Nchi, parity):
            if l > 8:
                continue
            Ct[:6, l] = rng.standard_normal(6) * 0.5 ** np.arange(6)  # smooth, decaying
        f_true = g.to_grid(Ct)
        s = bg.box0(Ct)
        Csol = solve_box0(bg, s, parity, bc_inf=f_true[-1, :])
        err = np.max(np.abs((g.to_grid(Csol) - f_true)[intr]))
        scale = np.max(np.abs(f_true[intr]))
        print(f"\n[A:{parity}] manufactured recover  rel {err/scale:.2e}  "
              f"(scale {scale:.2e})")
        assert err / scale < 1e-8

    # (B) real Stage-I solve
    Hsg = {i: g.fit(g.field_values(raw, f"sGBH{i}"), "even") for i in (1, 2, 3, 4)}
    Hcs = {i: g.fit(g.field_values(raw, f"dCSH{i}"), "even") for i in (1, 2, 3, 4)}
    phi = g.fit(g.field_values(raw, "sGBPhi"), "even")
    th = g.fit(g.field_values(raw, "dCSPhi"), "odd")

    phi11, th11 = solve_mixed_scalars(bg, phi, th, Hsg, Hcs)

    # residual: box0(solution) - source  (should be ~ solver tolerance)
    h_cs, dh_cs, d2h_cs = assemble_h(bg, Hcs)
    h_sg, dh_sg, d2h_sg = assemble_h(bg, Hsg)
    s_phi = -(box_pert(bg, h_cs, dh_cs, phi) + gb_invariant_pert(bg, h_cs, dh_cs, d2h_cs))
    s_th = -(box_pert(bg, h_sg, dh_sg, th) + pontryagin_pert(bg, h_sg, dh_sg, d2h_sg))
    rphi = np.max(np.abs((bg.box0(phi11) - s_phi)[intr])) / np.max(np.abs(s_phi[intr]))
    rth = np.max(np.abs((bg.box0(th11) - s_th)[intr])) / np.max(np.abs(s_th[intr]))
    print(f"\n[B] phi^(1/2,1)  box0-residual  rel {rphi:.2e}   "
          f"max|phi11|={np.max(np.abs(g.to_grid(phi11)[intr])):.3e}")
    print(f"[B] theta^(1,1/2) box0-residual rel {rth:.2e}   "
          f"max|th11| ={np.max(np.abs(g.to_grid(th11)[intr])):.3e}")
    assert rphi < 1e-3 and rth < 1e-3

    print("\nself-test PASSED\n")


if __name__ == "__main__":
    import sys
    _selftest(sys.argv[1] if len(sys.argv) > 1 else "data_a_50_0.99.json")
