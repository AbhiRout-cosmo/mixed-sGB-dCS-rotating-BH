"""
stage2_solver.py
================

Stage II of the mixed sector: solve the linear metric equation (report Eq. 149)

    G^(1)_{mu nu}[ h^(1,1) ] = S_{mu nu}

for the mixed metric deformation h^(1,1), expanded in the same 4-function
ansatz (Eq. 35) as the pure sectors.  G^(1) is the validated linearized
Einstein operator (einstein_operator.G1_lin).

The unknowns are the even-l spectral coefficients of H_1..H_4 (the odd,
non-circular components vanish at this order).  The 5 independent even-parity
tensor components (tt, t-phi, rr, chi-chi, phi-phi) are collocated on the
interior; decay (H_i -> 0 at r=inf) fixes the asymptotic behaviour.  The source
S_{mu nu} is assembled separately (Stage-II source module, next step).

This file provides the inversion harness and a manufactured-solution test.
"""

from __future__ import annotations

import numpy as np

from kerr_background import KerrBackground
from perturbation import assemble_h
from einstein_operator import G1_lin
from stage1_solver import _parity_ls

# the 5 independent even-parity components carried by the H_i ansatz
COMPS = [(0, 0), (0, 3), (1, 1), (2, 2), (3, 3)]


def _g1_of_Hcoeffs(bg, Hc):
    """G^(1)[h(H_i)] tensor for a given dict of H_i coefficient arrays."""
    h, dh, d2h = assemble_h(bg, Hc)
    return G1_lin(bg, h, dh, d2h)


def build_operator(bg, trunc=3, bc="peeling"):
    """
    Assemble the linear map  (H_1..4 coeffs) -> [G^(1) even comps on interior ; BC].
    bc = "peeling" (Eq. 37, production) or "decay" (H_i(inf)=value; operator test).
    Solves the even (circular) sector; the odd rchi component is handled separately.
    """
    g = bg.grid
    Nz, Nchi = g.Nz, g.Nchi
    ls = _parity_ls(Nchi, "even")
    lmax = max(ls) - 2 * (trunc - 1)
    ncols = [(n, l) for l in ls if l <= lmax for n in range(Nz + 1 - trunc)]
    cols = [(i, n, l) for i in (1, 2, 3, 4) for (n, l) in ncols]

    intr = bg.interior
    nrow_op = int(intr.sum()) * len(COMPS)
    nb = Nchi + 1
    nrow_bc = nb * 4
    A = np.zeros((nrow_op + nrow_bc, len(cols)))
    rp = bg.grid.r_plus
    b0 = nrow_op

    z0 = np.zeros((Nz + 1, Nchi + 1))
    for j, (i, n, l) in enumerate(cols):
        e = np.zeros((Nz + 1, Nchi + 1)); e[n, l] = 1.0
        Hc = {k: (e if k == i else z0) for k in (1, 2, 3, 4)}
        G = _g1_of_Hcoeffs(bg, Hc)
        A[:nrow_op, j] = np.concatenate([G[..., a, b][intr] for (a, b) in COMPS])
        val = g.to_grid(e)[-1, :]
        if bc == "decay":
            A[b0 + (i - 1) * nb:b0 + i * nb, j] = val
            continue
        dz = g.d_dz(e)[-1, :]                        # peeling BCs (Eq. 37, M=1)
        if i == 1:
            A[b0:b0 + nb, j] = val
        if i == 2:
            A[b0 + nb:b0 + 2 * nb, j] = val
        if i == 3:
            A[b0 + nb:b0 + 2 * nb, j] += -rp * dz
            A[b0 + 2 * nb:b0 + 3 * nb, j] = val + 2 * rp * dz
            A[b0 + 3 * nb:b0 + 4 * nb, j] += 2 * rp * dz
        if i == 4:
            A[b0 + 3 * nb:b0 + 4 * nb, j] += val
    return A, cols


def solve_mixed_metric(bg, S, trunc=3, A_cols=None):
    """
    Solve G^(1)[h^(1,1)] = S for the H_i^(1,1) coefficients.
    S : (...,4,4) source tensor.  Returns dict {i: coeff array}.
    """
    g = bg.grid
    intr = bg.interior
    A, cols = A_cols if A_cols is not None else build_operator(bg, trunc)
    b = np.concatenate([np.concatenate([S[..., a, b][intr] for (a, b) in COMPS]),
                        np.zeros((g.Nchi + 1) * 4)])
    # equilibrate: the near-horizon rows (G^(1) ~ Sigma/Delta) otherwise dominate
    # and drive cond(A) ~ 1e14; row + column scaling brings it to ~1e5 and makes
    # the solution stable w.r.t. truncation/resolution.
    rw = 1.0 / np.maximum(np.max(np.abs(A), axis=1), 1e-300)
    Aw = A * rw[:, None]
    bw = b * rw
    cw = 1.0 / np.maximum(np.linalg.norm(Aw, axis=0), 1e-300)
    sol, *_ = np.linalg.lstsq(Aw * cw[None, :], bw, rcond=None)
    sol = sol * cw
    Hc = {i: np.zeros((g.Nz + 1, g.Nchi + 1)) for i in (1, 2, 3, 4)}
    for (i, n, l), v in zip(cols, sol):
        Hc[i][n, l] = v
    return Hc


# ====================================================================== #
# self-test: manufactured solution
# ====================================================================== #
def _selftest(path: str = "data_a_50_0.99.json") -> None:
    import os
    from spectral_grid import SpectralGrid

    print("=" * 68)
    print("stage2_solver self-test (manufactured solution)")
    print("=" * 68)
    # moderate resolution for a fast operator build
    spin = 0.9
    if os.path.exists(path):
        _, raw = SpectralGrid.from_file(path); spin = float(raw["spin"])
    g = SpectralGrid(Nz=20, Nchi=20, spin=spin)
    bg = KerrBackground(g)
    intr = bg.interior

    # target H_i (even, smooth, decaying), build S = G^(1)[h(H_target)]
    rng = np.random.default_rng(0)
    Ht = {}
    for i in (1, 2, 3, 4):
        C = np.zeros((g.Nz + 1, g.Nchi + 1))
        for l in _parity_ls(g.Nchi, "even"):
            if l <= 4:
                C[:5, l] = rng.standard_normal(5) * 0.4 ** np.arange(5)
        Ht[i] = C
    S = _g1_of_Hcoeffs(bg, Ht)
    bc_true = {i: g.to_grid(Ht[i])[-1, :] for i in (1, 2, 3, 4)}

    print("\nbuilding G^(1) operator (this is the slow step)...")
    A, cols = build_operator(bg, trunc=3, bc="decay")   # operator test: plain decay BC
    print(f"operator shape {A.shape}, rank {np.linalg.matrix_rank(A)}")

    # solve with the true asymptotic values as BC
    b = np.concatenate([np.concatenate([S[..., a, b][intr] for (a, b) in COMPS]),
                        np.concatenate([bc_true[i] for i in (1, 2, 3, 4)])])
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    Hsol = {i: np.zeros((g.Nz + 1, g.Nchi + 1)) for i in (1, 2, 3, 4)}
    for (i, n, l), v in zip(cols, sol):
        Hsol[i][n, l] = v

    # compare recovered h to target h (gauge/normalization-independent check)
    ht = assemble_h(bg, Ht)[0]
    hs = assemble_h(bg, Hsol)[0]
    with np.errstate(invalid="ignore"):
        err = np.nanmax(np.abs((ht - hs)[intr]))
        sc = np.nanmax(np.abs(ht[intr]))
    print(f"\nrecovered h^(1,1) vs target : rel {err/sc:.2e}  (scale {sc:.2e})")

    # residual of the solve
    res = np.max(np.abs((_g1_of_Hcoeffs(bg, Hsol) - S)[intr]))
    print(f"G^(1)[h_sol]-S residual     : {res:.2e}")
    assert err / sc < 1e-2          # ~1e-5 typical; one near-null mode at low spin
    print("\nself-test PASSED\n")


if __name__ == "__main__":
    import sys
    _selftest(sys.argv[1] if len(sys.argv) > 1 else "data_a_50_0.99.json")
