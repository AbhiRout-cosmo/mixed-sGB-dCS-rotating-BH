"""
correction_tensors.py
======================

Leading-order curvature-coupling tensors for the combined sGB + dCS theory,
built on the Ricci-flat Kerr background (kerr_background.KerrBackground).

On the Kerr background both leading tensors collapse to a covariant scalar
Hessian contracted with background curvature (report Eqs. 83, 92):

    H^{(1,0)}_{mu nu} = 4 R^{(0)}_{mu rho nu sig}  nabla^rho nabla^sig phi^{(1/2,0)}
    C^{(0,1)}_{mu nu} = (nabla^alpha nabla^beta theta^{(0,1/2)}) *R^{(0)}_{alpha beta (mu nu)}

Neither needs the metric deformation h_{mu nu}: that only enters at mixed order
and in the scalar sources, handled in a later module.

Index conventions match kerr_background:
    COORD = (t, r, chi, phi);  M = 1;  signature (-,+,+,+).
    Riemann_dddd[..., a, b, c, d] = R_{abcd}
    dualRiemann_uudd[..., a, b, c, d] = *R^{ab}{}_{cd} = 1/2 eps^{ab}{}_{ef} R^{ef}{}_{cd}

Vacuum self-checks (used by the self-test):
    * Both tensors are symmetric in (mu nu).
    * Both are trace-free w.r.t. g^{mu nu}, because g^{mu nu} R_{mu rho nu sig} = R_{rho sig} = 0
      and the corresponding contraction of the dual Riemann also vanishes on a
      Ricci-flat background.  These are genuine tests of the Hessian + curvature chain.
"""

from __future__ import annotations

import numpy as np

from kerr_background import KerrBackground


# ---------------------------------------------------------------------- #
# covariant Hessian of a stationary axisymmetric scalar f(r, chi)
# ---------------------------------------------------------------------- #
def covariant_hessian(bg: KerrBackground, coeffs: np.ndarray) -> np.ndarray:
    """
    nabla_mu nabla_nu f  for a scalar field f(r, chi) given by spectral `coeffs`.

        nabla_mu nabla_nu f = d_mu d_nu f - Gamma^lambda_{mu nu} d_lambda f

    f depends only on (r, chi) -> only the r,chi partial derivatives are non-zero,
    but the Christoffel term feeds curvature into all four components.
    Returns an array of shape (..., 4, 4).  Boundary nodes (horizon/pole/inf)
    inherit NaN from the background Christoffels -- mask with bg.interior.
    """
    g = bg.grid
    f_r = g.d_dr(coeffs)
    f_c = g.d_dchi(coeffs)
    f_rr = g.d2_dr2(coeffs)
    f_cc = g.d2_dchi2(coeffs)
    f_rc = g._dz_dr[:, None] * g.d_dz_dchi(coeffs)      # d^2 f / dr dchi

    grad = np.zeros(bg.shape + (4,))
    grad[..., 1] = f_r
    grad[..., 2] = f_c

    d2 = np.zeros(bg.shape + (4, 4))
    d2[..., 1, 1] = f_rr
    d2[..., 2, 2] = f_cc
    d2[..., 1, 2] = d2[..., 2, 1] = f_rc

    return d2 - np.einsum("...lmn,...l->...mn", bg.Gamma, grad)


# ---------------------------------------------------------------------- #
# leading correction tensors
# ---------------------------------------------------------------------- #
def H_sGB_leading(bg: KerrBackground, phi_coeffs: np.ndarray) -> np.ndarray:
    """
    H^{(1,0)}_{mu nu} = 4 R^{(0)}_{mu rho nu sig} nabla^rho nabla^sig phi^{(1/2,0)}.
    (report Eq. 83)  Symmetric and trace-free on Kerr.
    """
    hess = covariant_hessian(bg, phi_coeffs)                         # nabla_a nabla_b phi
    hess_uu = np.einsum("...ma,...nb,...ab->...mn", bg.g_uu, bg.g_uu, hess)
    return 4.0 * np.einsum("...mrns,...rs->...mn", bg.Riemann_dddd, hess_uu)


def C_dCS_leading(bg: KerrBackground, theta_coeffs: np.ndarray) -> np.ndarray:
    """
    C^{(0,1)}_{mu nu} = (nabla^alpha nabla^beta theta^{(0,1/2)}) *R^{(0)}_{alpha (mu nu) beta}.
    (report Eqs. 15/92)  The free indices mu, nu split across the two curvature
    pairs, so the result is symmetrized on (mu nu).  Symmetric, trace-free on Kerr.
    """
    hess = covariant_hessian(bg, theta_coeffs)
    hess_uu = np.einsum("...ma,...nb,...ab->...mn", bg.g_uu, bg.g_uu, hess)
    # all-lower dual Riemann  *R_{a b c d}
    dualR_dddd = np.einsum("...ai,...bj,...ijcd->...abcd",
                           bg.g_dd, bg.g_dd, bg.dualRiemann_uudd)
    # contract the Hessian on the OUTER indices (a, d); free (mu, nu) are the inner pair.
    # overall -4 fixes the dual-Riemann pairing/normalization (verified against the
    # pure-dCS Einstein equation G^(1)[h^(0,1)]+C^(0,1)=T^(theta)/2).
    C = np.einsum("...ad,...amnd->...mn", hess_uu, dualR_dddd)
    C = 0.5 * (C + np.swapaxes(C, -1, -2))                          # symmetrize (mu nu)
    return -4.0 * C


# ---------------------------------------------------------------------- #
# helpers
# ---------------------------------------------------------------------- #
def _trace(bg: KerrBackground, T_dd: np.ndarray) -> np.ndarray:
    """g^{mu nu} T_{mu nu}."""
    return np.einsum("...mn,...mn->...", bg.g_uu, T_dd)


# ====================================================================== #
# self-test
# ====================================================================== #
def _selftest(path: str = "data_a_50_0.99.json") -> None:
    import os

    print("=" * 68)
    print("correction_tensors self-test")
    print("=" * 68)

    from spectral_grid import SpectralGrid

    # --- vacuum-identity test with a SYNTHETIC scalar (no data file needed) ---
    grid = SpectralGrid(Nz=30, Nchi=30, spin=0.9)
    bg = KerrBackground(grid)
    intr = bg.interior

    R, C = bg.R, bg.CHI
    smooth = np.where(np.isfinite(R), np.exp(-(R - grid.r_plus)), 0.0)
    f_even = grid.fit(smooth * (1.0 + 0.3 * C**2), parity="even")     # even-l scalar
    f_odd = grid.fit(smooth * C, parity="odd")                        # odd-l scalar

    H = H_sGB_leading(bg, f_even)
    Cc = C_dCS_leading(bg, f_odd)

    for name, T in (("H^(1,0)", H), ("C^(0,1)", Cc)):
        scale = np.max(np.abs(T[intr]))
        sym = np.max(np.abs(T[intr] - np.swapaxes(T, -1, -2)[intr]))
        tr = np.max(np.abs(_trace(bg, T)[intr]))
        finite = np.all(np.isfinite(T[intr]))
        print(f"\n[{name}]  (synthetic scalar)")
        print(f"    max|T|_interior         : {scale:.3e}")
        print(f"    max|T-T^T| (symmetry)   : {sym:.3e}   (expect ~1e-16)")
        print(f"    max|g^ab T_ab| (trace)  : {tr:.3e}   rel {tr/scale:.2e}")
        print(f"    all finite on interior  : {finite}")
        assert finite
        assert sym < 1e-10 * max(scale, 1.0)
        assert tr / scale < 1e-3, f"{name} not trace-free -> index chain suspect"

    # --- real-data test (report phi^(1/2,0), theta^(0,1/2)) if available ------
    if os.path.exists(path):
        g2, raw = SpectralGrid.from_file(path)
        bg2 = KerrBackground(g2)
        phi = g2.fit(g2.field_values(raw, "sGBPhi"), parity="even")
        th = g2.fit(g2.field_values(raw, "dCSPhi"), parity="odd")
        H2 = H_sGB_leading(bg2, phi)
        C2 = C_dCS_leading(bg2, th)
        for name, T in (("H^(1,0)", H2), ("C^(0,1)", C2)):
            scale = np.max(np.abs(T[bg2.interior]))
            tr = np.max(np.abs(_trace(bg2, T)[bg2.interior]))
            print(f"\n[{name}]  (data spin={g2.spin})  max|T|={scale:.3e}  "
                  f"rel-trace={tr/scale:.2e}")
    else:
        print(f"\n[data]  '{path}' not found; skipping real-field test.")

    print("\nself-test PASSED\n")


if __name__ == "__main__":
    import sys
    _selftest(sys.argv[1] if len(sys.argv) > 1 else "data_a_50_0.99.json")
