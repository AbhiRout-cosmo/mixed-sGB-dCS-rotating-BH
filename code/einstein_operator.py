"""
einstein_operator.py
====================

Linearized Einstein operator G^(1)_{mu nu}[h] on the Ricci-flat Kerr
background (report Eqs. 65, 68, 72), plus the scalar stress-energy tensor.

G^(1)[h] is the operator inverted in Stage II.  It is validated here against
the pure-sector Einstein equations (Eqs. 101 / 103), whose every other term is
already known:

    G^(1)[h^(1,0)] + H^(1,0) = 1/2 T^(phi)(1,0)      (sGB)
    G^(1)[h^(0,1)] + C^(0,1) = 1/2 T^(theta)(0,1)    (dCS)
"""

from __future__ import annotations

import numpy as np

from kerr_background import KerrBackground
from perturbation import assemble_h, covariant_derivs_h
from correction_tensors import H_sGB_leading, C_dCS_leading


def stress_energy_scalar(bg: KerrBackground, coeffs: np.ndarray) -> np.ndarray:
    """T_{mu nu} = d_mu f d_nu f - 1/2 g_{mu nu} (grad f)^2  for scalar f."""
    g = bg.grid
    grad = np.zeros(bg.shape + (4,))
    grad[..., 1] = g.d_dr(coeffs)
    grad[..., 2] = g.d_dchi(coeffs)
    sq = np.einsum("...mn,...m,...n->...", bg.g_uu, grad, grad)
    return np.einsum("...m,...n->...mn", grad, grad) - 0.5 * bg.g_dd * sq[..., None, None]


def G1_lin(bg: KerrBackground, h, dh, d2h) -> np.ndarray:
    """
    Linearized Einstein tensor G^(1)_{mu nu}[h] on Kerr (Ricci-flat).
        R^(1)_{mu nu} = -1/2 box h_{mu nu} - 1/2 nabla_mu nabla_nu h
                        + nabla_alpha nabla_{(mu} h^alpha_{nu)} - R_{mu a nu b} h^{ab}
        R^(1)        = -box h + nabla_mu nabla_nu h^{mu nu}
        G^(1)_{mu nu} = R^(1)_{mu nu} - 1/2 g_{mu nu} R^(1)
    """
    guu = bg.g_uu
    _, DDh = covariant_derivs_h(bg, h, dh, d2h)          # DDh[rho,nu,mu,sig]=nabla_rho nabla_nu h_{mu sig}

    # exact covariant linearized Ricci (Riemann is implicit in the DDh ordering):
    #   dR_{mu nu} = 1/2( nabla^a nabla_mu h_{a nu} + nabla^a nabla_nu h_{a mu}
    #                     - box h_{mu nu} - nabla_mu nabla_nu h )
    t1 = np.einsum("...ab,...bman->...mn", guu, DDh)      # nabla^a nabla_mu h_{a nu}
    t2 = np.einsum("...ab,...bnam->...mn", guu, DDh)      # nabla^a nabla_nu h_{a mu}
    box_h = np.einsum("...ab,...abmn->...mn", guu, DDh)   # box h_{mu nu}
    hess_tr = np.einsum("...ab,...mnab->...mn", guu, DDh) # nabla_mu nabla_nu h
    R1_dd = 0.5 * (t1 + t2 - box_h - hess_tr)

    box_tr = np.einsum("...mn,...mn->...", guu, hess_tr)              # box h (trace)
    divdiv = np.einsum("...ma,...nb,...mnab->...", guu, guu, DDh)     # nabla_mu nabla_nu h^{mu nu}
    R1 = -box_tr + divdiv

    return R1_dd - 0.5 * bg.g_dd * R1[..., None, None]


# ====================================================================== #
# self-test: pure-sector Einstein residuals
# ====================================================================== #
def _selftest(path: str = "data_a_50_0.99.json") -> None:
    import os
    from spectral_grid import SpectralGrid

    print("=" * 68)
    print("einstein_operator self-test")
    print("=" * 68)
    if not os.path.exists(path):
        print(f"[data] '{path}' not found; exiting.")
        return

    g, raw = SpectralGrid.from_file(path)
    bg = KerrBackground(g)
    intr = bg.interior

    Hsg = {i: g.fit(g.field_values(raw, f"sGBH{i}"), "even") for i in (1, 2, 3, 4)}
    Hcs = {i: g.fit(g.field_values(raw, f"dCSH{i}"), "even") for i in (1, 2, 3, 4)}
    phi = g.fit(g.field_values(raw, "sGBPhi"), "even")
    th = g.fit(g.field_values(raw, "dCSPhi"), "odd")

    # sGB: G^(1)[h^(1,0)] + H^(1,0) - 1/2 T^(phi)
    h, dh, d2h = assemble_h(bg, Hsg)
    res_sg = (G1_lin(bg, h, dh, d2h) + H_sGB_leading(bg, phi)
              - 0.5 * stress_energy_scalar(bg, phi))
    # scale from the individual terms
    sc_sg = np.max(np.abs(H_sGB_leading(bg, phi)[intr]))
    print(f"\n[sGB] |G^(1)[h]+H^(1,0)-T/2|_inf : {np.max(np.abs(res_sg[intr])):.3e}"
          f"   rel {np.max(np.abs(res_sg[intr]))/sc_sg:.2e}")

    # dCS: G^(1)[h^(0,1)] + C^(0,1) - 1/2 T^(theta)
    h, dh, d2h = assemble_h(bg, Hcs)
    res_cs = (G1_lin(bg, h, dh, d2h) + C_dCS_leading(bg, th)
              - 0.5 * stress_energy_scalar(bg, th))
    sc_cs = np.max(np.abs(C_dCS_leading(bg, th)[intr]))
    print(f"[dCS] |G^(1)[h]+C^(0,1)-T/2|_inf : {np.max(np.abs(res_cs[intr])):.3e}"
          f"   rel {np.max(np.abs(res_cs[intr]))/sc_cs:.2e}")

    print("\n(expected: data noise floor ~1e-6..1e-3 of the exported H/phi data)\n")


if __name__ == "__main__":
    import sys
    _selftest(sys.argv[1] if len(sys.argv) > 1 else "data_a_50_0.99.json")
