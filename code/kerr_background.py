"""
kerr_background.py
==================

Kerr background geometry on the spectral (z, chi) grid, in coordinates
(t, r, chi, phi) with M = 1, chi = cos(theta).  This is the geometry layer the
mixed-order solver builds on: metric, inverse, Christoffels, Riemann, Ricci,
dual Riemann, the curvature-invariant sources (Gauss-Bonnet / Pontryagin), and
the regular scalar d'Alembertian.

Design
------
The Kerr metric has *coordinate* singularities at the horizon (z = +1, i.e.
Delta = 0) and at the poles (chi = +-1).  Spectrally differentiating the metric
components there would be hopeless (g_rr ~ 1/Delta, g_chichi ~ 1/(1-chi^2)).
So the metric and its first/second coordinate derivatives are obtained
*analytically* with sympy, and every curvature tensor is then assembled by exact
tensor algebra (numpy einsum).  This yields machine-precision background tensors;
the R_{mu nu} = 0 self-test validates the whole chain.

Coordinate-basis tensor components still diverge at the horizon/pole grid nodes
(that is physical for Boyer-Lindquist coordinates).  Two consequences:
  * The interior block [1:Nz, 1:Nchi] is fully regular; curvature *scalars*
    (GB, Pontryagin) are finite everywhere but are only evaluated component-wise
    on the interior to avoid 0/0 at the coordinate singularities.  For the
    scalar *sources* the solver should use the closed forms `gb_scalar` /
    `pontryagin` (regular everywhere, matching report Eqs. 131/136).
  * The scalar d'Alembertian is provided in its regular form
        box0 phi = (1/rho^2)[ d_r(Delta d_r phi) + d_chi((1-chi^2) d_chi phi) ],
    which is finite at horizon and poles, and reuses the tested SpectralGrid
    derivative machinery.

Conventions
-----------
signature (-,+,+,+);  Riemann  R^rho_{sig mu nu} = d_mu Gamma^rho_{nu sig} - ...;
Ricci  R_{mu nu} = R^a_{mu a nu};  Levi-Civita tensor oriented so that
    eps^{mu nu rho sig} = +(1/sqrt(-g)) [mu nu rho sig],   [t r chi phi] = +1,
which reproduces the report's Pontryagin density Eq. (136) as written.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import permutations
from typing import Dict, Tuple

import numpy as np
import sympy as sp

from spectral_grid import SpectralGrid


# ---------------------------------------------------------------------- #
# symbolic Kerr metric and its coordinate derivatives (built once per a-form)
# ---------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def _symbolic_kerr() -> Tuple[tuple, tuple, tuple]:
    """
    Return component-wise lambdified callables (functions of r, chi, a) for
    g_{mu nu}, its first derivatives d_k g (k = r, chi) and second derivatives
    d2_{kl} g.  Cached so the sympy work happens only once per process.

    Returns
    -------
    gL      : 4x4 nested tuple of callables f(r, chi, a)
    dgL     : dict {1: 4x4, 2: 4x4}          (1 = d/dr, 2 = d/dchi)
    d2gL    : dict {(1,1),(2,2),(1,2): 4x4}
    """
    r, chi, a = sp.symbols("r chi a", real=True)
    rho2 = r**2 + a**2 * chi**2
    Delta = r**2 - 2 * r + a**2
    omc = 1 - chi**2

    g = sp.zeros(4, 4)
    g[0, 0] = -(1 - 2 * r / rho2)
    g[0, 3] = g[3, 0] = -2 * a * r * omc / rho2
    g[1, 1] = rho2 / Delta
    g[2, 2] = rho2 / omc
    g[3, 3] = (r**2 + a**2 + 2 * a**2 * r * omc / rho2) * omc

    dg = {1: sp.diff(g, r), 2: sp.diff(g, chi)}
    d2g = {(1, 1): sp.diff(g, r, r),
           (2, 2): sp.diff(g, chi, chi),
           (1, 2): sp.diff(g, r, chi)}

    def clam(M):
        return tuple(tuple(sp.lambdify((r, chi, a), M[i, j], "numpy")
                           for j in range(4)) for i in range(4))

    gL = clam(g)
    dgL = {k: clam(v) for k, v in dg.items()}
    d2gL = {k: clam(v) for k, v in d2g.items()}
    return gL, dgL, d2gL


def _levi_civita_symbol() -> np.ndarray:
    """Totally antisymmetric symbol [mu nu rho sig], [0123] = +1."""
    eps = np.zeros((4, 4, 4, 4))
    for p in permutations(range(4)):
        s = 1
        pl = list(p)
        for i in range(4):
            for j in range(i + 1, 4):
                if pl[i] > pl[j]:
                    s = -s
        eps[p] = s
    return eps


_LC = _levi_civita_symbol()


class KerrBackground:
    """
    All background geometry needed by the solver, evaluated on a SpectralGrid.

    Attributes (numpy arrays over the (Nz+1, Nchi+1) grid unless noted)
    --------------------------------------------------------------------
    R, CHI            : coordinate meshes (R = inf at z = -1)
    rho2, Delta       : rho^2 = r^2 + a^2 chi^2 ,  Delta = r^2 - 2r + a^2
    sqrt_mg           : sqrt(-g) = rho^2
    g_dd, g_uu        : metric and inverse, shape (...,4,4)
    Gamma             : Christoffel Gamma^l_{mn}, shape (...,4,4,4) = [...,l,m,n]
    Riemann_dddd      : R_{a b c d}, shape (...,4,4,4,4)
    Ricci_dd          : R_{mu nu} (== 0 up to round-off)
    dualRiemann_uudd  : *R^{a b}{}_{c d}  (upper-upper-lower-lower)
    gb_scalar         : Gauss-Bonnet invariant (closed form, Eq. 131) [regular]
    pontryagin        : Pontryagin density  (closed form, Eq. 136) [regular]

    Coordinate-basis tensors diverge at the horizon node (z=+1) and pole nodes
    (chi=+-1); those boundary entries are set to NaN so misuse is loud rather
    than silent.  The `interior` mask selects the regular block.
    """

    COORD = ("t", "r", "chi", "phi")

    def __init__(self, grid: SpectralGrid):
        self.grid = grid
        self.a = grid.spin
        self.R, self.CHI = grid.rchi_mesh()          # (Nz+1, Nchi+1); R[inf] at z=-1
        self.shape = self.R.shape

        # interior mask: drop horizon row (z=+1 -> index 0) and pole cols (chi=+-1)
        self.interior = np.ones(self.shape, bool)
        self.interior[0, :] = False                   # horizon (Delta = 0)
        self.interior[:, 0] = False                   # chi = -1
        self.interior[:, -1] = False                  # chi = +1
        # also the r = inf row (z = -1 -> last index) is not usable for curvature
        self._rinf_row = self.shape[0] - 1
        self.interior[self._rinf_row, :] = False

        with np.errstate(divide="ignore", invalid="ignore"):
            self.rho2 = self.R**2 + self.a**2 * self.CHI**2
            self.Delta = self.R**2 - 2 * self.R + self.a**2
        self.sqrt_mg = self.rho2.copy()               # sqrt(-g) = rho^2

        self._build_tensors()
        self._build_scalars()

    # ------------------------------------------------------------------ #
    def _eval_component_form(self, L) -> np.ndarray:
        """Evaluate a 4x4 nested-callable form on the (R, CHI) mesh -> (...,4,4).

        Boundary nodes (horizon, poles, r=inf) produce inf/nan in individual
        components; that is expected for Boyer-Lindquist coordinates and those
        entries are masked out downstream. Warnings are suppressed here.
        """
        out = np.zeros(self.shape + (4, 4))
        with np.errstate(divide="ignore", invalid="ignore"):
            for i in range(4):
                for j in range(4):
                    val = L[i][j](self.R, self.CHI, self.a)
                    out[..., i, j] = np.broadcast_to(np.asarray(val, float), self.shape)
        return out

    def _build_tensors(self) -> None:
        gL, dgL, d2gL = _symbolic_kerr()

        # metric & coordinate derivatives (analytic)
        g = self._eval_component_form(gL)                      # g_{mu nu}
        # first derivatives, indexed by coordinate direction c in 0..3
        dgc = np.zeros(self.shape + (4, 4, 4))                 # [...,c,mu,nu]
        dgc[..., 1, :, :] = self._eval_component_form(dgL[1])  # d/dr
        dgc[..., 2, :, :] = self._eval_component_form(dgL[2])  # d/dchi
        # second derivatives [...,c,d,mu,nu]
        d2gc = np.zeros(self.shape + (4, 4, 4, 4))
        d2gc[..., 1, 1, :, :] = self._eval_component_form(d2gL[(1, 1)])
        d2gc[..., 2, 2, :, :] = self._eval_component_form(d2gL[(2, 2)])
        d2gc[..., 1, 2, :, :] = self._eval_component_form(d2gL[(1, 2)])
        d2gc[..., 2, 1, :, :] = d2gc[..., 1, 2, :, :]

        # inverse metric (per grid point); horizon/pole nodes are singular -> mask
        ginv = np.full_like(g, np.nan)
        good = self.interior
        ginv[good] = np.linalg.inv(g[good])

        # derivative of inverse: d g^{-1} = - g^{-1} (dg) g^{-1}
        dginv = -np.einsum("...ma,...cab,...bn->...cmn", ginv, dgc, ginv)

        # Christoffel  Gamma^l_{mn} = 1/2 g^{ls}(d_m g_{ns} + d_n g_{ms} - d_s g_{mn})
        Gam = 0.5 * (
            np.einsum("...ls,...mns->...lmn", ginv, dgc)
            + np.einsum("...ls,...nms->...lmn", ginv, dgc)
            - np.einsum("...ls,...smn->...lmn", ginv, dgc)
        )

        # d_c Gamma^l_{mn}
        dGam = 0.5 * (
            np.einsum("...cls,...mns->...clmn", dginv, dgc)
            + np.einsum("...cls,...nms->...clmn", dginv, dgc)
            - np.einsum("...cls,...smn->...clmn", dginv, dgc)
            + np.einsum("...ls,...cmns->...clmn", ginv, d2gc)
            + np.einsum("...ls,...cnms->...clmn", ginv, d2gc)
            - np.einsum("...ls,...csmn->...clmn", ginv, d2gc)
        )

        # Riemann  R^rho_{sig mu nu} = d_mu Gamma^rho_{nu sig} - d_nu Gamma^rho_{mu sig}
        #                              + Gamma^rho_{mu l} Gamma^l_{nu sig}
        #                              - Gamma^rho_{nu l} Gamma^l_{mu sig}
        Riem_u = (
            np.einsum("...mrns->...rsmn", dGam)      # d_mu Gamma^rho_{nu sig}
            - np.einsum("...nrms->...rsmn", dGam)     # d_nu Gamma^rho_{mu sig}
            + np.einsum("...rml,...lns->...rsmn", Gam, Gam)
            - np.einsum("...rnl,...lms->...rsmn", Gam, Gam)
        )

        Ricci = np.einsum("...msmn->...sn", Riem_u)   # R_{sig nu} = R^m_{sig m nu}
        Riem_dddd = np.einsum("...ar,...rbcd->...abcd", g, Riem_u)  # lower all

        # store
        self.g_dd = g
        self.g_uu = ginv
        self.dg_ddd = dgc                              # [...,c,mu,nu]
        self.Gamma = Gam                               # [...,l,m,n]
        self.Riemann_dddd = Riem_dddd
        self.Ricci_dd = Ricci

        # dual Riemann  *R^{ab}{}_{cd} = 1/2 eps^{ab}{}_{ef} R^{ef}{}_{cd}
        # eps^{mu nu rho sig} = +(1/sqrt(-g)) [symbol]   (report orientation)
        eps_uuuu = (1.0 / self.sqrt_mg)[..., None, None, None, None] * _LC
        eps_uudd = np.einsum("...abef,...ce,...df->...abcd", eps_uuuu, g, g)  # eps^{ab}{}_{cd}
        Riem_uudd = np.einsum("...ai,...bj,...ijcd->...abcd", ginv, ginv, Riem_dddd)  # R^{ab}{}_{cd}
        self.dualRiemann_uudd = 0.5 * np.einsum("...abef,...efcd->...abcd",
                                                eps_uudd, Riem_uudd)

        # component-derived scalars (interior only) for validation / cross-check
        Riem_uuuu = np.einsum("...ai,...bj,...ck,...dl,...ijkl->...abcd",
                              ginv, ginv, ginv, ginv, Riem_dddd)
        self._kretschmann = np.einsum("...abcd,...abcd->...", Riem_dddd, Riem_uuuu)
        dualstar_uuuu = 0.5 * np.einsum("...abef,...efcd->...abcd",
                                        eps_uudd, Riem_uuuu)  # *R^{ab}{}^{... }
        self._pontryagin_comp = np.einsum("...abcd,...abcd->...",
                                          dualstar_uuuu, Riem_dddd)

    def _build_scalars(self) -> None:
        """Closed-form curvature invariants (regular everywhere), report Eqs. 131/136."""
        R, C, a = self.R, self.CHI, self.a
        Sig = R**2 + a**2 * C**2
        with np.errstate(divide="ignore", invalid="ignore"):
            self.gb_scalar = 48 * (R**6 - 15 * a**2 * R**4 * C**2
                                   + 15 * a**4 * R**2 * C**4 - a**6 * C**6) / Sig**6
            self.pontryagin = -96 * a * R * C * (R**2 - 3 * a**2 * C**2) \
                * (3 * R**2 - a**2 * C**2) / Sig**6
        # r = inf row -> 0 (corrections vanish there)
        self.gb_scalar[self._rinf_row, :] = 0.0
        self.pontryagin[self._rinf_row, :] = 0.0

    # ------------------------------------------------------------------ #
    # regular scalar d'Alembertian (for the scalar sector)
    # ------------------------------------------------------------------ #
    def box0(self, field_coeffs: np.ndarray) -> np.ndarray:
        """
        Scalar d'Alembertian on a stationary axisymmetric field phi(z, chi):

            box0 phi = (1/rho^2) [ d_r(Delta d_r phi) + d_chi((1-chi^2) d_chi phi) ]
                     = (1/rho^2) [ Delta phi_rr + Delta' phi_r
                                   + (1-chi^2) phi_chichi - 2 chi phi_chi ]

        Regular at the horizon (Delta -> 0) and poles ((1-chi^2) -> 0).
        `field_coeffs` are spectral coefficients (from SpectralGrid.fit).
        Returns grid values.
        """
        g = self.grid
        phi_r = g.d_dr(field_coeffs)
        phi_rr = g.d2_dr2(field_coeffs)
        phi_c = g.d_dchi(field_coeffs)
        phi_cc = g.d2_dchi2(field_coeffs)

        Delta_p = 2 * self.R - 2                      # d Delta / dr
        omc = 1 - self.CHI**2
        with np.errstate(divide="ignore", invalid="ignore"):
            radial = self.Delta * phi_rr + Delta_p * phi_r
            angular = omc * phi_cc - 2 * self.CHI * phi_c
            out = (radial + angular) / self.rho2
        # r = inf row: rho^2 -> inf, box -> 0 for decaying fields; guard NaN
        out[self._rinf_row, :] = 0.0
        return out


# ====================================================================== #
# self-test
# ====================================================================== #
def _selftest(path: str = "data_a_50_0.99.json") -> None:
    import os

    print("=" * 68)
    print("KerrBackground self-test")
    print("=" * 68)

    grid = SpectralGrid(Nz=30, Nchi=30, spin=0.9)
    bg = KerrBackground(grid)
    intr = bg.interior

    # 1) Ricci-flatness
    ricci_max = np.max(np.abs(bg.Ricci_dd[intr]))
    print(f"\n[1] max|R_mu_nu| (interior)            : {ricci_max:.2e}   (expect ~1e-11)")
    assert ricci_max < 1e-8

    # 2) Kretschmann (component) vs closed-form GB scalar (Eq. 131)
    rel_gb = np.max(np.abs(bg._kretschmann[intr] - bg.gb_scalar[intr])
                    / np.abs(bg.gb_scalar[intr]))
    print(f"[2] max rel err  Kretschmann vs Eq.131 : {rel_gb:.2e}   (expect ~1e-12)")
    assert rel_gb < 1e-9

    # 3) Pontryagin (component) vs closed-form (Eq. 136)
    #    equator chi=0 has Pontryagin=0, so compare absolute on interior
    err_p = np.max(np.abs(bg._pontryagin_comp[intr] - bg.pontryagin[intr]))
    scale = np.max(np.abs(bg.pontryagin[intr]))
    print(f"[3] max|Pontryagin_comp - Eq.136|      : {err_p:.2e}   "
          f"(scale {scale:.1e}, expect err ~1e-12)")
    assert err_p < 1e-6 * scale

    # 4) determinant identity sqrt(-g) = rho^2  (already used; sanity via det)
    det = np.linalg.det(bg.g_dd[intr])
    err_det = np.max(np.abs(np.sqrt(-det) - bg.rho2[intr]))
    print(f"[4] max|sqrt(-det g) - rho^2|          : {err_det:.2e}   (expect ~1e-13)")
    assert err_det < 1e-9

    # 5) regular scalar box: box0(1) = 0, and box0 of a known field is finite everywhere
    one = grid.fit(np.ones(bg.shape), parity="even")
    box_one = bg.box0(one)
    print(f"[5] max|box0(1)|                       : {np.max(np.abs(box_one)):.2e}   "
          f"(expect ~1e-10)")
    assert np.max(np.abs(box_one)) < 1e-6

    # 6) data cross-check: box0 phi^(1/2,0) + GB^(0) should ~ 0  (report Eq. 111/130b)
    #    This is the first genuine physics residual test using real data.
    if os.path.exists(path):
        g2, raw = SpectralGrid.from_file(path)
        bg2 = KerrBackground(g2)
        phi = g2.fit(g2.field_values(raw, "sGBPhi"), parity="even")
        res = bg2.box0(phi) + bg2.gb_scalar          # box phi = -GB  ->  residual
        # evaluate residual on interior, relative to the source magnitude
        rmax = np.max(np.abs(res[bg2.interior]))
        smax = np.max(np.abs(bg2.gb_scalar[bg2.interior]))
        print(f"\n[6] PURE-sGB SCALAR RESIDUAL  |box phi + GB|_inf : {rmax:.2e}")
        print(f"    source scale |GB|_inf                        : {smax:.2e}")
        print(f"    relative residual                            : {rmax/smax:.2e}")
        print("    (expect ~1e-8..1e-6: the noise floor of Lam's exported H/phi data)")
    else:
        print(f"\n[6] data file '{path}' not found; skipping physics residual test.")

    print("\nself-test PASSED\n")


if __name__ == "__main__":
    import sys
    _selftest(sys.argv[1] if len(sys.argv) > 1 else "data_a_50_0.99.json")
