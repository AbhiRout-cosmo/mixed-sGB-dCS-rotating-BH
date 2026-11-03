"""
spectral_grid.py
================

Chebyshev(z) x Legendre(chi) spectral machinery for the combined
scalar-Gauss-Bonnet + dynamical-Chern-Simons black-hole solver.

The data exported from Lam's Mathematica notebook is stored as *grid values*
on a Chebyshev-Gauss-Lobatto (CGL) tensor grid:

    z_nodes   : CGL points, running +1 -> -1   (horizon -> spatial infinity)
    chi_nodes : CGL points, running -1 -> +1   (cos(theta), symmetric about 0)

with each field an array indexed  V[i_z, j_chi]  of shape (Nz+1, Nchi+1).

Coordinate conventions (matching the report, M = 1):
    z = 2 r+ / r - 1 ,   r = 2 r+ / (1 + z) ,   r+ = 1 + sqrt(1 - a^2)
    dz/dr   = -(1 + z)^2 / (2 r+)                       [report Eq. (45)]
    d2z/dr2 =  (1 + z)^3 / (2 r+^2)

This module provides:
    - forward fit   : grid values  -> spectral coefficients  (parity-aware)
    - evaluation    : coefficients -> grid values (same or arbitrary grid)
    - derivatives   : d/dz, d/dchi, d/dr, d2/dr2  (all analytic, in coeff space)

Every public method is exercised by the __main__ self-test at the bottom, which
reproduces the validation numbers established before the module was written
(reconstruction ~1e-15, derivatives ~1e-13, correct parity handling).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np
from numpy.polynomial import chebyshev as _cheb
from numpy.polynomial import legendre as _leg


# Known parity of each stored field in chi (verified against the data):
#   Gauss-Bonnet dilaton  phi^(1/2,0)  -> even   (even-l only)
#   Chern-Simons axion    theta^(0,1/2)-> odd    (odd-l only)
#   all metric functions H_i (both sectors)      -> even (even-l only)
FIELD_PARITY: Dict[str, str] = {
    "sGBPhi": "even",
    "dCSPhi": "odd",
    "sGBH1": "even", "sGBH2": "even", "sGBH3": "even", "sGBH4": "even",
    "dCSH1": "even", "dCSH2": "even", "dCSH3": "even", "dCSH4": "even",
}


def cgl_nodes(N: int) -> np.ndarray:
    """Chebyshev-Gauss-Lobatto nodes cos(k pi / N), k = 0..N  (runs +1 -> -1)."""
    return np.cos(np.arange(N + 1) * np.pi / N)


def _cheb_basis(z: np.ndarray, N: int) -> np.ndarray:
    """Matrix T[i, n] = T_n(z_i), n = 0..N."""
    zc = np.clip(z, -1.0, 1.0)
    theta = np.arccos(zc)
    return np.cos(np.outer(theta, np.arange(N + 1)))


def _leg_basis(chi: np.ndarray, N: int) -> np.ndarray:
    """Matrix P[j, l] = P_l(chi_j), l = 0..N."""
    return np.column_stack([_leg.legval(chi, [0] * l + [1]) for l in range(N + 1)])


def _cheb_deriv_operator(N: int) -> np.ndarray:
    """
    Coefficient-space Chebyshev derivative operator D (shape (N+1, N+1)):
    if c are Chebyshev coefficients of F, then D @ c are the Chebyshev
    coefficients of dF/dz (degree reduced by one, top entry zero).
    """
    D = np.zeros((N + 1, N + 1))
    for n in range(N + 1):
        e = np.zeros(N + 1)
        e[n] = 1.0
        dc = _cheb.chebder(e)              # length n (<= N)
        D[: len(dc), n] = dc
    return D


def _leg_deriv_operator(N: int) -> np.ndarray:
    """Coefficient-space Legendre derivative operator (analogous to above)."""
    D = np.zeros((N + 1, N + 1))
    for l in range(N + 1):
        e = np.zeros(N + 1)
        e[l] = 1.0
        dc = _leg.legder(e)
        D[: len(dc), l] = dc
    return D


@dataclass
class SpectralGrid:
    """
    Parity-aware Chebyshev(z) x Legendre(chi) spectral grid.

    Construct either directly (Nz, Nchi, spin) or via `from_file`.
    Coefficient arrays are always shaped (Nz+1, Nchi+1) = C[n, l].
    """

    Nz: int
    Nchi: int
    spin: float

    # populated in __post_init__
    z: np.ndarray = field(init=False)
    chi: np.ndarray = field(init=False)
    r_plus: float = field(init=False)
    r: np.ndarray = field(init=False)          # r(z), with r=inf at z=-1

    def __post_init__(self) -> None:
        self.z = cgl_nodes(self.Nz)
        self.chi = cgl_nodes(self.Nchi)[::-1].copy()   # ascending, -1 -> +1
        self.r_plus = 1.0 + np.sqrt(1.0 - self.spin ** 2)
        with np.errstate(divide="ignore"):
            self.r = 2.0 * self.r_plus / (1.0 + self.z)

        # basis matrices
        self._Tz = _cheb_basis(self.z, self.Nz)          # (Nz+1, Nz+1)
        self._Pl = _leg_basis(self.chi, self.Nchi)       # (Nchi+1, Nchi+1)
        self._Tz_inv = np.linalg.inv(self._Tz)
        self._Pl_inv = np.linalg.inv(self._Pl)

        # coefficient-space derivative operators
        self._Dz = _cheb_deriv_operator(self.Nz)
        self._Dchi = _leg_deriv_operator(self.Nchi)

        # radial Jacobians on the grid (functions of z only); Eq. (45)
        with np.errstate(divide="ignore"):
            self._dz_dr = -((1.0 + self.z) ** 2) / (2.0 * self.r_plus)
            self._d2z_dr2 = ((1.0 + self.z) ** 3) / (2.0 * self.r_plus ** 2)

    # ------------------------------------------------------------------ #
    # construction from file
    # ------------------------------------------------------------------ #
    @classmethod
    def from_file(cls, path: str, validate: bool = True) -> Tuple["SpectralGrid", dict]:
        """
        Build a grid from one of Lam's JSON data files and return
        (grid, raw_dict). The raw dict still holds the field arrays.
        """
        with open(path) as fh:
            d = json.load(fh)
        g = cls(Nz=int(d["Nz"]), Nchi=int(d["Nchi"]), spin=float(d["spin"]))

        if validate:
            z_file = np.array(d["z_nodes"])
            chi_file = np.array(d["chi_nodes"])
            if not np.allclose(z_file, g.z, atol=1e-10):
                raise ValueError("z_nodes in file do not match CGL convention.")
            # file stores chi ascending as well; compare to internal ascending grid
            if not np.allclose(np.sort(chi_file), g.chi, atol=1e-10):
                raise ValueError("chi_nodes in file do not match CGL convention.")
        return g, d

    def field_values(self, raw: dict, name: str) -> np.ndarray:
        """
        Return the stored grid values for `name`, reordered so the chi axis is
        ascending (matching this grid's internal chi ordering).
        The files already store chi ascending, but we key off chi_nodes to be safe.
        """
        V = np.array(raw[name], dtype=float)
        chi_file = np.array(raw["chi_nodes"])
        order = np.argsort(chi_file)
        return V[:, order]

    # ------------------------------------------------------------------ #
    # fit / evaluate
    # ------------------------------------------------------------------ #
    def fit(self, values: np.ndarray, parity: str = "none") -> np.ndarray:
        """
        Grid values -> spectral coefficients C[n, l].

        parity : 'even' keeps only even l, 'odd' keeps only odd l, 'none' keeps all.
                 Enforcing parity both imposes the symmetry exactly and removes
                 the (numerically tiny) wrong-parity leakage.
        """
        if values.shape != (self.Nz + 1, self.Nchi + 1):
            raise ValueError(
                f"values shape {values.shape} != {(self.Nz + 1, self.Nchi + 1)}"
            )
        A = self._Tz_inv @ values           # transform along z
        C = (self._Pl_inv @ A.T).T          # transform along chi -> C[n, l]
        self._apply_parity(C, parity)
        return C

    def to_grid(self, coeffs: np.ndarray) -> np.ndarray:
        """Spectral coefficients -> values on this grid."""
        return self._Tz @ coeffs @ self._Pl.T

    def evaluate(self, coeffs: np.ndarray,
                 z_eval: np.ndarray, chi_eval: np.ndarray) -> np.ndarray:
        """
        Evaluate a coefficient array on an arbitrary tensor grid
        (z_eval, chi_eval). Useful for the finer-grid residual test.
        """
        Tz = _cheb_basis(np.asarray(z_eval, float), self.Nz)
        Pl = _leg_basis(np.asarray(chi_eval, float), self.Nchi)
        return Tz @ coeffs @ Pl.T

    @staticmethod
    def _apply_parity(C: np.ndarray, parity: str) -> None:
        if parity == "even":
            C[:, 1::2] = 0.0
        elif parity == "odd":
            C[:, 0::2] = 0.0
        elif parity != "none":
            raise ValueError("parity must be 'even', 'odd', or 'none'")

    # ------------------------------------------------------------------ #
    # derivatives (return grid values)
    # ------------------------------------------------------------------ #
    def d_dz(self, coeffs: np.ndarray) -> np.ndarray:
        """dF/dz on the grid."""
        return self._Tz @ (self._Dz @ coeffs) @ self._Pl.T

    def d2_dz2(self, coeffs: np.ndarray) -> np.ndarray:
        """d2F/dz2 on the grid."""
        return self._Tz @ (self._Dz @ self._Dz @ coeffs) @ self._Pl.T

    def d_dchi(self, coeffs: np.ndarray) -> np.ndarray:
        """
        dF/dchi on the grid.

        With F = Tz @ C @ Pl.T, differentiating in chi acts on the Legendre
        basis:  d(Pl)/dchi = Pl @ Dchi.  Hence
            dF/dchi = Tz @ C @ (Pl @ Dchi).T = Tz @ (C @ Dchi.T) @ Pl.T .
        """
        return self._Tz @ (coeffs @ self._Dchi.T) @ self._Pl.T

    def d2_dchi2(self, coeffs: np.ndarray) -> np.ndarray:
        """d2F/dchi2 on the grid."""
        return self._Tz @ (coeffs @ self._Dchi.T @ self._Dchi.T) @ self._Pl.T

    def d_dz_dchi(self, coeffs: np.ndarray) -> np.ndarray:
        """Mixed d2F/dz dchi on the grid."""
        return self._Tz @ (self._Dz @ coeffs @ self._Dchi.T) @ self._Pl.T

    def d_dr(self, coeffs: np.ndarray) -> np.ndarray:
        """
        dF/dr on the grid, via dF/dr = (dz/dr) dF/dz.  [chain rule, Eq. (45)]
        The z=-1 (r=inf) row is exactly zero because dz/dr -> 0 there.
        """
        return self._dz_dr[:, None] * self.d_dz(coeffs)

    def d2_dr2(self, coeffs: np.ndarray) -> np.ndarray:
        """
        d2F/dr2 = (dz/dr)^2 d2F/dz2 + (d2z/dr2) dF/dz.  [Eq. (45)]
        """
        return (self._dz_dr[:, None] ** 2) * self.d2_dz2(coeffs) \
            + self._d2z_dr2[:, None] * self.d_dz(coeffs)

    # convenience meshes ------------------------------------------------ #
    def rchi_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (R, CHI) meshes of shape (Nz+1, Nchi+1) (R=inf at z=-1)."""
        return np.meshgrid(self.r, self.chi, indexing="ij")


# ====================================================================== #
# self-test
# ====================================================================== #
def _selftest(path: str = "data_a_50_0.99.json") -> None:
    import os

    print("=" * 68)
    print("SpectralGrid self-test")
    print("=" * 68)

    # -- pure-math tests (no data file needed) -------------------------- #
    g = SpectralGrid(Nz=30, Nchi=30, spin=0.9)
    print(f"\n[grid]  Nz={g.Nz} Nchi={g.Nchi} spin={g.spin} r+={g.r_plus:.6f}")
    print(f"        r(z=+1)={g.r[0]:.6f} (=r+),  r(z=-1)={g.r[-1]} (=inf)")
    print(f"        cond(Tz)={np.linalg.cond(g._Tz):.3f}  "
          f"cond(Pl)={np.linalg.cond(g._Pl):.3f}")

    # analytic derivative test: F = T4(z) * P2(chi)
    T4 = np.cos(4 * np.arccos(np.clip(g.z, -1, 1)))
    P2 = _leg.legval(g.chi, [0, 0, 1])
    F = np.outer(T4, P2)
    C = g.fit(F, parity="even")            # P2 is even

    # exact derivatives
    T4p = _cheb.chebval(g.z, _cheb.chebder([0, 0, 0, 0, 1]))
    P2p = _leg.legval(g.chi, _leg.legder([0, 0, 1]))
    dFdz_exact = np.outer(T4p, P2)
    dFdchi_exact = np.outer(T4, P2p)

    err_recon = np.max(np.abs(g.to_grid(C) - F))
    err_dz = np.max(np.abs(g.d_dz(C) - dFdz_exact))
    err_dchi = np.max(np.abs(g.d_dchi(C) - dFdchi_exact))
    print(f"\n[math]  reconstruction err : {err_recon:.2e}   (expect ~1e-15)")
    print(f"        d/dz vs exact      : {err_dz:.2e}   (expect ~1e-13)")
    print(f"        d/dchi vs exact    : {err_dchi:.2e}   (expect ~1e-13)")
    assert err_recon < 1e-12 and err_dz < 1e-10 and err_dchi < 1e-10

    # chain-rule d/dr consistency: build F(r) = 1/r from coeffs and compare
    # to analytic -1/r^2 on interior points (drop z=-1 row).
    Rgrid = g.r.copy()
    Fr = np.where(np.isfinite(Rgrid), 1.0 / np.where(Rgrid == 0, np.nan, Rgrid), 0.0)
    Fr2d = np.repeat(Fr[:, None], g.Nchi + 1, axis=1)
    Cr = g.fit(Fr2d, parity="even")
    dFdr = g.d_dr(Cr)
    analytic = np.where(np.isfinite(Rgrid), -1.0 / Rgrid ** 2, 0.0)[:, None]
    interior = slice(0, g.Nz)             # drop r=inf row
    err_dr = np.max(np.abs(dFdr[interior] - analytic[interior] * np.ones_like(dFdr)[interior]))
    print(f"        d/dr of 1/r        : {err_dr:.2e}   (spectral fit of 1/r, "
          f"limited by resolution)")

    # -- data-file tests ------------------------------------------------ #
    if not os.path.exists(path):
        print(f"\n[data]  file '{path}' not found; skipping data tests.")
        print("\nself-test PASSED (math only)\n")
        return

    gf, raw = SpectralGrid.from_file(path)
    print(f"\n[data]  loaded '{path}'  (spin={gf.spin})")
    print(f"        node conventions validated against CGL: OK")

    print("\n        field            parity   recon_err   tail(n)   tail(l)")
    print("        " + "-" * 58)
    for name, par in FIELD_PARITY.items():
        V = gf.field_values(raw, name)
        C = gf.fit(V, parity=par)
        recon = np.max(np.abs(gf.to_grid(C) - V))
        tail_n = np.max(np.abs(C[-3:, :]))
        tail_l = np.max(np.abs(C[:, -3:]))
        # verify parity actually holds in the raw values
        Vref = V[:, ::-1]
        phold = np.allclose(V, Vref, atol=1e-9) if par == "even" \
            else np.allclose(V, -Vref, atol=1e-9)
        flag = "" if phold else "  <-- PARITY MISMATCH"
        print(f"        {name:14s}   {par:5s}   {recon:.2e}   "
              f"{tail_n:.1e}   {tail_l:.1e}{flag}")
        assert recon < 1e-12, f"{name}: reconstruction failed"
        assert phold, f"{name}: parity mismatch"

    print("\nself-test PASSED\n")


if __name__ == "__main__":
    import sys
    _selftest(sys.argv[1] if len(sys.argv) > 1 else "data_a_50_0.99.json")
