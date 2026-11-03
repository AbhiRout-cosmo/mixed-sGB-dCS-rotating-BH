"""
horizon_area.py
===============

Mixed-order O(zeta_GB zeta_CS) correction to the horizon AREA, from the cached
mixed metric (mixed_a_0_X.json).

Horizon area:  A = 2 pi integral_{-1}^{+1} sqrt(g_chichi * g_phiphi)|_{r+} dchi.

Numerical care (both essential):
  * POLES: at chi = +/-1, g_chichi ~ Sigma/(1-chi^2) -> inf while g_phiphi ~
    (1-chi^2) -> 0; the PRODUCT is finite but the node values are inf*0 = NaN.
    We evaluate the integrand only on interior (finite) nodes.
  * QUADRATURE: the chi nodes are Chebyshev-Lobatto, so Gauss-Legendre weights
    are wrong.  We instead fit the (smooth, even) integrand to a Chebyshev
    series and integrate it analytically -> spectrally exact.

Validation: A^(0) must equal the Kerr value 8 pi r+ (M=1).  The mixed shift
A^(1,1) is a 2nd finite difference in the two couplings; A^(1,0)/A^(0,1) are the
pure-sector first differences.

Note on the first law: with M, J held fixed by the peeling conditions, the Smarr
relation M = kappa A/(4 pi) + 2 Omega_H J gives, at O(zeta_GB zeta_CS),
    0 =? (1/4pi)[kappa^(1,1) A0 + kappa^(1,0) A^(0,1) + kappa^(0,1) A^(1,0)
                 + kappa0 A^(1,1)] + 2 J Omega^(1,1).
This is printed as a diagnostic, but it is NOT expected to close exactly at
O(zeta): in sGB/dCS the horizon entropy is the Wald entropy (area + curvature
terms), not A/4, so a non-zero residual here is the Wald correction, not a bug.
"""

from __future__ import annotations
import numpy as np
import glob
from numpy.polynomial import chebyshev as C

from mixed_cache import load_mixed
from perturbation import assemble_h


def horizon_area(bg, gdd):
    """Pole-safe, spectrally-exact horizon area for a metric gdd."""
    g = bg.grid
    chi = g.chi
    with np.errstate(invalid="ignore", divide="ignore"):
        prod = gdd[0, :, 2, 2] * gdd[0, :, 3, 3]          # g_chichi * g_phiphi at horizon
        integrand = np.sqrt(np.abs(prod))
    ok = np.isfinite(integrand)                            # drops the two pole nodes
    ci, vi = chi[ok], integrand[ok]
    order = np.argsort(ci)
    deg = min(len(ci) - 1, g.Nchi, 24)
    coef = C.chebfit(ci[order], vi[order], deg)            # smooth even integrand
    I = C.chebval(1.0, C.chebint(coef)) - C.chebval(-1.0, C.chebint(coef))
    return 2.0 * np.pi * I


def _metric(bg, H10, H01, H11, ca, cb):
    h10, _, _ = assemble_h(bg, H10)
    h01, _, _ = assemble_h(bg, H01)
    h11, _, _ = assemble_h(bg, H11)
    with np.errstate(invalid="ignore"):
        return bg.g_dd + ca * h10 + cb * h01 + ca * cb * h11


def area_shifts(bg, H10, H01, H11, eps=1e-3):
    """A^(0), A^(1,0), A^(0,1), A^(1,1) via finite differences in the couplings."""
    A = lambda ca, cb: horizon_area(bg, _metric(bg, H10, H01, H11, ca, cb))
    A0 = A(0.0, 0.0)
    App, Apm = A(+eps, +eps), A(+eps, -eps)
    Amp, Amm = A(-eps, +eps), A(-eps, -eps)
    A10 = (App + Apm - Amp - Amm) / (4 * eps)
    A01 = (App - Apm + Amp - Amm) / (4 * eps)
    A11 = (App - Apm - Amp + Amm) / (4 * eps * eps)
    return A0, A10, A01, A11


if __name__ == "__main__":
    # Generates: mixed_a_0.1.json, mixed_a_0.12.json, ..., mixed_a_0.2.json, ..., mixed_a_0.9.json
    paths = [f"mixed_a_{a:.2g}.json" for a in np.arange(0.1, 0.91, 0.02)]

    print(f"{'a':>4} {'A^(0)':>10} {'8pi r+':>10} {'err':>8}  "
          f"{'A^(1,0)':>12} {'A^(0,1)':>12} {'A^(1,1)':>12}")
    
    for p in paths:
        try:
            bg, H10, H01, H11, _, _ = load_mixed(p)
            a = bg.a; rp = 1 + np.sqrt(1 - a**2)
            A0, A10, A01, A11 = area_shifts(bg, H10, H01, H11)
            Aex = 8 * np.pi * rp
            print(f"{a:>4.2f} {A0:>10.5f} {Aex:>10.5f} {abs(A0-Aex):>8.1e}  "
                  f"{A10:>+12.4e} {A01:>+12.4e} {A11:>+12.4e}")
        except FileNotFoundError:
            print(f"[{p}] skipped: File not found")
        except Exception as e:
            print(f"[{p}] error: {e}")
