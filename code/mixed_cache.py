"""
mixed_cache.py
==============

Run ONCE per spin.  Solves Stage I (mixed scalars) and Stage II (mixed metric
h^(1,1)) and writes everything the observable scripts need into a single JSON,
so observables can be computed later WITHOUT re-solving.

Usage
-----
    python mixed_cache.py data_a_0_99.json                 # -> mixed_a_0_99.json
    python mixed_cache.py data_a_0_9.json  mixed_a09.json  # explicit output name

The cache stores spectral COEFFICIENTS (resolution-tagged by Nz, Nchi):
    spin, Nz, Nchi
    H10 (sGB), H01 (dCS), H11 (mixed)   -- each a dict {1,2,3,4}
    phi11, theta11                       -- mixed scalars

Load it elsewhere with `load_mixed(path)` -> (bg, H10, H01, H11, phi11, theta11).
"""

from __future__ import annotations
import json, sys
import numpy as np

from spectral_grid import SpectralGrid
from kerr_background import KerrBackground
from perturbation import assemble_h
from einstein_operator import G1_lin
from stage1_solver import solve_mixed_scalars
from stage2_solver import build_operator, solve_mixed_metric, COMPS
from stage2_source import build_source


# ---------------------------------------------------------------------- #
def solve_and_save(data_path: str, cache_path: str | None = None, trunc: int = 3):
    """Solve Stage I + II for one spin and write the cache JSON."""
    if cache_path is None:
        cache_path = data_path.replace("data_a_", "mixed_a_")
        if cache_path == data_path:
            cache_path = "mixed_" + data_path

    g, raw = SpectralGrid.from_file(data_path)
    bg = KerrBackground(g)
    H10 = {i: g.fit(g.field_values(raw, f"sGBH{i}"), "even") for i in (1, 2, 3, 4)}
    H01 = {i: g.fit(g.field_values(raw, f"dCSH{i}"), "even") for i in (1, 2, 3, 4)}
    p10 = g.fit(g.field_values(raw, "sGBPhi"), "even")
    t10 = g.fit(g.field_values(raw, "dCSPhi"), "odd")

    print(f"[{data_path}]  spin={g.spin}  grid {g.Nz}x{g.Nchi}  solving ...", flush=True)
    p11, t11 = solve_mixed_scalars(bg, p10, t10, H10, H01)
    S = build_source(bg, p10, t10, p11, t11, H10, H01)
    A, cols = build_operator(bg, trunc=trunc)
    H11 = solve_mixed_metric(bg, S, A_cols=(A, cols))

    # quality report: even-sector residual (physical) at the current resolution
    bulk = np.zeros(bg.shape, bool); bulk[5:-5, 5:-5] = True
    R = G1_lin(bg, *assemble_h(bg, H11)) - S
    ev = max(np.max(np.abs(R[..., a, b][bulk])) for (a, b) in COMPS)
    evsc = max(np.max(np.abs(S[..., a, b][bulk])) for (a, b) in COMPS)
    print(f"    even-sector residual rel = {ev / evsc:.2e}", flush=True)

    def dump(d):
        return {str(k): np.asarray(v).tolist() for k, v in d.items()}

    out = {
        "spin": float(g.spin), "Nz": int(g.Nz), "Nchi": int(g.Nchi),
        "trunc": int(trunc), "even_residual_rel": float(ev / evsc),
        "H10": dump(H10), "H01": dump(H01), "H11": dump(H11),
        "phi11": np.asarray(p11).tolist(), "theta11": np.asarray(t11).tolist(),
    }
    with open(cache_path, "w") as fh:
        json.dump(out, fh)
    print(f"    saved -> {cache_path}")
    return cache_path


# ---------------------------------------------------------------------- #
def load_mixed(cache_path: str):
    """Load a cache -> (bg, H10, H01, H11, phi11, theta11).  bg is rebuilt."""
    with open(cache_path) as fh:
        d = json.load(fh)
    g = SpectralGrid(Nz=d["Nz"], Nchi=d["Nchi"], spin=d["spin"])
    bg = KerrBackground(g)
    coef = lambda blk: {int(k): np.array(v) for k, v in d[blk].items()}
    H10, H01, H11 = coef("H10"), coef("H01"), coef("H11")
    p11 = np.array(d["phi11"]); t11 = np.array(d["theta11"])
    return bg, H10, H01, H11, p11, t11


if __name__ == "__main__":
    solve_and_save("data_a_50_0.8.json")
