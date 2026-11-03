"""
mixed_cache_fixed.py -- corrected replacement for mixed_cache.solve_and_save.

    python mixed_cache_fixed.py data_a_0.5.json [data_a_0.52.json ...]

Writes mixed_a_X.json in the SAME format as mixed_cache.py, so every observable script
(generate_observables, kappa_mixed, horizon_area, isco_photon, qnm_eikonal, wald_*) runs unchanged.
An existing mixed_a_X.json is first renamed to mixed_a_X.old.json (never overwritten twice).
Extra keys stored: residuals (even sector and r-chi), ADM shifts, peeling right-hand sides.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np

from spectral_grid import SpectralGrid
from kerr_background import KerrBackground
from perturbation import assemble_h
from einstein_operator import G1_lin
from stage2_solver import build_operator, COMPS
import fix_source as F


def solve_and_save(data_path, cache_path=None, trunc=3):
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

    t0 = time.time()
    print(f"[{data_path}]  spin={g.spin}  grid {g.Nz}x{g.Nchi}  solving ...", flush=True)
    p11, t11 = F.solve_mixed_scalars_full(bg, p10, t10, H10, H01)
    S = F.build_source_full(bg, p10, t10, p11, t11, H10, H01)
    A, cols = build_operator(bg, trunc=trunc)
    bc, c10, c01 = F.peeling_rhs(bg, H10, H01)
    H11 = F.solve_metric_bc(bg, S, A, cols, bc)

    bulk = np.zeros(bg.shape, bool); bulk[5:-5, 5:-5] = True
    R = G1_lin(bg, *assemble_h(bg, H11)) - S
    ev = max(np.max(np.abs(R[..., a, b][bulk])) for (a, b) in COMPS)
    evsc = max(np.max(np.abs(S[..., a, b][bulk])) for (a, b) in COMPS)
    rchi = np.max(np.abs(R[..., 1, 2][bulk])) / np.max(np.abs(S[..., 1, 2][bulk]))
    dM_rr, dM_tt, dJ = F.adm_shifts(bg, H10, H01, H11)
    print(f"    even-sector residual rel = {ev/evsc:.2e}   r-chi residual rel = {rchi:.2e}   "
          f"dM = {dM_rr:+.1e} (g_rr) {dM_tt:+.1e} (g_tt)   dJ/a = {dJ:+.1e}   [{time.time()-t0:.0f}s]", flush=True)

    dump = lambda d: {str(k): np.asarray(v).tolist() for k, v in d.items()}
    out = {"spin": float(g.spin), "Nz": int(g.Nz), "Nchi": int(g.Nchi), "trunc": int(trunc),
           "even_residual_rel": float(ev / evsc), "rchi_residual_rel": float(rchi),
           "peeling_rhs": [float(x) for x in bc], "c10": c10, "c01": c01,
           "dM_grr": float(dM_rr), "dM_gtt": float(dM_tt), "dJ_over_a": float(dJ),
           "source": "fix_source.build_source_full (2026-10-05)",
           "H10": dump(H10), "H01": dump(H01), "H11": dump(H11),
           "phi11": np.asarray(p11).tolist(), "theta11": np.asarray(t11).tolist()}
    if os.path.exists(cache_path):
        old = cache_path.replace(".json", ".old.json")
        if not os.path.exists(old):
            os.rename(cache_path, old); print(f"    existing cache kept as {old}")
    with open(cache_path, "w") as fh:
        json.dump(out, fh)
    print(f"    saved -> {cache_path}")
    return cache_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(0)
    for p in sys.argv[1:]:
        solve_and_save(p)
