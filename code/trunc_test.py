"""
trunc_test.py -- truncation / equilibration test of the corrected Stage-II solve (appendix on conditioning).

    cd data && python ../code/trunc_test.py data_a_0.9.json trunc_0.9.json [2,3,4,5,6]

For each spectral truncation it solves the Stage-II least-squares system twice, without ("naive") and with
row/column equilibration ("equil"), and records the condition number, the mixed horizon observables and the
field-equation residuals.
"""
import sys, json, time, warnings, numpy as np
warnings.filterwarnings("ignore")
from spectral_grid import SpectralGrid
from kerr_background import KerrBackground
from perturbation import assemble_h
from einstein_operator import G1_lin
from stage2_solver import build_operator, COMPS
import fix_source as F
from mixed_observables import omega_mixed
from kappa_horizon import kappa_coeffs
from horizon_area import area_shifts

path, outp = sys.argv[1], sys.argv[2]
truncs = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [2, 3, 4, 5, 6]
g, raw = SpectralGrid.from_file(path); bg = KerrBackground(g); chi = g.chi
H10 = {i: g.fit(g.field_values(raw, f"sGBH{i}"), "even") for i in (1, 2, 3, 4)}
H01 = {i: g.fit(g.field_values(raw, f"dCSH{i}"), "even") for i in (1, 2, 3, 4)}
p10 = g.fit(g.field_values(raw, "sGBPhi"), "even"); t10 = g.fit(g.field_values(raw, "dCSPhi"), "odd")
t0 = time.time()
p11, t11 = F.solve_mixed_scalars_full(bg, p10, t10, H10, H01)
S = F.build_source_full(bg, p10, t10, p11, t11, H10, H01)
print("source", time.time() - t0, flush=True)
A2, cols2 = build_operator(bg, trunc=min(truncs))
print("operator", A2.shape, time.time() - t0, flush=True)
bc, c10, c01 = F.peeling_rhs(bg, H10, H01)
intr = bg.interior; nb = g.Nchi + 1
Sz = np.where(np.isnan(S), 0.0, S)
b = np.concatenate([np.concatenate([Sz[..., a, c][intr] for (a, c) in COMPS])] + [np.full(nb, v) for v in bc])
bulk = np.zeros(bg.shape, bool); bulk[5:-5, 5:-5] = True
eq = lambda v: float(np.interp(0, chi, v))
out = []
for tr in truncs:
    ls = sorted(set(l for (_, _, l) in cols2)); lmax = max(l for l in range(0, g.Nchi + 1, 2)) - 2 * (tr - 1)
    keep = [j for j, (i, n, l) in enumerate(cols2) if l <= lmax and n < g.Nz + 1 - tr]
    A = A2[:, keep]; cols = [cols2[j] for j in keep]
    row = {"trunc": tr, "ncols": len(cols)}
    for mode in ("naive", "equil"):
        if mode == "naive":
            sv = np.linalg.svd(A, compute_uv=False); sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        else:
            rw = 1.0 / np.maximum(np.max(np.abs(A), axis=1), 1e-300); Aw = A * rw[:, None]
            cw = 1.0 / np.maximum(np.linalg.norm(Aw, axis=0), 1e-300); Ae = Aw * cw[None, :]
            sv = np.linalg.svd(Ae, compute_uv=False); sol, *_ = np.linalg.lstsq(Ae, b * rw, rcond=None); sol = sol * cw
        H11 = {i: np.zeros((g.Nz + 1, g.Nchi + 1)) for i in (1, 2, 3, 4)}
        for (i, n, l), v in zip(cols, sol): H11[i][n, l] = v
        R = G1_lin(bg, *assemble_h(bg, H11)) - S
        ev = max(np.max(np.abs(R[..., a, c][bulk])) for (a, c) in COMPS) / max(np.max(np.abs(S[..., a, c][bulk])) for (a, c) in COMPS)
        rchi = np.max(np.abs(R[..., 1, 2][bulk])) / np.max(np.abs(S[..., 1, 2][bulk]))
        om = omega_mixed(bg, H10, H01, H11); k11 = kappa_coeffs(bg, H10, H01, H11)[2]
        A11 = area_shifts(bg, H10, H01, H11)[3]
        sp = lambda v: float(np.nanmax(v[3:-3]) - np.nanmin(v[3:-3]))
        row[mode] = dict(cond=float(sv[0] / sv[-1]), rank_cut=int(np.sum(sv < sv[0] * np.finfo(float).eps * max(A.shape))),
                         Om11=eq(om), Om11_spread=sp(np.asarray(om)), k11=eq(k11), k11_spread=sp(np.asarray(k11)), A11=float(A11),
                         even_res=float(ev), rchi_res=float(rchi))
        print(tr, mode, row[mode], f"[{time.time()-t0:.0f}s]", flush=True)
    out.append(row)
    json.dump(out, open(outp, "w"), indent=1)
