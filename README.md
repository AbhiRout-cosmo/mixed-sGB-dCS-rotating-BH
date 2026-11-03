# Mixed-order sGB–dCS corrections to rotating black holes

Code and data for

> A. Rout and B. Altschul, *Mixed-Order Parity-Violating Corrections to Rotating Black Holes in Combined Scalar Gauss–Bonnet and Dynamical Chern–Simons Gravity* (2026), arXiv number to follow.

The pure-sector (leading-order sGB and dCS) solutions on which the mixed-order calculation is built are those of

> K. K.-H. Lam, A. K.-W. Chung, and N. Yunes, *Analytic and accurate approximate metrics for black holes with arbitrary rotation in beyond-Einstein gravity using spectral methods*, Phys. Rev. D **113**, 024030 (2026), [doi:10.1103/txxb-z8bx](https://doi.org/10.1103/txxb-z8bx), [arXiv:2510.05208](https://arxiv.org/abs/2510.05208).

The repository contains everything needed to reproduce the numerical results of the paper: the pure-sector inputs, the solver for the mixed-order $\mathcal{O}(\zeta_{\rm GB}\zeta_{\rm CS})$ scalars and metric correction, the solved mixed-order solutions, and the scripts that extract every observable, table entry and consistency check. Plotting code is not included; every plotted quantity is produced by the scripts below.

## Requirements

- Python ≥ 3.10 with NumPy, SciPy and SymPy (`pip install -r requirements.txt`). Tested with Python 3.13, NumPy 2.5, SciPy 1.18, SymPy 1.14.
- Mathematica, only to regenerate the pure-sector input files (they are included).

## Layout

```
code/          Python modules and scripts (flat; run them from data/, see below)
data/          inputs data_a_X.json and solutions mixed_a_X.json
results/       outputs of the scripts, as used in the paper
validation/    published pure-sector benchmarks of Lam, Chung & Yunes
mathematica/   notebook that generates the pure-sector inputs from the Lam et al. expressions
```

All scripts read and write files in the **current directory**. Run them from `data/`:

```bash
cd data
python ../code/sweep_table.py
```

Python puts `code/` on the import path automatically because the script lives there.

## Data

| File | Contents |
|---|---|
| `data/data_a_X.json` | Pure-sector inputs at spin `X`: the metric functions $H_i^{(1,0)}$ (`sGBH1..4`), $H_i^{(0,1)}$ (`dCSH1..4`) and the scalars $\varphi^{(1/2,0)}$ (`sGBPhi`), $\vartheta^{(0,1/2)}$ (`dCSPhi`) at the collocation nodes. 30×30 for X = 0.10, 0.12, …, 0.90 and 0.99; 50×50 for `data_a_50_0.8`, `_0.9`, `_0.99`. |
| `data/mixed_a_X.json` | Mixed-order solutions at 30×30, X = 0.10, 0.12, …, 0.90: spectral coefficients of $H_i^{(1,0)}, H_i^{(0,1)}, H_i^{(1,1)}$, the mixed scalars $\varphi^{(1/2,1)}$ and $\vartheta^{(1,1/2)}$, field-equation residuals, peeling right-hand sides and ADM shifts. Load with `mixed_cache.load_mixed`. |

Conventions: $M = 1$, Boyer–Lindquist-type coordinates $(t, r, \chi = \cos\theta, \phi)$, compactified radius $z = 2r_+/r - 1$, Chebyshev × Legendre expansions; all quantities at fixed ADM mass and angular momentum. Expansion coefficients $X^{(0)}, X^{(1,0)}, X^{(0,1)}, X^{(1,1)}$ are written `X0, X10, X01, X11`.

The 50×50 solutions and the a = 0.99 solution are not included; they are regenerated with step 1 below (about 1.5 minutes per spin at 30×30 and 12 minutes at 50×50 on two cores, under 1 GB of memory).

## Pipeline

**1. Mixed-order solution** (Sec. III, App. B–C). For each input file:

```bash
python ../code/mixed_cache_fixed.py data_a_0.5.json            # -> mixed_a_0.5.json  (~1.5 min at 30x30)
python ../code/mixed_cache_fixed.py data_a_50_0.9.json         # -> mixed_a_50_0.9.json
```

Stage I solves for the mixed scalars, Stage II assembles the complete source (`fix_source.build_source_full`) and solves the equilibrated least-squares system with the inhomogeneous peeling conditions of eq. (peeling11). An existing output is renamed `*.old.json` first. Re-solving `data_a_0.5.json` reproduces the shipped `mixed_a_0.5.json` to $10^{-9}$.

**2. Observables** (Secs. IV–VI, Tables II, IV, V, VII–XIII):

```bash
python ../code/sweep_table.py                     # all mixed_a_*.json except 50x50 -> sweep_corrected.csv/.json
python ../code/sweep_table.py mixed_a_50_0.8.json mixed_a_50_0.9.json    # 50x50 rows for Table II
```

One row per spin: residuals, ADM shifts, $\Omega_H$, $\kappa$ (on the horizon, and the near-horizon extrapolation for reference), area, ISCO and photon radii, $\Omega_c$, $\lambda$ (Richardson-extrapolated in the coupling step, with the plain $\epsilon = 10^{-3}$ value in `*_eps1e-3`), Wald and first-law entropy, and the size of the mixed fields. About 30 s per spin.

**3. Thermodynamics** (Sec. IV D, Fig. 5): `python ../code/thermo_check.py` → `thermo_data.csv` (Wald entropy, its three contributions, first-law values and their relative difference).

**4. Non-equatorial photon orbits** (Sec. V C, App. G, Fig. 7):

```bash
python ../code/spherical_orbits_sweep.py mixed_a_0.5.json pro   spherical_orbits_pro.json
python ../code/spherical_orbits_sweep.py mixed_a_0.5.json retro spherical_orbits_retro.json
```

About 10 minutes per branch; the two can run in parallel.

**5. Numbers quoted in Sec. VII** (perturbative control, observability, low-spin slopes): `python ../code/analysis_numbers.py` → `analysis_numbers.json`.

**6. Conditioning test** (App. C, Table I): `python ../code/trunc_test.py data_a_0.9.json trunc_0.9.json`.

**7. Cross-check of the closed-form $\Omega_H^{(1,1)}$** (App. E, Table III): `python ../code/omega_crossval.py`.

**8. Source audit** (App. B.7): `python ../code/audit_source.py` (needs no data) compares each linearized source term with a finite difference of the exact nonlinear expression.

## Where each result comes from

| Paper | Script | Output |
|---|---|---|
| Residuals, ADM shifts (Sec. III E, App. B) | `mixed_cache_fixed.py`, `sweep_table.py` | `even_residual_rel`, `rchi_residual_rel`, `dM_*`, `dJ_over_a` |
| Fig. 1 (meridional sections) | `mixed_a_X.json` | `H11` coefficients |
| Figs. 2–4, Tables VII–IX, IV | `sweep_table.py` | `Om*`, `k*`, `A*` |
| Fig. 5, Table X | `thermo_check.py`, `sweep_table.py` | `S*_wald`, `S*_fl` |
| Fig. 6, Tables XI–XII, V | `sweep_table.py` | `risco*`, `rph*` |
| Fig. 7 | `spherical_orbits_sweep.py` | `spherical_orbits_{pro,retro}.json` |
| Fig. 8, Table XIII | `sweep_table.py` | `Omc*`, `lam*` |
| Figs. 10–12, Sec. VII E–G | `analysis_numbers.py` | `analysis_numbers.json` |
| Table I | `trunc_test.py` | `trunc_0.9.json` |
| Table II | `mixed_cache_fixed.py` on `data_a_50_*`, then `sweep_table.py` | 50×50 rows |
| Table III | `omega_crossval.py` | printed |

`results/` holds the outputs of steps 2–6 (`sweep_corrected.*`, `thermo_data.csv`, `spherical_orbits_*.json`, `analysis_numbers.json`, `trunc_0.9.json`), regenerated from the files in `data/`. They reproduce every value quoted in the paper to the digits shown: the horizon, photon-ring, $\Omega_c$ and entropy coefficients agree with the original runs to $10^{-8}$ (relative) or better, $\lambda^{(1,1)}$ to about $10^{-6}$ (absolute), and the ISCO coefficients to within their root-finding noise ($\lesssim 4\times10^{-4}$ absolute). Fig. 9 (eikonal fidelity) compares the zero-coupling eikonal construction with the Berti–Cardoso–Will fits for Kerr and involves no mixed-order data.

## Modules

| Module | Role |
|---|---|
| `spectral_grid.py` | Chebyshev × Legendre grid, fits, derivatives, evaluation |
| `kerr_background.py` | Kerr metric, connection, curvature on the grid (SymPy-generated) |
| `perturbation.py`, `einstein_operator.py`, `correction_tensors.py` | linearized curvature, Einstein operator, $\mathcal{H}$ and $\mathcal{C}$ tensors |
| `stage1_solver.py`, `stage2_solver.py` | inversion of $\Box^{(0)}$; Stage-II operator and equilibrated solve |
| `stage2_source.py` | original mixed source pieces |
| `fix_source.py` | complete mixed source (adds the terms missing from `stage2_source.py` and `perturbation.gb_invariant_pert`), peeling conditions, ADM diagnostics |
| `mixed_cache.py` | `load_mixed` (cache reader) |
| `mixed_cache_fixed.py` | the solver used for all results |
| `mixed_observables.py`, `horizon_area.py`, `kappa_horizon.py` | $\Omega_H$, area, surface gravity on the horizon |
| `isco_photon.py`, `qnm_eikonal.py`, `lyapunov_fix.py` | equatorial orbits, eikonal frequency, Lyapunov exponent (five-point stencil) |
| `wald_sgb.py`, `wald_dcs.py`, `wald_mixed.py` | Wald entropy integrals |
| `spherical_orbits.py` | null geodesics off the equator and the turning-point shooting |

Superseded code kept because other modules import from these files: `mixed_cache.solve_and_save` and `stage2_source.build_source` assemble the incomplete source used before the correction; `kappa_mixed.py` extracts $\kappa$ by near-horizon extrapolation (reported only for reference); `qnm_eikonal.lyapunov` uses a three-point stencil. None of these produces a quoted number.

## Pure-sector inputs

The pure-sector metric functions and scalars are the spectral solutions of Lam, Chung & Yunes [Phys. Rev. D 113, 024030 (2026), arXiv:2510.05208]. `mathematica/datagenerator_a.nb` evaluates them at the collocation nodes and writes `data_a_X.json`. It reads their solutions as Mathematica expression files (`sGBH1Expr_raw.wdx` … `sGBH4Expr_raw.wdx`, `sGBPhiFunc_symbolic.wdx`, `dCSH1Func_symbolic.wdx` … `dCSH4Func_symbolic.wdx`, `dCSPhiFunc_symbolic.wdx`), which belong to that work and are not redistributed here; they are needed only to regenerate the inputs, which are included in `data/`. `validation/lam_published_pure_sector.txt` lists their published $\kappa^{(1)}$ and $\Omega_H^{(1)}$ values, which `sweep_table.py` reproduces (`k10`, `k01`, `Om01`).

## Citation

If you use this code or data, please cite the paper above (see `CITATION.cff`) and Lam, Chung & Yunes, Phys. Rev. D 113, 024030 (2026), for the pure-sector solutions.

## License

MIT (see `LICENSE`). The pure-sector inputs in `data/` are evaluations of the solutions of Lam, Chung & Yunes; please cite their paper when using them.
