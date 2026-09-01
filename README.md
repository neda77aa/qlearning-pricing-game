# Algorithmic Collusion under Reference Dependence — Replication Code

Simulation code for the paper on algorithmic collusion when consumers are
**reference-dependent**. Firms are Q-learning (tabular) or TD3 (deep RL) pricing
agents; consumers evaluate prices relative to a learned **reference price**. The
key parameter is `γ` (reference dependence); the paper sweeps `γ` (and, in the
appendix, `γ × λ` and `γ × δ`) and reports equilibrium prices, profits, and the
implied collusion gains, plus impulse-response (deviation/punishment) tests.

---

## 1. Repository layout

```
input/                       core model + learning code (imported by every driver)
  init.py                    base Model (logit demand, defaults c=1, μ=0.25)
  init_linear.py             LinearModel (linear demand)
  qlearning.py               tabular Q-learning firms (+ dual firm/reference convergence rule)
  qlearning_reference.py     Q-learning consumer-reference agent
  td3learning.py             TD3 (continuous-action) firms
  ConvResults*.py            experiment runners / convergence + cycle statistics
  visualization.py           heatmap / panel helpers

main.py                      single-experiment entry point (γ×δ grid is the active default)
main_gamma_only_mu_c.py      γ sweeps at varied μ and c
main_linear.py               linear-demand γ sweep (exponential-smoothing reference)
main_td3.py                  quick TD3 smoke run

<driver scripts>             produce results under ../Results/experiments  (see §3)
<figure scripts>             turn results into the paper's PNGs             (see §4)

paper_overleaf/              the LaTeX paper + its Images/ (the compiled figure set)
reviewer_notes_fixes/        reviewer-response notes
```

> **Results are written outside the repo.** Every driver writes to the sibling
> folder `../Results/experiments/` (i.e.
> `.../research_term_4/Results/experiments/`). This keeps large simulation
> output out of git.

---

## 2. Setup & conventions

- **Interpreter:** the project uses a local venv. Substitute your own if needed:
  ```
  PY=/Users/neda/llm_venv/bin/python      # or: PY=python
  ```
- **Always run from the repo root** (`Algorithmic-Collusion-Replication/`).
  Every script uses the relative path `../Results/experiments`.
- **Two-step pipeline for every result:** first run a **driver** to generate the
  simulation data (§3), then run a **figure script** to render the paper PNG (§4).
- **Base 4-panel logit figures** (`figure1_gamma_only_q_{price,profit,price_gain,
  profit_gain}.png`) and most appendix grids are produced by the
  `creating_results.ipynb` notebook. The `.py` figure scripts add the
  `_altbench` gain variants and the linear / TD3 / IRF / cycle robustness blocks.
- **Reproducibility:** per-session seeds are drawn from OS entropy, so absolute
  numbers vary slightly run-to-run; qualitative curves are stable. Independent
  session blocks are i.i.d. and may be pooled.

---

## 3. Data drivers (run first)

All commands are `$PY <script>` from the repo root. Output folder is under
`../Results/experiments/`.

| Script | Output folder | What it runs |
|---|---|---|
| `tabular_sweep_beta4e6.py` | `gamma_nloss_reference_True_beta4e-6_ESref/` | Tabular logit γ sweep (30 γ∈[0.05,3], 50 sess, β=4e-6, ES reference). Source for the **tabular IRF**. |
| `tabular_sweep_beta4e6_qref.py` | `gamma_nloss_reference_True_qref_beta4e-6_dualconv/` | Baseline γ sweep with a **Q-learning consumer reference**. |
| `paper_reruns_stage2.py` | `gamma_nloss_reference_True{c_0,mu_0,...}_qref_beta4e-6_dualconv/` | Market-structure (c=0, μ=0.05), misspecification, and firm-specific (CR=False) variants. |
| `paper_rerun_lossaversion.py` | `lossaversion_reverse_beta4e-6/` | Loss-aversion sweep (φ∈[1,3], γ=1). |
| `main_gamma_only_mu_c.py` | `sweeps/gamma_only_{c,mu}/…` | γ sweeps across a grid of c and μ. |
| `main.py` (`Desired_Experiment='gamma_delta'`) | `gamma_delta/gamma_delta_reference_True_contref/` | γ×δ heatmap grid (30×30, 50 sess). |
| `main_linear.py` | `linear_benchmark/gamma_only_linear[_beta4e-6]/` | Linear-demand γ sweep, **ES reference**. |
| `run_linear_qref.py` | `linear_benchmark/gamma_only_linear_qref_beta4e-6[_dualconv]/` | Linear-demand γ sweep, **Q-learning reference**. |
| `production_sweep.py` | `td3_production_reference_15g_20s/` | TD3 production sweep, lr=3e-4 (baseline). |
| `run_td3_lr1e4.py` | `…_15g_20s_lr1e-4/` | TD3 sweep, lr=1e-4, 20 sess. |
| `run_td3_lr1e4_50s.py` | `td3_production_reference_15g_50s_lr1e-4/` | TD3 sweep, lr=1e-4, **50 sess** — source for the paper's TD3 figures. |
| `run_td3_qref.py` | `…_15g_50s_lr1e-4_qref/` | TD3 sweep with a Q-learning reference. |
| `impulse_response.py {tabular|td3|both}` | `impulse_response/irf[_td3]_gamma_*.npz` | Deviation/punishment simulations from converged strategies. |
| `irf_new_sweep.py` | `…_ESref/Figures/irf_gamma_*_dev-*.npz` | Tabular IRF sweep (imports `impulse_response`). |

---

## 4. Regenerating each paper figure

`$PY <script>` from the repo root. Figures land under
`paper_overleaf/Images/…` (some scripts hard-code a
`Final_Paper__Reference_Dependence__Copy2_/Images/` output path — see §5; repoint
that constant to `paper_overleaf/Images` before running).

### Main text

| Paper figure (label) | Image(s) | How to regenerate |
|---|---|---|
| Fig 3 — γ only (`fig:gammaonly_q`) | `4_seperate_figures/benchmark/…` | base panels: `creating_results.ipynb`; alt-benchmark gains: `gen_altbench_gains.py` |
| Market structure (`fig:gamma_c_mu`) | `4_seperate_figures/market_structure/…` | notebook + `gen_altbench_gains.py` |
| Deviations/punishments (`fig:irf_mechanism`, `fig:irf_by_gamma`, Tab `irf_tabular`) | `impulse_response/irf_*` | `tabular_sweep_beta4e6.py` → `irf_new_sweep.py` → `paper_irf_figures.py` (table rows print to stdout) |
| Misspecification (`fig:gammaonly_refmiss_crtrue`) | `4_seperate_figures/misspecification/…` | notebook + `gen_altbench_gains.py` |
| Loss aversion (`fig:loss_aversion`) | `4_seperate_figures/lossaversion/…` | notebook (data: `paper_rerun_lossaversion.py`) |
| Firm-specific reference (`fig:gammaonly_crtruefalse_qlr`) | `4_seperate_figures/Firm-specific/…` | notebook + `gen_altbench_gains.py` |
| Q-learning reference (`fig:qlr_crtrue`) | `4_seperate_figures/exp_smooth/…` | notebook + `gen_altbench_gains.py` |
| Linear demand (`fig:linear_gamma`) | `4_seperate_figures_beta4e6/linear/…` | `run_linear_qref.py` → `recolor_linear_td3_purple.py` (price/profit) → `recompute_linear_gains_longterm.py` (overwrites the two gain panels) |
| TD3 (`fig:td3_gamma`, Tab `irf_td3`) | `4_seperate_figures_lr1e-4/td3/…` | `run_td3_lr1e4_50s.py` → `recolor_linear_td3_purple.py` + `gen_altbench_gains.py`; TD3 IRF table: `impulse_response.py td3` → `paper_irf_figures.py` |
| TD3 cycles (`fig:td3_cycles`) | `4_seperate_figures_lr1e-4/td3_cycles/td3_cycle_examples.png` | `plot_td3_cycles.py` (reads `rollout_paths.npz`) |
| Intro schematics | `Images/idea.png`, `Images/framework.png` | static assets (no script) |
| Consumer-welfare tables (`tab:consumer_*`) | — | hand-authored LaTeX (no script) |

`gen_altbench_gains.py` has two modes: `gen_altbench_gains.py` (benchmark,
market_structure, misspecification, Firm-specific, exp_smooth, td3, gamma_lambda,
gamma_delta) and `gen_altbench_gains.py new` (the appendix `exp_smoothing_*` and
`Separated_Panels_*`/`Seperated_Panels_CR` blocks).

### Appendix (`appendix_extension.tex`)

| Paper figure | Image(s) | How to regenerate |
|---|---|---|
| Cycle examples | `4_seperate_figures/appendix_cycles/…` | `creating_results.ipynb` |
| Cycle histograms | `4_seperate_figures/histogram/…` | `creating_results.ipynb` |
| α–β diff panels | `4_seperate_figures/Separated_Panels_AlphaBeta_DiffOnly/…` | `creating_results.ipynb` |
| γ×δ heatmaps (`fig:gamma_delta`) | `gamma_delta/…_heatmap[_altbench].png` | data: `main.py`; alt-benchmark: `gen_altbench_gains.py` |
| γ×λ heatmaps | `4_seperate_figures/gamma_lambda/…` | notebook + `gen_altbench_gains.py new` |
| exp-smoothing misspec / firm-specific | `4_seperate_figures/exp_smoothing_*/…` | notebook + `gen_altbench_gains.py new` |
| Separated misspec / CR panels | `4_seperate_figures/{Separated_Panels_miss,Seperated_Panels_CR}/…` | notebook + `gen_altbench_gains.py new` |
| Linear ES (`fig:linear_es`) | `4_seperate_figures_beta4e6/linear_es/…` | data: `main_linear.py`; gain panels: `recompute_linear_gains_longterm.py` |
| γ cycle probability | `4_seperate_figures/cycle/…` | `creating_results.ipynb` |

Publication-style linear line plots / panels (alternative styling, not the main
Fig 11) come from `postprocess_linear_benchmarks.py` →
`paper_figures_linear.py` / `paper_panels.py`.

---

## 5. Notes & caveats

- **Output-path rewrite.** `gen_altbench_gains.py`, `rebuild_paper_figures.py`,
  `recolor_linear_td3_purple.py`, `recompute_linear_gains_longterm.py`,
  `paper_irf_figures.py`, and `plot_td3_cycles.py` hard-code their output to a
  `Final_Paper__Reference_Dependence__Copy2_/Images/` directory. Point that
  constant at `paper_overleaf/Images/` before running so the figures land where
  the paper `\includegraphics` reads them.
- **`_dualconv` folders.** The newest drivers append `_dualconv` to their output
  folder name (dual firm+reference convergence rule — a robustness check). The
  paper's figure scripts read the **non-`_dualconv`** folders; the `_dualconv`
  runs are archived separately under
  `../Result_double_convergence/` and analysed with
  `analyze_gamma_ref_qlearn.py` / `preview_*_dualconv.py`.
- **Dual convergence rule** (in `input/qlearning.py`) is opt-in via
  `require_reference_stability=True`; default is off so committed paper runs are
  unchanged. See `run_gamma_ref_qlearn_diag.py` for the diagnostic driver.
- **`creating_results.ipynb`** produces the base logit 4-panel figures and most
  appendix grids. The `.py` figure scripts cover everything else (alt-benchmark
  gains, linear, TD3, IRF, cycles).
