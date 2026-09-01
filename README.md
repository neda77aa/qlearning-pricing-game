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
main_td3.py                  TD3 (deep RL) sweep — the paper's TD3 run

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
  `creating_results.ipynb` notebook. Everything else (the `_altbench` gain
  variants and the linear / TD3 / IRF / cycle robustness blocks) is produced by
  the single `make_figures.py` script — one subcommand per figure family (§4).
- **Reproducibility:** per-session seeds are drawn from OS entropy, so absolute
  numbers vary slightly run-to-run; qualitative curves are stable. Independent
  session blocks are i.i.d. and may be pooled.

---

## 3. Data drivers (run first)

All commands are `$PY <script>` from the repo root. Output folder is under
`../Results/experiments/`.

| Script | Output folder | What it runs |
|---|---|---|
| `main.py` (`Desired_Experiment='gamma_only'`) | `2*2_2/gamma_only_reference*/` | Tabular logit γ sweep (β=4e-6, the paper default). Source for the **tabular IRF**. Edit the γ grid / session count / `ref_prediction` at the top of the `gamma_only` block. |
| `paper_reruns_stage2.py` | `gamma_nloss_reference_True{c_0,mu_0,...}_qref_beta4e-6_dualconv/` | Market-structure (c=0, μ=0.05), misspecification, and firm-specific (CR=False) variants. |
| `paper_rerun_lossaversion.py` | `lossaversion_reverse_beta4e-6/` | Loss-aversion sweep (φ∈[1,3], γ=1). |
| `main_gamma_only_mu_c.py` | `sweeps/gamma_only_{c,mu}/…` | γ sweeps across a grid of c and μ. |
| `main.py` (`Desired_Experiment='gamma_delta'`) | `gamma_delta/gamma_delta_reference_True_contref/` | γ×δ heatmap grid (30×30, 50 sess). |
| `main_linear.py` | `linear_benchmark/gamma_only_linear[_beta4e-6]/` | Linear-demand γ sweep, **ES reference**. |
| `run_linear_qref.py` | `linear_benchmark/gamma_only_linear_qref_beta4e-6[_dualconv]/` | Linear-demand γ sweep, **Q-learning reference**. |
| `main_td3.py` | `td3_production_reference_15g_50s_lr1e-4/` | **TD3 (deep RL) sweep — the paper's TD3 run.** Self-contained: 15 γ, 50 sess, lr=1e-4, full-history buffer; also writes `rollout_paths.npz` + per-gamma price/profit grids. Source for all TD3 figures. |
| `impulse_response.py {tabular|td3|both}` | `impulse_response/irf[_td3]_gamma_*.npz` | Deviation/punishment simulations from converged strategies. |
| `irf_new_sweep.py` | `…_ESref/Figures/irf_gamma_*_dev-*.npz` | Tabular IRF sweep (imports `impulse_response`). |

---

## 4. Regenerating each paper figure

All figures are produced by **`make_figures.py`**, one subcommand per figure
family, run as `$PY make_figures.py <subcommand>` from the repo root. Output
always lands under `paper_overleaf/Images/…` where the paper `\includegraphics`
reads it — no output-path editing needed.

```
$PY make_figures.py all                    # run the standard committed set (best effort)
$PY make_figures.py altbench [--new]       # alt-benchmark gain panels + heatmaps
$PY make_figures.py irf                    # deviation/punishment panels + LaTeX table rows
$PY make_figures.py recolor-linear-td3     # linear + TD3 price/profit/gain panels (purple)
$PY make_figures.py linear-gains-longterm  # overwrite linear gain panels w/ long-term benchmark
$PY make_figures.py rebuild                # benchmark/market/misspec/firm-specific/lossaversion
$PY make_figures.py td3-cycles             # combined 2x2 TD3 cycle panel
$PY make_figures.py td3-cycles-appendix    # appendix-style per-panel TD3 cycles + legend
$PY make_figures.py linear-postprocess [experiment]   # both-benchmark summary CSV + plots
$PY make_figures.py linear-paper-figs [experiment]    # publication-style linear line plots
$PY make_figures.py panels {linear|td3} <experiment> <out_dir>
```

### Main text

| Paper figure (label) | Image(s) | How to regenerate |
|---|---|---|
| Fig 3 — γ only (`fig:gammaonly_q`) | `4_seperate_figures/benchmark/…` | base panels: `creating_results.ipynb`; alt-benchmark gains: `make_figures.py altbench` |
| Market structure (`fig:gamma_c_mu`) | `4_seperate_figures/market_structure/…` | notebook + `make_figures.py altbench` |
| Deviations/punishments (`fig:irf_mechanism`, `fig:irf_by_gamma`, Tab `irf_tabular`) | `impulse_response/irf_*` | `main.py` (`gamma_only`, β=4e-6) → `irf_new_sweep.py` → `make_figures.py irf` (table rows print to stdout) |
| Misspecification (`fig:gammaonly_refmiss_crtrue`) | `4_seperate_figures/misspecification/…` | notebook + `make_figures.py altbench` |
| Loss aversion (`fig:loss_aversion`) | `4_seperate_figures/lossaversion/…` | notebook (data: `paper_rerun_lossaversion.py`) |
| Firm-specific reference (`fig:gammaonly_crtruefalse_qlr`) | `4_seperate_figures/Firm-specific/…` | notebook + `make_figures.py altbench` |
| Q-learning reference (`fig:qlr_crtrue`) | `4_seperate_figures/exp_smooth/…` | notebook + `make_figures.py altbench` |
| Linear demand (`fig:linear_gamma`) | `4_seperate_figures_beta4e6/linear/…` | `run_linear_qref.py` → `make_figures.py recolor-linear-td3` (price/profit) → `make_figures.py linear-gains-longterm` (overwrites the two gain panels) |
| TD3 (`fig:td3_gamma`, Tab `irf_td3`) | `4_seperate_figures_lr1e-4/td3/…` | `main_td3.py` → `make_figures.py recolor-linear-td3` + `make_figures.py altbench`; TD3 IRF table: `impulse_response.py td3` → `make_figures.py irf` |
| TD3 cycles (`fig:td3_cycles`) | `4_seperate_figures_lr1e-4/td3_cycles/td3_cycle_examples.png` | `make_figures.py td3-cycles` (reads `rollout_paths.npz`) |
| Intro schematics | `Images/idea.png`, `Images/framework.png` | static assets (no script) |
| Consumer-welfare tables (`tab:consumer_*`) | — | hand-authored LaTeX (no script) |

`make_figures.py altbench` has two modes: without flags (benchmark,
market_structure, misspecification, Firm-specific, exp_smooth, td3, gamma_lambda,
gamma_delta) and `make_figures.py altbench --new` (the appendix `exp_smoothing_*`
and `Separated_Panels_*`/`Seperated_Panels_CR` blocks).

### Appendix (`appendix_extension.tex`)

| Paper figure | Image(s) | How to regenerate |
|---|---|---|
| Cycle examples | `4_seperate_figures/appendix_cycles/…` | `creating_results.ipynb` |
| Cycle histograms | `4_seperate_figures/histogram/…` | `creating_results.ipynb` |
| α–β diff panels | `4_seperate_figures/Separated_Panels_AlphaBeta_DiffOnly/…` | `creating_results.ipynb` |
| γ×δ heatmaps (`fig:gamma_delta`) | `gamma_delta/…_heatmap[_altbench].png` | data: `main.py`; alt-benchmark: `make_figures.py altbench` |
| γ×λ heatmaps | `4_seperate_figures/gamma_lambda/…` | notebook + `make_figures.py altbench --new` |
| exp-smoothing misspec / firm-specific | `4_seperate_figures/exp_smoothing_*/…` | notebook + `make_figures.py altbench --new` |
| Separated misspec / CR panels | `4_seperate_figures/{Separated_Panels_miss,Seperated_Panels_CR}/…` | notebook + `make_figures.py altbench --new` |
| Linear ES (`fig:linear_es`) | `4_seperate_figures_beta4e6/linear_es/…` | data: `main_linear.py`; gain panels: `make_figures.py linear-gains-longterm` |
| TD3 cycles, appendix style | `4_seperate_figures_lr1e-4/td3_cycles/td3_cycle_L{1,2,4,6}.png` + `td3_cycle_legend.png` | `make_figures.py td3-cycles-appendix` (used by `appendix_extension_revised.tex`) |
| γ cycle probability | `4_seperate_figures/cycle/…` | `creating_results.ipynb` |

Publication-style linear line plots / panels (alternative styling, not the main
Fig 11) come from `make_figures.py linear-postprocess` →
`make_figures.py linear-paper-figs` / `make_figures.py panels`.

---

## 5. Notes & caveats

- **Figure output path.** All of `make_figures.py` writes under
  `paper_overleaf/Images/…` (the paths are set by the `IMG` constant at the top
  of the file), so figures land where the paper `\includegraphics` reads them
  with no editing needed.
- **`_dualconv` folders.** The newest drivers append `_dualconv` to their output
  folder name (dual firm+reference convergence rule — a robustness check). The
  paper's figures read the **non-`_dualconv`** folders; the `_dualconv` runs are
  archived separately under `../Result_double_convergence/`.
- **Dual convergence rule** (in `input/qlearning.py`) is opt-in via
  `require_reference_stability=True`; default is off so committed paper runs are
  unchanged.
- **`creating_results.ipynb`** produces the base logit 4-panel figures and most
  appendix grids. `make_figures.py` covers everything else (alt-benchmark gains,
  linear, TD3, IRF, cycles).
