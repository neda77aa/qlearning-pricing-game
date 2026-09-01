"""TD3 (deep RL) continuous-action pricing sweep — the paper's TD3 robustness run.

Purpose
-------
Show that the supracompetitive (collusive) pricing found by the tabular
Q-learners is NOT an artifact of the discrete price grid: firms here choose a
*continuous* price on exactly the same interval the grid spanned, using TD3
(``input.td3learning``). The demand, profit, reference-price dynamics and
Nash/collusive benchmarks are the SAME ``input.init.model`` used everywhere else
in the paper — only the learner and the (continuous) action space change.

This single script is the whole TD3 driver (it absorbs what used to be split
across ``production_sweep.py`` + ``run_td3_lr1e4_50s.py``). Running it with the
DEFAULT config below reproduces the paper's TD3 results:

    ../Results/experiments/td3_production_reference_15g_50s_lr1e-4/
        gamma_<g>/cycle_statistics.csv        per-gamma summary stats
        gamma_<g>/rollout_paths.npz           frozen-rollout prices/profits, every session
        Figures/exp<NN>_gamma_<g>_prices.png  grid: converged prices, all sessions
        Figures/exp<NN>_gamma_<g>_profits.png grid: converged profits, all sessions
        Figures/{price,profit,price_gain,profit_gain,cycle_length}.png  gamma-only heatmaps

These feed the paper's TD3 figures via ``recolor_linear_td3_purple.py`` /
``gen_altbench_gains.py`` (Fig td3_gamma) and ``plot_td3_cycles.py`` (Fig
td3_cycles, which reads rollout_paths.npz).

Config: 15 gammas ∈ [0.05, 3], 50 sessions/gamma (matches the tabular/linear
paper runs), lr=1e-4 (the continuous-action analog of the corrected
more-exploration tabular config), full-history replay buffer, anneal-to-freeze,
Δπ policy-stability convergence.

Resume: the runner skips gammas whose output already exists, so re-running
continues an interrupted sweep.

Performance: each TD3 step trains two small networks (~3 ms/step on CPU), so a
session is far slower than a tabular one but needs far fewer environment steps.
For a quick smoke test, drop NUM_SESSIONS and the number of GAMMA_VALUES.

Run:  /Users/neda/llm_venv/bin/python main_td3.py
"""
import os
from multiprocessing import freeze_support

import numpy as np

from input.td3learning import run_experiment_parallel_td3

# --------------------------------------------------------------------------- #
# CONFIG  (defaults reproduce the paper's TD3 run)
# --------------------------------------------------------------------------- #
GAMMA_VALUES = np.round(np.linspace(0.05, 3.0, 15), 4)
NUM_SESSIONS = 50
NUM_PROCESSES = 10
MAIN_DIR = "../Results/experiments"
EXPERIMENT = "td3_production_reference_15g_50s_lr1e-4"

# Economic model (identical to the paper's reference runs).
GAME_KWARGS = dict(n=2, k=15, memory=1, demand_type="reference",
                   common_reference=True, lossaversion=1)

# Passed straight through to simulate_game_td3 (see its docstring).
SIM_KWARGS = dict(
    tmax=150_000, start_steps=1_000,
    expl_noise=0.30, expl_min=0.02, expl_decay=15_000,
    anneal_steps=40_000, freeze_tail=10_000,
    min_steps=50_000,
    pol_check_every=1_000, pol_tol_frac=4e-3, pol_stable_checks=2,
    pol_probe_size=256,
    cycle_rollout=1_000, cycle_tol_frac=2e-3,
    hidden=128, lr=1e-4, batch_size=256,
    buffer_size=150_000,   # = tmax: full history (prevents punishment forgetting)
    device="cpu",
)

# figure style (reference palette, light mode)
S1, S2 = "#2a78d6", "#eb6834"
SURF, INK, SEC, MUT = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
BASE = "#c3c2b7"
T_SHOW = 100          # last T_SHOW periods of the rollout shown per cell


def _grid_figure(paths, kind, refs, gamma, out_png):
    """Grid of per-session converged paths ('prices' or 'profits'),
    one cell per session showing the last ``T_SHOW`` rollout periods."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = sorted((k for k in paths.files if k.startswith(kind)),
                  key=lambda k: int(k.split("_s")[1]))
    n_sess = len(keys)
    ncol = 10 if n_sess > 25 else 5
    nrow = int(np.ceil(n_sess / ncol))
    lo_ref, hi_ref = refs

    # common y-limits across all cells of this figure
    gmin, gmax = np.inf, -np.inf
    cells = []
    for k in keys:
        P = paths[k]
        W = P[:, -T_SHOW:]
        cells.append(W)
        gmin, gmax = min(gmin, W.min()), max(gmax, W.max())
    ylo = min(lo_ref, gmin) - 0.03 * (gmax - gmin + 1e-9)
    yhi = max(hi_ref, gmax) + 0.03 * (gmax - gmin + 1e-9)

    fig, axes = plt.subplots(nrow, ncol, figsize=(20, 2.0 * nrow), dpi=110)
    fig.patch.set_facecolor(SURF)
    axes = np.atleast_2d(axes)
    for ax in axes.flat:
        ax.set_visible(False)
    for i, (k, W) in enumerate(zip(keys, cells)):
        ax = axes.flat[i]
        ax.set_visible(True)
        ax.set_facecolor(SURF)
        for y in refs:
            ax.axhline(y, color=MUT, lw=0.7, ls=(0, (4, 3)), zorder=1)
        ax.plot(W[0], color=S1, lw=0.9, zorder=3)
        ax.plot(W[1], color=S2, lw=0.9, zorder=3)
        ax.set_ylim(ylo, yhi)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(BASE); s.set_linewidth(0.5)
        ax.set_title(k.replace(kind + "_", ""), color=SEC, fontsize=6, pad=2)
    fig.suptitle(f"gamma = {gamma:g} — converged {kind} "
                 f"(last {T_SHOW} periods of each session's frozen rollout; "
                 f"dashed = Nash / Coop)",
                 color=INK, fontsize=11, x=0.05, y=0.995, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, facecolor=SURF, bbox_inches="tight")
    plt.close(fig)


def make_gamma_figures(game, run_dir, gamma):
    """per_gamma_callback: draw the converged price/profit grids."""
    idx = int(np.argmin(np.abs(GAMMA_VALUES - gamma))) + 1
    fig_dir = os.path.join(MAIN_DIR, EXPERIMENT, "Figures")
    os.makedirs(fig_dir, exist_ok=True)
    paths = np.load(os.path.join(run_dir, "rollout_paths.npz"))
    _grid_figure(paths, "prices",
                 (float(game.p_nash[0]), float(game.p_coop[0])), gamma,
                 os.path.join(fig_dir, f"exp{idx:02d}_gamma_{gamma:g}_prices.png"))
    _grid_figure(paths, "profits",
                 (float(game.NashProfits[0]), float(game.CoopProfits[0])), gamma,
                 os.path.join(fig_dir, f"exp{idx:02d}_gamma_{gamma:g}_profits.png"))
    print(f"  figures saved for gamma={gamma:g} (exp{idx:02d})")


def main():
    print(f"TD3 sweep: {len(GAMMA_VALUES)} gammas x {NUM_SESSIONS} sessions, "
          f"lr={SIM_KWARGS['lr']:g} -> {EXPERIMENT}")
    run_experiment_parallel_td3(
        game_kwargs=GAME_KWARGS,
        gamma_values=GAMMA_VALUES,
        num_sessions=NUM_SESSIONS,
        experiment_name=EXPERIMENT,
        main_dir=MAIN_DIR,
        num_processes=NUM_PROCESSES,
        base_seed=1000,
        per_gamma_callback=make_gamma_figures,
        **SIM_KWARGS,
    )

    # standard gamma-only heatmaps
    try:
        from input.visualization import create_single_heatmap_gamma_only
        fig_dir = os.path.join(MAIN_DIR, EXPERIMENT, "Figures")
        os.makedirs(fig_dir, exist_ok=True)
        for metric, fname in [("Profit", "profit.png"), ("Price", "price.png"),
                              ("Price Gain", "price_gain.png"),
                              ("Profit Gain", "profit_gain.png"),
                              ("Cycle Length", "cycle_length.png")]:
            fig = create_single_heatmap_gamma_only(
                MAIN_DIR, experiment_name=EXPERIMENT, metric_name=metric)
            fig.savefig(os.path.join(fig_dir, fname))
        print(f"heatmaps saved in {fig_dir}")
    except Exception as e:
        print(f"[warn] heatmap generation skipped: {e}")


if __name__ == "__main__":
    freeze_support()
    main()
