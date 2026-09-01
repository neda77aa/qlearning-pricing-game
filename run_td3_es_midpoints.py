"""Densify the TD3 exponential-smoothing gamma sweep.

The existing run td3_production_reference_15g_50s_lr1e-4 has 15 gammas at
np.linspace(0.05, 3.0, 15). This script computes the 14 MIDPOINTS between those
consecutive gammas and runs ONLY those new points (50 sessions each, same
lr=1e-4 config), writing into the SAME experiment folder. skip_existing=True
means the original 15 are left untouched -- we do not re-simulate them.

Combined, the folder then holds a uniform 29-point grid
(== np.linspace(0.05, 3.0, 29)), so recolor_linear_td3_purple.py regenerates a
denser TD3 figure automatically.

Note: 15 points have 14 gaps, so "exactly in between" is 14 new midpoints.

Run:  /Users/neda/llm_venv/bin/python run_td3_es_midpoints.py
"""
import numpy as np
from multiprocessing import freeze_support

import production_sweep as ps
from input.td3learning import run_experiment_parallel_td3

ORIG = np.round(np.linspace(0.05, 3.0, 15), 4)           # existing 15
MIDS = np.round((ORIG[:-1] + ORIG[1:]) / 2.0, 4)         # 14 new midpoints
FULL = np.round(np.sort(np.concatenate([ORIG, MIDS])), 4)  # 29-point grid

EXPERIMENT = "td3_production_reference_15g_50s_lr1e-4"    # same folder

if __name__ == "__main__":
    freeze_support()

    # so make_gamma_figures numbers cells by position in the full 29-pt grid
    ps.EXPERIMENT = EXPERIMENT
    ps.GAMMA_VALUES = FULL
    ps.NUM_SESSIONS = 50
    ps.SIM_KWARGS["lr"] = 1e-4

    print(f"TD3 ES midpoint densify -> {EXPERIMENT}")
    print(f"  existing 15: {list(ORIG)}")
    print(f"  new 14 mids: {list(MIDS)}")

    run_experiment_parallel_td3(
        game_kwargs=ps.GAME_KWARGS,
        gamma_values=MIDS,             # only the new midpoints
        num_sessions=ps.NUM_SESSIONS,
        experiment_name=EXPERIMENT,
        main_dir=ps.MAIN_DIR,
        num_processes=10,
        base_seed=1000,
        per_gamma_callback=ps.make_gamma_figures,
        skip_existing=True,            # never touch the existing 15
        **ps.SIM_KWARGS,
    )

    # refresh the gamma-only heatmaps over the full 29-point grid
    try:
        import os
        from input.visualization import create_single_heatmap_gamma_only
        fig_dir = os.path.join(ps.MAIN_DIR, EXPERIMENT, "Figures")
        os.makedirs(fig_dir, exist_ok=True)
        for metric, fname in [("Profit", "profit.png"), ("Price", "price.png"),
                              ("Price Gain", "price_gain.png"),
                              ("Profit Gain", "profit_gain.png"),
                              ("Cycle Length", "cycle_length.png")]:
            fig = create_single_heatmap_gamma_only(
                ps.MAIN_DIR, experiment_name=EXPERIMENT, metric_name=metric)
            fig.savefig(os.path.join(fig_dir, fname))
        print(f"heatmaps refreshed in {fig_dir}")
    except Exception as e:
        print(f"[warn] heatmap generation skipped: {e}")

    print("TD3 ES midpoint densify complete.")
