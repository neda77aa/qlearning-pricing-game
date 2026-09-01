"""
Driver for the TD3 continuous-action robustness experiment.

Purpose
-------
Show that the supracompetitive (collusive) pricing found by the tabular
Q-learners is NOT an artifact of the discrete price grid: firms here choose a
*continuous* price on exactly the same interval the grid spanned, using TD3
(``input.td3learning``). The demand, profit, reference-price dynamics and
Nash/collusive benchmarks are the SAME ``input.init.model`` used everywhere else
in the paper -- only the learner and the (continuous) action space change.

Output layout mirrors the tabular gamma-only runs so the existing plotting code
(``visualization.create_single_heatmap_gamma_only``) works unchanged:

    <main_dir>/<experiment_name>/gamma_<value>/cycle_statistics.csv

Performance note
----------------
Each TD3 step trains two small networks, so a session is far slower than a
tabular one (~3 ms/step on CPU). Deep RL needs far fewer *environment* steps
than the tabular 1e7, though. Use the CONFIG block below to keep the sweep
tractable (fewer gamma points / sessions / steps) for a first pass, then scale
up. Sessions and gamma points are embarrassingly parallel -- run several
gamma values in separate processes if you need the full sweep.
"""

import os
from multiprocessing import freeze_support
import numpy as np

from input.td3learning import run_experiment_parallel_td3


# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
Desired_Experiment = "gamma_only_reference"

# Economic model settings (kept identical to the paper's reference runs).
# These are forwarded verbatim to model(...) for every gamma; do NOT put
# gamma / num_sessions / aprint here (the runner sets those).
GAME_KWARGS = dict(
    n=2,
    k=15,                 # grid size only sets the *interval* [A[0], A[-1]]
    memory=1,
    demand_type="reference",   # your primary case: logit reference-dependent
    common_reference=True,
    lossaversion=1,            # 1 => no loss aversion
)

# Sweep settings. Defaults here are a quick first pass; raise for final.
GAMMA_VALUES = np.linspace(0.0, 3.0, 4)   # e.g. np.linspace(0.05, 3.0, 30) for final
NUM_SESSIONS = 8                          # e.g. 50 for final
NUM_PROCESSES = None                      # None => cpu_count() - 2
MAIN_DIR = "../Results/experiments"

# Passed straight through to simulate_game_td3 (see its docstring).
TD3_KWARGS = dict(
    tmax=150_000,
    start_steps=1_000,
    expl_noise=0.30,
    expl_min=0.02,
    expl_decay=15_000,     # tuned: fast decay (B_fastdecay config)
    anneal_steps=40_000,   # final freeze: exploration + lr -> 0
    min_steps=50_000,
    # convergence: policy stability (Delta_pi)
    pol_check_every=1_000,
    pol_tol_frac=1.5e-3,
    pol_stable_checks=2,
    pol_probe_size=256,
    cycle_rollout=200,
    hidden=128,
    lr=3e-4,
    batch_size=256,
    buffer_size=150_000,   # = tmax: full-history buffer (smaller buffers evict
                           # off-path punishment data -> collusion erodes)
    device="cpu",
)


def run_gamma_sweep():
    experiment_name = (f"td3_{GAME_KWARGS['demand_type']}_"
                       f"{GAME_KWARGS['common_reference']}")

    run_experiment_parallel_td3(
        game_kwargs=GAME_KWARGS,
        gamma_values=GAMMA_VALUES,
        num_sessions=NUM_SESSIONS,
        experiment_name=experiment_name,
        main_dir=MAIN_DIR,
        num_processes=NUM_PROCESSES,
        base_seed=1000,
        **TD3_KWARGS,
    )

    # ---- heatmaps (reuse the existing gamma-only plotting) ---------------- #
    try:
        from input.visualization import create_single_heatmap_gamma_only
        exp_dir = os.path.join(MAIN_DIR, experiment_name)
        figures_dir = os.path.join(exp_dir, "Figures")
        os.makedirs(figures_dir, exist_ok=True)
        for metric, fname in [
            ("Profit", "profit.png"),
            ("Price", "price.png"),
            ("Price Gain", "price_gain.png"),
            ("Profit Gain", "profit_gain.png"),
            ("Cycle Length", "cycle_length.png"),
        ]:
            fig = create_single_heatmap_gamma_only(
                MAIN_DIR, experiment_name=experiment_name, metric_name=metric)
            fig.savefig(os.path.join(figures_dir, fname))
        print(f"\nFigures saved in {figures_dir}")
    except Exception as e:
        print(f"\n[warn] heatmap generation skipped: {e}")


if __name__ == "__main__":
    freeze_support()
    if Desired_Experiment == "gamma_only_reference":
        run_gamma_sweep()
