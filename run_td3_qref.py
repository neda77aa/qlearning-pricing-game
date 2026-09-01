"""TD3 gamma sweep with a Q-learning consumer reference agent.

The continuous-action counterpart of tabular_sweep_beta4e6_qref.py /
run_linear_qref.py: firms are TD3 agents (as in the paper's TD3 robustness run
at lr=1e-4) but consumers form the common reference price with a pretrained
Q-learning reference agent (ref_prediction='qlearning', T_ref=2e5) instead of
exponential smoothing.

Same 15 gammas x 50 sessions x lr=1e-4 as td3_production_reference_15g_50s_lr1e-4
so the two are directly comparable (ES reference vs Q-learning reference).

Run:  /Users/neda/llm_venv/bin/python run_td3_qref.py
Results -> ../Results/experiments/td3_production_reference_15g_50s_lr1e-4_qref
"""
import os
import numpy as np
from multiprocessing import freeze_support

from input.td3learning import run_experiment_parallel_td3
import production_sweep as ps

EXPERIMENT = "td3_production_reference_15g_50s_lr1e-4_qref"
GAMMA_VALUES = np.round(np.linspace(0.05, 3.0, 15), 4)
NUM_SESSIONS = 50
NUM_PROCESSES = 8
MAIN_DIR = "../Results/experiments"
T_REF = int(2e5)

# firm model: reference-aware, common reference, Q-learning reference formation
GAME_KWARGS = dict(n=2, k=15, memory=1, demand_type="reference",
                   common_reference=True, lossaversion=1,
                   ref_prediction="qlearning")

SIM_KWARGS = dict(ps.SIM_KWARGS)   # copy the validated TD3 config
SIM_KWARGS["lr"] = 1e-4            # match the paper's TD3 robustness run

if __name__ == "__main__":
    freeze_support()

    # let production_sweep's per-gamma figure callback write into our folder
    ps.EXPERIMENT = EXPERIMENT
    ps.GAMMA_VALUES = GAMMA_VALUES
    ps.NUM_SESSIONS = NUM_SESSIONS

    print(f"TD3 + Q-learning reference sweep -> {EXPERIMENT}")
    run_experiment_parallel_td3(
        game_kwargs=GAME_KWARGS,
        gamma_values=GAMMA_VALUES,
        num_sessions=NUM_SESSIONS,
        experiment_name=EXPERIMENT,
        main_dir=MAIN_DIR,
        num_processes=NUM_PROCESSES,
        base_seed=1000,
        per_gamma_callback=ps.make_gamma_figures,
        use_reference_pretraining=True,
        T_ref=T_REF,
        train_reference=True,
        session_timeout=1800,
        **SIM_KWARGS,
    )

    # standard gamma-only heatmaps (same set/names as the other sweeps)
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

    print("TD3 qref sweep complete ->", os.path.join(MAIN_DIR, EXPERIMENT))
