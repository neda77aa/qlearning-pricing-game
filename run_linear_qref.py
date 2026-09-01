"""Linear-demand gamma sweep at the corrected (low) exploration rate with a
Q-learning consumer reference agent.

Mirrors the linear ES run (main_linear.py defaults: 30 gammas, 50 sessions,
beta=4e-6, alpha=0.15, lambda=0.6, lossaversion=1) but forms the consumer
reference with a pretrained Q-learning agent (ref_prediction='qlearning',
T_ref=2e5) instead of exponential smoothing -- the linear counterpart of the
tabular Q-reference sweep (main.py, gamma_only block with
ref_prediction='qlearning').

Note: continuous_reference does not apply here (the ES branch is bypassed when
ref_prediction='qlearning').

Run:  /Users/neda/llm_venv/bin/python run_linear_qref.py
Results -> ../Results/experiments/linear_benchmark/gamma_only_linear_qref_beta4e-6
"""
import os
import numpy as np
from multiprocessing import freeze_support

from input.init_linear import LinearModel
from input.ConvResults_gamma_lambda import run_experiment_parallel_gamma_only
from input.visualization import create_single_heatmap_gamma_only

EXPERIMENT = "linear_benchmark/gamma_only_linear_qref_beta4e-6_dualconv"
GAMMAS = [round(float(g), 4) for g in np.linspace(0.05, 3.0, 30)]
NUM_SESSIONS = 50
NUM_PROCESSES = 8
MAIN_DIR = "../Results/experiments"

if __name__ == "__main__":
    freeze_support()

    # ---- one fixed price grid spanning the whole gamma range (as in main_linear.py) ----
    n_firms = 2
    g_lo, g_hi = min(GAMMAS), max(GAMMAS)
    p_nash_min = 1.0 / (1.0 + n_firms * (1.0 + g_hi))   # lowest competitive price (gamma_max)
    p_coop_max = 1.0 / (2.0 + g_lo)                      # highest collusive price (gamma_min)
    pad = 0.1 * (p_coop_max - p_nash_min)
    grid_bounds = (max(0.0, p_nash_min - pad), p_coop_max + pad)
    print(f"linear grid bounds: {grid_bounds}")

    game = LinearModel(
        n=n_firms, k=15, memory=1,
        num_sessions=NUM_SESSIONS, aprint=False,
        common_reference=True,
        ref_prediction="qlearning",
        grid_bounds=grid_bounds,
        continuous_reference=False,   # ES branch bypassed for qlearning reference
        require_reference_stability=True,   # dual firm+reference convergence
        track_q_stabilization=True,         # Q-value stabilization diagnostic
    )

    game = run_experiment_parallel_gamma_only(
        game, GAMMAS,
        num_sessions=NUM_SESSIONS,
        experiment_name=EXPERIMENT,
        demand_type="reference",
        num_processes=NUM_PROCESSES,
        use_reference_pretraining=True,
        T_ref=int(2e5),
        lambda_fixed=0.6,
        alpha=0.15,
        beta=4e-6,
        lossaversion_fixed=1.0,
        session_timeout=1800,
    )

    # ---- figures (same set/names as the ES linear + qref tabular folders) ----
    figures_dir = os.path.join(MAIN_DIR, EXPERIMENT, "Figures")
    os.makedirs(figures_dir, exist_ok=True)
    for metric, fname in [("Price", "price.png"),
                          ("Price Gain", "price_gain.png"),
                          ("Profit", "profit.png"),
                          ("Profit Gain", "profit_gain.png"),
                          ("Cycle Length", "cycle_length.png")]:
        fig = create_single_heatmap_gamma_only(MAIN_DIR, experiment_name=EXPERIMENT, metric_name=metric)
        fig.savefig(os.path.join(figures_dir, fname), dpi=300, bbox_inches="tight")
    print("linear qref sweep complete ->", os.path.join(MAIN_DIR, EXPERIMENT))
