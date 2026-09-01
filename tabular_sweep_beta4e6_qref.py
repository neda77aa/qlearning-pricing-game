"""Q-learning-reference gamma sweep at the corrected exploration rate.

Mirror of paper_results/qqlearning/gamma_nloss_only_reference_True
(30 gammas, demand_type='reference', common_reference=True, lossaversion=1,
lambda=0.6, alpha=0.15, consumer reference formed by a Q-learning agent with
pretraining T_ref=2e5) -- changing ONLY beta: 4e-5 -> 4e-6.

Note: continuous_reference does not apply here (the ES branch is bypassed when
ref_prediction='qlearning'; the consumer agent predicts the reference index).

Run:  /Users/neda/llm_venv/bin/python tabular_sweep_beta4e6_qref.py
"""
import numpy as np
from multiprocessing import freeze_support

from input.init import model
from input.ConvResults_gamma_lambda import run_experiment_parallel_gamma_only

EXPERIMENT = "gamma_nloss_reference_True_qref_beta4e-6_dualconv"
GAMMAS = [round(float(g), 4) for g in np.linspace(0.05, 3.0, 30)]
NUM_SESSIONS = 50
NUM_PROCESSES = 8

if __name__ == "__main__":
    freeze_support()
    game = model(n=2, k=15, memory=1, demand_type="reference",
                 common_reference=True, lossaversion=1, gamma=1.0,
                 num_sessions=NUM_SESSIONS, aprint=False,
                 ref_prediction="qlearning",
                 require_reference_stability=True,   # dual firm+reference convergence
                 track_q_stabilization=True)         # Q-value stabilization diagnostic
    run_experiment_parallel_gamma_only(
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
    print("qref sweep complete ->", f"../Results/experiments/{EXPERIMENT}")
