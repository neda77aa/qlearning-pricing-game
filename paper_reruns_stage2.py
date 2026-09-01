"""Stage 2 of the paper rerun: the four QL-reference gamma-only variants at
beta=4e-6 (market_structure c=0 / mu=0.05, misspecification, firm-specific).

All mirror the paper originals exactly (30 gammas, 50 sessions, alpha=0.15,
lambda=0.6, lossaversion=1, ref_prediction='qlearning', pretraining T_ref=2e5)
-- only beta changes: 4e-5 -> 4e-6.

Run:  /Users/neda/llm_venv/bin/python paper_reruns_stage2.py
"""
import numpy as np
from multiprocessing import freeze_support

from input.init import model
from input.ConvResults_gamma_lambda import run_experiment_parallel_gamma_only

GAMMAS = [round(float(g), 4) for g in np.linspace(0.05, 3.0, 30)]
N_SESS, N_PROC = 50, 8

# (experiment_name, demand_type, common_reference, extra model kwargs)
JOBS = [
    ("gamma_nloss_reference_Truec_0_qref_beta4e-6_dualconv", "reference", True, dict(c=0)),
    ("gamma_nloss_reference_Truemu_0_qref_beta4e-6_dualconv", "reference", True, dict(mu=0.05)),
    ("gamma_nloss_misspecification_True_qref_beta4e-6_dualconv", "misspecification", True, dict()),
    ("gamma_nloss_reference_False_qref_beta4e-6_dualconv", "reference", False, dict()),
]

if __name__ == "__main__":
    freeze_support()
    for name, dtype, cr, extra in JOBS:
        print(f"\n######## {name} ########")
        game = model(n=2, k=15, memory=1, demand_type=dtype,
                     common_reference=cr, lossaversion=1, gamma=1.0,
                     num_sessions=N_SESS, aprint=False,
                     ref_prediction="qlearning",
                     require_reference_stability=True,   # dual firm+reference convergence
                     track_q_stabilization=True,         # Q-value stabilization diagnostic
                     **extra)
        run_experiment_parallel_gamma_only(
            game, GAMMAS, num_sessions=N_SESS, experiment_name=name,
            demand_type=dtype, num_processes=N_PROC,
            use_reference_pretraining=True, T_ref=int(2e5),
            lambda_fixed=0.6, alpha=0.15, beta=4e-6,
            lossaversion_fixed=1.0, session_timeout=1800,
        )
    print("\nstage 2 complete")
