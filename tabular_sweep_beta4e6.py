"""Tabular benchmark re-run with corrected configuration.

Changes vs the published gamma_nloss_only_reference_True benchmark:
  beta = 4e-6                (paper runs: hard-coded 4e-5 -> 10x less exploration)
  lossaversion = 1.0         (paper runners silently passed 1.5 into sessions)
  continuous_reference=True  (ES-path reference: smoothed as continuous float,
                              indexed only when reading PI / forming the state)

Same grid as the paper: 30 gammas in [0.05, 3.0], 50 sessions each.
Estimated runtime: ~6 h at 8 processes (sessions converge ~1.7M steps).

Run:  /Users/neda/llm_venv/bin/python tabular_sweep_beta4e6.py
"""
import numpy as np
from multiprocessing import freeze_support

from input.init import model
from input.ConvResults_gamma_lambda import run_experiment_parallel_gamma_only

EXPERIMENT = "gamma_nloss_reference_True_beta4e-6_ESref"
GAMMAS = [round(float(g), 4) for g in np.linspace(0.05, 3.0, 30)]
NUM_SESSIONS = 50
NUM_PROCESSES = 8

if __name__ == "__main__":
    freeze_support()
    game = model(n=2, k=15, memory=1, demand_type="reference",
                 common_reference=True, lossaversion=1, gamma=1.0,
                 num_sessions=NUM_SESSIONS, aprint=False,
                 continuous_reference=True)
    run_experiment_parallel_gamma_only(
        game, GAMMAS,
        num_sessions=NUM_SESSIONS,
        experiment_name=EXPERIMENT,
        demand_type="reference",
        num_processes=NUM_PROCESSES,
        lambda_fixed=0.6,
        alpha=0.15,
        beta=4e-6,
        lossaversion_fixed=1.0,
    )
    print("tabular sweep complete ->", f"../Results/experiments/{EXPERIMENT}")
