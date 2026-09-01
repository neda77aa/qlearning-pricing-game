"""Stage 3: lossaversion sweep at beta=4e-6 (paper block 'lossaversion').

Mirror of RES/loss_extreme/lossaversion_reverse: phi = linspace(1,3,20),
gamma=1, lambda=0.5, demand='reference', common_reference=True,
ref_prediction='exponentially_smoothing' (continuous ES reference), alpha=0.15,
50 sessions. Only beta changes: 4e-5 -> 4e-6. Loss aversion varies BY DESIGN
here (this is the dedicated loss-aversion experiment); all values >= 1 are
passed explicitly per lossaversion point by the runner.

Run:  /Users/neda/llm_venv/bin/python paper_rerun_lossaversion.py
"""
import numpy as np
from multiprocessing import freeze_support

from input.init import model
from input.ConvResults_gamma_lambda import run_experiment_parallel_lossaversion

EXPERIMENT = "lossaversion_reverse_beta4e-6"
PHI = [round(float(x), 4) for x in np.linspace(1.0, 3.0, 20)]
N_SESS, N_PROC = 50, 8

if __name__ == "__main__":
    freeze_support()
    game = model(n=2, k=15, memory=1, demand_type="reference",
                 common_reference=True, lossaversion=1, gamma=1.0,
                 num_sessions=N_SESS, aprint=False,
                 ref_prediction="exponentially_smoothing",
                 continuous_reference=True)
    run_experiment_parallel_lossaversion(
        game, PHI, num_sessions=N_SESS, experiment_name=EXPERIMENT,
        demand_type="reference", num_processes=N_PROC,
        alpha=0.15, beta=4e-6, gamma_fixed=1, lambda_fixed=0.5,
        session_timeout=1800,
    )
    print("lossaversion sweep complete ->", f"../Results/experiments/{EXPERIMENT}")
