"""
Reference-Q-learning gamma sweep with a dual (firm + reference) convergence
rule and a Q-value stabilization diagnostic.

Requested run:
  * 30 gamma values, 200 sessions each
  * beta = 4e-6 (the impulse-response exploration-decay rate)
  * demand_type='reference', common_reference=True, ref_prediction='qlearning'
    (consumer reference is learned by a Q-agent, pretrained then kept learning)
  * convergence requires BOTH the firms' Q-argmax policy AND the reference-price
    index to be stable for the usual tstable window
    (game.require_reference_stability=True)
  * every 1000 steps record the mean absolute per-cell change of the firms' and
    the consumer reference agent's Q-tables, saved to q_stabilization.npz per
    gamma (game.track_q_stabilization=True)

Run from the Algorithmic-Collusion-Replication directory:
    python run_gamma_ref_qlearn_diag.py smoke   # quick validation
    python run_gamma_ref_qlearn_diag.py full     # the real 30x200 run
"""
import sys
import numpy as np
from multiprocessing import freeze_support

from input.init import model
from input.ConvResults_gamma_lambda import run_experiment_parallel_gamma_only


def build_and_run(mode):
    if mode == "smoke":
        gamma_values = np.array([0.0, 1.0])
        num_sessions = 2
        tmax = int(4e5)
        tstable = int(2e3)
        q_stab_interval = 500
        T_ref = int(2e4)
        beta = 1e-4            # faster decay so the smoke test converges quickly
        num_processes = 2
        experiment_name = "smoke/gamma_ref_qlearn_diag"
    elif mode in ("full", "batch2"):
        gamma_values = np.linspace(0, 3, 30)   # 30 gamma values
        num_sessions = 50
        tmax = int(1e7)
        tstable = int(1e5)
        q_stab_interval = 1000
        T_ref = int(2e5)
        beta = 4e-6            # impulse-response exploration-decay rate
        num_processes = 10
        # batch2 = a second independent 50-session block (different folder) that
        # is pooled with the first for the 100-session/gamma main figure. Seeds
        # are drawn per-session from OS entropy, so the two blocks are i.i.d.
        experiment_name = ("gamma_ref_qlearn_beta4e-6_dualconv"
                           if mode == "full"
                           else "gamma_ref_qlearn_beta4e-6_dualconv_batch2")
    else:
        raise SystemExit("usage: run_gamma_ref_qlearn_diag.py [smoke|full|batch2]")

    game = model(
        n=2,
        k=15,
        memory=1,
        lossaversion=1,                     # no loss aversion (benchmark logit)
        num_sessions=num_sessions,
        aprint=False,
        demand_type='reference',
        common_reference=True,
        ref_prediction='qlearning',         # <-- reference learned by a Q-agent
        beta=beta,
        tmax=tmax,
        tstable=tstable,
        require_reference_stability=True,   # dual convergence rule
        track_q_stabilization=True,         # Q-value stabilization diagnostic
        q_stab_interval=q_stab_interval,
    )

    print(f"[{mode}] gammas={list(np.round(gamma_values, 4))}")
    print(f"[{mode}] num_sessions={num_sessions}, beta={beta:g}, "
          f"tmax={tmax:g}, tstable={tstable:g}, T_ref={T_ref:g}, "
          f"procs={num_processes}")
    print(f"[{mode}] experiment_name={experiment_name}")

    run_experiment_parallel_gamma_only(
        game,
        gamma_values,
        num_sessions=num_sessions,
        experiment_name=experiment_name,
        demand_type='reference',
        num_processes=num_processes,
        use_reference_pretraining=True,     # pretrain the consumer reference Q
        T_ref=T_ref,
        alpha=0.15,
        beta=beta,                          # OVERRIDES game.beta inside sessions
    )
    print(f"[{mode}] done.")


if __name__ == "__main__":
    freeze_support()
    mode = sys.argv[1] if len(sys.argv) > 1 else "smoke"
    build_and_run(mode)
