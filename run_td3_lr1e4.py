"""TD3 production sweep at lr=1e-4 (baseline sweep used 3e-4).

Motivated by the 4-seed check where lr=1e-4 raised collusion at gamma>=1
(pg +0.53/+0.53/+0.72 vs +0.37/+0.42/+0.52) -- the continuous-action analog
of Calvano's 'persistent learning' corner. Same 15 gammas x 20 sessions as
td3_production_reference_15g_20s, so results are directly comparable.

Run:  /Users/neda/llm_venv/bin/python run_td3_lr1e4.py
"""
from multiprocessing import freeze_support

import production_sweep as ps

if __name__ == "__main__":
    freeze_support()
    ps.EXPERIMENT = "td3_production_reference_15g_20s_lr1e-4"
    ps.SIM_KWARGS["lr"] = 1e-4
    ps.NUM_PROCESSES = 8
    print(f"TD3 sweep at lr=1e-4 -> {ps.EXPERIMENT}")
    ps.main()
