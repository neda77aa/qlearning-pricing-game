"""TD3 production sweep at lr=1e-4 with 50 sessions/gamma.

Same validated config and 15 gammas as td3_production_reference_15g_20s_lr1e-4,
but raises sessions 20 -> 50 so the sample size matches the tabular/linear
paper runs (50 sessions/gamma). lr=1e-4 is the continuous-action analog of the
corrected (more-exploration) tabular config.

Run:  /Users/neda/llm_venv/bin/python run_td3_lr1e4_50s.py
"""
from multiprocessing import freeze_support

import production_sweep as ps

if __name__ == "__main__":
    freeze_support()
    ps.EXPERIMENT = "td3_production_reference_15g_50s_lr1e-4"
    ps.NUM_SESSIONS = 50
    ps.SIM_KWARGS["lr"] = 1e-4
    ps.NUM_PROCESSES = 10
    print(f"TD3 sweep at lr=1e-4, 50 sessions -> {ps.EXPERIMENT}")
    ps.main()
