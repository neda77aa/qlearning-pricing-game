"""Impulse responses / cheating profitability for the corrected tabular sweep
(gamma_nloss_reference_True_beta4e-6_ESref: beta=4e-6, lossaversion=1,
continuous ES reference, 50 sessions per gamma).

For each gamma and both deviation depths (static BR, Nash price):
  - Calvano-Fig4-style figure -> <exp>/Figures/irf_gamma_<g>_dev-<dt>.png
  - pooled % of deviations unprofitable + mean %gain in discounted profits
  - per-session breakdown (sessions where cheating pays)

Old-benchmark comparison (beta=4e-5, legacy ref): BR ~41-47%, Nash ~50-55%
unprofitable.

Run:  /Users/neda/llm_venv/bin/python irf_new_sweep.py
"""
import os
import numpy as np
import pandas as pd

import impulse_response as ir

EXP = "../Results/experiments/gamma_nloss_reference_True_beta4e-6_ESref"
GAMMAS = ["0.05", "1.0672", "2.0845", "3.0"]

if __name__ == "__main__":
    fig_dir = os.path.join(EXP, "Figures")
    os.makedirs(fig_dir, exist_ok=True)
    rows = []
    for g in GAMMAS:
        print(f"gamma={g} ...")
        res_by, sess_by = ir.irf_tabular_es(EXP, g)
        for dt in ir.DEV_TARGETS:
            res, sg = res_by[dt], sess_by[dt]
            out = os.path.join(fig_dir, f"irf_gamma_{g}_dev-{dt}.png")
            lbl = "static BR" if dt == "br" else "Nash price"
            ir.plot_irf(res, f"{g}, beta=4e-6 ES, deviation: {lbl}",
                        "Tabular Q (corrected config)", out)
            np.savez(os.path.join(fig_dir, f"irf_gamma_{g}_dev-{dt}.npz"),
                     session_pct_gain=sg,
                     **{k: np.asarray(v) for k, v in res.items()})
            rows.append(dict(
                gamma=float(g), dev=dt, n_obs=res["n_obs"],
                frac_unprofitable=res["frac_unprofitable"],
                mean_pct_gain=float(sg.mean()),
                sessions_cheating_loses=int((sg < 0).sum()),
                n_sessions=len(sg)))
            print(f"  dev={dt}: {res['frac_unprofitable']*100:.1f}% of "
                  f"deviations unprofitable | session mean gain "
                  f"{sg.mean():+.2f}% | cheating loses in "
                  f"{(sg < 0).sum()}/{len(sg)} sessions")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(fig_dir, "irf_summary.csv"), index=False)
    print("\n" + df.to_string(index=False))
    print(f"\nfigures + summary -> {fig_dir}")
