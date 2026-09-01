"""Preview of paper Figure 5 (market structure) under the NEW dual firm+reference
convergence rule, side-by-side with the ORIGINAL paper rule, for comparison.

Figure 5 overlays 3 curves vs gamma (each from gamma_*/cycle_statistics.csv,
firm-averaged, mean +/- 0.8*sigma band -- the paper's convention):
    solid   baseline (c=1, mu=0.25)
    dashed  (c down, mu=0.25)   -> c=0 variant
    dotted  (c=1, mu down)      -> mu=0.05 variant

Layout: 2 rows (Original rule / New dual-convergence rule) x 4 metric columns.
The new mu=0.05 run may still be in progress -> that curve is drawn over
whatever gammas are available so far.

Run:  /Users/neda/llm_venv/bin/python preview_fig5_dualconv.py
Out:  fig5_compare_old_vs_dualconv.png (repo root)
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = "../Results/experiments"                                    # OLD paper runs (in place)
DC = "/Users/neda/Desktop/UBC/PHD/research_term_4/Result_double_convergence"  # NEW dual-conv archive

# curve key -> (OLD folder full path, NEW/dual-conv folder full path)
FOLDERS = {
    "base": (os.path.join(RES, "gamma_nloss_reference_True_qref_beta4e-6"),
             os.path.join(DC, "baseline_common_reference")),  # baseline already had (100s/grid0-3)
    "c0":   (os.path.join(RES, "gamma_nloss_reference_Truec_0_qref_beta4e-6"),
             os.path.join(DC, "market_structure_cost_0")),
    "mu0":  (os.path.join(RES, "gamma_nloss_reference_Truemu_0_qref_beta4e-6"),
             os.path.join(DC, "market_structure_mu_0.05")),
}
# plot style per curve (matches rebuild_paper_figures.py market_structure block)
STYLE = [
    ("base", "-",          3.0, 0.18, "baseline (c = 1, μ = 0.25)"),
    ("c0",   (0, (6, 3)),  2.2, 0.12, "(c ↓, μ = 0.25)"),
    ("mu0",  (0, (1, 2)),  2.2, 0.12, "(c = 1, μ ↓)"),
]
COL = "tab:purple"
METRICS = [("price", "Price"), ("profit", "Profit"),
           ("price_gain", "Price Gain"), ("profit_gain", "Profit Gain")]


def load(exp_dir, metric):
    """gamma, firm-averaged mean, firm-averaged std from cycle_statistics.csv."""
    rows = []
    for d in sorted(glob.glob(os.path.join(exp_dir, "gamma_*"))):
        f = os.path.join(d, "cycle_statistics.csv")
        if not os.path.exists(f):
            continue
        r = pd.read_csv(f).iloc[0]
        g = float(os.path.basename(d).split("gamma_")[1])
        mu = np.mean([float(r[f"mean_{metric}_p1"]), float(r[f"mean_{metric}_p2"])])
        sd = np.mean([float(r[f"std_{metric}_p1"]), float(r[f"std_{metric}_p2"])])
        rows.append((g, mu, sd))
    rows.sort()
    a = np.array(rows)
    return (a[:, 0], a[:, 1], a[:, 2]) if len(a) else (np.array([]),)*3


def main():
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 12,
                         "xtick.labelsize": 10, "ytick.labelsize": 10})
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for row, col_idx in [(0, 0), (1, 1)]:   # row 0 = OLD, row 1 = NEW
        rule = "Original rule (firm-only convergence)" if row == 0 \
            else "New rule (firm + reference convergence)"
        for j, (metric, ylabel) in enumerate(METRICS):
            ax = axes[row, j]
            for key, ls, lw, ba, label in STYLE:
                exp = FOLDERS[key][col_idx]
                g, mu, sd = load(exp, metric)
                if not len(g):
                    continue
                ax.plot(g, mu, color=COL, linestyle=ls, lw=lw, label=label)
                ax.fill_between(g, mu - 0.8 * sd, mu + 0.8 * sd,
                                color=COL, alpha=ba, lw=0)
            ax.set_xlabel(r"$\gamma$")
            if j == 0:
                ax.set_ylabel(f"{rule}\n\n{ylabel}")
            else:
                ax.set_ylabel(ylabel)
            ax.grid(True, ls="--", alpha=0.4)
            if row == 0 and j == 0:
                ax.legend(fontsize=9, loc="best")
            if row == 0:
                ax.set_title(ylabel, fontsize=13)
    fig.suptitle("Figure 5 (market structure): original vs new dual-convergence "
                 "rule  —  Q-learning reference, β=4e-6", fontsize=15, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = "fig5_compare_old_vs_dualconv.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print("wrote", os.path.abspath(out))

    # per-curve gamma coverage note
    print("\ngamma coverage (new/_dualconv folders):")
    for key in ("base", "c0", "mu0"):
        g, *_ = load(FOLDERS[key][1], "price")
        print(f"  {key:5s}: {len(g)} gammas"
              + ("" if len(g) >= 30 else "  <-- still running"))


if __name__ == "__main__":
    main()
