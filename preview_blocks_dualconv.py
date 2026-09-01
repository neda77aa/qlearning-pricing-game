"""Old-vs-new comparison figures for each paper block that we re-ran under the
NEW dual firm+reference convergence rule.

For each block we emit ONE image: top row = the paper's previous 4 panels
(Price / Profit / Price Gain / Profit Gain), bottom row = the newly generated
4 panels under the dual-convergence rule. Curves, colors and linestyles match
rebuild_paper_figures.py; bands are mean +/- 0.8*sigma (paper convention).

Blocks: market_structure (Fig 5), misspecification, Firm-specific.
Runs still in progress are drawn over whatever gammas exist so far.

Run:  /Users/neda/llm_venv/bin/python preview_blocks_dualconv.py
Out:  compare_<block>_old_vs_dualconv.png (repo root)
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
    "ref_T":  (os.path.join(RES, "gamma_nloss_reference_True_qref_beta4e-6"),
               os.path.join(DC, "baseline_common_reference")),   # baseline (not rerun; already had)
    "c0":     (os.path.join(RES, "gamma_nloss_reference_Truec_0_qref_beta4e-6"),
               os.path.join(DC, "market_structure_cost_0")),
    "mu0":    (os.path.join(RES, "gamma_nloss_reference_Truemu_0_qref_beta4e-6"),
               os.path.join(DC, "market_structure_mu_0.05")),
    "miss_T": (os.path.join(RES, "gamma_nloss_misspecification_True_qref_beta4e-6"),
               os.path.join(DC, "misspecification_reference_naive")),
    "ref_F":  (os.path.join(RES, "gamma_nloss_reference_False_qref_beta4e-6"),
               os.path.join(DC, "firm_specific_reference_CRfalse")),
}

# block -> list of curves: (key, color, linestyle, lw, band_alpha, label)
BLOCKS = {
    "market_structure": [
        ("ref_T", "tab:purple", "-",         3.0, 0.18, "baseline (c = 1, μ = 0.25)"),
        ("c0",    "tab:purple", (0, (6, 3)), 2.2, 0.12, "(c ↓, μ = 0.25)"),
        ("mu0",   "tab:purple", (0, (1, 2)), 2.2, 0.12, "(c = 1, μ ↓)")],
    "misspecification": [
        ("ref_T",  "tab:purple", "-", 2.2, 0.15, "Reference-Aware Firms"),
        ("miss_T", "tab:red",    "-", 2.2, 0.15, "Reference-Naive Firms")],
    "Firm-specific": [
        ("ref_T", "tab:purple", "-",  2.2, 0.15, "Reference-Aware (CR=True)"),
        ("ref_F", "tab:purple", "--", 2.2, 0.15, "Reference-Aware (CR=False)")],
}

METRICS = [("price", "Price"), ("profit", "Profit"),
           ("price_gain", "Price Gain"), ("profit_gain", "Profit Gain")]


def load(exp_dir, metric):
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


def build(block, curves):
    plt.rcParams.update({"font.size": 12, "axes.labelsize": 12,
                         "xtick.labelsize": 10, "ytick.labelsize": 10})
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    coverage = {}
    for row, col_idx in [(0, 0), (1, 1)]:      # row0 = OLD, row1 = NEW
        rowlab = "Paper (firm-only convergence)" if row == 0 \
            else "New rule (firm + reference convergence)"
        for j, (metric, ylabel) in enumerate(METRICS):
            ax = axes[row, j]
            for key, color, ls, lw, ba, label in curves:
                exp = FOLDERS[key][col_idx]
                g, mu, sd = load(exp, metric)
                if not len(g):
                    continue
                if row == 1 and j == 0:
                    coverage[key] = len(g)
                ax.plot(g, mu, color=color, linestyle=ls, lw=lw, label=label)
                ax.fill_between(g, mu - 0.8 * sd, mu + 0.8 * sd,
                                color=color, alpha=ba, lw=0)
            ax.set_xlabel(r"$\gamma$")
            ax.set_ylabel(f"{rowlab}\n\n{ylabel}" if j == 0 else ylabel)
            ax.grid(True, ls="--", alpha=0.4)
            if row == 0:
                ax.set_title(ylabel, fontsize=13)
            if row == 0 and j == 0:
                ax.legend(fontsize=9, loc="best")
    partial = [f"{k} ({n}/30)" for k, n in coverage.items() if n < 30]
    sub = "  [new still running: " + ", ".join(partial) + "]" if partial else ""
    fig.suptitle(f"{block}: paper vs new dual-convergence rule "
                 f"(Q-learning reference, β=4e-6){sub}", fontsize=15, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = f"compare_{block}_old_vs_dualconv.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.abspath(out), "coverage(new):", coverage)


if __name__ == "__main__":
    for block, curves in BLOCKS.items():
        build(block, curves)
