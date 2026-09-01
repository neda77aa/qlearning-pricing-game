"""Rebuild the paper's figure blocks from the beta=4e-6 reruns, replicating
the original notebook styling (creating_results.ipynb cell 77/84) exactly:
same filenames, fonts, colors, linestyles, and the ORIGINAL band conventions
(gamma panels: mean +/- 0.8*sigma, alpha .15; lossaversion: +/- 0.5*sigma).

Output: Final_Paper__Reference_Dependence__Copy2_/Images/4_seperate_figures_beta4e6/<block>/

Blocks rebuilt here: benchmark, market_structure, misspecification,
Firm-specific, lossaversion. (exp_smooth needs the gamma-lambda grid; linear
and td3 have their own generators -- deferred.)

Run:  /Users/neda/llm_venv/bin/python rebuild_paper_figures.py
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = "../Results/experiments"
OUT = ("/Users/neda/Desktop/UBC/PHD/research_term_4/Algorithmic-Collusion-"
       "Replication/Final_Paper__Reference_Dependence__Copy2_/Images/"
       "4_seperate_figures_beta4e6")

METRICS = [("price", "Price"), ("profit", "Profit"),
           ("price_gain", "Price Gain"), ("profit_gain", "Profit Gain")]

E = {  # new experiment dirs
    "ref_T":  f"{RES}/gamma_nloss_reference_True_qref_beta4e-6",
    "c0":     f"{RES}/gamma_nloss_reference_Truec_0_qref_beta4e-6",
    "mu0":    f"{RES}/gamma_nloss_reference_Truemu_0_qref_beta4e-6",
    "miss_T": f"{RES}/gamma_nloss_misspecification_True_qref_beta4e-6",
    "ref_F":  f"{RES}/gamma_nloss_reference_False_qref_beta4e-6",
    "loss":   f"{RES}/lossaversion_reverse_beta4e-6",
}

# block -> list of curves: (exp, color, ls, lw, band_mult, band_alpha, label)
BLOCKS = {
    "benchmark": [
        ("ref_T", "tab:purple", "-", 2.0, 0.8, 0.15, "QLearning Refrence Pred")],
    "market_structure": [
        ("ref_T", "tab:purple", "-", 3.0, 0.8, 0.18, "baseline(c = 1, μ = 0.25)"),
        ("c0", "tab:purple", (0, (6, 3)), 2.2, 0.8, 0.12, "(c ↓, μ = 0.25)"),
        ("mu0", "tab:purple", (0, (1, 2)), 2.2, 0.8, 0.12, "(c = 1, μ ↓)")],
    "misspecification": [
        ("ref_T", "tab:purple", "-", 2.0, 0.8, 0.15, "Reference-Aware Firms"),
        ("miss_T", "tab:red", "-", 2.0, 0.8, 0.15, "Reference-Naive Firms")],
    "Firm-specific": [
        ("ref_T", "tab:purple", "-", 2.0, 0.8, 0.15, "Reference-Aware Firms (CR=True)"),
        ("ref_F", "tab:purple", "--", 2.0, 0.8, 0.15, "Reference-Aware Firms (CR=False)")],
}

RC = {"font.size": 18, "axes.labelsize": 18, "xtick.labelsize": 16,
      "ytick.labelsize": 16, "legend.fontsize": 14}


def load_gamma_only_metric(exp_dir, metric):
    """gamma, mean, std -- firm-averaged from each gamma_*/cycle_statistics.csv."""
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
    return a[:, 0], a[:, 1], a[:, 2]


def build_block(name, curves):
    out_dir = os.path.join(OUT, name)
    os.makedirs(out_dir, exist_ok=True)
    with plt.rc_context(RC):
        for metric, ylabel in METRICS:
            fig, ax = plt.subplots(figsize=(6, 4))
            for exp, color, ls, lw, bm, ba, label in curves:
                g, mu, sd = load_gamma_only_metric(E[exp], metric)
                ax.plot(g, mu, color=color, linestyle=ls, lw=lw, label=label)
                ax.fill_between(g, mu - bm * sd, mu + bm * sd,
                                color=color, alpha=ba, lw=0)
            ax.set_xlabel(r"$\gamma$")
            ax.set_ylabel(ylabel)
            ax.grid(True, ls="--", alpha=0.4)
            fig.savefig(os.path.join(out_dir, f"figure1_gamma_only_q_{metric}.png"),
                        dpi=300, bbox_inches="tight")
            plt.close(fig)
        if len(curves) > 1:   # separate legend PNG (blocks with >1 curve)
            fig = plt.figure(figsize=(8, 0.7))
            handles = [plt.Line2D([], [], color=c, linestyle=ls, lw=lw, label=lab)
                       for _, c, ls, lw, _, _, lab in curves]
            fig.legend(handles=handles, loc="center", ncol=2, frameon=False)
            fig.savefig(os.path.join(out_dir, "legend_reference_naive.png"),
                        dpi=300, bbox_inches="tight")
            plt.close(fig)
    print(f"  {name}: done")


def build_lossaversion():
    out_dir = os.path.join(OUT, "lossaversion")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for d in sorted(glob.glob(os.path.join(E["loss"], "lossaversion_*"))):
        f = os.path.join(d, "cycle_statistics.csv")
        if not os.path.exists(f):
            continue
        r = pd.read_csv(f).iloc[0]
        phi = float(os.path.basename(d).split("lossaversion_")[1])
        for metric in ("price", "profit"):
            mu = np.mean([float(r[f"mean_{metric}_p1"]), float(r[f"mean_{metric}_p2"])])
            sd = np.mean([float(r[f"std_{metric}_p1"]), float(r[f"std_{metric}_p2"])])
            rows.append((phi, metric, mu, sd))
    df = pd.DataFrame(rows, columns=["phi", "metric", "mu", "sd"]).sort_values("phi")
    with plt.rc_context({**RC, "font.family": "serif",
                         "mathtext.fontset": "dejavuserif"}):
        for metric, ylabel in (("price", "Price"), ("profit", "Profit")):
            sub = df[df.metric == metric]
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.plot(sub.phi, sub.mu, color="tab:purple", ls="-", lw=2)
            ax.fill_between(sub.phi, sub.mu - 0.5 * sub.sd, sub.mu + 0.5 * sub.sd,
                            color="tab:purple", alpha=0.2, lw=0)
            ax.set_xlabel(r"Loss Aversion ($\phi$)")
            ax.set_ylabel(ylabel)
            ax.grid(True, ls="--", alpha=0.4)
            fig.savefig(os.path.join(out_dir, f"lossaversion_{metric}.png"),
                        dpi=300, bbox_inches="tight")
            plt.close(fig)
    print("  lossaversion: done")


if __name__ == "__main__":
    print(f"rebuilding into {OUT}")
    for name, curves in BLOCKS.items():
        missing = [e for e, *_ in curves
                   if not glob.glob(os.path.join(E[e], "gamma_*", "cycle_statistics.csv"))]
        if missing:
            print(f"  {name}: SKIPPED (data not ready: {missing})")
            continue
        build_block(name, curves)
    if glob.glob(os.path.join(E["loss"], "lossaversion_*", "cycle_statistics.csv")):
        build_lossaversion()
    else:
        print("  lossaversion: SKIPPED (data not ready)")
    print("done")
