"""Regenerate the linear and TD3 robustness figures in the paper's benchmark
PURPLE style (tab:purple, band = 0.8*std, alpha 0.15), matching build_block in
rebuild_paper_figures.py, so the robustness sections use the same "our baseline
reference-aware result" color as the main analysis.

Overwrites the exact PNG filenames the paper already \includegraphics:
  linear -> Images/4_seperate_figures_beta4e6/linear/figure1_gamma_only_q_*.png
  td3    -> Images/4_seperate_figures_lr1e-4/td3/figure1_gamma_only_q_*.png

Reads gamma_*/cycle_statistics.csv only (no re-simulation).

Run:  /Users/neda/llm_venv/bin/python recolor_linear_td3_purple.py
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/Users/neda/Desktop/UBC/PHD/research_term_4"
RES  = os.path.join(ROOT, "Results", "experiments")
IMG  = os.path.join(ROOT, "Algorithmic-Collusion-Replication",
                    "Final_Paper__Reference_Dependence__Copy2_", "Images")

# match build_block exactly
RC = {"font.size": 18, "axes.labelsize": 18, "xtick.labelsize": 16,
      "ytick.labelsize": 16, "legend.fontsize": 14}
COLOR, LW, BAND_MULT, BAND_ALPHA = "tab:purple", 2.0, 0.8, 0.15

METRICS = [("price", "Price"), ("profit", "Profit"),
           ("price_gain", "Price Gain"), ("profit_gain", "Profit Gain")]

# figure block -> (source experiment dir, output image dir)
BLOCKS = {
    "linear": (
        os.path.join(RES, "linear_benchmark", "gamma_only_linear_qref_beta4e-6"),
        os.path.join(IMG, "4_seperate_figures_beta4e6", "linear")),
    "td3": (
        os.path.join(RES, "td3_production_reference_15g_50s_lr1e-4"),
        os.path.join(IMG, "4_seperate_figures_lr1e-4", "td3")),
}


def load_metric(exp_dir, metric):
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


if __name__ == "__main__":
    with plt.rc_context(RC):
        for block, (exp_dir, out_dir) in BLOCKS.items():
            if not os.path.isdir(exp_dir):
                print(f"!! missing source for {block}: {exp_dir}")
                continue
            os.makedirs(out_dir, exist_ok=True)
            n_g = len(glob.glob(os.path.join(exp_dir, "gamma_*")))
            print(f"{block}: {n_g} gammas  <-  {exp_dir}")
            for metric, ylabel in METRICS:
                g, mu, sd = load_metric(exp_dir, metric)
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(g, mu, color=COLOR, lw=LW)
                ax.fill_between(g, mu - BAND_MULT * sd, mu + BAND_MULT * sd,
                                color=COLOR, alpha=BAND_ALPHA, lw=0)
                ax.set_xlabel(r"$\gamma$")
                ax.set_ylabel(ylabel)
                ax.grid(True, ls="--", alpha=0.4)
                out = os.path.join(out_dir, f"figure1_gamma_only_q_{metric}.png")
                fig.savefig(out, dpi=300, bbox_inches="tight")
                plt.close(fig)
                print(f"    wrote {out}")
    print("done.")
