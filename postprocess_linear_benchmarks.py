"""
Post-process a linear-demand gamma sweep to save BOTH price gain and profit gain
under BOTH collusive benchmarks:

  * naive    : p_coop = 1/(2+g),  Pi_coop = (1+g)/(2+g)^2
               (reference taken as GIVEN; this is what the pipeline already uses)
  * true     : p_mono = 1/2,      Pi_mono = 1/4
               (reference INTERNALIZED: at the symmetric steady state r = p, so the
                gamma*(p-r) term vanishes and demand collapses to D = 1 - p, whose
                monopoly price is the textbook 1/2, independent of gamma)

Nash / competitive benchmark is unchanged (n = 2 closed form):
    p_nash = 1/(3+2g),  Pi_nash = 2(1+g)/(3+2g)^2

Everything is derived from the already-saved per-gamma cycle_statistics.csv
(realized mean_profit_p*, mean_price_p*, and their stds), so the simulation is
NOT re-run and nothing in the pipeline is modified.

Usage:
    python postprocess_linear_benchmarks.py <experiment_name>
    # default experiment_name = linear_benchmark/gamma_only_linear_full
"""

import os
import re
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MAIN_DIR = "../Results/experiments"
N = 2  # firms


def closed_forms(g):
    """Return the benchmark dict for gamma g (n = 2)."""
    p_nash = 1.0 / (3.0 + 2.0 * g)
    Pi_nash = 2.0 * (1.0 + g) / (3.0 + 2.0 * g) ** 2
    p_coop_naive = 1.0 / (2.0 + g)
    Pi_coop_naive = (1.0 + g) / (2.0 + g) ** 2
    p_mono_true = 0.5
    Pi_mono_true = 0.25
    return dict(p_nash=p_nash, Pi_nash=Pi_nash,
                p_coop_naive=p_coop_naive, Pi_coop_naive=Pi_coop_naive,
                p_mono_true=p_mono_true, Pi_mono_true=Pi_mono_true)


def build_summary(exp_dir):
    rows = []
    for d in sorted(glob.glob(os.path.join(exp_dir, "gamma_*"))):
        m = re.search(r"gamma_([0-9.]+)", os.path.basename(d))
        f = os.path.join(d, "cycle_statistics.csv")
        if not m or not os.path.isfile(f):
            continue
        g = float(m.group(1))
        s = pd.read_csv(f).iloc[0]
        cf = closed_forms(g)

        # realized levels (average over the two symmetric firms)
        profit = np.mean([s["mean_profit_p1"], s["mean_profit_p2"]])
        std_profit = np.mean([s["std_profit_p1"], s["std_profit_p2"]])
        price = np.mean([s["mean_price_p1"], s["mean_price_p2"]])
        std_price = np.mean([s["std_price_p1"], s["std_price_p2"]])

        # gains under each benchmark (Nash is common to both)
        prof_den_naive = cf["Pi_coop_naive"] - cf["Pi_nash"]
        prof_den_true = cf["Pi_mono_true"] - cf["Pi_nash"]
        price_den_naive = cf["p_coop_naive"] - cf["p_nash"]
        price_den_true = cf["p_mono_true"] - cf["p_nash"]

        rows.append(dict(
            gamma=g,
            convergence_rate=s["convergence_rate"],
            mean_cycle_length=s["mean_cycle_length"],
            std_cycle_length=s["std_cycle_length"],
            # benchmarks
            p_nash=cf["p_nash"], p_coop_naive=cf["p_coop_naive"], p_mono_true=cf["p_mono_true"],
            Pi_nash=cf["Pi_nash"], Pi_coop_naive=cf["Pi_coop_naive"], Pi_mono_true=cf["Pi_mono_true"],
            # realized levels
            mean_price=price, std_price=std_price,
            mean_profit=profit, std_profit=std_profit,
            mean_reference_price=s["mean_reference_price"],
            # PROFIT GAIN
            profit_gain_naive=(profit - cf["Pi_nash"]) / prof_den_naive,
            std_profit_gain_naive=std_profit / prof_den_naive,
            profit_gain_true=(profit - cf["Pi_nash"]) / prof_den_true,
            std_profit_gain_true=std_profit / prof_den_true,
            # PRICE GAIN
            price_gain_naive=(price - cf["p_nash"]) / price_den_naive,
            std_price_gain_naive=std_price / price_den_naive,
            price_gain_true=(price - cf["p_nash"]) / price_den_true,
            std_price_gain_true=std_price / price_den_true,
        ))
    df = pd.DataFrame(rows).sort_values("gamma").reset_index(drop=True)
    return df


def line_plot(df, col, std_col, ylabel, title, color, out, floor=None):
    fig, ax = plt.subplots(figsize=(10, 8))
    x = df["gamma"].values
    y = df[col].values
    ax.plot(x, y, marker="o", color=color, label=ylabel)
    if std_col in df:
        lo = y - df[std_col].values
        hi = y + df[std_col].values
        if floor is not None:
            lo = np.maximum(lo, floor)
        ax.fill_between(x, lo, hi, color=color, alpha=0.2, label=f"{ylabel} ± std")
    ax.axhline(1.0, color="grey", ls="--", lw=1, alpha=0.7)
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, ls="--", alpha=0.6)
    ax.legend()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main():
    exp = sys.argv[1] if len(sys.argv) > 1 else "linear_benchmark/gamma_only_linear_full"
    exp_dir = os.path.join(MAIN_DIR, exp)
    df = build_summary(exp_dir)
    if df.empty:
        print(f"No gamma results found in {exp_dir}")
        return

    csv_out = os.path.join(exp_dir, "gamma_summary_both_benchmarks.csv")
    df.to_csv(csv_out, index=False)
    print(f"[{len(df)} gammas] saved {csv_out}")

    fig_dir = os.path.join(exp_dir, "Figures_both_benchmarks")
    os.makedirs(fig_dir, exist_ok=True)
    line_plot(df, "profit_gain_naive", "std_profit_gain_naive", "Profit Gain (naive coop)",
              "Profit Gain vs Gamma  —  naive benchmark p_coop=1/(2+γ)",
              "blue", os.path.join(fig_dir, "profit_gain_naive.png"))
    line_plot(df, "profit_gain_true", "std_profit_gain_true", "Profit Gain (true monopoly)",
              "Profit Gain vs Gamma  —  true monopoly p=1/2, Π=1/4",
              "navy", os.path.join(fig_dir, "profit_gain_true.png"))
    line_plot(df, "price_gain_naive", "std_price_gain_naive", "Price Gain (naive coop)",
              "Price Gain vs Gamma  —  naive benchmark p_coop=1/(2+γ)",
              "green", os.path.join(fig_dir, "price_gain_naive.png"))
    line_plot(df, "price_gain_true", "std_price_gain_true", "Price Gain (true monopoly)",
              "Price Gain vs Gamma  —  true monopoly p=1/2",
              "darkgreen", os.path.join(fig_dir, "price_gain_true.png"))
    print(f"figures saved in {fig_dir}")

    show = ["gamma", "profit_gain_naive", "profit_gain_true",
            "price_gain_naive", "price_gain_true", "mean_price", "convergence_rate"]
    print(df[show].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
