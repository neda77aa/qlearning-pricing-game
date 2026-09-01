"""
Paper-style figures for the linear-demand gamma sweep.

Reads the already-saved  gamma_summary_both_benchmarks.csv  (produced by
postprocess_linear_benchmarks.py) and redraws four curves in a clean
publication style:

    price          (level)  with naive Nash + coop benchmark curves
    profit         (level)  with naive Nash + coop benchmark curves
    price gain     (naive)  with Nash = 0 and coop = 1 reference lines
    profit gain    (naive)  with Nash = 0 and coop = 1 reference lines

Style: solid line, NO markers, light +/- std band, despined axes, y-only
light grid, serif fonts, 300 dpi.

Usage:
    python paper_figures_linear.py [experiment_name]
    # default = linear_benchmark/gamma_only_linear_full
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MAIN_DIR = "../Results/experiments"

# ---- publication-style rcParams -----------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.linewidth": 1.0,
    "figure.dpi": 120,
    "savefig.dpi": 300,
})

# one hue per entity (color follows the quantity, not the panel)
C_PROFIT = "#1f5fa8"   # blue
C_PRICE = "#20794d"    # green
C_NASH = "#b0392b"     # muted red  -> Nash
C_COOP = "#555555"     # dark grey  -> collusive (naive coop)


def _despine(ax, x):
    ax.set_xlabel(r"reference dependence  $\gamma$")
    ax.set_xlim(x.min(), x.max())
    ax.grid(axis="y", ls=":", color="0.8", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def level_plot(x, y, std, nash, coop, ylabel, title, color, out,
               show_benchmarks=True):
    """A level curve (price or profit), optionally with naive Nash + coop."""
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(x, y, color=color, lw=2.2, solid_capstyle="round",
            zorder=3, label="learned")
    if std is not None:
        ax.fill_between(x, y - std, y + std, color=color, alpha=0.16,
                        linewidth=0, zorder=1)

    if show_benchmarks:
        ax.plot(x, coop, color=C_COOP, lw=1.6, ls="--", zorder=2,
                label="collusive (naive coop)")
        ax.plot(x, nash, color=C_NASH, lw=1.6, ls="-.", zorder=2,
                label="Nash")
        ax.legend(frameon=False, loc="upper right")

    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=10)
    _despine(ax, x)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def gain_plot(x, y, std, ylabel, title, color, out, show_refs=True):
    """A normalized-gain curve, optionally with Nash = 0 / coop = 1 ref lines."""
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(x, y, color=color, lw=2.2, solid_capstyle="round", zorder=3)
    if std is not None:
        ax.fill_between(x, y - std, y + std, color=color, alpha=0.16,
                        linewidth=0, zorder=1)

    if show_refs:
        ax.axhline(0.0, color=C_NASH, ls="-.", lw=1.2, zorder=2)
        ax.axhline(1.0, color=C_COOP, ls="--", lw=1.2, zorder=2)
        ax.text(x.max(), 0.0, "Nash  ", color=C_NASH, va="bottom", ha="right",
                fontsize=10.5)
        ax.text(x.max(), 1.0, "collusive (naive coop)  ", color=C_COOP,
                va="bottom", ha="right", fontsize=10.5)

    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=10)
    _despine(ax, x)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def main():
    exp = sys.argv[1] if len(sys.argv) > 1 else "linear_benchmark/gamma_only_linear_full"
    exp_dir = os.path.join(MAIN_DIR, exp)
    csv = os.path.join(exp_dir, "gamma_summary_both_benchmarks.csv")
    df = pd.read_csv(csv).sort_values("gamma").reset_index(drop=True)
    x = df["gamma"].values

    out_dir = os.path.join(exp_dir, "Figures_paper")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[{exp}] {len(df)} gammas -> {out_dir}")

    # ---- levels: price, profit (with naive Nash + coop benchmarks) -------
    level_plot(x, df["mean_price"].values, df["std_price"].values,
               df["p_nash"].values, df["p_coop_naive"].values,
               "price  $p$", "Price vs. reference dependence",
               C_PRICE, os.path.join(out_dir, "price.png"))

    level_plot(x, df["mean_profit"].values, df["std_profit"].values,
               df["Pi_nash"].values, df["Pi_coop_naive"].values,
               "profit  $\\Pi$", "Profit vs. reference dependence",
               C_PROFIT, os.path.join(out_dir, "profit.png"))

    # ---- levels without benchmarks (learned curve only) ------------------
    level_plot(x, df["mean_price"].values, df["std_price"].values,
               None, None, "price  $p$", "Price vs. reference dependence",
               C_PRICE, os.path.join(out_dir, "price_nobench.png"),
               show_benchmarks=False)

    level_plot(x, df["mean_profit"].values, df["std_profit"].values,
               None, None, "profit  $\\Pi$", "Profit vs. reference dependence",
               C_PROFIT, os.path.join(out_dir, "profit_nobench.png"),
               show_benchmarks=False)

    # ---- gains: price gain (no ref lines), profit gain (with ref lines) --
    gain_plot(x, df["price_gain_naive"].values, df["std_price_gain_naive"].values,
              "price gain", "Price gain vs. reference dependence",
              C_PRICE, os.path.join(out_dir, "price_gain.png"), show_refs=False)

    gain_plot(x, df["profit_gain_naive"].values, df["std_profit_gain_naive"].values,
              "profit gain  $\\Delta$", "Profit gain vs. reference dependence",
              C_PROFIT, os.path.join(out_dir, "profit_gain.png"))


if __name__ == "__main__":
    main()
