"""
Paper-style 4-panel figures for the robustness section (linear demand + deep RL).

Reproduces EXACTLY the styling of the paper's main gamma panels
(creating_results.ipynb cell that saves ``figure1_gamma_only_q_*.png``):

    figsize (6, 4), serif default, font 18 / ticks 16 / legend 14,
    solid line lw 2 (tab:blue), fill_between CI band alpha 0.15,
    grid dashed alpha 0.4, xlabel $\\gamma$, ylabel = metric, dpi 300, tight bbox.

Two producers:
  * linear : reads gamma_summary_both_benchmarks.csv (from
             postprocess_linear_benchmarks.py) and uses the TRUE monopoly
             benchmark for the gain panels (price_gain_true / profit_gain_true).
  * td3    : aggregates per-gamma cycle_statistics.csv (deep-RL / TD3 sweep),
             averaging the two firms; gains use the paper's coop benchmark
             (same convention as the paper's logit main figures).

Bands are mean +/- 1.96 * SE (SE = std / sqrt(n_sessions)) -> 95% CI, matching
the paper captions.

Usage:
    python paper_panels.py linear <experiment_name> <out_dir>
    python paper_panels.py td3    <experiment_name> <out_dir>
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
Z = 1.96  # 95% CI multiplier
CURVE = "tab:blue"  # paper "reference-aware" hue

plt.rcParams.update({
    "font.size": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 14,
})

PANELS = [
    ("Price",       "figure1_gamma_only_q_price.png"),
    ("Profit",      "figure1_gamma_only_q_profit.png"),
    ("Price Gain",  "figure1_gamma_only_q_price_gain.png"),
    ("Profit Gain", "figure1_gamma_only_q_profit_gain.png"),
]


def _panel(g, mu, band, ylabel, out):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(g, mu, color=CURVE, linestyle="-", marker=" ", linewidth=2)
    if band is not None:
        ax.fill_between(g, mu - band, mu + band, color=CURVE, alpha=0.15)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------- linear -----
def make_linear(exp, out_dir):
    csv = os.path.join(MAIN_DIR, exp, "gamma_summary_both_benchmarks.csv")
    df = pd.read_csv(csv).sort_values("gamma").reset_index(drop=True)
    g = df["gamma"].values
    # n sessions: infer from a per-gamma cycle_statistics.csv if present, else 200
    n = _infer_n_sessions(os.path.join(MAIN_DIR, exp), default=200)
    se = lambda s: s / np.sqrt(n)

    series = [
        ("Price",       df["mean_price"].values,        se(df["std_price"].values)),
        ("Profit",      df["mean_profit"].values,       se(df["std_profit"].values)),
        # TRUE monopoly benchmark for the gains (user choice)
        ("Price Gain",  df["price_gain_true"].values,   se(df["std_price_gain_true"].values)),
        ("Profit Gain", df["profit_gain_true"].values,  se(df["std_profit_gain_true"].values)),
    ]
    os.makedirs(out_dir, exist_ok=True)
    print(f"[linear {exp}] {len(g)} gammas, n={n} -> {out_dir}")
    for (ylabel, mu, sb), (_, fname) in zip(series, PANELS):
        _panel(g, mu, Z * sb, ylabel, os.path.join(out_dir, fname))


def _infer_n_sessions(exp_dir, default):
    for cs in glob.glob(os.path.join(exp_dir, "gamma_*", "cycle_statistics.csv")):
        try:
            return int(pd.read_csv(cs).iloc[0]["num_sessions"])
        except Exception:
            pass
    return default


# ------------------------------------------------------------------ td3 ------
def _gamma_from_dir(d):
    m = re.search(r"gamma_([0-9.]+)", os.path.basename(d))
    return float(m.group(1)) if m else np.nan


def make_td3(exp, out_dir):
    exp_dir = os.path.join(MAIN_DIR, exp)
    rows = []
    for d in glob.glob(os.path.join(exp_dir, "gamma_*")):
        cs = os.path.join(d, "cycle_statistics.csv")
        if not os.path.isfile(cs):
            continue
        r = pd.read_csv(cs).iloc[0]
        g = _gamma_from_dir(d)
        n = float(r["num_sessions"])
        # average the two firms; combine per-firm std as sqrt(mean of variances)
        def avg(a, b):
            return 0.5 * (float(r[a]) + float(r[b]))

        def std2(a, b):
            return np.sqrt(0.5 * (float(r[a]) ** 2 + float(r[b]) ** 2))

        rows.append(dict(
            gamma=g, n=n,
            price=avg("mean_price_p1", "mean_price_p2"),
            price_sd=std2("std_price_p1", "std_price_p2"),
            profit=avg("mean_profit_p1", "mean_profit_p2"),
            profit_sd=std2("std_profit_p1", "std_profit_p2"),
            pgain=avg("mean_price_gain_p1", "mean_price_gain_p2"),
            pgain_sd=std2("std_price_gain_p1", "std_price_gain_p2"),
            prgain=avg("mean_profit_gain_p1", "mean_profit_gain_p2"),
            prgain_sd=std2("std_profit_gain_p1", "std_profit_gain_p2"),
        ))
    df = pd.DataFrame(rows).sort_values("gamma").reset_index(drop=True)
    g = df["gamma"].values
    se = lambda s: s / np.sqrt(df["n"].values)

    series = [
        ("Price",       df["price"].values,  se(df["price_sd"].values)),
        ("Profit",      df["profit"].values, se(df["profit_sd"].values)),
        ("Price Gain",  df["pgain"].values,  se(df["pgain_sd"].values)),
        ("Profit Gain", df["prgain"].values, se(df["prgain_sd"].values)),
    ]
    os.makedirs(out_dir, exist_ok=True)
    print(f"[td3 {exp}] {len(g)} gammas, n={int(df['n'].iloc[0])} -> {out_dir}")
    for (ylabel, mu, sb), (_, fname) in zip(series, PANELS):
        _panel(g, mu, Z * sb, ylabel, os.path.join(out_dir, fname))


if __name__ == "__main__":
    mode = sys.argv[1]
    exp = sys.argv[2]
    out_dir = sys.argv[3]
    if mode == "linear":
        make_linear(exp, out_dir)
    elif mode == "td3":
        make_td3(exp, out_dir)
    else:
        raise SystemExit(f"unknown mode {mode!r} (use 'linear' or 'td3')")
