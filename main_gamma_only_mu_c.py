"""
Gamma-only sweeps over c and μ + end-to-end visualization.

- Runs:
    ../Results/experiments/{experiment_base}/c_{c}/gamma_{γ}/...
    ../Results/experiments/{experiment_base}/mu_{μ}/gamma_{γ}/...

- Plots:
    2×2 panels of Price/Profit/PriceGain/ProfitGain vs γ
    Heatmaps of metric(γ, c) or metric(γ, μ)
"""

import os
import re
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import freeze_support

# --- your project imports ---------------------------------------------------
from input.init import model
from input.ConvResults_gamma_lambda import run_experiment_parallel_gamma_only


# =========================
# RUNNERS
# =========================

def run_gamma_only_c_sweep(
    c_values, gamma_values, base_game_kwargs, *,
    experiment_base="sweeps/gamma_only_c", num_sessions=200,
    demand_type="reference",
):
    """
    For each c in c_values, runs gamma_only across gamma_values.
    Results go to: ../Results/experiments/{experiment_base}/c_{c}/gamma_{γ}/...
    """
    for c in c_values:
        kwargs = dict(base_game_kwargs)
        kwargs["c"] = float(c)                      # vary c
        game = model(**kwargs)
        exp_name = f"{experiment_base}/c_{c:.2f}"
        run_experiment_parallel_gamma_only(
            game,
            gamma_values,
            num_sessions=num_sessions,
            experiment_name=exp_name,
            demand_type=demand_type,
        )


def run_gamma_only_mu_sweep(
    mu_values, gamma_values, base_game_kwargs, *,
    experiment_base="sweeps/gamma_only_mu", num_sessions=200,
    demand_type="reference",
):
    """
    For each μ in mu_values, runs gamma_only across gamma_values.
    Results go to: ../Results/experiments/{experiment_base}/mu_{μ}/gamma_{γ}/...
    """
    for mu in mu_values:
        kwargs = dict(base_game_kwargs)
        kwargs["mu"] = float(mu)                    # vary μ
        game = model(**kwargs)
        exp_name = f"{experiment_base}/mu_{mu:.2f}"
        run_experiment_parallel_gamma_only(
            game,
            gamma_values,
            num_sessions=num_sessions,
            experiment_name=exp_name,
            demand_type=demand_type,
        )


# =========================
# LOADERS
# =========================

_GAMMA_DIR_RE = re.compile(r"^gamma_([-+]?\d*\.?\d+)$")

def load_gamma_only_metric(
    experiment_dir: str,
    metric_name: str,
    *,
    fallback_gain_std: str = "relative",   # 'none' | 'relative'
    relative_gain_std_scale: float = 0.10, # 10% of (price/profit std / mean) as proxy
    drop_nan: bool = True,
) -> pd.DataFrame:
    """
    Load one metric across gamma_* runs inside `experiment_dir`.

    Returns a DataFrame sorted by gamma with columns:
      ['gamma', 'mean', 'std']

    - For metrics with per-player stds saved (Price/Profit), std is RMS across players.
    - For '*_gain' metrics (often without stds), you can set `fallback_gain_std`:
        * 'relative': uses relative proxy based on price/profit variability (default)
        * 'none'    : leaves std as NaN
    """
    key = metric_name.strip().lower().replace(" ", "_")  # 'Price Gain' -> 'price_gain'
    rows = []

    for run_dir in glob.glob(os.path.join(experiment_dir, "gamma_*")):
        base = os.path.basename(run_dir.rstrip(os.sep))
        m = _GAMMA_DIR_RE.match(base)
        if not m:
            continue
        gamma = float(m.group(1))

        stats_file = os.path.join(run_dir, "cycle_statistics.csv")
        if not os.path.exists(stats_file):
            continue

        df = pd.read_csv(stats_file)
        if df.empty:
            continue
        row = df.iloc[0]

        mean_cols = [c for c in row.index if c.lower().startswith(f"mean_{key}_p")]
        std_cols  = [c for c in row.index if c.lower().startswith(f"std_{key}_p")]

        if not mean_cols:
            # metric not present in this file
            continue

        mean_val = float(np.nanmean([row[c] for c in mean_cols]))

        if std_cols:
            std_val = float(np.sqrt(np.nanmean([row[c] ** 2 for c in std_cols])))
        else:
            std_val = np.nan
            if key.endswith("_gain") and fallback_gain_std != "none":
                base_metric = "price" if key.startswith("price") else "profit"
                base_mean_cols = [c for c in row.index if c.lower().startswith(f"mean_{base_metric}_p")]
                base_std_cols  = [c for c in row.index if c.lower().startswith(f"std_{base_metric}_p")]
                if base_mean_cols and base_std_cols:
                    base_mean = float(np.nanmean([row[c] for c in base_mean_cols]))
                    base_std  = float(np.sqrt(np.nanmean([row[c] ** 2 for c in base_std_cols])))
                    if np.isfinite(base_mean) and base_mean != 0:
                        std_val = relative_gain_std_scale * (base_std / abs(base_mean))

        rows.append(dict(gamma=gamma, mean=mean_val, std=std_val))

    if not rows:
        raise ValueError(f"No gamma-only data in {experiment_dir} for '{metric_name}'")

    out = (
        pd.DataFrame(rows)
        .sort_values("gamma")
        .drop_duplicates(subset=["gamma"])
        .reset_index(drop=True)
    )
    if drop_nan:
        out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["mean"])
    return out


# --------- optional: read Nash/Coop baselines from any gamma/config.csv ----

def _read_benchmarks_from_any_gamma(base_dir, exp_name):
    """
    Looks for ../exp_name/gamma_*/config.csv and returns:
    {
      price_nash=..., price_coop=...,
      profit_nash=..., profit_coop=...
    }
    Robust to Code1/Code2 naming differences you mentioned.
    """
    for cfg in glob.glob(os.path.join(base_dir, exp_name, "gamma_*", "config.csv")):
        try:
            df = pd.read_csv(cfg)
        except Exception:
            continue
        if df.empty:
            continue
        r = df.iloc[0].to_dict()

        # prices (preferred: NashP1, NashP2...), fallbacks: Pricenash
        p_nash, p_coop = [], []
        for i in range(1, 6):  # allow up to 5 players gracefully
            if f"NashP{i}" in r and f"CoopP{i}" in r:
                p_nash.append(float(r[f"NashP{i}"]))
                p_coop.append(float(r[f"CoopP{i}"]))
        # profits (preferred: NashProfit1/2..., CoopProfit1/2...)
        pi_nash, pi_coop = [], []
        for i in range(1, 6):
            if f"NashProfit{i}" in r and f"CoopProfit{i}" in r:
                pi_nash.append(float(r[f"NashProfit{i}"]))
                pi_coop.append(float(r[f"CoopProfit{i}"]))

        # fallbacks if needed (arrays stored under 'Pricenash'/'Pricecoop')
        if (not p_nash or not p_coop) and isinstance(r.get("Pricenash"), str):
            try:
                p_nash = list(map(float, json.loads(r["Pricenash"])))
                p_coop = list(map(float, json.loads(r["Pricecoop"])))
            except Exception:
                pass

        out = {}
        if p_nash and p_coop:
            out["price_nash"] = float(np.mean(p_nash))
            out["price_coop"] = float(np.mean(p_coop))
        if pi_nash and pi_coop:
            out["profit_nash"] = float(np.mean(pi_nash))
            out["profit_coop"] = float(np.mean(pi_coop))
        if out:
            return out
    return {}


# =========================
# PLOTTING
# =========================

def _maybe_overlay_benchmarks(ax, base_dir, exp_name, metric_key,
                              color_nash="k", color_coop="k"):
    bm = _read_benchmarks_from_any_gamma(base_dir, exp_name)
    if metric_key == "price":
        if "price_nash" in bm:
            ax.axhline(bm["price_nash"], linestyle="--", linewidth=1, alpha=0.6,
                       color=color_nash, label="Nash")
        if "price_coop" in bm:
            ax.axhline(bm["price_coop"], linestyle=":", linewidth=1, alpha=0.6,
                       color=color_coop, label="Coop")
    elif metric_key == "profit":
        if "profit_nash" in bm:
            ax.axhline(bm["profit_nash"], linestyle="--", linewidth=1, alpha=0.6,
                       color=color_nash, label="Nash")
        if "profit_coop" in bm:
            ax.axhline(bm["profit_coop"], linestyle=":", linewidth=1, alpha=0.6,
                       color=color_coop, label="Coop")


def plot_gamma_lines_across_param_v2(
    base_dir, experiment_base, param_values, param_name,
    label_fmt="{param_name}={val:.2f}", alpha_fill=0.15,
    overlay_benchmarks=True,
):
    """
    2×2 panel (Price, Profit, Price Gain, Profit Gain) with a γ-curve
    for each value in param_values. Uses `load_gamma_only_metric` so
    shaded ribbons work even when *_gain stds are missing.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    panels = [
        ("Price",       "price"),
        ("Profit",      "profit"),
        ("Price Gain",  "price_gain"),
        ("Profit Gain", "profit_gain"),
    ]

    for val in param_values:
        exp_name = f"{experiment_base}/{param_name}_{val:.2f}"

        loaded = {}
        for _, key in panels:
            try:
                dfm = load_gamma_only_metric(
                    os.path.join(base_dir, exp_name),
                    key,
                    fallback_gain_std="relative",
                    relative_gain_std_scale=0.10,
                )
                loaded[key] = dfm
            except Exception:
                continue

        for ax, (title, key) in zip(axes.ravel(), panels):
            if key not in loaded:
                continue
            dfm = loaded[key]
            x, y = dfm["gamma"].values, dfm["mean"].values
            ystd = dfm["std"].values

            ax.plot(x, y, marker="o", lw=2,
                    label=label_fmt.format(param_name=param_name, val=val))
            if np.isfinite(ystd).any():
                ax.fill_between(x, y - ystd, y + ystd, alpha=alpha_fill)

            ax.set_title(title)
            ax.set_xlabel("γ")
            ax.grid(True, alpha=0.3)

            if overlay_benchmarks and key in ("price", "profit"):
                _maybe_overlay_benchmarks(ax, base_dir, exp_name, key)

    # y-labels
    axes[0, 0].set_ylabel("Price")
    axes[0, 1].set_ylabel("Profit")
    axes[1, 0].set_ylabel("Price Gain")
    axes[1, 1].set_ylabel("Profit Gain")

    # legend on top, unique labels
    handles, labels = [], []
    for ax in axes.ravel():
        h, l = ax.get_legend_handles_labels()
        handles += h; labels += l
    uniq = dict(zip(labels, handles))
    if uniq:
        fig.legend(uniq.values(), uniq.keys(), loc="upper center",
                   ncol=min(6, len(uniq)))

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_heatmap_gamma_vs_param_v2(
    base_dir, experiment_base, param_values, param_name, metric="price",
    cmap="viridis",
):
    """
    γ×param heatmap for a chosen metric ('price'/'profit'/'price_gain'/'profit_gain').
    """
    # collect union of γ across all param values
    all_gammas = set()
    curves = {}
    for val in param_values:
        exp_name = f"{experiment_base}/{param_name}_{val:.2f}"
        try:
            dfm = load_gamma_only_metric(os.path.join(base_dir, exp_name), metric)
        except Exception:
            dfm = pd.DataFrame(columns=["gamma", "mean", "std"])
        curves[val] = dfm
        all_gammas.update(dfm.get("gamma", []))
    gammas = sorted(all_gammas)
    if not gammas:
        raise ValueError("No gamma data found for heatmap.")

    # grid: rows=param values, cols=γ
    Z = np.full((len(param_values), len(gammas)), np.nan)
    for i, val in enumerate(param_values):
        dfm = curves[val]
        for j, g in enumerate(gammas):
            row = dfm[dfm["gamma"] == g]
            if not row.empty:
                Z[i, j] = float(row.iloc[0]["mean"])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    im = ax.imshow(
        Z, aspect="auto", origin="lower", cmap=cmap,
        extent=[min(gammas), max(gammas), min(param_values), max(param_values)],
    )
    ax.set_xlabel("γ")
    ax.set_ylabel(param_name)
    ax.set_title(f"{metric.replace('_', ' ').title()} (heatmap)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    freeze_support()

    # Sweep ranges
    gamma_values = np.linspace(0.0, 3.0, 10)  # γ grid

    # (A) c-sweep with μ fixed at 0.25
    c_values = np.linspace(0.0, 2.0, 5)
    base_kwargs_c = dict(
        n=2, k=15, memory=1,
        mu=0.25,           # fixed here
        c=1.0,             # overwritten in the runner
        num_sessions=100,  # adjust for reliability/runtime
        aprint=True,
        demand_type="reference",
        common_reference=False,
        ref_prediction="qlearning",
    )
    # run_gamma_only_c_sweep(
    #     c_values, gamma_values, base_kwargs_c,
    #     experiment_base="mu_c/gamma_only_c", num_sessions=base_kwargs_c["num_sessions"]
    # )

    # (B) μ-sweep with c fixed at 1.0
    # NOTE: avoid μ=0 to prevent division-by-zero in logit demand.
    mu_values = [0.05, 0.25, 0.50, 0.75, 1.00, 5.00]
    #mu_values = np.linspace(0.1, 2.0, 5)
    base_kwargs_mu = dict(
        n=2, k=15, memory=1,
        mu=0.25,           # overwritten in the runner
        c=1.0,             # fixed here
        num_sessions=10,
        aprint=True,
        demand_type="reference",
        common_reference=False,
        ref_prediction="qlearning",
    )
    # run_gamma_only_mu_sweep(
    #     mu_values, gamma_values, base_kwargs_mu,
    #     experiment_base="mu_c/gamma_only_mu", num_sessions=base_kwargs_mu["num_sessions"]
    # )

    # ---------- Visualization ----------
    base_dir = "../Results_sockeye/experiments"

    # Ensure figure folders exist
    os.makedirs(os.path.join(base_dir, "mu_c/gamma_only_c/Figures"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "mu_c/gamma_only_mu/Figures"), exist_ok=True)

    # 2×2 panels across c
    fig_c = plot_gamma_lines_across_param_v2(
        base_dir, "mu_c/gamma_only_c", c_values, "c",
        label_fmt="c={val:.2f}", overlay_benchmarks=False
    )
    fig_c.savefig(os.path.join(base_dir, "mu_c/gamma_only_c/Figures", "gamma_lines_by_c.png"), dpi=220)

    # 2×2 panels across μ
    fig_mu = plot_gamma_lines_across_param_v2(
        base_dir, "mu_c/gamma_only_mu", mu_values, "mu",
        label_fmt="μ={val:.2f}", overlay_benchmarks=False
    )
    fig_mu.savefig(os.path.join(base_dir, "mu_c/gamma_only_mu/Figures", "gamma_lines_by_mu.png"), dpi=220)

    # Heatmaps
    fig_hc = plot_heatmap_gamma_vs_param_v2(
        base_dir, "mu_c/gamma_only_c", c_values, "c", metric="price"
    )
    fig_hc.savefig(os.path.join(base_dir, "mu_c/gamma_only_c/Figures", "heatmap_price_gamma_vs_c.png"), dpi=220)

    fig_hm = plot_heatmap_gamma_vs_param_v2(
        base_dir, "mu_c/gamma_only_mu", mu_values, "mu", metric="profit_gain"
    )
    fig_hm.savefig(os.path.join(base_dir, "mu_c/gamma_only_mu/Figures", "heatmap_profitgain_gamma_vs_mu.png"), dpi=220)

    print("Done.")
