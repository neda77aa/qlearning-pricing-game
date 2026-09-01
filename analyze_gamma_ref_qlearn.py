"""
Post-process the beta=4e-6 reference-Q-learning dual-convergence sweep.

For each gamma it reads cycle_statistics.csv (metrics) and q_stabilization.npz
(per-session Q-value change trajectories + convergence flags), then:

  * writes summary_metrics.csv into the run folder (gamma-indexed price, profit,
    price-gain, profit-gain, plus convergence rate / time), player-averaged with
    session dispersion (std) and standard error;
  * saves figures into <run>/Figures/:
      - metrics_vs_gamma.png   : 2x2 Price / Profit / Price Gain / Profit Gain
      - price_gamma.png, profit_gamma.png,
        price_gain_gamma.png, profit_gain_gamma.png   (individual panels)
      - convergence_rate_vs_gamma.png
      - q_stabilization.png    : mean|dQ| for firm & reference Q vs t,
                                 converged vs non-converged sessions.

Run from the repo root:  python analyze_gamma_ref_qlearn.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Double-convergence result archive (moved out of ../Results/experiments).
DC = "/Users/neda/Desktop/UBC/PHD/research_term_4/Result_double_convergence"
RUN = os.path.join(DC, "baseline_common_reference")
# Pool the primary run with any additional independent session blocks
# (batch2) so the main figure uses all sessions per gamma. Each block
# contributes i.i.d. sessions (independent OS-entropy seeds), so pooling is
# equivalent to one larger run.
RUNS = [RUN] + [r for r in [os.path.join(DC, "baseline_common_reference_batch2")]
                if os.path.isdir(r)]
FIGDIR = os.path.join(RUN, "Figures")
os.makedirs(FIGDIR, exist_ok=True)

# session_summaries.csv column stems for each reported metric
KEYMAP = {"price": "cycle_mean_price", "profit": "cycle_mean_profit",
          "price_gain": "cycle_price_gain", "profit_gain": "cycle_profit_gain"}

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
})

XLAB = r"$\gamma$ (Reference Dependence)"
COL = "tab:purple"


def gamma_dirs():
    gs = [d for d in os.listdir(RUN) if d.startswith("gamma_")]
    return sorted(gs, key=lambda s: float(s.split("_")[1]))


def pooled_summaries(g):
    """Concatenate session_summaries.csv for gamma dir `g` across all RUNS
    (batch1 + batch2 + ...). Returns a single DataFrame of all sessions."""
    dfs = []
    for r in RUNS:
        p = os.path.join(r, g, "session_summaries.csv")
        if os.path.exists(p):
            dfs.append(pd.read_csv(p))
    return pd.concat(dfs, ignore_index=True) if dfs else None


def build_summary():
    rows = []
    for g in gamma_dirs():
        gv = float(g.split("_")[1])
        df_s = pooled_summaries(g)
        if df_s is None:
            continue
        n_sess = len(df_s)
        n_conv = int(df_s["converged"].sum())
        row = {"gamma": gv,
               # nanmean over ALL sessions of the convergence flag == conv rate
               "convergence_rate": float(df_s["converged"].mean()),
               "mean_convergence_time": float(np.nanmean(df_s["time_to_convergence"])),
               "n_sessions": n_sess, "n_converged": n_conv}
        # player-average mean and std for each metric, nanmean over sessions
        # (non-converged sessions carry NaN gains and are excluded, matching
        # cycle_statistics semantics).
        for key, col in KEYMAP.items():
            p1, p2 = df_s[f"{col}_p1"].values, df_s[f"{col}_p2"].values
            m = 0.5 * (np.nanmean(p1) + np.nanmean(p2))
            # combine two players' session-std as RMS (they are near-identical)
            s = np.sqrt(0.5 * (np.nanstd(p1) ** 2 + np.nanstd(p2) ** 2))
            row[f"mean_{key}"] = m
            row[f"std_{key}"] = s
            row[f"se_{key}"] = s / np.sqrt(max(n_conv, 1))
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("gamma").reset_index(drop=True)
    out = os.path.join(RUN, "summary_metrics.csv")
    df.to_csv(out, index=False)
    print("wrote", out, "(pooled over %d run block(s))" % len(RUNS))
    return df


def _panel(ax, df, key, title, ylab):
    x = df["gamma"].values
    y = df[f"mean_{key}"].values
    band = df[f"std_{key}"].values
    ax.plot(x, y, "-o", color=COL, ms=3, lw=1.8)
    ax.fill_between(x, y - band, y + band, color=COL, alpha=0.18,
                    label=r"$\pm 1$ std (sessions)")
    ax.set_title(title)
    ax.set_xlabel(XLAB)
    ax.set_ylabel(ylab)
    ax.grid(alpha=0.3)


METRICS = [
    ("price", "Price", "Mean price"),
    ("profit", "Profit", "Mean profit"),
    ("price_gain", "Price Gain", "Mean price gain"),
    ("profit_gain", "Profit Gain", "Mean profit gain"),
]


def plot_metrics(df):
    # combined 2x2
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (key, title, ylab) in zip(axes.ravel(), METRICS):
        _panel(ax, df, key, title, ylab)
    axes.ravel()[0].legend(fontsize=9, loc="best")
    fig.tight_layout()
    p = os.path.join(FIGDIR, "metrics_vs_gamma.png")
    fig.savefig(p, dpi=200); plt.close(fig); print("wrote", p)

    # individual panels
    for key, title, ylab in METRICS:
        fig, ax = plt.subplots(figsize=(6, 4.2))
        _panel(ax, df, key, title, ylab)
        ax.legend(fontsize=9)
        fig.tight_layout()
        p = os.path.join(FIGDIR, f"{key}_gamma.png")
        fig.savefig(p, dpi=200); plt.close(fig); print("wrote", p)


def plot_convergence(df):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(df["gamma"], df["convergence_rate"], "-o", color="tab:red", ms=4)
    ax.set_ylim(0, 1.02)
    ax.set_title("Convergence rate (firm + reference stable)")
    ax.set_xlabel(XLAB); ax.set_ylabel("Fraction of sessions converged")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = os.path.join(FIGDIR, "convergence_rate_vs_gamma.png")
    fig.savefig(p, dpi=200); plt.close(fig); print("wrote", p)


FLOOR = 1e-12  # floor |dQ| before log so exact-zero snapshots don't underflow


def _group_band(trajs, col):
    """Median and 10-90 percentile of |dQ| over sessions at each recording bin,
    computed only where >=5 sessions are still present. Values floored at FLOOR."""
    if not trajs:
        return None, None, None, None
    maxlen = max(a.shape[0] for a in trajs)
    step = None
    vals = np.full((len(trajs), maxlen), np.nan)
    for i, a in enumerate(trajs):
        vals[i, :a.shape[0]] = np.maximum(a[:, col], FLOOR)
        if a.shape[0] >= 2 and step is None:
            step = a[1, 0] - a[0, 0]
    step = step or 1000
    tgrid = np.arange(1, maxlen + 1) * step
    n_present = np.sum(~np.isnan(vals), axis=0)
    keep = n_present >= 5
    med = np.nanmedian(vals, axis=0)
    lo = np.nanpercentile(vals, 10, axis=0)
    hi = np.nanpercentile(vals, 90, axis=0)
    return tgrid[keep], med[keep], lo[keep], hi[keep]


def plot_q_stabilization():
    conv, non = [], []
    for g in gamma_dirs():
        for r in RUNS:
            f = os.path.join(r, g, "q_stabilization.npz")
            if not os.path.exists(f):
                continue
            npz = np.load(f)
            # convergence map is per-file, so session-id collisions across
            # batches are harmless (resolved within each file).
            conv_map = {int(sid): bool(c) for sid, c, _ in npz["convergence"]}
            for k in npz.files:
                if not k.startswith("session_"):
                    continue
                a = npz[k]
                if a.size == 0:
                    continue
                (conv if conv_map.get(int(k.split("_")[1]), False) else non).append(a)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, (col, title) in zip(axes, [(1, "Firm Q"), (2, "Consumer reference Q")]):
        for trajs, c, lab in [(conv, "tab:green", "converged"),
                              (non, "tab:red", "not converged")]:
            t, med, lo, hi = _group_band(trajs, col)
            if t is None:
                continue
            ax.fill_between(t, lo, hi, color=c, alpha=0.15)
            ax.plot(t, med, color=c, lw=2.2, label=f"{lab} (n={len(trajs)})")
        ax.set_yscale("log")
        ax.set_ylim(FLOOR, 1.0)
        ax.set_title(f"{title}: median |$\\Delta Q$| per cell / 1000 steps")
        ax.set_xlabel("Training step $t$")
        ax.set_ylabel(r"|$\Delta Q$| per cell (median, 10-90 pct band)")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=9)
    fig.suptitle("Q-value stabilization before convergence "
                 "(floored at %g)" % FLOOR, y=1.02, fontsize=13)
    fig.tight_layout()
    p = os.path.join(FIGDIR, "q_stabilization.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig); print("wrote", p)


if __name__ == "__main__":
    df = build_summary()
    plot_metrics(df)
    plot_convergence(df)
    plot_q_stabilization()
    print("all figures saved to", FIGDIR)
