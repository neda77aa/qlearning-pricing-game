"""Unified figure generator for the reference-dependence paper.

This single script replaces the former collection of one-off figure scripts
(gen_altbench_gains, paper_irf_figures, recolor_linear_td3_purple,
recompute_linear_gains_longterm, rebuild_paper_figures, plot_td3_cycles,
plot_td3_cycles_appendix_style, postprocess_linear_benchmarks,
paper_figures_linear, paper_panels). Each is now a subcommand.

All figures are written under ``paper_overleaf/Images/`` (the Overleaf-synced
paper), so no output-path rewrite is needed any more.

Run from the repo root:

    PY=/Users/neda/llm_venv/bin/python

    $PY make_figures.py altbench              # Figs 3,4,7,9,10,12 gain panels + gamma-lambda/gamma-delta heatmaps
    $PY make_figures.py altbench --new        # appendix ES line figs + separated profit-gain heatmaps
    $PY make_figures.py irf                    # deviation/punishment panels + LaTeX table rows
    $PY make_figures.py recolor-linear-td3     # linear + TD3 price/profit/gain panels (purple style)
    $PY make_figures.py linear-gains-longterm  # overwrite linear gain panels with long-term benchmark
    $PY make_figures.py rebuild                # benchmark/market/misspec/firm-specific/lossaversion blocks
    $PY make_figures.py td3-cycles             # Fig td3_cycles (2x2 combined panel)
    $PY make_figures.py td3-cycles-appendix    # appendix-style per-panel TD3 cycles + legend
    $PY make_figures.py linear-postprocess [experiment]   # both-benchmark summary CSV + diagnostic plots
    $PY make_figures.py linear-paper-figs [experiment]    # publication-style linear line plots
    $PY make_figures.py panels {linear|td3} <experiment> <out_dir>   # alt-styling 4-panel figures
    $PY make_figures.py all                    # run the standard committed-figure set (best effort)
"""
import os
import re
import sys
import glob
import time
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

# --------------------------------------------------------------------------- #
# Shared paths & style
# --------------------------------------------------------------------------- #
ROOT = "/Users/neda/Desktop/UBC/PHD/research_term_4"
RES = os.path.join(ROOT, "Results", "experiments")          # simulation output
PR = os.path.join(ROOT, "paper_results")                    # archived paper runs
IMG = os.path.join(ROOT, "Algorithmic-Collusion-Replication",
                   "paper_overleaf", "Images")              # figures land here

# gamma / gain panel style (used by altbench, recolor, linear-gains, rebuild, panels)
RC = {"font.size": 18, "axes.labelsize": 18, "xtick.labelsize": 16,
      "ytick.labelsize": 16, "legend.fontsize": 14}
COLOR, LW, BAND_MULT, BAND_ALPHA = "tab:purple", 2.0, 0.8, 0.15

METRICS = [("price", "Price"), ("profit", "Profit"),
           ("price_gain", "Price Gain"), ("profit_gain", "Profit Gain")]


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def ff(s):
    """First float in a string like '[1.45 1.45]'."""
    return float(re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(s))[0])


def read_csv_retry(path, tries=5):
    """pd.read_csv with retries -- guards against transient disk I/O timeouts."""
    last = None
    for k in range(tries):
        try:
            return pd.read_csv(path)
        except (OSError, TimeoutError) as e:
            last = e
            time.sleep(0.5 * (k + 1))
    raise last


def profit_from_price(p, a=2.0, c=1.0, mu=0.25, a0=0.0, n=2):
    """Symmetric steady-state per-firm logit profit at price p (reference=p)."""
    e = np.exp((a - p) / mu)
    return (p - c) * e / (n * e + np.exp(a0 / mu))


def cfg_params(cfg_row):
    a = float(cfg_row["a1"]) if "a1" in cfg_row else 2.0
    c = float(cfg_row["c1"]) if "c1" in cfg_row else 1.0
    mu = float(cfg_row["mu"]) if "mu" in cfg_row else 0.25
    a0 = float(cfg_row["a0"]) if "a0" in cfg_row else 0.0
    return a, c, mu, a0


def load_gamma_metric(exp_dir, metric):
    """gamma, firm-averaged mean, firm-averaged std from each
    gamma_*/cycle_statistics.csv."""
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


# =========================================================================== #
# 1. ALTERNATIVE-BENCHMARK GAIN FIGURES  (was gen_altbench_gains.py)
# =========================================================================== #
# For each target figure we regenerate ONLY the two gain panels with a frozen
# collusive benchmark (Pcoop at gamma=0), an exact rescale of the stored gains:
#     gain_new(g) = gain_std(g) * R(g),  R(g) = (coop(g)-nash(g))/(coop(0)-nash(g)).
GAIN_METRICS = [("price_gain", "Price Gain", "figure1_gamma_only_q_price_gain_altbench.png"),
                ("profit_gain", "Profit Gain", "figure1_gamma_only_q_profit_gain_altbench.png")]


def _ab_benchmarks_gamma_only(root):
    g, Pn, Pc, PIn, PIc = [], [], [], [], []
    for d in sorted(glob.glob(os.path.join(root, "gamma_*"))):
        f = os.path.join(d, "config.csv")
        if not os.path.exists(f):
            continue
        cfg = pd.read_csv(f).iloc[0]
        gamma = ff(os.path.basename(d).split("gamma_")[1])
        a, c, mu, a0 = cfg_params(cfg)
        pn, pc = ff(cfg["Pnash"]), ff(cfg["Pcoop"])
        g.append(gamma)
        Pn.append(pn); Pc.append(pc)
        PIn.append(profit_from_price(pn, a, c, mu, a0))
        PIc.append(profit_from_price(pc, a, c, mu, a0))
    idx = np.argsort(g)
    A = lambda x: np.array(x)[idx]
    return {"price_gain": (A(g), A(Pn), A(Pc)),
            "profit_gain": (A(g), A(PIn), A(PIc))}


def _ab_rescale_factor(gammas, nash, coop):
    """R(g) = (coop(g)-nash(g)) / (coop(0)-nash(g)); coop(0) = coop at min gamma."""
    coop0 = coop[np.argmin(gammas)]
    return (coop - nash) / (coop0 - nash)


def _ab_series_gamma_only(root, gain_key):
    g, mu, sd = [], [], []
    for d in sorted(glob.glob(os.path.join(root, "gamma_*"))):
        f = os.path.join(d, "cycle_statistics.csv")
        if not os.path.exists(f):
            continue
        r = read_csv_retry(f).iloc[0]
        g.append(ff(os.path.basename(d).split("gamma_")[1]))
        mu.append(np.mean([r[f"mean_{gain_key}_p1"], r[f"mean_{gain_key}_p2"]]))
        sd.append(np.mean([r[f"std_{gain_key}_p1"], r[f"std_{gain_key}_p2"]]))
    idx = np.argsort(g)
    return np.array(g)[idx], np.array(mu)[idx], np.array(sd)[idx]


def _ab_series_grid_pooled(root, gain_key):
    rows = []
    for d in glob.glob(os.path.join(root, "gamma_*_lambda_*")):
        f = os.path.join(d, "cycle_statistics.csv")
        if not os.path.exists(f):
            continue
        base = os.path.basename(d)
        gamma = float(base.split("gamma_")[1].split("_lambda_")[0])
        r = read_csv_retry(f).iloc[0]
        m = np.mean([r[f"mean_{gain_key}_p1"], r[f"mean_{gain_key}_p2"]])
        s = np.mean([r[f"std_{gain_key}_p1"], r[f"std_{gain_key}_p2"]])
        rows.append((gamma, m, s))
    df = pd.DataFrame(rows, columns=["gamma", "m", "s"])
    out = []
    for gamma, gdf in df.groupby("gamma"):
        ms, ss = gdf["m"].values, gdf["s"].values
        mbar = float(np.mean(ms))
        within = float(np.nanmean(ss ** 2)) if np.any(np.isfinite(ss)) else 0.0
        between = float(np.var(ms, ddof=1)) if len(ms) > 1 else 0.0
        sbar = float(np.sqrt(max(within + between, 0.0)))
        out.append((gamma, mbar, sbar, len(ms)))
    out.sort()
    a = np.array(out)
    return a[:, 0], a[:, 1], a[:, 2], a[:, 3]


def _ab_benchmarks_grid(root):
    seen = {}
    for d in glob.glob(os.path.join(root, "gamma_*_lambda_*")):
        base = os.path.basename(d)
        gamma = float(base.split("gamma_")[1].split("_lambda_")[0])
        if gamma in seen:
            continue
        f = os.path.join(d, "config.csv")
        if not os.path.exists(f):
            continue
        seen[gamma] = read_csv_retry(f).iloc[0]
    g = sorted(seen)
    Pn, Pc, PIn, PIc = [], [], [], []
    for gamma in g:
        cfg = seen[gamma]
        a, c, mu, a0 = cfg_params(cfg)
        pn, pc = ff(cfg["Pnash"]), ff(cfg["Pcoop"])
        Pn.append(pn); Pc.append(pc)
        PIn.append(profit_from_price(pn, a, c, mu, a0))
        PIc.append(profit_from_price(pc, a, c, mu, a0))
    g = np.array(g)
    return {"price_gain": (g, np.array(Pn), np.array(Pc)),
            "profit_gain": (g, np.array(PIn), np.array(PIc))}


def _ab_td3_benchmarks(root):
    g, Pn, Pc, PIn, PIc = [], [], [], [], []
    for d in sorted(glob.glob(os.path.join(root, "gamma_*"))):
        f = os.path.join(d, "cycle_statistics.csv")
        if not os.path.exists(f):
            continue
        r = read_csv_retry(f).iloc[0]
        g.append(ff(os.path.basename(d).split("gamma_")[1]))
        pn = 0.5 * (r["p_nash_p1"] + r["p_nash_p2"])
        pc = 0.5 * (r["p_coop_p1"] + r["p_coop_p2"])
        Pn.append(pn); Pc.append(pc)
        PIn.append(profit_from_price(pn))
        PIc.append(profit_from_price(pc))
    idx = np.argsort(g)
    A = lambda x: np.array(x)[idx]
    return {"price_gain": (A(g), A(Pn), A(Pc)),
            "profit_gain": (A(g), A(PIn), A(PIc))}


def _ab_build_line_figure(name, out_dir, curves):
    os.makedirs(out_dir, exist_ok=True)
    with plt.rc_context(RC):
        for gain_key, ylabel, fname in GAIN_METRICS:
            fig, ax = plt.subplots(figsize=(6, 4))
            for cv in curves:
                root = cv["root"]
                if cv["kind"] == "grid":
                    g, mu, sd, n_eff = _ab_series_grid_pooled(root, gain_key)
                    bench = _ab_benchmarks_grid(root)[gain_key]
                else:  # gonly / td3
                    g, mu, sd = _ab_series_gamma_only(root, gain_key)
                    n_eff = None
                    if cv["kind"] == "td3":
                        bench = _ab_td3_benchmarks(root)[gain_key]
                    else:
                        bench = _ab_benchmarks_gamma_only(root)[gain_key]
                bg, nash, coop = bench
                R = _ab_rescale_factor(bg, nash, coop)
                Rmap = {round(float(x), 6): r for x, r in zip(bg, R)}
                rr = np.array([Rmap[round(float(x), 6)] for x in g])
                mu_new, sd_new = mu * rr, sd * rr

                bkind, bmult = cv["band"]
                if bkind == "se" and n_eff is not None:
                    half = bmult * (sd_new / np.sqrt(n_eff))
                else:
                    half = bmult * sd_new
                ax.plot(g, mu_new, color=cv["color"], linestyle=cv["ls"],
                        lw=cv["lw"], marker=" ", label=cv.get("label", ""))
                ax.fill_between(g, mu_new - half, mu_new + half,
                                color=cv["color"], alpha=cv["alpha"], lw=0)
            ax.grid(True, ls="--", alpha=0.4)
            ax.set_xlabel(r"$\gamma$")
            ax.set_ylabel(ylabel)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, fname), dpi=300, bbox_inches="tight")
            plt.close(fig)
    print(f"  [line] {name}: wrote 2 altbench gain panels -> {out_dir}")


def _ab_gamma_lambda_heatmaps(root, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cfg = {"price_gain": ("PuRd", "gamma_lambda_price_gain_altbench.png"),
           "profit_gain": ("OrRd", "gamma_lambda_profit_gain_altbench.png")}
    bench_all = _ab_benchmarks_grid(root)
    rc = {**RC, "font.family": "serif", "mathtext.fontset": "dejavuserif",
          "xtick.labelsize": 16, "ytick.labelsize": 16}
    for gain_key, (cmap, fname) in cfg.items():
        rows = []
        for d in glob.glob(os.path.join(root, "gamma_*_lambda_*")):
            f = os.path.join(d, "cycle_statistics.csv")
            if not os.path.exists(f):
                continue
            base = os.path.basename(d)
            gamma = float(base.split("gamma_")[1].split("_lambda_")[0])
            lam = float(base.split("lambda_")[1].split("_")[0])
            r = read_csv_retry(f).iloc[0]
            v = np.mean([r[f"mean_{gain_key}_p1"], r[f"mean_{gain_key}_p2"]])
            rows.append((gamma, lam, v))
        df = pd.DataFrame(rows, columns=["gamma", "lambda", "v"])
        piv = df.pivot_table(index="gamma", columns="lambda", values="v", aggfunc="mean")
        gammas = piv.index.values.astype(float)
        lams = piv.columns.values.astype(float)
        grid = piv.values.astype(float)
        bg, nash, coop = bench_all[gain_key]
        R = _ab_rescale_factor(bg, nash, coop)
        Rmap = {round(float(x), 6): r for x, r in zip(bg, R)}
        rr = np.array([Rmap[round(float(x), 6)] for x in gammas])
        grid = grid * rr[:, None]

        with plt.rc_context(rc):
            fig, ax = plt.subplots(figsize=(7, 5))
            dl = (lams[1] - lams[0]) / 2 if len(lams) > 1 else 0.5
            dg = (gammas[1] - gammas[0]) / 2 if len(gammas) > 1 else 0.5
            im = ax.imshow(grid, aspect="auto", origin="lower",
                           extent=[lams.min() - dl, lams.max() + dl,
                                   gammas.min() - dg, gammas.max() + dg],
                           cmap=cmap)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=12)
            ax.set_xlabel(r"$\lambda$ (Memory Weight)")
            ax.set_ylabel(r"$\gamma$")
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, fname), dpi=300, bbox_inches="tight")
            plt.close(fig)
    print(f"  [heatmap] gamma_lambda: wrote 2 altbench panels -> {out_dir}")


def _ab_gamma_delta_heatmaps(root, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cfg = {"price_gain": ("Price Gain", "price_gain_heatmap_altbench.png"),
           "profit_gain": ("Profit Gain", "profit_gain_heatmap_altbench.png")}
    seen = {}
    for d in glob.glob(os.path.join(root, "gamma_*_delta_*")):
        base = os.path.basename(d)
        gamma = float(base.split("gamma_")[1].split("_delta_")[0])
        if gamma in seen:
            continue
        f = os.path.join(d, "config.csv")
        if os.path.exists(f):
            seen[gamma] = pd.read_csv(f).iloc[0]
    gsort = sorted(seen)
    Pn = {g: ff(seen[g]["Pnash"]) for g in gsort}
    Pc = {g: ff(seen[g]["Pcoop"]) for g in gsort}
    PIn, PIc = {}, {}
    for g in gsort:
        a, c, mu, a0 = cfg_params(seen[g])
        PIn[g] = profit_from_price(Pn[g], a, c, mu, a0)
        PIc[g] = profit_from_price(Pc[g], a, c, mu, a0)
    gmin = min(gsort)

    for gain_key, (title, fname) in cfg.items():
        rows = []
        for d in glob.glob(os.path.join(root, "gamma_*_delta_*")):
            f = os.path.join(d, "cycle_statistics.csv")
            if not os.path.exists(f):
                continue
            base = os.path.basename(d)
            gamma = float(base.split("gamma_")[1].split("_delta_")[0])
            delta = float(base.split("delta_")[1].split("_")[0])
            r = read_csv_retry(f).iloc[0]
            v = np.mean([r[f"mean_{gain_key}_p1"], r[f"mean_{gain_key}_p2"]])
            rows.append((gamma, delta, v))
        df = pd.DataFrame(rows, columns=["gamma", "delta", "v"])
        ug = np.sort(df["gamma"].unique())
        ud = np.sort(df["delta"].unique())
        grid = np.full((len(ud), len(ug)), np.nan)
        if gain_key == "price_gain":
            nash = {g: Pn[g] for g in gsort}; coop = {g: Pc[g] for g in gsort}; coop0 = Pc[gmin]
        else:
            nash = {g: PIn[g] for g in gsort}; coop = {g: PIc[g] for g in gsort}; coop0 = PIc[gmin]
        for _, rr in df.iterrows():
            g, dlt, v = rr["gamma"], rr["delta"], rr["v"]
            R = (coop[g] - nash[g]) / (coop0 - nash[g])
            i = np.where(ud == dlt)[0][0]
            j = np.where(ug == g)[0][0]
            grid[i, j] = v * R
        fig = plt.figure(figsize=(10, 8))
        im = plt.imshow(grid, aspect="auto", origin="lower",
                        extent=[ug.min(), ug.max(), ud.min(), ud.max()], cmap="Reds")
        plt.colorbar(im, label=title)
        plt.xlabel(r"$\gamma$")
        plt.ylabel(r"$\delta$")
        plt.title(title)
        fig.savefig(os.path.join(out_dir, fname), dpi=300, bbox_inches="tight")
        plt.close(fig)
    print(f"  [heatmap] gamma_delta: wrote 2 altbench panels -> {out_dir}")


def _ab_load_grid_gl(root, gain_key):
    rows = []
    for d in glob.glob(os.path.join(root, "gamma_*_lambda_*")):
        f = os.path.join(d, "cycle_statistics.csv")
        if not os.path.exists(f):
            continue
        base = os.path.basename(d)
        gamma = float(base.split("gamma_")[1].split("_lambda_")[0])
        lam = float(base.split("lambda_")[1].split("_")[0])
        r = read_csv_retry(f).iloc[0]
        v = np.mean([r[f"mean_{gain_key}_p1"], r[f"mean_{gain_key}_p2"]])
        rows.append((gamma, lam, v))
    df = pd.DataFrame(rows, columns=["gamma", "lambda", "v"])
    piv = df.pivot_table(index="gamma", columns="lambda", values="v", aggfunc="mean")
    gammas = piv.index.values.astype(float)
    lams = piv.columns.values.astype(float)
    grid = piv.values.astype(float)
    bg, nash, coop = _ab_benchmarks_grid(root)[gain_key]
    R = _ab_rescale_factor(bg, nash, coop)
    Rmap = {round(float(x), 6): rr for x, rr in zip(bg, R)}
    rr = np.array([Rmap[round(float(x), 6)] for x in gammas])
    return gammas, lams, grid * rr[:, None]


def _ab_heatmap_ax(ax, gammas, lams, grid, title, cmap, norm):
    dl = (lams[1] - lams[0]) / 2 if len(lams) > 1 else 0.5
    dg = (gammas[1] - gammas[0]) / 2 if len(gammas) > 1 else 0.5
    im = ax.imshow(grid, aspect="auto", origin="lower",
                   extent=[lams.min() - dl, lams.max() + dl,
                           gammas.min() - dg, gammas.max() + dg],
                   cmap=cmap, norm=norm)
    ax.set_title(title)
    ax.set_xlabel(r"$\lambda$ (Memory Weight)")
    ax.set_ylabel(r"$\gamma$ (Ref. Dependence)")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    for lab in ax.get_xticklabels():
        lab.set_rotation(45)
        lab.set_ha("right")
    gc = np.where(np.isfinite(grid), grid, np.nanmean(grid))
    X, Y = np.meshgrid(lams, gammas)
    ax.contour(X, Y, gc, levels=6, colors="black", linewidths=0.5, alpha=0.7)
    return im


def _ab_separated_heatmaps(out_dir, root_a, root_b, titles, fnames, figsize):
    os.makedirs(out_dir, exist_ok=True)
    gk = "profit_gain"
    ga, la, grid_a = _ab_load_grid_gl(root_a, gk)
    gb, lb, grid_b = _ab_load_grid_gl(root_b, gk)
    g_all = np.sort(np.unique(np.concatenate([ga, gb])))
    l_all = np.sort(np.unique(np.concatenate([la, lb])))

    def onto(gs, ls, grid):
        out = np.full((len(g_all), len(l_all)), np.nan)
        for i, gg in enumerate(gs):
            ii = int(np.argmin(np.abs(g_all - gg)))
            for j, ll in enumerate(ls):
                jj = int(np.argmin(np.abs(l_all - ll)))
                out[ii, jj] = grid[i, j]
        return out

    A, B = onto(ga, la, grid_a), onto(gb, lb, grid_b)
    D = B - A
    valid = np.hstack([A[np.isfinite(A)], B[np.isfinite(B)]])
    norm_abs = mcolors.Normalize(vmin=valid.min(), vmax=valid.max())
    md = float(np.max(np.abs(D[np.isfinite(D)])))
    norm_diff = mcolors.TwoSlopeNorm(vcenter=0, vmin=-md, vmax=md)
    rc = {**RC, "font.family": "serif", "mathtext.fontset": "dejavuserif",
          "axes.titlesize": 22, "axes.labelsize": 20,
          "xtick.labelsize": 18, "ytick.labelsize": 18}
    with plt.rc_context(rc):
        for grid, title, fname, cmap, norm in [
                (A, titles[0], fnames[0], "OrRd", norm_abs),
                (B, titles[1], fnames[1], "OrRd", norm_abs),
                (D, titles[2], fnames[2], "BrBG", norm_diff)]:
            fig, ax = plt.subplots(figsize=figsize)
            im = _ab_heatmap_ax(ax, g_all, l_all, grid, title, cmap, norm)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.savefig(os.path.join(out_dir, fname), dpi=300, bbox_inches="tight")
            plt.close(fig)
    print(f"  [heatmap] separated: wrote 3 altbench panels -> {out_dir}")


def _C(root, color, ls, lw, band, alpha, kind="gonly", label=""):
    return dict(root=root, color=color, ls=ls, lw=lw, band=band, alpha=alpha,
                kind=kind, label=label)


def _ab_line_figs():
    return {
        "benchmark": (f"{IMG}/4_seperate_figures/benchmark", [
            _C(f"{PR}/benchmark_figure/gamma_nloss_only_reference_True",
               "tab:purple", "-", 2.0, ("sigma", 0.8), 0.15)]),
        "market_structure": (f"{IMG}/4_seperate_figures/market_structure", [
            _C(f"{PR}/market_structure/gamma_nloss_only_reference_True",
               "tab:purple", "-", 3.0, ("sigma", 0.8), 0.18),
            _C(f"{PR}/market_structure/gamma_nloss_only_reference_Truec_0",
               "tab:purple", (0, (6, 3)), 2.2, ("sigma", 0.8), 0.12),
            _C(f"{PR}/market_structure/gamma_nloss_only_reference_Truemu_0",
               "tab:purple", (0, (1, 2)), 2.2, ("sigma", 0.8), 0.12)]),
        "misspecification": (f"{IMG}/4_seperate_figures/misspecification", [
            _C(f"{PR}/misspecification/gamma_nloss_only_reference_True",
               "tab:purple", "-", 2.0, ("sigma", 0.8), 0.15),
            _C(f"{PR}/misspecification/gamma_nloss_only_misspecification_True",
               "tab:red", "-", 2.0, ("sigma", 0.8), 0.15)]),
        "Firm-specific": (f"{IMG}/4_seperate_figures/Firm-specific", [
            _C(f"{PR}/Firm_specific/gamma_nloss_only_reference_True",
               "tab:purple", "-", 2.0, ("sigma", 0.8), 0.15),
            _C(f"{PR}/Firm_specific/gamma_nloss_only_reference_False",
               "tab:purple", "--", 2.0, ("sigma", 0.8), 0.15)]),
        "exp_smooth": (f"{IMG}/4_seperate_figures/exp_smooth", [
            _C(f"{PR}/exp_smoothing/gamma_lambda_reference_True",
               "tab:blue", "-", 2.0, ("se", 1.8), 0.15, kind="grid"),
            _C(f"{PR}/qqlearning/gamma_nloss_only_reference_True",
               "tab:purple", "-", 2.0, ("sigma", 0.8), 0.15)]),
        "td3": (f"{IMG}/4_seperate_figures_lr1e-4/td3", [
            _C(f"{RES}/td3_production_reference_15g_50s_lr1e-4",
               "tab:purple", "-", 2.0, ("sigma", 0.8), 0.15, kind="td3")]),
    }


def _ab_line_figs_new():
    return {
        "exp_smoothing_misspecification": (
            f"{IMG}/4_seperate_figures/exp_smoothing_misspecification", [
                _C(f"{PR}/exp_smoothing_misspecification/gamma_lambda_reference_True",
                   "tab:blue", "-", 2.0, ("se", 1.8), 0.15, kind="grid"),
                _C(f"{PR}/exp_smoothing_misspecification/gamma_lambda_misspecification_True",
                   "tab:orange", "-", 2.0, ("se", 1.8), 0.15, kind="grid")]),
        "exp_smoothing_firmspecific": (
            f"{IMG}/4_seperate_figures/exp_smoothing_firmspecific", [
                _C(f"{PR}/exp_smoothing_firm_specific/gamma_lambda_reference_True",
                   "tab:blue", "-", 2.0, ("se", 1.8), 0.15, kind="grid"),
                _C(f"{PR}/exp_smoothing_firm_specific/gamma_lambda_reference_False",
                   "tab:blue", "--", 2.0, ("se", 1.8), 0.15, kind="grid")]),
    }


def cmd_altbench(args):
    if args.new:
        print("Generating alternative-benchmark gain figures (Figs 27-30)...")
        for name, (out_dir, curves) in _ab_line_figs_new().items():
            _ab_build_line_figure(name, out_dir, curves)
        _ab_separated_heatmaps(
            f"{IMG}/4_seperate_figures/Separated_Panels_miss",
            f"{PR}/Separated_Panels_miss/gamma_lambda_reference_True",
            f"{PR}/Separated_Panels_miss/gamma_lambda_misspecification_True",
            ("Profit Gain\nReference-Aware", "Profit Gain\nReference-Naive",
             "Profit Gain\nDifference (Naive − Aware)"),
            ("profit_gain_aware_altbench.png", "profit_gain_naive_altbench.png",
             "profit_gain_difference_altbench.png"),
            (7, 6))
        _ab_separated_heatmaps(
            f"{IMG}/4_seperate_figures/Seperated_Panels_CR",
            f"{PR}/Separated_Panels_CR_True_vs_False/gamma_lambda_reference_True",
            f"{PR}/Separated_Panels_CR_True_vs_False/gamma_lambda_reference_False",
            ("Profit Gain\nReference-Aware (CR=True)", "Profit Gain\nReference-Aware (CR=False)",
             "Profit Gain\nDiff. (CR=False − CR=True)"),
            ("profit_gain_cr_true_altbench.png", "profit_gain_cr_false_altbench.png",
             "profit_gain_cr_diff_altbench.png"),
            (6, 4))
        print("done")
    else:
        print("Generating alternative-benchmark gain figures...")
        for name, (out_dir, curves) in _ab_line_figs().items():
            _ab_build_line_figure(name, out_dir, curves)
        _ab_gamma_lambda_heatmaps(f"{PR}/gamma_lambda/gamma_lambda_reference_True",
                                  f"{IMG}/4_seperate_figures/gamma_lambda")
        _ab_gamma_delta_heatmaps(f"{RES}/gamma_delta/gamma_delta_reference_True_contref",
                                 f"{IMG}/gamma_delta")
        print("done")


# =========================================================================== #
# 2. IMPULSE-RESPONSE FIGURES  (was paper_irf_figures.py)
# =========================================================================== #
IRF_TAB = os.path.join(RES, "gamma_nloss_reference_True_beta4e-6_ESref", "Figures")
IRF_TD3 = os.path.join(RES, "impulse_response")
IRF_OUT = os.path.join(IMG, "impulse_response")

IRF_TAB_GAMMAS = ["0.05", "1.0672", "2.0845", "3.0"]
IRF_TD3_GAMMAS = ["0.05", "1.1036", "2.1571", "3.0"]
IRF_REP = "1.0672"                        # representative gamma for the mechanism fig

_S_DEV, _S_NON = "#c02f2f", "#2a5fb0"     # deviator (red), non-deviator (blue)
_MUT, _BASE = "#8a8a8a", "#b8b8b8"
_LS_MONO, _LS_NASH, _LS_PRE = (0, (1, 2)), (0, (4, 3)), "-"
IRF_RC = {"font.size": 15, "axes.labelsize": 16, "axes.titlesize": 16,
          "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 14}


def _irf_load(dirpath, g, dt, prefix="irf"):
    return dict(np.load(os.path.join(dirpath, f"{prefix}_gamma_{g}_dev-{dt}.npz")))


def _irf_panel(res, out, ylim=None):
    t = np.arange(len(res["dev_price"]))
    fig, ax = plt.subplots(figsize=(5.6, 4.3))
    ax.axhline(float(res["p_coop"]), color=_MUT, lw=1.1, ls=_LS_MONO, zorder=1)
    ax.axhline(float(res["p_nash"]), color=_MUT, lw=1.1, ls=_LS_NASH, zorder=1)
    ax.axhline(float(res["long_run"]), color=_BASE, lw=1.2, ls=_LS_PRE, zorder=1)
    ax.axvline(1, color="#d8d8d8", lw=1.0, zorder=0)
    ax.plot(t, res["dev_price"], color=_S_DEV, lw=2.0, marker="o", ms=4.5, zorder=3)
    ax.plot(t, res["nondev_price"], color=_S_NON, lw=2.0, marker="^", ms=4.5,
            ls="--", zorder=3)
    ax.set_xlabel(r"Period ($\tau$); deviation at $\tau=1$")
    ax.set_ylabel("Price")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", color="#ececec", lw=0.7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def _irf_legend(out):
    handles = [
        Line2D([], [], color=_S_DEV, lw=2.0, marker="o", ms=6, label="Deviating firm"),
        Line2D([], [], color=_S_NON, lw=2.0, marker="^", ms=6, ls="--",
               label="Non-deviating (rival) firm"),
        Line2D([], [], color=_BASE, lw=1.4, ls=_LS_PRE,
               label="Pre-deviation (collusive) price"),
        Line2D([], [], color=_MUT, lw=1.4, ls=_LS_NASH, label="Nash price"),
        Line2D([], [], color=_MUT, lw=1.4, ls=_LS_MONO, label="Monopoly price"),
    ]
    fig = plt.figure(figsize=(12, 0.6))
    fig.legend(handles=handles, loc="center", ncol=5, frameon=False,
               handlelength=2.6, columnspacing=1.6)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def _irf_common_ylim(*ress, pad=0.03):
    ys = []
    for r in ress:
        ys += [float(r["dev_price"].min()), float(r["dev_price"].max()),
               float(r["nondev_price"].min()), float(r["nondev_price"].max()),
               float(r["p_nash"]), float(r["p_coop"])]
    lo, hi = min(ys), max(ys)
    m = pad * (hi - lo)
    return lo - m, hi + m


def cmd_irf(args):
    os.makedirs(IRF_OUT, exist_ok=True)
    with plt.rc_context(IRF_RC):
        br, na = _irf_load(IRF_TAB, IRF_REP, "br"), _irf_load(IRF_TAB, IRF_REP, "nash")
        yl = _irf_common_ylim(br, na)
        _irf_panel(br, os.path.join(IRF_OUT, "irf_mechanism_a.png"), ylim=yl)
        _irf_panel(na, os.path.join(IRF_OUT, "irf_mechanism_b.png"), ylim=yl)
        _irf_panel(_irf_load(IRF_TAB, "0.05", "nash"),
                   os.path.join(IRF_OUT, "irf_by_gamma_a.png"))
        _irf_panel(_irf_load(IRF_TAB, "3.0", "nash"),
                   os.path.join(IRF_OUT, "irf_by_gamma_b.png"))
        _irf_legend(os.path.join(IRF_OUT, "irf_legend.png"))

    def row(dirpath, g, prefix):
        b = _irf_load(dirpath, g, "br", prefix)
        n = _irf_load(dirpath, g, "nash", prefix)
        return (float(g), float(b["frac_unprofitable"]) * 100,
                float(n["frac_unprofitable"]) * 100, int(b["n_obs"]))
    print("\n% ---- TABULAR rows ----")
    for g in IRF_TAB_GAMMAS:
        gm, ubr, un, n = row(IRF_TAB, g, "irf")
        print(f"${gm:.2f}$ & {ubr:.1f}\\% & {un:.1f}\\% & {n} \\\\")
    print("\n% ---- TD3 rows ----")
    for g in IRF_TD3_GAMMAS:
        gm, ubr, un, n = row(IRF_TD3, g, "irf_td3")
        print(f"${gm:.2f}$ & {ubr:.1f}\\% & {un:.1f}\\% & {n} \\\\")


# =========================================================================== #
# 3. LINEAR + TD3 PURPLE PANELS  (was recolor_linear_td3_purple.py)
# =========================================================================== #
def cmd_recolor_linear_td3(args):
    blocks = {
        "linear": (
            os.path.join(RES, "linear_benchmark", "gamma_only_linear_qref_beta4e-6"),
            os.path.join(IMG, "4_seperate_figures_beta4e6", "linear")),
        "td3": (
            os.path.join(RES, "td3_production_reference_15g_50s_lr1e-4"),
            os.path.join(IMG, "4_seperate_figures_lr1e-4", "td3")),
    }
    with plt.rc_context(RC):
        for block, (exp_dir, out_dir) in blocks.items():
            if not os.path.isdir(exp_dir):
                print(f"!! missing source for {block}: {exp_dir}")
                continue
            os.makedirs(out_dir, exist_ok=True)
            n_g = len(glob.glob(os.path.join(exp_dir, "gamma_*")))
            print(f"{block}: {n_g} gammas  <-  {exp_dir}")
            for metric, ylabel in METRICS:
                g, mu, sd = load_gamma_metric(exp_dir, metric)
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


# =========================================================================== #
# 4. LINEAR LONG-TERM GAIN PANELS  (was recompute_linear_gains_longterm.py)
# =========================================================================== #
P_LC, PI_LC = 0.5, 0.25                    # long-term collusive benchmark (r = p)


def _lin_p_nash(g):
    return 1.0 / (3.0 + 2.0 * g)           # n=2, c=0


def _lin_pi_nash(g):
    return 2.0 * (1.0 + g) / (3.0 + 2.0 * g) ** 2


def _lin_longterm_gain(exp_dir, which):
    rows = []
    for d in sorted(glob.glob(os.path.join(exp_dir, "gamma_*"))):
        f = os.path.join(d, "cycle_statistics.csv")
        if not os.path.exists(f):
            continue
        r = pd.read_csv(f).iloc[0]
        g = float(os.path.basename(d).split("gamma_")[1])
        mu = np.mean([float(r[f"mean_{which}_p1"]), float(r[f"mean_{which}_p2"])])
        sd = np.mean([float(r[f"std_{which}_p1"]), float(r[f"std_{which}_p2"])])
        if which == "price":
            lo, hi = _lin_p_nash(g), P_LC
        else:
            lo, hi = _lin_pi_nash(g), PI_LC
        span = hi - lo
        rows.append((g, (mu - lo) / span, sd / span))
    rows.sort()
    a = np.array(rows)
    return a[:, 0], a[:, 1], a[:, 2]


def cmd_linear_gains_longterm(args):
    blocks = [
        (os.path.join(RES, "linear_benchmark", "gamma_only_linear_qref_beta4e-6"),
         os.path.join(IMG, "4_seperate_figures_beta4e6", "linear")),
        (os.path.join(RES, "linear_benchmark", "gamma_only_linear_beta4e-6"),
         os.path.join(IMG, "4_seperate_figures_beta4e6", "linear_es")),
    ]
    panels = [("price", "Price Gain", "figure1_gamma_only_q_price_gain.png"),
              ("profit", "Profit Gain", "figure1_gamma_only_q_profit_gain.png")]
    with plt.rc_context(RC):
        for exp_dir, out_dir in blocks:
            if not os.path.isdir(exp_dir):
                print(f"!! missing source: {exp_dir}")
                continue
            os.makedirs(out_dir, exist_ok=True)
            n_g = len(glob.glob(os.path.join(exp_dir, "gamma_*")))
            print(f"linear long-term gains: {n_g} gammas  <-  {exp_dir}")
            for which, ylabel, fname in panels:
                g, mu, sd = _lin_longterm_gain(exp_dir, which)
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(g, mu, color=COLOR, lw=LW)
                ax.fill_between(g, mu - BAND_MULT * sd, mu + BAND_MULT * sd,
                                color=COLOR, alpha=BAND_ALPHA, lw=0)
                ax.set_xlabel(r"$\gamma$")
                ax.set_ylabel(ylabel)
                ax.grid(True, ls="--", alpha=0.4)
                out = os.path.join(out_dir, fname)
                fig.savefig(out, dpi=300, bbox_inches="tight")
                plt.close(fig)
                print(f"    wrote {out}  (range {mu.min():.3f}..{mu.max():.3f})")
    print("done.")


# =========================================================================== #
# 5. MAIN GAMMA BLOCKS FROM beta=4e-6 RERUNS  (was rebuild_paper_figures.py)
# =========================================================================== #
_RB_OUT = os.path.join(IMG, "4_seperate_figures_beta4e6")
_RB_E = {
    "ref_T":  f"{RES}/gamma_nloss_reference_True_qref_beta4e-6",
    "c0":     f"{RES}/gamma_nloss_reference_Truec_0_qref_beta4e-6",
    "mu0":    f"{RES}/gamma_nloss_reference_Truemu_0_qref_beta4e-6",
    "miss_T": f"{RES}/gamma_nloss_misspecification_True_qref_beta4e-6",
    "ref_F":  f"{RES}/gamma_nloss_reference_False_qref_beta4e-6",
    "loss":   f"{RES}/lossaversion_reverse_beta4e-6",
}
_RB_BLOCKS = {
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


def _rb_build_block(name, curves):
    out_dir = os.path.join(_RB_OUT, name)
    os.makedirs(out_dir, exist_ok=True)
    with plt.rc_context(RC):
        for metric, ylabel in METRICS:
            fig, ax = plt.subplots(figsize=(6, 4))
            for exp, color, ls, lw, bm, ba, label in curves:
                g, mu, sd = load_gamma_metric(_RB_E[exp], metric)
                ax.plot(g, mu, color=color, linestyle=ls, lw=lw, label=label)
                ax.fill_between(g, mu - bm * sd, mu + bm * sd,
                                color=color, alpha=ba, lw=0)
            ax.set_xlabel(r"$\gamma$")
            ax.set_ylabel(ylabel)
            ax.grid(True, ls="--", alpha=0.4)
            fig.savefig(os.path.join(out_dir, f"figure1_gamma_only_q_{metric}.png"),
                        dpi=300, bbox_inches="tight")
            plt.close(fig)
        if len(curves) > 1:
            fig = plt.figure(figsize=(8, 0.7))
            handles = [plt.Line2D([], [], color=c, linestyle=ls, lw=lw, label=lab)
                       for _, c, ls, lw, _, _, lab in curves]
            fig.legend(handles=handles, loc="center", ncol=2, frameon=False)
            fig.savefig(os.path.join(out_dir, "legend_reference_naive.png"),
                        dpi=300, bbox_inches="tight")
            plt.close(fig)
    print(f"  {name}: done")


def _rb_build_lossaversion():
    out_dir = os.path.join(_RB_OUT, "lossaversion")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for d in sorted(glob.glob(os.path.join(_RB_E["loss"], "lossaversion_*"))):
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


def cmd_rebuild(args):
    print(f"rebuilding into {_RB_OUT}")
    for name, curves in _RB_BLOCKS.items():
        missing = [e for e, *_ in curves
                   if not glob.glob(os.path.join(_RB_E[e], "gamma_*", "cycle_statistics.csv"))]
        if missing:
            print(f"  {name}: SKIPPED (data not ready: {missing})")
            continue
        _rb_build_block(name, curves)
    if glob.glob(os.path.join(_RB_E["loss"], "lossaversion_*", "cycle_statistics.csv")):
        _rb_build_lossaversion()
    else:
        print("  lossaversion: SKIPPED (data not ready)")
    print("done")


# =========================================================================== #
# 6. TD3 CYCLES  (combined 2x2 panel; was plot_td3_cycles.py)
# =========================================================================== #
_TD3_EXP = os.path.join(RES, "td3_production_reference_15g_50s_lr1e-4")
_TD3_CYC_OUT = os.path.join(IMG, "4_seperate_figures_lr1e-4", "td3_cycles")
_TD3_GAME_KWARGS = dict(n=2, k=15, memory=1, demand_type="reference",
                        common_reference=True, lossaversion=1)
_NDISP = 12                                # periods to display


def _td3_load_session(g_dir):
    npz = np.load(os.path.join(g_dir, "rollout_paths.npz"))
    keys = [k for k in npz.files if k.startswith("prices_s")]
    return [npz[k] for k in sorted(keys, key=lambda s: int(s.split("s")[-1]))]


def _td3_find_session(gamma, sidx):
    for g_dir in glob.glob(os.path.join(_TD3_EXP, "gamma_*")):
        g = float(os.path.basename(g_dir).split("gamma_")[1])
        if abs(round(g, 2) - gamma) < 1e-6:
            return g, sidx, _td3_load_session(g_dir)[sidx]
    raise ValueError(f"gamma {gamma} not found")


def _td3_reconstruct_reference(g, P):
    """Iterate the ES reference update over the full stored price path."""
    from input.init import model
    from input.td3learning import init_reference, update_reference
    game = model(gamma=g, num_sessions=1, aprint=False, **_TD3_GAME_KWARGS)
    T = P.shape[1]
    r = init_reference(game, P[:, 0])
    ref = np.empty(T)
    ref[0] = r
    for t in range(1, T):
        r = update_reference(game, r, P[:, t])
        ref[t] = r
    return ref


def cmd_td3_cycles(args):
    # hand-picked clean representatives (gamma, session)
    picks_sel = {1: (0.79, 6), 2: (0.05, 22), 4: (1.52, 26), 6: (0.37, 30)}
    os.makedirs(_TD3_CYC_OUT, exist_ok=True)
    picks = {}
    for L, (gamma, sidx) in picks_sel.items():
        g, si, P = _td3_find_session(gamma, sidx)
        picks[L] = (g, si, P)

    titles = {1: "(a) Stationary outcome ($L=1$)",
              2: "(b) High-low cycle ($L=2$)",
              4: "(c) Four-period cycle ($L=4$)",
              6: "(d) Six-period cycle ($L=6$)"}
    rc = {"font.size": 15, "axes.labelsize": 15, "xtick.labelsize": 12,
          "ytick.labelsize": 12, "legend.fontsize": 13, "axes.titlesize": 15}
    C_F1, C_F2, C_REF = "#2a78d6", "black", "#d62728"

    with plt.rc_context(rc):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
        for ax, L in zip(axes.ravel(), (1, 2, 4, 6)):
            g, sidx, P = picks[L]
            ref = _td3_reconstruct_reference(g, P)
            p1, p2, rr = P[0, -_NDISP:], P[1, -_NDISP:], ref[-_NDISP:]
            x = np.arange(1, _NDISP + 1)
            ax.plot(x, p1, color=C_F1, lw=2.2, marker="o", ms=4, label="Firm 1 price")
            ax.plot(x, p2, color=C_F2, lw=2.2, marker="s", ms=4, label="Firm 2 price")
            ax.plot(x, rr, color=C_REF, lw=2.0, ls=":", label="Market reference")
            ax.set_title(f"{titles[L]},  $\\gamma={g:.2f}$")
            ax.set_xlabel("Period")
            ax.set_ylabel("Price")
            ax.set_xlim(1, _NDISP)
            if L == 1:
                c = 0.5 * (p1.mean() + p2.mean())
                ax.set_ylim(c - 0.10, c + 0.10)
            ax.grid(True, ls="--", alpha=0.4)
            print(f"L={L}: gamma={g:.4f} session={sidx}")

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3,
                   frameon=False, bbox_to_anchor=(0.5, 1.02))
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        out = os.path.join(_TD3_CYC_OUT, "td3_cycle_examples.png")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("wrote", out)


# =========================================================================== #
# 7. TD3 CYCLES, APPENDIX STYLE  (per-panel; was plot_td3_cycles_appendix_style.py)
# =========================================================================== #
_APPX_RC = {
    "font.family": "serif", "font.size": 11, "axes.titlesize": 12,
    "axes.labelsize": 11, "legend.fontsize": 10, "xtick.labelsize": 10,
    "ytick.labelsize": 10, "axes.linewidth": 1.0, "grid.linewidth": 0.5,
    "grid.alpha": 0.3, "grid.linestyle": "-", "figure.dpi": 200,
    "savefig.dpi": 300, "lines.linewidth": 2,
}


def _appx_panel(y1, y2, y_ref, out_stub, y_lims, T=_NDISP):
    import matplotlib.ticker as ticker
    c_f1, c_f2, c_ref = "#1f77b4", "#000000", "#d62728"
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    t_base = np.arange(1, T + 1)
    off_x_f1, off_x_ref, off_x_f2 = 0.00, 0.04, 0.08
    if np.ptp(y1) < 1e-5 and np.ptp(y2) < 1e-5 and np.allclose(y1, y2, atol=1e-4):
        y1p, y2p = y1 + 0.005, y2 - 0.005
    else:
        y1p, y2p = y1, y2
    l1 = ax.step(t_base + off_x_f1, y1p, where="post", color=c_f1,
                 linewidth=3.6, linestyle="-", label="Firm 1", zorder=2)[0]
    l2 = ax.step(t_base + off_x_f2, y2p, where="post", color=c_f2,
                 linewidth=3.6, linestyle="-", label="Firm 2", zorder=3)[0]
    lref = ax.step(t_base + off_x_ref, y_ref, where="post", color=c_ref,
                   linewidth=2.8, linestyle=":", label="Reference", zorder=5)[0]
    ax.set_xlabel("Period")
    ax.set_ylabel("Price")
    ax.set_xlim(1, T + 0.2)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, which="major", axis="both")
    ax.set_ylim(y_lims)
    fig.tight_layout()
    os.makedirs(_TD3_CYC_OUT, exist_ok=True)
    fig.savefig(os.path.join(_TD3_CYC_OUT, out_stub + ".png"), bbox_inches="tight")
    plt.close(fig)
    return [l1, l2, lref], ["Firm 1", "Firm 2", "Reference"]


def cmd_td3_cycles_appendix(args):
    # identical picks to plot_td3_cycles except L=2 uses a cleaner representative
    picks_sel = {1: (0.79, 6), 2: (0.37, 29), 4: (1.52, 26), 6: (0.37, 30)}
    with plt.rc_context(_APPX_RC):
        series = {}
        all_vals = []
        for L, (gamma, sidx) in picks_sel.items():
            g, si, P = _td3_find_session(gamma, sidx)
            ref = _td3_reconstruct_reference(g, P)
            y1, y2, rr = P[0, -_NDISP:], P[1, -_NDISP:], ref[-_NDISP:]
            series[L] = (y1, y2, rr, g)
            all_vals.extend(np.concatenate([y1, y2, rr]).tolist())

        g_min, g_max = float(min(all_vals)), float(max(all_vals))
        rng = g_max - g_min
        pad = 0.10 if rng < 0.01 else 0.15 * rng
        y_lims = (g_min - pad, g_max + pad)

        stubs = {1: "td3_cycle_L1", 2: "td3_cycle_L2",
                 4: "td3_cycle_L4", 6: "td3_cycle_L6"}
        legend = None
        for L in (1, 2, 4, 6):
            y1, y2, rr, g = series[L]
            handles, labels = _appx_panel(y1, y2, rr, stubs[L], y_lims)
            if legend is None:
                legend = (handles, labels)
            print(f"L={L}: gamma={g:.2f} -> {stubs[L]}.png")

        fig = plt.figure(figsize=(7.2, 0.8))
        fig.legend(*legend, loc="center", ncol=3, frameon=False,
                   handlelength=3.2, columnspacing=2.2)
        fig.tight_layout(pad=0)
        fig.savefig(os.path.join(_TD3_CYC_OUT, "td3_cycle_legend.png"),
                    bbox_inches="tight", transparent=True)
        plt.close(fig)
    print("wrote panels + legend to", _TD3_CYC_OUT)


# =========================================================================== #
# 8. LINEAR both-benchmark post-processing  (was postprocess_linear_benchmarks.py)
# =========================================================================== #
def _lin_closed_forms(g):
    """Benchmark dict for gamma g (n = 2)."""
    return dict(
        p_nash=1.0 / (3.0 + 2.0 * g),
        Pi_nash=2.0 * (1.0 + g) / (3.0 + 2.0 * g) ** 2,
        p_coop_naive=1.0 / (2.0 + g),
        Pi_coop_naive=(1.0 + g) / (2.0 + g) ** 2,
        p_mono_true=0.5, Pi_mono_true=0.25,
    )


def _lin_build_summary(exp_dir):
    rows = []
    for d in sorted(glob.glob(os.path.join(exp_dir, "gamma_*"))):
        m = re.search(r"gamma_([0-9.]+)", os.path.basename(d))
        f = os.path.join(d, "cycle_statistics.csv")
        if not m or not os.path.isfile(f):
            continue
        g = float(m.group(1))
        s = pd.read_csv(f).iloc[0]
        cf = _lin_closed_forms(g)
        profit = np.mean([s["mean_profit_p1"], s["mean_profit_p2"]])
        std_profit = np.mean([s["std_profit_p1"], s["std_profit_p2"]])
        price = np.mean([s["mean_price_p1"], s["mean_price_p2"]])
        std_price = np.mean([s["std_price_p1"], s["std_price_p2"]])
        prof_den_naive = cf["Pi_coop_naive"] - cf["Pi_nash"]
        prof_den_true = cf["Pi_mono_true"] - cf["Pi_nash"]
        price_den_naive = cf["p_coop_naive"] - cf["p_nash"]
        price_den_true = cf["p_mono_true"] - cf["p_nash"]
        rows.append(dict(
            gamma=g,
            convergence_rate=s["convergence_rate"],
            mean_cycle_length=s["mean_cycle_length"],
            std_cycle_length=s["std_cycle_length"],
            p_nash=cf["p_nash"], p_coop_naive=cf["p_coop_naive"], p_mono_true=cf["p_mono_true"],
            Pi_nash=cf["Pi_nash"], Pi_coop_naive=cf["Pi_coop_naive"], Pi_mono_true=cf["Pi_mono_true"],
            mean_price=price, std_price=std_price,
            mean_profit=profit, std_profit=std_profit,
            mean_reference_price=s["mean_reference_price"],
            profit_gain_naive=(profit - cf["Pi_nash"]) / prof_den_naive,
            std_profit_gain_naive=std_profit / prof_den_naive,
            profit_gain_true=(profit - cf["Pi_nash"]) / prof_den_true,
            std_profit_gain_true=std_profit / prof_den_true,
            price_gain_naive=(price - cf["p_nash"]) / price_den_naive,
            std_price_gain_naive=std_price / price_den_naive,
            price_gain_true=(price - cf["p_nash"]) / price_den_true,
            std_price_gain_true=std_price / price_den_true,
        ))
    return pd.DataFrame(rows).sort_values("gamma").reset_index(drop=True)


def _lin_line_plot(df, col, std_col, ylabel, title, color, out, floor=None):
    fig, ax = plt.subplots(figsize=(10, 8))
    x = df["gamma"].values
    y = df[col].values
    ax.plot(x, y, marker="o", color=color, label=ylabel)
    if std_col in df:
        lo, hi = y - df[std_col].values, y + df[std_col].values
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


def cmd_linear_postprocess(args):
    exp = args.experiment or "linear_benchmark/gamma_only_linear_full"
    exp_dir = os.path.join(RES, exp)
    df = _lin_build_summary(exp_dir)
    if df.empty:
        print(f"No gamma results found in {exp_dir}")
        return
    csv_out = os.path.join(exp_dir, "gamma_summary_both_benchmarks.csv")
    df.to_csv(csv_out, index=False)
    print(f"[{len(df)} gammas] saved {csv_out}")

    fig_dir = os.path.join(exp_dir, "Figures_both_benchmarks")
    os.makedirs(fig_dir, exist_ok=True)
    _lin_line_plot(df, "profit_gain_naive", "std_profit_gain_naive", "Profit Gain (naive coop)",
                   "Profit Gain vs Gamma  —  naive benchmark p_coop=1/(2+γ)",
                   "blue", os.path.join(fig_dir, "profit_gain_naive.png"))
    _lin_line_plot(df, "profit_gain_true", "std_profit_gain_true", "Profit Gain (true monopoly)",
                   "Profit Gain vs Gamma  —  true monopoly p=1/2, Π=1/4",
                   "navy", os.path.join(fig_dir, "profit_gain_true.png"))
    _lin_line_plot(df, "price_gain_naive", "std_price_gain_naive", "Price Gain (naive coop)",
                   "Price Gain vs Gamma  —  naive benchmark p_coop=1/(2+γ)",
                   "green", os.path.join(fig_dir, "price_gain_naive.png"))
    _lin_line_plot(df, "price_gain_true", "std_price_gain_true", "Price Gain (true monopoly)",
                   "Price Gain vs Gamma  —  true monopoly p=1/2",
                   "darkgreen", os.path.join(fig_dir, "price_gain_true.png"))
    print(f"figures saved in {fig_dir}")
    show = ["gamma", "profit_gain_naive", "profit_gain_true",
            "price_gain_naive", "price_gain_true", "mean_price", "convergence_rate"]
    print(df[show].round(3).to_string(index=False))


# =========================================================================== #
# 9. LINEAR publication-style line plots  (was paper_figures_linear.py)
# =========================================================================== #
_LINPUB_RC = {
    "font.family": "serif", "font.size": 13, "axes.titlesize": 15,
    "axes.labelsize": 14, "legend.fontsize": 11, "xtick.labelsize": 12,
    "ytick.labelsize": 12, "axes.linewidth": 1.0, "figure.dpi": 120,
    "savefig.dpi": 300,
}
_C_PROFIT, _C_PRICE, _C_NASH, _C_COOP = "#1f5fa8", "#20794d", "#b0392b", "#555555"


def _linpub_despine(ax, x):
    ax.set_xlabel(r"reference dependence  $\gamma$")
    ax.set_xlim(x.min(), x.max())
    ax.grid(axis="y", ls=":", color="0.8", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _linpub_level_plot(x, y, std, nash, coop, ylabel, title, color, out,
                       show_benchmarks=True):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, y, color=color, lw=2.2, solid_capstyle="round", zorder=3, label="learned")
    if std is not None:
        ax.fill_between(x, y - std, y + std, color=color, alpha=0.16, linewidth=0, zorder=1)
    if show_benchmarks:
        ax.plot(x, coop, color=_C_COOP, lw=1.6, ls="--", zorder=2, label="collusive (naive coop)")
        ax.plot(x, nash, color=_C_NASH, lw=1.6, ls="-.", zorder=2, label="Nash")
        ax.legend(frameon=False, loc="upper right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=10)
    _linpub_despine(ax, x)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def _linpub_gain_plot(x, y, std, ylabel, title, color, out, show_refs=True):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, y, color=color, lw=2.2, solid_capstyle="round", zorder=3)
    if std is not None:
        ax.fill_between(x, y - std, y + std, color=color, alpha=0.16, linewidth=0, zorder=1)
    if show_refs:
        ax.axhline(0.0, color=_C_NASH, ls="-.", lw=1.2, zorder=2)
        ax.axhline(1.0, color=_C_COOP, ls="--", lw=1.2, zorder=2)
        ax.text(x.max(), 0.0, "Nash  ", color=_C_NASH, va="bottom", ha="right", fontsize=10.5)
        ax.text(x.max(), 1.0, "collusive (naive coop)  ", color=_C_COOP,
                va="bottom", ha="right", fontsize=10.5)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=10)
    _linpub_despine(ax, x)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def cmd_linear_paper_figs(args):
    exp = args.experiment or "linear_benchmark/gamma_only_linear_full"
    exp_dir = os.path.join(RES, exp)
    csv = os.path.join(exp_dir, "gamma_summary_both_benchmarks.csv")
    df = pd.read_csv(csv).sort_values("gamma").reset_index(drop=True)
    x = df["gamma"].values
    out_dir = os.path.join(exp_dir, "Figures_paper")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[{exp}] {len(df)} gammas -> {out_dir}")
    with plt.rc_context(_LINPUB_RC):
        _linpub_level_plot(x, df["mean_price"].values, df["std_price"].values,
                           df["p_nash"].values, df["p_coop_naive"].values,
                           "price  $p$", "Price vs. reference dependence",
                           _C_PRICE, os.path.join(out_dir, "price.png"))
        _linpub_level_plot(x, df["mean_profit"].values, df["std_profit"].values,
                           df["Pi_nash"].values, df["Pi_coop_naive"].values,
                           "profit  $\\Pi$", "Profit vs. reference dependence",
                           _C_PROFIT, os.path.join(out_dir, "profit.png"))
        _linpub_level_plot(x, df["mean_price"].values, df["std_price"].values,
                           None, None, "price  $p$", "Price vs. reference dependence",
                           _C_PRICE, os.path.join(out_dir, "price_nobench.png"),
                           show_benchmarks=False)
        _linpub_level_plot(x, df["mean_profit"].values, df["std_profit"].values,
                           None, None, "profit  $\\Pi$", "Profit vs. reference dependence",
                           _C_PROFIT, os.path.join(out_dir, "profit_nobench.png"),
                           show_benchmarks=False)
        _linpub_gain_plot(x, df["price_gain_naive"].values, df["std_price_gain_naive"].values,
                          "price gain", "Price gain vs. reference dependence",
                          _C_PRICE, os.path.join(out_dir, "price_gain.png"), show_refs=False)
        _linpub_gain_plot(x, df["profit_gain_naive"].values, df["std_profit_gain_naive"].values,
                          "profit gain  $\\Delta$", "Profit gain vs. reference dependence",
                          _C_PROFIT, os.path.join(out_dir, "profit_gain.png"))


# =========================================================================== #
# 10. ALT-STYLING 4-PANEL FIGURES  (was paper_panels.py)
# =========================================================================== #
_Z = 1.96                                  # 95% CI multiplier
_PANELS_CURVE = "tab:blue"
_PANELS = [
    ("Price", "figure1_gamma_only_q_price.png"),
    ("Profit", "figure1_gamma_only_q_profit.png"),
    ("Price Gain", "figure1_gamma_only_q_price_gain.png"),
    ("Profit Gain", "figure1_gamma_only_q_profit_gain.png"),
]


def _panels_panel(g, mu, band, ylabel, out):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(g, mu, color=_PANELS_CURVE, linestyle="-", marker=" ", linewidth=2)
    if band is not None:
        ax.fill_between(g, mu - band, mu + band, color=_PANELS_CURVE, alpha=0.15)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


def _panels_infer_n(exp_dir, default):
    for cs in glob.glob(os.path.join(exp_dir, "gamma_*", "cycle_statistics.csv")):
        try:
            return int(pd.read_csv(cs).iloc[0]["num_sessions"])
        except Exception:
            pass
    return default


def _panels_make_linear(exp, out_dir):
    csv = os.path.join(RES, exp, "gamma_summary_both_benchmarks.csv")
    df = pd.read_csv(csv).sort_values("gamma").reset_index(drop=True)
    g = df["gamma"].values
    n = _panels_infer_n(os.path.join(RES, exp), default=200)
    se = lambda s: s / np.sqrt(n)
    series = [
        ("Price", df["mean_price"].values, se(df["std_price"].values)),
        ("Profit", df["mean_profit"].values, se(df["std_profit"].values)),
        ("Price Gain", df["price_gain_true"].values, se(df["std_price_gain_true"].values)),
        ("Profit Gain", df["profit_gain_true"].values, se(df["std_profit_gain_true"].values)),
    ]
    os.makedirs(out_dir, exist_ok=True)
    print(f"[linear {exp}] {len(g)} gammas, n={n} -> {out_dir}")
    with plt.rc_context(RC):
        for (ylabel, mu, sb), (_, fname) in zip(series, _PANELS):
            _panels_panel(g, mu, _Z * sb, ylabel, os.path.join(out_dir, fname))


def _panels_make_td3(exp, out_dir):
    exp_dir = os.path.join(RES, exp)
    rows = []
    for d in glob.glob(os.path.join(exp_dir, "gamma_*")):
        cs = os.path.join(d, "cycle_statistics.csv")
        if not os.path.isfile(cs):
            continue
        r = pd.read_csv(cs).iloc[0]
        m = re.search(r"gamma_([0-9.]+)", os.path.basename(d))
        g = float(m.group(1)) if m else np.nan
        n = float(r["num_sessions"])
        avg = lambda a, b: 0.5 * (float(r[a]) + float(r[b]))
        std2 = lambda a, b: np.sqrt(0.5 * (float(r[a]) ** 2 + float(r[b]) ** 2))
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
        ("Price", df["price"].values, se(df["price_sd"].values)),
        ("Profit", df["profit"].values, se(df["profit_sd"].values)),
        ("Price Gain", df["pgain"].values, se(df["pgain_sd"].values)),
        ("Profit Gain", df["prgain"].values, se(df["prgain_sd"].values)),
    ]
    os.makedirs(out_dir, exist_ok=True)
    print(f"[td3 {exp}] {len(g)} gammas, n={int(df['n'].iloc[0])} -> {out_dir}")
    with plt.rc_context(RC):
        for (ylabel, mu, sb), (_, fname) in zip(series, _PANELS):
            _panels_panel(g, mu, _Z * sb, ylabel, os.path.join(out_dir, fname))


def cmd_panels(args):
    if args.mode == "linear":
        _panels_make_linear(args.experiment, args.out_dir)
    else:
        _panels_make_td3(args.experiment, args.out_dir)


# =========================================================================== #
# Convenience: run the standard committed-figure set (best effort)
# =========================================================================== #
def cmd_all(args):
    steps = [
        ("altbench", lambda: cmd_altbench(argparse.Namespace(new=False))),
        ("altbench --new", lambda: cmd_altbench(argparse.Namespace(new=True))),
        ("rebuild", lambda: cmd_rebuild(args)),
        ("recolor-linear-td3", lambda: cmd_recolor_linear_td3(args)),
        ("linear-gains-longterm", lambda: cmd_linear_gains_longterm(args)),
        ("irf", lambda: cmd_irf(args)),
        ("td3-cycles", lambda: cmd_td3_cycles(args)),
        ("td3-cycles-appendix", lambda: cmd_td3_cycles_appendix(args)),
    ]
    for name, fn in steps:
        print(f"\n=== {name} ===")
        try:
            fn()
        except Exception as e:
            print(f"[warn] {name} failed: {e}")


# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(
        description="Unified figure generator for the reference-dependence paper.")
    sub = p.add_subparsers(dest="command", required=True)

    ab = sub.add_parser("altbench", help="alternative-benchmark gain panels + heatmaps")
    ab.add_argument("--new", action="store_true",
                    help="appendix ES line figs + separated profit-gain heatmaps")
    ab.set_defaults(func=cmd_altbench)

    sub.add_parser("irf", help="deviation/punishment panels + LaTeX table rows"
                   ).set_defaults(func=cmd_irf)
    sub.add_parser("recolor-linear-td3", help="linear + TD3 purple 4-panel figures"
                   ).set_defaults(func=cmd_recolor_linear_td3)
    sub.add_parser("linear-gains-longterm", help="linear gain panels, long-term benchmark"
                   ).set_defaults(func=cmd_linear_gains_longterm)
    sub.add_parser("rebuild", help="benchmark/market/misspec/firm-specific/lossaversion blocks"
                   ).set_defaults(func=cmd_rebuild)
    sub.add_parser("td3-cycles", help="Fig td3_cycles combined 2x2 panel"
                   ).set_defaults(func=cmd_td3_cycles)
    sub.add_parser("td3-cycles-appendix", help="appendix-style per-panel TD3 cycles + legend"
                   ).set_defaults(func=cmd_td3_cycles_appendix)

    lp = sub.add_parser("linear-postprocess", help="both-benchmark summary CSV + diagnostic plots")
    lp.add_argument("experiment", nargs="?", default=None)
    lp.set_defaults(func=cmd_linear_postprocess)

    lf = sub.add_parser("linear-paper-figs", help="publication-style linear line plots")
    lf.add_argument("experiment", nargs="?", default=None)
    lf.set_defaults(func=cmd_linear_paper_figs)

    pn = sub.add_parser("panels", help="alt-styling 4-panel figures (linear/td3)")
    pn.add_argument("mode", choices=["linear", "td3"])
    pn.add_argument("experiment")
    pn.add_argument("out_dir")
    pn.set_defaults(func=cmd_panels)

    sub.add_parser("all", help="run the standard committed-figure set (best effort)"
                   ).set_defaults(func=cmd_all)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
