"""
Alternative-benchmark Price/Profit Gain figures.

For each target paper figure we regenerate ONLY the two gain panels using an
alternative normalization:

    * SAME Nash benchmark (gamma-dependent, falls with gamma)
    * COLLUSIVE benchmark FROZEN at its gamma=0 value (constant across gamma)

    price_gain_new(g)  = (mean_price(g)  - Pnash(g)) / (Pcoop(0)  - Pnash(g))
    profit_gain_new(g) = (mean_profit(g) - PInash(g)) / (PIcoop(0) - PInash(g))

Because the stored gains already use the standard benchmark
    gain_std(g) = (x(g) - nash(g)) / (coop(g) - nash(g)),
the alternative gain is an exact RESCALE of the stored gain (mean AND std):
    gain_new(g) = gain_std(g) * R(g),   R(g) = (coop(g)-nash(g)) / (coop(0)-nash(g)).
This guarantees the new gain panels are numerically consistent with the
existing price/profit panels and the existing (standard) gain panels.

Profit benchmarks are recomputed from the Nash/Coop PRICES via the symmetric
steady-state logit profit  pi(p) = (p-c)*e/(n*e + exp(a0/mu)),  e=exp((a-p)/mu),
which reproduces every stored benchmark exactly (validated).

Outputs are written next to the originals with a `_altbench` suffix.

Run:  /Users/neda/llm_venv/bin/python gen_altbench_gains.py
"""
import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors

PR  = "/Users/neda/Desktop/UBC/PHD/research_term_4/paper_results"
RES = "/Users/neda/Desktop/UBC/PHD/research_term_4/Results/experiments"
IMG = ("/Users/neda/Desktop/UBC/PHD/research_term_4/Algorithmic-Collusion-"
       "Replication/Final_Paper__Reference_Dependence__Copy2_/Images")

RC = {"font.size": 18, "axes.labelsize": 18, "xtick.labelsize": 16,
      "ytick.labelsize": 16, "legend.fontsize": 14}

GAIN_METRICS = [("price_gain", "Price Gain", "figure1_gamma_only_q_price_gain_altbench.png"),
                ("profit_gain", "Profit Gain", "figure1_gamma_only_q_profit_gain_altbench.png")]


# ---------------------------------------------------------------- helpers ----
def ff(s):
    """First float in a string like '[1.45 1.45]'."""
    return float(re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(s))[0])


def read_csv_retry(path, tries=5):
    """pd.read_csv with retries -- guards against transient disk I/O timeouts."""
    import time
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
    a  = float(cfg_row["a1"]) if "a1" in cfg_row else 2.0
    c  = float(cfg_row["c1"]) if "c1" in cfg_row else 1.0
    mu = float(cfg_row["mu"]) if "mu" in cfg_row else 0.25
    a0 = float(cfg_row["a0"]) if "a0" in cfg_row else 0.0
    return a, c, mu, a0


def benchmarks_gamma_only(root):
    """
    root/gamma_*/config.csv  ->  dict metric -> (gammas, nash, coop) arrays,
    where metric in {'price_gain','profit_gain'}. Prices from Pnash/Pcoop;
    profits from profit_from_price with per-folder (a,c,mu,a0).
    """
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


def rescale_factor(gammas, nash, coop):
    """R(g) = (coop(g)-nash(g)) / (coop(0)-nash(g)); coop(0) = coop at min gamma."""
    coop0 = coop[np.argmin(gammas)]
    return (coop - nash) / (coop0 - nash)


# ---------------------------------------------- per-curve series loaders -----
def series_gamma_only(root, gain_key):
    """gammas, firm-avg stored gain mean, firm-avg stored gain std (gamma-only)."""
    g, mu, sd = [], [], []
    for d in sorted(glob.glob(os.path.join(root, "gamma_*"))):
        f = os.path.join(d, "cycle_statistics.csv")
        if not os.path.exists(f):
            continue
        r = read_csv_retry(f).iloc[0]
        g.append(ff(os.path.basename(d).split("gamma_")[1]))
        mu.append(np.mean([r[f"mean_{gain_key}_p1"], r[f"mean_{gain_key}_p2"]]))
        sd.append(np.mean([r[f"std_{gain_key}_p1"],  r[f"std_{gain_key}_p2"]]))
    idx = np.argsort(g)
    return np.array(g)[idx], np.array(mu)[idx], np.array(sd)[idx]


def series_grid_pooled(root, gain_key):
    """
    gammas, mean (avg over lambda cells), pooled std, n_eff  -- mirrors the
    notebook load_gamma_metric_n fallback (no per-cell counts).
    """
    rows = []
    for d in glob.glob(os.path.join(root, "gamma_*_lambda_*")):
        f = os.path.join(d, "cycle_statistics.csv")
        if not os.path.exists(f):
            continue
        base = os.path.basename(d)
        gamma = float(base.split("gamma_")[1].split("_lambda_")[0])
        r = read_csv_retry(f).iloc[0]
        m = np.mean([r[f"mean_{gain_key}_p1"], r[f"mean_{gain_key}_p2"]])
        s = np.mean([r[f"std_{gain_key}_p1"],  r[f"std_{gain_key}_p2"]])
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


def benchmarks_grid(root):
    """Per-gamma benchmarks from one lambda cell per gamma (indep. of lambda)."""
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


# ----------------------------------------------------------- line figures ----
def build_line_figure(name, out_dir, curves, legend_curves=None):
    """
    curves: list of dicts with keys:
      root, kind ('gonly'|'grid'|'td3'), color, ls, lw, band ('sigma'|'se', mult), alpha
    Emits <out_dir>/figure1_gamma_only_q_{price,profit}_gain_altbench.png
    """
    os.makedirs(out_dir, exist_ok=True)
    with plt.rc_context(RC):
        for gain_key, ylabel, fname in GAIN_METRICS:
            fig, ax = plt.subplots(figsize=(6, 4))
            for cv in curves:
                root = cv["root"]
                if cv["kind"] == "grid":
                    g, mu, sd, n_eff = series_grid_pooled(root, gain_key)
                    bench = benchmarks_grid(root)[gain_key]
                else:  # gonly / td3
                    g, mu, sd = series_gamma_only(root, gain_key)
                    n_eff = None
                    if cv["kind"] == "td3":
                        bench = td3_benchmarks(root)[gain_key]
                    else:
                        bench = benchmarks_gamma_only(root)[gain_key]
                # align benchmark gammas to series gammas
                bg, nash, coop = bench
                R = rescale_factor(bg, nash, coop)
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


def td3_benchmarks(root):
    """TD3: price benchmarks from p_nash/p_coop in cycle_statistics; profit from model."""
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


# -------------------------------------------------- gamma-lambda heatmaps ----
def build_gamma_lambda_heatmaps(root, out_dir):
    """Fig 19 style (PuRd/OrRd, ylabel gamma, xlabel lambda, no title)."""
    os.makedirs(out_dir, exist_ok=True)
    cfg = {"price_gain": ("PuRd", "gamma_lambda_price_gain_altbench.png"),
           "profit_gain": ("OrRd", "gamma_lambda_profit_gain_altbench.png")}
    bench_all = benchmarks_grid(root)
    rc = {**RC, "font.family": "serif", "mathtext.fontset": "dejavuserif",
          "xtick.labelsize": 16, "ytick.labelsize": 16}
    for gain_key, (cmap, fname) in cfg.items():
        # collect per-cell values
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
        # rescale each gamma row
        bg, nash, coop = bench_all[gain_key]
        R = rescale_factor(bg, nash, coop)
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


# --------------------------------------------------- gamma-delta heatmaps ----
def build_gamma_delta_heatmaps(root, out_dir):
    """Fig 18 style (Reds, title, xlabel gamma, ylabel delta, figsize (10,8))."""
    os.makedirs(out_dir, exist_ok=True)
    cfg = {"price_gain": ("Price Gain", "price_gain_heatmap_altbench.png"),
           "profit_gain": ("Profit Gain", "profit_gain_heatmap_altbench.png")}
    # per-gamma benchmarks (independent of delta)
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


# ------------------------------- separated aware/naive (cr) profit-gain -----
def _load_grid_gl(root, gain_key):
    """gammas, lambdas, grid[gamma,lambda] of firm-avg stored gain, rescaled by
    R(gamma) (frozen collusive benchmark at gamma=0)."""
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
    bg, nash, coop = benchmarks_grid(root)[gain_key]
    R = rescale_factor(bg, nash, coop)
    Rmap = {round(float(x), 6): rr for x, rr in zip(bg, R)}
    rr = np.array([Rmap[round(float(x), 6)] for x in gammas])
    return gammas, lams, grid * rr[:, None]


def _heatmap_ax(ax, gammas, lams, grid, title, cmap, norm):
    """Mirror the notebook plot_heatmap_on_ax (extent-centred, contours, ticks)."""
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


def build_separated_heatmaps(out_dir, root_a, root_b, titles, fnames, figsize):
    """Profit-gain aware/naive(/cr) heatmaps + diff (=b-a), alt benchmark.
    OrRd for the two absolute panels (shared norm); BrBG diverging for diff."""
    os.makedirs(out_dir, exist_ok=True)
    gk = "profit_gain"
    ga, la, grid_a = _load_grid_gl(root_a, gk)
    gb, lb, grid_b = _load_grid_gl(root_b, gk)
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
            im = _heatmap_ax(ax, g_all, l_all, grid, title, cmap, norm)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.savefig(os.path.join(out_dir, fname), dpi=300, bbox_inches="tight")
            plt.close(fig)
    print(f"  [heatmap] separated: wrote 3 altbench panels -> {out_dir}")


# ----------------------------------------------------------------- config ----
def C(root, color, ls, lw, band, alpha, kind="gonly", label=""):
    return dict(root=root, color=color, ls=ls, lw=lw, band=band, alpha=alpha,
                kind=kind, label=label)


LINE_FIGS = {
    # Fig 3
    "benchmark": (f"{IMG}/4_seperate_figures/benchmark", [
        C(f"{PR}/benchmark_figure/gamma_nloss_only_reference_True",
          "tab:purple", "-", 2.0, ("sigma", 0.8), 0.15)]),
    # Fig 4
    "market_structure": (f"{IMG}/4_seperate_figures/market_structure", [
        C(f"{PR}/market_structure/gamma_nloss_only_reference_True",
          "tab:purple", "-", 3.0, ("sigma", 0.8), 0.18),
        C(f"{PR}/market_structure/gamma_nloss_only_reference_Truec_0",
          "tab:purple", (0, (6, 3)), 2.2, ("sigma", 0.8), 0.12),
        C(f"{PR}/market_structure/gamma_nloss_only_reference_Truemu_0",
          "tab:purple", (0, (1, 2)), 2.2, ("sigma", 0.8), 0.12)]),
    # Fig 7
    "misspecification": (f"{IMG}/4_seperate_figures/misspecification", [
        C(f"{PR}/misspecification/gamma_nloss_only_reference_True",
          "tab:purple", "-", 2.0, ("sigma", 0.8), 0.15),
        C(f"{PR}/misspecification/gamma_nloss_only_misspecification_True",
          "tab:red", "-", 2.0, ("sigma", 0.8), 0.15)]),
    # Fig 9
    "Firm-specific": (f"{IMG}/4_seperate_figures/Firm-specific", [
        C(f"{PR}/Firm_specific/gamma_nloss_only_reference_True",
          "tab:purple", "-", 2.0, ("sigma", 0.8), 0.15),
        C(f"{PR}/Firm_specific/gamma_nloss_only_reference_False",
          "tab:purple", "--", 2.0, ("sigma", 0.8), 0.15)]),
    # Fig 10 (exp smoothing blue grid + Q-learning purple gamma-only)
    "exp_smooth": (f"{IMG}/4_seperate_figures/exp_smooth", [
        C(f"{PR}/exp_smoothing/gamma_lambda_reference_True",
          "tab:blue", "-", 2.0, ("se", 1.8), 0.15, kind="grid"),
        C(f"{PR}/qqlearning/gamma_nloss_only_reference_True",
          "tab:purple", "-", 2.0, ("sigma", 0.8), 0.15)]),
    # Fig 12 (TD3, lr 1e-4)
    "td3": (f"{IMG}/4_seperate_figures_lr1e-4/td3", [
        C(f"{RES}/td3_production_reference_15g_50s_lr1e-4",
          "tab:purple", "-", 2.0, ("sigma", 0.8), 0.15, kind="td3")]),
}

# Line figures added later (Figs 27 & 28): exponential-smoothing gamma-only
# curves (lambda-averaged grid data, SE band 1.8*sigma/sqrt(n_eff)).
LINE_FIGS_NEW = {
    # Fig 27: ES misspecification -- reference-aware (blue) vs naive (orange)
    "exp_smoothing_misspecification": (
        f"{IMG}/4_seperate_figures/exp_smoothing_misspecification", [
            C(f"{PR}/exp_smoothing_misspecification/gamma_lambda_reference_True",
              "tab:blue", "-", 2.0, ("se", 1.8), 0.15, kind="grid"),
            C(f"{PR}/exp_smoothing_misspecification/gamma_lambda_misspecification_True",
              "tab:orange", "-", 2.0, ("se", 1.8), 0.15, kind="grid")]),
    # Fig 28: ES firm-specific -- CR=True (solid) vs CR=False (dashed), both blue
    "exp_smoothing_firmspecific": (
        f"{IMG}/4_seperate_figures/exp_smoothing_firmspecific", [
            C(f"{PR}/exp_smoothing_firm_specific/gamma_lambda_reference_True",
              "tab:blue", "-", 2.0, ("se", 1.8), 0.15, kind="grid"),
            C(f"{PR}/exp_smoothing_firm_specific/gamma_lambda_reference_False",
              "tab:blue", "--", 2.0, ("se", 1.8), 0.15, kind="grid")]),
}


def run_new():
    """Figs 27, 28 (ES line) and 29, 30 (separated profit-gain heatmaps)."""
    print("Generating alternative-benchmark gain figures (Figs 27-30)...")
    for name, (out_dir, curves) in LINE_FIGS_NEW.items():
        build_line_figure(name, out_dir, curves)
    build_separated_heatmaps(
        f"{IMG}/4_seperate_figures/Separated_Panels_miss",
        f"{PR}/Separated_Panels_miss/gamma_lambda_reference_True",
        f"{PR}/Separated_Panels_miss/gamma_lambda_misspecification_True",
        ("Profit Gain\nReference-Aware", "Profit Gain\nReference-Naive",
         "Profit Gain\nDifference (Naive − Aware)"),
        ("profit_gain_aware_altbench.png", "profit_gain_naive_altbench.png",
         "profit_gain_difference_altbench.png"),
        (7, 6))
    build_separated_heatmaps(
        f"{IMG}/4_seperate_figures/Seperated_Panels_CR",
        f"{PR}/Separated_Panels_CR_True_vs_False/gamma_lambda_reference_True",
        f"{PR}/Separated_Panels_CR_True_vs_False/gamma_lambda_reference_False",
        ("Profit Gain\nReference-Aware (CR=True)", "Profit Gain\nReference-Aware (CR=False)",
         "Profit Gain\nDiff. (CR=False − CR=True)"),
        ("profit_gain_cr_true_altbench.png", "profit_gain_cr_false_altbench.png",
         "profit_gain_cr_diff_altbench.png"),
        (6, 4))
    print("done")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "new":
        run_new()
    else:
        print("Generating alternative-benchmark gain figures...")
        for name, (out_dir, curves) in LINE_FIGS.items():
            build_line_figure(name, out_dir, curves)
        build_gamma_lambda_heatmaps(f"{PR}/gamma_lambda/gamma_lambda_reference_True",
                                    f"{IMG}/4_seperate_figures/gamma_lambda")
        build_gamma_delta_heatmaps(f"{RES}/gamma_delta/gamma_delta_reference_True_contref",
                                   f"{IMG}/gamma_delta")
        print("done")
