"""Build clean, paper-style impulse-response (deviation/punishment) figures for
the Deviations & Punishments subsection, matching the paper's multi-panel
convention: each panel is a SEPARATE png (no baked-in title/legend), plus a
SHARED legend png placed at the top of the figure in LaTeX, and (a)/(b)
subcaptions added in the .tex.

Reads the ALREADY-COMPUTED IRF npz files (no retraining, no new simulations):
  tabular (corrected config, beta=4e-6, continuous ES reference, 50 sessions):
    ../Results/experiments/gamma_nloss_reference_True_beta4e-6_ESref/Figures/
        irf_gamma_<g>_dev-<dt>.npz
  td3 (continuous-action learner):
    ../Results/experiments/impulse_response/irf_td3_gamma_<g>_dev-<dt>.npz

Outputs (Final_Paper.../Images/impulse_response/):
  irf_legend.png         shared legend (2 firm curves + 3 benchmark lines)
  irf_mechanism_a.png    gamma=1.07, static-BR deviation
  irf_mechanism_b.png    gamma=1.07, Nash-price deviation
  irf_by_gamma_a.png     gamma=0.05, Nash-price deviation
  irf_by_gamma_b.png     gamma=3.0,  Nash-price deviation
  and prints the LaTeX rows for the summary tables.

Run:  /Users/neda/llm_venv/bin/python paper_irf_figures.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

TAB = ("../Results/experiments/gamma_nloss_reference_True_beta4e-6_ESref/"
       "Figures")
TD3 = "../Results/experiments/impulse_response"
OUT = ("/Users/neda/Desktop/UBC/PHD/research_term_4/Algorithmic-Collusion-"
       "Replication/Final_Paper__Reference_Dependence__Copy2_/Images/"
       "impulse_response")

TAB_GAMMAS = ["0.05", "1.0672", "2.0845", "3.0"]
TD3_GAMMAS = ["0.05", "1.1036", "2.1571", "3.0"]
REP = "1.0672"          # representative gamma for the mechanism figure

S_DEV, S_NON = "#c02f2f", "#2a5fb0"      # deviator (red), non-deviator (blue)
MUT, BASE = "#8a8a8a", "#b8b8b8"
LS_MONO, LS_NASH, LS_PRE = (0, (1, 2)), (0, (4, 3)), "-"   # dotted / dashed / solid

plt.rcParams.update({
    "font.size": 15, "axes.labelsize": 16, "axes.titlesize": 16,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 14,
})


def _load(dirpath, g, dt, prefix="irf"):
    return dict(np.load(os.path.join(dirpath,
                                     f"{prefix}_gamma_{g}_dev-{dt}.npz")))


def _panel(res, out, ylim=None):
    """One standalone panel: no title, no legend, no annotations."""
    t = np.arange(len(res["dev_price"]))
    fig, ax = plt.subplots(figsize=(5.6, 4.3))
    ax.axhline(float(res["p_coop"]), color=MUT, lw=1.1, ls=LS_MONO, zorder=1)
    ax.axhline(float(res["p_nash"]), color=MUT, lw=1.1, ls=LS_NASH, zorder=1)
    ax.axhline(float(res["long_run"]), color=BASE, lw=1.2, ls=LS_PRE, zorder=1)
    ax.axvline(1, color="#d8d8d8", lw=1.0, zorder=0)
    ax.plot(t, res["dev_price"], color=S_DEV, lw=2.0, marker="o", ms=4.5,
            zorder=3)
    ax.plot(t, res["nondev_price"], color=S_NON, lw=2.0, marker="^", ms=4.5,
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


def _shared_legend(out):
    handles = [
        Line2D([], [], color=S_DEV, lw=2.0, marker="o", ms=6,
               label="Deviating firm"),
        Line2D([], [], color=S_NON, lw=2.0, marker="^", ms=6, ls="--",
               label="Non-deviating (rival) firm"),
        Line2D([], [], color=BASE, lw=1.4, ls=LS_PRE,
               label="Pre-deviation (collusive) price"),
        Line2D([], [], color=MUT, lw=1.4, ls=LS_NASH, label="Nash price"),
        Line2D([], [], color=MUT, lw=1.4, ls=LS_MONO, label="Monopoly price"),
    ]
    fig = plt.figure(figsize=(12, 0.6))
    fig.legend(handles=handles, loc="center", ncol=5, frameon=False,
               handlelength=2.6, columnspacing=1.6)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def _common_ylim(*ress, pad=0.03):
    ys = []
    for r in ress:
        ys += [float(r["dev_price"].min()), float(r["dev_price"].max()),
               float(r["nondev_price"].min()), float(r["nondev_price"].max()),
               float(r["p_nash"]), float(r["p_coop"])]
    lo, hi = min(ys), max(ys)
    m = pad * (hi - lo)
    return lo - m, hi + m


def figures():
    # mechanism: same gamma, both deviation depths -> share y-limits
    br, na = _load(TAB, REP, "br"), _load(TAB, REP, "nash")
    yl = _common_ylim(br, na)
    _panel(br, os.path.join(OUT, "irf_mechanism_a.png"), ylim=yl)
    _panel(na, os.path.join(OUT, "irf_mechanism_b.png"), ylim=yl)
    # by-gamma: different gamma -> independent y-limits
    _panel(_load(TAB, "0.05", "nash"), os.path.join(OUT, "irf_by_gamma_a.png"))
    _panel(_load(TAB, "3.0", "nash"), os.path.join(OUT, "irf_by_gamma_b.png"))
    _shared_legend(os.path.join(OUT, "irf_legend.png"))


def table_rows():
    def row(dirpath, g, prefix):
        b = _load(dirpath, g, "br", prefix)
        n = _load(dirpath, g, "nash", prefix)
        return (float(g), float(b["frac_unprofitable"]) * 100,
                float(n["frac_unprofitable"]) * 100, int(b["n_obs"]))
    print("\n% ---- TABULAR rows ----")
    for g in TAB_GAMMAS:
        gm, ubr, un, n = row(TAB, g, "irf")
        print(f"${gm:.2f}$ & {ubr:.1f}\\% & {un:.1f}\\% & {n} \\\\")
    print("\n% ---- TD3 rows ----")
    for g in TD3_GAMMAS:
        gm, ubr, un, n = row(TD3, g, "irf_td3")
        print(f"${gm:.2f}$ & {ubr:.1f}\\% & {un:.1f}\\% & {n} \\\\")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    figures()
    table_rows()
