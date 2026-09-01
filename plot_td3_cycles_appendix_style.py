"""Render TD3 price cycles in the EXACT style of Appendix B.1 (tabular cycles).

Same hand-picked TD3 representatives as ``plot_td3_cycles.py`` (L = 1, 2, 4, 6),
but drawn as step plots with no markers, one PNG per panel plus a shared legend,
matching ``creating_results.ipynb`` cell 95 (the appendix_cycles figure):
Firm 1 (blue solid), Firm 2 (black solid), market reference (red dotted).

Run:  /Users/neda/llm_venv/bin/python plot_td3_cycles_appendix_style.py
"""
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from input.init import model
from input.td3learning import init_reference, update_reference

ROOT = "/Users/neda/Desktop/UBC/PHD/research_term_4"
EXP = os.path.join(ROOT, "Results", "experiments",
                   "td3_production_reference_15g_50s_lr1e-4")
OUT_DIR = os.path.join(ROOT, "Algorithmic-Collusion-Replication", "paper_overleaf",
                       "Images", "4_seperate_figures_lr1e-4", "td3_cycles")

GAME_KWARGS = dict(n=2, k=15, memory=1, demand_type="reference",
                   common_reference=True, lossaversion=1)

NDISP = 12          # periods to display

# hand-picked clean representatives (gamma, session), identical to plot_td3_cycles.py
PICKS = {1: (0.79, 6), 2: (0.37, 29), 4: (1.52, 26), 6: (0.37, 30)}


# ---- appendix B.1 styling (creating_results.ipynb cell 95) -----------------
def set_paper_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 1.0,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.3,
        "grid.linestyle": "-",
        "figure.dpi": 200,
        "savefig.dpi": 300,
        "lines.linewidth": 2,
    })


def load_session(g_dir):
    npz = np.load(os.path.join(g_dir, "rollout_paths.npz"))
    keys = [k for k in npz.files if k.startswith("prices_s")]
    return [npz[k] for k in sorted(keys, key=lambda s: int(s.split("s")[-1]))]


def find_session(gamma, sidx):
    for g_dir in glob.glob(os.path.join(EXP, "gamma_*")):
        g = float(os.path.basename(g_dir).split("gamma_")[1])
        if abs(round(g, 2) - gamma) < 1e-6:
            return g, sidx, load_session(g_dir)[sidx]
    raise ValueError(f"gamma {gamma} not found")


def reconstruct_reference(g, P):
    game = model(gamma=g, num_sessions=1, aprint=False, **GAME_KWARGS)
    T = P.shape[1]
    r = init_reference(game, P[:, 0])
    ref = np.empty(T)
    ref[0] = r
    for t in range(1, T):
        r = update_reference(game, r, P[:, t])
        ref[t] = r
    return ref


def plot_panel(y1, y2, y_ref, out_stub, y_lims, T=NDISP):
    c_f1, c_f2, c_ref = "#1f77b4", "#000000", "#d62728"
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    t_base = np.arange(1, T + 1)
    off_x_f1, off_x_ref, off_x_f2 = 0.00, 0.04, 0.08

    # keep coincident flat series visible (same trick as appendix)
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

    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, out_stub + ".png"), bbox_inches="tight")
    plt.close(fig)
    return [l1, l2, lref], ["Firm 1", "Firm 2", "Reference"]


def save_global_legend(handles, labels):
    fig = plt.figure(figsize=(7.2, 0.8))
    fig.legend(handles, labels, loc="center", ncol=3, frameon=False,
               handlelength=3.2, columnspacing=2.2)
    fig.tight_layout(pad=0)
    fig.savefig(os.path.join(OUT_DIR, "td3_cycle_legend.png"),
                bbox_inches="tight", transparent=True)
    plt.close(fig)


def main():
    set_paper_style()
    # gather the four displayed panels
    series = {}
    all_vals = []
    for L, (gamma, sidx) in PICKS.items():
        g, si, P = find_session(gamma, sidx)
        ref = reconstruct_reference(g, P)
        y1, y2, rr = P[0, -NDISP:], P[1, -NDISP:], ref[-NDISP:]
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
        handles, labels = plot_panel(y1, y2, rr, stubs[L], y_lims)
        if legend is None:
            legend = (handles, labels)
        print(f"L={L}: gamma={g:.2f} -> {stubs[L]}.png")

    save_global_legend(*legend)
    print("wrote panels + legend to", OUT_DIR)


if __name__ == "__main__":
    main()
