"""Sample TD3 price cycles, styled like the tabular Figure 13.

Scans the TD3 production rollouts, auto-selects the cleanest representative of
each target cycle length (L = 1 stationary, 2 high-low, 4 and 6 longer cycles),
reconstructs the exponential-smoothing market reference from the stored price
path via the actual ``td3learning`` update, and renders a 2x2 panel:
Firm 1 price (blue solid), Firm 2 price (black solid), market reference
(red dotted), over 12 periods with the detected cycle repeated as needed.

Run:  /Users/neda/llm_venv/bin/python plot_td3_cycles.py
"""
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from input.init import model
from input.td3learning import init_reference, update_reference

ROOT = "/Users/neda/Desktop/UBC/PHD/research_term_4"
EXP = os.path.join(ROOT, "Results", "experiments",
                   "td3_production_reference_15g_50s_lr1e-4")
OUT_DIR = os.path.join(ROOT, "Algorithmic-Collusion-Replication",
                       "Final_Paper__Reference_Dependence__Copy2_", "Images",
                       "4_seperate_figures_lr1e-4", "td3_cycles")

GAME_KWARGS = dict(n=2, k=15, memory=1, demand_type="reference",
                   common_reference=True, lossaversion=1)

TAIL = 200          # window assumed converged/periodic
TOL = 0.01          # relative tolerance for period match
NDISP = 12          # periods to display


def detect_period(P, max_L=12):
    """Smallest L in 1..max_L such that the tail is L-periodic to TOL."""
    W = P[:, -TAIL:]
    scale = max(W.max() - W.min(), 1e-9)
    for L in range(1, max_L + 1):
        a, b = W[:, L:], W[:, :-L]
        if np.max(np.abs(a - b)) / scale < TOL:
            return L
    return None


def load_session(g_dir):
    npz = np.load(os.path.join(g_dir, "rollout_paths.npz"))
    keys = [k for k in npz.files if k.startswith("prices_s")]
    return [npz[k] for k in sorted(keys, key=lambda s: int(s.split("s")[-1]))]


def scan():
    """Return {L: list of (score, gamma, sidx, P)} candidates."""
    cands = {L: [] for L in (1, 2, 4, 6)}
    for g_dir in sorted(glob.glob(os.path.join(EXP, "gamma_*"))):
        try:
            g = float(os.path.basename(g_dir).split("gamma_")[1])
        except ValueError:
            continue
        if not os.path.exists(os.path.join(g_dir, "rollout_paths.npz")):
            continue
        for sidx, P in enumerate(load_session(g_dir)):
            L = detect_period(P)
            if L not in cands:
                continue
            W = P[:, -TAIL:]
            amp = float(W.max() - W.min())
            lvl = float(W.mean())
            if L == 1:
                # clean stationary: symmetric (small firm gap) and high level
                gap = abs(P[0, -1] - P[1, -1])
                score = lvl - 5.0 * gap
            else:
                # clear, well-formed cycle at a healthy price level
                score = amp + 0.5 * lvl
            cands[L].append((score, g, sidx, P))
    for L in cands:
        cands[L].sort(key=lambda t: -t[0])
    return cands


def reconstruct_reference(g, P):
    """Iterate the ES reference update over the full stored price path."""
    game = model(gamma=g, num_sessions=1, aprint=False, **GAME_KWARGS)
    T = P.shape[1]
    r = init_reference(game, P[:, 0])
    ref = np.empty(T)
    ref[0] = r
    for t in range(1, T):
        r = update_reference(game, r, P[:, t])
        ref[t] = r
    return ref, game


# hand-picked clean representatives (gamma, session), chosen from the gallery
# so cycles oscillate within [Nash, monopoly] rather than pinning the ceiling
PICKS = {1: (0.79, 6), 2: (0.05, 22), 4: (1.52, 26), 6: (0.37, 30)}


def find_session(gamma, sidx):
    for g_dir in glob.glob(os.path.join(EXP, "gamma_*")):
        g = float(os.path.basename(g_dir).split("gamma_")[1])
        if abs(round(g, 2) - gamma) < 1e-6:
            P = load_session(g_dir)[sidx]
            return g, sidx, P
    raise ValueError(f"gamma {gamma} not found")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    picks = {}
    for L, (gamma, sidx) in PICKS.items():
        g, si, P = find_session(gamma, sidx)
        picks[L] = (0.0, g, si, P)

    titles = {1: "(a) Stationary outcome ($L=1$)",
              2: "(b) High-low cycle ($L=2$)",
              4: "(c) Four-period cycle ($L=4$)",
              6: "(d) Six-period cycle ($L=6$)"}

    RC = {"font.size": 15, "axes.labelsize": 15, "xtick.labelsize": 12,
          "ytick.labelsize": 12, "legend.fontsize": 13,
          "axes.titlesize": 15}
    C_F1, C_F2, C_REF = "#2a78d6", "black", "#d62728"

    with plt.rc_context(RC):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
        for ax, L in zip(axes.ravel(), (1, 2, 4, 6)):
            score, g, sidx, P = picks[L]
            ref, game = reconstruct_reference(g, P)
            p1 = P[0, -NDISP:]
            p2 = P[1, -NDISP:]
            rr = ref[-NDISP:]
            x = np.arange(1, NDISP + 1)
            ax.plot(x, p1, color=C_F1, lw=2.2, marker="o", ms=4,
                    label="Firm 1 price")
            ax.plot(x, p2, color=C_F2, lw=2.2, marker="s", ms=4,
                    label="Firm 2 price")
            ax.plot(x, rr, color=C_REF, lw=2.0, ls=":",
                    label="Market reference")
            ax.set_title(f"{titles[L]},  $\\gamma={g:.2f}$")
            ax.set_xlabel("Period")
            ax.set_ylabel("Price")
            ax.set_xlim(1, NDISP)
            if L == 1:
                # widen the y-range so the two nearly-equal firm prices read
                # as close together rather than being blown up by autoscale
                c = 0.5 * (p1.mean() + p2.mean())
                ax.set_ylim(c - 0.10, c + 0.10)
            ax.grid(True, ls="--", alpha=0.4)
            print(f"L={L}: gamma={g:.4f} session={sidx} "
                  f"amp={P[:,-TAIL:].max()-P[:,-TAIL:].min():.3f} "
                  f"level={P[:,-TAIL:].mean():.3f}")

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3,
                   frameon=False, bbox_to_anchor=(0.5, 1.02))
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        out = os.path.join(OUT_DIR, "td3_cycle_examples.png")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    main()
