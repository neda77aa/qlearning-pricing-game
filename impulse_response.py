"""Deviation / punishment test (Calvano et al. 2020, section 5.2, Figure 4).

Protocol
--------
Start from the converged ("limit") strategies on their limit orbit. In period
tau = 1 one firm is exogenously forced to defect to the STATIC BEST RESPONSE
to the rival's pre-deviation price (given the current reference price); the
rival keeps playing its learned strategy. From tau = 2 on, both firms play
their learned strategies. For sessions that converged to a price cycle, the
deviation is started from every phase of the cycle and the impulse responses
are averaged (Calvano fn. 29). Both firms take a turn as the deviator.

Outputs (in <OUT_DIR>):
  irf_tabular_gamma_<g>.png / irf_td3_gamma_<g>.png   Calvano-Fig.4-style plot
  irf_summary.csv          per (method, gamma): punishment stats + incentive
                           compatibility (% of deviations that are unprofitable
                           in present discounted value, delta = game.delta)

Tabular: uses the paper's SAVED converged strategies (session_details.npz in
the benchmark results); no re-training.
TD3: re-trains sessions from their production seeds (exactly reproducible),
then tests the frozen policies.

Run:
  /Users/neda/llm_venv/bin/python impulse_response.py tabular
  /Users/neda/llm_venv/bin/python impulse_response.py td3
  /Users/neda/llm_venv/bin/python impulse_response.py both
"""
import os
import sys
import glob
import multiprocessing as mp

import numpy as np
import pandas as pd

from input.init import model

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
BENCH_DIR = ("/Users/neda/Desktop/UBC/PHD/research_term_4/paper_results/"
             "benchmark_figure/gamma_nloss_only_reference_True")
OUT_DIR = "../Results/experiments/impulse_response"

GAME_KWARGS = dict(n=2, k=15, memory=1, demand_type="reference",
                   common_reference=True, lossaversion=1)

# gammas to test (tabular names must exist under BENCH_DIR)
TABULAR_GAMMAS = ["0.05", "1.0672", "2.0845", "3.0"]
TD3_GAMMAS = [0.05, 1.1036, 2.1571, 3.0]     # production grid values
TD3_SEEDS = [1000, 1001, 1002, 1003]

T_IR = 16          # periods shown in the figure (tau = 0 .. T_IR-1)
T_IC = 100         # horizon for the discounted incentive-compatibility sums
BURN = 200         # burn-in periods to reach the limit orbit
MAX_PHASES = 25    # max cycle phases used as deviation starting points

# deviation depths tested: 'br' = static best response (mild undercut),
# 'nash' = force the deviator all the way down to the Nash price (harsh cut)
DEV_TARGETS = ("br", "nash")

# TD3 training config = the validated production config
TD3_SIM = dict(
    tmax=150_000, start_steps=1_000,
    expl_noise=0.30, expl_min=0.02, expl_decay=15_000,
    anneal_steps=40_000, freeze_tail=10_000,
    min_steps=50_000,
    pol_check_every=1_000, pol_tol_frac=4e-3, pol_stable_checks=2,
    pol_probe_size=256,
    cycle_rollout=1_000, cycle_tol_frac=2e-3,
    hidden=128, lr=3e-4, batch_size=256, buffer_size=150_000, device="cpu",
)


# =========================================================================== #
# TABULAR: transition + IRF from saved index_strategies
# =========================================================================== #
def _tab_ref_update(game, r_idx, a_idx, d):
    """Legacy (paper) common-reference update, INDEX space, memory=1:
    r' = round( lambda*r + (1-lambda) * sum(a*d)/sum(d) ), clipped to grid."""
    tot = float(np.sum(d))
    weighted = float(np.sum(a_idx * d)) / tot if tot > 0 else float(np.mean(a_idx))
    r_cont = game.lambda_ * r_idx + (1.0 - game.lambda_) * weighted
    return int(np.clip(np.round(r_cont), 0, game.k - 1))


def _tab_step(game, strat, a_prev, r_idx, forced=None):
    """One tabular period from state (a_prev, r_idx).

    forced : None or (firm, action_idx) -- exogenous action override.
    Returns (a, r_new, prices, profits). Mirrors qlearning.detect_cycle:
    demand uses the OLD reference; profits are recorded at the NEW reference.
    """
    a = strat[(slice(None),) + tuple(a_prev) + (r_idx,)].copy()
    if forced is not None:
        a[forced[0]] = forced[1]
    d = np.asarray(game.demand(np.asarray(game.A[a]),
                               np.asarray(game.R[r_idx])), dtype=float)
    r_new = _tab_ref_update(game, r_idx, a, d)
    profits = np.asarray(game.PI[tuple(np.append(a, r_new))], dtype=float)
    return a, r_new, np.asarray(game.A[a], dtype=float), profits


def _tab_limit_cycle(game, strat, a0, r0):
    """Burn in, then return the exact limit cycle as a list of (a, r) states."""
    a, r = np.asarray(a0, dtype=int), int(r0)
    for _ in range(BURN):
        a, r, _, _ = _tab_step(game, strat, a, r)
    seen = {}
    orbit = []
    for t in range(2000):
        key = (tuple(a), r)
        if key in seen:
            return orbit[seen[key]:]          # the cycle part only
        seen[key] = t
        orbit.append((a.copy(), r))
        a, r, _, _ = _tab_step(game, strat, a, r)
    return orbit[-MAX_PHASES:]                # fallback: last states


def _tab_static_br(game, i, a_rival, r_idx):
    """Static best response (grid) of firm i to the rival's action index."""
    if i == 0:
        prof = game.PI[:, a_rival, r_idx, 0]
    else:
        prof = game.PI[a_rival, :, r_idx, 1]
    return int(np.argmax(prof))


def _tab_dev_action(game, strat, deviator, a, r, dev_target):
    """Deviation action index: 'br' = static best response to the rival's
    pre-deviation action; 'nash' = grid price closest to the Nash price."""
    if dev_target == "nash":
        return int(np.argmin(np.abs(np.asarray(game.A) -
                                    game.p_nash[deviator])))
    return _tab_static_br(game, deviator, a[1 - deviator], r)


def _tab_paths(game, strat, state, deviator, dev_target="br"):
    """Baseline and deviation paths of length T_IC from a cycle state.
    Returns dict with price/profit arrays, shape (2, T)."""
    (a0, r0) = state
    # tau = 0 is the pre-deviation period: both on-path
    out = {}
    for dev in (False, True):
        a, r = a0.copy(), r0
        prices = np.zeros((2, T_IC))
        profits = np.zeros((2, T_IC))
        # period 0: on-path action from the state
        a, r, prices[:, 0], profits[:, 0] = _tab_step(game, strat, a, r)
        for t in range(1, T_IC):
            forced = None
            if dev and t == 1:
                forced = (deviator,
                          _tab_dev_action(game, strat, deviator, a, r,
                                          dev_target))
            a, r, prices[:, t], profits[:, t] = _tab_step(
                game, strat, a, r, forced=forced)
        out["dev" if dev else "base"] = (prices, profits)
    return out


def irf_tabular(gamma_name):
    """IRF averaged over all sessions x cycle phases x deviator identity."""
    d = os.path.join(BENCH_DIR, f"gamma_{gamma_name}")
    z = np.load(os.path.join(d, "session_details.npz"))
    strat_all = z["index_strategies"]          # (2, k, k, R, sessions)
    n_sess = strat_all.shape[-1]

    game = model(gamma=float(gamma_name), num_sessions=1, aprint=False,
                 **GAME_KWARGS)
    game.continuous_reference = False          # paper runs used legacy update
    # sanity: Nash price must match the config the benchmark was run with
    cfg = pd.read_csv(os.path.join(d, "config.csv")).iloc[0]
    p_nash_cfg = float(str(cfg["Pnash"]).strip("[]").split()[0])
    if abs(p_nash_cfg - game.p_nash[0]) > 1e-4:
        print(f"  [warn] rebuilt model Nash {game.p_nash[0]:.6f} != saved "
              f"config Nash {p_nash_cfg:.6f} -- check model defaults!")

    # starting states: map first saved cycle prices/reference to indices
    cyc_prices = z["cycle_prices"]             # values (2, pad, sessions)
    cyc_refs = z["cycle_reference_prices"]     # indices (1, pad, sessions)

    accs = {dt: _Accumulator() for dt in DEV_TARGETS}
    for s in range(n_sess):
        strat = strat_all[..., s]
        a0 = np.array([int(np.argmin(np.abs(game.A - cyc_prices[i, 0, s])))
                       for i in range(2)])
        r0 = int(cyc_refs[0, 0, s])
        orbit = _tab_limit_cycle(game, strat, a0, r0)
        phases = orbit if len(orbit) <= MAX_PHASES else \
            [orbit[j] for j in np.linspace(0, len(orbit) - 1, MAX_PHASES,
                                           dtype=int)]
        for state in phases:
            for deviator in (0, 1):
                for dt in DEV_TARGETS:
                    paths = _tab_paths(game, strat, state, deviator,
                                       dev_target=dt)
                    accs[dt].add(paths, deviator, game.delta)
    return {dt: accs[dt].result(game) for dt in DEV_TARGETS}


# =========================================================================== #
# TABULAR, ES (continuous) reference: transition + IRF
# Strategies trained with continuous_reference=True smooth the reference as a
# continuous float in PRICE units; the index is used only to read PI / form
# the state (qlearning.compute_reference_price, ES branch). The frozen-policy
# state is therefore (action indices, reference index, continuous reference).
# =========================================================================== #
def _tab_step_es(game, strat, a_prev, r_idx, r_cont, forced=None):
    """One ES-reference tabular period from state (a_prev, r_idx, r_cont)."""
    a = strat[(slice(None),) + tuple(a_prev) + (r_idx,)].copy()
    if forced is not None:
        a[forced[0]] = forced[1]
    prices = np.asarray(game.A[a], dtype=float)
    d = np.asarray(game.demand(prices, np.asarray(game.R[r_idx])), dtype=float)
    tot = float(np.sum(d))
    weighted = float(np.sum(prices * d)) / tot if tot > 0 else float(np.mean(prices))
    r_cont_new = game.lambda_ * r_cont + (1.0 - game.lambda_) * weighted
    r_new = int(np.argmin(np.abs(np.asarray(game.R) - r_cont_new)))
    profits = np.asarray(game.PI[tuple(np.append(a, r_new))], dtype=float)
    return a, r_new, r_cont_new, prices, profits


def _tab_limit_cycle_es(game, strat):
    """Burn in from a fixed midpoint state, then return the limit cycle as a
    list of (a, r_idx, r_cont) states. r_cont converges geometrically, so
    after the burn-in a rounded key detects the exact orbit."""
    k = game.k
    a = np.array([k // 2, k // 2])
    r_cont = float(np.mean(np.asarray(game.A)))
    r_idx = int(np.argmin(np.abs(np.asarray(game.R) - r_cont)))
    for _ in range(2 * BURN):
        a, r_idx, r_cont, _, _ = _tab_step_es(game, strat, a, r_idx, r_cont)
    seen, orbit = {}, []
    for t in range(2000):
        key = (tuple(a), r_idx, round(r_cont, 9))
        if key in seen:
            return orbit[seen[key]:]
        seen[key] = t
        orbit.append((a.copy(), r_idx, r_cont))
        a, r_idx, r_cont, _, _ = _tab_step_es(game, strat, a, r_idx, r_cont)
    return orbit[-MAX_PHASES:]


def _tab_paths_es(game, strat, state, deviator, dev_target="br"):
    """Baseline and deviation paths (ES reference), length T_IC."""
    (a0, r_idx0, r_cont0) = state
    out = {}
    for dev in (False, True):
        a, r_idx, r_cont = a0.copy(), r_idx0, r_cont0
        prices = np.zeros((2, T_IC))
        profits = np.zeros((2, T_IC))
        a, r_idx, r_cont, prices[:, 0], profits[:, 0] = _tab_step_es(
            game, strat, a, r_idx, r_cont)
        for t in range(1, T_IC):
            forced = None
            if dev and t == 1:
                forced = (deviator,
                          _tab_dev_action(game, strat, deviator, a, r_idx,
                                          dev_target))
            a, r_idx, r_cont, prices[:, t], profits[:, t] = _tab_step_es(
                game, strat, a, r_idx, r_cont, forced=forced)
        out["dev" if dev else "base"] = (prices, profits)
    return out


def irf_tabular_es(exp_dir, gamma_name, lambda_=0.6):
    """IRF for an ES-reference tabular experiment (e.g. the beta=4e-6 sweep).

    Returns ({dev_target: result}, {dev_target: per-session %gain array}).
    """
    z = np.load(os.path.join(exp_dir, f"gamma_{gamma_name}",
                             "session_details.npz"))
    strat_all = z["index_strategies"]
    game = model(gamma=float(gamma_name), num_sessions=1, aprint=False,
                 lambda_=lambda_, continuous_reference=True, **GAME_KWARGS)
    disc = game.delta ** np.arange(1, T_IC)

    accs = {dt: _Accumulator() for dt in DEV_TARGETS}
    sess_gain = {dt: [] for dt in DEV_TARGETS}
    for s in range(strat_all.shape[-1]):
        strat = strat_all[..., s]
        orbit = _tab_limit_cycle_es(game, strat)
        phases = orbit if len(orbit) <= MAX_PHASES else \
            [orbit[j] for j in np.linspace(0, len(orbit) - 1, MAX_PHASES,
                                           dtype=int)]
        for dt in DEV_TARGETS:
            gains = []
            for state in phases:
                for deviator in (0, 1):
                    p = _tab_paths_es(game, strat, state, deviator,
                                      dev_target=dt)
                    accs[dt].add(p, deviator, game.delta)
                    base = np.sum(disc * p["base"][1][deviator, 1:])
                    devv = np.sum(disc * p["dev"][1][deviator, 1:])
                    gains.append((devv - base) / base * 100)
            sess_gain[dt].append(float(np.mean(gains)))
    return ({dt: accs[dt].result(game) for dt in DEV_TARGETS},
            {dt: np.array(v) for dt, v in sess_gain.items()})


# =========================================================================== #
# TD3: retrain (reproducible), then IRF on frozen actors
# =========================================================================== #
def _td3_step(game, agents, price_hist, r, lo, hi, forced=None):
    """One frozen-policy TD3 period. forced: None or (firm, price)."""
    from input.td3learning import build_state, _action_to_price, update_reference
    s = build_state(game, price_hist, r, lo, hi)
    prices = np.array([_action_to_price(ag.act(s), lo, hi) for ag in agents])
    if forced is not None:
        prices[forced[0]] = forced[1]
    profits = np.asarray(game.compute_profits(prices, r), dtype=float)
    new_hist = price_hist.copy()
    if game.memory > 1:
        new_hist[:, 1:] = price_hist[:, :-1]
    new_hist[:, 0] = prices
    r_new = update_reference(game, r, prices)
    return new_hist, r_new, prices, profits


def _td3_static_br(game, i, p_rival, r, lo, hi, npts=2001):
    """Static best response of firm i on a fine price grid."""
    grid = np.linspace(lo, hi, npts)
    best_p, best_v = grid[0], -np.inf
    p = np.empty(2)
    for pv in grid:
        p[i], p[1 - i] = pv, p_rival
        v = float(np.asarray(game.compute_profits(p, r), dtype=float)[i])
        if v > best_v:
            best_v, best_p = v, pv
    return best_p


def _td3_paths(game, agents, state, deviator, lo, hi, dev_target="br"):
    (ph0, r0) = state
    out = {}
    for dev in (False, True):
        ph, r = ph0.copy(), r0
        prices = np.zeros((2, T_IC))
        profits = np.zeros((2, T_IC))
        ph, r, prices[:, 0], profits[:, 0] = _td3_step(game, agents, ph, r,
                                                       lo, hi)
        for t in range(1, T_IC):
            forced = None
            if dev and t == 1:
                if dev_target == "nash":
                    p_dev = float(game.p_nash[deviator])
                else:
                    # static BR to the rival's PRE-deviation price (period 0)
                    p_dev = _td3_static_br(game, deviator,
                                           prices[1 - deviator, 0], r, lo, hi)
                forced = (deviator, p_dev)
            ph, r, prices[:, t], profits[:, t] = _td3_step(
                game, agents, ph, r, lo, hi, forced=forced)
        out["dev" if dev else "base"] = (prices, profits)
    return out


def _td3_session_irf(job):
    """Worker: train one session, run the IRF on its frozen policies."""
    gamma, seed = job
    import torch
    torch.set_num_threads(1)
    from input.td3learning import simulate_game_td3, _bounds

    game = model(gamma=gamma, num_sessions=1, aprint=False, **GAME_KWARGS)
    game.Q = None
    roll, agents = simulate_game_td3(game, seed=seed, verbose=False, **TD3_SIM)
    lo, hi = _bounds(game)

    # reconstruct the end-of-rollout state, burn to the limit orbit
    n = game.n
    ph = np.tile(roll["prices"][:, -1].reshape(n, 1), (1, game.memory))
    r = roll["reference"][:, -1]
    r = float(r[0]) if game.common_reference else r.copy()
    for _ in range(BURN):
        ph, r, _, _ = _td3_step(game, agents, ph, r, lo, hi)

    # phases: sample the orbit over the next MAX_PHASES periods
    states = []
    for _ in range(MAX_PHASES):
        states.append((ph.copy(), r if np.isscalar(r) else r.copy()))
        ph, r, _, _ = _td3_step(game, agents, ph, r, lo, hi)

    accs = {dt: _Accumulator() for dt in DEV_TARGETS}
    for state in states:
        for deviator in (0, 1):
            for dt in DEV_TARGETS:
                paths = _td3_paths(game, agents, state, deviator, lo, hi,
                                   dev_target=dt)
                accs[dt].add(paths, deviator, game.delta)
    return accs


def irf_td3(gamma, seeds=TD3_SEEDS, processes=None):
    game = model(gamma=gamma, num_sessions=1, aprint=False, **GAME_KWARGS)
    jobs = [(gamma, s) for s in seeds]
    if processes is None:
        processes = min(len(jobs), max(1, mp.cpu_count() - 4))
    with mp.Pool(processes=processes) as pool:
        per_session = pool.map(_td3_session_irf, jobs)
    out = {}
    for dt in DEV_TARGETS:
        acc = _Accumulator()
        for accs in per_session:
            acc.merge(accs[dt])
        out[dt] = acc.result(game)
    return out


# =========================================================================== #
# Shared: accumulation, stats, plotting
# =========================================================================== #
class _Accumulator:
    """Pools impulse responses over sessions/phases/deviator identity."""

    def __init__(self):
        self.dev_price, self.nondev_price = [], []
        self.base_price = []
        self.ic_gain = []          # PDV(deviation) - PDV(baseline), deviator

    def add(self, paths, deviator, delta):
        pb, fb = paths["base"]
        pdv, fd = paths["dev"]
        self.dev_price.append(pdv[deviator, :T_IR])
        self.nondev_price.append(pdv[1 - deviator, :T_IR])
        self.base_price.append(pb.mean(axis=0)[:T_IR])
        disc = delta ** np.arange(1, T_IC)          # PDV from tau=1 onward
        self.ic_gain.append(float(np.sum(disc * (fd[deviator, 1:] -
                                                 fb[deviator, 1:]))))

    def merge(self, other):
        self.dev_price += other.dev_price
        self.nondev_price += other.nondev_price
        self.base_price += other.base_price
        self.ic_gain += other.ic_gain

    def result(self, game):
        ic = np.asarray(self.ic_gain)
        return {
            "dev_price": np.mean(self.dev_price, axis=0),
            "nondev_price": np.mean(self.nondev_price, axis=0),
            "long_run": float(np.mean(self.base_price)),
            "n_obs": len(ic),
            "frac_unprofitable": float(np.mean(ic < 0)),
            "mean_ic_gain": float(np.mean(ic)),
            "p_nash": float(game.p_nash[0]),
            "p_coop": float(game.p_coop[0]),
        }


def plot_irf(res, gamma, method, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    S1, S2 = "#2a78d6", "#eb6834"
    SURF, INK, SEC, MUT = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
    GRID, BASE = "#e1e0d9", "#c3c2b7"

    t = np.arange(T_IR)
    fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=150)
    fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
    for y, lbl in [(res["p_nash"], "Nash"), (res["p_coop"], "Coop"),
                   (res["long_run"], "long-run")]:
        style = dict(color=BASE, lw=1.2) if lbl == "long-run" else \
            dict(color=MUT, lw=1.0, ls=(0, (4, 3)))
        ax.axhline(y, zorder=1, **style)
        ax.annotate(lbl, xy=(1.005, y), xycoords=("axes fraction", "data"),
                    color=SEC, fontsize=8.5, va="center", ha="left",
                    annotation_clip=False)
    ax.plot(t, res["dev_price"], color=S2, lw=2.0, marker="o", ms=4,
            zorder=3, label="Deviating firm")
    ax.plot(t, res["nondev_price"], color=S1, lw=2.0, marker="^", ms=4,
            ls="--", zorder=3, label="Non-deviating firm")
    ax.set_title(f"{method} — impulse response to a one-period deviation "
                 f"(γ = {gamma})\n"
                 f"{res['n_obs']} deviations pooled; "
                 f"{res['frac_unprofitable']*100:.0f}% unprofitable "
                 f"(mean PDV gain {res['mean_ic_gain']:+.4f})",
                 color=INK, fontsize=10, loc="left", pad=10)
    ax.set_xlabel("Period (deviation at t = 1)", color=SEC, fontsize=9)
    ax.set_ylabel("Price", color=SEC, fontsize=9)
    ax.tick_params(colors=MUT, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color(BASE)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.13), ncol=2,
              frameon=False, fontsize=8.5, labelcolor=SEC)
    fig.tight_layout()
    fig.savefig(out_png, facecolor=SURF, bbox_inches="tight")
    plt.close(fig)


# =========================================================================== #
def main(which):
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []

    def _emit(res_by_target, method, label, g):
        for dt, res in res_by_target.items():
            np.savez(os.path.join(OUT_DIR,
                                  f"irf_{method}_gamma_{g}_dev-{dt}.npz"),
                     **{k: np.asarray(v) for k, v in res.items()})
            out = os.path.join(OUT_DIR,
                               f"irf_{method}_gamma_{g}_dev-{dt}.png")
            dt_label = ("deviation: static BR" if dt == "br"
                        else "deviation: Nash price")
            plot_irf(res, f"{g}, {dt_label}", label, out)
            rows.append(dict(method=method, gamma=float(g), dev=dt, **{
                k: res[k] for k in ("n_obs", "frac_unprofitable",
                                    "mean_ic_gain", "long_run")}))
            print(f"  dev={dt}: {res['n_obs']} deviations, "
                  f"{res['frac_unprofitable']*100:.1f}% unprofitable -> {out}")

    if which in ("tabular", "both"):
        for g in TABULAR_GAMMAS:
            print(f"tabular gamma={g} ...")
            _emit(irf_tabular(g), "tabular", "Tabular Q", g)

    if which in ("td3", "both"):
        for g in TD3_GAMMAS:
            print(f"td3 gamma={g} (retraining {len(TD3_SEEDS)} sessions) ...")
            _emit(irf_td3(g), "td3", "TD3 (continuous)", g)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "irf_summary.csv"), index=False)
    print(f"\nsummary -> {os.path.join(OUT_DIR, 'irf_summary.csv')}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    mp.freeze_support()
    main(sys.argv[1] if len(sys.argv) > 1 else "both")
