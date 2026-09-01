"""
TD3 (Twin Delayed DDPG) continuous-action learner.

Robustness counterpart to the tabular Q-learning in ``input.qlearning``: it
answers the reviewer question "is the collusion an artifact of the discrete
price grid?" by letting each firm choose a *continuous* price on the SAME
interval the grid was sampling.

Design principles
-----------------
* **Reuse the economics unchanged.** Demand, profit, reference-price mean,
  Nash/collusive benchmarks and the price interval all come from the existing
  ``input.init.model`` instance (``game``). This module never re-derives any
  demand or profit math; it calls ``game.compute_profits(p, r)`` /
  ``game.demand(p, r)`` exactly like ``qlearning.py`` does. Loss aversion,
  ``price_sensitivity`` and the ``LinearModel`` subclass therefore all work for
  free, since they are just different ``compute_profits`` implementations.
* **Same action interval as the grid.** Actions are squeezed into
  ``[A[0], A[-1]]`` -- the exact lower/upper bounds ``init_actions`` built for
  the discrete grid -- so the only difference from the tabular run is
  continuity, which is precisely the robustness claim.
* **Same state as the tabular game.** State = last ``memory`` prices of every
  firm (newest first, matching ``game.last_observed_prices``) plus the
  reference price(s) when ``demand_type == 'reference'``. Prices/reference are
  fed to the networks in normalized ``[-1, 1]`` units.
* **Independent learners.** Each firm has its own actor/critics and its own
  replay buffer, and treats the rival as part of the (non-stationary)
  environment -- the continuous analogue of the per-firm ``game.Q[n]`` tables.

Exploration and horizon are decoupled from the tabular hyper-parameters
(``game.beta``/``game.tmax`` are tuned for 1e7 tabular steps, which is far more
than a neural network needs). They are passed explicitly to
:func:`simulate_game_td3` with deep-RL-appropriate defaults. ``delta`` is reused
from ``game``.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Networks
# --------------------------------------------------------------------------- #
class Actor(nn.Module):
    """Deterministic policy: state -> action in [-1, 1] (one price per firm)."""

    def __init__(self, state_dim, hidden=128):
        super().__init__()
        self.l1 = nn.Linear(state_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, 1)

    def forward(self, s):
        x = F.relu(self.l1(s))
        x = F.relu(self.l2(x))
        return torch.tanh(self.l3(x))  # in [-1, 1]


class Critic(nn.Module):
    """Twin Q-networks Q1, Q2 over (state, own-action)."""

    def __init__(self, state_dim, hidden=128):
        super().__init__()
        # Q1
        self.l1 = nn.Linear(state_dim + 1, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, 1)
        # Q2
        self.l4 = nn.Linear(state_dim + 1, hidden)
        self.l5 = nn.Linear(hidden, hidden)
        self.l6 = nn.Linear(hidden, 1)

    def forward(self, s, a):
        sa = torch.cat([s, a], dim=1)
        q1 = F.relu(self.l1(sa)); q1 = F.relu(self.l2(q1)); q1 = self.l3(q1)
        q2 = F.relu(self.l4(sa)); q2 = F.relu(self.l5(q2)); q2 = self.l6(q2)
        return q1, q2

    def Q1(self, s, a):
        sa = torch.cat([s, a], dim=1)
        q1 = F.relu(self.l1(sa)); q1 = F.relu(self.l2(q1)); q1 = self.l3(q1)
        return q1


# --------------------------------------------------------------------------- #
# Replay buffer (one per firm)
# --------------------------------------------------------------------------- #
class ReplayBuffer:
    """Fixed-size ring buffer. Small on purpose: in self-play, stale opponent
    transitions bias each firm's target, so we keep the buffer recent."""

    def __init__(self, state_dim, capacity, device):
        self.capacity = int(capacity)
        self.device = device
        self.s = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.a = np.zeros((self.capacity, 1), dtype=np.float32)
        self.r = np.zeros((self.capacity, 1), dtype=np.float32)
        self.s2 = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, s, a, r, s2):
        i = self.ptr
        self.s[i] = s
        self.a[i] = a
        self.r[i] = r
        self.s2[i] = s2
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng):
        idx = rng.integers(0, self.size, size=batch_size)
        to = lambda x: torch.as_tensor(x[idx], device=self.device)
        return to(self.s), to(self.a), to(self.r), to(self.s2)


# --------------------------------------------------------------------------- #
# Per-firm TD3 agent
# --------------------------------------------------------------------------- #
class TD3Agent:
    """One firm's TD3 learner: actor + twin critics with target copies."""

    def __init__(self, state_dim, delta, device,
                 hidden=128, lr=3e-4, tau=0.005,
                 policy_noise=0.2, noise_clip=0.5, policy_delay=2,
                 buffer_size=100_000):
        self.device = device
        self.gamma = float(delta)          # reuse the game's discount factor
        self.tau = tau
        self.policy_noise = policy_noise   # in normalized action units [-1, 1]
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.total_it = 0

        self.actor = Actor(state_dim, hidden).to(device)
        self.actor_target = Actor(state_dim, hidden).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic(state_dim, hidden).to(device)
        self.critic_target = Critic(state_dim, hidden).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.buffer = ReplayBuffer(state_dim, buffer_size, device)

    @torch.no_grad()
    def act(self, state_np):
        """Greedy (deterministic) normalized action for a single state."""
        s = torch.as_tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        return float(self.actor(s).item())

    @torch.no_grad()
    def act_batch(self, states_np):
        """Greedy normalized actions for a batch of states, shape (m,)."""
        s = torch.as_tensor(states_np, dtype=torch.float32, device=self.device)
        return self.actor(s).cpu().numpy().reshape(-1)

    def train(self, batch_size, rng):
        self.total_it += 1
        s, a, r, s2 = self.buffer.sample(batch_size, rng)

        with torch.no_grad():
            # Target policy smoothing: noisy target action, clipped to valid range.
            noise = (torch.randn_like(a) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            a2 = (self.actor_target(s2) + noise).clamp(-1.0, 1.0)
            q1_t, q2_t = self.critic_target(s2, a2)
            q_t = torch.min(q1_t, q2_t)               # clipped double-Q
            target = r + self.gamma * q_t

        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Delayed actor + target updates.
        if self.total_it % self.policy_delay == 0:
            actor_loss = -self.critic.Q1(s, self.actor(s)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)

    def _soft_update(self, net, target):
        for p, tp in zip(net.parameters(), target.parameters()):
            tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)


# --------------------------------------------------------------------------- #
# Price / state helpers (bridge continuous actions <-> game economics)
# --------------------------------------------------------------------------- #
def _bounds(game):
    """Continuous action interval = the discrete grid's endpoints."""
    lo = float(np.ravel(game.A).min())
    hi = float(np.ravel(game.A).max())
    return lo, hi


def _action_to_price(a_norm, lo, hi):
    """Map normalized action a in [-1, 1] to a price in [lo, hi]."""
    return lo + (a_norm + 1.0) * 0.5 * (hi - lo)


def _norm(x, lo, hi):
    """Map a price/reference in [lo, hi] to [-1, 1] for network input."""
    return 2.0 * (np.asarray(x, dtype=np.float64) - lo) / (hi - lo) - 1.0


def _has_reference(game):
    return game.demand_type in ("reference", "misspecification")


def _ref_dim(game):
    if not _has_reference(game):
        return 0
    return 1 if game.common_reference else game.n


def state_dim(game):
    """State = last `memory` prices of all firms (+ reference(s) if any)."""
    return game.n * game.memory + _ref_dim(game)


def init_reference(game, prices):
    """Initial continuous reference from the initial prices, matching
    ``game.reference_price`` (mean for common, own price otherwise)."""
    if game.common_reference:
        return float(game.reference_price(prices))          # scalar mean
    return np.asarray(prices, dtype=np.float64).copy()      # per-firm vector


def update_reference(game, r_prev, prices):
    """Continuous exponential-smoothing reference update.

    Mirrors ``qlearning.compute_reference_price`` but stays in continuous price
    units (no rounding to grid indices -- that discretization is exactly what
    this robustness check removes).

      common_reference : r' = lambda*r + (1-lambda)*(demand-weighted mean price)
      firm-specific    : r'_i = lambda*r_i + (1-lambda)*p_i
    """
    lam = game.lambda_
    prices = np.asarray(prices, dtype=np.float64)
    if game.common_reference:
        d = np.asarray(game.demand(prices, r_prev), dtype=np.float64)
        denom = np.sum(d)
        weighted = np.sum(prices * d) / denom if denom > 0 else np.mean(prices)
        return float(lam * r_prev + (1.0 - lam) * weighted)
    else:
        return lam * np.asarray(r_prev, dtype=np.float64) + (1.0 - lam) * prices


# --------------------------------------------------------------------------- #
# Q-learning consumer reference (continuous-action port of the tabular agent)
# --------------------------------------------------------------------------- #
def _price_grid(game):
    """Sorted unique discrete price grid = the k reference/action price levels."""
    return np.unique(np.ravel(np.asarray(game.A, dtype=np.float64)))


def _nearest_index(grid, prices):
    """Nearest grid index for each (continuous) price, into ``grid``."""
    p = np.atleast_1d(np.asarray(prices, dtype=np.float64))
    return np.argmin(np.abs(grid[None, :] - p[:, None]), axis=1).astype(int)


def qref_reference(cref, grid, hist_idx, t, cycle=False):
    """Q-learning consumer reference: forecast the reference price(s) from a
    grid-index price history ``hist_idx`` (n, reference_memory), newest-first.

    Returns ``(r_price, r_idx)`` where ``r_price`` is a float (common reference)
    or an (n,) array (firm-specific), and ``r_idx`` is the raw agent action
    (grid index / index vector) needed for the agent's own Q-update.
    """
    pred = cref.predict(hist_idx, t, cycle=cycle)
    if cref.common_reference:
        idx = int(pred)
        return float(grid[idx]), idx
    idx = np.asarray(pred, dtype=int)
    return grid[idx].astype(np.float64), idx


def build_state(game, price_hist, r, lo, hi):
    """Assemble the normalized network state vector.

    price_hist : (n, memory) array of actual prices, newest in column 0
                 (same layout as game.last_observed_prices).
    r          : scalar (common) or (n,) array (firm-specific) reference price.
    """
    parts = [_norm(price_hist.flatten(), lo, hi)]
    if _has_reference(game):
        parts.append(np.atleast_1d(_norm(r, lo, hi)))
    return np.concatenate(parts).astype(np.float32)


# --------------------------------------------------------------------------- #
# Reward benchmarks (for logging profit gain in the same units as tabular)
# --------------------------------------------------------------------------- #
def profit_gain(game, mean_profit, i):
    denom = game.CoopProfits[i] - game.NashProfits[i]
    if denom == 0:
        return np.nan
    return (mean_profit - game.NashProfits[i]) / denom


def price_gain(game, mean_price, i):
    denom = game.p_coop[i] - game.p_nash[i]
    if denom == 0:
        return np.nan
    return (mean_price - game.p_nash[i]) / denom


# --------------------------------------------------------------------------- #
# Consumer surplus (continuous prices) -- same formula as qlearning
# --------------------------------------------------------------------------- #
def consumer_surplus(game, prices, r):
    """Consumer surplus at continuous prices, mirroring
    ``qlearning.compute_consumer_surplus``."""
    prices = np.asarray(prices, dtype=np.float64)
    if getattr(game, "is_linear", False):
        # Linear model: use realized quantity * ... proxy consistent with the
        # linear demand (integral of inverse demand is model-specific; we use
        # the same demand-based proxy the linear branch expects). Fall back to
        # total demand-weighted surplus 0.5 * sum(D^2)/slope is not defined
        # generically here, so we report total demand as a monotone proxy.
        return float(np.sum(game.demand(prices, r)))
    if game.demand_type == "noreference":
        e = np.exp((game.a - prices) / game.mu)
        return float(game.mu * np.log(np.sum(e) + np.exp(game.a0 / game.mu)))
    if game.demand_type == "price_sensitivity":
        e = np.exp((game.a - (1 + game.gamma) * prices) / game.mu)
        return float(game.mu * np.log(np.sum(e) + np.exp(game.a0 / game.mu)))
    # reference / misspecification
    p_eff = prices + game.gamma * (prices - r)
    e = np.exp((game.a - p_eff) / game.mu)
    return float(game.mu * np.log(np.sum(e) + np.exp(game.a0 / game.mu)))


# --------------------------------------------------------------------------- #
# One learning session
# --------------------------------------------------------------------------- #
def simulate_game_td3(
    game,
    seed=None,
    tmax=300_000,
    start_steps=1_000,
    batch_size=256,
    expl_noise=0.30,       # initial exploration std (normalized action units)
    expl_min=0.02,         # floor exploration std
    expl_decay=100_000,    # steps for exploration to decay by ~1/e
    anneal_steps=40_000,   # final phase: exploration AND lr decay linearly to 0
                           # (continuous analog of tabular eps->0; lets the
                           # policy genuinely freeze so convergence is testable)
    freeze_tail=10_000,    # after the anneal: lr=0, noise=0 exactly, so the
                           # policy is constant and the Delta_pi flag can fire
                           # BEFORE tmax (convergence is then by construction
                           # of the schedule -- footnote it in the paper)
    hidden=128,
    lr=3e-4,
    tau=0.005,
    buffer_size=100_000,
    # --- convergence: policy stability (Delta_pi) --------------------------- #
    min_steps=50_000,       # do not test convergence before this many steps
    pol_check_every=1_000,  # lag L and test interval (steps)
    pol_tol_frac=1.5e-3,    # eps_p: mean |greedy price change| per lag, frac of range
    pol_stable_checks=2,    # require this many consecutive sub-tolerance checks
    pol_probe_size=256,     # number of recent on-path states in the probe set
    reward_scale=1.0,
    cycle_rollout=200,      # deterministic post-convergence rollout length
    cycle_tol_frac=2e-3,    # cycle-length descriptor tolerance (frac of range)
    consumer_reference_agent=None,  # pretrained ConsumerQReference; enables the
                                    # Q-learning reference path when the game has
                                    # ref_prediction == "qlearning"
    train_reference=True,   # keep updating the reference agent during firm learning
    device="cpu",
    verbose=False,
):
    """Run one TD3 self-play session on ``game``.

    Returns a dict with convergence status and the post-convergence
    deterministic price/profit/reference/consumer-surplus paths.
    """
    device = torch.device(device)
    rng = np.random.default_rng(seed)
    if seed is not None:
        torch.manual_seed(int(seed))

    n = game.n
    lo, hi = _bounds(game)
    prange = hi - lo
    pol_tol = pol_tol_frac * prange
    sdim = state_dim(game)

    agents = [
        TD3Agent(sdim, game.delta, device, hidden=hidden, lr=lr, tau=tau,
                 buffer_size=buffer_size)
        for _ in range(n)
    ]

    # ---- initial state ----------------------------------------------------- #
    # Random initial prices on the grid interval (continuous), newest-first
    # history of width `memory` (all slots equal to the initial price).
    p0 = lo + rng.random(n) * prange
    price_hist = np.tile(p0.reshape(n, 1), (1, game.memory))  # (n, memory)

    # Q-learning reference path: forecast r from a grid-index price history via
    # the pretrained consumer agent, instead of exponential smoothing.
    use_qref = (_has_reference(game)
                and getattr(game, "ref_prediction", None) == "qlearning"
                and consumer_reference_agent is not None)
    if use_qref:
        cref = consumer_reference_agent
        # ConsumerQReference.predict/update draw from the GLOBAL numpy RNG; seed
        # it per session so exploration is independent across worker processes.
        np.random.seed(int(seed) % (2**32) if seed is not None else None)
        grid = _price_grid(game)
        ref_mem = int(game.reference_memory)
        hist_idx = np.tile(_nearest_index(grid, p0).reshape(n, 1), (1, ref_mem))
        r, r_idx = qref_reference(cref, grid, hist_idx, 0, cycle=False)
    elif _has_reference(game):
        r = init_reference(game, p0)
    else:
        r = 0.0
    s = build_state(game, price_hist, r, lo, hi)

    # ---- policy-stability (Delta_pi) convergence state ------------------- #
    # Ring buffer of recent on-path states; at each check we compare every
    # firm's greedy prices on a fixed snapshot of these states now vs. one
    # check (lag L = pol_check_every) ago.
    probe_buf = np.zeros((pol_probe_size, sdim), dtype=np.float32)
    prev_S = None            # snapshot of probe states from the last check
    prev_prices = None       # (n, m) greedy prices on prev_S at the last check
    stable_count = 0         # consecutive sub-tolerance checks

    converged = False
    t_conv = tmax
    for t in range(int(tmax)):
        f = 1.0   # learning-rate/exploration scale (1 = full, 0 = frozen tail)
        # ---- action selection -------------------------------------------- #
        if t < start_steps:
            a_norm = rng.uniform(-1.0, 1.0, size=n)
            det_a = a_norm.copy()  # no policy yet
        else:
            det_a = np.array([ag.act(s) for ag in agents])   # greedy (no noise)
            sigma = expl_min + (expl_noise - expl_min) * np.exp(-t / expl_decay)
            # Anneal-to-freeze schedule: over the `anneal_steps` before the
            # freeze tail, exploration std AND both optimizers' learning rates
            # scale linearly to 0; during the last `freeze_tail` steps they are
            # exactly 0 (policy constant -> Delta_pi flag fires before tmax).
            # Tabular convergence relies on eps -> 0 plus argmax hysteresis;
            # continuous actions have no hysteresis, so the freeze must come
            # from the learning itself winding down.
            t_freeze = tmax - freeze_tail
            if t >= t_freeze:
                f = 0.0
            elif anneal_steps and t > t_freeze - anneal_steps:
                f = (t_freeze - t) / float(anneal_steps)
            else:
                f = 1.0
            if f < 1.0:
                sigma *= f
                for ag in agents:
                    for pg_ in ag.actor_opt.param_groups:
                        pg_["lr"] = lr * f
                    for pg_ in ag.critic_opt.param_groups:
                        pg_["lr"] = lr * f
            a_norm = np.clip(det_a + rng.normal(0.0, sigma, size=n), -1.0, 1.0)

        prices = _action_to_price(a_norm, lo, hi)            # actual prices (n,)

        # ---- environment step (reuse game economics) --------------------- #
        if _has_reference(game):
            pi = np.asarray(game.compute_profits(prices, r), dtype=np.float64)
            if use_qref:
                # advance the reference agent's grid-index history, let it learn
                # (frozen in the tail), and forecast next period's reference
                a_idx = _nearest_index(grid, prices)
                new_hist_idx = hist_idx.copy()
                if ref_mem > 1:
                    new_hist_idx[:, 1:] = hist_idx[:, :-1]
                new_hist_idx[:, 0] = a_idx
                if train_reference and f > 0.0:
                    cref.update(r_idx, hist_idx, a_idx, new_hist_idx)
                r_next, r_next_idx = qref_reference(
                    cref, grid, new_hist_idx, t, cycle=(f == 0.0))
            else:
                r_next = update_reference(game, r, prices)
        else:
            pi = np.asarray(game.compute_profits(prices), dtype=np.float64)
            r_next = 0.0

        # ---- advance price history (newest in col 0) --------------------- #
        new_hist = price_hist.copy()
        if game.memory > 1:
            new_hist[:, 1:] = price_hist[:, :-1]
        new_hist[:, 0] = prices
        s_next = build_state(game, new_hist, r_next, lo, hi)

        # ---- store transition + learn (per firm) ------------------------- #
        for i, ag in enumerate(agents):
            ag.buffer.add(s, a_norm[i], reward_scale * pi[i], s_next)
        if t >= start_steps and f > 0.0:   # lr=0 in the freeze tail: skip
            for ag in agents:
                ag.train(batch_size, rng)

        # ---- convergence: policy stability (Delta_pi) -------------------- #
        # Converged when the *policy* stops changing: for a fixed set of recent
        # on-path states, each firm's greedy price moves by < pol_tol (mean
        # absolute change) between two checks L=pol_check_every steps apart, for
        # pol_stable_checks checks in a row. This tests the strategy, not the
        # realized price path, so it is agnostic to fixed-point vs cyclic
        # equilibria and is not fooled by a still-drifting-but-periodic path.
        probe_buf[t % pol_probe_size] = s
        if ((not converged) and t >= max(min_steps, pol_probe_size)
                and (t % pol_check_every == 0)):
            if prev_S is not None:
                cur = np.array([_action_to_price(ag.act_batch(prev_S), lo, hi)
                                for ag in agents])            # (n, m)
                delta = float(np.max(np.mean(np.abs(cur - prev_prices), axis=1)))
                if delta < pol_tol:
                    stable_count += 1
                    if stable_count >= pol_stable_checks:
                        converged = True
                        t_conv = t
                        if verbose:
                            print(f"  converged at t={t}, delta_pi={delta:.5f}")
                        break
                else:
                    stable_count = 0
                if verbose:
                    sys.stdout.write(f"\r  t={t} delta_pi={delta:.5f} "
                                     f"stable={stable_count}   ")
                    sys.stdout.flush()
            # refresh snapshot: current on-path states + current greedy prices
            prev_S = probe_buf.copy()
            prev_prices = np.array(
                [_action_to_price(ag.act_batch(prev_S), lo, hi) for ag in agents])

        s, price_hist, r = s_next, new_hist, r_next
        if use_qref:
            hist_idx, r_idx = new_hist_idx, r_next_idx

    # ---- post-convergence deterministic rollout (metrics) --------------- #
    roll = _deterministic_rollout(
        game, agents, price_hist, r, lo, hi, cycle_rollout,
        cref=(cref if use_qref else None),
        grid=(grid if use_qref else None),
        hist_idx=(hist_idx if use_qref else None))
    cyc_len = _approx_cycle_length(roll["prices"], cycle_tol_frac * prange)
    roll["cycle_length"] = cyc_len
    roll["converged"] = converged
    roll["t_conv"] = t_conv
    return roll, agents


def _deterministic_rollout(game, agents, price_hist, r, lo, hi, steps,
                           cref=None, grid=None, hist_idx=None):
    """Roll the greedy (noise-free) joint policy forward `steps` periods and
    record prices, profits, reference and consumer surplus each period.

    If ``cref``/``grid``/``hist_idx`` are given, the reference is formed by the
    (frozen, greedy) Q-learning consumer agent instead of exponential smoothing.
    """
    n = game.n
    price_hist = price_hist.copy()
    use_qref = cref is not None
    if use_qref:
        hist_idx = hist_idx.copy()
        ref_mem = hist_idx.shape[1]
    prices_log = np.zeros((n, steps))
    profits_log = np.zeros((n, steps))
    ref_dim = _ref_dim(game)
    ref_log = np.zeros((max(ref_dim, 1), steps))
    cs_log = np.zeros(steps)

    for k in range(steps):
        s = build_state(game, price_hist, r, lo, hi)
        det_a = np.array([ag.act(s) for ag in agents])
        prices = _action_to_price(det_a, lo, hi)

        if _has_reference(game):
            pi = np.asarray(game.compute_profits(prices, r), dtype=np.float64)
        else:
            pi = np.asarray(game.compute_profits(prices), dtype=np.float64)

        prices_log[:, k] = prices
        profits_log[:, k] = pi
        if ref_dim:
            ref_log[:, k] = np.atleast_1d(r)
        cs_log[k] = consumer_surplus(game, prices, r)

        # advance
        new_hist = price_hist.copy()
        if game.memory > 1:
            new_hist[:, 1:] = price_hist[:, :-1]
        new_hist[:, 0] = prices
        price_hist = new_hist
        if _has_reference(game):
            if use_qref:
                a_idx = _nearest_index(grid, prices)
                new_hist_idx = hist_idx.copy()
                if ref_mem > 1:
                    new_hist_idx[:, 1:] = hist_idx[:, :-1]
                new_hist_idx[:, 0] = a_idx
                r, _ = qref_reference(cref, grid, new_hist_idx, k, cycle=True)
                hist_idx = new_hist_idx
            else:
                r = update_reference(game, r, prices)

    return {"prices": prices_log, "profits": profits_log,
            "reference": ref_log, "consumer_surplus": cs_log}


def _detect_period(traj, tol, max_period, min_reps=3, stab_window=None):
    """Smallest period of a *stable* periodic orbit in a deterministic price path.

    Parameters
    ----------
    traj : (T, n) array
        Chronological deterministic prices (oldest first, newest last), one
        column per firm.
    tol : float
        Absolute tolerance on the per-phase standard deviation across
        repetitions (in price units, i.e. already scaled by the price range by
        the caller).
    max_period : int
        Longest period to test.
    min_reps : int
        Minimum number of clean repetitions required to accept a period. Guards
        against a single coincidental column match reporting a spurious cycle.
    stab_window : int or None
        If given, test each candidate period over at least this many of the most
        recent samples (so a short period must be stable for a long stretch, not
        just ``min_reps`` steps). If None, exactly ``min_reps`` repetitions are
        checked.

    Method (phase-folded stability)
    -------------------------------
    For a candidate period ``p`` we take the last ``reps * p`` samples, fold them
    into ``reps`` consecutive blocks of length ``p``, and measure, for every
    phase within the cycle and every firm, the **standard deviation across the
    repetitions**. If that std is ``< tol`` everywhere, the trajectory repeats
    with period ``p`` to within ``tol`` -- i.e. a stable period-``p`` orbit that
    tolerates small continuous jitter but rejects drift (drift inflates the
    across-repetition std). Returns the smallest such ``p`` (1 == fixed point),
    or ``None`` if no stable short period is found.

    Why std rather than range (max - min): the range of noisy samples grows with
    the number of samples (~sigma * sqrt(2 ln N)), so with ``stab_window`` a
    small period folds into *more* repetitions and its range inflates -- which
    would wrongly reject short stable orbits in favour of longer ones. The
    standard deviation is (asymptotically) invariant to the repetition count, so
    the criterion is monotone in the true jitter, not in ``p``.
    """
    T, n = traj.shape
    max_p = min(int(max_period), T // int(min_reps))
    for p in range(1, max_p + 1):
        reps = min_reps
        if stab_window:
            reps = max(min_reps, -(-int(stab_window) // p))  # ceil division
        reps = min(reps, T // p)
        if reps < min_reps:
            continue
        seg = traj[T - p * reps:]                 # (reps*p, n), newest at end
        folded = seg.reshape(reps, p, n)          # axis 0 = repetitions
        disp = folded.std(axis=0)                 # (p, n) std across repetitions
        if np.all(disp < tol):
            return p
    return None


def _approx_cycle_length(prices_log, tol, min_reps=3):
    """Cycle length of a frozen-policy deterministic rollout.

    Wraps :func:`_detect_period` (phase-folded, tolerant of small continuous
    jitter, requires ``min_reps`` repetitions). Returns the period (1 == fixed
    point) or the rollout length if no stable short cycle fits in the rollout."""
    _, T = prices_log.shape
    period = _detect_period(prices_log.T, tol, max_period=T, min_reps=min_reps)
    return period if period is not None else T


# --------------------------------------------------------------------------- #
# Session summary <-> game-array plumbing (shared by serial and parallel paths)
# --------------------------------------------------------------------------- #
def _summarize_roll(game, roll, steady_frac=0.5):
    """Collapse a post-convergence rollout into per-session scalars.

    ``steady_frac`` : use only the last fraction of the deterministic rollout to
    form steady-state means (drops any transient at the start of the rollout).
    Returned dict is plain floats/arrays so it is picklable across processes.
    """
    T = roll["prices"].shape[1]
    k0 = int(T * (1.0 - steady_frac))
    pr = roll["prices"][:, k0:]
    pf = roll["profits"][:, k0:]
    return {
        "converged": bool(roll["converged"]),
        "t_conv": float(roll["t_conv"]),
        "cycle_length": int(roll["cycle_length"]),
        "mean_cs": float(np.mean(roll["consumer_surplus"][k0:])),
        "mean_price": np.array([float(np.mean(pr[i])) for i in range(game.n)]),
        "mean_profit": np.array([float(np.mean(pf[i])) for i in range(game.n)]),
    }


def _alloc_session_arrays(game):
    """Allocate the per-session result arrays on ``game`` (mirrors the fields
    the tabular ``run_sessions`` fills, but for *continuous* prices)."""
    n, ns = game.n, game.num_sessions
    game.converged = np.zeros(ns, dtype=bool)
    game.time_to_convergence = np.zeros(ns, dtype=float)
    game.cycle_length = np.zeros(ns, dtype=int)
    game.td3_mean_price = np.zeros((n, ns))
    game.td3_mean_profit = np.zeros((n, ns))
    game.td3_profit_gain = np.zeros((n, ns))
    game.td3_price_gain = np.zeros((n, ns))
    game.td3_mean_cs = np.zeros(ns)


def _store_summary(game, i_sess, summ):
    """Write one session summary into the pre-allocated game arrays."""
    game.converged[i_sess] = summ["converged"]
    game.time_to_convergence[i_sess] = summ["t_conv"]
    game.cycle_length[i_sess] = summ["cycle_length"]
    game.td3_mean_cs[i_sess] = summ["mean_cs"]
    for i in range(game.n):
        mp = float(summ["mean_price"][i])
        mprof = float(summ["mean_profit"][i])
        game.td3_mean_price[i, i_sess] = mp
        game.td3_mean_profit[i, i_sess] = mprof
        game.td3_profit_gain[i, i_sess] = profit_gain(game, mprof, i)
        game.td3_price_gain[i, i_sess] = price_gain(game, mp, i)


# --------------------------------------------------------------------------- #
# Many sessions (serial)
# --------------------------------------------------------------------------- #
def run_sessions_td3(game, verbose=True, base_seed=0, steady_frac=0.5, **sim_kwargs):
    """Run ``game.num_sessions`` independent TD3 sessions serially.

    Stores per-session summaries on ``game``:
        game.converged / time_to_convergence / cycle_length
        game.td3_mean_price / td3_mean_profit / td3_profit_gain /
        game.td3_price_gain / td3_mean_cs
    """
    _alloc_session_arrays(game)
    for i_sess in range(game.num_sessions):
        if verbose:
            print(f"\nTD3 session {i_sess + 1}/{game.num_sessions}")
        roll, _ = simulate_game_td3(game, seed=base_seed + i_sess,
                                    verbose=verbose, **sim_kwargs)
        _store_summary(game, i_sess, _summarize_roll(game, roll, steady_frac))

    if verbose:
        print("\nAll TD3 sessions completed.")
        print(f"  converged: {game.converged.sum()}/{game.num_sessions}")
        print(f"  mean price gain: {np.nanmean(game.td3_price_gain):.4f}")
        print(f"  mean profit gain: {np.nanmean(game.td3_profit_gain):.4f}")
    return game


# --------------------------------------------------------------------------- #
# Parallel gamma sweep (mirrors ConvResults' run_experiment_parallel_gamma_only)
# --------------------------------------------------------------------------- #
def _td3_session_worker(game, session_id, seed, steady_frac, sim_kwargs,
                        consumer_reference_agent=None, train_reference=True):
    """Top-level worker: run ONE TD3 session and return a picklable summary.

    Pin torch to a single thread so ``num_processes`` workers do not
    oversubscribe the CPU (each already parallelizes poorly on tiny nets).

    ``consumer_reference_agent`` (if given) is a pretrained ConsumerQReference;
    each worker process receives its own unpickled copy, so per-session updates
    do not clash across processes.
    """
    torch.set_num_threads(1)
    roll, _ = simulate_game_td3(
        game, seed=seed, verbose=False,
        consumer_reference_agent=consumer_reference_agent,
        train_reference=train_reference,
        **sim_kwargs)
    summ = _summarize_roll(game, roll, steady_frac)
    summ["session_id"] = session_id
    # ship the frozen-rollout price/profit paths back too (float32, ~16 KB per
    # session) so the runner can archive them and per-gamma figures of the
    # converged prices/profits can be drawn later
    summ["rollout_prices"] = roll["prices"].astype(np.float32)
    summ["rollout_profits"] = roll["profits"].astype(np.float32)
    return summ


def run_experiment_parallel_td3(
    game_kwargs, gamma_values, num_sessions,
    experiment_name, main_dir="../Results/experiments",
    num_processes=None, base_seed=1000, steady_frac=0.5,
    skip_existing=True, session_timeout=None, model_cls=None,
    per_gamma_callback=None,
    use_reference_pretraining=False, T_ref=200_000, train_reference=True,
    **sim_kwargs,
):
    """Run a gamma sweep with sessions parallelized across processes.

    Parameters
    ----------
    game_kwargs : dict
        Keyword args used to build the model for each gamma (e.g. n, k, memory,
        demand_type, common_reference, lossaversion). ``gamma``, ``num_sessions``
        and ``aprint`` are set by this runner and should be omitted.
    gamma_values : iterable of float
    num_sessions : int
    experiment_name : str
        Results are written to ``<main_dir>/<experiment_name>/gamma_<value>/``.
    model_cls : class or None
        Model class to instantiate (defaults to ``input.init.model``; pass
        ``LinearModel`` for the linear robustness variant).
    per_gamma_callback : callable or None
        Called as ``per_gamma_callback(game, run_dir, gamma)`` after each
        gamma's CSV and rollout archive are written (e.g. to draw figures).
    **sim_kwargs : forwarded to ``simulate_game_td3``.

    Besides ``cycle_statistics.csv``, each gamma directory also gets
    ``rollout_paths.npz`` with the frozen-rollout ``prices_s<i>`` /
    ``profits_s<i>`` arrays (n x cycle_rollout) for every session.
    """
    import multiprocessing as mp
    import pandas as pd
    from input.init import model as _default_model

    Model = model_cls if model_cls is not None else _default_model
    if num_processes is None:
        num_processes = max(1, mp.cpu_count() - 2)
    print(f"TD3 parallel sweep: {num_processes} processes, "
          f"{num_sessions} sessions/gamma, {len(list(gamma_values))} gammas")

    exp_dir = os.path.join(main_dir, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)

    # ---- optional Q-learning consumer reference (two-stage protocol) -------- #
    # Pretrain ONE reference agent (structure is gamma-independent) and hand a
    # copy to every session worker. Each worker process gets its own unpickled
    # copy, so its in-session updates do not clash with other workers.
    cref_pretrained = None
    gamma_list = list(gamma_values)
    if use_reference_pretraining:
        from input.qlearning import pretrain_consumer_reference
        tmpl = Model(gamma=float(gamma_list[0]), num_sessions=1, aprint=False,
                     **game_kwargs)
        if getattr(tmpl, "ref_prediction", None) != "qlearning":
            print("[warn] use_reference_pretraining=True but ref_prediction != "
                  "'qlearning'; the reference agent will be ignored.")
        print(f"pretraining consumer Q-reference for T_ref={int(T_ref)} steps ...")
        cref_pretrained = pretrain_consumer_reference(tmpl, T_ref=int(T_ref))
        print("  reference pretraining done.")

    for gamma in gamma_list:
        gamma = float(gamma)
        run_dir = os.path.join(exp_dir, f"gamma_{gamma}")
        stats_file = os.path.join(run_dir, "cycle_statistics.csv")
        if skip_existing and os.path.exists(stats_file):
            print(f"gamma_{gamma}: already exists, skipping")
            continue

        # Fresh model per gamma so ALL gamma-dependent quantities (Nash/coop
        # prices & profits, price interval) are recomputed cleanly.
        game = Model(gamma=gamma, num_sessions=num_sessions, aprint=False,
                     **game_kwargs)
        game.Q = None   # TD3 never uses the tabular Q; drop it to slim pickling
        _alloc_session_arrays(game)

        print(f"\n{'='*60}\nTD3 gamma = {gamma:.5g}  "
              f"(Nash {game.NashProfits[0]:.4f}, Coop {game.CoopProfits[0]:.4f})"
              f"\n{'='*60}")

        rollout_paths = {}
        with mp.Pool(processes=num_processes) as pool:
            async_res = [
                pool.apply_async(
                    _td3_session_worker,
                    args=(game, i, base_seed + i, steady_frac, sim_kwargs,
                          cref_pretrained, train_reference),
                )
                for i in range(num_sessions)
            ]
            done = 0
            for res in async_res:
                try:
                    summ = res.get(timeout=session_timeout)
                    sid = summ["session_id"]
                    rollout_paths[f"prices_s{sid}"] = summ.pop("rollout_prices")
                    rollout_paths[f"profits_s{sid}"] = summ.pop("rollout_profits")
                    _store_summary(game, sid, summ)
                    done += 1
                    print(f"  session {sid} done "
                          f"({done}/{num_sessions})  "
                          f"conv={summ['converged']}")
                except Exception as e:
                    print(f"  a session failed/timed out: {e}")

        os.makedirs(run_dir, exist_ok=True)
        np.savez_compressed(os.path.join(run_dir, "rollout_paths.npz"),
                            **rollout_paths)
        df = save_cycle_statistics_td3(game, run_dir)
        # Match the tabular gamma-only CSV: append per-player Nash/coop prices.
        for i, v in enumerate(game.p_nash):
            df[f"p_nash_p{i+1}"] = v
        for i, v in enumerate(game.p_coop):
            df[f"p_coop_p{i+1}"] = v
        df.to_csv(stats_file, index=False)
        print(f"  gamma_{gamma}: mean price gain "
              f"{np.nanmean(game.td3_price_gain):.4f}, "
              f"profit gain {np.nanmean(game.td3_profit_gain):.4f}  -> {stats_file}")

        if per_gamma_callback is not None:
            try:
                per_gamma_callback(game, run_dir, gamma)
            except Exception as e:
                print(f"  [warn] per_gamma_callback failed: {e}")

    print("\nTD3 parallel sweep complete.")


# --------------------------------------------------------------------------- #
# CSV output compatible with the existing gamma-only heatmaps
# --------------------------------------------------------------------------- #
def save_cycle_statistics_td3(game, run_dir):
    """Write ``cycle_statistics.csv`` with the SAME columns the tabular saver
    produces, so ``visualization.create_single_heatmap_gamma_only`` /
    ``extract_metric_data`` read it unchanged."""
    import pandas as pd
    os.makedirs(run_dir, exist_ok=True)

    row = {
        "gamma": f"{game.gamma:.5g}",
        "lambda": f"{game.lambda_:.5g}",
        "lossaversion": f"{game.lossaversion:.5g}",
        "num_sessions": game.num_sessions,
        "convergence_rate": f"{np.mean(game.converged):.5g}",
        "mean_cycle_length": f"{np.nanmean(game.cycle_length):.5g}",
        "std_cycle_length": f"{np.nanstd(game.cycle_length):.5g}",
        "mean_consumer_surplus": f"{np.nanmean(game.td3_mean_cs):.5g}",
        "std_consumer_surplus": f"{np.nanstd(game.td3_mean_cs):.5g}",
    }
    for i in range(game.n):
        pn = i + 1
        row[f"mean_profit_p{pn}"] = f"{np.nanmean(game.td3_mean_profit[i]):.5g}"
        row[f"std_profit_p{pn}"] = f"{np.nanstd(game.td3_mean_profit[i]):.5g}"
        row[f"mean_profit_gain_p{pn}"] = f"{np.nanmean(game.td3_profit_gain[i]):.5g}"
        row[f"std_profit_gain_p{pn}"] = f"{np.nanstd(game.td3_profit_gain[i]):.5g}"
        row[f"mean_price_gain_p{pn}"] = f"{np.nanmean(game.td3_price_gain[i]):.5g}"
        row[f"std_price_gain_p{pn}"] = f"{np.nanstd(game.td3_price_gain[i]):.5g}"
        row[f"mean_price_p{pn}"] = f"{np.nanmean(game.td3_mean_price[i]):.5g}"
        row[f"std_price_p{pn}"] = f"{np.nanstd(game.td3_mean_price[i]):.5g}"

    df = pd.DataFrame([row])
    df.to_csv(os.path.join(run_dir, "cycle_statistics.csv"), index=False)
    return df
