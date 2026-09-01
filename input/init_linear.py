"""
Linear reference-dependent demand model (robustness check).

Implements the "Simple Linear Demand" of the paper (eq. 3):

    D_i = 1 - p_i - gamma * (p_i - r) + (1 + gamma) * (p_j - p_i)      (n = 2)

with zero marginal cost and per-period profit  pi_i = p_i * D_i.

Closed-form symmetric rational-expectation benchmarks (r = symmetric price),
generalized to n firms (own-price slope dD_i/dp_i = -n*(1+gamma)):

    Nash / competitive:   p*    = 1 / (1 + n*(1 + gamma))
                          Pi*   = n*(1 + gamma) / (1 + n*(1 + gamma))**2
    Collusive / monopoly: p_col = 1 / (2 + gamma)          (independent of n)
                          Pi_col= (1 + gamma) / (2 + gamma)**2

For n = 2 these reduce to the paper's p* = 1/(3+2*gamma), Pi* = 2(1+g)/(3+2g)^2,
p_col = 1/(2+gamma), Pi_col = (1+g)/(2+g)^2.

Design
------
This is a thin subclass of ``input.init.model``. It **reuses the entire
'reference' machinery** (state includes the reference price, reference-price
formation, cycle detection, the parallel experiment runners, and the
gamma-only heatmaps) unchanged. Only the parts that are genuinely different
for a *linear* demand are overridden here:

    * ``demand``                        -> eq. (3)
    * ``compute_p_competitive_monopoly``-> analytical Nash / collusive prices
                                           (no fsolve / no FOC needed)
    * ``init_actions``                  -> price grid built from the linear
                                           benchmarks (or an explicit
                                           ``grid_bounds`` frozen across gamma)

The instance still reports ``demand_type == 'reference'`` so that every
``demand_type in ["reference", "misspecification"]`` branch in the shared code
fires correctly. The ``is_linear`` flag is used *only* by
``compute_consumer_surplus`` in ``input.qlearning`` to select the linear
consumer-surplus proxy. Nothing in the paper's original code paths is affected.
"""

import numpy as np

from input.init import model


class LinearModel(model):
    """Linear reference-dependent demand variant of ``model``."""

    def __init__(self, **kwargs):
        # Reuse the reference-demand structure (reference price in the state).
        kwargs.setdefault("demand_type", "reference")
        # The closed-form benchmarks above assume zero marginal cost.
        kwargs.setdefault("c", 0.0)
        # Loss aversion is not part of the linear specification (eq. 3).
        kwargs.setdefault("lossaversion", 1)

        # Flag read by compute_consumer_surplus() to pick the linear CS proxy.
        # Set before super().__init__ so it exists throughout construction.
        self.is_linear = True

        super().__init__(**kwargs)

    # ------------------------------------------------------------------ #
    # Demand: paper eq. (3), n-firm generalization.
    # ------------------------------------------------------------------ #
    def _raw_linear_demand(self, p, r=None):
        """UNCLIPPED linear reference-dependent demand.

        D_i = 1 - p_i - gamma*(p_i - r) + (1 + gamma) * sum_{j != i}(p_j - p_i)

        For n = 2 this is exactly eq. (3). This can go negative off-equilibrium.
        It is used only to *score* profits (see ``compute_profits``): letting the
        pre-clip demand contribute p_i * D_i < 0 penalizes infeasibly-high prices
        and cancels the "free spillover" a clipped reward would give the rival
        through the cross term, restoring the analytical interior optimum as the
        unique profit maximizer.
        """
        p = np.asarray(p, dtype=float)
        if r is None:
            r = self.reference_price(p)  # mean(p) when common_reference

        # sum_{j != i}(p_j - p_i) = (sum_p - p_i) - (n - 1) * p_i
        sum_p = np.sum(p)
        cross = (sum_p - p) - (self.n - 1) * p

        return 1.0 - p - self.gamma * (p - r) + (1.0 + self.gamma) * cross

    def demand(self, p, r=None):
        """Linear reference-dependent demand, clipped at 0 (physical quantity).

        Used everywhere a *quantity* is needed: the demand-weighted reference
        price update and the consumer-surplus proxy. Negative quantities are
        meaningless there (and would produce NaNs / negative CS). Profit scoring
        uses the UNCLIPPED ``_raw_linear_demand`` instead (see ``compute_profits``).
        """
        return np.maximum(self._raw_linear_demand(p, r), 0.0)

    # ------------------------------------------------------------------ #
    # Profit scoring uses UNCLIPPED demand (see _raw_linear_demand).
    # ------------------------------------------------------------------ #
    def compute_profits(self, p, r=None):
        """Per-period profit pi_i = (p_i - c) * D_i using UNCLIPPED demand.

        This overrides the base ``compute_profits`` so the profit matrix ``PI``
        (which is both the Q-learning reward and the reported profit) reflects the
        continuous linear game's incentives rather than the clipped ones. At the
        symmetric interior equilibrium D_i > 0, so this matches the clipped value;
        it differs only for off-equilibrium price pairs, where it removes the
        clip-induced corner-solution artifact.
        """
        p = np.asarray(p, dtype=float)
        if r is None:
            r = self.reference_price(p)
        d = self._raw_linear_demand(p, r)
        return (p - self.c) * d

    # ------------------------------------------------------------------ #
    # Analytical benchmarks (no FOC / fsolve needed).
    # ------------------------------------------------------------------ #
    def compute_p_competitive_monopoly(self):
        """Return (Nash prices, collusive prices) from the closed forms.

        Both are returned as length-``n`` arrays of the symmetric price, matching
        the shape produced by the base ``fsolve`` implementation so that
        ``compute_profits_nash_coop`` and the runners work unchanged.
        """
        g = self.gamma
        n = self.n
        p_nash = 1.0 / (1.0 + n * (1.0 + g))   # n=2 -> 1/(3+2g)
        p_coop = 1.0 / (2.0 + g)               # independent of n
        return np.full(n, p_nash), np.full(n, p_coop)

    # ------------------------------------------------------------------ #
    # Price grid tailored to the linear benchmarks.
    # ------------------------------------------------------------------ #
    def init_actions(self):
        """Build the discrete price/reference grid for the linear model.

        If ``grid_bounds`` is supplied it is used verbatim (used to freeze ONE
        common grid across the whole gamma sweep, since the linear Nash/coop
        prices shrink as gamma grows). Otherwise the grid spans this gamma's
        Nash/coop prices with the usual ``extend`` padding.
        """
        if self.grid_bounds is not None:
            lower_bound, upper_bound = self.grid_bounds
        else:
            p_nash = float(np.min(self.p_minmax[0]))
            p_coop = float(np.max(self.p_minmax[1]))
            lower_bound = p_nash - self.extend * (p_coop - p_nash)
            upper_bound = p_coop + self.extend * (p_coop - p_nash)
            lower_bound = max(0.0, lower_bound)

        A = np.linspace(lower_bound, upper_bound, self.k)
        R = np.linspace(lower_bound, upper_bound, self.k)
        return A, R
