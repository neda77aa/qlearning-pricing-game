"""
Model of algorithms and competition
"""

import numpy as np
from itertools import product
from scipy.optimize import fsolve


class model(object):
    """
    model

    Attributes
    ----------
    n : int
        number of players
    alpha : float
        product differentiation parameter
    beta : float
        exploration parameter
    delta : float
        discount factor
    mu : float
        product differentiation parameter
    a : int
        value of the products
    a0 : float
        value of the outside option
    c : float
        marginal cost
    k : int
        dimension of the grid
    stable: int
        periods of game stability
    """

    def __init__(self, **kwargs):
        """Initialize game with default values"""
        # Default properties
        self.n = kwargs.get('n', 2)
        self.alpha = kwargs.get('alpha', 0.15) # learning parameter
        self.beta = kwargs.get('beta', 4e-6)
        self.delta = kwargs.get('delta', 0.95) # 
        self.gamma = kwargs.get('gamma', 1)  # reference effect(in logit model)
        self.lossaversion = kwargs.get('lossaversion', 1.5)
        self.lambda_ = kwargs.get('lambda', 0.5)  # reference update rate
        self.c = kwargs.get('c', 1) # cost
        self.a = kwargs.get('a', 2) # quality
        self.a0 = kwargs.get('a0', 0) # outside option
        self.mu = kwargs.get('mu', 0.25) #horizontal differentiation parameter 
        self.extend = kwargs.get('extend', 0.1)
        self.k = kwargs.get('k', 15) # number of discrete prices
        self.memory = kwargs.get('memory', 1)
        self.reference_memory = kwargs.get('reference_memory', 1)

        # Get demand_type from kwargs, defaulting to 'noreference'
        valid_demand_types = {'reference', 'noreference'}  # Allowed values
        self.demand_type = kwargs.get('demand_type', 'noreference')

        # Validate input
        if self.demand_type not in valid_demand_types:
            raise ValueError(f"Invalid demand_type: {self.demand_type}. Allowed values: {valid_demand_types}")
        self.ref_type = "average"

        # if lossaversion coefficient is larger than 1 we will go with the loss aversion approach
        if self.lossaversion>1:
            self.reference_loss_aversion = True
        else:
            self.reference_loss_aversion = False

    
        self.num_sessions = kwargs.get('num_sessions', 100)
        self.tstable = kwargs.get('tstable', 1e5)
        self.tmax = kwargs.get('tmax', 1e7)
        self.aprint = kwargs.get('aprint', True)

        # Derived state and action space
        self.adim, self.sdim, self.s0 = self.init_state()
        self.p_minmax = self.compute_p_competitive_monopoly()
        demand_type = self.demand_type
        self.demand_type = 'noreference'
        self.pmin_max_noreference = self.compute_p_competitive_monopoly()
        self.demand_type = 'reference'
        self.pmin_max_reference = self.compute_p_competitive_monopoly()
        self.demand_type = demand_type
        

        # Compute Nash and cooperative prices and profits
        self.NashProfits,  self.CoopProfits = self.compute_profits_nash_coop()
        self.p_nash, self.p_coop = self.p_minmax[0], self.p_minmax[1]
        self.A, self.R = self.init_actions()
        self.PI = self.init_PI()
        self.PG = self.init_PG()
        self.Q = self.init_Q()


        # Derived properties
        if self.demand_type == 'noreference':
            self.num_states = self.k ** (self.n * self.memory)
        if self.demand_type == 'reference':
            self.num_states = self.k ** (self.n * self.memory) * self.k 

        self.num_actions = self.k ** self.n
        self.num_periods = self.num_states + 1

        # Game logs 
        # Initialize all the variables with zeros
        self.converged = np.zeros(self.num_sessions, dtype=bool)  # Convergence status
        self.time_to_convergence = np.zeros(self.num_sessions, dtype=float)  # Time to convergence
        # Initialize with same shape as Q but without the last axis (k)
        self.index_strategies = np.zeros((self.n,) + self.sdim + (self.num_sessions,), dtype=int)
        self.index_last_state = np.zeros((self.n, self.memory, self.num_sessions), dtype=int)  # Last states
        self.index_last_reference = np.zeros((1, self.num_sessions),  dtype=int)
        self.cycle_length = np.zeros(self.num_sessions, dtype=int)  # Cycle length
        self.cycle_states = np.zeros((self.num_periods, self.num_sessions), dtype=int)  # Cycle states
        self.cycle_prices = np.zeros((self.n, self.num_periods, self.num_sessions), dtype=float)  # Cycle prices
        self.cycle_reference_prices = np.zeros((self.num_periods, self.num_sessions), dtype=float)  # Cycle ref prices
        self.cycle_consumer_surplus = np.zeros((self.num_periods, self.num_sessions), dtype=float)  # Cycle consumer surplus
        self.cycle_profits = np.zeros((self.n, self.num_periods, self.num_sessions), dtype=float)  # Cycle profits
        self.index_actions = np.zeros((self.num_actions, self.n), dtype=int)  # Action indices
        self.profit_gains = np.zeros((self.num_actions, self.n), dtype=float)  # Profit gains
        self.last_observed_prices = np.zeros((self.n, self.memory), dtype=int)  # last prices
        self.last_observed_reference = np.zeros(1, dtype=int)  # last prices
        self.last_reference_observed_prices = np.zeros((self.n, self.reference_memory), dtype=int)  # last prices
        self.last_observed_demand = np.zeros((self.n, self.reference_memory), dtype=float)  # last shares for each firm

    def reference_price(self, p):
        """
        Compute the reference price `r` based on all firms' prices.
        Default: Simple average of all prices.
        """
        if self.ref_type == "average":
            return np.mean(p)  # Average price
        elif self.ref_type == "min":
            return np.min(p)  # Minimum price (e.g., price leader effect)
        elif self.ref_type == "max":
            return np.max(p)  # Maximum price (e.g., anchoring effect)
        else:
            raise ValueError("Invalid ref_type. Choose from 'average', 'min', 'max'.")
    
    ### Calculating demand function 
    def demand(self, p, r=None):
        """Computes demand, optionally using a given reference price"""
        
        if self.demand_type == 'noreference':
            e = np.exp((self.a - p) / self.mu)
            d = e / (np.sum(e) + np.exp(self.a0 / self.mu))

        elif self.demand_type == 'reference':
            """
            Logit demand with reference dependence.

            Parameters:
            -----------
            p : array-like of shape (n,)
                Prices set by each firm.
            r : scalar or array-like of shape (1,), optional
                Reference price. If None, it is computed from `self.reference_price(p)`.
                
            Returns:
            --------
            d : array of demands [d1, ..., dn].
            """
            if r is None:
                r = self.reference_price(p)  # Compute reference price if not provided

            # Effective price: p_i^eff = p_i + gamma * (p_i - r)
            p_eff = p + self.gamma * (p - r)  # elementwise adjustment

            if self.reference_loss_aversion:
                # Compute loss aversion adjustment
                price_above_r = p >= r  # Boolean array: True if p_i > r
                price_below_r = p < r  # Boolean array: True if p_i < r

                # Loss aversion effect: Consumers react strongly if p > r
                p_eff = (
                    p + self.gamma * (p - r) * price_below_r  # Standard reference adjustment
                    + self.lossaversion * self.gamma * (p - r) * price_above_r  # Stronger penalty if price > r
                )

            # Logit exponent: exp((a_i - p_eff_i)/mu)
            e = np.exp((self.a - p_eff) / self.mu)
            
            # Compute demand shares
            d = e / (np.sum(e) + np.exp(self.a0 / self.mu))

        return d

    

    ###### FOC 
    def foc(self, p):
        """Compute first order condition"""
        d = self.demand(p)
        if self.demand_type == 'noreference':
            zero = 1 - (p - self.c) * (1 - d) / self.mu
        if self.demand_type == 'reference':
            if self.reference_loss_aversion:
                p_c = p.copy()
                r  = self.reference_price(p_c)
                # Compute smooth price indicators
                price_above_r = 1 / (1 + np.exp(-10 * (p - r)))  # Smooth transition for p > r
                price_below_r = 1 / (1 + np.exp(10 * (p - r)))   # Smooth transition for p < r

                zero = 1 - (1 + self.gamma * price_below_r  + self.gamma * self.lossaversion * price_above_r) * (p - self.c) * (1 - d) / self.mu  # Adjusted FOC
            else: 
                zero = 1 - (1 + self.gamma) * (p - self.c) * (1 - d) / self.mu

        return np.squeeze(zero)
    
    def foc_monopoly(self, p):
        """Compute first-order condition for collusion with n firms (Vectorized)"""
        d = self.demand(p)  # Compute market shares (shape: (n,))
        
        total_contribution = np.sum((p - self.c) * d)  # Sum over all firms
        own_contribution = (p - self.c) * d  # Each firm's individual term

        if self.demand_type == 'noreference':
            zero = 1 - ((p - self.c) * (1 - d) / self.mu) + ((total_contribution - own_contribution) / self.mu)
        if self.demand_type == 'reference':
            if self.reference_loss_aversion:
                p_c = p.copy()
                r  = self.reference_price(p_c)
                # Compute smooth price indicators
                price_above_r = 1 / (1 + np.exp(-10 * (p - r)))  # Smooth transition for p > r
                price_below_r = 1 / (1 + np.exp(10 * (p - r)))   # Smooth transition for p < r

                zero = 1 - (1 + self.gamma * price_below_r  + self.gamma * self.lossaversion * price_above_r) * ((p - self.c) * (1 - d) / self.mu) + (1 + self.gamma * price_below_r  + self.gamma * self.lossaversion * price_above_r) * ((total_contribution - own_contribution) / self.mu)
            else: 
                zero = 1 - (1 + self.gamma) * ((p - self.c) * (1 - d) / self.mu) + (1 + self.gamma) * ((total_contribution - own_contribution) / self.mu)
        return np.squeeze(zero)
    
    def compute_p_competitive_monopoly(self):
        """Computes competitive and monopoly prices"""
        p0 = np.ones((1, self.n)) * 3 * self.c
        p_competitive = fsolve(self.foc, p0)
        p_monopoly = fsolve(self.foc_monopoly, p0)
        return p_competitive, p_monopoly
    
    
    def compute_profits_nash_coop(self):
        """Compute Nash and cooperative profits for the agents."""
        # Extract Nash and cooperative prices
        NashPrices, CoopPrices  = self.p_minmax 

        # Compute Nash market shares using demand function
        NashMarketShares = self.demand(NashPrices)
        # Compute Nash profits
        NashProfits = (NashPrices - self.c) * NashMarketShares

        # Compute cooperative market shares using demand function
        CoopMarketShares = self.demand(CoopPrices)
        # Compute cooperative profits
        CoopProfits = (CoopPrices - self.c) * CoopMarketShares

        return NashProfits, CoopProfits


    def compute_profit_gain(self):
        """
        Compute profit gains for each agent.
        
        Profit gain is calculated as:
        PG(s, i) = (PI(s, i) - NashProfits(i)) / (CoopProfits(i) - NashProfits(i))
        
        Returns
        -------
        profit_gain : ndarray
            A matrix of profit gains for all states and agents.
        avg_profit_gain : ndarray
            The average profit gain across all states for each agent.
        """
        # Ensure PI, NashProfits, and CoopProfits are initialized
        if not hasattr(self, 'PI') or not hasattr(self, 'NashProfits') or not hasattr(self, 'CoopProfits'):
            raise ValueError("PI, NashProfits, and CoopProfits must be initialized before computing profit gain.")

        # Calculate profit gains
        profit_gain = np.zeros(self.PI.shape)  # Same shape as PI (k^n x n)
        for i in range(self.n):  # Loop over agents
            profit_gain[:, :, i] = (self.PI[:, :, i] - self.NashProfits[i]) / (self.CoopProfits[i] - self.NashProfits[i])
        
        # Calculate average profit gain across states for each agent
        avg_profit_gain = np.mean(profit_gain, axis=2)  # Average across all states
        
        return avg_profit_gain
    

    def init_actions(self):
        """
        Get action space of the firms and reference prices.
        
        Returns
        -------
        A : ndarray
            Discretized set of feasible prices.
        R : ndarray
            Discretized set of reference prices.
        """
        type = 2
        if type == 1:
            # Compute the lower and upper bounds of the price grid
            p_nash, p_coop = min(self.p_minmax[0]), max(self.p_minmax[1])

        if type == 2:
            # Compute the lower and upper bounds considering both demand types
            p_nash = np.min([np.min(self.pmin_max_noreference), np.min(self.pmin_max_reference)])
            p_coop = np.max([np.max(self.pmin_max_noreference), np.max(self.pmin_max_reference)])
        
        # Calculate bounds
        lower_bound = p_nash - self.extend * (p_coop - p_nash)
        upper_bound = p_coop + self.extend * (p_coop - p_nash)

        # Ensure bounds are non-negative
        lower_bound = np.maximum(0, lower_bound)
        upper_bound = np.maximum(0, upper_bound)

        # Create the price grid for each agent
        A = np.linspace(lower_bound, upper_bound, self.k)

        # Create reference price grid (same as price grid)
        R = np.linspace(lower_bound, upper_bound, self.k)

        return A, R
    
    def init_state(self):
        """Get state dimension and initial state"""
        """Each Player action space is a grid of k points(prices)"""
        if self.demand_type == 'noreference':
            sdim = tuple([self.k] * (self.n * self.memory))
            adim = tuple([self.k] * self.n)
        if self.demand_type == 'reference':
            # State space includes both past prices and a reference price
            sdim = tuple([self.k] * (self.n * self.memory)) + (self.k,)  # Last value for reference price
            adim = tuple([self.k] * (self.n))
       
        s0 = np.zeros(len(sdim)).astype(int)
        return adim, sdim, s0

    def compute_profits(self, p, r=None):
        """Compute payoffs considering reference price"""
        
        if self.demand_type == 'noreference':
            d = self.demand(p)  # Demand without reference
        elif self.demand_type == 'reference':
            if r is None:
                r = self.reference_price(p)  # Compute reference price if not provided
            d = self.demand(p, r)  # Demand with reference price

        pi = (p - self.c) * d  # Compute profits
        return pi


    
    def init_PI(game):
        """Initialize Profits (actions x reference prices x agents)"""
        
        if game.demand_type == 'noreference':
            # Profits depend only on prices (adim, n)
            PI = np.zeros(game.adim + (game.n,))
            for a in product(*[range(i) for i in game.adim]):  # Iterate over actions
                p = np.asarray(game.A[np.asarray(a)])  # Convert action indices to prices
                # Convert action index to prices
                PI[a] = game.compute_profits(p)  # Compute profits
        
        elif game.demand_type == 'reference':
            # Profits depend on both prices and reference price
            PI = np.zeros(game.adim + (game.k, game.n))  
            # Shape: (actions, reference price, agents)

            for a in product(*[range(i) for i in game.adim]):  # Iterate over all actions
                p = np.asarray(game.A[np.asarray(a)])  # Convert action indices to prices
                
                for r_idx, r in enumerate(game.R):  # Iterate over actual reference prices
                    PI[a + (r_idx,)] = game.compute_profits(p, r)  # Compute profits
        return PI
    
    def init_PG(game):
        """
        Compute profit gains for each agent.
        
        PG(s, i) = (PI(s, i) - NashProfits(i)) / (CoopProfits(i) - NashProfits(i))
        
        Parameters
        ----------
        game : Model
            The initialized game object containing PI, NashProfits, and CoopProfits.

        Returns
        -------
        profit_gain : ndarray
            A matrix of profit gains with the same dimensions as PI.
        """
        # Ensure PI, NashProfits, and CoopProfits are initialized
        if not hasattr(game, 'PI') or not hasattr(game, 'NashProfits') or not hasattr(game, 'CoopProfits'):
            raise ValueError("PI, NashProfits, and CoopProfits must be initialized before computing profit gain.")

        # Initialize profit gain matrix
        profit_gain = np.zeros(game.PI.shape)

        # Compute profit gain for each agent
        for i in range(game.n):  # Loop over agents
            profit_gain[..., i] = (game.PI[..., i] - game.NashProfits[i]) / (game.CoopProfits[i] - game.NashProfits[i])
        
        return profit_gain

    def init_Q(game):
        """Initialize Q function (n x #s x k)"""
        Q = np.zeros((game.n,) + game.sdim + (game.k,))
        for n in range(game.n):
            # pi = np.mean(game.PI[:, :, n], axis=1 - n)
            # Q[n] = np.tile(pi, game.sdim + (1,)) / (1 - game.delta)

            # Define other agents (all agents except n)
            other_agents = [i for i in range(game.n) if i != n]

            # **Handle reference price**
            if game.demand_type == 'noreference':
                
                # Compute expected profits by summing over other agents' actions
                pi_summed = np.sum(
                    game.PI.take(indices=n, axis=-1),  # Select profits for agent n
                    axis=tuple([j for j in other_agents])  # Sum over other agents
                )

                # Normalize by the number of possible actions for other agents
                num_actions_other_agents = np.prod([game.k for _ in other_agents])  # |A|^(n-1)

            elif game.demand_type == 'reference':
                # Sum over other agents' actions + reference price
                pi_summed = np.sum(
                    game.PI.take(indices=n, axis=-1),  # Extract agent n's profits
                    axis=tuple(other_agents) + (-1,)  # Sum over all other agents' actions + reference price
                )
                num_actions_other_agents = np.prod([game.k for _ in other_agents]) * game.k  # Normalize for reference

            # Normalize Q-values
            Q[n] = pi_summed / ((1 - game.delta) * num_actions_other_agents)
                
        return Q
    