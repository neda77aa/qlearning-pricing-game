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
        self.alpha = kwargs.get('alpha', 0.15)
        self.beta = kwargs.get('beta', 4e-6)
        self.delta = kwargs.get('delta', 0.95)
        self.c = kwargs.get('c', 1)
        self.a = kwargs.get('a', 2)
        self.a0 = kwargs.get('a0', 0)
        self.mu = kwargs.get('mu', 0.25)
        self.extend = kwargs.get('extend', 0.1)
        self.k = kwargs.get('k', 15)
        self.memory = kwargs.get('memory', 1)
        self.num_sessions = kwargs.get('num_sessions', 100)
        self.tstable = kwargs.get('tstable', 1e5)
        self.tmax = kwargs.get('tmax', 1e8)
        self.aprint = kwargs.get('aprint', True)

        # Derived state and action space
        self.adim, self.sdim, self.s0 = self.init_state()
        self.p_minmax = self.compute_p_competitive_monopoly()


        # Compute Nash and cooperative prices and profits
        self.NashProfits,  self.CoopProfits = self.compute_profits_nash_coop()
        self.A = self.init_actions()
        self.PI = self.init_PI()
        self.PG = self.init_PG()
        self.Q = self.init_Q()


        # Derived properties
        self.num_states = self.k ** (self.n * self.memory)
        self.num_actions = self.k ** self.n
        self.num_periods = self.num_states + 1

        # Game logs 
        # Initialize all the variables with zeros
        self.converged = np.zeros(self.num_sessions, dtype=bool)  # Convergence status
        self.time_to_convergence = np.zeros(self.num_sessions, dtype=float)  # Time to convergence
        # Initialize with same shape as Q but without the last axis (k)
        self.index_strategies = np.zeros((self.n,) + self.sdim + (self.num_sessions,), dtype=int)
        self.index_last_state = np.zeros((self.n, self.memory, self.num_sessions), dtype=int)  # Last states
        self.cycle_length = np.zeros(self.num_sessions, dtype=int)  # Cycle length
        self.cycle_states = np.zeros((self.num_periods, self.num_sessions), dtype=int)  # Cycle states
        self.cycle_prices = np.zeros((self.n, self.num_periods, self.num_sessions), dtype=float)  # Cycle prices
        self.cycle_profits = np.zeros((self.n, self.num_periods, self.num_sessions), dtype=float)  # Cycle profits
        self.index_actions = np.zeros((self.num_actions, self.n), dtype=int)  # Action indices
        self.profit_gains = np.zeros((self.num_actions, self.n), dtype=float)  # Profit gains
        self.last_observed_prices = np.zeros((self.n, self.memory), dtype=int)  # last prices




    def demand(self, p):
        """Computes demand"""
        e = np.exp((self.a - p) / self.mu)
        d = e / (np.sum(e) + np.exp(self.a0 / self.mu))
        return d
    
    # Implement later spacial case that mu = 0
    # def demand_mu0(self, p):
    #     """Computes demand"""
    #     e = np.exp((self.a - p) / self.mu)
    #     d = e / (np.sum(e) + np.exp(self.a0 / self.mu))
    #     return d

    def foc(self, p):
        """Compute first order condition"""
        d = self.demand(p)
        zero = 1 - (p - self.c) * (1 - d) / self.mu
        return np.squeeze(zero)

    def foc_monopoly(self, p):
        """Compute first order condition of a monopolist"""
        d = self.demand(p)
        d1 = np.flip(d)
        p1 = np.flip(p)
        zero = 1 - (p - self.c) * (1 - d) / self.mu + (p1 - self.c) * d1 / self.mu
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
        
        print(profit_gain.shape)
        # Calculate average profit gain across states for each agent
        avg_profit_gain = np.mean(profit_gain, axis=2)  # Average across all states
        
        return avg_profit_gain


    def init_actions_initial(self):
        """Get action space of the firms"""
        a = np.linspace(min(self.p_minmax[0]), max(self.p_minmax[1]), self.k - 2)
        delta = a[1] - a[0]
        A = np.linspace(min(a) - delta, max(a) + delta, self.k)
        return A
    
    def init_actions(self):
        """
        Get action space of the firms.
        
        Parameters
        ----------
        extend : tuple
            A tuple (extend1, extend2) that defines how much to extend the range
            beyond Nash and monopoly prices.
        
        Returns
        -------
        A : ndarray
            Discretized set of feasible prices.
        """
        # Compute the lower and upper bounds of the price grid
        p_nash, p_coop = min(self.p_minmax[0]), max(self.p_minmax[1])

        # Calculate bounds
        lower_bound = p_nash - self.extend * (p_coop - p_nash)
        upper_bound = p_coop + self.extend * (p_coop - p_nash)

        # Ensure bounds are non-negative
        lower_bound = np.maximum(0, lower_bound)
        upper_bound = np.maximum(0, upper_bound)

        # Create the price grid for each agent
        A = np.linspace(lower_bound, upper_bound, self.k)

        return A
    
    def init_state(self):
        """Get state dimension and initial state"""
        """Each Player action space is a grid of k points(prices)"""
        sdim = tuple([self.k] * (self.n * self.memory))
        adim = tuple([self.k] * self.n )
        #sdim = (self.k, self.k)
        s0 = np.zeros(len(sdim)).astype(int)
        return adim, sdim, s0

    def compute_profits(self, p):
        """Compute payoffs"""
        d = self.demand(p)
        pi = (p - self.c) * d
        return pi

    # def init_PI(game):
    #     """Initialize Profits (k^n x kp x n)"""
    #     PI = np.zeros(game.adim + (game.n,))
    #     for s in product(*[range(i) for i in game.sdim]):
    #         p = np.asarray(game.A[np.asarray(s)])
    #         PI[s] = game.compute_profits(p)
    #     return PI
    
    def init_PI(game):
        """Initialize Profits (k^n x kp x n)"""
        PI = np.zeros(game.adim + (game.n,))
        # Adjusted in initial code for memory
        for a in product(*[range(i) for i in game.adim]):
            p = np.asarray(game.A[np.asarray(a)])
            PI[a] = game.compute_profits(p)
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

            # Sum PI values for agent n over the other agents' action spaces
            pi_summed = np.sum(
                game.PI.take(indices=n, axis=-1),  # Select profits for agent n
                axis=tuple([j for j in other_agents])  # Sum over other agents
            )
            
            # Calculate Q values by dividing the summed profits by the size of the action space of other agents
            num_actions_other_agents = np.prod([game.k for _ in other_agents])  # |A|^(n-1)
            Q[n] = pi_summed / ((1 - game.delta) * num_actions_other_agents)
        return Q
    