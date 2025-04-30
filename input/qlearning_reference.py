import sys
import numpy as np

import numpy as np
from itertools import product

class ConsumerQReference:
    def __init__(self, n_firms, k, memory, common_reference=True, alpha=0.1, beta = 0.1 / 2500, delta=0.95):
        """
        Initializes the consumer Q-learning reference model.

        Parameters:
        -----------
        n_firms : int
            Number of firms
        k : int
            Number of discrete price levels
        memory : int
            Number of past time steps to consider
        common_reference : bool
            Whether a single reference price is used for all firms
        alpha : float
            Learning rate
        delta : float
            Discount factor
        """
        self.n = n_firms
        self.k = k
        self.memory = memory
        self.alpha = alpha
        self.beta = beta 
        self.delta = delta
        self.common_reference = common_reference

        self.sdim = tuple([k] * (self.n * self.memory))  # State: last m prices of all firms
        self.adim = (k,) if self.common_reference else tuple([k] * self.n)  # Action: 1 price or n prices

        # Initialize Q-table: shape = (|S| x |A|), as a dictionary for sparse storage or as array
        self.Q = self.init_Q_with_prior()



    def init_Q_with_prior(self):
        q_table = np.zeros(self.sdim + self.adim)

        for state in np.ndindex(self.sdim):
            # Decode past price indices per firm from the state
            past_price_indices = np.array(state).reshape(self.n, self.memory)

            if self.common_reference:
                avg_index = np.mean(past_price_indices)

                for action_idx in range(self.k):
                    dist = (action_idx - avg_index) ** 2
                    q_table[state + (action_idx,)] = -dist

            else:
                # Compute average index separately for each firm
                avg_indices = np.mean(past_price_indices, axis=1)  # shape: (n,)
                for action_idx in range(np.prod(self.adim)):
                    # Action index is a flat index over (k,) * n shape
                    action_tuple = np.unravel_index(action_idx, self.adim)  # shape: (n,)
                    dist = np.mean([(a_i - avg_i) ** 2 for a_i, avg_i in zip(action_tuple, avg_indices)])
                    q_table[state + action_tuple] = -dist

        return q_table

    
    def decode_state(self, state_idx):
        """Convert flat state index into matrix of past prices (as indices): shape (n, memory)"""
        flat_state = np.array(state_idx)
        return flat_state.reshape(self.n, self.memory)


    def encode_state(self, price_history):
        """Encodes a (n x memory) matrix into a Q-table state tuple"""
        return tuple(price_history.flatten())


    def get_action_index(self, predicted_price):
        """Convert predicted price(s) back to an action index"""
        if self.common_reference:
            return predicted_price  # scalar
        else:
            return tuple(predicted_price)

    def predict(self, price_history, t, cycle = False):
        """Predict reference price using epsilon-greedy strategy with time-based exponential decay"""
        
        # assuming the prices are shifted in each step
        recent_history = price_history[:, :self.memory]  # already ordered from newest to oldest
        state = self.encode_state(recent_history)

        # Exploration probability decays over time
        pr_explore = np.exp(- self.beta * t)

        if cycle:
            pr_explore = 0 

        # Choose action
        if np.random.rand() < pr_explore:
            if self.common_reference:
                action_idx = np.random.randint(self.k)
            else:
                action_idx = np.random.randint(0, self.k, size=self.n)
        else:
            action_values = self.Q[state]
            action_idx = np.argmax(action_values)

            if not self.common_reference:
                # Decode back to (n,) shape if needed
                action_idx = np.unravel_index(action_idx, [self.k] * self.n)
                action_idx = np.array(action_idx) 

        return action_idx


    def update(self, predicted_price, s, actual_price, sprime):
        """Q-learning update step"""
        # Encode current and next states
        current_state = self.encode_state(s)
        next_state = self.encode_state(sprime)

        # Get action index
        action_idx = self.get_action_index(predicted_price)

        # Decode to action tuple
        if self.common_reference:
            action_tuple = (action_idx,)  # scalar action
        else:
            action_tuple = np.unravel_index(action_idx, [self.k] * self.n)  # tuple of length n

        # Compute reward
        if self.common_reference:
            reward = -((predicted_price - np.mean(actual_price)) ** 2)
        else:
            reward = -np.mean((np.array(predicted_price) - np.array(actual_price)) ** 2)

        # Q-learning update
        current_Q = self.Q[current_state + action_tuple]
        next_Q_max = np.max(self.Q[next_state])
        self.Q[current_state + action_tuple] = (
            (1 - self.alpha) * current_Q + self.alpha * (reward + self.delta * next_Q_max)
        )