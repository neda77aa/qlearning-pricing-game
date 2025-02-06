"""
Q-learning Functions
"""

import sys
import numpy as np

def generate_init_state(game):
    """
    Generates a random initial state for price setup.

    Parameters:
    -----------
    game : object
        The game instance containing information about the price grid and state dimensions.

    Returns:
    --------
    init_state : np.ndarray
        A tuple representing the randomly selected initial state, with indices corresponding to price levels.
    """
    # Generate random numbers between 0 and 1 for each dimension of the state space
    u_init_price = np.random.uniform(low=0, high=1, size=(len(game.sdim)))

    # Scale by number of price levels (k) and convert to integer indices
    init_state = (u_init_price * game.k).astype(int)  # Assuming game.k is the number of price levels

    return init_state



def pick_strategies(game, s, t):
    """Pick strategies by exploration vs exploitation"""
    a = np.zeros(game.n).astype(int)
    pr_explore = np.exp(- t * game.beta)
    e = (pr_explore > np.random.rand(game.n))
    for n in range(game.n):
        if e[n]:
            a[n] = np.random.randint(0, game.k)
        else:
            a[n] = np.argmax(game.Q[(n,) + tuple(s)])
    return a


def update_q(game, s, a, sprime, pi, stable):
    """Update Q matrix"""
    for n in range(game.n):
        subj_state = (n,) + tuple(s) + (a[n],)
        old_value = game.Q[subj_state]
        max_qprime = np.max(game.Q[(n,) + tuple(sprime)])
        new_value = pi[n] + game.delta * max_qprime
        old_argmax = np.argmax(game.Q[(n,) + tuple(s)])
        game.Q[subj_state] = (1 - game.alpha) * old_value + game.alpha * new_value
        # Check stability
        new_argmax = np.argmax(game.Q[(n,) + tuple(s)])
        same_argmax = (old_argmax == new_argmax)
        stable = (stable + same_argmax) * same_argmax
    return game.Q, stable

def check_convergence(game, t, stable):
    """Check if game converged"""
    if (t % game.tstable == 0) & (t > 0):
        sys.stdout.write("\rt=%i" % t)
        sys.stdout.flush()
    if stable > game.tstable:
        print('Converged!')
        return True
    if t == game.tmax:
        print('ERROR! Not Converged!')
        return True
    return False


def simulate_game(game):
    """Simulate game"""
    s = generate_init_state(game)

    # Initialize last observed prices 
    game.last_observed_prices = np.reshape(s, (game.n, game.memory))

    #s = game.s0
    stable = 0
    converged = False
    # Iterate until convergence
    for t in range(int(game.tmax)):
        a = pick_strategies(game, s, t)
        pi = game.PI[tuple(a)]

        # **Update Last Observed Prices**
        game.last_observed_prices[:, 1:] = game.last_observed_prices[:, :-1]  # Shift old prices
        game.last_observed_prices[:, 0] = a  # Insert new prices at first position

        # **Update State**
        sprime = game.last_observed_prices.flatten()  # Flatten to match state shape
        #sprime = a
        game.Q, stable = update_q(game, s, a, sprime, pi, stable) # Q-learning update
        s = sprime
        if check_convergence(game, t, stable):
            converged = True
            break
    return game, converged, t


def run_sessions(game):
    """
    Runs multiple learning sessions, calling simulate_game() for each session.

    Parameters:
    -----------
    game : object
        The game instance containing learning parameters and state space.

    Returns:
    --------
    game : object
        The updated game instance after running multiple sessions.
    """
    for iSession in range(game.num_sessions):
        print(f"\nStarting Session {iSession + 1}/{game.num_sessions}")

        # Run Q-learning for this session
        game, converged, t_convergence = simulate_game(game)

        # Store convergence results
        game.converged[iSession] = converged
        game.time_to_convergence[iSession] = t_convergence

        # If converged, analyze post-convergence cycles
        if converged:
            post_convergence_analysis(game)

    print("\nAll sessions completed.")
    return game


def detect_cycle(game, optimal_strategy):
    """
    Detects cycles in observed price states.

    Parameters:
    -----------
    game : object
        Game instance containing state information.
    optimal_strategy : np.ndarray
        The optimal strategy matrix for each state.

    Returns:
    --------
    cycle_length : int
        The detected cycle length.
    visited_states : np.ndarray
        The states visited before cycle detection.
    visited_profits : np.ndarray
        The profits recorded before cycle detection.
    """
    visited_states = []
    visited_profits = []
    p_hist = []
    
    # Initialize with last observed prices
    p = np.copy(game.last_observed_prices)
    
    # Compute initial state and action
    state = game.compute_state_number(p)
    action = optimal_strategy[state]
    
    for iPeriod in range(game.num_periods):
        # Update price history
        p_hist.append(action)
        
        # Compute new state number and record it
        state = game.compute_state_number(p)
        visited_states.append(state)
        
        # Compute profit and store
        profit = game.PI[tuple(action)]
        visited_profits.append(profit)
        
        # Check for cycle (if state repeats)
        if state in visited_states[:-1]:  # Ignore the last entry (current state)
            first_occurrence = visited_states.index(state)
            cycle_length = iPeriod - first_occurrence
            return cycle_length, np.array(visited_states), np.array(visited_profits)
        
        # Update action based on strategy
        action = optimal_strategy[state]
    
    return game.num_periods, np.array(visited_states), np.array(visited_profits)  # If no cycle detected, return full period





def save_convergence_results(game, iSession, cycle_length, visited_states, visited_profits):
    """
    Saves the results of convergence analysis.

    Parameters:
    -----------
    game : object
        Game instance containing session data.
    iSession : int
        The session number.
    cycle_length : int
        The length of the detected cycle.
    visited_states : np.ndarray
        The sequence of visited states before convergence.
    visited_profits : np.ndarray
        The profits recorded before convergence.
    """
    filename = "convergence_results.txt"

    with open(filename, "a") as file:
        file.write(f"Session {iSession}\n")
        file.write(f"Converged: {game.converged[iSession]}\n")
        file.write(f"Time to Convergence: {game.time_to_convergence[iSession]}\n")
        file.write(f"Cycle Length: {cycle_length}\n")
        file.write(f"Visited States: {visited_states.tolist()}\n")
        file.write(f"Visited Profits: {visited_profits.tolist()}\n")
        file.write(f"Optimal Strategy: {game.index_strategies[:, iSession].tolist()}\n\n")
