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


def compute_reference_price(game, last_reference_price, last_reference_observed_prices, last_observed_demand):
    """
    Computes the reference price as a single value by averaging past observed prices across firms
    and weighting them by their demand shares.

    Parameters:
    -----------
    game : object
        The game instance containing observed prices and demands.

    Returns:
    --------
    ref_price : int
        The computed reference price (rounded to the nearest discrete price index).
        If common_reference=True, returns single integer.
        If common_reference=False, returns ndarray with shape (n,).
    """
    if game.common_reference:
        # Weighted average by demand
        weighted_prices = last_reference_observed_prices * last_observed_demand
        total_weighted_prices = np.sum(weighted_prices, axis=0) / np.sum(last_observed_demand, axis=0)
        ref_price_continuous = (game.lambda_ * last_reference_price +
                                (1 - game.lambda_) * np.mean(total_weighted_prices))
        ref_price_index = int(np.clip(np.round(ref_price_continuous), 0, game.k - 1))

    else:
        # Individual exponential smoothing (no weighting by demand)
        ref_price_continuous = (game.lambda_ * last_reference_price +
                                (1 - game.lambda_) * np.mean(last_reference_observed_prices, axis=1))
        ref_price_index = np.clip(np.round(ref_price_continuous), 0, game.k - 1).astype(int)

    return ref_price_index

def compute_reference_price_corr(game, price_history, last_reference_price, last_reference_observed_prices, last_observed_demand, lambda_smooth=0.5, cycle_threshold=0.9):
    """
    Updates reference price considering potential cycles.
    - price_history: list of recent prices
    """
    if price_history.shape[1] < 10:
        # Element-wise multiplication of past prices and past demands
        weighted_prices = last_reference_observed_prices * last_observed_demand  # Shape: (n, reference_memory)

        # Sum across firms (axis=0) to get total market-weighted prices for each past period
        total_weighted_prices = np.sum(weighted_prices, axis=0)/ np.sum(last_observed_demand, axis = 0)  # Shape: (reference_memory,)

        # Compute the average over reference_memory periods
        ref_price_continuous = game.lambda_ *  last_reference_price + (1- game.lambda_) * np.mean(total_weighted_prices)  # Single scalar value

        # Round to the nearest valid price index (ensuring it stays within [0, game.k-1])
        ref_price_index = int(np.clip(np.round(ref_price_continuous), 0, game.k - 1))

    else:
        # Enough history: check for cycle
        recent_sequence = price_history[:, -5:]   # shape (n, 5)
        previous_sequence = price_history[:, -10:-5]  # shape (n, 5)

        # Flatten arrays for correlation check across all prices jointly
        recent_flat = recent_sequence.flatten()
        previous_flat = previous_sequence.flatten()

        # Check correlation to detect cycle
        correlation = np.corrcoef(recent_flat, previous_flat)[0, 1]

        if correlation > cycle_threshold:
            print('corr')
            # Recognized cycle: average all recent prices in the cycle period
            rp = np.mean(recent_sequence)
        else:
            # No clear cycle: use smoothing based on the mean of firms' prices
            # Element-wise multiplication of past prices and past demands
            weighted_prices = last_reference_observed_prices * last_observed_demand  # Shape: (n, reference_memory)

            # Sum across firms (axis=0) to get total market-weighted prices for each past period
            total_weighted_prices = np.sum(weighted_prices, axis=0)/ np.sum(last_observed_demand, axis = 0)  # Shape: (reference_memory,)

            # Compute the average over reference_memory periods
            rp = game.lambda_ *  last_reference_price + (1- game.lambda_) * np.mean(total_weighted_prices)  # Single scalar value


        # Convert to discrete price index
        ref_price_index = int(np.clip(np.round(rp), 0, game.k - 1))

    return ref_price_index






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
        if game.aprint:
            sys.stdout.write("\rt=%i" % t)
            sys.stdout.flush()
    if stable > game.tstable:
        if game.aprint:
            print('Converged!')
        return True
    if t == game.tmax:
        if game.aprint:
            print('ERROR! Not Converged!')
        return True
    return False


def simulate_game(game):
    """Simulate game"""
    s = generate_init_state(game)

    # Initialize last observed prices 
    game.last_observed_prices = np.reshape(s[:game.n * game.memory], (game.n, game.memory)).copy()  

    if game.demand_type in ["reference", "misspecification"]:
        # **Initialize reference price tracking**
        initial_price = s[:game.n].reshape(game.n, 1)  # Extract initial prices
        game.last_reference_observed_prices = np.tile(initial_price, (1, game.reference_memory))  # Fill all reference slots

        # **Initialize observed demands (equal share assumption)**
        game.last_observed_demand[:, :] = 1 / game.n  # Set initial demands as equal weights


    #s = game.s0
    stable = 0
    converged = False
    # Iterate until convergence
    for t in range(int(game.tmax)):
        a = pick_strategies(game, s, t)
        if game.demand_type == 'noreference':
            pi = game.PI[tuple(a)]
        if game.demand_type == 'reference':
            if game.common_reference:
                # Single common reference price (last element)
                pi = game.PI[tuple(np.append(a, s[-1]))]
            else:
                # Individual reference prices per firm (last n elements)
                pi = game.PI[tuple(np.concatenate([a, s[-game.n:]]))]
        if game.demand_type == 'misspecification':
            if game.common_reference:
                # single common reference price
                pi = game.PI[tuple(np.append(a, game.last_observed_reference))]
            else:
                # each firm has individual reference prices
                # Use each firm's reference price from game.last_observed_reference array
                pi_indices = tuple(np.concatenate([a, game.last_observed_reference]))
                pi = game.PI[pi_indices]

        # **Update Last Observed Prices**
        game.last_observed_prices[:, 1:] = game.last_observed_prices[:, :-1]  # Shift old prices
        game.last_observed_prices[:, 0] = a  # Insert new prices at first position
        
        if game.demand_type in ["reference", "misspecification"]:
            game.last_reference_observed_prices[:, 1:] = game.last_reference_observed_prices[:, :-1]  # Shift old prices
            game.last_reference_observed_prices[:, 0] = a  # Insert new prices at first position
            game.last_observed_demand[:, 1:] = game.last_observed_demand[:, :-1]
            p = np.asarray(game.A[np.asarray(a)])
            ref = np.asarray(game.R[np.asarray(game.last_observed_reference)])
            game.last_observed_demand[:, 0] = game.demand(p, ref)
            game.last_observed_reference = compute_reference_price(game, game.last_observed_reference, game.last_reference_observed_prices, game.last_observed_demand)
            # Update price history
            game.price_history = np.hstack((game.price_history, a.reshape(game.n, 1)))  # appending new prices
            #game.last_observed_reference_corr = compute_reference_price_corr(game, game.price_history, game.last_observed_reference, game.last_reference_observed_prices, game.last_observed_demand)

        # **Update State**
        if game.demand_type == 'noreference':
            sprime = game.last_observed_prices.flatten()

        elif game.demand_type == 'reference':
            if game.common_reference:
                # single reference price
                sprime = np.append(game.last_observed_prices.flatten(), game.last_observed_reference)
            else:
                # individual reference prices per firm
                sprime = np.concatenate([game.last_observed_prices.flatten(), game.last_observed_reference.flatten()])

        elif game.demand_type == 'misspecification':
            sprime = game.last_observed_prices.flatten()


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
        if game.aprint:
            print(f"\nStarting Session {iSession + 1}/{game.num_sessions}")

        game.Q = game.init_Q()  # Reset Q-values
        game.last_observed_prices = np.zeros((game.n, game.memory), dtype=int)  # Reset prices

        if game.demand_type in ["reference", "misspecification"]:
            game.last_reference_observed_prices = np.zeros((game.n, game.reference_memory), dtype=int)  # last prices
            game.last_observed_demand = np.zeros((game.n, game.reference_memory), dtype=float)  # last shares for each firm

        # Run Q-learning for this session
        game, converged, t_convergence = simulate_game(game)

        # Store convergence results
        game.converged[iSession] = converged
        game.time_to_convergence[iSession] = t_convergence

        # Store last observed prices
        game.index_last_state[:, :,iSession] = game.last_observed_prices

        # Store the learned strategies (optimal strategies at convergence)
        game.index_strategies[..., iSession] = game.Q.argmax(axis=-1)

        if game.demand_type == 'reference':
            game.index_last_reference[:, iSession] = game.last_observed_reference

        # If converged, analyze post-convergence cycles
        if converged:
            if game.demand_type == 'noreference':
                # Pass iSession to detect_cycle function
                cycle_length, visited_states, visited_profits, price_history, _, consumer_surplus_history = detect_cycle(game, iSession)  # Now passing iSession
                cycle_data = {
                    'cycle_length': cycle_length,
                    'visited_states': visited_states,
                    'visited_profits': visited_profits,
                    'price_history': price_history,
                    'consumer_surplus_history': consumer_surplus_history
                }
            if game.demand_type in ["reference", "misspecification"]:
                # Pass iSession to detect_cycle function
                cycle_length, visited_states, visited_profits, price_history, reference_price_history, consumer_surplus_history = detect_cycle(game, iSession)  # Now passing iSession
                cycle_data = {
                    'cycle_length': cycle_length,
                    'visited_states': visited_states,
                    'visited_profits': visited_profits,
                    'price_history': price_history,
                    'reference_price_history': reference_price_history,
                    'consumer_surplus_history': consumer_surplus_history
                }

        #post_convergence_analysis(game)

    print("\nAll sessions completed.")
    return game


def compute_consumer_surplus(game, prices, r):
    """
    Computes consumer surplus given current prices and reference price.
    """
    if game.demand_type == 'noreference':
        e = np.exp((game.a - prices) / game.mu)
        surplus = game.mu * np.log(np.sum(e) + np.exp(game.a0 / game.mu))
    elif game.demand_type in ["reference", "misspecification"]:
        # Effective price: p_i^eff = p_i + gamma * (p_i - r)
        p_eff = prices + game.gamma * (prices - r)  # elementwise adjustment
        e = np.exp((game.a - p_eff) / game.mu)
        surplus = game.mu * np.log(np.sum(e) + np.exp(game.a0 / game.mu))
            
    return surplus


def detect_cycle(game, session_idx):
    """
    Detects cycles in the game states and computes related metrics.
    Updates game parameters with cycle information.
    
    Parameters:
    -----------
    game : object
        Game instance containing state information and parameters
    optimal_strategy : ndarray
        Array containing optimal strategies for each state
    session_idx : int
        Current session index for updating game parameters
    
    Returns:
    --------
    tuple:
        - cycle_length: Length of the detected cycle
        - visited_states: Array of states visited during simulation
        - visited_profits: Array of profits for each state visited
        - price_history: History of prices chosen by agents
        - consumer_surplus_history: History of consumer surplus values
    """
    # Initialize arrays to store visited states and profits
    visited_states = np.zeros(game.num_periods, dtype=int)
    visited_profits = np.zeros((game.n, game.num_periods))
    price_history = np.zeros((game.n, game.num_periods), dtype=int)
    demand_history = np.zeros((game.n, game.num_periods), dtype=int)
    if game.common_reference:
        reference_price_history = np.zeros((1, game.num_periods), dtype=int)
    else:
        reference_price_history = np.zeros((game.n, game.num_periods), dtype=int)
    consumer_surplus_history = np.zeros(game.num_periods)  # Store consumer surplus at each period
    
    # Initialize based on the demand type and common_reference
    p = np.copy(game.last_observed_prices)
    d = np.copy(game.last_observed_demand)

    if game.demand_type == 'noreference':
        r = np.zeros(1, dtype=int)
        state = p.flatten()
    
    if game.demand_type == 'reference':
        if game.common_reference:
            r = np.copy(game.last_observed_reference)  # Single value
            state = np.append(p.flatten(), r)
        else:
            r = np.copy(game.last_observed_reference)  # Vector of firm-specific reference prices
            state = np.concatenate([p.flatten(), r])
        
    if game.demand_type == 'misspecification':
        r = np.copy(game.last_observed_reference)
        state = p.flatten()
    
    # Initial optimal action from current state
    p_prime = game.index_strategies[(slice(None),) + tuple(state) + (session_idx,)]


    # Main loop for detecting cycles
    for i_period in range(game.num_periods):

        if game.demand_type == 'noreference':
            # Update price history
            if game.memory > 1:
                p[:, 1:] = p[:, :-1]  # Shift older prices up
            p[:, 0] = p_prime  # Update most recent prices

        
        if game.demand_type in ["reference", "misspecification"]:
            # Update price history
            if game.memory > 1:
                p[:, 1:] = p[:, :-1]  # Shift older prices up
                d[:, 1:] = d[:, :-1]  # Shift older demands up

            p[:, 0] = p_prime  # Update most recent prices
            d[:, 0] = game.demand(np.asarray(game.A[np.asarray(p_prime)]),np.asarray(game.R[np.asarray(r)])) 
            r = compute_reference_price(game, r, p, d)


        price_history[:, i_period] = p_prime
        
        if game.demand_type == 'noreference':
            # Record current state
            visited_states[i_period] = compute_state_number(game, p, 1)
            # Compute and record profits
            visited_profits[:, i_period] = game.PI[tuple(p_prime)]

        if game.demand_type == 'reference':
            reference_price_history[:, i_period] = r
            # Record current state
            visited_states[i_period] = compute_state_number(game, p, r)
            # Compute and record profits
            visited_profits[:, i_period] = game.PI[tuple(np.append(p_prime, r))]

        if game.demand_type == 'misspecification':
            reference_price_history[:, i_period] = r
            # Record current state
            visited_states[i_period] = compute_state_number(game, p, r)
            # Compute and record profits
            visited_profits[:, i_period] = game.PI[tuple(np.append(p_prime, r))]
        
        # **Compute Consumer Surplus**
        current_prices = np.asarray(game.A[np.asarray(p_prime)])  # Convert price indices to actual prices
        consumer_surplus_history[i_period] = compute_consumer_surplus(game, current_prices, np.asarray(game.R[np.asarray(r)]))


        # Check if we've seen this state before (cycle detection)
        if i_period >= 1:
            for prev_period in range(i_period):
                if visited_states[prev_period] == visited_states[i_period]:
                    # Cycle found
                    cycle_length = i_period - prev_period
                    
                    # Trim arrays to only include the cycle
                    cycle_start = i_period - cycle_length + 1
                    visited_states = visited_states[cycle_start:i_period + 1]
                    visited_profits = visited_profits[:, cycle_start:i_period + 1]
                    price_history = price_history[:, cycle_start:i_period + 1]
                    consumer_surplus_history = consumer_surplus_history[cycle_start:i_period + 1]
                    
                    # Update game parameters
                    game.cycle_length[session_idx] = cycle_length
                    
                    # Update cycle states (pad with zeros if needed)
                    game.cycle_states[:len(visited_states), session_idx] = visited_states

                    if game.demand_type in ["reference", "misspecification"]:
                        reference_price_history = reference_price_history[:, cycle_start:i_period + 1]
                        if game.common_reference:
                            game.cycle_reference_prices[0, :cycle_length, session_idx] = reference_price_history
                        else:
                            game.cycle_reference_prices[:, :cycle_length, session_idx] = reference_price_history

                    # Update cycle prices and profits
                    for i in range(game.n):
                        game.cycle_prices[i, :cycle_length, session_idx] = price_history[i, :cycle_length]
                        game.cycle_profits[i, :cycle_length, session_idx] = visited_profits[i, :cycle_length]
                    
                    game.cycle_consumer_surplus[:cycle_length, session_idx] = consumer_surplus_history
                    
                    if game.demand_type == 'noreference':
                        return cycle_length, visited_states, visited_profits, price_history, reference_price_history, consumer_surplus_history 
                    if game.demand_type in ["reference", "misspecification"]:
                        return cycle_length, visited_states, visited_profits, price_history, reference_price_history, consumer_surplus_history
                
        if game.demand_type == 'noreference':
            # Update p_prime for next iteration
            p_prime = game.index_strategies[(slice(None),) + tuple(p.flatten()) + (session_idx,)]

        if game.demand_type == 'reference':
            s = np.append(p.flatten(), r)
            # Get initial optimal actions from current state
            p_prime = game.index_strategies[(slice(None),) + tuple(s) + (session_idx,)]

        if game.demand_type == 'misspecification':
            # Get initial optimal actions from current state
            p_prime = game.index_strategies[(slice(None),) + tuple(p.flatten()) + (session_idx,)]
    
    # If no cycle found, update game parameters with full period data
    game.cycle_length[session_idx] = game.num_periods
    game.cycle_states[:, session_idx] = visited_states
    
    for i in range(game.n):
        game.cycle_prices[i, :, session_idx] = price_history[:, i]
        game.cycle_profits[i, :, session_idx] = visited_profits[:, i]
    
    game.cycle_consumer_surplus[:, session_idx] = consumer_surplus_history
    
    return game.num_periods, visited_states, visited_profits, price_history, reference_price_history, consumer_surplus_history


def compute_state_number(game, prices, reference_price):
    """
    Compute the state number from a price configuration.
    
    Parameters:
    -----------
    game : object
        Game instance containing grid parameters
    prices : ndarray
        Array of prices, shape (memory, n)
    
    Returns:
    --------
    int : State number
    """
    state = 0
    multiplier = 1
    
    # Flatten prices array and convert to index in state space
    flat_prices = prices.flatten()
    for i in range(len(flat_prices)):
        state += flat_prices[i] * multiplier
        multiplier *= game.k

    # Handle multiple reference prices
    if game.demand_type in ["reference", "misspecification"]:
        if game.common_reference:
            state += reference_price * multiplier
        else:
            for ref_price in reference_price:
                state += ref_price * multiplier
                multiplier *= game.k
    return state