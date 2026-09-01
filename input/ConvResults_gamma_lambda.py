import numpy as np
import pandas as pd
import os
from datetime import datetime
from input.qlearning import simulate_game, run_sessions, detect_cycle, pretrain_consumer_reference
import matplotlib.pyplot as plt
import os
from glob import glob
import multiprocessing as mp
from functools import partial
import copy
import secrets
import random

# When the collusive and competitive prices are essentially equal (market has
# collapsed to ~cost, e.g. price_sensitivity at high gamma), the price/profit
# gain denominators -> 0 and the gains blow up. In that case report the gains as
# NaN so they simply drop out of the figures. (Temporary guard; revisit later.)
_COLLAPSE_EPS = 1e-3


###############################################
######## Saving Experimat


class ExperimentSaver:
    def __init__(self, experiment_name):
        self.base_dir = "../Results/experiments"
        self.experiment_name = experiment_name
        self.experiment_dir = os.path.join(self.base_dir, experiment_name)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Add this line
        os.makedirs(self.experiment_dir, exist_ok=True)
        
    def _setup_directories(self):
        """Create necessary directories if they don't exist"""
        os.makedirs(self.experiment_dir, exist_ok=True)
            
    def get_run_dir(self, gamma, lambda_):
        """Create directory for specific gamma, lambda combination"""
        run_name = f"gamma_{gamma}_lambda_{lambda_}" #_{self.timestamp}"
        run_dir = os.path.join(self.experiment_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        return run_dir
    
    def get_run_dir_gd(self, gamma, delta):
        """Create directory for a specific gamma, delta combination"""
        run_name = f"gamma_{gamma}_delta_{delta}"
        run_dir = os.path.join(self.experiment_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    def get_run_dir_lossaversion(self, lossaversion):
        """Create directory for specific gamma, lambda combination"""
        run_name = f"lossaversion_{lossaversion}" #_{self.timestamp}"
        run_dir = os.path.join(self.experiment_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        return run_dir
    
    def get_run_dir_gamma_only(self, gamma):
        """Create directory for specific gamma, lambda combination"""
        run_name = f"gamma_{gamma}" #_{self.timestamp}"
        run_dir = os.path.join(self.experiment_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        return run_dir
        
    def save_experiment_config(self, game, run_dir):
        """Save experiment configuration to CSV"""
        config = {
            'Model': 1,  # Assuming this is always 1 based on your example
            'PrintQ': 0,  # Based on your example
            'Alpha1': game.alpha,
            'Alpha2': game.alpha,
            'Beta_1': game.beta,
            'Beta_2': game.beta,
            'Delta': game.delta,
            'Gamma': game.gamma,
            'Lossaversion_aversion': game.lossaversion,
            'Lambda' : game.lambda_,
            'Pnash': game.p_minmax[0],
            'Pcoop': game.p_minmax[1],
            'Pricenash': game.compute_p_competitive_monopoly(),
            'Pricecoop': game.compute_p_competitive_monopoly(),
            'a0': game.a0,
            'a1': game.a,
            'a2': game.a,
            'c1': game.c,
            'c2': game.c,
            'mu': game.mu,
            'extend1': game.extend,
            'extend2': game.extend,
            'NashP1': game.p_nash[0],
            'CoopP1': game.p_coop[0],
            'NashProfit1': game.NashProfits[0],
            'NashProfit2': game.NashProfits[1],
            'CoopProfit1': game.CoopProfits[0],
            'CoopProfit2': game.CoopProfits[1],
            'typeQ1': 'O',  # Based on your example
            'par1Q1': 0,
            'par2Q1': 0,
            'typeQ2': 'O',
            'par1Q2': 0,
            'par2Q2': 0
        }
        
        df = pd.DataFrame([config])
        config_path = os.path.join(run_dir, "config.csv")
        df.to_csv(config_path, index=False)
        
    def save_session_results(self, game, run_dir):
        """Save aggregated session results"""
        os.makedirs(run_dir, exist_ok=True)
        
        # Save session summaries
        session_summaries = {
            'session_id': range(game.num_sessions),
            'converged': game.converged,
            'time_to_convergence': game.time_to_convergence,
            'cycle_length': np.where(game.converged, game.cycle_length, np.nan),
            'Nash profit': game.NashProfits[0],
            'Coop profit': game.CoopProfits[0]
        }

        # Add cycle prices and profits for each player
        for i_player in range(game.n):
            player_num = i_player + 1
            prices_list = []
            prices_value_list = []
            profits_list = []
            reference_prices_list = []
            actual_reference_prices_list = []
            consumer_surplus_list = []
            mean_profit_list = []
            mean_price_list = []
            profit_gains_list = []
            price_gains_list = []

            # Extract prices and profits only up to cycle length for each session
            for i_session in range(game.num_sessions):
                cycle_len = game.cycle_length[i_session]
                prices = game.cycle_prices[i_player, :cycle_len, i_session]
                profits = game.cycle_profits[i_player, :cycle_len, i_session]
                consumer_surplus = game.cycle_consumer_surplus[:cycle_len, i_session]  # Extract CS

                mean_profits = np.mean(game.cycle_profits[i_player, :cycle_len, i_session])

                # Convert price indexes to actual price values
                actual_prices = np.asarray(game.A[np.asarray(game.cycle_prices[i_player, :cycle_len, i_session], dtype=int)])
                mean_prices = np.mean(actual_prices)

                # Guard: NaN gains when the market has collapsed (p_coop ~= p_nash)
                price_denom = game.p_coop[i_player] - game.p_nash[i_player]
                profit_denom = game.CoopProfits[i_player] - game.NashProfits[i_player]
                if abs(price_denom) < _COLLAPSE_EPS:
                    price_gains = np.nan
                    profit_gains = np.nan
                else:
                    profit_gains = (mean_profits - game.NashProfits[i_player]) / profit_denom
                    price_gains = (mean_prices - game.p_nash[i_player]) / price_denom


                if game.demand_type in ["reference", "misspecification"]:
                    # ↳ change this block
                    if game.common_reference:
                        ref_slice = game.cycle_reference_prices[0, :cycle_len, i_session]
                        reference_prices_list.append(','.join(f"{r:.5g}" for r in ref_slice))
                        actual_reference_prices_list.append(','.join(f"{game.A[int(r)]:.5g}" for r in ref_slice))
                    else:
                        # one string per firm, separated by ‘;’
                        firm_strings = []
                        for f in range(game.n):
                            ref_slice = game.cycle_reference_prices[f, :cycle_len, i_session]
                            firm_strings.append(','.join(f"{r:.5g}" for r in ref_slice))
                        reference_prices_list.append(';'.join(firm_strings))
                        actual_reference_prices_list = reference_prices_list
                
                # Convert arrays to strings with comma separation, formatting to 5 digits
                prices_str = ','.join([f"{p:.5g}" for p in prices])
                actual_prices_value_str = ','.join([f"{game.A[int(p)]:.5g}" for p in prices])
                profits_str = ','.join([f"{p:.5g}" for p in profits])
                consumer_surplus_str = ','.join([f"{cs:.5g}" for cs in consumer_surplus])

                
                prices_list.append(prices_str)
                prices_value_list.append(actual_prices_value_str)
                profits_list.append(profits_str)
                mean_profit_list.append(mean_profits)
                mean_price_list.append(mean_prices)
                profit_gains_list.append(profit_gains)
                price_gains_list.append(price_gains)
                consumer_surplus_list.append(consumer_surplus_str)
            
            session_summaries[f'cycle_prices_p{player_num}'] = prices_list
            session_summaries[f'cycle_prices_value_p{player_num}'] = prices_value_list
            session_summaries[f'cycle_profits_p{player_num}'] = profits_list
            session_summaries[f'cycle_mean_price_p{player_num}'] = mean_price_list
            session_summaries[f'cycle_mean_profit_p{player_num}'] = mean_profit_list
            session_summaries[f'cycle_price_gain_p{player_num}'] = price_gains_list
            session_summaries[f'cycle_profit_gain_p{player_num}'] = profit_gains_list
        session_summaries[f'cycle_consumer_surplus'] = consumer_surplus_list

        # Add reference prices if reference demand is used
        if game.demand_type in ["reference", "misspecification"]:
            session_summaries[f'cycle_reference_prices'] = reference_prices_list
            session_summaries[f'cycle_reference_prices_value'] = actual_reference_prices_list
        
        df_summaries = pd.DataFrame(session_summaries)
        df_summaries.to_csv(os.path.join(run_dir, "session_summaries.csv"), index=False)
        
        # Save compressed arrays for detailed data
        np.savez_compressed(
            os.path.join(run_dir, "session_details.npz"),
            cycle_states=game.cycle_states,
            cycle_prices=game.cycle_prices,
            cycle_profits=game.cycle_profits,
            cycle_consumer_surplus=game.cycle_consumer_surplus,
            index_strategies=game.index_strategies,
            cycle_reference_prices=game.cycle_reference_prices  # Include reference prices in saved file
        )

    def save_cycle_statistics(self, game, run_dir):
        """Save cycle statistics across all sessions"""
        os.makedirs(run_dir, exist_ok=True)

        # Calculate mean profits only up to cycle length for each session
        mean_profits = np.zeros((game.n, game.num_sessions))
        profit_gains = np.zeros((game.n, game.num_sessions))
        price_gains = np.zeros((game.n, game.num_sessions))
        mean_prices = np.zeros((game.n, game.num_sessions))
        mean_consumer_surplus = np.zeros(game.num_sessions)

        # If using reference pricing, store reference price statistics
        if game.demand_type in ["reference", "misspecification"]:
            if game.common_reference:                        # single reference
                mean_reference_prices = np.zeros(game.num_sessions)
                std_reference_prices  = np.zeros(game.num_sessions)
            else:                                            # one per firm
                mean_reference_prices = np.zeros((game.n, game.num_sessions))
                std_reference_prices  = np.zeros((game.n, game.num_sessions))

               
        for i_session in range(game.num_sessions):
            cycle_len = game.cycle_length[i_session]
            # Compute mean consumer surplus
            mean_consumer_surplus[i_session] = np.mean(game.cycle_consumer_surplus[:cycle_len, i_session])
             
            for i_player in range(game.n):
                mean_profits[i_player, i_session] = np.mean(game.cycle_profits[i_player, :cycle_len, i_session])

                # Convert price indexes to actual price values
                actual_prices = np.asarray(game.A[np.asarray(game.cycle_prices[i_player, :cycle_len, i_session], dtype=int)])
                mean_prices[i_player, i_session] = np.mean(actual_prices)

                # Guard: if the market has collapsed (p_coop ~= p_nash) the gain
                # denominators are ~0 -> report NaN instead of exploding values.
                price_denom = game.p_coop[i_player] - game.p_nash[i_player]
                profit_denom = game.CoopProfits[i_player] - game.NashProfits[i_player]
                if abs(price_denom) < _COLLAPSE_EPS:
                    price_gains[i_player, i_session] = np.nan
                    profit_gains[i_player, i_session] = np.nan
                else:
                    price_gains[i_player, i_session] = (mean_prices[i_player, i_session] - game.p_nash[i_player]) / price_denom
                    profit_gains[i_player, i_session] = (mean_profits[i_player, i_session] - game.NashProfits[i_player]) / profit_denom

                if game.demand_type in ["reference", "misspecification"]:
                    if game.common_reference:
                        ref = game.cycle_reference_prices[0, :cycle_len, i_session]
                        ref = game.A[ref.astype(int)]
                        mean_reference_prices[i_session] = ref.mean()
                        std_reference_prices[i_session]  = ref.std()
                    else:
                        for f in range(game.n):
                            ref = game.cycle_reference_prices[f, :cycle_len, i_session]
                            ref = game.A[ref.astype(int)]
                            mean_reference_prices[f, i_session] = ref.mean()
                            std_reference_prices[f,  i_session] = ref.std()

        

        print("loss_save",game.NashProfits[0],game.CoopProfits[0], game.p_nash[0],game.p_coop[0]) 
        print()
        # Calculate statistics
        cycle_stats = {
            'mean_cycle_length': f"{np.nanmean(game.cycle_length):.5g}",
            'std_cycle_length': f"{np.nanstd(game.cycle_length):.5g}",
            'convergence_rate': f"{np.nanmean(game.converged):.5g}",
            'mean_convergence_time': f"{np.nanmean(game.time_to_convergence):.5g}",
            'convergence_rate': f"{np.nanmean(game.converged):.5g}",
            'mean_convergence_time': f"{np.nanmean(game.time_to_convergence):.5g}"
        }
        
        # Add statistics for each player
        for i_player in range(game.n):
            player_num = i_player + 1
            cycle_stats.update({
                f'mean_profit_p{player_num}': f"{np.nanmean(mean_profits[i_player]):.5g}",
                f'std_profit_p{player_num}': f"{np.nanstd(mean_profits[i_player]):.5g}",
                f'mean_profit_gain_p{player_num}': f"{np.nanmean(profit_gains[i_player]):.5g}",
                f'std_profit_gain_p{player_num}': f"{np.nanstd(profit_gains[i_player]):.5g}",
                f'mean_price_gain_p{player_num}': f"{np.nanmean(price_gains[i_player]):.5g}",
                f'std_price_gain_p{player_num}': f"{np.nanstd(price_gains[i_player]):.5g}",
                f'mean_price_p{player_num}': f"{np.nanmean(mean_prices[i_player]):.5g}",
                f'std_price_p{player_num}': f"{np.nanstd(mean_prices[i_player]):.5g}",
            })

        cycle_stats.update({
                'mean_consumer_surplus': f"{np.nanmean(mean_consumer_surplus):.5g}",
            })

        
        # Add reference price statistics if applicable
        if game.demand_type in ["reference", "misspecification"]:
            cycle_stats.update({
                'mean_reference_price': f"{np.nanmean(mean_reference_prices):.5g}",
                'std_reference_price': f"{np.nanstd(mean_reference_prices):.5g}"
            })

        df_stats = pd.DataFrame([cycle_stats])
        df_stats.to_csv(os.path.join(run_dir, "cycle_statistics.csv"), index=False)

def save_experiment(game, experiment_name, gamma, lambda_):
    """Main function to save all experiment data"""
    saver = ExperimentSaver(experiment_name)
    run_dir = saver.get_run_dir(gamma, lambda_)  # Use the get_run_dir method
    
    # Save all components
    saver.save_experiment_config(game, run_dir)
    saver.save_session_results(game, run_dir)
    saver.save_cycle_statistics(game, run_dir)
    
    return run_dir

def save_experiment_gd(game, experiment_name, gamma, delta):
    """Main function to save all experiment data for a gamma-delta run"""
    saver = ExperimentSaver(experiment_name)
    run_dir = saver.get_run_dir_gd(gamma, delta)  # Use the get_run_dir_gd method

    # Save all components
    saver.save_experiment_config(game, run_dir)
    saver.save_session_results(game, run_dir)
    saver.save_cycle_statistics(game, run_dir)

    return run_dir

def save_experiment_lossaversion(game, experiment_name, lossaversion):
    """Main function to save all experiment data"""
    saver = ExperimentSaver(experiment_name)
    run_dir = saver.get_run_dir_lossaversion(lossaversion)  # Use the get_run_dir method
    
    # Save all components
    saver.save_experiment_config(game, run_dir)
    saver.save_session_results(game, run_dir)
    saver.save_cycle_statistics(game, run_dir)
    
    return run_dir

def save_experiment_gamma_only(game, experiment_name, gamma):
    """Main function to save all experiment data"""
    saver = ExperimentSaver(experiment_name)
    run_dir = saver.get_run_dir_gamma_only(gamma)  # Use the get_run_dir method
    
    # Save all components
    saver.save_experiment_config(game, run_dir)
    saver.save_session_results(game, run_dir)
    saver.save_cycle_statistics(game, run_dir)
    
    return run_dir


###############################################
######## Run Experiment 


###############################

## Single Computing Session 

def run_experiment_gl(game, gamma_values, lambda_values, num_sessions=1000, demand_type = 'noreference', experiment_name = 'test'):
    """
    Run experiments with different gamma and lambda values
    
    Parameters:
    -----------
    game : object
        Game instance
    gamma_values : array-like
        Array of gamma values to test
    lambda_values : array-like
        Array of lambda values to test
    num_sessions : int
        Number of sessions per experiment
    num_processes : int, optional
        Number of processes to use. If None, uses CPU count - 1
    """

    # Fixed values
    alpha_fixed = 0.15
    beta_fixed = 0.1 / 2500
    
    for i, gamma in enumerate(gamma_values):
        for j, lambda_ in enumerate(lambda_values):

            gamma = round(gamma, 3)
            lambda_= round(lambda_, 3)
        
            # Configure experiment
            experiment_id = f"gamma_{gamma}_lambda_{lambda_}"
            
            # Update game parameters for this experiment
            game.alpha = alpha_fixed
            game.beta = beta_fixed
            game.gamma = gamma  # Varying gamma
            game.lambda_ = lambda_  # Varying lambda
            game.p_minmax = game.compute_p_competitive_monopoly()
            game.NashProfits,  game.CoopProfits = game.compute_profits_nash_coop()
            game.p_nash, game.p_coop = game.p_minmax[0], game.p_minmax[1]
            game.PI = game.init_PI()
            game.Q = game.init_Q()
            game.num_sessions = num_sessions
            game.demand_type = demand_type
            
            # Game logs 
            if game.common_reference:
                ref_shape = (1,)  # single common reference price
            else:
                ref_shape = (game.n,)  # each firm has its own reference price
            # Reset and initialize game arrays for the new experiment
            game.converged = np.zeros(game.num_sessions, dtype=bool)
            game.time_to_convergence = np.zeros(game.num_sessions, dtype=float)
            game.index_last_state = np.zeros((game.n, game.memory, game.num_sessions), dtype=int)
            game.index_last_reference = np.zeros(ref_shape + (game.num_sessions,), dtype=int)
            game.cycle_length = np.zeros(game.num_sessions, dtype=int)
            game.cycle_states = np.zeros((game.num_periods, game.num_sessions), dtype=int)
            game.cycle_prices = np.zeros((game.n, game.num_periods, game.num_sessions), dtype=float)
            game.cycle_profits = np.zeros((game.n, game.num_periods, game.num_sessions), dtype=float)
            game.cycle_reference_prices = np.zeros(ref_shape + (game.num_periods, game.num_sessions), dtype=float)
            game.cycle_consumer_surplus = np.zeros((game.num_periods, game.num_sessions), dtype=float) 
            game.index_strategies = np.zeros((game.n,) + game.sdim + (game.num_sessions,), dtype=int)
            game.last_observed_prices = np.zeros((game.n, game.memory), dtype=int)  # last prices
            game.last_observed_reference = np.zeros(ref_shape, dtype=int)
            game.last_reference_observed_prices = np.zeros((game.n, game.reference_memory), dtype=int)  # last prices
            game.last_observed_demand = np.zeros((game.n, game.reference_memory), dtype=float)  # last shares for each firm

            # Run all sessions for this gamma_lambda combination
            for iSession in range(game.num_sessions):
                if game.aprint:
                    print(f"\nStarting Session {iSession + 1}/{game.num_sessions}")
                    print(f"Current gamma: {gamma}, lambda: {lambda_}")

                game.Q = game.init_Q()  # Reset Q-values
                game.last_observed_prices = np.zeros((game.n, game.memory), dtype=int)  # Reset prices


                # Run Q-learning for this session
                game, converged, t_convergence, consumer_reference_agent  = simulate_game(game)

                # Store convergence results
                game.converged[iSession] = converged
                game.time_to_convergence[iSession] = t_convergence

                # Store last observed prices
                game.index_last_state[:, :, iSession] = game.last_observed_prices

                # Store the learned strategies (optimal strategies at convergence)
                game.index_strategies[..., iSession] = game.Q.argmax(axis=-1)

                if game.demand_type in ["reference", "misspecification"]:
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
                        cycle_length, visited_states, visited_profits, price_history, reference_price_history, consumer_surplus_history = detect_cycle(game, iSession, consumer_reference_agent)  # Now passing iSession
                        cycle_data = {
                            'cycle_length': cycle_length,
                            'visited_states': visited_states,
                            'visited_profits': visited_profits,
                            'price_history': price_history,
                            'reference_price_history': reference_price_history,
                            'consumer_surplus_history': consumer_surplus_history
                        }
            # Save results for this gamma-lambda combination
            run_dir = save_experiment(game, experiment_name, gamma, lambda_)

            if game.aprint:
                print(f"\nCompleted experiment for gamma={gamma}, lambda={lambda_}")
                print(f"Results saved under experiment ID: {experiment_id}")

    print("\nAll experiments completed.")
    return game




###############################

## Parallel Computing Section

def run_single_session(game, gamma, lambda_, lossaversion, iSession,
                       use_reference_pretraining=False, T_ref=200000,
                       alpha=0.15, beta=0.1 / 2500):
    """
    Run a single session of the game

    Parameters:
    -----------
    game : object
        Game instance
    iSession : int
        Session number
    alpha, beta : float
        Q-learning rate and exploration-decay rate. Defaults preserve the
        historical hard-coded values (0.15, 4e-5); pass explicitly to vary.
        NOTE: these OVERRIDE whatever game.alpha/game.beta were set to.

    Returns:
    --------
    dict : Session results
    """

    # # Update game parameters
    game.alpha = alpha
    game.beta = beta
    game.gamma = gamma  # Varying gamma
    game.lambda_ = lambda_  # Varying lambda
    game.lossaversion = lossaversion
    game.p_minmax = game.compute_p_competitive_monopoly()
    game.NashProfits,  game.CoopProfits  = game.compute_profits_nash_coop()
    game.p_nash, game.p_coop = game.p_minmax[0], game.p_minmax[1]
    game.PI = game.init_PI()
    game.Q = game.init_Q()

        # Game logs 
    if game.common_reference:
        ref_shape = (1,)  # single common reference price
    else:
        ref_shape = (game.n,)  # each firm has its own reference price
    # Reset and initialize game arrays for the new experiment
    game.converged = np.zeros(game.num_sessions, dtype=bool)
    game.time_to_convergence = np.zeros(game.num_sessions, dtype=float)
    game.index_last_state = np.zeros((game.n, game.memory, game.num_sessions), dtype=int)
    game.index_last_reference = np.zeros(ref_shape + (game.num_sessions,), dtype=int)
    game.cycle_length = np.zeros(game.num_sessions, dtype=int)
    game.cycle_states = np.zeros((game.num_periods, game.num_sessions), dtype=int)
    game.cycle_prices = np.zeros((game.n, game.num_periods, game.num_sessions), dtype=float)
    game.cycle_profits = np.zeros((game.n, game.num_periods, game.num_sessions), dtype=float)
    game.cycle_reference_prices = np.zeros(ref_shape + (game.num_periods, game.num_sessions), dtype=float)
    game.cycle_consumer_surplus = np.zeros((game.num_periods, game.num_sessions), dtype=float) 
    game.index_strategies = np.zeros((game.n,) + game.sdim + (game.num_sessions,), dtype=int)
    game.last_observed_prices = np.zeros((game.n, game.memory), dtype=int)  # last prices
    game.last_observed_reference = np.zeros(ref_shape, dtype=int)
    game.last_reference_observed_prices = np.zeros((game.n, game.reference_memory), dtype=int)  # last prices
    game.last_observed_demand = np.zeros((game.n, game.reference_memory), dtype=float)  # last shares for each firm


    # Create a deep copy of game to avoid shared state issues
    game_copy = copy.deepcopy(game)
    
    # Initialize session-specific variables
    game_copy.Q = game_copy.init_Q()
    game_copy.last_observed_prices = np.zeros((game_copy.n, game_copy.memory), dtype=int)

    if game.demand_type in ["reference", "misspecification"]:
        # Initialize reference-related variables
        game_copy.last_observed_reference = np.zeros(ref_shape, dtype=int)
        game_copy.last_reference_observed_prices = np.zeros((game_copy.n, game_copy.reference_memory), dtype=int)
        game_copy.last_observed_demand = np.zeros((game_copy.n, game_copy.reference_memory), dtype=float)


    # 🔐 Set unique random seed for each session
    seed = secrets.randbits(32)   # 32-bit integer from OS entropy
    np.random.seed(seed)
    random.seed(seed)

    # Optionally pretrain a consumer reference agent. Unlike a strict
    # two-stage protocol, we *do not* freeze learning afterwards; the
    # pretrained agent continues to explore and update during
    # firm-learning. Pretraining simply provides a better initialization
    # for the consumer-side Q-table.
    consumer_reference_agent = None

    if (use_reference_pretraining and
        game_copy.demand_type in ["reference", "misspecification"] and
        game_copy.ref_prediction == "qlearning"):

        consumer_reference_agent = pretrain_consumer_reference(game_copy, T_ref=T_ref)

    # Run simulation (consumer_reference_agent may be None or pretrained),
    # with reference learning active throughout.
    game_copy, converged, t_convergence, consumer_reference_agent = simulate_game(
        game_copy,
        consumer_reference_agent=consumer_reference_agent,
        train_reference=True,
    )

    # Store convergence results in game_copy
    game_copy.converged[iSession] = converged
    game_copy.time_to_convergence[iSession] = t_convergence
    
    # Store last observed prices in game_copy
    game_copy.index_last_state[:, :, iSession] = game_copy.last_observed_prices
    
    # Store the learned strategies in game_copy
    game_copy.index_strategies[..., iSession] = game_copy.Q.argmax(axis=-1)

    # Store reference pricing data if applicable
    last_reference_price = None
    last_reference_prices = None
    last_observed_demand = None

    if game.demand_type in ["reference", "misspecification"]:
        last_reference_price = game_copy.last_observed_reference
        last_reference_prices = game_copy.last_reference_observed_prices
        last_observed_demand = game_copy.last_observed_demand


    # If converged, get cycle data
    cycle_data = None
    if converged:
        if game_copy.demand_type in ('noreference', 'price_sensitivity'):
            # Pass iSession to detect_cycle function
            cycle_length, visited_states, visited_profits, price_history, _, consumer_surplus_history = detect_cycle(game_copy, iSession)  # Now passing iSession
            cycle_data = {
                'cycle_length': cycle_length,
                'visited_states': visited_states,
                'visited_profits': visited_profits,
                'price_history': price_history,
                'consumer_surplus_history': consumer_surplus_history
            }
        if game.demand_type in ["reference", "misspecification"]:
            # Pass iSession to detect_cycle function
            cycle_length, visited_states, visited_profits, price_history, reference_price_history, consumer_surplus_history = detect_cycle(game_copy, iSession, consumer_reference_agent)  # Now passing iSession
            cycle_data = {
                'cycle_length': cycle_length,
                'visited_states': visited_states,
                'visited_profits': visited_profits,
                'price_history': price_history,
                'reference_price_history': reference_price_history,
                'consumer_surplus_history': consumer_surplus_history
            }

    # Q-value stabilization trajectory (only populated when
    # game.track_q_stabilization is True; see qlearning.simulate_game).
    q_diag = getattr(game_copy, 'q_diag', None)

    if game.demand_type in ["reference", "misspecification"]:
        # Return results
        return {
            'session_id': iSession,
            'converged': converged,
            'time_to_convergence': t_convergence,
            'last_observed_prices': game_copy.last_observed_prices,
            'optimal_strategies': game_copy.Q.argmax(axis=-1),
            'cycle_data': cycle_data,
            'last_observed_reference': last_reference_price,
            'last_reference_prices': last_reference_prices,
            'last_observed_demand': last_observed_demand,
            'q_diag': q_diag
        }


    # Return results
    return {
        'session_id': iSession,
        'converged': converged,
        'time_to_convergence': t_convergence,
        'last_observed_prices': game_copy.last_observed_prices,
        'optimal_strategies': game_copy.Q.argmax(axis=-1),
        'cycle_data': cycle_data,
        'q_diag': q_diag
    }


def run_experiment_parallel_gl(game, gamma_values, lambda_values, num_sessions=1000, experiment_name='test',  demand_type = 'noreference', num_processes=None):
    """
    Run experiments with different gamma and lambda values using parallel processing
    """
    if num_processes is None:
        num_processes = max(1, mp.cpu_count() - 2)
        print('num_process', num_processes)

    # Run sessions in parallel with error handling
    print(f"Starting parallel processing with {num_processes} processes for {num_sessions} sessions")
    
    # Fixed values
    alpha_fixed = 0.15
    beta_fixed = 0.1 / 2500
    loss_aversion_fixed = 1.5

    for i, gamma in enumerate(gamma_values):
        for j, lambda_ in enumerate(lambda_values):

            gamma = round(gamma, 3)
            lambda_= round(lambda_, 3)

            # Check if this gamma-lambda combination has already been run
            run_dir = os.path.join("../Results/experiments", experiment_name, f"gamma_{gamma}_lambda_{lambda_}")
            stats_file = os.path.join(run_dir, "cycle_statistics.csv")

            if os.path.exists(stats_file):
                print(f"Skipping gamma_{gamma}_lambda_{lambda_} (already exists in {run_dir})")
                continue  # Skip running simulation again


            # Update game parameters
            game.alpha = alpha_fixed
            game.beta = beta_fixed
            game.gamma = gamma  # Varying gamma
            game.lambda_ = lambda_  # Varying lambda
            game.p_minmax = game.compute_p_competitive_monopoly()
            game.NashProfits,  game.CoopProfits = game.compute_profits_nash_coop()
            game.p_nash, game.p_coop = game.p_minmax[0], game.p_minmax[1]
            game.PI = game.init_PI()
            game.Q = game.init_Q()
            game.num_sessions = num_sessions
            game.demand_type = demand_type
            
            # Game logs 
            if game.common_reference:
                ref_shape = (1,)  # single common reference price
            else:
                ref_shape = (game.n,)  # each firm has its own reference price
            # Reset and initialize game arrays for the new experiment
            game.converged = np.zeros(game.num_sessions, dtype=bool)
            game.time_to_convergence = np.zeros(game.num_sessions, dtype=float)
            game.index_last_state = np.zeros((game.n, game.memory, game.num_sessions), dtype=int)
            game.index_last_reference = np.zeros(ref_shape + (game.num_sessions,), dtype=int)
            game.cycle_length = np.zeros(game.num_sessions, dtype=int)
            game.cycle_states = np.zeros((game.num_periods, game.num_sessions), dtype=int)
            game.cycle_prices = np.zeros((game.n, game.num_periods, game.num_sessions), dtype=float)
            game.cycle_profits = np.zeros((game.n, game.num_periods, game.num_sessions), dtype=float)
            game.cycle_reference_prices = np.zeros(ref_shape + (game.num_periods, game.num_sessions), dtype=float)
            game.cycle_consumer_surplus = np.zeros((game.num_periods, game.num_sessions), dtype=float) 
            game.index_strategies = np.zeros((game.n,) + game.sdim + (game.num_sessions,), dtype=int)
            game.last_observed_prices = np.zeros((game.n, game.memory), dtype=int)  # last prices
            game.last_observed_reference = np.zeros(ref_shape, dtype=int)
            game.last_reference_observed_prices = np.zeros((game.n, game.reference_memory), dtype=int)  # last prices
            game.last_observed_demand = np.zeros((game.n, game.reference_memory), dtype=float)  # last shares for each firm

            #if game.aprint:
            print(f"\nStarting gamma={gamma}, lambda={lambda_} with {num_processes} processes")
            
            try:
                # Run sessions in parallel with error handling
                with mp.Pool(processes=num_processes) as pool:
                    session_results = []
                    for iSession in range(num_sessions):
                        result = pool.apply_async(run_single_session, args=(game, gamma, lambda_, loss_aversion_fixed, iSession))
                        session_results.append(result)
                    
                    # ✅ Use improved result collection here
                    results = []
                    for i, res in enumerate(session_results):
                        try:
                            result = res.get(timeout=600)
                            results.append(result)
                        except Exception as e:
                            print(f"Session {i} failed or timed out: {e}")
                            continue
                

                # Process results
                for result in results:
                    iSession = result['session_id']
                    game.converged[iSession] = result['converged']
                    game.time_to_convergence[iSession] = result['time_to_convergence']
                    game.index_last_state[:, :, iSession] = result['last_observed_prices']
                    game.index_strategies[..., iSession] = result['optimal_strategies']

                    # If using reference pricing, store reference-related results
                    if game.demand_type in ["reference", "misspecification"]:
                        game.index_last_reference[:, iSession] = result['last_observed_reference']

                    
                    if result['cycle_data'] is not None:
                        cycle_data = result['cycle_data']
                        game.cycle_length[iSession] = cycle_data['cycle_length']
                        cycle_len = cycle_data['cycle_length']
                        game.cycle_states[:cycle_len, iSession] = cycle_data['visited_states']
                        game.cycle_prices[:, :cycle_len, iSession] = cycle_data['price_history']
                        game.cycle_profits[:, :cycle_len, iSession] = cycle_data['visited_profits']
                        game.cycle_consumer_surplus[:cycle_len, iSession] = cycle_data['consumer_surplus_history']
                        if game.demand_type in ["reference", "misspecification"]:
                            game.cycle_reference_prices[:, :cycle_len, iSession] = cycle_data['reference_price_history']

                # Save results for this gamma-lambda combination
                run_dir = save_experiment(game, experiment_name, gamma, lambda_)
                
                if game.aprint:
                    print(f"Completed gamma={gamma}, lambda={lambda_}")
                    print(f"Results saved in {run_dir}")
                    
            except Exception as e:
                print(f"Error processing gamma={gamma}, lambda={lambda_}: {str(e)}")
                import traceback
                traceback.print_exc()  # Print full error details
                continue

    print("\nAll experiments completed.")
    return game


def run_experiment_parallel_gd(game, gamma_values, delta_values, lambda_fixed=0.5,
                               num_sessions=1000, experiment_name='test',
                               demand_type='noreference', num_processes=None):
    """
    Run experiments over a gamma x delta grid using parallel processing.

    delta (the discount factor) is the only new sweep dimension compared to the
    gamma-lambda machinery: it is set on `game` before dispatch and flows into
    each worker through the deep-copy inside run_single_session (which never
    overrides game.delta). lambda_ is held fixed at lambda_fixed.
    """
    if num_processes is None:
        num_processes = max(1, mp.cpu_count() - 2)
        print('num_process', num_processes)

    # Run sessions in parallel with error handling
    print(f"Starting parallel processing with {num_processes} processes for {num_sessions} sessions")

    # Fixed values
    alpha_fixed = 0.15
    beta_fixed = 4e-6   # paper default (corrected from 4e-5)
    loss_aversion_fixed = 1.5

    for i, gamma in enumerate(gamma_values):
        for j, delta in enumerate(delta_values):

            gamma = round(gamma, 3)
            delta = round(delta, 3)

            # Check if this gamma-delta combination has already been run
            run_dir = os.path.join("../Results/experiments", experiment_name, f"gamma_{gamma}_delta_{delta}")
            stats_file = os.path.join(run_dir, "cycle_statistics.csv")

            if os.path.exists(stats_file):
                print(f"Skipping gamma_{gamma}_delta_{delta} (already exists in {run_dir})")
                continue  # Skip running simulation again

            # Update game parameters
            game.alpha = alpha_fixed
            game.beta = beta_fixed
            game.gamma = gamma            # Varying gamma
            game.delta = delta            # Varying delta (discount factor)
            game.lambda_ = lambda_fixed   # Fixed reference update rate
            game.p_minmax = game.compute_p_competitive_monopoly()
            game.NashProfits,  game.CoopProfits = game.compute_profits_nash_coop()
            game.p_nash, game.p_coop = game.p_minmax[0], game.p_minmax[1]
            game.PI = game.init_PI()
            game.Q = game.init_Q()
            game.num_sessions = num_sessions
            game.demand_type = demand_type

            # Game logs
            if game.common_reference:
                ref_shape = (1,)  # single common reference price
            else:
                ref_shape = (game.n,)  # each firm has its own reference price
            # Reset and initialize game arrays for the new experiment
            game.converged = np.zeros(game.num_sessions, dtype=bool)
            game.time_to_convergence = np.zeros(game.num_sessions, dtype=float)
            game.index_last_state = np.zeros((game.n, game.memory, game.num_sessions), dtype=int)
            game.index_last_reference = np.zeros(ref_shape + (game.num_sessions,), dtype=int)
            game.cycle_length = np.zeros(game.num_sessions, dtype=int)
            game.cycle_states = np.zeros((game.num_periods, game.num_sessions), dtype=int)
            game.cycle_prices = np.zeros((game.n, game.num_periods, game.num_sessions), dtype=float)
            game.cycle_profits = np.zeros((game.n, game.num_periods, game.num_sessions), dtype=float)
            game.cycle_reference_prices = np.zeros(ref_shape + (game.num_periods, game.num_sessions), dtype=float)
            game.cycle_consumer_surplus = np.zeros((game.num_periods, game.num_sessions), dtype=float)
            game.index_strategies = np.zeros((game.n,) + game.sdim + (game.num_sessions,), dtype=int)
            game.last_observed_prices = np.zeros((game.n, game.memory), dtype=int)  # last prices
            game.last_observed_reference = np.zeros(ref_shape, dtype=int)
            game.last_reference_observed_prices = np.zeros((game.n, game.reference_memory), dtype=int)  # last prices
            game.last_observed_demand = np.zeros((game.n, game.reference_memory), dtype=float)  # last shares for each firm

            #if game.aprint:
            print(f"\nStarting gamma={gamma}, delta={delta} with {num_processes} processes")

            try:
                # Run sessions in parallel with error handling
                with mp.Pool(processes=num_processes) as pool:
                    session_results = []
                    for iSession in range(num_sessions):
                        result = pool.apply_async(run_single_session, args=(game, gamma, lambda_fixed, loss_aversion_fixed, iSession))
                        session_results.append(result)

                    # ✅ Use improved result collection here
                    results = []
                    for k, res in enumerate(session_results):
                        try:
                            result = res.get(timeout=600)
                            results.append(result)
                        except Exception as e:
                            print(f"Session {k} failed or timed out: {e}")
                            continue

                # Process results
                for result in results:
                    iSession = result['session_id']
                    game.converged[iSession] = result['converged']
                    game.time_to_convergence[iSession] = result['time_to_convergence']
                    game.index_last_state[:, :, iSession] = result['last_observed_prices']
                    game.index_strategies[..., iSession] = result['optimal_strategies']

                    # If using reference pricing, store reference-related results
                    if game.demand_type in ["reference", "misspecification"]:
                        game.index_last_reference[:, iSession] = result['last_observed_reference']

                    if result['cycle_data'] is not None:
                        cycle_data = result['cycle_data']
                        game.cycle_length[iSession] = cycle_data['cycle_length']
                        cycle_len = cycle_data['cycle_length']
                        game.cycle_states[:cycle_len, iSession] = cycle_data['visited_states']
                        game.cycle_prices[:, :cycle_len, iSession] = cycle_data['price_history']
                        game.cycle_profits[:, :cycle_len, iSession] = cycle_data['visited_profits']
                        game.cycle_consumer_surplus[:cycle_len, iSession] = cycle_data['consumer_surplus_history']
                        if game.demand_type in ["reference", "misspecification"]:
                            game.cycle_reference_prices[:, :cycle_len, iSession] = cycle_data['reference_price_history']

                # Save results for this gamma-delta combination
                run_dir = save_experiment_gd(game, experiment_name, gamma, delta)

                if game.aprint:
                    print(f"Completed gamma={gamma}, delta={delta}")
                    print(f"Results saved in {run_dir}")

            except Exception as e:
                print(f"Error processing gamma={gamma}, delta={delta}: {str(e)}")
                import traceback
                traceback.print_exc()  # Print full error details
                continue

    print("\nAll experiments completed.")
    return game



###############################################
######## Run Experiment loss aversion


###############################

## Single Computing Session 

def run_experiment_lossaversion(game, lossaversion_values, num_sessions=1000, demand_type = 'noreference', experiment_name = 'test'):
    """
    Run experiments with different gamma and lambda values
    
    Parameters:
    -----------
    game : object
        Game instance
    gamma_values : array-like
        Array of gamma values to test
    lambda_values : array-like
        Array of lambda values to test
    num_sessions : int
        Number of sessions per experiment
    num_processes : int, optional
        Number of processes to use. If None, uses CPU count - 1
    """

    # Fixed values
    alpha_fixed = 0.15
    beta_fixed = 0.1 / 2500
    gamma_fixed = 1
    lambda_fixed = 0.5

    for i, lossaversion in enumerate(lossaversion_values):
        # Configure experiment
        experiment_id = f"lossaversion_{lossaversion}"
            
        # Update game parameters
        game.alpha = alpha_fixed
        game.beta = beta_fixed
        game.gamma = gamma_fixed  
        game.lambda_ = lambda_fixed  
        game.lossaversion = lossaversion # Varying lossaversion
        game.p_minmax = game.compute_p_competitive_monopoly()
        game.p_nash, game.p_coop = game.p_minmax[0], game.p_minmax[1]
        game.PI = game.init_PI()
        game.Q = game.init_Q()
        game.num_sessions = num_sessions
        game.demand_type = demand_type
            
        # Game logs 
        if game.common_reference:
            ref_shape = (1,)  # single common reference price
        else:
            ref_shape = (game.n,)  # each firm has its own reference price
        # Reset and initialize game arrays for the new experiment
        game.converged = np.zeros(game.num_sessions, dtype=bool)
        game.time_to_convergence = np.zeros(game.num_sessions, dtype=float)
        game.index_last_state = np.zeros((game.n, game.memory, game.num_sessions), dtype=int)
        game.index_last_reference = np.zeros(ref_shape + (game.num_sessions,), dtype=int)
        game.cycle_length = np.zeros(game.num_sessions, dtype=int)
        game.cycle_states = np.zeros((game.num_periods, game.num_sessions), dtype=int)
        game.cycle_prices = np.zeros((game.n, game.num_periods, game.num_sessions), dtype=float)
        game.cycle_profits = np.zeros((game.n, game.num_periods, game.num_sessions), dtype=float)
        game.cycle_reference_prices = np.zeros(ref_shape + (game.num_periods, game.num_sessions), dtype=float)
        game.cycle_consumer_surplus = np.zeros((game.num_periods, game.num_sessions), dtype=float) 
        game.index_strategies = np.zeros((game.n,) + game.sdim + (game.num_sessions,), dtype=int)
        game.last_observed_prices = np.zeros((game.n, game.memory), dtype=int)  # last prices
        game.last_observed_reference = np.zeros(ref_shape, dtype=int)
        game.last_reference_observed_prices = np.zeros((game.n, game.reference_memory), dtype=int)  # last prices
        game.last_observed_demand = np.zeros((game.n, game.reference_memory), dtype=float)  # last shares for each firm


        # Run all sessions for this gamma_lambda combination
        for iSession in range(game.num_sessions):
            if game.aprint:
                print(f"\nStarting Session {iSession + 1}/{game.num_sessions}")
                print(f"Current lossaversion: {lossaversion}")

            game.Q = game.init_Q()  # Reset Q-values
            game.last_observed_prices = np.zeros((game.n, game.memory), dtype=int)  # Reset prices


            # Run Q-learning for this session
            game, converged, t_convergence, consumer_reference_agent = simulate_game(game)

            # Store convergence results
            game.converged[iSession] = converged
            game.time_to_convergence[iSession] = t_convergence

            # Store last observed prices
            game.index_last_state[:, :, iSession] = game.last_observed_prices

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
                    cycle_length, visited_states, visited_profits, price_history, reference_price_history, consumer_surplus_history = detect_cycle(game, iSession, consumer_reference_agent)  # Now passing iSession
                    cycle_data = {
                        'cycle_length': cycle_length,
                        'visited_states': visited_states,
                        'visited_profits': visited_profits,
                        'price_history': price_history,
                        'reference_price_history': reference_price_history,
                        'consumer_surplus_history': consumer_surplus_history
                    }
        # Save results for this gamma-lambda combination
        run_dir = save_experiment_lossaversion(game, experiment_name, lossaversion)

        if game.aprint:
            print(f"\nCompleted experiment for lossaversion = {lossaversion}")
            print(f"Results saved under experiment ID: {experiment_id}")

    print("\nAll experiments completed.")
    return game






def run_experiment_parallel_lossaversion(game, lossaversion_values, num_sessions=1000, experiment_name='test',  demand_type = 'noreference', num_processes=None,
                                         alpha=0.15, beta=0.1 / 2500,
                                         gamma_fixed=1, lambda_fixed=0.5,
                                         session_timeout=1800):
    """
    Run experiments with different lossaversion values using parallel processing
    """
    if num_processes is None:
        num_processes = max(1, mp.cpu_count() - 2)
        print('num_process', num_processes)

    # Run sessions in parallel with error handling
    print(f"Starting parallel processing with {num_processes} processes for {num_sessions} sessions")
    
    # Fixed values
    alpha_fixed = alpha
    beta_fixed = beta
    # gamma_fixed / lambda_fixed now function parameters


    for i, lossaversion in enumerate(lossaversion_values):
        
        # Check if this gamma-lambda combination has already been run
        run_dir = os.path.join("../Results/experiments", experiment_name, f"lossaversion_{lossaversion}")
        stats_file = os.path.join(run_dir, "cycle_statistics.csv")

        # Update game parameters
        game.alpha = alpha_fixed
        game.beta = beta_fixed
        game.gamma = gamma_fixed  
        game.lambda_ = lambda_fixed  
        game.lossaversion = lossaversion # Varying lossaversion
        game.p_minmax = game.compute_p_competitive_monopoly()
        game.NashProfits,  game.CoopProfits  = game.compute_profits_nash_coop()
        game.p_nash, game.p_coop = game.p_minmax[0], game.p_minmax[1]
        #print('f',game.NashProfits[0],  game.CoopProfits[0])
        game.PI = game.init_PI()
        game.Q = game.init_Q()
        game.num_sessions = num_sessions
        game.demand_type = demand_type
        
        if os.path.exists(stats_file):
            print(f"lossaversion_{lossaversion} (already exists in {run_dir})")
            # Load existing stats
            df = pd.read_csv(stats_file)

            # Only add p_nash and p_coop if not already present
            if 'p_nash_p1' not in df.columns:
                for i, val in enumerate(game.p_nash):
                    df[f'p_nash_p{i+1}'] = val
                for i, val in enumerate(game.p_coop):
                    df[f'p_coop_p{i+1}'] = val

                df.to_csv(stats_file, index=False)
                print(f"Added p_nash and p_coop per player to {stats_file}")

            else:
                print("p_nash and p_coop already present.")
            continue  # Skip running simulation again


        # Game logs 
        if game.common_reference:
            ref_shape = (1,)  # single common reference price
        else:
            ref_shape = (game.n,)  # each firm has its own reference price
        # Reset and initialize game arrays for the new experiment
        game.converged = np.zeros(game.num_sessions, dtype=bool)
        game.time_to_convergence = np.zeros(game.num_sessions, dtype=float)
        game.index_last_state = np.zeros((game.n, game.memory, game.num_sessions), dtype=int)
        game.index_last_reference = np.zeros(ref_shape + (game.num_sessions,), dtype=int)
        game.cycle_length = np.zeros(game.num_sessions, dtype=int)
        game.cycle_states = np.zeros((game.num_periods, game.num_sessions), dtype=int)
        game.cycle_prices = np.zeros((game.n, game.num_periods, game.num_sessions), dtype=float)
        game.cycle_profits = np.zeros((game.n, game.num_periods, game.num_sessions), dtype=float)
        game.cycle_reference_prices = np.zeros(ref_shape + (game.num_periods, game.num_sessions), dtype=float)
        game.cycle_consumer_surplus = np.zeros((game.num_periods, game.num_sessions), dtype=float) 
        game.index_strategies = np.zeros((game.n,) + game.sdim + (game.num_sessions,), dtype=int)
        game.last_observed_prices = np.zeros((game.n, game.memory), dtype=int)  # last prices
        game.last_observed_reference = np.zeros(ref_shape, dtype=int)
        game.last_reference_observed_prices = np.zeros((game.n, game.reference_memory), dtype=int)  # last prices
        game.last_observed_demand = np.zeros((game.n, game.reference_memory), dtype=float)  # last shares for each firm


        #if game.aprint:
        print(f"\n Lossaversion_{lossaversion} with {num_processes} processes")
        
        try:
            # Run sessions in parallel with error handling
            with mp.Pool(processes=num_processes) as pool:
                session_results = []
                for iSession in range(num_sessions):
                    result = pool.apply_async(
                        run_single_session,
                        args=(game, gamma_fixed, lambda_fixed, lossaversion, iSession,
                              False, 200000, alpha, beta))
                    session_results.append(result)
                
                # ✅ Use improved result collection here
                results = []
                for i, res in enumerate(session_results):
                    try:
                        result = res.get(timeout=session_timeout)
                        results.append(result)
                    except Exception as e:
                        print(f"Session {i} failed or timed out: {e}")
                        continue
    
            # Process results
            for result in results:
                old_gamma = game.gamma
                old_lossaversion = game.lossaversion

                # game.lossaversion = 1
                # game.gamma = gamma_fixed * lossaversion
                # game.p_minmax = game.compute_p_competitive_monopoly()
                # game.p_nash = game.p_minmax[0]
                # NashProfits, _ = game.compute_profits_nash_coop()
                # game.NashProfits = NashProfits  # CoopProfits stays as baseline


                # game.gamma = gamma_fixed 
                # game.p_minmax = game.compute_p_competitive_monopoly()
                # game.p_coop = game.p_minmax[1]
                # _, CoopProfits = game.compute_profits_nash_coop()
                # game.CoopProfits = CoopProfits  # CoopProfits stays as baseline


                game.lossaversion = 1
                game.gamma = gamma_fixed * lossaversion
                game.p_minmax = game.compute_p_competitive_monopoly()
                game.p_nash, game.p_coop = game.p_minmax[0], game.p_minmax[1]
                NashProfits, CoopProfits = game.compute_profits_nash_coop()
                game.CoopProfits = CoopProfits  # CoopProfits stays as baseline
                game.NashProfits = NashProfits

                # Restore state for the rest of the loop (learning part)
                game.gamma = old_gamma
                game.lossaversion = old_lossaversion
                print('lossaversion = ',game.lossaversion, 'Nash Coop Profits = ',game.NashProfits[0],  game.CoopProfits[0])

                # game.lossaversion = lossaversion
                iSession = result['session_id']
                game.converged[iSession] = result['converged']
                game.time_to_convergence[iSession] = result['time_to_convergence']
                game.index_last_state[:, :, iSession] = result['last_observed_prices']
                game.index_strategies[..., iSession] = result['optimal_strategies']

                # If using reference pricing, store reference-related results
                if game.demand_type in ["reference", "misspecification"]:
                    game.index_last_reference[:, iSession] = result['last_observed_reference']

                
                if result['cycle_data'] is not None:
                    cycle_data = result['cycle_data']
                    game.cycle_length[iSession] = cycle_data['cycle_length']
                    cycle_len = cycle_data['cycle_length']
                    game.cycle_states[:cycle_len, iSession] = cycle_data['visited_states']
                    game.cycle_prices[:, :cycle_len, iSession] = cycle_data['price_history']
                    game.cycle_profits[:, :cycle_len, iSession] = cycle_data['visited_profits']
                    game.cycle_consumer_surplus[:cycle_len, iSession] = cycle_data['consumer_surplus_history']
                    if game.demand_type in ["reference", "misspecification"]:
                        game.cycle_reference_prices[:, :cycle_len, iSession] = cycle_data['reference_price_history']


            # Save results for this gamma-lambda combination
            run_dir = save_experiment_lossaversion(game, experiment_name, lossaversion)
            
            if game.aprint:
                print(f"Completed lossaversion = {lossaversion}")
                print(f"Results saved in {run_dir}")
                
        except Exception as e:
            print(f"Error processing lossaversion = {lossaversion}: {str(e)}")
            import traceback
            traceback.print_exc()  # Print full error details
            continue

    print("\nAll experiments completed.")
    return game



def run_experiment_parallel_gamma_only(game, gamma_values, num_sessions=1000,
                                       experiment_name='test',  demand_type='noreference',
                                       num_processes=None, use_reference_pretraining=False,
                                       T_ref=200000, lambda_fixed=0.6,
                                       alpha=0.15, beta=4e-6,
                                       lossaversion_fixed=1.5,
                                       session_timeout=1800):
    """
    Run experiments with different lossaversion values using parallel processing

    lambda_fixed defaults to 0.6 (the paper value); pass another value to sweep
    the reference-smoothing weight without changing any existing paper run.

    alpha / beta / lossaversion_fixed default to the values that were
    historically HARD-CODED inside this function (0.15, 4e-5, 1.5) so old
    behavior is reproducible -- but note these are what every session actually
    trains with, regardless of the attributes on the ``game`` object passed in
    (config.csv used to record the parent game's values, which could differ).
    Pass them explicitly to change.
    """
    if num_processes is None:
        num_processes = max(1, mp.cpu_count() - 2)
        print('num_process', num_processes)

    # Run sessions in parallel with error handling
    print(f"Starting parallel processing with {num_processes} processes for {num_sessions} sessions")
    print(f"  alpha={alpha}, beta={beta:g}, lambda={lambda_fixed}, "
          f"lossaversion={lossaversion_fixed}, "
          f"continuous_reference={getattr(game, 'continuous_reference', False)}")

    alpha_fixed = alpha
    beta_fixed = beta

    for i, gamma in enumerate(gamma_values):
        
        # Check if this gamma-lambda combination has already been run
        run_dir = os.path.join("../Results/experiments", experiment_name, f"gamma_{gamma}")
        stats_file = os.path.join(run_dir, "cycle_statistics.csv")

        # Update game parameters
        game.alpha = alpha_fixed
        game.beta = beta_fixed
        game.gamma = gamma
        game.lambda_ = lambda_fixed
        game.demand_type = demand_type
        game.p_minmax = game.compute_p_competitive_monopoly()
        game.NashProfits,  game.CoopProfits = game.compute_profits_nash_coop()
        game.p_nash, game.p_coop = game.p_minmax[0], game.p_minmax[1]
        game.PI = game.init_PI()
        game.Q = game.init_Q()
        game.num_sessions = num_sessions

        if os.path.exists(stats_file):
            print(f"gamma_{gamma} (already exists in {run_dir})")
            # Load existing stats
            df = pd.read_csv(stats_file)

            # Only add p_nash and p_coop if not already present
            if 'p_nash_p1' not in df.columns:
                for i, val in enumerate(game.p_nash):
                    df[f'p_nash_p{i+1}'] = val
                for i, val in enumerate(game.p_coop):
                    df[f'p_coop_p{i+1}'] = val

                df.to_csv(stats_file, index=False)
                print(f"Added p_nash and p_coop per player to {stats_file}")

            else:
                print("p_nash and p_coop already present.")
            continue  # Skip running simulation again


        # Game logs 
        if game.common_reference:
            ref_shape = (1,)  # single common reference price
        else:
            ref_shape = (game.n,)  # each firm has its own reference price
        # Reset and initialize game arrays for the new experiment
        game.converged = np.zeros(game.num_sessions, dtype=bool)
        game.time_to_convergence = np.zeros(game.num_sessions, dtype=float)
        game.index_last_state = np.zeros((game.n, game.memory, game.num_sessions), dtype=int)
        game.index_last_reference = np.zeros(ref_shape + (game.num_sessions,), dtype=int)
        game.cycle_length = np.zeros(game.num_sessions, dtype=int)
        game.cycle_states = np.zeros((game.num_periods, game.num_sessions), dtype=int)
        game.cycle_prices = np.zeros((game.n, game.num_periods, game.num_sessions), dtype=float)
        game.cycle_profits = np.zeros((game.n, game.num_periods, game.num_sessions), dtype=float)
        game.cycle_reference_prices = np.zeros(ref_shape + (game.num_periods, game.num_sessions), dtype=float)
        game.cycle_consumer_surplus = np.zeros((game.num_periods, game.num_sessions), dtype=float) 
        game.index_strategies = np.zeros((game.n,) + game.sdim + (game.num_sessions,), dtype=int)
        game.last_observed_prices = np.zeros((game.n, game.memory), dtype=int)  # last prices
        game.last_observed_reference = np.zeros(ref_shape, dtype=int)
        game.last_reference_observed_prices = np.zeros((game.n, game.reference_memory), dtype=int)  # last prices
        game.last_observed_demand = np.zeros((game.n, game.reference_memory), dtype=float)  # last shares for each firm


        #if game.aprint:
        print(f"\n Gamma_{gamma} with {num_processes} processes")
        
        try:
            # Run sessions in parallel with error handling
            with mp.Pool(processes=num_processes) as pool:
                session_results = []
                for iSession in range(num_sessions):
                    result = pool.apply_async(
                        run_single_session,
                        args=(game, gamma, lambda_fixed, lossaversion_fixed, iSession,
                              use_reference_pretraining, T_ref, alpha, beta),
                    )
                    session_results.append(result)
                
                # ✅ Use improved result collection here
                results = []
                for i, res in enumerate(session_results):
                    try:
                        result = res.get(timeout=session_timeout)
                        results.append(result)
                    except Exception as e:
                        print(f"Session {i} failed or timed out: {e}")
                        continue
            

            # Process results
            for result in results:
                iSession = result['session_id']
                game.converged[iSession] = result['converged']
                game.time_to_convergence[iSession] = result['time_to_convergence']
                game.index_last_state[:, :, iSession] = result['last_observed_prices']
                game.index_strategies[..., iSession] = result['optimal_strategies']

                # If using reference pricing, store reference-related results
                if game.demand_type in ["reference", "misspecification"]:
                    game.index_last_reference[:, iSession] = result['last_observed_reference']

                
                if result['cycle_data'] is not None:
                    cycle_data = result['cycle_data']
                    game.cycle_length[iSession] = cycle_data['cycle_length']
                    cycle_len = cycle_data['cycle_length']
                    game.cycle_states[:cycle_len, iSession] = cycle_data['visited_states']
                    game.cycle_prices[:, :cycle_len, iSession] = cycle_data['price_history']
                    game.cycle_profits[:, :cycle_len, iSession] = cycle_data['visited_profits']
                    game.cycle_consumer_surplus[:cycle_len, iSession] = cycle_data['consumer_surplus_history']
                    if game.demand_type in ["reference", "misspecification"]:
                        game.cycle_reference_prices[:, :cycle_len, iSession] = cycle_data['reference_price_history']

            # Save results for this gamma-lambda combination
            run_dir = save_experiment_gamma_only(game, experiment_name, gamma)

            # Persist per-session Q-stabilization trajectories when tracked.
            # Each session's trajectory is an (m, 3) array of
            # [t, mean|dQ_firm|, mean|dQ_ref|]; also record whether the session
            # converged and its convergence time for later joint analysis.
            q_diag_arrays = {
                f"session_{r['session_id']}": r['q_diag']
                for r in results
                if r.get('q_diag') is not None and np.size(r['q_diag']) > 0
            }
            if q_diag_arrays:
                conv_flags = np.array(
                    [[r['session_id'], int(bool(r['converged'])),
                      float(r['time_to_convergence'])] for r in results],
                    dtype=float)
                np.savez_compressed(
                    os.path.join(run_dir, "q_stabilization.npz"),
                    convergence=conv_flags,
                    **q_diag_arrays)
                if game.aprint:
                    print(f"Saved Q-stabilization trajectories for "
                          f"{len(q_diag_arrays)} sessions.")

            if game.aprint:
                print(f"Completed gamma = {gamma}")
                print(f"Results saved in {run_dir}")
                
        except Exception as e:
            print(f"Error processing gamma = {gamma}: {str(e)}")
            import traceback
            traceback.print_exc()  # Print full error details
            continue

    print("\nAll experiments completed.")
    return game
