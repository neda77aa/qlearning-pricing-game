import numpy as np
import pandas as pd
import os
from datetime import datetime
from input.qlearning import simulate_game, run_sessions, detect_cycle
import matplotlib.pyplot as plt
import os
from glob import glob
import multiprocessing as mp
from functools import partial
import copy


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
            
    def get_run_dir(self, alpha, beta):
        """Create directory for specific alpha-beta combination"""
        run_name = f"alpha_{alpha}_beta_{beta}" #_{self.timestamp}"
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
            'Lossaversion_aversion': game.lossaversion,
            'Lambda' : game.lambda_,
            'a0': game.a0,
            'a1': game.a,
            'a2': game.a,
            'c1': game.c,
            'c2': game.c,
            'mu': game.mu,
            'extend1': game.extend,
            'extend2': game.extend,
            'NashP1': game.NashProfits[0],
            'NashP2': game.NashProfits[1],
            'CoopP1': game.CoopProfits[0],
            'CoopP2': game.CoopProfits[1],
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
            'cycle_length': np.where(game.converged, game.cycle_length, np.nan)
        }

        # Add cycle prices and profits for each player
        for i_player in range(game.n):
            player_num = i_player + 1
            prices_list = []
            profits_list = []
            reference_prices_list = []
            consumer_surplus_list = []

            # Extract prices and profits only up to cycle length for each session
            for i_session in range(game.num_sessions):
                cycle_len = game.cycle_length[i_session]
                prices = game.cycle_prices[i_player, :cycle_len, i_session]
                profits = game.cycle_profits[i_player, :cycle_len, i_session]
                consumer_surplus = game.cycle_consumer_surplus[:cycle_len, i_session]  # Extract CS


                if game.demand_type in ["reference", "misspecification"]:
                    # ↳ change this block
                    if game.common_reference:
                        ref_slice = game.cycle_reference_prices[0, :cycle_len, i_session]
                        reference_prices_list.append(','.join(f"{r:.5g}" for r in ref_slice))
                    else:
                        # one string per firm, separated by ‘;’
                        firm_strings = []
                        for f in range(game.n):
                            ref_slice = game.cycle_reference_prices[f, :cycle_len, i_session]
                            firm_strings.append(','.join(f"{r:.5g}" for r in ref_slice))
                        reference_prices_list.append(';'.join(firm_strings))
                
                # Convert arrays to strings with comma separation, formatting to 5 digits
                prices_str = ','.join([f"{p:.5g}" for p in prices])
                profits_str = ','.join([f"{p:.5g}" for p in profits])
                consumer_surplus_str = ','.join([f"{cs:.5g}" for cs in consumer_surplus])

                
                prices_list.append(prices_str)
                profits_list.append(profits_str)
                consumer_surplus_list.append(consumer_surplus_str)
            
            session_summaries[f'cycle_prices_p{player_num}'] = prices_list
            session_summaries[f'cycle_profits_p{player_num}'] = profits_list
        session_summaries[f'cycle_consumer_surplus'] = consumer_surplus_list

        # Add reference prices if reference demand is used
        if game.demand_type in ["reference", "misspecification"]:
            session_summaries[f'cycle_reference_prices'] = reference_prices_list
        
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

        if game.demand_type in ["reference", "misspecification"]:
            if game.common_reference:
                mean_reference_prices = np.zeros(game.num_sessions)
                std_reference_prices  = np.zeros(game.num_sessions)
            else:
                mean_reference_prices = np.zeros((game.n, game.num_sessions))
                std_reference_prices  = np.zeros((game.n, game.num_sessions))

                
        for i_session in range(game.num_sessions):
            cycle_len = game.cycle_length[i_session]
            # Compute mean consumer surplus
            mean_consumer_surplus[i_session] = np.mean(game.cycle_consumer_surplus[:cycle_len, i_session])

            for i_player in range(game.n):
                mean_profits[i_player, i_session] = np.mean(game.cycle_profits[i_player, :cycle_len, i_session])
                profit_gains[i_player, i_session] = (mean_profits[i_player, i_session] - game.NashProfits[i_player]) / (game.CoopProfits[i_player] - game.NashProfits[i_player])

                # Convert price indexes to actual price values
                actual_prices = np.asarray(game.A[np.asarray(game.cycle_prices[i_player, :cycle_len, i_session], dtype=int)])
                mean_prices[i_player, i_session] = np.mean(actual_prices)
                price_gains[i_player, i_session] = (mean_prices[i_player, i_session] - game.p_nash[i_player]) / (game.p_coop[i_player] - game.p_nash[i_player])
               
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
                f'mean_price_p{player_num}': f"{np.nanmean(mean_prices[i_player]):.5g}"
            })

        cycle_stats.update({
                'mean_consumer_surplus': f"{np.nanmean(mean_consumer_surplus):.5g}",
            })

        
        if game.demand_type in ["reference", "misspecification"]:
            if game.common_reference:
                cycle_stats.update({
                    'mean_reference_price': f"{np.nanmean(mean_reference_prices):.5g}",
                    'std_reference_price' : f"{np.nanstd(std_reference_prices):.5g}",
                })
            else:
                for f in range(game.n):
                    cycle_stats.update({
                        f'mean_reference_price_p{f+1}': f"{np.nanmean(mean_reference_prices[f]):.5g}",
                        f'std_reference_price_p{f+1}' : f"{np.nanstd(std_reference_prices[f]):.5g}",
                    })


        df_stats = pd.DataFrame([cycle_stats])
        df_stats.to_csv(os.path.join(run_dir, "cycle_statistics.csv"), index=False)

def save_experiment(game, experiment_name, alpha, beta):
    """Main function to save all experiment data"""
    saver = ExperimentSaver(experiment_name)
    run_dir = saver.get_run_dir(alpha, beta)  # Use the get_run_dir method
    
    # Save all components
    saver.save_experiment_config(game, run_dir)
    saver.save_session_results(game, run_dir)
    saver.save_cycle_statistics(game, run_dir)
    
    return run_dir




###############################################
######## Run Experiment 


###############################

## Single Computing Session 

def run_experiment(game, alpha_values, beta_values, num_sessions=1000, demand_type = 'noreference', experiment_name = 'test'):
    """
    Run experiments with different alpha and beta values
    
    Parameters:
    -----------
    game : object
        Game instance
    alpha_values : array-like
        Array of alpha values to test
    beta_values : array-like
        Array of beta values to test
    num_sessions : int
        Number of sessions per experiment
    num_processes : int, optional
        Number of processes to use. If None, uses CPU count - 1
    """
    
    for i, alpha in enumerate(alpha_values):
        for j, beta in enumerate(beta_values):
            # Configure experiment
            experiment_id = f"alpha_{alpha}_beta_{beta}"
            
            # Update game parameters for this experiment
            game.alpha = alpha
            game.beta = beta
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
            

            # Run all sessions for this alpha-beta combination
            for iSession in range(game.num_sessions):
                if game.aprint:
                    print(f"\nStarting Session {iSession + 1}/{game.num_sessions}")
                    print(f"Current alpha: {alpha}, beta: {beta}")

                game.Q = game.init_Q()  # Reset Q-values
                game.last_observed_prices = np.zeros((game.n, game.memory), dtype=int)  # Reset prices


                # Run Q-learning for this session
                game, converged, t_convergence = simulate_game(game)

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
                        cycle_length, visited_states, visited_profits, price_history, reference_price_history, consumer_surplus_history = detect_cycle(game, iSession)  # Now passing iSession
                        cycle_data = {
                            'cycle_length': cycle_length,
                            'visited_states': visited_states,
                            'visited_profits': visited_profits,
                            'price_history': price_history,
                            'reference_price_history': reference_price_history,
                            'consumer_surplus_history': consumer_surplus_history
                        }
            # Save results for this alpha-beta combination
            run_dir = save_experiment(game, experiment_name, alpha, beta)

            if game.aprint:
                print(f"\nCompleted experiment for alpha={alpha}, beta={beta}")
                print(f"Results saved under experiment ID: {experiment_id}")

    print("\nAll experiments completed.")
    return game




###############################

## Parallel Computing Section

def run_single_session(game, alpha, beta, iSession):
    """
    Run a single session of the game
    
    Parameters:
    -----------
    game : object
        Game instance
    iSession : int
        Session number
    
    Returns:
    --------
    dict : Session results
    """
    # Create a deep copy of game to avoid shared state issue

    # # Update game parameters
    game.alpha = alpha
    game.beta = beta

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
        game_copy.last_observed_reference = np.zeros(1 if game_copy.common_reference
                                             else game_copy.n, dtype=int)
        game_copy.last_reference_observed_prices = np.zeros((game_copy.n, game_copy.reference_memory), dtype=int)
        game_copy.last_observed_demand = np.zeros((game_copy.n, game_copy.reference_memory), dtype=float)

    # Run simulation
    game_copy, converged, t_convergence = simulate_game(game_copy)

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
        if game_copy.demand_type == 'noreference':
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
            cycle_length, visited_states, visited_profits, price_history, reference_price_history, consumer_surplus_history = detect_cycle(game_copy, iSession)  # Now passing iSession
            cycle_data = {
                'cycle_length': cycle_length,
                'visited_states': visited_states,
                'visited_profits': visited_profits,
                'price_history': price_history,
                'reference_price_history': reference_price_history,
                'consumer_surplus_history': consumer_surplus_history
            }

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
            'last_observed_demand': last_observed_demand
        }
    
    
    # Return results
    return {
        'session_id': iSession,
        'converged': converged,
        'time_to_convergence': t_convergence,
        'last_observed_prices': game_copy.last_observed_prices,
        'optimal_strategies': game_copy.Q.argmax(axis=-1),
        'cycle_data': cycle_data
    }


def run_experiment_parallel(game, alpha_values, beta_values, num_sessions=1000, experiment_name='test',  demand_type = 'noreference', num_processes=None):
    """
    Run experiments with different alpha and beta values using parallel processing
    """
    if num_processes is None:
        num_processes = max(1, mp.cpu_count() - 2)
        print('num_process', num_processes)

    # Run sessions in parallel with error handling
    print(f"Starting parallel processing with {num_processes} processes for {num_sessions} sessions")
    
    for i, alpha in enumerate(alpha_values):
        for j, beta in enumerate(beta_values):

            # Check if this alpha-beta combination has already been run
            run_dir = os.path.join("../Results/experiments", experiment_name, f"alpha_{alpha}_beta_{beta}")
            stats_file = os.path.join(run_dir, "cycle_statistics.csv")

            if os.path.exists(stats_file):
                print(f"Skipping alpha={alpha}, beta={beta} (already exists in {run_dir})")
                continue  # Skip already completed experiments

            # Update game parameters
            game.alpha = alpha
            game.beta = beta
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
            print(f"\nStarting alpha={alpha}, beta={beta} with {num_processes} processes")
            
            try:
                # Run sessions in parallel with error handling
                with mp.Pool(processes=num_processes) as pool:
                    session_results = []
                    for iSession in range(num_sessions):
                        result = pool.apply_async(run_single_session, args=(game, alpha, beta, iSession))
                        session_results.append(result)
                    
                    # Collect results with timeout
                    results = [res.get(timeout=600) for res in session_results]
                

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

                # Save results for this alpha-beta combination
                run_dir = save_experiment(game, experiment_name, alpha, beta)
                
                if game.aprint:
                    print(f"Completed alpha={alpha}, beta={beta}")
                    print(f"Results saved in {run_dir}")
                    
            except Exception as e:
                print(f"Error processing alpha={alpha}, beta={beta}: {str(e)}")
                import traceback
                traceback.print_exc()  # Print full error details
                continue

    print("\nAll experiments completed.")
    return game


