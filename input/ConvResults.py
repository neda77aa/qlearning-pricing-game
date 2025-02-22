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
        run_name = f"alpha_{alpha}_beta_{beta}_{self.timestamp}"
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
            'cycle_length': game.cycle_length
        }

        # Add cycle prices and profits for each player
        for i_player in range(game.n):
            player_num = i_player + 1
            prices_list = []
            profits_list = []
            reference_prices_list = []

            # Extract prices and profits only up to cycle length for each session
            for i_session in range(game.num_sessions):
                cycle_len = game.cycle_length[i_session]
                prices = game.cycle_prices[i_player, :cycle_len, i_session]
                profits = game.cycle_profits[i_player, :cycle_len, i_session]

                # If reference pricing is enabled, store reference prices
                if game.demand_type == 'reference':
                    reference_prices = game.cycle_reference_prices[0, :cycle_len, i_session]  # Extract ref prices
                    reference_prices_str = ','.join([f"{r:.5g}" for r in reference_prices])  # Format as string
                    reference_prices_list.append(reference_prices_str)
                
                # Convert arrays to strings with comma separation, formatting to 5 digits
                prices_str = ','.join([f"{p:.5g}" for p in prices])
                profits_str = ','.join([f"{p:.5g}" for p in profits])
                
                prices_list.append(prices_str)
                profits_list.append(profits_str)
            
            session_summaries[f'cycle_prices_p{player_num}'] = prices_list
            session_summaries[f'cycle_profits_p{player_num}'] = profits_list

            # Add reference prices if reference demand is used
            if game.demand_type == 'reference':
                session_summaries[f'cycle_reference_prices_p{player_num}'] = reference_prices_list
            
        df_summaries = pd.DataFrame(session_summaries)
        df_summaries.to_csv(os.path.join(run_dir, "session_summaries.csv"), index=False)
        
        # Save compressed arrays for detailed data
        np.savez_compressed(
            os.path.join(run_dir, "session_details.npz"),
            cycle_states=game.cycle_states,
            cycle_prices=game.cycle_prices,
            cycle_profits=game.cycle_profits,
            index_strategies=game.index_strategies,
            cycle_reference_prices=game.cycle_reference_prices  # Include reference prices in saved file
        )

    def save_cycle_statistics(self, game, run_dir):
        """Save cycle statistics across all sessions"""
        os.makedirs(run_dir, exist_ok=True)

        # Calculate mean profits only up to cycle length for each session
        mean_profits = np.zeros((game.n, game.num_sessions))
        profit_gains = np.zeros((game.n, game.num_sessions))
        mean_prices = np.zeros((game.n, game.num_sessions))

        # If using reference pricing, store reference price statistics
        if game.demand_type == 'reference':
            mean_reference_prices = np.zeros(game.num_sessions)
            std_reference_prices = np.zeros(game.num_sessions)
        
        for i_session in range(game.num_sessions):
            cycle_len = game.cycle_length[i_session]
            for i_player in range(game.n):
                mean_profits[i_player, i_session] = np.mean(game.cycle_profits[i_player, :cycle_len, i_session])
                profit_gains[i_player, i_session] = (mean_profits[i_player, i_session] - game.NashProfits[i_player]) / (game.CoopProfits[i_player] - game.NashProfits[i_player])
                # Convert price indexes to actual price values
                actual_prices = np.asarray(game.A[np.asarray(game.cycle_prices[i_player, :cycle_len, i_session], dtype=int)])
                mean_prices[i_player, i_session] = np.mean(actual_prices)

            # Compute reference price statistics
            if game.demand_type == 'reference':
                reference_prices = np.asarray(game.A[np.asarray(game.cycle_reference_prices[0, :cycle_len, i_session], dtype=int)]) # Extract reference prices
                mean_reference_prices[i_session] = np.mean(reference_prices)
                std_reference_prices[i_session] = np.std(reference_prices)

                # Convert reference prices to string format and store
                ref_prices_str = ','.join([f"{p:.5g}" for p in reference_prices])
        

        # Calculate statistics
        cycle_stats = {
            'mean_cycle_length': f"{np.mean(game.cycle_length):.5g}",
            'std_cycle_length': f"{np.std(game.cycle_length):.5g}",
            'convergence_rate': f"{np.mean(game.converged):.5g}",
            'mean_convergence_time': f"{np.mean(game.time_to_convergence):.5g}"
        }
        
        # Add statistics for each player
        for i_player in range(game.n):
            player_num = i_player + 1
            cycle_stats.update({
                f'mean_profit_p{player_num}': f"{np.mean(mean_profits[i_player]):.5g}",
                f'std_profit_p{player_num}': f"{np.std(mean_profits[i_player]):.5g}",
                f'mean_profit_gain_p{player_num}': f"{np.mean(profit_gains[i_player]):.5g}",
                f'std_profit_gain_p{player_num}': f"{np.std(profit_gains[i_player]):.5g}",
                f'mean_price_p{player_num}': f"{np.mean(mean_prices[i_player]):.5g}"
            })
        
        # Add reference price statistics if applicable
        if game.demand_type == 'reference':
            cycle_stats.update({
                'mean_reference_price': f"{np.mean(mean_reference_prices):.5g}",
                'std_reference_price': f"{np.std(mean_reference_prices):.5g}"
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
            
            # Reset and initialize game arrays for the new experiment
            game.converged = np.zeros(game.num_sessions, dtype=bool)
            game.time_to_convergence = np.zeros(game.num_sessions, dtype=float)
            game.index_last_state = np.zeros((game.n, game.memory, game.num_sessions), dtype=int)
            game.cycle_length = np.zeros(game.num_sessions, dtype=int)
            game.cycle_states = np.zeros((game.num_periods, game.num_sessions), dtype=int)
            game.cycle_prices = np.zeros((game.n, game.num_periods, game.num_sessions), dtype=float)
            game.cycle_profits = np.zeros((game.n, game.num_periods, game.num_sessions), dtype=float)
            game.index_strategies = np.zeros((game.n,) + game.sdim + (game.num_sessions,), dtype=int)
            game.last_observed_prices = np.zeros((game.n, game.memory), dtype=int)  # last prices
            game.last_observed_reference = np.zeros(1, dtype=int)  # last prices
            game.last_reference_observed_prices = np.zeros((game.n, game.reference_memory), dtype=int)  # last prices
            game.last_observed_demand = np.zeros((game.n, game.reference_memory), dtype=float)  # last shares for each firm


            # Run all sessions for this alpha-beta combination
            for iSession in range(game.num_sessions):
                if game.aprint:
                    print(f"\nStarting Session {iSession + 1}/{game.num_sessions}")
                    print(f"Current alpha: {alpha}, beta: {beta}")

                game.Q = game.init_Q()  # Reset Q-values
                game.last_observed_prices = np.zeros((game.n, game.memory), dtype=int)  # Reset prices

                if game.demand_type == 'reference':
                    game.last_reference_observed_prices = np.zeros((game.n, game.reference_memory), dtype=int)  # last prices
                    game.last_observed_demand = np.zeros((game.n, game.reference_memory), dtype=float)  # last shares for each firm


                # Run Q-learning for this session
                game, converged, t_convergence = simulate_game(game)

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
                    cycle_length, visited_states, visited_profits, price_history = detect_cycle(game, iSession)
                    if game.aprint:
                        print('cycle length:', cycle_length)
                        print('visited_states:', visited_states)
                        print('visited_profits:', visited_profits)
                        print('price_history:', price_history)

            # Save results for this alpha-beta combination
            run_dir = save_experiment(game, experiment_name, alpha, beta)

            if game.aprint:
                print(f"\nCompleted experiment for alpha={alpha}, beta={beta}")
                print(f"Results saved under experiment ID: {experiment_id}")

    print("\nAll experiments completed.")
    return game



def create_profit_gain_heatmap(results_dir, player_num=1, experiment_name="*", metric_name="Profit Gain",figsize=(10, 8)):
    """
    Creates a heatmap of mean profit gains for a specific player across different alpha and beta values.
    
    Parameters:
    -----------
    results_dir : str
        Directory containing experiment results (can be relative or absolute)
    player_num : int
        Player number to plot (1, 2, etc.)
    experiment_name : str
        Pattern to match specific experiments (default "*" matches all)
    figsize : tuple
        Figure size in inches
    """

    # Resolve the full path if a relative path is provided
    results_dir = os.path.abspath(results_dir)
    
    # First get the experiment directory
    exp_dir = os.path.join(results_dir, experiment_name)
    if not os.path.exists(exp_dir):
        raise ValueError(f"No experiment directory found: {exp_dir}")
    
    
    # Get all alpha-beta directories (including timestamps)
    pattern = "alpha_*_beta_*_*"  # Matches "alpha_X_beta_Y_timestamp"
    run_dirs = glob(os.path.join(exp_dir, pattern))
    
    if not run_dirs:
        raise ValueError(f"No run directories found matching pattern '{pattern}' in {exp_dir}")
    

    # Extract alpha and beta values from directory names
    alpha_values = []
    beta_values = []
    metric_values = []
    
    for run_dir in run_dirs:
        try:
            # Extract alpha and beta from directory name
            dir_name = os.path.basename(run_dir)
            # The format should be "alpha_X_beta_Y_timestamp"
            alpha_str = dir_name.split('alpha_')[1].split('_beta_')[0]
            beta_str = dir_name.split('beta_')[1].split('_')[0]
            
            alpha = float(alpha_str)
            beta = float(beta_str)
            
            # Read cycle statistics
            stats_file = os.path.join(run_dir, "cycle_statistics.csv")
            if os.path.exists(stats_file):
                df = pd.read_csv(stats_file)
                metric_value = float(df[f'mean_{metric_name.lower().replace(" ", "_")}_p{player_num}'].iloc[0])
                
                alpha_values.append(alpha)
                beta_values.append(beta)
                metric_values.append(metric_value)
                
        except (IndexError, ValueError) as e:
            print(f"Skipping directory {dir_name} due to parsing error: {e}")
            continue
    
    if not alpha_values:
        raise ValueError("No valid data found to create heatmap")
    
    
    # Create and return the heatmap
    fig = _create_heatmap(alpha_values, beta_values, metric_values, player_num, metric_name, figsize)
    return fig


def _create_heatmap(alpha_values, beta_values, data_values, player_num, metric_name, figsize):
    """
    Helper function to create heatmaps for different metrics (Profit Gain or Mean Price).

    Parameters:
    -----------
    alpha_values : list
        List of alpha values
    beta_values : list
        List of beta values
    data_values : list
        Corresponding values for the heatmap
    player_num : int
        Player number being plotted
    metric_name : str
        Title of the heatmap (e.g., "Profit Gain", "Mean Price")
    figsize : tuple
        Size of the figure
    """

    # Convert inputs to numpy arrays
    alpha_values = np.array(alpha_values)
    beta_values = np.array(beta_values)
    data_values = np.array(data_values)

    # Get unique values for the grid
    unique_alphas = np.sort(np.unique(alpha_values))
    unique_betas = np.sort(np.unique(beta_values))

    # Create grid for the heatmap
    data_grid = np.zeros((len(unique_alphas), len(unique_betas)))

    # Fill grid with values
    for alpha, beta, value in zip(alpha_values, beta_values, data_values):
        i = np.where(unique_alphas == alpha)[0][0]  # Find row index
        j = np.where(unique_betas == beta)[0][0]  # Find column index
        data_grid[i, j] = value

    if figsize is not None:
        plt.figure(figsize=figsize)

    # Set colormap based on metric
    cmap_choice = 'Reds' if "Profit Gain" in metric_name else 'Blues'

    # Create the heatmap
    im = plt.imshow(
        data_grid, 
        aspect='auto',
        origin='lower',
        extent=[min(unique_betas), max(unique_betas), min(unique_alphas), max(unique_alphas)],
        cmap=cmap_choice
    )

    # Add color bar
    plt.colorbar(im, label=metric_name)

    # Set labels and title
    plt.xlabel(r'$\beta \times 10^5$')
    plt.ylabel(r'$\alpha$')
    plt.title(f'{metric_name} - Player {player_num}')

    return plt.gcf()


###############################

## Parallel Computing Section

def run_single_session(game, iSession):
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
    # Create a deep copy of game to avoid shared state issues
    game_copy = copy.deepcopy(game)
    
    # Initialize session-specific variables
    game_copy.Q = game_copy.init_Q()
    game_copy.last_observed_prices = np.zeros((game_copy.n, game_copy.memory), dtype=int)

    # Run simulation
    game_copy, converged, t_convergence = simulate_game(game_copy)

    # Store convergence results in game_copy
    game_copy.converged[iSession] = converged
    game_copy.time_to_convergence[iSession] = t_convergence
    
    # Store last observed prices in game_copy
    game_copy.index_last_state[:, :, iSession] = game_copy.last_observed_prices
    
    # Store the learned strategies in game_copy
    game_copy.index_strategies[..., iSession] = game_copy.Q.argmax(axis=-1)

    # If converged, get cycle data
    cycle_data = None
    if converged:
        # Pass iSession to detect_cycle function
        cycle_length, visited_states, visited_profits, price_history = detect_cycle(game_copy, iSession)  # Now passing iSession
        cycle_data = {
            'cycle_length': cycle_length,
            'visited_states': visited_states,
            'visited_profits': visited_profits,
            'price_history': price_history
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


def run_experiment_parallel(game, alpha_values, beta_values, num_sessions=1000, experiment_name='test', num_processes=None):
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
            # Update game parameters
            game.alpha = alpha
            game.beta = beta
            game.num_sessions = num_sessions
            
            # Initialize arrays for results
            game.converged = np.zeros(game.num_sessions, dtype=bool)
            game.time_to_convergence = np.zeros(game.num_sessions, dtype=float)
            game.index_last_state = np.zeros((game.n, game.memory, game.num_sessions), dtype=int)
            game.cycle_length = np.zeros(game.num_sessions, dtype=int)
            game.cycle_states = np.zeros((game.num_periods, game.num_sessions), dtype=int)
            game.cycle_prices = np.zeros((game.n, game.num_periods, game.num_sessions), dtype=float)
            game.cycle_profits = np.zeros((game.n, game.num_periods, game.num_sessions), dtype=float)
            game.index_strategies = np.zeros((game.n,) + game.sdim + (game.num_sessions,), dtype=int)
            game.last_observed_reference = np.zeros(1, dtype=int)  # last prices
            game.last_reference_observed_prices = np.zeros((game.n, game.reference_memory), dtype=int)  # last prices
            game.last_observed_demand = np.zeros((game.n, game.reference_memory), dtype=float)  # last shares for each firm


            #if game.aprint:
            print(f"\nStarting alpha={alpha}, beta={beta} with {num_processes} processes")
            
            try:
                # Run sessions in parallel with error handling
                with mp.Pool(processes=num_processes) as pool:
                    session_results = []
                    for iSession in range(num_sessions):
                        result = pool.apply_async(run_single_session, args=(game, iSession))
                        session_results.append(result)
                    
                    # Collect results with timeout
                    results = [res.get(timeout=300) for res in session_results]
                

                # Process results
                for result in results:
                    
                    iSession = result['session_id']
                    game.converged[iSession] = result['converged']
                    game.time_to_convergence[iSession] = result['time_to_convergence']
                    game.index_last_state[:, :, iSession] = result['last_observed_prices']
                    game.index_strategies[..., iSession] = result['optimal_strategies']

            
                    
                    if result['cycle_data'] is not None:
                        cycle_data = result['cycle_data']
                        game.cycle_length[iSession] = cycle_data['cycle_length']
                        cycle_len = cycle_data['cycle_length']
                        game.cycle_states[:cycle_len, iSession] = cycle_data['visited_states']
                        game.cycle_prices[:, :cycle_len, iSession] = cycle_data['price_history']
                        game.cycle_profits[:, :cycle_len, iSession] = cycle_data['visited_profits']
                
                # Save results for this alpha-beta combination
                run_dir = save_experiment(game, experiment_name, alpha, beta)
                
                if game.aprint:
                    print(f"Completed alpha={alpha}, beta={beta}")
                    print(f"Results saved in {run_dir}")
                    
            except Exception as e:
                print(f"Error processing alpha={alpha}, beta={beta}: {str(e)}")
                continue

    print("\nAll experiments completed.")
    return game