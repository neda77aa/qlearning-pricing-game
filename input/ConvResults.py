import numpy as np
import pandas as pd
import os
from datetime import datetime
from input.qlearning import simulate_game, run_sessions, detect_cycle
import matplotlib.pyplot as plt
import os
from glob import glob

class ExperimentSaver:
    def __init__(self, base_dir="experiments"):
        """Initialize experiment saver with base directory"""
        self.base_dir = base_dir
        self._setup_directories()
        
    def _setup_directories(self):
        """Create necessary directories if they don't exist"""
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
            
    def create_experiment_id(self, params):
        """Create unique experiment identifier"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"exp_{timestamp}"
        
    def save_experiment_config(self, game, experiment_id):
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
        config_path = os.path.join(self.base_dir, experiment_id, "config.csv")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        df.to_csv(config_path, index=False)
        
    def save_session_results(self, game, experiment_id):
        """Save aggregated session results"""
        session_dir = os.path.join(self.base_dir, experiment_id, "sessions")
        os.makedirs(session_dir, exist_ok=True)
        
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
            
            # Extract prices and profits only up to cycle length for each session
            for i_session in range(game.num_sessions):
                cycle_len = game.cycle_length[i_session]
                prices = game.cycle_prices[i_player, :cycle_len, i_session]
                profits = game.cycle_profits[i_player, :cycle_len, i_session]
                
                # Convert arrays to strings with comma separation
                prices_str = ','.join(map(str, prices))
                profits_str = ','.join(map(str, profits))
                
                prices_list.append(prices_str)
                profits_list.append(profits_str)
            
            session_summaries[f'cycle_prices_p{player_num}'] = prices_list
            session_summaries[f'cycle_profits_p{player_num}'] = profits_list
        
        
        df_summaries = pd.DataFrame(session_summaries)
        df_summaries.to_csv(os.path.join(session_dir, "session_summaries.csv"), index=False)
        
        # Save compressed arrays for detailed data
        np.savez_compressed(
            os.path.join(session_dir, "session_details.npz"),
            cycle_states=game.cycle_states,
            cycle_prices=game.cycle_prices,
            cycle_profits=game.cycle_profits,
            index_strategies=game.index_strategies
        )
        
    def save_cycle_statistics(self, game, experiment_id):
        """Save cycle statistics across all sessions"""
        stats_dir = os.path.join(self.base_dir, experiment_id)
        os.makedirs(stats_dir, exist_ok=True)

        # Calculate mean profits only up to cycle length for each session
        mean_profits = np.zeros((game.n, game.num_sessions))
        profit_gains = np.zeros((game.n, game.num_sessions))
        
        for i_session in range(game.num_sessions):
            cycle_len = game.cycle_length[i_session]
            for i_player in range(game.n):
                mean_profits[i_player, i_session] = np.mean(game.cycle_profits[i_player, :cycle_len, i_session])
                profit_gains[i_player, i_session] = (mean_profits[i_player, i_session] - game.NashProfits[i_player]) / (game.CoopProfits[i_player] - game.NashProfits[i_player])
        
       # Calculate statistics
        cycle_stats = {
            'mean_cycle_length': np.mean(game.cycle_length),
            'std_cycle_length': np.std(game.cycle_length),
            'convergence_rate': np.mean(game.converged),
            'mean_convergence_time': np.mean(game.time_to_convergence)
        }
        
        # Add statistics for each player
        for i_player in range(game.n):
            player_num = i_player + 1  # For naming convention (player 1, player 2, etc.)
            cycle_stats.update({
                f'mean_profit_p{player_num}': f"{np.mean(mean_profits[i_player]):.5g}",
                f'std_profit_p{player_num}': f"{np.std(mean_profits[i_player]):.5g}",
                f'mean_profit_gain_p{player_num}': f"{np.mean(profit_gains[i_player]):.5g}",
                f'std_profit_gain_p{player_num}': f"{np.std(profit_gains[i_player]):.5g}"
            })

        df_stats = pd.DataFrame([cycle_stats])
        df_stats.to_csv(os.path.join(stats_dir, "cycle_statistics.csv"), index=False)


def save_experiment(game):
    """Main function to save all experiment data"""
    saver = ExperimentSaver()
    experiment_id = saver.create_experiment_id({})
    
    # Save all components
    saver.save_experiment_config(game, experiment_id)
    saver.save_session_results(game, experiment_id)
    saver.save_cycle_statistics(game, experiment_id)
    
    return experiment_id


def run_experiment(game, alpha_values, beta_values, num_sessions=1000):
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
    """
    results_handler = ExperimentSaver()
    
    for i, alpha in enumerate(alpha_values):
        for j, beta in enumerate(beta_values):
            # Configure experiment
            experiment_id = f"alpha_{alpha}_beta_{beta}"
            
            # Update game parameters for this experiment
            game.alpha = alpha
            game.beta = beta
            game.num_sessions = num_sessions
            
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

                # If converged, analyze post-convergence cycles
                if converged:
                    cycle_length, visited_states, visited_profits, price_history = detect_cycle(game,game.index_strategies[:, iSession], iSession)
                    if game.aprint:
                        print('cycle length:', cycle_length)
                        print('visited_states:', visited_states)
                        print('visited_profits:', visited_profits)
                        print('price_history:', price_history)

            # Save results for this alpha-beta combination
            results_handler.save_experiment_config(game, experiment_id)
            results_handler.save_session_results(game, experiment_id)
            results_handler.save_cycle_statistics(game, experiment_id)

            if game.aprint:
                print(f"\nCompleted experiment for alpha={alpha}, beta={beta}")
                print(f"Results saved under experiment ID: {experiment_id}")

    print("\nAll experiments completed.")
    return game


def create_profit_gain_heatmap(base_dir, player_num=1, figsize=(10, 8)):
    """
    Creates a heatmap of mean profit gains for a specific player across different alpha and beta values.
    
    Parameters:
    -----------
    base_dir : str
        Base directory containing experiment results
    player_num : int
        Player number to plot (1, 2, etc.)
    figsize : tuple
        Figure size in inches
    """
    # Get all experiment directories
    exp_dirs = glob(os.path.join(base_dir, "alpha_*"))
    
    # Extract alpha and beta values from directory names
    alpha_values = []
    beta_values = []
    profit_gains = []
    
    for exp_dir in exp_dirs:
        # Extract alpha and beta from directory name
        dir_name = os.path.basename(exp_dir)
        alpha = float(dir_name.split('_')[1])
        beta = float(dir_name.split('_')[3])
        
        # Read cycle statistics
        stats_file = os.path.join(exp_dir, "cycle_statistics.csv")
        if os.path.exists(stats_file):
            df = pd.read_csv(stats_file)
            profit_gain = float(df[f'mean_profit_gain_p{player_num}'].iloc[0])
            
            alpha_values.append(alpha)
            beta_values.append(beta)
            profit_gains.append(profit_gain)
    
    # Convert to numpy arrays
    alpha_values = np.array(alpha_values)
    beta_values = np.array(beta_values)
    profit_gains = np.array(profit_gains)
    
    # Get unique values for grid
    unique_alphas = np.sort(np.unique(alpha_values))
    unique_betas = np.sort(np.unique(beta_values))
    
    # Create grid
    profit_gain_grid = np.zeros((len(unique_alphas), len(unique_betas)))
    
    # Fill grid
    for alpha, beta, gain in zip(alpha_values, beta_values, profit_gains):
        i = np.where(unique_alphas == alpha)[0][0]
        j = np.where(unique_betas == beta)[0][0]
        profit_gain_grid[i, j] = gain
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create heatmap
    im = plt.imshow(profit_gain_grid, 
                   aspect='auto',
                   origin='lower',
                   extent=[min(unique_betas), max(unique_betas), 
                          min(unique_alphas), max(unique_alphas)],
                   cmap='Reds')
    
    # Add colorbar
    plt.colorbar(im)
    
    # Set labels
    plt.xlabel(r'$\beta \times 10^5$')
    plt.ylabel(r'$\alpha$')
    
    # Add title
    plt.title(f'Mean Profit Gain - Player {player_num}')
    
    # Adjust layout
    plt.tight_layout()
    
    return plt.gcf()

def plot_all_players_heatmaps(base_dir, num_players):
    """
    Creates heatmaps for all players' profit gains.
    
    Parameters:
    -----------
    base_dir : str
        Base directory containing experiment results
    num_players : int
        Number of players in the game
    """
    fig, axes = plt.subplots(1, num_players, figsize=(6*num_players, 5))
    if num_players == 1:
        axes = [axes]
    
    for i in range(num_players):
        plt.sca(axes[i])
        create_profit_gain_heatmap(base_dir, i+1, figsize=None)
        axes[i].set_title(f'Player {i+1}')
    
    plt.tight_layout()
    return fig