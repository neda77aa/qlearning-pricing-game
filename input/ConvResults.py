import numpy as np
import pandas as pd
import os
from datetime import datetime

class ExperimentResults:
    """Handles saving and loading of experiment results"""
    
    def __init__(self, base_dir="experiments"):
        """Initialize with base directory for experiments"""
        self.base_dir = base_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = None
        self._setup_directories()
        
    def _setup_directories(self):
        """Create necessary directories if they don't exist"""
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
            
    def start_experiment(self, experiment_id):
        """Set up a new experiment directory"""
        self.experiment_dir = os.path.join(self.base_dir, f"experiment_{experiment_id}_{self.timestamp}")
        os.makedirs(self.experiment_dir, exist_ok=True)
        
    def save_experiment_config(self, config_dict):
        """Save experiment configuration"""
        config_df = pd.DataFrame([config_dict])
        config_path = os.path.join(self.experiment_dir, "experiment_config.csv")
        config_df.to_csv(config_path, index=False)
        
    def save_session_results(self, game, session_id, cycle_data):
        """
        Save results for a single session.
        
        Parameters:
        -----------
        game : object
            Game instance containing session data
        session_id : int
            Session identifier
        cycle_data : dict
            Dictionary containing cycle_length, visited_states, visited_profits
        """
        # Save summary statistics for the session
        session_summary = {
            'session_id': session_id,
            'converged': game.converged[session_id],
            'time_to_convergence': game.time_to_convergence[session_id],
            'cycle_length': cycle_data['cycle_length'],
            'mean_profit_player1': np.mean(cycle_data['visited_profits'][:, 0]),
            'mean_profit_player2': np.mean(cycle_data['visited_profits'][:, 1]),
        }
        
        summary_path = os.path.join(self.experiment_dir, "session_summaries.csv")
        pd.DataFrame([session_summary]).to_csv(summary_path, 
                                             mode='a', 
                                             header=not os.path.exists(summary_path),
                                             index=False)
        
        # Save detailed session data as compressed numpy arrays
        session_dir = os.path.join(self.experiment_dir, f"session_{session_id}")
        os.makedirs(session_dir, exist_ok=True)
        
        np.savez_compressed(
            os.path.join(session_dir, "detailed_data.npz"),
            visited_states=cycle_data['visited_states'],
            visited_profits=cycle_data['visited_profits'],
            optimal_strategies=game.index_strategies[:, session_id]
        )
        
    def save_experiment_summary(self, experiment_results):
        """
        Save summary statistics for the entire experiment
        
        Parameters:
        -----------
        experiment_results : dict
            Dictionary containing experiment-wide statistics
        """
        summary_path = os.path.join(self.experiment_dir, "experiment_summary.csv")
        pd.DataFrame([experiment_results]).to_csv(summary_path, index=False)
        
    def load_experiment_results(self, experiment_dir):
        """
        Load results from a specific experiment
        
        Returns:
        --------
        dict : Contains all experiment data
        """
        config = pd.read_csv(os.path.join(experiment_dir, "experiment_config.csv"))
        summaries = pd.read_csv(os.path.join(experiment_dir, "session_summaries.csv"))
        experiment_summary = pd.read_csv(os.path.join(experiment_dir, "experiment_summary.csv"))
        
        return {
            'config': config,
            'session_summaries': summaries,
            'experiment_summary': experiment_summary
        }

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
    results_handler = ExperimentResults()
    
    for i, alpha in enumerate(alpha_values):
        for j, beta in enumerate(beta_values):
            # Configure experiment
            experiment_id = f"alpha_{alpha}_beta_{beta}"
            results_handler.start_experiment(experiment_id)
            
            # Save configuration
            config = {
                'alpha': alpha,
                'beta': beta,
                'delta': game.delta,
                'a0': game.a0,
                'a': game.a,
                'c': game.c,
                'mu': game.mu,
                'extend': game.extend,
                'NashProfits': game.NashProfits.tolist(),
                'CoopProfits': game.CoopProfits.tolist()
            }
            results_handler.save_experiment_config(config)
            
            # Run sessions
            game.alpha = alpha
            game.beta = beta
            
            for session in range(num_sessions):

                if game.aprint:
                    print(f"\nStarting Session {iSession + 1}/{game.num_sessions}")

                game.Q = game.init_Q()  # Reset Q-values
                game.last_observed_prices = np.zeros((game.n, game.memory), dtype=int)  # Reset prices
                
                # Run single session
                game, converged, t_convergence = simulate_game(game)
                
                # Get cycle data
                cycle_data = detect_cycle(game, game.index_strategies[:, session])
                
                # Save session results
                results_handler.save_session_results(game, session, {
                    'cycle_length': cycle_data[0],
                    'visited_states': cycle_data[1],
                    'visited_profits': cycle_data[2]
                })
            
            # Calculate and save experiment summary
            summaries = pd.read_csv(os.path.join(results_handler.experiment_dir, "session_summaries.csv"))
            experiment_summary = {
                'mean_convergence_time': summaries['time_to_convergence'].mean(),
                'convergence_rate': summaries['converged'].mean(),
                'mean_cycle_length': summaries['cycle_length'].mean(),
                'mean_profit_player1': summaries['mean_profit_player1'].mean(),
                'mean_profit_player2': summaries['mean_profit_player2'].mean()
            }
            results_handler.save_experiment_summary(experiment_summary)