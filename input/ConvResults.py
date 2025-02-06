import numpy as np

def compute_conv_results(game, experiment_id):
    """
    Computes statistics for one model.
    
    Parameters:
    -----------
    game : object
        The game environment containing all necessary attributes.
    experiment_id : int
        Experiment identifier.
    
    Returns:
    --------
    None (Updates game attributes directly)
    """
    # Declare variables
    num_sessions = game.num_sessions
    num_agents = game.n
    num_states = game.num_states
    num_periods = game.num_periods
    depth_state = game.memory  # Equivalent to DepthState in Fortran

    # Initialize storage arrays
    profits = np.zeros((num_sessions, num_agents))
    freq_states = np.zeros((num_sessions, num_states))
    mean_profit = np.zeros(num_agents)
    se_profit = np.zeros(num_agents)
    mean_profit_gain = np.zeros(num_agents)
    se_profit_gain = np.zeros(num_agents)

    avg_profits = np.zeros(num_sessions)
    mean_avg_profit = 0.0
    se_avg_profit = 0.0
    mean_avg_profit_gain = 0.0
    se_avg_profit_gain = 0.0

    mean_freq_states = np.zeros(num_states)
    
    print(f"Computing convergence results (average profits and frequency of prices)")
