"""
Replication of Artificial Intelligence, Algorithmic Pricing, and Collusion
    by: Calvano, Calzolari, Denicolò (2020)
    at: https://www.aeaweb.org/articles?id=10.1257/aer.20190623
Code
    author: Matteo Courthoud
    date: 07/05/2021
    git: https://github.com/matteocourthoud
    myself: https://matteocourthoud.github.io/
"""
import os
import numpy as np
from multiprocessing import freeze_support
from input.init import model
from input.qlearning import simulate_game, run_sessions
from input.ConvResults import run_experiment, create_profit_gain_heatmap, run_experiment_parallel
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Add freeze_support
    freeze_support()

    ####
    # game = model(n=2, k = 15, memory = 1,alpha=0.01, num_sessions = 5, aprint = True)
    #game_equilibrium = run_sessions(game)


    # Define parameter ranges to test
    alpha_values = np.linspace(0.0025, 0.15, 20)  # 10 values between 0.001 and 0.01
    beta_values = np.linspace(0.005/25000, 0.2/25000, 20)   # 10 values between 0.001 and 0.01


    experiment_name = "baseline_experiment_20,20_parallel"
    num_sessions = 20
    aprint = False

    game = model(n=2, k = 15, memory = 1,alpha=0.0075, beta=0.01/25000, num_sessions = num_sessions, aprint = aprint)

    # Run experiments Single core
    #game = run_experiment(game, alpha_values, beta_values, num_sessions= num_sessions, experiment_name = experiment_name)


    # Or specify number of processes
    game = run_experiment_parallel(game, alpha_values, beta_values, num_sessions=num_sessions, experiment_name = experiment_name) #, num_processes=4)

    # For a single player
    fig = create_profit_gain_heatmap("../Results/experiments", player_num=1, experiment_name = experiment_name)

    # Create Figures directory inside the experiment directory
    figures_dir = os.path.join("../Results/experiments", experiment_name, "Figures")
    os.makedirs(figures_dir, exist_ok=True)

    # Create the full path for saving the figure
    fig_path = os.path.join(figures_dir, f"player1_heatmap.png")

    # Save the figure
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()