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
    game = model(n=2, k = 15, memory = 1,alpha=0.01, beta = 2e-6,demand_type = 'reference', num_sessions = 2, aprint = True)
    #game_equilibrium = simulate_game(game)
    game_equilibrium = run_sessions(game)


    # Define parameter ranges to test
    # alpha_values = np.linspace(0.0025, 0.01, 3)  # 10 values between 0.001 and 0.01
    # beta_values = np.linspace(0.005/25000, 0.05/25000, 3)   # 10 values between 0.001 and 0.01


    # experiment_name = "baseline_experiment_2,2_parallel"
    # num_sessions = 2
    # aprint = True

    # game = model(n=2, k = 15, memory = 1,alpha=0.0075, beta=0.01/25000, num_sessions = num_sessions, aprint = aprint, demand_type = 'noreference')

    # # Run experiments Single core
    # game = run_experiment(game, alpha_values, beta_values, num_sessions= num_sessions, experiment_name = experiment_name, demand_type = 'noreference')

    # Or specify number of processes
    #game = run_experiment_parallel(game, alpha_values, beta_values, num_sessions=num_sessions, experiment_name = experiment_name) #, num_processes=4)

    # Generate heatmaps
    # fig_profit = create_profit_gain_heatmap("../Results/experiments", player_num=1, experiment_name=experiment_name, metric_name="Profit Gain")
    # fig_price = create_profit_gain_heatmap("../Results/experiments", player_num=1, experiment_name=experiment_name, metric_name="Price")

    # # Create "Figures" directory
    # figures_dir = os.path.join("../Results/experiments", experiment_name, "Figures")
    # os.makedirs(figures_dir, exist_ok=True)

    # # Save figures
    # fig_profit.savefig(os.path.join(figures_dir, "profit_gain_heatmap.png"))
    # fig_price.savefig(os.path.join(figures_dir, "price_heatmap.png"))

    # # Show plots (optional)
    # plt.show()