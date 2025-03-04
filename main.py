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
from input.ConvResults import create_heatmap_change, compute_experiment_differences



if __name__ == '__main__':
    # Add freeze_support
    freeze_support()

    ####
    #game = model(n=2, k = 15, memory = 1,alpha=0.003, beta = 5e-6, demand_type = 'noreference', num_sessions = 5, aprint = True)
    # # #game_equilibrium = simulate_game(game)
    # game_equilibrium = run_sessions(game)


    # Define parameter ranges to test
    alpha_values = np.linspace(0.0025, 0.25, 25)  # 10 values between 0.001 and 0.01
    beta_values = np.linspace(0.006/25000, 0.5/25000, 25)   # 10 values between 0.001 and 0.01


    experiment_base_name = "reference_impact_exponentially_smoothed/march3_10_10_8"
    num_sessions = 8
    aprint = True

    # Store experiment directories for later comparison
    experiment_dirs = {}


    for demand_type in ['noreference','reference']:
        experiment_name = experiment_base_name + "_" + demand_type

        game = model(n=2, k = 15, memory = 1,alpha=0.0075, beta=0.01/25000, num_sessions = num_sessions, aprint = aprint, demand_type = demand_type)

        # Run experiments Single core
        #game = run_experiment(game, alpha_values, beta_values, num_sessions= num_sessions, experiment_name = experiment_name, demand_type = demand_type)

        # Or specify number of processes
        game = run_experiment_parallel(game, alpha_values, beta_values, num_sessions=num_sessions, experiment_name = experiment_name, demand_type = demand_type, num_processes=8)
        # Store experiment directory
        experiment_dirs[demand_type] = os.path.join("../Results/experiments", experiment_name)

        # Generate heatmaps
        fig_profit = create_profit_gain_heatmap("../Results/experiments", player_num=1, experiment_name=experiment_name, metric_name="Profit")
        fig_price = create_profit_gain_heatmap("../Results/experiments", player_num=1, experiment_name=experiment_name, metric_name="Price")

        # Create "Figures" directory
        figures_dir = os.path.join("../Results/experiments", experiment_name, "Figures")
        os.makedirs(figures_dir, exist_ok=True)

        # Save figures
        fig_profit.savefig(os.path.join(figures_dir, "profit_gain_heatmap.png"))
        fig_price.savefig(os.path.join(figures_dir, "price_heatmap.png"))

    # Compute differences between reference and no-reference experiments
    matched_alpha, matched_beta, price_change, profit_change, consumer_surplus_change = compute_experiment_differences(experiment_dirs)

    # Generate heatmaps
    figures_dir = os.path.join("../Results/experiments", experiment_base_name, "Figures_comparison")
    os.makedirs(figures_dir, exist_ok=True)

    fig1 = create_heatmap_change(matched_alpha, matched_beta, price_change, "Price Change (Ref - NoRef)")
    if fig1:  # Ensure fig1 is not None
        fig1.savefig(os.path.join(figures_dir, "price_change_heatmap.png"))

    fig2 = create_heatmap_change(matched_alpha, matched_beta, profit_change, "Profit Change (Ref - NoRef)")
    if fig2:
        fig2.savefig(os.path.join(figures_dir, "profit_change_heatmap.png"))

    fig3 = create_heatmap_change(matched_alpha, matched_beta, consumer_surplus_change, "Consumer Surplus Change (Ref - NoRef)")
    if fig3:
        fig3.savefig(os.path.join(figures_dir, "consumer_surplus_change_heatmap.png"))