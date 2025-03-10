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
from input.ConvResults import run_experiment, run_experiment_parallel
from input.ConvResults_gamma_lambda import run_experiment_gl, run_experiment_parallel_gl
import matplotlib.pyplot as plt
from input.visualization import create_comparative_heatmaps, create_single_heatmap, create_comparative_heatmaps_gl, create_single_heatmap_gl



if __name__ == '__main__':
    # Add freeze_support
    freeze_support()

    Desired_Experiment = 'gamma_lambda'

    ###########################################
    # generating alpha beta figures
    if Desired_Experiment == 'trial_test':

        game = model(n=2, k = 15, memory = 1,alpha=0.003, beta = 5e-6, demand_type = 'reference', num_sessions = 5, aprint = True, gamma = 1)
        # # #game_equilibrium = simulate_game(game)
        game_equilibrium = run_sessions(game)

    
    ###########################################
    # generating alpha beta figures
    if Desired_Experiment == 'alpha_beta':
        # Define parameter ranges to test
        alpha_values = np.linspace(0.0075, 0.075, 4)  # 10 values between 0.001 and 0.01
        beta_values = np.linspace(0.01/25000, 0.07/25000, 4)   # 10 values between 0.001 and 0.01

        experiment_base_name =  "reference_impact_loss_aversion/alpha_beta"
        num_sessions = 4
        aprint = True

        # Store experiment directories for later comparison
        experiment_dirs = {}


        for demand_type in ['noreference','reference']:
            experiment_name = experiment_base_name + "_" + demand_type

            game = model(n=2, k = 15, memory = 1,alpha=0.0075, beta=0.01/25000, num_sessions = num_sessions, aprint = aprint, demand_type = demand_type)

            # Run experiments Single core
            # game = run_experiment(game, alpha_values, beta_values, num_sessions= num_sessions, experiment_name = experiment_name, demand_type = demand_type)

            # Or specify number of processes
            game = run_experiment_parallel(game, alpha_values, beta_values, num_sessions=num_sessions, experiment_name = experiment_name, demand_type = demand_type, num_processes=4)
            # Store experiment directory
            experiment_dirs[demand_type] = os.path.join("../Results/experiments", experiment_name)

            # Generate heatmaps
            fig_profit = create_single_heatmap("../Results/experiments",  experiment_name=experiment_name, metric_name="Profit")
            fig_price_gain = create_single_heatmap("../Results/experiments", experiment_name=experiment_name, metric_name="Price Gain")
            fig_price = create_single_heatmap("../Results/experiments", experiment_name=experiment_name, metric_name="Price")

            # Create "Figures" directory
            figures_dir = os.path.join("../Results/experiments", experiment_name, "Figures")
            os.makedirs(figures_dir, exist_ok=True)

            # Save figures
            fig_profit.savefig(os.path.join(figures_dir, "profit_heatmap.png"))
            fig_price_gain.savefig(os.path.join(figures_dir, "price_gain_heatmap.png"))
            fig_price.savefig(os.path.join(figures_dir, "price_heatmap.png"))

        # # Compute differences between reference and no-reference experiments
        # Create Figures directory inside the experiment directory
        figures_dir = os.path.join("../Results/experiments", experiment_base_name, "Figures")
        os.makedirs(figures_dir, exist_ok=True)
        # Run side-by-side heatmaps for price, profit, and cycle length
        fig1 = create_comparative_heatmaps("../Results/experiments", experiment_dirs, metric_name="Price")
        fig2 = create_comparative_heatmaps("../Results/experiments", experiment_dirs, metric_name="Profit")
        fig3 = create_comparative_heatmaps("../Results/experiments", experiment_dirs, metric_name="Surplus")
        fig4 = create_comparative_heatmaps("../Results/experiments", experiment_dirs, metric_name="Cycle Length")

        fig1.savefig(os.path.join(figures_dir, "price_dual_heatmap.png"))
        fig2.savefig(os.path.join(figures_dir, "profit_dual_heatmap.png"))
        fig3.savefig(os.path.join(figures_dir, "consumer_surplus_dual_heatmap.png"))
        fig4.savefig(os.path.join(figures_dir, "cyclelength_dual_heatmap.png"))




    #################################################
    # Generate gamma lambda values

    if Desired_Experiment == 'gamma_lambda':
       # Define parameter ranges to test
        gamma_values = np.linspace(0, 3, 4)  # 10 values between 0.001 and 0.01
        lambda_values = np.linspace(0, 1, 4)   # 10 values between 0.001 and 0.01

        experiment_base_name =  "reference_impact_loss_aversion/gamma_lambda"
        num_sessions = 4
        aprint = True

        # Store experiment directories for later comparison
        experiment_dirs = {}


        for demand_type in ['noreference','reference']:
            experiment_name = experiment_base_name + "_" + demand_type

            game = model(n=2, k = 15, memory = 1, num_sessions = num_sessions, aprint = aprint, demand_type = demand_type)

            # Run experiments Single core
            # game = run_experiment_gl(game, gamma_values, lambda_values, num_sessions= num_sessions, experiment_name = experiment_name, demand_type = demand_type)

            # Or specify number of processes
            game = run_experiment_parallel_gl(game, gamma_values, lambda_values, num_sessions=num_sessions, experiment_name = experiment_name, demand_type = demand_type, num_processes=4)
            # Store experiment directory
            experiment_dirs[demand_type] = os.path.join("../Results/experiments", experiment_name)

            # Generate heatmaps
            fig_profit = create_single_heatmap_gl("../Results/experiments",  experiment_name=experiment_name, metric_name="Profit")
            fig_price_gain = create_single_heatmap_gl("../Results/experiments", experiment_name=experiment_name, metric_name="Price Gain")
            fig_price = create_single_heatmap_gl("../Results/experiments", experiment_name=experiment_name, metric_name="Price")

            # Create "Figures" directory
            figures_dir = os.path.join("../Results/experiments", experiment_name, "Figures")
            os.makedirs(figures_dir, exist_ok=True)

            # Save figures
            fig_profit.savefig(os.path.join(figures_dir, "profit_heatmap.png"))
            fig_price_gain.savefig(os.path.join(figures_dir, "price_gain_heatmap.png"))
            fig_price.savefig(os.path.join(figures_dir, "price_heatmap.png"))

        # # Compute differences between reference and no-reference experiments
        # Create Figures directory inside the experiment directory
        figures_dir = os.path.join("../Results/experiments", experiment_base_name, "Figures")
        os.makedirs(figures_dir, exist_ok=True)
        # Run side-by-side heatmaps for price, profit, and cycle length
        fig1 = create_comparative_heatmaps_gl("../Results/experiments", experiment_dirs, metric_name="Price")
        fig2 = create_comparative_heatmaps_gl("../Results/experiments", experiment_dirs, metric_name="Profit")
        fig3 = create_comparative_heatmaps_gl("../Results/experiments", experiment_dirs, metric_name="Surplus")
        fig4 = create_comparative_heatmaps_gl("../Results/experiments", experiment_dirs, metric_name="Cycle Length")

        fig1.savefig(os.path.join(figures_dir, "price_dual_heatmap.png"))
        fig2.savefig(os.path.join(figures_dir, "profit_dual_heatmap.png"))
        fig3.savefig(os.path.join(figures_dir, "consumer_surplus_dual_heatmap.png"))
        fig4.savefig(os.path.join(figures_dir, "cyclelength_dual_heatmap.png"))


    #################################################