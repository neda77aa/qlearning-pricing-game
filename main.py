"""
Clean experiment runner for reference-dependence pricing simulations.

Usage examples:
- python main.py --experiment gamma_only
- python main.py --experiment alpha_beta
- python main.py --experiment misspecification --misspecification-test gamma_lambda
- python main.py --experiment gamma_only --output-root /path/to/my/results
"""

import argparse
import os
from multiprocessing import freeze_support
from input.init import model
from input.qlearning import simulate_game, run_sessions
from input.ConvResults import run_experiment, run_experiment_parallel
from input.ConvResults_gamma_lambda import run_experiment_gl, run_experiment_parallel_gl, run_experiment_parallel_lossaversion, run_experiment_parallel_gamma_only, run_experiment_lossaversion, run_experiment_parallel_gd
from input.ConvResults_mu import run_experiment_parallel_mu_only  # 👈 add this
import matplotlib.pyplot as plt
from input.visualization import create_comparative_heatmaps, create_single_heatmap, create_single_heatmap_gamma_only, create_comparative_heatmaps_gl, create_single_heatmap_gl, create_single_heatmap_lossaversion, create_comparative_heatmaps_miss,create_single_heatmap_mu_only, create_single_heatmap_gd



if __name__ == '__main__':
    # Add freeze_support
    freeze_support()

    Desired_Experiment = 'gamma_delta'

    ###########################################
    # generating alpha beta figures
    if Desired_Experiment == 'trial_test':
        #gamma_values = np.linspace(0, 3, 5)
        gamma_values = [1]
        for gamma in gamma_values:
            game = model(n=2, k = 15, memory = 1, lossaversion = 1,alpha=0.1, beta = 0.1 / 2500, demand_type = 'reference', num_sessions = 5, aprint = True, gamma = gamma, common_reference = False, ref_prediction = 'qlearning')
            # game_equilibrium = simulate_game(game)
            #game_equilibrium = run_sessions(game)
            print('gamma = ', game.NashProfits,  game.CoopProfits)

import numpy as np

        experiment_base_name =  "reference_qlearning/alpha_beta"
        num_sessions = 5
        aprint = True

        experiment_dirs = {}


        for demand_type in ['reference']:
            ref_prediction = 'qlearning'
            for common_reference in [True,False]:
                experiment_name = experiment_base_name + "_" + demand_type + str(common_reference)

                game = model(n=2, k = 15, memory = 1,alpha=0.0075, beta=0.01/25000, num_sessions = num_sessions, aprint = aprint, demand_type = demand_type, common_reference = common_reference, ref_prediction = ref_prediction)

                # Run experiments Single core
                # game = run_experiment(game, alpha_values, beta_values, num_sessions= num_sessions, experiment_name = experiment_name, demand_type = demand_type)

                # Or specify number of processes
                game = run_experiment_parallel(game, alpha_values, beta_values, num_sessions=num_sessions, experiment_name = experiment_name, demand_type = demand_type, num_processes=6)
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
        gamma_values = np.linspace(0, 3, 30)  
        lambda_values = np.linspace(0, 0.95, 30) 
        common_reference = False  

        experiment_base_name =  "noloss/gamma_lambda"
        num_sessions = 50
        aprint = True
        lossaversion = 1

        experiment_dirs = {}

        for demand_type in ["reference", "misspecification"]:
            experiment_name = f"{base_subfolder}/{demand_type}"

            game = model(
                n=2,
                k=15,
                memory=1,
                num_sessions=num_sessions,
                aprint=aprint,
                demand_type=demand_type,
                common_reference=common_reference,
            )

            run_experiment_parallel_gl(
                game,
                gamma_values,
                lambda_values,
                num_sessions=num_sessions,
                experiment_name=experiment_name,
                demand_type=demand_type,
                num_processes=8,
            )

            experiment_dirs[demand_type] = os.path.join(results_dir, experiment_name)

            fig_profit = create_single_heatmap_gl(results_dir, experiment_name=experiment_name, metric_name="Profit")
            fig_price_gain = create_single_heatmap_gl(results_dir, experiment_name=experiment_name, metric_name="Price Gain")
            fig_price = create_single_heatmap_gl(results_dir, experiment_name=experiment_name, metric_name="Price")
            fig_cycle = create_single_heatmap_gl(results_dir, experiment_name=experiment_name, metric_name="mean_cycle_length")
            fig_price_min = create_single_heatmap_gl(results_dir, experiment_name=experiment_name, metric_name="Price", price_plot="min")
            fig_price_max = create_single_heatmap_gl(results_dir, experiment_name=experiment_name, metric_name="Price", price_plot="max")

            figures_dir = os.path.join(results_dir, experiment_name, "Figures")
            _save_figures(
                figures_dir,
                [
                    (fig_profit, "profit_heatmap.png"),
                    (fig_price_gain, "price_gain_heatmap.png"),
                    (fig_price, "price_heatmap.png"),
                    (fig_cycle, "cyclelength_heatmap.png"),
                    (fig_price_min, "price_min_heatmap.png"),
                    (fig_price_max, "price_max_heatmap.png"),
                ],
            )

        figures_dir = os.path.join(results_dir, base_subfolder, "Figures")
        fig_price = create_comparative_heatmaps_gl(results_dir, experiment_dirs, metric_name="Price")
        fig_profit = create_comparative_heatmaps_gl(results_dir, experiment_dirs, metric_name="Profit")
        fig_surplus = create_comparative_heatmaps_gl(results_dir, experiment_dirs, metric_name="Surplus")
        fig_cycle = create_comparative_heatmaps_gl(results_dir, experiment_dirs, metric_name="Cycle Length")
        fig_price_gain = create_comparative_heatmaps_gl(results_dir, experiment_dirs, metric_name="Price Gain")

        _save_figures(
            figures_dir,
            [
                (fig_price, "price_dual_heatmap.png"),
                (fig_profit, "profit_dual_heatmap.png"),
                (fig_surplus, "consumer_surplus_dual_heatmap.png"),
                (fig_cycle, "cyclelength_dual_heatmap.png"),
                (fig_price_gain, "price_gain_dual_heatmap.png"),
            ],
        )

    else:
        raise ValueError(f"Unknown misspecification test mode: {test_mode}")


EXPERIMENT_RUNNERS = {
    "trial_test": run_trial_test,
    "alpha_beta": run_alpha_beta,
    "gamma_lambda": run_gamma_lambda,
    "loss_aversion": run_loss_aversion,
    "gamma_only": run_gamma_only,
    "mu_only": run_mu_only,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run paper experiments with clean entrypoints.")
    parser.add_argument(
        "--experiment",
        default="gamma_only",
        choices=[*EXPERIMENT_RUNNERS.keys(), "misspecification"],
        help="Experiment block to run.",
    )
    parser.add_argument(
        "--misspecification-test",
        default="gamma_lambda",
        choices=["alpha_beta", "gamma_lambda"],
        help="Sub-mode used when --experiment misspecification.",
    )
    parser.add_argument(
        "--output-root",
        default="../Results/experiments",
        help="Main output folder for all experiments.",
    )
    parser.add_argument(
        "--common-reference",
        type=_parse_bool,
        default=True,
        help="Use common reference price (true/false). Default: true.",
    )
    return parser.parse_args()


def main():
    freeze_support()
    args = parse_args()
    results_dir = _prepare_results_root(args.output_root)

        for demand_type in ["reference",
                            #'misspecification'
                            ]:
            ref_prediction = 'exponentially_smoothing'
            for common_reference in [False]:
                experiment_name = experiment_base_name + "_" + demand_type +"_" + str(common_reference)

                game = model(n=2, k = 15, memory = 1, lossaversion = lossaversion, num_sessions = num_sessions, aprint = aprint, demand_type = demand_type, common_reference = common_reference, ref_prediction = ref_prediction)

                # Run experiments Single core
                # game = run_experiment_gl(game, gamma_values, lambda_values, num_sessions= num_sessions, experiment_name = experiment_name, demand_type = demand_type)

                # Or specify number of processes
                game = run_experiment_parallel_gl(game, gamma_values, lambda_values, num_sessions=num_sessions, experiment_name = experiment_name, demand_type = demand_type, num_processes=4)
                # Store experiment directory
                experiment_dirs[demand_type] = os.path.join("../Results/experiments", experiment_name)

                # Generate heatmaps
                fig_profit = create_single_heatmap_gl("../Results/experiments",  experiment_name=experiment_name, metric_name="Profit")
                fig_price_gain = create_single_heatmap_gl("../Results/experiments", experiment_name=experiment_name, metric_name="Price Gain")
                fig_profit_gain = create_single_heatmap_gl("../Results/experiments", experiment_name=experiment_name, metric_name="Profit Gain")
                fig_price = create_single_heatmap_gl("../Results/experiments", experiment_name=experiment_name, metric_name="Price")
                fig_cycle = create_single_heatmap_gl("../Results/experiments", experiment_name=experiment_name, metric_name="mean_cycle_length")
                # Create "Figures" directory
                figures_dir = os.path.join("../Results/experiments", experiment_name, "Figures")
                os.makedirs(figures_dir, exist_ok=True)

                # Save figures
                fig_profit.savefig(os.path.join(figures_dir, "profit_heatmap.png"))
                fig_price_gain.savefig(os.path.join(figures_dir, "price_gain_heatmap.png"))
                fig_price_gain.savefig(os.path.join(figures_dir, "profit_gain_heatmap.png"))
                fig_price.savefig(os.path.join(figures_dir, "price_heatmap.png"))
                fig_cycle.savefig(os.path.join(figures_dir, "cyclelength_heatmap.png"))


    #################################################
    # Generate gamma x delta values (discount factor on the y-axis)
    #
    # x-axis = 10 gamma values, y-axis = 10 delta (discount factor) values from
    # 0.1 to 0.95 (default delta is 0.95). Color intensity = average price /
    # profit, same style as the gamma_lambda figures. lambda is held fixed.
    if Desired_Experiment == 'gamma_delta':
        # Define parameter ranges to test
        gamma_values = np.linspace(0.05, 3.0, 30)   # x-axis
        delta_values = np.linspace(0.1, 0.95, 30)   # y-axis (discount factor)
        lambda_fixed = 0.5                          # model default, held fixed

        experiment_base_name = "gamma_delta/gamma_delta"
        num_sessions = 50   # lower this (e.g. 4) for a quick smoke test
        aprint = True
        lossaversion = 1
        demand_type = 'reference'
        common_reference = True
        ref_prediction = 'exponentially_smoothing'
        # Continuous-reference robustness fix: smooth the reference in price
        # units and index it only at the PI/profit lookup (matches the linear fix).
        continuous_reference = True

        experiment_name = experiment_base_name + "_" + demand_type + "_" + str(common_reference) + "_contref"

        game = model(
            n=2, k=15, memory=1,
            lossaversion=lossaversion,
            num_sessions=num_sessions,
            aprint=aprint,
            demand_type=demand_type,
            common_reference=common_reference,
            ref_prediction=ref_prediction,
            continuous_reference=continuous_reference,
        )

        # Run the gamma x delta sweep in parallel
        game = run_experiment_parallel_gd(
            game,
            gamma_values,
            delta_values,
            lambda_fixed=lambda_fixed,
            num_sessions=num_sessions,
            experiment_name=experiment_name,
            demand_type=demand_type,
            num_processes=4,
        )

        main_dir = "../Results/experiments"
        # Generate heatmaps (Price and Profit are the requested ones)
        fig_profit = create_single_heatmap_gd(main_dir, experiment_name=experiment_name, metric_name="Profit")
        fig_price = create_single_heatmap_gd(main_dir, experiment_name=experiment_name, metric_name="Price")
        fig_price_gain = create_single_heatmap_gd(main_dir, experiment_name=experiment_name, metric_name="Price Gain")
        fig_profit_gain = create_single_heatmap_gd(main_dir, experiment_name=experiment_name, metric_name="Profit Gain")
        fig_cycle = create_single_heatmap_gd(main_dir, experiment_name=experiment_name, metric_name="mean_cycle_length")

        # Create "Figures" directory
        figures_dir = os.path.join(main_dir, experiment_name, "Figures")
        os.makedirs(figures_dir, exist_ok=True)

        # Save figures
        fig_profit.savefig(os.path.join(figures_dir, "profit_heatmap.png"))
        fig_price.savefig(os.path.join(figures_dir, "price_heatmap.png"))
        fig_price_gain.savefig(os.path.join(figures_dir, "price_gain_heatmap.png"))
        fig_profit_gain.savefig(os.path.join(figures_dir, "profit_gain_heatmap.png"))
        fig_cycle.savefig(os.path.join(figures_dir, "cyclelength_heatmap.png"))
        print(f"gamma_delta figures saved in {figures_dir}")


    #################################################
    # generating lossaversion figures
    if Desired_Experiment == 'loss_aversion':
       # Define parameter ranges to test
        lossaversion_values = np.linspace(1, 3, 20)  
        experiment_base_name =  "fixing_qlearning/gamma_only"
        experiment_name =  "loss_extreme/lossaversion_reverse"
        num_sessions = 50
        aprint = True
        demand_type = 'reference'
        common_reference = True
        ref_prediction = 'exponentially_smoothing'

        # Store experiment directories for later comparison
        experiment_dirs = {}

        game = model(n=2, k = 15, memory = 1, num_sessions = num_sessions, aprint = aprint, demand_type = 'reference', common_reference = common_reference, ref_prediction = ref_prediction)

        # Run experiments Single core
        # game = run_experiment_lossaversion(game, lossaversion_values, num_sessions= num_sessions, experiment_name = experiment_name, demand_type = demand_type)

        # Or specify number of processes
        game = run_experiment_parallel_lossaversion(game, lossaversion_values, num_sessions=num_sessions, experiment_name = experiment_name, demand_type = demand_type, num_processes=4)

        # Generate heatmaps
        fig_profit = create_single_heatmap_lossaversion("../Results/experiments",  experiment_name=experiment_name, metric_name="Profit")
        fig_price_gain = create_single_heatmap_lossaversion("../Results/experiments", experiment_name=experiment_name, metric_name="Price Gain")
        fig_profit_gain = create_single_heatmap_lossaversion("../Results/experiments", experiment_name=experiment_name, metric_name="Profit Gain")
        fig_price = create_single_heatmap_lossaversion("../Results/experiments", experiment_name=experiment_name, metric_name="Price")
        fig_cycle = create_single_heatmap_lossaversion("../Results/experiments", experiment_name=experiment_name, metric_name="Cycle Length")
        #fig_foc = create_single_heatmap_lossaversion("../Results/experiments", experiment_name=experiment_name, metric_name="FOC")
        # Create "Figures" directory
        figures_dir = os.path.join("../Results/experiments", experiment_name, "Figures")
        os.makedirs(figures_dir, exist_ok=True)

        # Save figures
        fig_profit.savefig(os.path.join(figures_dir, "profit_heatmap.png"))
        fig_price_gain.savefig(os.path.join(figures_dir, "price_gain_heatmap.png"))
        fig_profit_gain.savefig(os.path.join(figures_dir, "profit_gain_heatmap.png"))
        fig_price.savefig(os.path.join(figures_dir, "price_heatmap.png"))
        fig_cycle.savefig(os.path.join(figures_dir, "cycle_length.png"))
        #fig_foc.savefig(os.path.join(figures_dir, "nash_coop.png"))

    #################################################
    # generating gamma-only figures (with optional Q-reference pretraining)
    if Desired_Experiment == 'gamma_only':
        # Define parameter ranges to test
        gamma_values = np.linspace(0, 3, 4)

        experiment_base_name = "2*2_2/gamma_only"
        num_sessions = 100
        aprint = True
        demand_type = 'reference'
        common_reference = True
        lossaversion = 1

        # Toggle this flag if you want to use the two-stage protocol where the
        # consumer reference Q-learner is pretrained and then frozen while
        # firms learn. When False, the original joint-learning behaviour is
        # used.
        use_reference_pretraining = True
        T_ref = int(2e5)  # length of reference pretraining (only used if flag is True)

        # Store experiment directories for later comparison
        experiment_dirs = {}
        ref_prediction = 'exponentially_smoothing'

        for common_reference in [True]:

            experiment_name = experiment_base_name + "_" + demand_type + str(common_reference)
            game = model(
                n=2,
                k=15,
                memory=1,
                lossaversion=lossaversion,
                num_sessions=num_sessions,
                aprint=aprint,
                demand_type=demand_type,
                common_reference=common_reference,
                ref_prediction=ref_prediction,
            )

            # Run experiments (parallel)
            game = run_experiment_parallel_gamma_only(
                game,
                gamma_values,
                num_sessions=num_sessions,
                experiment_name=experiment_name,
                demand_type=demand_type,
                num_processes=4,
                use_reference_pretraining=use_reference_pretraining,
                T_ref=T_ref,
            )

            main_dir = "../Results/experiments"
            # Generate heatmaps
            fig_profit = create_single_heatmap_gamma_only(main_dir, experiment_name=experiment_name, metric_name="Profit")
            fig_price_gain = create_single_heatmap_gamma_only(main_dir, experiment_name=experiment_name, metric_name="Price Gain")
            fig_profit_gain = create_single_heatmap_gamma_only(main_dir, experiment_name=experiment_name, metric_name="Profit Gain")
            fig_price = create_single_heatmap_gamma_only(main_dir, experiment_name=experiment_name, metric_name="Price")
            fig_cycle = create_single_heatmap_gamma_only(main_dir, experiment_name=experiment_name, metric_name="Cycle Length")

            # Create "Figures" directory
            figures_dir = os.path.join(main_dir, experiment_name, "Figures")
            os.makedirs(figures_dir, exist_ok=True)

            # Save figures
            fig_profit.savefig(os.path.join(figures_dir, "profit_heatmap.png"))
            fig_price_gain.savefig(os.path.join(figures_dir, "price_gain_heatmap.png"))
            fig_profit_gain.savefig(os.path.join(figures_dir, "profit_gain_heatmap.png"))
            fig_price.savefig(os.path.join(figures_dir, "price_heatmap.png"))
            fig_cycle.savefig(os.path.join(figures_dir, "cycle_length.png"))


    #################################################
    # Reviewer-2 benchmark: pure price sensitivity  u = a - (1+gamma)*p
    # NO reference term and NO reference dimension in the state.
    #
    # The REFERENCE curve is REUSED from the paper run (not re-run) so its
    # converged prices are byte-identical to the paper and the reviewer cannot
    # flag a change. Only the price_sensitivity benchmark is simulated here.
    if Desired_Experiment == 'price_sensitivity':
        # Match the paper reference sweep exactly:
        gamma_values = np.linspace(0.05, 3.0, 30)   # same 30 gamma points as the paper
        num_sessions = 50                            # same sessions as the paper
        aprint = True
        lossaversion = 1
        main_dir = "../Results/experiments"

        # Existing paper reference folder (reused as-is for the overlay):
        paper_reference_dir = "/Users/neda/Desktop/UBC/PHD/research_term_4/paper_results/benchmark_figure/gamma_nloss_only_reference_True"

        # ---- price_sensitivity grid: paper ceiling, floor lowered to cost ----
        # Reference keeps the paper grid (we reuse the paper folder). The
        # benchmark needs a lower floor because its Nash/Coop fall to ~cost at
        # high gamma. The ceiling uses the same construction as the paper grid
        # (no-reference monopoly + 10% padding); only the floor is extended down.
        _tmp_lo = model(n=2, k=15, memory=1, demand_type='reference', common_reference=True, gamma=2.0, aprint=False)
        _tmp_hi = model(n=2, k=15, memory=1, demand_type='reference', common_reference=True, gamma=1.0, aprint=False)
        grid_low  = float(np.ravel(_tmp_lo.A).min())   # ~1.0622  (reaches cost)
        grid_high = float(np.ravel(_tmp_hi.A).max())   # ~1.9957  (paper ceiling family)
        ps_grid_bounds = (grid_low, grid_high)
        print(f"price_sensitivity grid bounds: [{grid_low:.4f}, {grid_high:.4f}]")

        experiment_name = 'price_sensitivity_benchmark/gamma_only_price_sensitivity'

        game = model(
            n=2, k=15, memory=1,
            lossaversion=lossaversion,
            num_sessions=num_sessions,
            aprint=aprint,
            demand_type='price_sensitivity',
            common_reference=False,
            grid_bounds=ps_grid_bounds,
        )

        game = run_experiment_parallel_gamma_only(
            game,
            gamma_values,
            num_sessions=num_sessions,
            experiment_name=experiment_name,
            demand_type='price_sensitivity',
            num_processes=4,
            use_reference_pretraining=False,
            T_ref=int(2e5),
        )

        # Single-curve figures for the benchmark
        figures_dir = os.path.join(main_dir, experiment_name, "Figures")
        os.makedirs(figures_dir, exist_ok=True)
        for metric, fname in [("Profit", "profit_heatmap.png"),
                              ("Price Gain", "price_gain_heatmap.png"),
                              ("Profit Gain", "profit_gain_heatmap.png"),
                              ("Price", "price_heatmap.png"),
                              ("Cycle Length", "cycle_length.png")]:
            fig = create_single_heatmap_gamma_only(main_dir, experiment_name=experiment_name, metric_name=metric)
            fig.savefig(os.path.join(figures_dir, fname))

        # ---- Overlay: paper reference (reused) vs price_sensitivity ----------
        # overlay_gamma_only joins main_dir with each value; an ABSOLUTE path
        # (the paper folder) is used as-is by os.path.join.
        from input.visualization import overlay_gamma_only
        overlay_exps = {
            'Reference (paper)':             paper_reference_dir,   # absolute -> used as-is
            'Price sensitivity (benchmark)': experiment_name,
        }
        overlay_dir = os.path.join(main_dir, "price_sensitivity_benchmark", "Overlay")
        os.makedirs(overlay_dir, exist_ok=True)
        for metric in ["Price", "Profit", "Price Gain", "Profit Gain", "Cycle Length"]:
            figo = overlay_gamma_only(main_dir, overlay_exps, metric_name=metric)
            figo.savefig(os.path.join(overlay_dir, f"overlay_{metric.replace(' ', '_')}.png"), dpi=150)
        print(f"Overlay figures saved in {overlay_dir}")


    #################################################
    # generating lossaversion figures
    if Desired_Experiment == 'mu_only':
        # Define μ values to test
        mu_values = np.linspace(0.05, 0.5, 10)

        experiment_name = "test_mu/mu_only_no"
        num_sessions = 10
        aprint = True
        demand_type = 'noreference'
        common_reference = False
        ref_prediction = 'qlearning'

        # Toggle this flag to decide whether to use the optional
        # two-stage protocol where the consumer reference Q-learner is
        # pretrained and then frozen while firms learn. When set to False,
        # the original joint-learning behaviour is used.
        use_reference_pretraining = True
        # Length of the pretraining phase for the consumer reference agent
        # (in Q-learning iterations). You can lower this for quick tests.
        T_ref = int(2e5)

        # Initialize model
        game = model(
            n=2, k=15, memory=1,
            num_sessions=num_sessions,
            aprint=aprint,
            demand_type=demand_type,
            common_reference=common_reference,
            ref_prediction=ref_prediction
        )

        # Run μ experiments in parallel
        game = run_experiment_parallel_mu_only(
            game,
            mu_values,
            num_sessions=num_sessions,
            experiment_name=experiment_name,
            demand_type=demand_type,
            num_processes=4
        )

        main_dir = "../Results/experiments"
        # Generate heatmaps
        fig_profit = create_single_heatmap_mu_only(main_dir,  experiment_name=experiment_name, metric_name="Profit")
        fig_price_gain = create_single_heatmap_mu_only(main_dir, experiment_name=experiment_name, metric_name="Price Gain")
        fig_profit_gain = create_single_heatmap_mu_only(main_dir, experiment_name=experiment_name, metric_name="Profit Gain")
        fig_price = create_single_heatmap_mu_only(main_dir, experiment_name=experiment_name, metric_name="Price")
        fig_cycle = create_single_heatmap_mu_only(main_dir, experiment_name=experiment_name, metric_name="Cycle Length")
        #fig_foc = create_single_heatmap_mu_only(main_dir, experiment_name=experiment_name, metric_name="FOC")
        # Create "Figures" directory
        figures_dir = os.path.join(main_dir, experiment_name, "Figures")
        os.makedirs(figures_dir, exist_ok=True)

        # Save figures
        fig_profit.savefig(os.path.join(figures_dir, "profit_heatmap.png"))
        fig_price_gain.savefig(os.path.join(figures_dir, "price_gain_heatmap.png"))
        fig_profit_gain.savefig(os.path.join(figures_dir, "profit_gain_heatmap.png"))
        fig_price.savefig(os.path.join(figures_dir, "price_heatmap.png"))
        fig_cycle.savefig(os.path.join(figures_dir, "cycle_length.png"))
        #fig_foc.savefig(os.path.join(figures_dir, "nash_coop.png"))


        #################################################
    # generating lossaversion figures
    if Desired_Experiment == 'misspecification':
        test = 'gamma_lambda'

        if test == 'alpha_beta':

            # Define parameter ranges to test
            alpha_values = np.linspace(0.0045, 0.25, 40)  # 10 values between 0.001 and 0.01
            beta_values = np.linspace(0.009/25000, 0.5/25000, 40)   # 10 values between 0.001 and 0.01

            experiment_base_name =  "reference_impact_misspecification/alpha_beta"
            num_sessions = 16
            aprint = True

            # Store experiment directories for later comparison
            experiment_dirs = {}


            for demand_type in ['misspecification']:  ##['noreference', 'reference', 'misspecification']:
                experiment_name = experiment_base_name + "_" + demand_type

                game = model(n=2, k = 15, memory = 1,alpha=0.0075, beta=0.01/25000, num_sessions = num_sessions, aprint = aprint, demand_type = demand_type)

                # Run experiments Single core
                game = run_experiment_parallel_gamma_only(
                    game,
                    gamma_values,
                    num_sessions=num_sessions,
                    experiment_name=experiment_name,
                    demand_type=demand_type,
                    num_processes=4,
                    use_reference_pretraining=use_reference_pretraining,
                    T_ref=T_ref,
                )

                # Or specify number of processes
                game = run_experiment_parallel(game, alpha_values, beta_values, num_sessions=num_sessions, experiment_name = experiment_name, demand_type = demand_type, num_processes=6)
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
            fig1 = create_comparative_heatmaps_miss("../Results/experiments", experiment_dirs, metric_name="Price")
            fig2 = create_comparative_heatmaps_miss("../Results/experiments", experiment_dirs, metric_name="Profit")
            fig3 = create_comparative_heatmaps_miss("../Results/experiments", experiment_dirs, metric_name="Surplus")
            fig4 = create_comparative_heatmaps_miss("../Results/experiments", experiment_dirs, metric_name="Cycle Length")

            fig1.savefig(os.path.join(figures_dir, "price_dual_heatmap.png"))
            fig2.savefig(os.path.join(figures_dir, "profit_dual_heatmap.png"))
            fig3.savefig(os.path.join(figures_dir, "consumer_surplus_dual_heatmap.png"))
            fig4.savefig(os.path.join(figures_dir, "cyclelength_dual_heatmap.png"))


        if test == 'gamma_lambda':
            # Define parameter ranges to test
            gamma_values = np.linspace(0, 3, 25)  
            lambda_values = np.linspace(0, 0.9, 5)
            #lambda_values = np.array([0, 0.0333334, 0.0666667])
            experiment_base_name =  "reference_impact_misspecification_high_num_sessions/gamma_lambda"
            num_sessions = 10
            aprint = True

            # Store experiment directories for later comparison
            experiment_dirs = {}


            for demand_type in ['reference','misspecification']:
                experiment_name = experiment_base_name + "_" + demand_type

                game = model(n=2, k = 15, memory = 1, num_sessions = num_sessions, aprint = aprint, demand_type = demand_type)

                # Run experiments Single core
                # game = run_experiment_gl(game, gamma_values, lambda_values, num_sessions= num_sessions, experiment_name = experiment_name, demand_type = demand_type)

                # Or specify number of processes
                game = run_experiment_parallel_gl(game, gamma_values, lambda_values, num_sessions=num_sessions, experiment_name = experiment_name, demand_type = demand_type, num_processes=8)
                # Store experiment directory
                experiment_dirs[demand_type] = os.path.join("../Results/experiments", experiment_name)

                # Generate heatmaps
                fig_profit = create_single_heatmap_gl("../Results/experiments",  experiment_name=experiment_name, metric_name="Profit")
                fig_price_gain = create_single_heatmap_gl("../Results/experiments", experiment_name=experiment_name, metric_name="Price Gain")
                fig_price = create_single_heatmap_gl("../Results/experiments", experiment_name=experiment_name, metric_name="Price")
                fig_cycle = create_single_heatmap_gl("../Results/experiments", experiment_name=experiment_name, metric_name="mean_cycle_length")
                fig_price_min = create_single_heatmap_gl("../Results/experiments", experiment_name=experiment_name, metric_name="Price", price_plot = 'min')
                fig_price_max = create_single_heatmap_gl("../Results/experiments", experiment_name=experiment_name, metric_name="Price", price_plot = 'max')
                # Create "Figures" directory
                figures_dir = os.path.join("../Results/experiments", experiment_name, "Figures")
                os.makedirs(figures_dir, exist_ok=True)

                # Save figures
                fig_profit.savefig(os.path.join(figures_dir, "profit_heatmap.png"))
                fig_price_gain.savefig(os.path.join(figures_dir, "price_gain_heatmap.png"))
                fig_price.savefig(os.path.join(figures_dir, "price_heatmap.png"))
                fig_cycle.savefig(os.path.join(figures_dir, "cyclelength_heatmap.png"))
                fig_price_min.savefig(os.path.join(figures_dir, "price_min_heatmap.png"))
                fig_price_max.savefig(os.path.join(figures_dir, "price_max_heatmap.png"))
        
        
            figures_dir = os.path.join("../Results/experiments", experiment_base_name, "Figures")
            os.makedirs(figures_dir, exist_ok=True)
            # Run side-by-side heatmaps for price, profit, and cycle length
            fig1 = create_comparative_heatmaps_gl("../Results/experiments", experiment_dirs, metric_name="Price")
            fig2 = create_comparative_heatmaps_gl("../Results/experiments", experiment_dirs, metric_name="Profit")
            fig3 = create_comparative_heatmaps_gl("../Results/experiments", experiment_dirs, metric_name="Surplus")
            fig4 = create_comparative_heatmaps_gl("../Results/experiments", experiment_dirs, metric_name="Cycle Length")
            fig5 = create_comparative_heatmaps_gl("../Results/experiments", experiment_dirs, metric_name="Price Gain")

            fig1.savefig(os.path.join(figures_dir, "price_dual_heatmap.png"))
            fig2.savefig(os.path.join(figures_dir, "profit_dual_heatmap.png"))
            fig3.savefig(os.path.join(figures_dir, "consumer_surplus_dual_heatmap.png"))
            fig4.savefig(os.path.join(figures_dir, "cyclelength_dual_heatmap.png"))
            fig5.savefig(os.path.join(figures_dir, "price_gain_dual_heatmap.png"))
