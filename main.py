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

import numpy as np

from input.init import model
from input.ConvResults import run_experiment_parallel
from input.ConvResults_gamma_lambda import (
    run_experiment_parallel_gamma_only,
    run_experiment_parallel_gl,
    run_experiment_parallel_lossaversion,
)
from input.ConvResults_mu import run_experiment_parallel_mu_only
from input.visualization import (
    create_comparative_heatmaps,
    create_comparative_heatmaps_gl,
    create_comparative_heatmaps_miss,
    create_single_heatmap,
    create_single_heatmap_gamma_only,
    create_single_heatmap_gl,
    create_single_heatmap_lossaversion,
    create_single_heatmap_mu_only,
)


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _save_figures(figures_dir, figure_specs):
    """Save [(figure, filename), ...] into figures_dir."""
    _ensure_dir(figures_dir)
    for fig, filename in figure_specs:
        fig.savefig(os.path.join(figures_dir, filename))


def _prepare_results_root(output_root):
    """
    Route all experiment writes to output_root without editing helper modules.

    The experiment helpers currently write to '../Results/experiments' relative
    to the repo root. We keep that contract and redirect it with a symlink when
    needed.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)  # ensure all relative paths are deterministic

    legacy_results_dir = os.path.abspath(os.path.join(script_dir, "../Results/experiments"))
    output_root = os.path.abspath(output_root)

    _ensure_dir(output_root)
    _ensure_dir(os.path.dirname(legacy_results_dir))

    if os.path.abspath(output_root) == os.path.abspath(legacy_results_dir):
        return output_root

    if os.path.lexists(legacy_results_dir):
        if os.path.islink(legacy_results_dir):
            current_target = os.path.realpath(legacy_results_dir)
            if os.path.abspath(current_target) != os.path.abspath(output_root):
                os.unlink(legacy_results_dir)
            else:
                return output_root
        else:
            raise RuntimeError(
                f"Cannot redirect results: '{legacy_results_dir}' already exists as a real directory. "
                "Please remove/rename it or use that directory as --output-root."
            )

    os.symlink(output_root, legacy_results_dir)
    return output_root


def _parse_bool(value):
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected true/false.")


def run_trial_test(results_dir, common_reference=True):
    """Quick sanity check: print Nash and cooperative profits for selected gamma values."""
    _ = results_dir
    gamma_values = [1]
    for gamma in gamma_values:
        game = model(
            n=2,
            k=15,
            memory=1,
            lossaversion=1,
            alpha=0.1,
            beta=0.1 / 2500,
            demand_type="reference",
            num_sessions=200,
            aprint=True,
            gamma=gamma,
            common_reference=common_reference,
            ref_prediction="qlearning",
        )
        print("gamma =", gamma, "NashProfits=", game.NashProfits, "CoopProfits=", game.CoopProfits)


def run_alpha_beta(results_dir, common_reference=True):
    """Alpha-Beta sweep for learning dynamics."""
    alpha_values = np.linspace(0.0045, 0.05, 30)
    beta_values = np.linspace(0.007 / 25000, 0.1 / 25000, 30)

    base_subfolder = "alpha_beta"
    num_sessions = 200
    aprint = True

    experiment_dirs = {}

    for demand_type in ["reference"]:
        ref_prediction = "qlearning"
        experiment_name = f"{base_subfolder}/{demand_type}_common_{str(common_reference).lower()}"

        game = model(
            n=2,
            k=15,
            memory=1,
            alpha=0.0075,
            beta=0.01 / 25000,
            num_sessions=num_sessions,
            aprint=aprint,
            demand_type=demand_type,
            common_reference=common_reference,
            ref_prediction=ref_prediction,
        )

        run_experiment_parallel(
            game,
            alpha_values,
            beta_values,
            num_sessions=num_sessions,
            experiment_name=experiment_name,
            demand_type=demand_type,
            num_processes=6,
        )

        experiment_dirs[demand_type] = os.path.join(results_dir, experiment_name)

        fig_profit = create_single_heatmap(results_dir, experiment_name=experiment_name, metric_name="Profit")
        fig_price_gain = create_single_heatmap(results_dir, experiment_name=experiment_name, metric_name="Price Gain")
        fig_price = create_single_heatmap(results_dir, experiment_name=experiment_name, metric_name="Price")

        figures_dir = os.path.join(results_dir, experiment_name, "Figures")
        _save_figures(
            figures_dir,
            [
                (fig_profit, "profit_heatmap.png"),
                (fig_price_gain, "price_gain_heatmap.png"),
                (fig_price, "price_heatmap.png"),
            ],
        )

    # Comparative panel requires both keys.
    if {"reference", "noreference"}.issubset(experiment_dirs.keys()):
        figures_dir = os.path.join(results_dir, base_subfolder, "Figures")
        fig_price = create_comparative_heatmaps(results_dir, experiment_dirs, metric_name="Price")
        fig_profit = create_comparative_heatmaps(results_dir, experiment_dirs, metric_name="Profit")
        fig_surplus = create_comparative_heatmaps(results_dir, experiment_dirs, metric_name="Surplus")
        fig_cycle = create_comparative_heatmaps(results_dir, experiment_dirs, metric_name="Cycle Length")

        _save_figures(
            figures_dir,
            [
                (fig_price, "price_dual_heatmap.png"),
                (fig_profit, "profit_dual_heatmap.png"),
                (fig_surplus, "consumer_surplus_dual_heatmap.png"),
                (fig_cycle, "cyclelength_dual_heatmap.png"),
            ],
        )


def run_gamma_lambda(results_dir, common_reference=True):
    """Gamma-Lambda sweep for reference-dependence and reference updating."""
    gamma_values = np.linspace(0, 3, 30)
    lambda_values = np.linspace(0, 0.95, 30)

    base_subfolder = "gamma_lambda"
    num_sessions = 200
    aprint = True
    lossaversion = 1

    for demand_type in ["reference"]:
        ref_prediction = "exponentially_smoothing"
        experiment_name = f"{base_subfolder}/{demand_type}_common_{str(common_reference).lower()}"

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

        run_experiment_parallel_gl(
            game,
            gamma_values,
            lambda_values,
            num_sessions=num_sessions,
            experiment_name=experiment_name,
            demand_type=demand_type,
            num_processes=4,
        )

        fig_profit = create_single_heatmap_gl(results_dir, experiment_name=experiment_name, metric_name="Profit")
        fig_price_gain = create_single_heatmap_gl(results_dir, experiment_name=experiment_name, metric_name="Price Gain")
        fig_profit_gain = create_single_heatmap_gl(results_dir, experiment_name=experiment_name, metric_name="Profit Gain")
        fig_price = create_single_heatmap_gl(results_dir, experiment_name=experiment_name, metric_name="Price")
        fig_cycle = create_single_heatmap_gl(results_dir, experiment_name=experiment_name, metric_name="mean_cycle_length")

        figures_dir = os.path.join(results_dir, experiment_name, "Figures")
        _save_figures(
            figures_dir,
            [
                (fig_profit, "profit_heatmap.png"),
                (fig_price_gain, "price_gain_heatmap.png"),
                (fig_profit_gain, "profit_gain_heatmap.png"),
                (fig_price, "price_heatmap.png"),
                (fig_cycle, "cyclelength_heatmap.png"),
            ],
        )


def run_loss_aversion(results_dir, common_reference=True):
    """Loss-aversion sweep with fixed baseline settings."""
    lossaversion_values = np.linspace(1, 3, 30)

    experiment_name = "loss_aversion/reference_common_true"
    num_sessions = 200
    aprint = True
    demand_type = "reference"
    ref_prediction = "exponentially_smoothing"

    game = model(
        n=2,
        k=15,
        memory=1,
        num_sessions=num_sessions,
        aprint=aprint,
        demand_type=demand_type,
        common_reference=common_reference,
        ref_prediction=ref_prediction,
    )

    run_experiment_parallel_lossaversion(
        game,
        lossaversion_values,
        num_sessions=num_sessions,
        experiment_name=experiment_name,
        demand_type=demand_type,
        num_processes=4,
    )

    fig_profit = create_single_heatmap_lossaversion(results_dir, experiment_name=experiment_name, metric_name="Profit")
    fig_price_gain = create_single_heatmap_lossaversion(results_dir, experiment_name=experiment_name, metric_name="Price Gain")
    fig_profit_gain = create_single_heatmap_lossaversion(results_dir, experiment_name=experiment_name, metric_name="Profit Gain")
    fig_price = create_single_heatmap_lossaversion(results_dir, experiment_name=experiment_name, metric_name="Price")
    fig_cycle = create_single_heatmap_lossaversion(results_dir, experiment_name=experiment_name, metric_name="Cycle Length")

    figures_dir = os.path.join(results_dir, experiment_name, "Figures")
    _save_figures(
        figures_dir,
        [
            (fig_profit, "profit_heatmap.png"),
            (fig_price_gain, "price_gain_heatmap.png"),
            (fig_profit_gain, "profit_gain_heatmap.png"),
            (fig_price, "price_heatmap.png"),
            (fig_cycle, "cycle_length.png"),
        ],
    )


def run_gamma_only(results_dir, common_reference=True):
    """Gamma-only sweep with optional two-stage reference pretraining."""
    gamma_values = np.linspace(0, 3, 30)

    base_subfolder = "gamma_only"
    num_sessions = 200
    aprint = True
    demand_type = "reference"
    lossaversion = 1
    ref_prediction = "exponentially_smoothing"

    use_reference_pretraining = True
    t_ref = int(2e5)

    experiment_name = f"{base_subfolder}/{demand_type}_common_{str(common_reference).lower()}"

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

    run_experiment_parallel_gamma_only(
        game,
        gamma_values,
        num_sessions=num_sessions,
        experiment_name=experiment_name,
        demand_type=demand_type,
        num_processes=4,
        use_reference_pretraining=use_reference_pretraining,
        T_ref=t_ref,
    )

    fig_profit = create_single_heatmap_gamma_only(results_dir, experiment_name=experiment_name, metric_name="Profit")
    fig_price_gain = create_single_heatmap_gamma_only(results_dir, experiment_name=experiment_name, metric_name="Price Gain")
    fig_profit_gain = create_single_heatmap_gamma_only(results_dir, experiment_name=experiment_name, metric_name="Profit Gain")
    fig_price = create_single_heatmap_gamma_only(results_dir, experiment_name=experiment_name, metric_name="Price")
    fig_cycle = create_single_heatmap_gamma_only(results_dir, experiment_name=experiment_name, metric_name="Cycle Length")

    figures_dir = os.path.join(results_dir, experiment_name, "Figures")
    _save_figures(
        figures_dir,
        [
            (fig_profit, "profit_heatmap.png"),
            (fig_price_gain, "price_gain_heatmap.png"),
            (fig_profit_gain, "profit_gain_heatmap.png"),
            (fig_price, "price_heatmap.png"),
            (fig_cycle, "cycle_length.png"),
        ],
    )


def run_mu_only(results_dir, common_reference=True):
    """Mu-only sweep for demand differentiation."""
    mu_values = np.linspace(0.05, 0.5, 30)

    experiment_name = "mu_only/noreference_common_false"
    num_sessions = 200
    aprint = True
    demand_type = "noreference"
    ref_prediction = "qlearning"

    game = model(
        n=2,
        k=15,
        memory=1,
        num_sessions=num_sessions,
        aprint=aprint,
        demand_type=demand_type,
        common_reference=common_reference,
        ref_prediction=ref_prediction,
    )

    run_experiment_parallel_mu_only(
        game,
        mu_values,
        num_sessions=num_sessions,
        experiment_name=experiment_name,
        demand_type=demand_type,
        num_processes=4,
    )

    fig_profit = create_single_heatmap_mu_only(results_dir, experiment_name=experiment_name, metric_name="Profit")
    fig_price_gain = create_single_heatmap_mu_only(results_dir, experiment_name=experiment_name, metric_name="Price Gain")
    fig_profit_gain = create_single_heatmap_mu_only(results_dir, experiment_name=experiment_name, metric_name="Profit Gain")
    fig_price = create_single_heatmap_mu_only(results_dir, experiment_name=experiment_name, metric_name="Price")
    fig_cycle = create_single_heatmap_mu_only(results_dir, experiment_name=experiment_name, metric_name="Cycle Length")

    figures_dir = os.path.join(results_dir, experiment_name, "Figures")
    _save_figures(
        figures_dir,
        [
            (fig_profit, "profit_heatmap.png"),
            (fig_price_gain, "price_gain_heatmap.png"),
            (fig_profit_gain, "profit_gain_heatmap.png"),
            (fig_price, "price_heatmap.png"),
            (fig_cycle, "cycle_length.png"),
        ],
    )


def run_misspecification(results_dir, test_mode="gamma_lambda", common_reference=True):
    """Misspecification experiments in alpha-beta or gamma-lambda mode."""
    if test_mode == "alpha_beta":
        alpha_values = np.linspace(0.0045, 0.25, 30)
        beta_values = np.linspace(0.009 / 25000, 0.5 / 25000, 30)

        base_subfolder = "misspecification/alpha_beta"
        num_sessions = 200
        aprint = True

        experiment_dirs = {}

        for demand_type in ["misspecification"]:
            experiment_name = f"{base_subfolder}/{demand_type}"

            game = model(
                n=2,
                k=15,
                memory=1,
                alpha=0.0075,
                beta=0.01 / 25000,
                num_sessions=num_sessions,
                aprint=aprint,
                demand_type=demand_type,
                common_reference=common_reference,
            )

            run_experiment_parallel(
                game,
                alpha_values,
                beta_values,
                num_sessions=num_sessions,
                experiment_name=experiment_name,
                demand_type=demand_type,
                num_processes=6,
            )

            experiment_dirs[demand_type] = os.path.join(results_dir, experiment_name)

            fig_profit = create_single_heatmap(results_dir, experiment_name=experiment_name, metric_name="Profit")
            fig_price_gain = create_single_heatmap(results_dir, experiment_name=experiment_name, metric_name="Price Gain")
            fig_price = create_single_heatmap(results_dir, experiment_name=experiment_name, metric_name="Price")

            figures_dir = os.path.join(results_dir, experiment_name, "Figures")
            _save_figures(
                figures_dir,
                [
                    (fig_profit, "profit_heatmap.png"),
                    (fig_price_gain, "price_gain_heatmap.png"),
                    (fig_price, "price_heatmap.png"),
                ],
            )

        required = {"noreference", "reference", "misspecification"}
        if required.issubset(experiment_dirs.keys()):
            figures_dir = os.path.join(results_dir, base_subfolder, "Figures")
            fig_price = create_comparative_heatmaps_miss(results_dir, experiment_dirs, metric_name="Price")
            fig_profit = create_comparative_heatmaps_miss(results_dir, experiment_dirs, metric_name="Profit")
            fig_surplus = create_comparative_heatmaps_miss(results_dir, experiment_dirs, metric_name="Surplus")
            fig_cycle = create_comparative_heatmaps_miss(results_dir, experiment_dirs, metric_name="Cycle Length")

            _save_figures(
                figures_dir,
                [
                    (fig_price, "price_dual_heatmap.png"),
                    (fig_profit, "profit_dual_heatmap.png"),
                    (fig_surplus, "consumer_surplus_dual_heatmap.png"),
                    (fig_cycle, "cyclelength_dual_heatmap.png"),
                ],
            )

    elif test_mode == "gamma_lambda":
        gamma_values = np.linspace(0, 3, 30)
        lambda_values = np.linspace(0, 0.9, 30)

        base_subfolder = "misspecification/gamma_lambda"
        num_sessions = 200
        aprint = True

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

    if args.experiment == "misspecification":
        run_misspecification(results_dir, args.misspecification_test, args.common_reference)
    else:
        EXPERIMENT_RUNNERS[args.experiment](results_dir, args.common_reference)


if __name__ == "__main__":
    main()
