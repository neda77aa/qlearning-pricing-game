"""
Robustness check: LINEAR reference-dependent demand (main analysis only).

This is a standalone entry point for the linear-demand robustness check. It does
NOT touch the paper pipeline (main.py) — it reuses the existing gamma-only
experiment runner and heatmaps, but with the LinearModel demand system
(input.init_linear) instead of the logit model.

Main analysis = the gamma sweep: profit gain / price / cycle metrics as a
function of the reference-dependence strength gamma, mirroring the paper's
"Simple Linear Demand" section (reference dependence reduces the collusive
price / makes collusion easier).

Run:
    python main_linear.py

Results are written to:  ../Results/experiments/linear_benchmark/gamma_only_linear
so nothing overwrites the paper (logit) results.
"""

import os
import argparse
import numpy as np
from multiprocessing import freeze_support

from input.init_linear import LinearModel
from input.ConvResults_gamma_lambda import run_experiment_parallel_gamma_only
from input.visualization import create_single_heatmap_gamma_only


if __name__ == '__main__':
    freeze_support()

    # CLI overrides; defaults reproduce the full large-scale sweep.
    parser = argparse.ArgumentParser(description="Linear-demand robustness (gamma sweep)")
    parser.add_argument("--n-gammas", type=int, default=30, help="number of gamma points in [0.05, 3.0]")
    parser.add_argument("--sessions", type=int, default=50, help="sessions per gamma")
    parser.add_argument("--procs", type=int, default=4, help="parallel processes")
    parser.add_argument("--lam", type=float, default=0.6,
                        help="reference-smoothing weight lambda (paper default 0.6)")
    parser.add_argument("--ref-prediction", type=str, default="exponentially_smoothing",
                        choices=["exponentially_smoothing", "qlearning"],
                        help="how the consumer reference price is formed")
    parser.add_argument("--pretrain", action="store_true",
                        help="pretrain the qlearning consumer-reference agent (T_ref steps)")
    parser.add_argument("--continuous-reference", action="store_true",
                        help="smooth the reference as a continuous float (round only for "
                             "the Q-state/PI lookup) instead of in grid-index space; removes "
                             "the index-rounding trap so r relaxes to p at a fixed point "
                             "(exponentially_smoothing path only)")
    parser.add_argument("--t-ref", type=float, default=2e5,
                        help="pretraining horizon for the qlearning reference agent")
    parser.add_argument("--name", type=str, default="linear_benchmark/gamma_only_linear",
                        help="experiment folder under ../Results/experiments")
    args = parser.parse_args()

    # ---- gamma sweep (same resolution as the paper's gamma benchmark) ----
    gamma_values = np.linspace(0.05, 3.0, args.n_gammas)
    n_firms = 2
    num_sessions = args.sessions
    num_processes = args.procs
    aprint = True

    main_dir = "../Results/experiments"
    experiment_name = args.name
    print(f"gamma_values ({len(gamma_values)}): {np.round(gamma_values,4)}")

    # ---- one fixed price grid spanning the whole gamma range --------------
    # Linear benchmarks shrink as gamma grows:
    #   p_nash = 1/(1+n(1+g))  (smallest at gamma_max),  p_coop = 1/(2+g) (largest at gamma_min)
    # Freeze a common grid (like the price_sensitivity benchmark) so prices are
    # comparable across the sweep and every gamma's Nash/coop lie inside it.
    g_lo, g_hi = float(gamma_values.min()), float(gamma_values.max())
    p_nash_min = 1.0 / (1.0 + n_firms * (1.0 + g_hi))   # lowest competitive price (gamma_max)
    p_coop_max = 1.0 / (2.0 + g_lo)                     # highest collusive price (gamma_min)
    pad = 0.1 * (p_coop_max - p_nash_min)
    grid_low = max(0.0, p_nash_min - pad)
    grid_high = p_coop_max + pad
    grid_bounds = (grid_low, grid_high)
    print(f"linear grid bounds: [{grid_low:.4f}, {grid_high:.4f}] "
          f"(p_nash@g={g_hi:.2f}={p_nash_min:.4f}, p_coop@g={g_lo:.2f}={p_coop_max:.4f})")

    # ---- build the linear model ------------------------------------------
    # demand_type defaults to 'reference' inside LinearModel so the reference
    # machinery (state, reference formation, cycle detection) is reused.
    game = LinearModel(
        n=n_firms,
        k=15,
        memory=1,
        num_sessions=num_sessions,
        aprint=aprint,
        common_reference=True,
        ref_prediction=args.ref_prediction,
        grid_bounds=grid_bounds,
        continuous_reference=args.continuous_reference,
    )

    # ---- run the gamma sweep (parallel) ----------------------------------
    game = run_experiment_parallel_gamma_only(
        game,
        gamma_values,
        num_sessions=num_sessions,
        experiment_name=experiment_name,
        demand_type='reference',          # keep the reference structure
        num_processes=num_processes,
        use_reference_pretraining=args.pretrain,   # pretrain Q reference agent?
        T_ref=int(args.t_ref),
        lambda_fixed=args.lam,            # reference-smoothing weight (ES only)
        alpha=0.15,
        beta=4e-6,                        # corrected exploration (paper runs: 4e-5)
        lossaversion_fixed=1.0,
        session_timeout=1800,
    )

    # ---- figures ----------------------------------------------------------
    figures_dir = os.path.join(main_dir, experiment_name, "Figures")
    os.makedirs(figures_dir, exist_ok=True)
    for metric, fname in [("Profit", "profit_heatmap.png"),
                          ("Price Gain", "price_gain_heatmap.png"),
                          ("Profit Gain", "profit_gain_heatmap.png"),
                          ("Price", "price_heatmap.png"),
                          ("Cycle Length", "cycle_length.png")]:
        fig = create_single_heatmap_gamma_only(main_dir, experiment_name=experiment_name, metric_name=metric)
        fig.savefig(os.path.join(figures_dir, fname))
    print(f"Linear robustness figures saved in {figures_dir}")
