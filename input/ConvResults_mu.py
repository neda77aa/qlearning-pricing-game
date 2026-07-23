import numpy as np
import pandas as pd
import os
from datetime import datetime
from input.qlearning import simulate_game, run_sessions, detect_cycle
import matplotlib.pyplot as plt
import os
from glob import glob
import multiprocessing as mp
from functools import partial
import copy
import secrets
import random


###############################################
######## Saving Experimat 


class ExperimentSaver:
    def __init__(self, experiment_name):
        self.base_dir = "../Results/experiments"
        self.experiment_name = experiment_name
        self.experiment_dir = os.path.join(self.base_dir, experiment_name)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Add this line
        os.makedirs(self.experiment_dir, exist_ok=True)
        
    def _setup_directories(self):
        """Create necessary directories if they don't exist"""
        os.makedirs(self.experiment_dir, exist_ok=True)
            
    
    def get_run_dir_mu_only(self, mu):
        """Create directory for specific mu (μ) value."""
        run_name = f"mu_{mu}"
        run_dir = os.path.join(self.experiment_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        return run_dir


    def save_experiment_config(self, game, run_dir):
        """Save experiment configuration to CSV"""
        config = {
            'Model': 1,  # Assuming this is always 1 based on your example
            'PrintQ': 0,  # Based on your example
            'Alpha1': game.alpha,
            'Alpha2': game.alpha,
            'Beta_1': game.beta,
            'Beta_2': game.beta,
            'Delta': game.delta,
            'Gamma': game.gamma,
            'Lossaversion_aversion': game.lossaversion,
            'Lambda' : game.lambda_,
            'Pnash': game.p_minmax[0],
            'Pcoop': game.p_minmax[1],
            'Pricenash': game.compute_p_competitive_monopoly(),
            'Pricecoop': game.compute_p_competitive_monopoly(),
            'a0': game.a0,
            'a1': game.a,
            'a2': game.a,
            'c1': game.c,
            'c2': game.c,
            'mu': game.mu,
            'extend1': game.extend,
            'extend2': game.extend,
            'NashP1': game.p_nash[0],
            'CoopP1': game.p_coop[0],
            'NashProfit1': game.NashProfits[0],
            'NashProfit2': game.NashProfits[1],
            'CoopProfit1': game.CoopProfits[0],
            'CoopProfit2': game.CoopProfits[1],
            'typeQ1': 'O',  # Based on your example
            'par1Q1': 0,
            'par2Q1': 0,
            'typeQ2': 'O',
            'par1Q2': 0,
            'par2Q2': 0
        }
        
        df = pd.DataFrame([config])
        config_path = os.path.join(run_dir, "config.csv")
        df.to_csv(config_path, index=False)
        
    def save_session_results(self, game, run_dir):
        """Save aggregated session results"""
        os.makedirs(run_dir, exist_ok=True)
        
        # Save session summaries
        session_summaries = {
            'session_id': range(game.num_sessions),
            'converged': game.converged,
            'time_to_convergence': game.time_to_convergence,
            'cycle_length': np.where(game.converged, game.cycle_length, np.nan)
        }

        # Add cycle prices and profits for each player
        for i_player in range(game.n):
            player_num = i_player + 1
            prices_list = []
            profits_list = []
            reference_prices_list = []
            consumer_surplus_list = []

            # Extract prices and profits only up to cycle length for each session
            for i_session in range(game.num_sessions):
                cycle_len = game.cycle_length[i_session]
                prices = game.cycle_prices[i_player, :cycle_len, i_session]
                profits = game.cycle_profits[i_player, :cycle_len, i_session]
                consumer_surplus = game.cycle_consumer_surplus[:cycle_len, i_session]  # Extract CS


                if game.demand_type in ["reference", "misspecification"]:
                    # ↳ change this block
                    if game.common_reference:
                        ref_slice = game.cycle_reference_prices[0, :cycle_len, i_session]
                        reference_prices_list.append(','.join(f"{r:.5g}" for r in ref_slice))
                    else:
                        # one string per firm, separated by ‘;’
                        firm_strings = []
                        for f in range(game.n):
                            ref_slice = game.cycle_reference_prices[f, :cycle_len, i_session]
                            firm_strings.append(','.join(f"{r:.5g}" for r in ref_slice))
                        reference_prices_list.append(';'.join(firm_strings))
                
                # Convert arrays to strings with comma separation, formatting to 5 digits
                prices_str = ','.join([f"{p:.5g}" for p in prices])
                profits_str = ','.join([f"{p:.5g}" for p in profits])
                consumer_surplus_str = ','.join([f"{cs:.5g}" for cs in consumer_surplus])

                
                prices_list.append(prices_str)
                profits_list.append(profits_str)
                consumer_surplus_list.append(consumer_surplus_str)
            
            session_summaries[f'cycle_prices_p{player_num}'] = prices_list
            session_summaries[f'cycle_profits_p{player_num}'] = profits_list
        session_summaries[f'cycle_consumer_surplus'] = consumer_surplus_list

        # Add reference prices if reference demand is used
        if game.demand_type in ["reference", "misspecification"]:
            session_summaries[f'cycle_reference_prices'] = reference_prices_list
        
        df_summaries = pd.DataFrame(session_summaries)
        df_summaries.to_csv(os.path.join(run_dir, "session_summaries.csv"), index=False)
        
        # Save compressed arrays for detailed data
        np.savez_compressed(
            os.path.join(run_dir, "session_details.npz"),
            cycle_states=game.cycle_states,
            cycle_prices=game.cycle_prices,
            cycle_profits=game.cycle_profits,
            cycle_consumer_surplus=game.cycle_consumer_surplus,
            index_strategies=game.index_strategies,
            cycle_reference_prices=game.cycle_reference_prices  # Include reference prices in saved file
        )

    def save_cycle_statistics(self, game, run_dir):
        """Save cycle statistics across all sessions"""
        os.makedirs(run_dir, exist_ok=True)

        # Calculate mean profits only up to cycle length for each session
        mean_profits = np.zeros((game.n, game.num_sessions))
        profit_gains = np.zeros((game.n, game.num_sessions))
        price_gains = np.zeros((game.n, game.num_sessions))
        mean_prices = np.zeros((game.n, game.num_sessions))
        mean_consumer_surplus = np.zeros(game.num_sessions)

        # If using reference pricing, store reference price statistics
        if game.demand_type in ["reference", "misspecification"]:
            if game.common_reference:                        # single reference
                mean_reference_prices = np.zeros(game.num_sessions)
                std_reference_prices  = np.zeros(game.num_sessions)
            else:                                            # one per firm
                mean_reference_prices = np.zeros((game.n, game.num_sessions))
                std_reference_prices  = np.zeros((game.n, game.num_sessions))

        print(game.NashProfits[0],game.CoopProfits[0], game.p_nash[0],game.p_coop[0])        
        for i_session in range(game.num_sessions):
            cycle_len = game.cycle_length[i_session]
            # Compute mean consumer surplus
            mean_consumer_surplus[i_session] = np.mean(game.cycle_consumer_surplus[:cycle_len, i_session])
             
            for i_player in range(game.n):
                mean_profits[i_player, i_session] = np.mean(game.cycle_profits[i_player, :cycle_len, i_session])
                profit_gains[i_player, i_session] = (mean_profits[i_player, i_session] - game.NashProfits[i_player]) / (game.CoopProfits[i_player] - game.NashProfits[i_player])

                # Convert price indexes to actual price values
                actual_prices = np.asarray(game.A[np.asarray(game.cycle_prices[i_player, :cycle_len, i_session], dtype=int)])
                mean_prices[i_player, i_session] = np.mean(actual_prices)
                price_gains[i_player, i_session] = (mean_prices[i_player, i_session] - game.p_nash[i_player]) / (game.p_coop[i_player] - game.p_nash[i_player])

                if game.demand_type in ["reference", "misspecification"]:
                    if game.common_reference:
                        ref = game.cycle_reference_prices[0, :cycle_len, i_session]
                        ref = game.A[ref.astype(int)]
                        mean_reference_prices[i_session] = ref.mean()
                        std_reference_prices[i_session]  = ref.std()
                    else:
                        for f in range(game.n):
                            ref = game.cycle_reference_prices[f, :cycle_len, i_session]
                            ref = game.A[ref.astype(int)]
                            mean_reference_prices[f, i_session] = ref.mean()
                            std_reference_prices[f,  i_session] = ref.std()

        

        # Calculate statistics
        cycle_stats = {
            'mean_cycle_length': f"{np.nanmean(game.cycle_length):.5g}",
            'std_cycle_length': f"{np.nanstd(game.cycle_length):.5g}",
            'convergence_rate': f"{np.nanmean(game.converged):.5g}",
            'mean_convergence_time': f"{np.nanmean(game.time_to_convergence):.5g}",
            'convergence_rate': f"{np.nanmean(game.converged):.5g}",
            'mean_convergence_time': f"{np.nanmean(game.time_to_convergence):.5g}"
        }
        print("profitgain", profit_gains[i_player])
        # Add statistics for each player
        for i_player in range(game.n):
            player_num = i_player + 1
            cycle_stats.update({
                f'mean_profit_p{player_num}': f"{np.nanmean(mean_profits[i_player]):.5g}",
                f'std_profit_p{player_num}': f"{np.nanstd(mean_profits[i_player]):.5g}",
                f'mean_profit_gain_p{player_num}': f"{np.nanmean(profit_gains[i_player]):.5g}",
                f'std_profit_gain_p{player_num}': f"{np.nanstd(profit_gains[i_player]):.5g}",
                f'mean_price_gain_p{player_num}': f"{np.nanmean(price_gains[i_player]):.5g}",
                f'std_price_gain_p{player_num}': f"{np.nanstd(price_gains[i_player]):.5g}",
                f'mean_price_p{player_num}': f"{np.nanmean(mean_prices[i_player]):.5g}",
                f'std_price_p{player_num}': f"{np.nanstd(mean_prices[i_player]):.5g}",
            })

        cycle_stats.update({
                'mean_consumer_surplus': f"{np.nanmean(mean_consumer_surplus):.5g}",
            })

        
        # Add reference price statistics if applicable
        if game.demand_type in ["reference", "misspecification"]:
            cycle_stats.update({
                'mean_reference_price': f"{np.nanmean(mean_reference_prices):.5g}",
                'std_reference_price': f"{np.nanstd(mean_reference_prices):.5g}"
            })

        df_stats = pd.DataFrame([cycle_stats])
        df_stats.to_csv(os.path.join(run_dir, "cycle_statistics.csv"), index=False)

def save_experiment_mu_only(game, experiment_name, mu):
    """Save all experiment data for a single μ value."""
    saver = ExperimentSaver(experiment_name)
    run_dir = saver.get_run_dir_mu_only(mu)

    saver.save_experiment_config(game, run_dir)
    saver.save_session_results(game, run_dir)
    saver.save_cycle_statistics(game, run_dir)

    return run_dir



###############################################
######## Run Experiment 

###############################

def run_single_session_mu(game, mu, gamma, lambda_, lossaversion, iSession):
    """
    Run a single session with μ varying and (alpha, beta, gamma, lambda) fixed.
    """
    # Fixed values (same as your style)
    alpha_fixed = 0.15
    beta_fixed  = 0.1 / 2500

    # Update game parameters
    game.alpha = alpha_fixed
    game.beta  = beta_fixed
    game.gamma = gamma          # fixed
    game.lambda_ = lambda_      # fixed
    game.mu = mu                # VARYING μ
    game.lossaversion = lossaversion

    game.p_minmax = game.compute_p_competitive_monopoly()
    game.NashProfits, game.CoopProfits = game.compute_profits_nash_coop()
    game.p_nash, game.p_coop = game.p_minmax[0], game.p_minmax[1]
    game.PI = game.init_PI()
    game.Q  = game.init_Q()

    # Logs/arrays init (same as your other runner)
    if game.common_reference: ref_shape = (1,)
    else:                     ref_shape = (game.n,)

    game.converged = np.zeros(game.num_sessions, dtype=bool)
    game.time_to_convergence = np.zeros(game.num_sessions, dtype=float)
    game.index_last_state = np.zeros((game.n, game.memory, game.num_sessions), dtype=int)
    game.index_last_reference = np.zeros(ref_shape + (game.num_sessions,), dtype=int)
    game.cycle_length = np.zeros(game.num_sessions, dtype=int)
    game.cycle_states = np.zeros((game.num_periods, game.num_sessions), dtype=int)
    game.cycle_prices = np.zeros((game.n, game.num_periods, game.num_sessions), dtype=float)
    game.cycle_profits = np.zeros((game.n, game.num_periods, game.num_sessions), dtype=float)
    game.cycle_reference_prices = np.zeros(ref_shape + (game.num_periods, game.num_sessions), dtype=float)
    game.cycle_consumer_surplus = np.zeros((game.num_periods, game.num_sessions), dtype=float)
    game.index_strategies = np.zeros((game.n,) + game.sdim + (game.num_sessions,), dtype=int)
    game.last_observed_prices = np.zeros((game.n, game.memory), dtype=int)
    game.last_observed_reference = np.zeros(ref_shape, dtype=int)
    game.last_reference_observed_prices = np.zeros((game.n, game.reference_memory), dtype=int)
    game.last_observed_demand = np.zeros((game.n, game.reference_memory), dtype=float)

    # Deep copy & per-session init
    game_copy = copy.deepcopy(game)
    game_copy.Q = game_copy.init_Q()
    game_copy.last_observed_prices = np.zeros((game_copy.n, game_copy.memory), dtype=int)

    if game.demand_type in ["reference", "misspecification"]:
        game_copy.last_observed_reference = np.zeros(ref_shape, dtype=int)
        game_copy.last_reference_observed_prices = np.zeros((game_copy.n, game_copy.reference_memory), dtype=int)
        game_copy.last_observed_demand = np.zeros((game_copy.n, game_copy.reference_memory), dtype=float)

    # Unique seed
    seed = secrets.randbits(32)
    np.random.seed(seed)
    random.seed(seed)

    # Simulate
    game_copy, converged, t_convergence, consumer_reference_agent = simulate_game(game_copy)

    # Store per-session outputs into game_copy
    game_copy.converged[iSession] = converged
    game_copy.time_to_convergence[iSession] = t_convergence
    game_copy.index_last_state[:, :, iSession] = game_copy.last_observed_prices
    game_copy.index_strategies[..., iSession]  = game_copy.Q.argmax(axis=-1)

    if game.demand_type in ["reference", "misspecification"]:
        last_reference_price   = game_copy.last_observed_reference
        last_reference_prices  = game_copy.last_reference_observed_prices
        last_observed_demand   = game_copy.last_observed_demand
    else:
        last_reference_price = last_reference_prices = last_observed_demand = None

    # Cycle data (same pattern as your code)
    cycle_data = None
    if converged:
        if game_copy.demand_type == 'noreference':
            cycle_length, visited_states, visited_profits, price_history, _, cs_hist = detect_cycle(game_copy, iSession)
            cycle_data = {
                'cycle_length': cycle_length,
                'visited_states': visited_states,
                'visited_profits': visited_profits,
                'price_history': price_history,
                'consumer_surplus_history': cs_hist
            }
        else:
            cycle_length, visited_states, visited_profits, price_history, ref_price_hist, cs_hist = detect_cycle(game_copy, iSession, consumer_reference_agent)
            cycle_data = {
                'cycle_length': cycle_length,
                'visited_states': visited_states,
                'visited_profits': visited_profits,
                'price_history': price_history,
                'reference_price_history': ref_price_hist,
                'consumer_surplus_history': cs_hist
            }

    # Return
    if game.demand_type in ["reference", "misspecification"]:
        return {
            'session_id': iSession,
            'converged': converged,
            'time_to_convergence': t_convergence,
            'last_observed_prices': game_copy.last_observed_prices,
            'optimal_strategies': game_copy.Q.argmax(axis=-1),
            'cycle_data': cycle_data,
            'last_observed_reference': last_reference_price,
            'last_reference_prices': last_reference_prices,
            'last_observed_demand': last_observed_demand
        }
    else:
        return {
            'session_id': iSession,
            'converged': converged,
            'time_to_convergence': t_convergence,
            'last_observed_prices': game_copy.last_observed_prices,
            'optimal_strategies': game_copy.Q.argmax(axis=-1),
            'cycle_data': cycle_data
        }


def run_experiment_parallel_mu_only(
    game,
    mu_values,
    num_sessions=1000,
    experiment_name='test',
    demand_type='noreference',
    num_processes=None,
    # fixed params
    alpha_fixed=0.15,
    beta_fixed=0.1/2500,
    gamma_fixed=1.0,
    lambda_fixed=0.5,
    lossaversion_fixed=1.5
):
    """
    Run experiments sweeping μ with (alpha, beta, gamma, lambda) fixed, in parallel.
    """
    if num_processes is None:
        num_processes = max(1, mp.cpu_count() - 2)
        print('num_process', num_processes)

    print(f"Starting parallel μ sweep with {num_processes} processes for {num_sessions} sessions")

    for mu in mu_values:
        # Where results live
        run_dir = os.path.join("../Results/experiments", experiment_name, f"mu_{mu}")
        stats_file = os.path.join(run_dir, "cycle_statistics.csv")

        # Set fixed parameters on the shared `game`
        game.alpha = alpha_fixed
        game.beta  = beta_fixed
        game.mu = mu  # varying
        game.p_minmax = game.compute_p_competitive_monopoly()
        game.NashProfits, game.CoopProfits = game.compute_profits_nash_coop()
        game.p_nash, game.p_coop = game.p_minmax[0], game.p_minmax[1]
        print(game.NashProfits, game.CoopProfits, game.p_nash, game.p_coop)
        game.PI = game.init_PI()
        game.Q  = game.init_Q()
        game.num_sessions = num_sessions
        game.demand_type = demand_type

        # print(game.p_nash, game.p_coop)
        # print(game.NashProfits, game.CoopProfits)

        # If already exists, optionally patch p_nash / p_coop columns then skip
        if os.path.exists(stats_file):
            print(f"mu_{mu} (already exists in {run_dir})")
            try:
                df = pd.read_csv(stats_file)
                if 'p_nash_p1' not in df.columns:
                    for i, val in enumerate(game.p_nash):
                        df[f'p_nash_p{i+1}'] = val
                    for i, val in enumerate(game.p_coop):
                        df[f'p_coop_p{i+1}'] = val
                    df.to_csv(stats_file, index=False)
                    print(f"Added p_nash and p_coop per player to {stats_file}")
                else:
                    print("p_nash and p_coop already present.")
            except Exception as e:
                print(f"Warning reading existing stats for mu_{mu}: {e}")
            continue

        # Allocate arrays on the shared `game` (same pattern as others)
        if game.common_reference: ref_shape = (1,)
        else:                     ref_shape = (game.n,)

        game.converged = np.zeros(game.num_sessions, dtype=bool)
        game.time_to_convergence = np.zeros(game.num_sessions, dtype=float)
        game.index_last_state = np.zeros((game.n, game.memory, game.num_sessions), dtype=int)
        game.index_last_reference = np.zeros(ref_shape + (game.num_sessions,), dtype=int)
        game.cycle_length = np.zeros(game.num_sessions, dtype=int)
        game.cycle_states = np.zeros((game.num_periods, game.num_sessions), dtype=int)
        game.cycle_prices = np.zeros((game.n, game.num_periods, game.num_sessions), dtype=float)
        game.cycle_profits = np.zeros((game.n, game.num_periods, game.num_sessions), dtype=float)
        game.cycle_reference_prices = np.zeros(ref_shape + (game.num_periods, game.num_sessions), dtype=float)
        game.cycle_consumer_surplus = np.zeros((game.num_periods, game.num_sessions), dtype=float)
        game.index_strategies = np.zeros((game.n,) + game.sdim + (game.num_sessions,), dtype=int)
        game.last_observed_prices = np.zeros((game.n, game.memory), dtype=int)
        game.last_observed_reference = np.zeros(ref_shape, dtype=int)
        game.last_reference_observed_prices = np.zeros((game.n, game.reference_memory), dtype=int)
        game.last_observed_demand = np.zeros((game.n, game.reference_memory), dtype=float)

        print(f"\n μ={mu} with {num_processes} processes")

        try:
            with mp.Pool(processes=num_processes) as pool:
                session_results = []
                for iSession in range(num_sessions):
                    r = pool.apply_async(
                        run_single_session_mu,
                        args=(game, mu, gamma_fixed, lambda_fixed, lossaversion_fixed, iSession)
                    )
                    session_results.append(r)

                results = []
                for i, res in enumerate(session_results):
                    try:
                        results.append(res.get(timeout=600))
                    except Exception as e:
                        print(f"Session {i} failed or timed out: {e}")
                        continue

            # Collect back into `game`
            for result in results:
                iSession = result['session_id']
                game.converged[iSession] = result['converged']
                game.time_to_convergence[iSession] = result['time_to_convergence']
                game.index_last_state[:, :, iSession] = result['last_observed_prices']
                game.index_strategies[..., iSession]  = result['optimal_strategies']

                if game.demand_type in ["reference", "misspecification"]:
                    game.index_last_reference[:, iSession] = result['last_observed_reference']

                if result['cycle_data'] is not None:
                    cd = result['cycle_data']
                    game.cycle_length[iSession] = cd['cycle_length']
                    L = cd['cycle_length']
                    game.cycle_states[:L, iSession] = cd['visited_states']
                    game.cycle_prices[:, :L, iSession] = cd['price_history']
                    game.cycle_profits[:, :L, iSession] = cd['visited_profits']
                    game.cycle_consumer_surplus[:L, iSession] = cd['consumer_surplus_history']
                    if game.demand_type in ["reference", "misspecification"]:
                        game.cycle_reference_prices[:, :L, iSession] = cd['reference_price_history']

            # Save
            run_dir = save_experiment_mu_only(game, experiment_name, mu)
            if game.aprint:
                print(f"Completed μ = {mu}")
                print(f"Results saved in {run_dir}")

        except Exception as e:
            print(f"Error processing μ = {mu}: {str(e)}")
            import traceback; traceback.print_exc()
            continue

    print("\nAll μ experiments completed.")
    return game
