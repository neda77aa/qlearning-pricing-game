import numpy as np
import pandas as pd
from scipy.optimize import fsolve
import warnings

# Suppress runtime warnings from fsolve if it encounters difficult points
warnings.filterwarnings('ignore', 'The iteration is not making good progress')

# ==============================================================================
# STEP 1: REPLICATE THE LOGIC FROM YOUR `model` CLASS TO CALCULATE game.A
# We will use the parameters from your code as defaults.
# ==============================================================================

def demand(p, params, demand_type):
    """
    Logit demand with or without reference dependence.
    """
    a = params['a']
    mu = params['mu']
    a0 = params['a0']
    gamma = params['gamma']
    phi = params.get('lossaversion', 1.0)

    if demand_type == 'noreference':
        e = np.exp((a - p) / mu)
        d = e / (np.sum(e) + np.exp(a0 / mu))
    elif demand_type == 'reference':
        r = p  # assume reference = price (stationary point)
        price_below_r = (p < r).astype(float)
        price_above_r = (p >= r).astype(float)
        p_eff = (
            p + gamma * (p - r) * price_below_r
              + gamma * phi * (p - r) * price_above_r
        )
        e = np.exp((a - p_eff) / mu)
        d = e / (np.sum(e) + np.exp(a0 / mu))
    return d

def foc(p, params, demand_type):
    """First-order condition for competition."""
    c = params['c']
    mu = params['mu']
    gamma = params['gamma']
    phi = params.get('lossaversion', 1.0)
    d = demand(p, params, demand_type)

    if demand_type == 'noreference':
        zero = 1 - (p - c) * (1 - d) / mu
    elif demand_type == 'reference':
        r = p
        # Compute smooth price indicators
        price_above_r = 1 / (1 + np.exp(-40 * (p - r)))  # Smooth transition for p > r
        price_below_r = 1 / (1 + np.exp(40 * (p - r)))   # Smooth transition for p < r

        factor = 1 + gamma * price_below_r + gamma * phi * price_above_r
        zero = 1 - factor * (p - c) * (1 - d) / mu
    return np.squeeze(zero)

def foc_monopoly(p, params, demand_type):
    """First-order condition for monopoly."""
    c = params['c']
    mu = params['mu']
    gamma = params['gamma']
    phi = params.get('lossaversion', 1.0)
    d = demand(p, params, demand_type)

    total_contribution = np.sum((p - c) * d)
    own_contribution = (p - c) * d

    if demand_type == 'noreference':
        zero = 1 - ((p - c) * (1 - d) / mu) + ((total_contribution - own_contribution) / mu)
    elif demand_type == 'reference':
        r = p
        # Compute smooth price indicators
        price_above_r = 1 / (1 + np.exp(-40 * (p - r)))  # Smooth transition for p > r
        price_below_r = 1 / (1 + np.exp(40 * (p - r)))   # Smooth transition for p < r
        factor = 1 + gamma * price_below_r + gamma * phi * price_above_r
        zero = 1 - factor * ((p - c) * (1 - d) / mu) + factor * ((total_contribution - own_contribution) / mu)
    return np.squeeze(zero)

def compute_price_grid(params, k=15, extend=0.1):
    # Step 1: p_nash under reference dependence
    ref_params = params.copy()
    ref_params['gamma'] = 3
    ref_params['lossaversion'] = 1.5
    demand_type_ref = 'reference'
    p0 = np.ones((params['n'],)) * 3 * params['c']
    p_nash = fsolve(foc, p0, args=(ref_params, demand_type_ref))

    # Step 2: p_coop under no-reference
    noreference_params = params.copy()
    noreference_params['gamma'] = 0
    demand_type_nr = 'noreference'
    p_coop = fsolve(foc_monopoly, p0, args=(noreference_params, demand_type_nr))

    # Step 3: Compute bounds
    p_nash = np.min(p_nash)
    p_coop = np.max(p_coop)
    lower_bound = max(0, p_nash - extend * (p_coop - p_nash))
    upper_bound = p_coop + extend * (p_coop - p_nash)

    # Step 4: Return grid
    return np.linspace(lower_bound, upper_bound, k)


def calculate_average_expected_utility_dual(filepath, A, gamma_decide=0, phi_decide=1.0, gamma_welfare=0, phi_welfare=1.0, a=2, mu=0.25, a0=0):
    import ast
    df = pd.read_csv(filepath)
    session_utilities = []

    for _, row in df.iterrows():
        try:
            p1_idx = ast.literal_eval(row['cycle_prices_p1'])
            p2_idx = ast.literal_eval(row['cycle_prices_p2'])
            ref_idx = ast.literal_eval(row['cycle_reference_prices'])

            p1 = np.atleast_1d(A[np.array(p1_idx, dtype=int)])
            p2 = np.atleast_1d(A[np.array(p2_idx, dtype=int)])
            r = np.atleast_1d(A[np.array(ref_idx, dtype=int)])

            EU_list = []
            for i in range(len(p1)):
                # ----- Step 1: Decision utility (u) -----
                if gamma_decide == 0:
                    u1 = a - p1[i]
                    u2 = a - p2[i]
                else:
                    u1 = a - p1[i]
                    u1 += gamma_decide * (r[i] - p1[i]) if p1[i] < r[i] else -gamma_decide * phi_decide * (p1[i] - r[i])
                    u2 = a - p2[i]
                    u2 += gamma_decide * (r[i] - p2[i]) if p2[i] < r[i] else -gamma_decide * phi_decide * (p2[i] - r[i])

                # ----- Step 2: Welfare utility (v) -----
                if gamma_welfare == 0:
                    v1 = a - p1[i]
                    v2 = a - p2[i]
                else:
                    v1 = a - p1[i]
                    v1 += gamma_welfare * (r[i] - p1[i]) if p1[i] < r[i] else -gamma_welfare * phi_welfare * (p1[i] - r[i])
                    v2 = a - p2[i]
                    v2 += gamma_welfare * (r[i] - p2[i]) if p2[i] < r[i] else -gamma_welfare * phi_welfare * (p2[i] - r[i])

                # ----- Step 3: Experienced Utility -----
                exp_u1 = np.exp(u1 / mu)
                exp_u2 = np.exp(u2 / mu)
                exp_u0 = np.exp(a0 / mu)

                denom = exp_u0 + exp_u1 + exp_u2
                P1 = exp_u1 / denom
                P2 = exp_u2 / denom

                gamma_const = 0.5772  # Euler-Mascheroni constant

                EU = (
                    P1 * (v1 - mu * np.log(P1)) +
                    P2 * (v2 - mu * np.log(P2)) +
                    (1 - P1 - P2) * (a0 - mu * np.log(1 - P1 - P2))
                )
                EU_list.append(EU)

            session_utilities.append(np.mean(EU_list))
        except Exception as e:
            continue  # skip malformed row

    return np.mean(session_utilities)

# File paths
rational_world_file = '/Users/neda/Desktop/UBC/PHD/research_term_4/Results/experiments/reference_qlearning_for_2*2/gamma_only/gamma_0.0/session_summaries.csv'
irrational_world_file = '/Users/neda/Desktop/UBC/PHD/research_term_4/Results/experiments/reference_qlearning_for_2*2/gamma_only/gamma_3.0/session_summaries.csv'

# rational_world_file = '/Users/neda/Desktop/UBC/PHD/research_term_4/Results_sockeye/experiments/final_results/gamma_lambda_reference_True/gamma_0.05_lambda_0.5241/session_summaries.csv'
# irrational_world_file = '/Users/neda/Desktop/UBC/PHD/research_term_4/Results_sockeye/experiments/final_results/gamma_lambda_reference_True/gamma_3.0_lambda_0.5241/session_summaries.csv'

model_params = {
    'n': 2,
    'gamma': 1,
    'c': 1,
    'a': 2,
    'a0': 0,
    'mu': 0.25,
    'extend': 0.1,
    'k': 15
}

print("--- Calculating Price Grid (game.A) ---")
game_A = compute_price_grid(model_params, k=model_params['k'], extend=model_params['extend'])
print("Calculated game.A price grid:")
print(np.round(game_A, 8))
print("-" * 35)

gamma_file = 3
# Generate all 8 cases
results = {}
for gamma_choice in [0, gamma_file]:
    for gamma_welfare in [0, gamma_file]:
        for world, path in [('rational', rational_world_file), ('irrational', irrational_world_file)]:
            key = (gamma_choice, gamma_welfare, world)
            results[key] = calculate_average_expected_utility_dual(
                filepath=path,
                A=game_A,
                gamma_decide=gamma_choice,              # FIXED
                phi_decide=1.5 if gamma_choice else 0,  # FIXED
                gamma_welfare=gamma_welfare,
                phi_welfare=1.5 if gamma_welfare else 0
            )

import pandas as pd

# Create and display two 2x2 tables
table1 = pd.DataFrame({
    "Everyone Else Rational": [
        results[(0, 0, 'rational')],
        results[(gamma_file, 0, 'rational')]
    ],
    "Everyone Else Irrational": [
        results[(0, 0, 'irrational')],
        results[(gamma_file, 0, 'irrational')]
    ]
}, index=["You Act Rationally", "You Act Irrationally"])

table2 = pd.DataFrame({
    "Everyone Else Rational": [
        results[(0, gamma_file, 'rational')],
        results[(gamma_file, gamma_file, 'rational')]
    ],
    "Everyone Else Irrational": [
        results[(0, gamma_file, 'irrational')],
        results[(gamma_file, gamma_file, 'irrational')]
    ]
}, index=["You Act Rationally", "You Act Irrationally"])


print("\n--- Welfare Table: Rational Consumption Utility ---")
print(table1)

print("\n--- Welfare Table: Irrational Consumption Utility ---")
print(table2)
