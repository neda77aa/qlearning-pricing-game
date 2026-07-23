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


# ==============================================================================
# STEP 2: FUNCTION TO PROCESS THE CSV FILES
# ==============================================================================

def compute_expected_utility(p1_idx, p2_idx, ref_idx, A, a=2, mu=0.25, a0=0.0, gamma=0, phi=1.5):
    # Ensure inputs are arrays, even if singleton
    p1 = np.atleast_1d(A[np.array(p1_idx, dtype=int)])
    p2 = np.atleast_1d(A[np.array(p2_idx, dtype=int)])
    r = np.atleast_1d(A[np.array(ref_idx, dtype=int)])

    EU_list = []

    for i in range(len(p1)):
        if gamma == 0:
            u1 = a - p1[i]
            u2 = a - p2[i]
        else:
            u1 = a - p1[i]
            u1 += gamma * (r[i] - p1[i]) if p1[i] < r[i] else -gamma * phi * (p1[i] - r[i])

            u2 = a - p2[i]
            u2 += gamma * (r[i] - p2[i]) if p2[i] < r[i] else -gamma * phi * (p2[i] - r[i])

        # Inclusive utility from logit model
        EU = mu * np.log(np.exp(a0 / mu) + np.exp(u1 / mu) + np.exp(u2 / mu))
        EU_list.append(EU)

    return np.mean(EU_list)




def calculate_average_expected_utility(filepath, A, gamma=0, phi=1.5, a=2, mu=0.25, a0=0):
    df = pd.read_csv(filepath)
    session_utilities = []

    for _, row in df.iterrows():
        try:
            p1_idx = ast.literal_eval(row['cycle_prices_p1'])
            p2_idx = ast.literal_eval(row['cycle_prices_p2'])
            ref_idx = ast.literal_eval(row['cycle_reference_prices'])
            surplus = ast.literal_eval(row['cycle_consumer_surplus'])


            EU = compute_expected_utility(p1_idx, p2_idx, ref_idx, A, a=a, mu=mu, a0=a0, gamma=gamma, phi=phi)
            session_utilities.append(EU)
        except:
            continue  # Skip malformed rows

    return np.mean(session_utilities)



# ==============================================================================
# STEP 3: MAIN SCRIPT EXECUTION
# ==============================================================================

if __name__ == "__main__":
    import pandas as pd
    import ast
    # Define the parameters from your model class to calculate game.A
    # These match the defaults in your provided code.
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

    # === File paths ===
    # rational_world_file = '/Users/neda/Desktop/UBC/PHD/research_term_4/Results/experiments/reference_qlearning_for_2*2/gamma_only/gamma_0.0/session_summaries.csv'
    # irrational_world_file = '/Users/neda/Desktop/UBC/PHD/research_term_4/Results/experiments/reference_qlearning_for_2*2/gamma_only/gamma_1.0/session_summaries.csv'

    rational_world_file = '/Users/neda/Desktop/UBC/PHD/research_term_4/Results_sockeye/experiments/final_results/gamma_lambda_reference_True/gamma_0.05_lambda_0.5241/session_summaries.csv'
    irrational_world_file = '/Users/neda/Desktop/UBC/PHD/research_term_4/Results_sockeye/experiments/final_results/gamma_lambda_reference_True/gamma_0.9655_lambda_0.5241/session_summaries.csv'

    print("\n--- Calculating Expected Utility Welfare ---")
    welfare_rationalword_rationalcunsomer = calculate_average_expected_utility(rational_world_file, A=game_A, gamma=0)
    welfare_rationalword_irrationalcunsomer = calculate_average_expected_utility(rational_world_file, A=game_A, gamma=1, phi=1.5)
    # γ = 1 (moderate reference dependence)
    welfare_rationalword_midgamma_consumer = calculate_average_expected_utility(irrational_world_file, A=game_A, gamma=0)
    welfare_irrationalword_midgamma_consumer = calculate_average_expected_utility(irrational_world_file, A=game_A, gamma=1, phi=1.5)

    if welfare_rationalword_rationalcunsomer is not None and welfare_irrationalword_midgamma_consumer is not None:
        print(f"Average Welfare in a 'Rational' World (γ=0):     {welfare_rationalword_rationalcunsomer:.4f}")
        print(f"Average Welfare in an 'Irrational' World (γ=3):  {welfare_irrationalword_midgamma_consumer:.4f}")
        print("-" * 35)

        print("\n--- The 2x2 Payoff Matrix (Your Welfare) ---")
        matrix = f"""
        +-----------------------------+----------------------------------+------------------------------------+
        |                             |    Everyone Else is Rational     |    Everyone Else is "Irrational"   |
        |                             | (Market Welfare = {welfare_rationalword_rationalcunsomer:.4f}) | (Market Welfare = {welfare_irrationalword_midgamma_consumer:.4f})  |
        +-----------------------------+----------------------------------+------------------------------------+
        |  Your Agent is Rational     |             {welfare_rationalword_rationalcunsomer:.4f}                |            {welfare_rationalword_midgamma_consumer:.4f}               |
        |                             |               (Low)              |              (High)             |
        +-----------------------------+----------------------------------+------------------------------------+
        |  Your Agent is "Irrational" |             {welfare_rationalword_irrationalcunsomer:.4f}              |          {welfare_irrationalword_midgamma_consumer:.4f}               |
        |                             |               (Low)              |                (High)              |
        +-----------------------------+----------------------------------+------------------------------------+
        """
        print(matrix)

        # print("\nAnalysis:")
        # print("1. INDIVIDUAL INCENTIVE: Your best outcome is to be a rational 'free-rider' in an irrational world.")
        # print("2. COLLECTIVE OUTCOME: If everyone acts selfishly, all agents become rational, and everyone ends up with the low welfare outcome.")
        # print("3. PARADOXICAL BENEFIT: If everyone adopts the 'suboptimal' irrational behavior, they achieve a better collective outcome.")

    # === File paths ===
    # rational_world_file = '/Users/neda/Desktop/UBC/PHD/research_term_4/Results/experiments/reference_qlearning_for_2*2/gamma_only/gamma_0.0/session_summaries.csv'
    # irrational_world_file = '/Users/neda/Desktop/UBC/PHD/research_term_4/Results/experiments/reference_qlearning_for_2*2/gamma_only/gamma_3.0/session_summaries.csv'

    rational_world_file = '/Users/neda/Desktop/UBC/PHD/research_term_4/Results_sockeye/experiments/final_results/gamma_lambda_reference_True/gamma_0.05_lambda_0.5241/session_summaries.csv'
    irrational_world_file = '/Users/neda/Desktop/UBC/PHD/research_term_4/Results_sockeye/experiments/final_results/gamma_lambda_reference_True/gamma_3.0_lambda_0.5241/session_summaries.csv'


    print("\n--- Calculating Expected Utility Welfare ---")
    welfare_rationalword_rationalcunsomer = calculate_average_expected_utility(rational_world_file, A=game_A, gamma=0)
    welfare_rationalword_irrationalcunsomer = calculate_average_expected_utility(rational_world_file, A=game_A, gamma=3, phi=1.5)
    # γ = 1 (moderate reference dependence)
    welfare_irrationalword_rationalcunsomer = calculate_average_expected_utility(irrational_world_file, A=game_A, gamma=0)
    welfare_irrationalword_irrationalcunsomer = calculate_average_expected_utility(irrational_world_file, A=game_A, gamma=3, phi=1.5)

    if welfare_rationalword_rationalcunsomer is not None and welfare_irrationalword_irrationalcunsomer is not None:
        print(f"Average Welfare in a 'Rational' World (γ=0):     {welfare_rationalword_rationalcunsomer:.4f}")
        print(f"Average Welfare in an 'Irrational' World (γ=3):  {welfare_irrationalword_irrationalcunsomer:.4f}")
        print("-" * 35)

        print("\n--- The 2x2 Payoff Matrix (Your Welfare) ---")
        matrix = f"""
        +-----------------------------+----------------------------------+------------------------------------+
        |                             |    Everyone Else is Rational     |    Everyone Else is "Irrational"   |
        |                             | (Market Welfare = {welfare_rationalword_rationalcunsomer:.4f}) | (Market Welfare = {welfare_irrationalword_irrationalcunsomer:.4f})  |
        +-----------------------------+----------------------------------+------------------------------------+
        |  Your Agent is Rational     |             {welfare_rationalword_rationalcunsomer:.4f}                |            {welfare_irrationalword_rationalcunsomer:.4f}               |
        |                             |               (Low)              |              (High)                 |
        +-----------------------------+----------------------------------+------------------------------------+
        |  Your Agent is "Irrational" |             {welfare_rationalword_irrationalcunsomer:.4f}              |          {welfare_irrationalword_irrationalcunsomer:.4f}               |
        |                             |               (Low)              |                (High)              |
        +-----------------------------+----------------------------------+------------------------------------+
        """
        print(matrix)

        # print("\nAnalysis:")
        # print("1. INDIVIDUAL INCENTIVE: Your best outcome is to be a rational 'free-rider' in an irrational world.")
        # print("2. COLLECTIVE OUTCOME: If everyone acts selfishly, all agents become rational, and everyone ends up with the low welfare outcome.")
        # print("3. PARADOXICAL BENEFIT: If everyone adopts the 'suboptimal' irrational behavior, they achieve a better collective outcome.")