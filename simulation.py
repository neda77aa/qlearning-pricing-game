import numpy as np
import matplotlib.pyplot as plt
from input.init import model
from input.qlearning import simulate_game

# Define parameter ranges for alpha and beta
alpha_values = np.linspace(0.0025, 0.2, 50)  # Range of alpha values
beta_values = np.linspace(0.005, 0.035, 50)  # Range of beta values

# Initialize storage for profit gains
profit_gains = np.zeros((len(alpha_values), len(beta_values)))

# Iterate over parameter grid
for i, alpha in enumerate(alpha_values):
    for j, beta in enumerate(beta_values):
        # Create a model with specific alpha and beta
        game = model(alpha=alpha, beta=beta)
        
        # Simulate the game and compute equilibrium
        simulated_game = simulate_game(game)
        
        # Compute average profit gain
        avg_profit_gain = np.mean((simulated_game.Q - game.c) / (game.a - game.c))
        profit_gains[i, j] = avg_profit_gain

# Plot the results as a heatmap
plt.figure(figsize=(10, 6))
plt.imshow(profit_gains, extent=[min(beta_values), max(beta_values),
                                 min(alpha_values), max(alpha_values)],
           aspect='auto', origin='lower', cmap='hot')

plt.colorbar(label="Average Profit Gain")
plt.xlabel(r"$\beta$")
plt.ylabel(r"$\alpha$")
plt.title("Average Profit Gain for Grid of Alpha and Beta Values")
plt.show()
