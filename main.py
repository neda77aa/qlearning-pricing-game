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
import numpy as np
from input.init import model
from input.qlearning import simulate_game, run_sessions

# Init algorithm
#game = model()

# Compute equilibrium
#game_equilibrium = simulate_game(game)


####
print("Special Case: beta = 0.005 and alpha = 0.0025")
game = model(n=2, k = 10, memory = 2,alpha=0.0025, beta=0.005, num_sessions = 10)
game_equilibrium = simulate_game(game)
