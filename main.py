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


####
#game = model(n=2, k = 15, memory = 1,alpha=0.0075, beta=0.01/25000, num_sessions = 5)
game = model(n=2, k = 15, memory = 1,alpha=0.01, num_sessions = 5, aprint = True)
game_equilibrium = run_sessions(game)