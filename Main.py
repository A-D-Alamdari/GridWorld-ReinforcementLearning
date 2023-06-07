"""
Name:       Amin
Surname:    Deldari Alamdari
Student ID: S033174
"""

import os.path
import time

import numpy as np
from matplotlib import pyplot as plt

import tree_search_agents
from Environment import Environment
from rl_agents.SARSA import SARSAAgent
from rl_agents.QLearning import QLearningAgent

GRID_DIR = "grid_worlds/"
FIGURES_DIR = "Figures/"

if __name__ == "__main__":
    file_name = input("Enter file name: ")

    assert os.path.exists(os.path.join(GRID_DIR, file_name)), "Invalid File"

    env = Environment(os.path.join(GRID_DIR, file_name))

    # Type your parameters
    agents = [
        SARSAAgent(env=env, seed=1, discount_rate=0.9, epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.05, alpha=1.0,
                   max_episode=1000),
        QLearningAgent(env=env, seed=1, discount_rate=0.9, epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.05, alpha=1.0,
                       max_episode=1000)
    ]

    actions = ["UP", "LEFT", "DOWN", "RIGHT"]

    for agent in agents:
        print("*" * 50)
        print()

        env.reset()

        start_time = time.time_ns()

        agent.train()

        end_time = time.time_ns()

        path, score = agent.validate()

        print("Actions:", [actions[i] for i in path])
        print("Score:", score)
        print("Elapsed Time (ms):", (end_time - start_time) * 1e-6)

        ALPHA = 1.0
        plt.plot(agent.episodes_list, label=agent.__class__.__name__)
        plt.xlabel('Episodes')
        plt.ylabel('Max TD')
        plt.legend()
        plt.savefig(
            os.path.join(FIGURES_DIR, str(ALPHA) + "_" + agent.__class__.__name__ + "_q_table_" + file_name + ".png"))
        plt.clf()

        plt.plot(agent.episodes_list, agent.rewards_list, label=agent.__class__.__name__)
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.legend()
        plt.savefig(
            os.path.join(FIGURES_DIR, str(ALPHA) + "_" + agent.__class__.__name__ + "_Reward_" + file_name + ".png"))
        plt.clf()

        print("*" * 50)
