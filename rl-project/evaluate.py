"""
Evaluation utilities for trained agents.
Run agents without exploration and compute summary statistics.
"""

import numpy as np
import gymnasium as gym
from typing import Dict, List

from agent import DQNAgent
from config import Config


def evaluate_agent(agent: DQNAgent, env_name: str, n_episodes: int = 20) -> Dict:
    """
    Run agent greedily for n_episodes and return metrics dict.
    """
    env = gym.make(env_name)
    rewards = []

    for _ in range(n_episodes):
        state, _ = env.reset()
        total = 0.0
        done = False
        while not done:
            action = agent.q_net.greedy_action(np.array(state, dtype=np.float32))
            state, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        rewards.append(total)

    env.close()
    rewards = np.array(rewards)
    return {
        "mean":   float(rewards.mean()),
        "std":    float(rewards.std()),
        "min":    float(rewards.min()),
        "max":    float(rewards.max()),
        "median": float(np.median(rewards)),
        "solved": float(rewards.mean()) >= 195.0,
    }


def compare_agents(
    dqn_history: Dict,
    ddqn_history: Dict,
) -> Dict:
    """
    Return a side-by-side comparison dict of the two training runs.
    """
    def _last_n_avg(rewards, n=50):
        arr = np.array(rewards)
        return float(arr[-n:].mean()) if len(arr) >= n else float(arr.mean())

    def _first_solve(rewards, threshold=195, window=100):
        arr = np.array(rewards)
        for i in range(window - 1, len(arr)):
            if arr[i - window + 1: i + 1].mean() >= threshold:
                return i + 1
        return None

    return {
        "dqn": {
            "last50_avg": _last_n_avg(dqn_history["rewards"]),
            "max_reward": float(max(dqn_history["rewards"])),
            "first_solve_episode": _first_solve(dqn_history["rewards"]),
            "avg_loss": float(np.mean([l for l in dqn_history["losses"] if l > 0]) or 0),
        },
        "double_dqn": {
            "last50_avg": _last_n_avg(ddqn_history["rewards"]),
            "max_reward": float(max(ddqn_history["rewards"])),
            "first_solve_episode": _first_solve(ddqn_history["rewards"]),
            "avg_loss": float(np.mean([l for l in ddqn_history["losses"] if l > 0]) or 0),
        },
    }
