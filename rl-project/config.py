"""
Hyperparameter configuration for DQN and Double DQN agents.
Modify these to run your own experiments.
"""

from dataclasses import dataclass


@dataclass
class Config:
    # Environment
    env_name: str = "CartPole-v1"
    seed: int = 42

    # Training
    episodes: int = 500
    max_steps: int = 500

    # Agent
    gamma: float = 0.99           # Discount factor
    lr: float = 1e-3              # Learning rate
    epsilon_start: float = 1.0    # Initial exploration rate
    epsilon_end: float = 0.01     # Minimum exploration rate
    epsilon_decay: float = 0.995  # Multiplicative decay per episode

    # Replay Buffer
    buffer_capacity: int = 10_000
    batch_size: int = 64
    min_buffer_size: int = 64    # Start training after this many transitions

    # Target Network
    target_update_freq: int = 10  # Episodes between target network syncs

    # Neural Network
    hidden_size: int = 128

    # Evaluation
    eval_every: int = 10          # Evaluate agent every N episodes
    eval_episodes: int = 5        # Number of eval episodes to average
    solve_threshold: float = 195  # CartPole is "solved" at avg reward >= 195


# Default config used across experiments
DEFAULT_CONFIG = Config()
