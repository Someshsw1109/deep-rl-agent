"""
Experience Replay Buffer.

Stores past (state, action, reward, next_state, done) transitions.
Random sampling breaks temporal correlations in training data,
which stabilises the Q-network update.
"""

import random
from collections import deque
from typing import List, Tuple
import numpy as np


Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]


class ReplayBuffer:
    """Fixed-size circular buffer for experience replay."""

    def __init__(self, capacity: int, seed: int = 42):
        self.buffer: deque = deque(maxlen=capacity)
        self.capacity = capacity
        random.seed(seed)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)

    @property
    def is_ready(self) -> bool:
        return len(self.buffer) >= self.capacity // 10
