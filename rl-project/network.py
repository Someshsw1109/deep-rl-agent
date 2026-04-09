"""
Lightweight Q-Network implemented in pure NumPy.

Architecture: Linear → ReLU → Linear → ReLU → Linear (Q-values)

No PyTorch dependency — keeps the project portable and fast for CartPole.
The network uses He initialisation and mini-batch SGD with MSE loss.
"""

import numpy as np
from typing import List


class QNetwork:
    """
    Two-hidden-layer fully-connected network.
    Inputs:  state vector  (state_size,)
    Outputs: Q-values      (action_size,)
    """

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128, seed: int = 42):
        np.random.seed(seed)
        self.layer_sizes = [state_size, hidden_size, hidden_size, action_size]
        # He initialisation: std = sqrt(2 / fan_in)
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        for i in range(len(self.layer_sizes) - 1):
            fan_in = self.layer_sizes[i]
            std = np.sqrt(2.0 / fan_in)
            W = np.random.randn(self.layer_sizes[i + 1], fan_in) * std
            b = np.zeros(self.layer_sizes[i + 1])
            self.weights.append(W)
            self.biases.append(b)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x : (state_size,) or (batch, state_size)
        Returns Q-values of shape (action_size,) or (batch, action_size).
        """
        single = x.ndim == 1
        if single:
            x = x[np.newaxis, :]      # (1, state_size)
        h = x
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            h = h @ W.T + b           # (batch, out)
            if i < len(self.weights) - 1:
                h = np.maximum(0, h)  # ReLU on all hidden layers
        return h[0] if single else h  # back to (action_size,) if single

    # ------------------------------------------------------------------
    # Training step (mini-batch gradient descent)
    # ------------------------------------------------------------------
    def train_step(
        self,
        states: np.ndarray,        # (batch, state_size)
        actions: np.ndarray,       # (batch,) int
        targets: np.ndarray,       # (batch,) float  — TD targets for chosen actions
        lr: float,
    ) -> float:
        """One SGD step; returns MSE loss."""
        batch = states.shape[0]

        # ---- Forward pass (store activations) -------------------------
        activations = [states]
        pre_acts = []
        h = states
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            z = h @ W.T + b
            pre_acts.append(z)
            h = np.maximum(0, z) if i < len(self.weights) - 1 else z
            activations.append(h)

        q_all = activations[-1]  # (batch, action_size)

        # ---- Loss: only on the taken action ---------------------------
        predicted = q_all[np.arange(batch), actions]
        error = predicted - targets                 # (batch,)
        loss = float(np.mean(error ** 2))

        # ---- Backward pass --------------------------------------------
        # Gradient of MSE w.r.t. q_all
        dL_dq = np.zeros_like(q_all)
        dL_dq[np.arange(batch), actions] = 2 * error / batch

        delta = dL_dq
        for i in reversed(range(len(self.weights))):
            inp = activations[i]
            dW = delta.T @ inp / batch
            db = delta.mean(axis=0)
            self.weights[i] -= lr * dW
            self.biases[i]  -= lr * db
            if i > 0:
                delta = delta @ self.weights[i]
                delta *= (pre_acts[i - 1] > 0).astype(float)  # ReLU grad

        return loss

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def copy_weights_from(self, other: "QNetwork") -> None:
        """Hard copy (used for target-network sync)."""
        self.weights = [W.copy() for W in other.weights]
        self.biases  = [b.copy() for b in other.biases]

    def greedy_action(self, state: np.ndarray) -> int:
        q = self.forward(state)
        return int(np.argmax(q))
