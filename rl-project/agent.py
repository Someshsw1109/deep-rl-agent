"""
DQN and Double DQN agents.

Key differences:
  DQN:         target = r + γ · max_a Q_target(s', a)
  Double DQN:  target = r + γ · Q_target(s', argmax_a Q_online(s', a))

Double DQN decouples action selection from action evaluation, which
reduces the overestimation bias inherent in standard DQN.
"""

import numpy as np
from typing import Literal

from network import QNetwork
from replay_buffer import ReplayBuffer
from config import Config


AgentType = Literal["dqn", "double_dqn"]


class DQNAgent:
    """
    Unified agent that implements both DQN and Double DQN.
    Toggle with `agent_type`.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        config: Config,
        agent_type: AgentType = "dqn",
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        self.agent_type = agent_type

        # Q-network & frozen target network
        self.q_net     = QNetwork(state_size, action_size, config.hidden_size, config.seed)
        self.target_net = QNetwork(state_size, action_size, config.hidden_size, config.seed)
        self.target_net.copy_weights_from(self.q_net)

        # Experience replay
        self.buffer = ReplayBuffer(config.buffer_capacity, config.seed)

        # Exploration
        self.epsilon = config.epsilon_start
        self._episode = 0

    # ------------------------------------------------------------------
    # Action selection (ε-greedy)
    # ------------------------------------------------------------------
    def select_action(self, state: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        return self.q_net.greedy_action(state)

    # ------------------------------------------------------------------
    # Store transition
    # ------------------------------------------------------------------
    def store(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.buffer.push(state, action, reward, next_state, done)

    # ------------------------------------------------------------------
    # Learning step
    # ------------------------------------------------------------------
    def learn(self) -> float:
        """
        Sample a mini-batch and perform one gradient step.
        Returns the MSE loss (0.0 if buffer not ready).
        """
        if len(self.buffer) < self.config.min_buffer_size:
            return 0.0

        batch = self.buffer.sample(self.config.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states      = np.array(states,      dtype=np.float32)
        actions     = np.array(actions,     dtype=np.int32)
        rewards     = np.array(rewards,     dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones       = np.array(dones,       dtype=np.float32)

        # Compute TD targets
        if self.agent_type == "double_dqn":
            # Online network chooses the action; target network evaluates it
            next_q_online  = self.q_net.forward(next_states)          # (batch, actions)
            best_actions   = np.argmax(next_q_online, axis=1)          # (batch,)
            next_q_target  = self.target_net.forward(next_states)      # (batch, actions)
            next_q_values  = next_q_target[np.arange(len(batch)), best_actions]
        else:
            # Standard DQN: target network picks the max Q directly
            next_q_target  = self.target_net.forward(next_states)      # (batch, actions)
            next_q_values  = np.max(next_q_target, axis=1)             # (batch,)

        targets = rewards + self.config.gamma * next_q_values * (1 - dones)

        loss = self.q_net.train_step(states, actions, targets, self.config.lr)
        return loss

    # ------------------------------------------------------------------
    # Episode bookkeeping
    # ------------------------------------------------------------------
    def on_episode_end(self) -> None:
        """Call once at the end of every episode."""
        self._episode += 1
        # Decay exploration
        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon * self.config.epsilon_decay,
        )
        # Sync target network periodically
        if self._episode % self.config.target_update_freq == 0:
            self.target_net.copy_weights_from(self.q_net)

    # ------------------------------------------------------------------
    # Greedy evaluation (no exploration)
    # ------------------------------------------------------------------
    def evaluate(self, env) -> float:
        state, _ = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action = self.q_net.greedy_action(np.array(state, dtype=np.float32))
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        return total_reward
