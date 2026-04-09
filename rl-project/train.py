"""
Training loop for DQN / Double DQN.

Returns per-episode reward, loss, and epsilon history for analysis.
"""

import numpy as np
import gymnasium as gym
from typing import Dict, List, Callable, Optional

from agent import DQNAgent, AgentType
from config import Config


def run_episode(agent: DQNAgent, env) -> tuple[float, float, int]:
    """
    Run a single training episode.
    Returns (total_reward, avg_loss, num_steps).
    """
    state, _ = env.reset()
    state = np.array(state, dtype=np.float32)
    total_reward = 0.0
    losses = []
    step = 0

    while True:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = np.array(next_state, dtype=np.float32)

        agent.store(state, action, float(reward), next_state, done)
        loss = agent.learn()
        if loss > 0:
            losses.append(loss)

        total_reward += reward
        state = next_state
        step += 1
        if done:
            break

    agent.on_episode_end()
    avg_loss = float(np.mean(losses)) if losses else 0.0
    return total_reward, avg_loss, step


def train(
    agent_type: AgentType = "dqn",
    config: Optional[Config] = None,
    progress_callback: Optional[Callable[[Dict], None]] = None,
) -> Dict[str, List]:
    """
    Train an agent and collect metrics.

    Args:
        agent_type:         "dqn" or "double_dqn"
        config:             hyperparameter config (uses DEFAULT_CONFIG if None)
        progress_callback:  called after each episode with a metrics dict

    Returns a dict with keys:
        episodes, rewards, losses, epsilons, eval_rewards, eval_episodes
    """
    if config is None:
        config = Config()

    np.random.seed(config.seed)
    env      = gym.make(config.env_name)
    eval_env = gym.make(config.env_name)

    state_size  = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size, config, agent_type)

    history: Dict[str, List] = {
        "episodes":      [],
        "rewards":       [],
        "losses":        [],
        "epsilons":      [],
        "eval_rewards":  [],
        "eval_episodes": [],
    }

    for ep in range(1, config.episodes + 1):
        reward, loss, _ = run_episode(agent, env)

        history["episodes"].append(ep)
        history["rewards"].append(reward)
        history["losses"].append(loss)
        history["epsilons"].append(agent.epsilon)

        # Periodic evaluation (greedy, no exploration)
        if ep % config.eval_every == 0:
            eval_rewards = [agent.evaluate(eval_env) for _ in range(config.eval_episodes)]
            avg_eval = float(np.mean(eval_rewards))
            history["eval_rewards"].append(avg_eval)
            history["eval_episodes"].append(ep)

        if progress_callback:
            progress_callback({
                "episode":   ep,
                "reward":    reward,
                "loss":      loss,
                "epsilon":   agent.epsilon,
                "progress":  ep / config.episodes,
            })

    env.close()
    eval_env.close()
    return history
