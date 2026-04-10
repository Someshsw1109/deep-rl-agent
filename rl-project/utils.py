"""
Plotting utilities.

All functions return Matplotlib Figure objects so they can be embedded
in Streamlit with st.pyplot() or saved to disk with fig.savefig().
"""

import matplotlib
matplotlib.use("Agg")  # headless backend — required for cloud/server environments
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import Dict, List, Optional


COLORS = {
    "dqn":        "#e74c3c",   # red
    "double_dqn": "#2ecc71",   # green
    "eval":       "#3498db",   # blue
    "loss":       "#9b59b6",   # purple
    "epsilon":    "#f39c12",   # orange
}

plt.rcParams.update({
    "figure.facecolor": "#1e1e2e",
    "axes.facecolor":   "#1e1e2e",
    "axes.edgecolor":   "#44475a",
    "axes.labelcolor":  "#cdd6f4",
    "xtick.color":      "#cdd6f4",
    "ytick.color":      "#cdd6f4",
    "text.color":       "#cdd6f4",
    "grid.color":       "#313244",
    "grid.linewidth":   0.6,
    "legend.facecolor": "#313244",
    "legend.edgecolor": "#44475a",
    "font.size":        11,
})


def _moving_avg(data: List[float], window: int = 20) -> np.ndarray:
    arr = np.array(data, dtype=float)
    kernel = np.ones(window) / window
    if len(arr) < window:
        return arr
    return np.convolve(arr, kernel, mode="valid")


def plot_reward_curve(
    history: Dict,
    agent_label: str = "DQN",
    color: str = COLORS["dqn"],
    window: int = 20,
    title: str = "Reward per Episode",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4))
    episodes = history["episodes"]
    rewards  = history["rewards"]
    raw      = np.array(rewards)

    ax.plot(episodes, raw, alpha=0.25, color=color, linewidth=0.8, label="Raw reward")
    ma = _moving_avg(rewards, window)
    offset = len(episodes) - len(ma)
    ax.plot(
        episodes[offset:], ma,
        color=color, linewidth=2,
        label=f"{agent_label} (MA-{window})",
    )
    if history.get("eval_episodes") and history.get("eval_rewards"):
        ax.scatter(
            history["eval_episodes"], history["eval_rewards"],
            color=COLORS["eval"], zorder=5, s=30, label="Eval (greedy)",
        )
    ax.axhline(195, color="#f1fa8c", linewidth=1, linestyle="--", label="Solved threshold (195)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title(title, pad=12)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, which="major")
    ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
    fig.tight_layout()
    return fig


def plot_comparison(
    dqn_history: Dict,
    ddqn_history: Dict,
    window: int = 20,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))

    for history, label, color in [
        (dqn_history,  "DQN",         COLORS["dqn"]),
        (ddqn_history, "Double DQN",  COLORS["double_dqn"]),
    ]:
        raw = np.array(history["rewards"])
        ax.plot(history["episodes"], raw, alpha=0.15, color=color, linewidth=0.7)
        ma = _moving_avg(history["rewards"], window)
        offset = len(history["episodes"]) - len(ma)
        ax.plot(
            history["episodes"][offset:], ma,
            color=color, linewidth=2.2, label=f"{label} (MA-{window})",
        )

    ax.axhline(195, color="#f1fa8c", linewidth=1, linestyle="--", label="Solved threshold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("DQN vs Double DQN — CartPole-v1", pad=12)
    ax.legend(loc="upper left")
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_loss_curves(
    dqn_history: Dict,
    ddqn_history: Optional[Dict] = None,
    window: int = 20,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 4))

    for history, label, color in [
        (dqn_history,  "DQN",        COLORS["dqn"]),
        *(([(ddqn_history, "Double DQN", COLORS["double_dqn"])] if ddqn_history else [])),
    ]:
        losses = [l for l in history["losses"] if l > 0]
        if not losses:
            continue
        episodes = list(range(1, len(losses) + 1))
        ax.plot(episodes, losses, alpha=0.2, color=color, linewidth=0.7)
        ma = _moving_avg(losses, window)
        offset = len(episodes) - len(ma)
        ax.plot(
            episodes[offset:], ma,
            color=color, linewidth=2, label=f"{label} loss (MA-{window})",
        )

    ax.set_xlabel("Training Step (episodes with loss)")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training Loss Curves", pad=12)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_epsilon_decay(history: Dict, color: str = COLORS["epsilon"]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(history["episodes"], history["epsilons"], color=color, linewidth=1.8)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Epsilon")
    ax.set_title("Exploration Rate (ε) Decay", pad=10)
    ax.set_ylim(0, 1.05)
    ax.grid(True)
    fig.tight_layout()
    return fig


def plot_eval_bar(comparison: Dict) -> plt.Figure:
    labels     = ["DQN", "Double DQN"]
    avg_rewards = [comparison["dqn"]["last50_avg"], comparison["double_dqn"]["last50_avg"]]
    colors     = [COLORS["dqn"], COLORS["double_dqn"]]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, avg_rewards, color=colors, edgecolor="#44475a", width=0.4)
    for bar, val in zip(bars, avg_rewards):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
            f"{val:.1f}", ha="center", va="bottom", fontweight="bold",
        )
    ax.axhline(195, color="#f1fa8c", linestyle="--", linewidth=1, label="Solved threshold")
    ax.set_ylabel("Avg Reward (last 50 episodes)")
    ax.set_title("Algorithm Comparison", pad=12)
    ax.set_ylim(0, max(avg_rewards) * 1.2)
    ax.legend()
    fig.tight_layout()
    return fig
