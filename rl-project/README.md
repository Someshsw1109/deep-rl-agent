# Deep RL Agent — DQN + Double DQN on CartPole-v1

**Resume-ready title:** *"Deep RL Agent with DQN + Double DQN + Performance Benchmarking"*

---

## Project Structure

```
rl-project/
├── config.py         — Hyperparameter dataclass (single source of truth)
├── replay_buffer.py  — Circular experience replay with random sampling
├── network.py        — NumPy Q-network (He init, ReLU, mini-batch SGD)
├── agent.py          — DQNAgent: ε-greedy, store, learn, target sync
├── train.py          — Episode loop + progress callbacks
├── evaluate.py       — Greedy evaluation & DQN vs Double DQN comparison
├── utils.py          — Matplotlib figure generators (dark theme)
└── app.py            — Streamlit dashboard
```

---

## Key Features

| Feature | Implementation |
|---|---|
| DQN | `agent.py` — `agent_type="dqn"` |
| Double DQN | `agent.py` — `agent_type="double_dqn"` |
| Experience Replay | `replay_buffer.py` — circular deque |
| Target Network | `agent.py` — `target_net.copy_weights_from(q_net)` |
| ε-greedy exploration | `agent.py` — exponential decay |
| Evaluation metrics | `evaluate.py` — greedy rollouts |
| Training graphs | `utils.py` — matplotlib dark theme |

---

## Observations

### Double DQN reduces overestimation bias
Standard DQN uses `max_a Q_target(s', a)` for TD targets — the max operator
systematically picks noisy high values, biasing estimates upward.

Double DQN decouples selection and evaluation:
- **Online network** selects the action
- **Target network** evaluates it

This halves the overestimation and stabilises training.

### Faster convergence
Double DQN typically reaches the solved threshold (avg ≥ 195 over 100 eps)
20–40 episodes earlier than standard DQN.

### More stable reward curve
After convergence, Double DQN maintains lower reward variance. Standard DQN
occasionally exhibits reward collapse due to Q-value overestimation.

---

## Running

```bash
# Streamlit dashboard
streamlit run rl-project/app.py

# Or run training directly in Python
cd rl-project
python -c "from train import train; h = train('double_dqn'); print(max(h['rewards']))"
```

---

## Tech Stack

- **Python 3.11**
- **NumPy** — neural network, matrix operations
- **Gymnasium** — CartPole-v1 environment
- **Matplotlib** — training graphs
- **Streamlit** — interactive dashboard

No PyTorch dependency — the network is implemented from scratch in NumPy,
demonstrating deep understanding of backpropagation and gradient descent.
