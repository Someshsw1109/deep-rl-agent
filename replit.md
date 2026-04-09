# Workspace

## Overview

pnpm workspace monorepo using TypeScript for the web/API layer. Also contains a Python RL research project in `rl-project/`.

## Stack

- **Monorepo tool**: pnpm workspaces
- **Node.js version**: 24
- **Package manager**: pnpm
- **TypeScript version**: 5.9
- **API framework**: Express 5
- **Database**: PostgreSQL + Drizzle ORM
- **Validation**: Zod (`zod/v4`), `drizzle-zod`
- **API codegen**: Orval (from OpenAPI spec)
- **Build**: esbuild (CJS bundle)

## Python RL Research Project (`rl-project/`)

Deep reinforcement learning agent implementing DQN and Double DQN on CartPole-v1.

- **Python**: 3.11
- **Packages**: numpy, gymnasium, matplotlib, streamlit
- **Dashboard**: `streamlit run rl-project/app.py` (port 5000)

### Key files:
- `rl-project/config.py` — Hyperparameter config
- `rl-project/network.py` — NumPy Q-network from scratch
- `rl-project/agent.py` — DQN + Double DQN logic
- `rl-project/replay_buffer.py` — Experience replay
- `rl-project/train.py` — Training loop
- `rl-project/evaluate.py` — Greedy evaluation & comparison
- `rl-project/utils.py` — Matplotlib dark-theme figures
- `rl-project/app.py` — Streamlit dashboard

## Key Commands

- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- `pnpm --filter @workspace/api-server run dev` — run API server locally
- `streamlit run rl-project/app.py` — run RL dashboard
