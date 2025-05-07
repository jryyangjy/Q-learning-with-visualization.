
# Multi-Agent Path Planning with Q-Learning

Q-learning based multi-agent path planning system with training visualization.

## 🛠️ Installation

# Pip
```bash
pip install -r requirements.txt
```
# Conda
```bash
conda create -n rl_env python=3.10
conda activate rl_env
conda install numpy matplotlib
```

## 🚀 Quick Start

### Default settings
```bash
python train.py
```

### Custom parameters
```bash
python train.py \
  --size 15 \          # Grid size (NxN)
  --agents 3 \         # Number of agents
  --episodes 1000 \    # Training episodes
  --obstacles 0.25 \   # Obstacle density (0.0-1.0)
  --alpha 0.15 \       # Learning rate
  --gamma 0.95 \       # Discount factor
  --epsilon 0.2        # Initial exploration rate
```

## ⚙️ Parameters

| Parameter     | Type  | Default | Description                |
|---------------|-------|---------|----------------------------|
| `--size`      | int   | 20      | Grid dimension (N x N)     |
| `--agents`    | int   | 3       | Number of agents           |
| `--episodes`  | int   | 500     | Training iterations        |
| `--obstacles` | float | 0.2     | Obstacle density (0-1)     |
| `--alpha`     | float | 0.2     | Learning rate (0-1)        |
| `--gamma`     | float | 0.9     | Discount factor (0-1)      |
| `--epsilon`   | float | 0.3     | Initial exploration rate   |

## 📊 Outputs

- `training_ep[XXX].png`: Q-value heatmap + convergence curve (per 100 episodes)
- `path_evolution.gif`: Animated trajectory evolution

## 🔍 Features

- Dynamic ε-decay strategy
- Collision detection mechanism
- Independent Q-tables per agent
- Real-time training visualization

## 📚 Reference
[Q-learning fundamentals](https://stanford-cs221.github.io/autumn2024/) | [Project Structure](https://github.com/ShirakawaSanae/USTC-DS4001-25sp/)