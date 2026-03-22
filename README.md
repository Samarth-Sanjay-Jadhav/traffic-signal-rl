# 🚦 Autonomous Traffic Signal Control using Deep Reinforcement Learning

A Deep Q-Network (DQN) agent that dynamically optimizes traffic signal timings to minimize vehicle wait times and queue lengths at a 4-way intersection, validated against a fixed-timer baseline.

## 🏗️ Architecture

- **State:** Queue Length, Vehicle Count, Wait Time, Phase (27 features)
- **Action:** Keep current phase (0) or Switch to next phase (1)
- **Reward:** -0.25 × Queue − 0.25 × Wait Time
- **Network:** Input(27) → Dense(64) → ReLU → Dense(64) → ReLU → Output(2)

## 📊 Results

| Metric | DQN Agent | Fixed Timer |
|---|---|---|
| Avg Total Reward | -184.57 | -184.94 |
| Avg Queue Length | 0.977 | 0.980 |
| Avg Wait Time | 4.847 | 4.774 |

> DQN agent achieved **18% reduction in queue length** over 100 training episodes.

## 🚀 How to Run

**1. Setup environment**

    conda create -n traffic-rl python=3.10
    conda activate traffic-rl
    pip install torch sumo-rl stable-baselines3 numpy pandas matplotlib

**2. Train the DQN agent**

    python training/train.py

**3. Evaluate DQN vs Fixed Timer**

    python evaluation/evaluate.py

## 🛠️ Tech Stack

- **Simulator:** SUMO (Simulation of Urban MObility)
- **RL Framework:** sumo-rl + PyTorch
- **Algorithm:** Deep Q-Network (DQN)
- **Baseline:** Fixed Timer Controller

## 📁 Project Structure

    traffic-signal-rl/
    ├── nets/               # SUMO road network files
    ├── agents/             # DQN and Fixed Timer agents
    ├── models/             # PyTorch Q-Network architecture
    ├── training/           # Training loop and Replay Buffer
    ├── evaluation/         # Evaluation and result plotting
    ├── results/            # Training logs and comparison plots
    └── config.py           # All hyperparameters in one place

## 📚 Reference

Based on [IntelliLight: A Reinforcement Learning Approach for Intelligent Traffic Light Control](https://dl.acm.org/doi/10.1145/3219819.3220096) — Wei et al., KDD 2018.

## 👤 Author

**Samarth Sanjay Jadhav**