\# Autonomous Traffic Signal Control using Deep Reinforcement Learning



A Deep Q-Network (DQN) agent that dynamically optimizes traffic signal 

timings to minimize vehicle wait times and queue lengths at a 4-way 

intersection, validated against a fixed-timer baseline.



\## 🏗️ Architecture

```

State  → \[Queue Length, Vehicle Count, Wait Time, Phase] (27 features)

Action → Keep current phase (0) or Switch to next phase (1)

Reward → -0.25 × Queue - 0.25 × Wait Time

```



\## 📊 Results



| Metric             | DQN Agent | Fixed Timer |

|--------------------|-----------|-------------|

| Avg Total Reward   | -184.57   | -184.94     |

| Avg Queue Length   |   0.977   |   0.980     |

| Avg Wait Time      |   4.847   |   4.774     |



!\[Comparison Plot](results/comparison\_plot.png)



\## 🚀 How to Run



\### Setup

```bash

conda create -n traffic-rl python=3.10

conda activate traffic-rl

pip install torch sumo-rl stable-baselines3 numpy pandas matplotlib

```



\### Train

```bash

python training/train.py

```



\### Evaluate

```bash

python evaluation/evaluate.py

```



\## 🛠️ Tech Stack

\- \*\*Simulator:\*\* SUMO (Simulation of Urban MObility)

\- \*\*RL Library:\*\* sumo-rl

\- \*\*Deep Learning:\*\* PyTorch

\- \*\*Algorithm:\*\* Deep Q-Network (DQN)



\## 📚 Reference

Based on \[IntelliLight (KDD 2018)](https://dl.acm.org/doi/10.1145/3219819.3220096)

by Wei et al. — A Reinforcement Learning Approach for Intelligent 

Traffic Light Control.



\## 📁 Project Structure

```

traffic-signal-rl/

├── nets/               # SUMO network files

├── agents/             # DQN and Fixed Timer agents

├── models/             # PyTorch Q-Network

├── training/           # Training loop + Replay Buffer

├── evaluation/         # Evaluation + Plotting

├── results/            # Training logs + Comparison plots

└── config.py           # All hyperparameters

```

