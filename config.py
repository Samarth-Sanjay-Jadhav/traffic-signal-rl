# =============================================================
#   Autonomous Traffic Signal Control using Deep RL
#   config.py — All hyperparameters and settings
# =============================================================

import os

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
NET_FILE   = os.path.join(BASE_DIR, "nets", "single-intersection", "network.net.xml")
ROUTE_FILE = os.path.join(BASE_DIR, "nets", "single-intersection", "routes.rou.xml")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# ── Simulation Settings ───────────────────────────────────────
NUM_SECONDS      = 1800   # Total simulation time per episode (seconds)
DELTA_TIME       = 5      # Agent acts every 5 seconds
YELLOW_TIME      = 3      # Yellow phase duration (seconds)
MIN_GREEN        = 10     # Minimum green phase duration (seconds)
MAX_GREEN        = 60     # Maximum green phase duration (seconds)

# ── DQN Hyperparameters ───────────────────────────────────────
LEARNING_RATE    = 0.001
GAMMA            = 0.99   # Discount factor for future rewards
EPSILON_START    = 1.0    # Starting exploration rate
EPSILON_END      = 0.05   # Minimum exploration rate
EPSILON_DECAY    = 0.995  # Decay per episode
BATCH_SIZE       = 64     # Mini-batch size for training
MEMORY_SIZE      = 50000  # Replay buffer capacity
TARGET_UPDATE    = 10     # Sync target network every N episodes

# ── Network Architecture ──────────────────────────────────────
STATE_SIZE       = 27     # Size of state vector (lanes x features + phase)
ACTION_SIZE      = 2      # 0 = keep phase, 1 = switch phase
HIDDEN_SIZE      = 64     # Neurons in hidden layers

# ── Training Settings ─────────────────────────────────────────
NUM_EPISODES     = 50    # Total training episodes
EVAL_EPISODES    = 10     # Episodes for evaluation
SAVE_MODEL_EVERY = 10     # Save model checkpoint every N episodes

# ── Reward Weights (from IntelliLight paper Equation 1) ───────
W_QUEUE     = -0.25   # Weight for queue length
W_WAIT      = -0.25   # Weight for waiting time
W_SWITCH    = -5.0    # Penalty for switching lights
W_DELAY     = -0.25   # Weight for vehicle delay

# ── Fixed Timer Baseline Settings ────────────────────────────
FIXED_GREEN_TIME = 30    # Fixed timer green phase duration (seconds)

# ── Logging ──────────────────────────────────────────────────
LOG_INTERVAL = 5         # Print training stats every N episodes
