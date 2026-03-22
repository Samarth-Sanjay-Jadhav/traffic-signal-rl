# =============================================================
#   agents/dqn_agent.py — DQN Agent for Traffic Signal Control
# =============================================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dqn_net import DQNetwork
from training.replay_buffer import ReplayBuffer
from config import (
    STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE,
    LEARNING_RATE, GAMMA,
    EPSILON_START, EPSILON_END, EPSILON_DECAY,
    BATCH_SIZE, MEMORY_SIZE, TARGET_UPDATE
)


class DQNAgent:
    """
    Deep Q-Network Agent for Traffic Signal Control.

    Key components:
    - Online Network  : learns Q-values at every step
    - Target Network  : frozen copy, synced every N episodes
                        (stabilizes training)
    - Replay Buffer   : stores past experiences, breaks correlation
    - Epsilon-Greedy  : balances exploration vs exploitation
    """

    def __init__(self):
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epsilon     = EPSILON_START
        self.steps_done  = 0
        self.episode     = 0

        print(f"[Agent] Using device: {self.device}")

        # ── Networks ─────────────────────────────────────────
        self.online_net = DQNetwork(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE).to(self.device)
        self.target_net = DQNetwork(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()  # Target net is never trained directly

        # ── Optimizer & Loss ──────────────────────────────────
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=LEARNING_RATE)
        self.loss_fn   = nn.MSELoss()

        # ── Replay Buffer ─────────────────────────────────────
        self.memory = ReplayBuffer(MEMORY_SIZE)

        print(f"[Agent] DQN Agent initialized.")
        print(f"        Epsilon : {self.epsilon}")
        print(f"        Memory  : {MEMORY_SIZE} experiences")

    # ── Action Selection ──────────────────────────────────────
    def select_action(self, state):
        """
        Epsilon-greedy action selection.
        - With prob epsilon  → explore (random action)
        - With prob 1-epsilon → exploit (best Q-value action)
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(ACTION_SIZE)   # Explore

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.online_net(state_tensor)
        return q_values.argmax().item()             # Exploit

    # ── Store Experience ──────────────────────────────────────
    def store_experience(self, state, action, reward, next_state, done):
        """Store a transition in replay buffer."""
        self.memory.push(state, action, reward, next_state, done)

    # ── Train Step ────────────────────────────────────────────
    def train_step(self):
        """
        Sample a mini-batch and perform one gradient update.
        Uses Bellman equation:
            Q(s,a) = r + gamma * max Q(s', a')
        """
        if not self.memory.is_ready(BATCH_SIZE):
            return None  # Not enough samples yet

        # Sample mini-batch from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)

        # Convert to tensors
        states      = torch.FloatTensor(states).to(self.device)
        actions     = torch.LongTensor(actions).to(self.device)
        rewards     = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones       = torch.FloatTensor(dones).to(self.device)

        # Current Q-values from online network
        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values from target network (Bellman equation)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0]
            target_q   = rewards + GAMMA * max_next_q * (1 - dones)

        # Compute loss and backpropagate
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent exploding gradients
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.steps_done += 1
        return loss.item()

    # ── Epsilon Decay ─────────────────────────────────────────
    def decay_epsilon(self):
        """Decay epsilon after each episode."""
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)
        self.episode += 1

    # ── Target Network Sync ───────────────────────────────────
    def update_target_network(self):
        """Copy online network weights to target network."""
        self.target_net.load_state_dict(self.online_net.state_dict())
        print(f"[Agent] Target network synced at episode {self.episode}")

    # ── Save / Load Model ─────────────────────────────────────
    def save(self, path):
        torch.save({
            'online_net'  : self.online_net.state_dict(),
            'target_net'  : self.target_net.state_dict(),
            'optimizer'   : self.optimizer.state_dict(),
            'epsilon'     : self.epsilon,
            'episode'     : self.episode,
            'steps_done'  : self.steps_done,
        }, path)
        print(f"[Agent] Model saved → {path}")

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(checkpoint['online_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon    = checkpoint['epsilon']
        self.episode    = checkpoint['episode']
        self.steps_done = checkpoint['steps_done']
        print(f"[Agent] Model loaded ← {path}")