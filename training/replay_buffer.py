# =============================================================
#   training/replay_buffer.py — Experience Replay Memory
#   Inspired by Memory Palace concept in IntelliLight paper
# =============================================================

import random
import numpy as np
from collections import deque

class ReplayBuffer:
    """
    Experience Replay Buffer.
    Stores (state, action, reward, next_state, done) tuples.
    Randomly samples mini-batches for DQN training.
    """

    def __init__(self, capacity):
        self.buffer   = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        self.buffer.append((
            np.array(state,      dtype=np.float32),
            int(action),
            float(reward),
            np.array(next_state, dtype=np.float32),
            bool(done)
        ))

    def sample(self, batch_size):
        """Randomly sample a mini-batch of experiences."""
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)

    def is_ready(self, batch_size):
        """Check if enough samples exist to train."""
        return len(self.buffer) >= batch_size
