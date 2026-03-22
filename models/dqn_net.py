# =============================================================
#   models/dqn_net.py — PyTorch Q-Network Architecture
# =============================================================

import torch
import torch.nn as nn

class DQNetwork(nn.Module):
    """
    Deep Q-Network (DQN) for traffic signal control.
    
    Architecture (inspired by IntelliLight paper):
    Input  → Dense(64) → ReLU → Dense(64) → ReLU → Output
    
    Input  : state vector (queue lengths, vehicle counts, phase)
    Output : Q-values for each action [keep, switch]
    """

    def __init__(self, state_size, action_size, hidden_size=64):
        super(DQNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

        # Initialize weights using He initialization (good for ReLU)
        self._initialize_weights()

    def forward(self, x):
        return self.network(x)

    def _initialize_weights(self):
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                nn.init.zeros_(layer.bias)


def build_model(state_size, action_size, hidden_size=64):
    """Helper to build and return a DQNetwork instance."""
    model = DQNetwork(state_size, action_size, hidden_size)
    print(f"[Model] DQNetwork built:")
    print(f"        Input  : {state_size} features")
    print(f"        Hidden : {hidden_size} neurons x 2 layers")
    print(f"        Output : {action_size} actions (keep / switch)")
    return model