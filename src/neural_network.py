"""
Neural Network for AlphaZero Catan Agent.

The network outputs:
- Policy: Probability distribution over actions
- Value: Expected outcome from current state [-1, 1]
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CatanNetwork(nn.Module):
    """
    Neural network for Catan AlphaZero agent.

    Architecture: Shared trunk with policy and value heads.
    """

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int = 256,
        num_layers: int = 4,
    ):
        super().__init__()

        self.state_size = state_size
        self.action_size = action_size

        # Shared trunk
        layers = []
        layers.append(nn.Linear(state_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_size))

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))

        self.trunk = nn.Sequential(*layers)

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Tanh(),  # Output in [-1, 1]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: State tensor of shape (batch, state_size)

        Returns:
            policy_logits: Raw logits of shape (batch, action_size)
            value: Value estimate of shape (batch, 1)
        """
        # Handle single sample (no batch dimension)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        trunk_out = self.trunk(x)
        policy_logits = self.policy_head(trunk_out)
        value = self.value_head(trunk_out)

        return policy_logits, value

    def predict(
        self,
        state: np.ndarray,
        valid_action_mask: np.ndarray = None
    ) -> tuple[np.ndarray, float]:
        """
        Get policy and value for a single state.

        Args:
            state: Game state as numpy array
            valid_action_mask: Binary mask of valid actions (optional)

        Returns:
            policy: Probability distribution over actions
            value: Scalar value estimate
        """
        self.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            policy_logits, value = self.forward(state_tensor)

            # Apply mask for invalid actions if provided
            if valid_action_mask is not None:
                mask_tensor = torch.FloatTensor(valid_action_mask)
                # Set invalid actions to very negative value
                policy_logits = policy_logits.squeeze(0)
                policy_logits[mask_tensor == 0] = float('-inf')

            policy = F.softmax(policy_logits, dim=-1).numpy()
            value = value.item()

        return policy.flatten(), value


# ActionEncoder is now in encoders.py - import from there
from src.encoders import get_action_size, get_state_size

if __name__ == "__main__":
    # Test the network with proper sizes
    state_size = get_state_size()
    action_size = get_action_size()

    print(f"State size: {state_size}")
    print(f"Action size: {action_size}")

    net = CatanNetwork(state_size, action_size)

    # Test forward pass
    dummy_state = torch.randn(1, state_size)
    policy, value = net(dummy_state)

    print(f"Policy shape: {policy.shape}")
    print(f"Value shape: {value.shape}")
    print(f"Value: {value.item():.4f}")

    # Test prediction
    state_np = np.random.randn(state_size).astype(np.float32)
    policy_np, value_np = net.predict(state_np)
    print(f"Policy sum: {policy_np.sum():.4f}")  # Should be ~1
    print(f"Value: {value_np:.4f}")
