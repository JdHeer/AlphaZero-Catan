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


class ActionEncoder:
    """
    Encodes Catan actions to fixed-size vector indices.

    Catan has a complex action space including:
    - Building settlements/cities/roads at specific locations
    - Buying/playing development cards
    - Trading with bank/ports/players
    - Moving the robber
    - Discarding cards

    This class maps all possible actions to unique indices.
    """

    def __init__(self):
        # Action type base indices
        self.action_types = {
            "BUILD_SETTLEMENT": 0,      # 54 possible locations
            "BUILD_ROAD": 54,           # 72 possible locations
            "BUILD_CITY": 126,          # 54 possible locations
            "BUY_DEVELOPMENT_CARD": 180,# 1 action
            "PLAY_KNIGHT_CARD": 181,    # 19 robber destinations
            "ROLL": 200,                # 1 action
            "END_TURN": 201,            # 1 action
            "MARITIME_TRADE": 202,      # Multiple trade options
            "DISCARD": 250,             # Multiple discard options
            # ... more action types
        }

        # Total action space size (conservative estimate)
        self.action_space_size = 500

    def encode(self, action) -> int:
        """Convert a Catan action to an integer index."""
        # Simplified encoding - full implementation needs
        # to handle all action types and their parameters
        action_type = action.action_type.name

        if action_type in self.action_types:
            base_idx = self.action_types[action_type]
            # Add offset based on action-specific parameters
            # This is a simplified version
            return base_idx

        return 0  # Default fallback

    def decode(self, index: int, valid_actions: list):
        """
        Convert an index back to a Catan action.

        Since actions are context-dependent, we need the list
        of valid actions to find the matching one.
        """
        # Find the action that maps to this index
        for action in valid_actions:
            if self.encode(action) == index:
                return action

        # Fallback: return first valid action
        return valid_actions[0] if valid_actions else None

    def get_valid_action_mask(self, valid_actions: list) -> np.ndarray:
        """Create a binary mask of valid actions."""
        mask = np.zeros(self.action_space_size, dtype=np.float32)
        for action in valid_actions:
            idx = self.encode(action)
            if 0 <= idx < self.action_space_size:
                mask[idx] = 1
        return mask


if __name__ == "__main__":
    # Test the network
    state_size = 32  # From game_wrapper
    action_size = 500

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
