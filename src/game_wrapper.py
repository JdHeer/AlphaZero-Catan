"""
Game wrapper for Catanatron - provides clean interface for AlphaZero training.
"""
import copy
import numpy as np
from catanatron import Color, Game, RandomPlayer

from src.encoders import StateEncoder, ActionEncoder, get_state_size as encoder_get_state_size


class CatanGameWrapper:
    """Wrapper around Catanatron game for AlphaZero training."""

    def __init__(self, num_players: int = 4):
        self.num_players = num_players
        self.colors = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE][:num_players]
        self.game = None
        self.state_encoder = StateEncoder(num_players)
        self.action_encoder = ActionEncoder()
        self.reset()

    def reset(self) -> np.ndarray:
        """Reset the game and return initial state."""
        players = [RandomPlayer(color) for color in self.colors]
        self.game = Game(players)
        return self.get_state()

    def get_state(self) -> np.ndarray:
        """Convert game state to neural network input using full encoder."""
        return self.state_encoder.encode(self.game)

    def get_valid_actions(self) -> list:
        """Get list of valid actions in current state."""
        return self.game.state.playable_actions

    def get_action_mask(self) -> np.ndarray:
        """Get binary mask of valid actions."""
        valid_actions = self.get_valid_actions()
        return self.action_encoder.get_valid_action_mask(valid_actions, self.game)

    def step(self, action) -> tuple:
        """
        Execute an action and return (next_state, reward, done, info).

        Args:
            action: A Catanatron Action object

        Returns:
            state: New game state as numpy array
            reward: Reward signal (1 for win, -1 for loss, 0 otherwise)
            done: Whether game is over
            info: Additional information dict
        """
        # Execute the action
        self.game.execute(action)

        # Check if game is over
        winner = self.game.winning_color()
        done = winner is not None

        # Calculate reward (from perspective of player who just moved)
        reward = 0
        if done:
            # Get the player who made the last move
            last_player = action.color
            reward = 1.0 if winner == last_player else -1.0

        new_state = self.get_state()
        info = {
            "winner": winner,
            "current_player": self.game.state.current_color(),
            "turn": self.game.state.num_turns,
        }

        return new_state, reward, done, info

    def get_current_player(self) -> int:
        """Get index of current player."""
        return self.game.state.current_player_index

    def clone(self) -> "CatanGameWrapper":
        """Create a deep copy of the game for MCTS simulation."""
        new_wrapper = CatanGameWrapper.__new__(CatanGameWrapper)
        new_wrapper.num_players = self.num_players
        new_wrapper.colors = self.colors
        new_wrapper.game = copy.deepcopy(self.game)
        new_wrapper.state_encoder = self.state_encoder
        new_wrapper.action_encoder = self.action_encoder
        return new_wrapper


def get_state_size(num_players: int = 4) -> int:
    """Get the state vector size from encoder."""
    return encoder_get_state_size(num_players)


if __name__ == "__main__":
    # Quick test
    wrapper = CatanGameWrapper(num_players=4)
    state = wrapper.get_state()
    print(f"State shape: {state.shape}")
    print(f"Expected: {get_state_size(4)}")
    print(f"Valid actions: {len(wrapper.get_valid_actions())}")
    print(f"Current player: {wrapper.get_current_player()}")
    print(f"Action mask sum: {wrapper.get_action_mask().sum()}")
