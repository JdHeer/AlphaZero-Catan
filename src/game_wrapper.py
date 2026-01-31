"""
Game wrapper for Catanatron - provides clean interface for AlphaZero training.
"""
import numpy as np
from catanatron import Color, Game, RandomPlayer


class CatanGameWrapper:
    """Wrapper around Catanatron game for AlphaZero training."""

    def __init__(self, num_players: int = 4):
        self.num_players = num_players
        self.colors = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE][:num_players]
        self.game = None
        self.reset()

    def reset(self) -> np.ndarray:
        """Reset the game and return initial state."""
        players = [RandomPlayer(color) for color in self.colors]
        self.game = Game(players)
        return self.get_state()

    def get_state(self) -> np.ndarray:
        """
        Convert game state to neural network input.

        Returns a flat numpy array encoding:
        - Board state (tiles, numbers, ports)
        - Building positions for each player
        - Resource counts for each player
        - Development cards
        - Victory points
        - Current player
        - Robber position
        """
        state = self.game.state
        ps = state.player_state  # Player state dictionary
        features = []
        
        # Current player one-hot encoding
        current_player_idx = state.current_player_index
        player_encoding = np.zeros(self.num_players)
        player_encoding[current_player_idx] = 1
        features.extend(player_encoding)
        
        # Resource counts for each player (normalized)
        for i in range(self.num_players):
            resources = [
                ps.get(f"P{i}_WOOD_IN_HAND", 0),
                ps.get(f"P{i}_BRICK_IN_HAND", 0),
                ps.get(f"P{i}_SHEEP_IN_HAND", 0),
                ps.get(f"P{i}_WHEAT_IN_HAND", 0),
                ps.get(f"P{i}_ORE_IN_HAND", 0),
            ]
            # Normalize resources (cap at 10 for stability)
            features.extend([min(r, 10) / 10.0 for r in resources])
        
        # Victory points for each player (normalized to 10)
        for i in range(self.num_players):
            vp = ps.get(f"P{i}_VICTORY_POINTS", 0)
            features.append(vp / 10.0)
        
        # Army size for each player
        for i in range(self.num_players):
            army = ps.get(f"P{i}_PLAYED_KNIGHT", 0)
            features.append(min(army, 10) / 10.0)
        
        # Longest road length for each player
        for i in range(self.num_players):
            road = ps.get(f"P{i}_LONGEST_ROAD_LENGTH", 0)
            features.append(min(road, 15) / 15.0)
        
        # Development cards in hand for each player
        for i in range(self.num_players):
            dev_cards = (
                ps.get(f"P{i}_KNIGHT_IN_HAND", 0) +
                ps.get(f"P{i}_YEAR_OF_PLENTY_IN_HAND", 0) +
                ps.get(f"P{i}_MONOPOLY_IN_HAND", 0) +
                ps.get(f"P{i}_ROAD_BUILDING_IN_HAND", 0) +
                ps.get(f"P{i}_VICTORY_POINT_IN_HAND", 0)
            )
            features.append(min(dev_cards, 10) / 10.0)
        
        # Buildings available (settlements, cities, roads)
        for i in range(self.num_players):
            features.append((5 - ps.get(f"P{i}_SETTLEMENTS_AVAILABLE", 5)) / 5.0)
            features.append((4 - ps.get(f"P{i}_CITIES_AVAILABLE", 4)) / 4.0)
            features.append((15 - ps.get(f"P{i}_ROADS_AVAILABLE", 15)) / 15.0)
        
        # TODO: Add more sophisticated features:
        # - Board topology encoding
        # - Settlement/city positions
        # - Road network
        # - Port access
        
        return np.array(features, dtype=np.float32)
    
    def get_valid_actions(self) -> list:
        """Get list of valid actions in current state."""
        return self.game.state.playable_actions
    
    def get_action_mask(self, action_space_size: int) -> np.ndarray:
        """
        Get binary mask of valid actions.
        
        Note: Catan has a variable action space. We need to map
        actions to fixed indices for the neural network.
        """
        valid_actions = self.get_valid_actions()
        # For now, return the valid actions list
        # Full implementation needs action indexing scheme
        return valid_actions

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
        import copy
        new_wrapper = CatanGameWrapper.__new__(CatanGameWrapper)
        new_wrapper.num_players = self.num_players
        new_wrapper.colors = self.colors
        new_wrapper.game = copy.deepcopy(self.game)
        return new_wrapper


def get_state_size(num_players: int = 4) -> int:
    """Calculate the state vector size."""
    # Current player: num_players (one-hot)
    # Resources per player: 5 * num_players
    # Victory points: num_players
    # Army size: num_players
    # Longest road: num_players
    # Dev cards: num_players
    # Buildings (settlements, cities, roads): 3 * num_players
    return num_players + (5 * num_players) + num_players + num_players + num_players + num_players + (3 * num_players)


if __name__ == "__main__":
    # Quick test
    wrapper = CatanGameWrapper(num_players=4)
    state = wrapper.get_state()
    print(f"State shape: {state.shape}")
    print(f"State: {state}")
    print(f"Valid actions: {len(wrapper.get_valid_actions())}")
    print(f"Current player: {wrapper.get_current_player()}")
