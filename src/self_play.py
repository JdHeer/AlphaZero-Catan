"""
Self-play module for generating training data.

Plays games using MCTS and collects (state, policy, value) tuples
for training the neural network.
"""
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np


@dataclass
class GameRecord:
    """Record of a single game for training."""
    states: List[np.ndarray]
    policies: List[np.ndarray]  # MCTS visit count distributions
    actions: List[int]  # Actual actions taken
    players: List[int]  # Which player made each move
    winner: int  # Index of winning player (-1 for draw)

    def get_training_data(self) -> List[tuple]:
        """
        Convert game record to training examples.

        Returns list of (state, policy, value) tuples where value
        is +1 for winner's moves, -1 for losers, 0 for draws.
        """
        examples = []

        for i, (state, policy, player) in enumerate(
            zip(self.states, self.policies, self.players)
        ):
            if self.winner == -1:
                value = 0.0  # Draw
            elif player == self.winner:
                value = 1.0  # This player won
            else:
                value = -1.0  # This player lost

            examples.append((state, policy, value))

        return examples


class SelfPlay:
    """
    Generates training data through self-play.
    """

    def __init__(
        self,
        game_wrapper_class,
        network,
        mcts_class,
        action_encoder,
        num_simulations: int = 100,
        num_players: int = 4,
    ):
        self.game_wrapper_class = game_wrapper_class
        self.network = network
        self.mcts_class = mcts_class
        self.action_encoder = action_encoder
        self.num_simulations = num_simulations
        self.num_players = num_players

    def play_game(
        self,
        temperature_threshold: int = 30,
        verbose: bool = False,
    ) -> GameRecord:
        """
        Play a single game of self-play.

        Args:
            temperature_threshold: Use temperature=1 for first N moves,
                                   then temperature=0 (greedy)
            verbose: Print game progress

        Returns:
            GameRecord with game history
        """
        # Initialize game
        game = self.game_wrapper_class(num_players=self.num_players)

        # Initialize MCTS
        mcts = self.mcts_class(
            network=self.network,
            action_encoder=self.action_encoder,
            num_simulations=self.num_simulations,
        )

        # Game history
        states = []
        policies = []
        actions = []
        players = []

        move_count = 0
        done = False

        while not done:
            current_player = game.get_current_player()
            current_state = game.get_state()

            # Choose temperature based on move count
            temp = 1.0 if move_count < temperature_threshold else 0.0

            # Run MCTS
            action, action_probs = mcts.search(game, temperature=temp)

            if action is None:
                # No valid actions (shouldn't happen normally)
                break

            # Record state and policy
            states.append(current_state.copy())

            # Convert action probs to full action space
            valid_actions = game.get_valid_actions()
            full_policy = np.zeros(self.action_encoder.action_space_size)
            for i, a in enumerate(valid_actions):
                idx = self.action_encoder.encode(a)
                if idx < len(full_policy) and i < len(action_probs):
                    full_policy[idx] = action_probs[i]
            policies.append(full_policy)

            actions.append(self.action_encoder.encode(action))
            players.append(current_player)

            # Execute action
            _, reward, done, info = game.step(action)
            move_count += 1

            if verbose and move_count % 10 == 0:
                print(f"Move {move_count}, Player {current_player}")

            # Safety check for very long games
            if move_count > 1000:
                if verbose:
                    print("Game exceeded 1000 moves, ending...")
                break

        # Determine winner
        winner_color = game.game.winning_color()
        if winner_color:
            winner = game.colors.index(winner_color)
        else:
            winner = -1  # Draw or no winner

        if verbose:
            print(f"Game finished after {move_count} moves. Winner: {winner}")

        return GameRecord(
            states=states,
            policies=policies,
            actions=actions,
            players=players,
            winner=winner,
        )

    def generate_games(
        self,
        num_games: int,
        save_path: str = None,
        verbose: bool = True,
    ) -> List[GameRecord]:
        """
        Generate multiple games of self-play.

        Args:
            num_games: Number of games to play
            save_path: Optional path to save game records
            verbose: Print progress

        Returns:
            List of GameRecord objects
        """
        games = []

        for i in range(num_games):
            if verbose:
                print(f"\nPlaying game {i + 1}/{num_games}...")

            record = self.play_game(verbose=verbose)
            games.append(record)

            if verbose:
                print(f"  Moves: {len(record.states)}, Winner: Player {record.winner}")

        if save_path:
            self.save_games(games, save_path)
            if verbose:
                print(f"\nSaved {num_games} games to {save_path}")

        return games

    @staticmethod
    def save_games(games: List[GameRecord], path: str):
        """Save game records to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(games, f)

    @staticmethod
    def load_games(path: str) -> List[GameRecord]:
        """Load game records from disk."""
        with open(path, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def games_to_training_data(games: List[GameRecord]) -> tuple:
        """
        Convert game records to training arrays.

        Returns:
            states: (N, state_size) array
            policies: (N, action_size) array
            values: (N,) array
        """
        all_examples = []
        for game in games:
            all_examples.extend(game.get_training_data())

        states = np.array([ex[0] for ex in all_examples])
        policies = np.array([ex[1] for ex in all_examples])
        values = np.array([ex[2] for ex in all_examples])

        return states, policies, values


if __name__ == "__main__":
    print("Self-play module loaded successfully")
    # Full testing requires all components
