"""
Evaluation module for testing trained agents.

Provides utilities to:
1. Play against baseline bots
2. Calculate win rates
3. Analyze learned strategies
"""
from typing import Dict, List

import numpy as np


class Evaluator:
    """
    Evaluates trained AlphaZero agents against baselines.
    """

    def __init__(
        self,
        game_wrapper_class,
        action_encoder,
        num_players: int = 4,
    ):
        self.game_wrapper_class = game_wrapper_class
        self.action_encoder = action_encoder
        self.num_players = num_players

    def play_game(
        self,
        agents: List,  # List of (agent, use_mcts, mcts_sims) tuples
        verbose: bool = False,
    ) -> Dict:
        """
        Play a single evaluation game.

        Args:
            agents: List of agents for each player position
                    Each agent should have a select_action(game, valid_actions) method
            verbose: Print game progress

        Returns:
            Dict with game results
        """
        game = self.game_wrapper_class(num_players=self.num_players)

        move_count = 0
        done = False

        while not done:
            current_player = game.get_current_player()
            valid_actions = game.get_valid_actions()

            if not valid_actions:
                break

            # Get action from appropriate agent
            agent = agents[current_player]
            action = agent.select_action(game, valid_actions)

            # Execute action
            _, _, done, info = game.step(action)
            move_count += 1

            if verbose and move_count % 20 == 0:
                print(f"Move {move_count}, Player {current_player}")

            if move_count > 1000:
                break

        winner_color = game.game.winning_color()
        winner = game.colors.index(winner_color) if winner_color else -1

        return {
            "winner": winner,
            "moves": move_count,
            "final_vps": self._get_victory_points(game),
        }

    def _get_victory_points(self, game) -> List[int]:
        """Get victory points for each player."""
        vps = []
        state = game.game.state
        for color in game.colors:
            vp = state.player_state.get(f"P{color.value}_VICTORY_POINTS", 0)
            vps.append(vp)
        return vps

    def evaluate(
        self,
        agents: List,
        num_games: int = 100,
        verbose: bool = True,
    ) -> Dict:
        """
        Evaluate agents over multiple games.

        Returns:
            Dict with win rates and statistics
        """
        results = {
            "wins": [0] * len(agents),
            "total_vps": [0] * len(agents),
            "games_played": 0,
        }

        for i in range(num_games):
            if verbose and (i + 1) % 10 == 0:
                print(f"Game {i + 1}/{num_games}...")

            game_result = self.play_game(agents, verbose=False)

            if game_result["winner"] >= 0:
                results["wins"][game_result["winner"]] += 1

            for j, vp in enumerate(game_result["final_vps"]):
                results["total_vps"][j] += vp

            results["games_played"] += 1

        # Calculate statistics
        results["win_rates"] = [
            w / num_games for w in results["wins"]
        ]
        results["avg_vps"] = [
            vp / num_games for vp in results["total_vps"]
        ]

        if verbose:
            print("\nEvaluation Results:")
            print("-" * 40)
            for i, (wr, avg_vp) in enumerate(zip(results["win_rates"], results["avg_vps"])):
                print(f"Player {i}: Win Rate={wr:.1%}, Avg VP={avg_vp:.1f}")

        return results


class RandomAgent:
    """Baseline agent that plays random valid actions."""

    def select_action(self, game, valid_actions):
        return np.random.choice(valid_actions)


class GreedyAgent:
    """Agent that prefers building actions over others."""

    def __init__(self, action_encoder):
        self.action_encoder = action_encoder

    def select_action(self, game, valid_actions):
        # Prioritize: cities > settlements > roads > dev cards > other
        from catanatron.models.enums import ActionType

        priorities = {
            ActionType.BUILD_CITY: 5,
            ActionType.BUILD_SETTLEMENT: 4,
            ActionType.BUY_DEVELOPMENT_CARD: 3,
            ActionType.BUILD_ROAD: 2,
        }

        best_action = None
        best_priority = -1

        for action in valid_actions:
            priority = priorities.get(action.action_type, 0)
            if priority > best_priority:
                best_priority = priority
                best_action = action

        return best_action if best_action else np.random.choice(valid_actions)


class AlphaZeroAgent:
    """Agent that uses the trained network with optional MCTS."""

    def __init__(
        self,
        network,
        action_encoder,
        mcts_class=None,
        num_simulations: int = 50,
        use_mcts: bool = True,
    ):
        self.network = network
        self.action_encoder = action_encoder
        self.mcts_class = mcts_class
        self.num_simulations = num_simulations
        self.use_mcts = use_mcts

    def select_action(self, game, valid_actions):
        if self.use_mcts and self.mcts_class:
            # Use MCTS for action selection
            mcts = self.mcts_class(
                network=self.network,
                action_encoder=self.action_encoder,
                num_simulations=self.num_simulations,
            )
            action, _ = mcts.search(game, temperature=0)
            return action
        else:
            # Use network policy directly
            state = game.get_state()
            valid_mask = self.action_encoder.get_valid_action_mask(valid_actions)
            policy, _ = self.network.predict(state, valid_mask)

            # Select action with highest probability
            best_idx = None
            best_prob = -1

            for action in valid_actions:
                idx = self.action_encoder.encode(action)
                if idx < len(policy) and policy[idx] > best_prob:
                    best_prob = policy[idx]
                    best_idx = action

            return best_idx if best_idx else valid_actions[0]


def run_evaluation(
    network,
    action_encoder,
    game_wrapper_class,
    mcts_class,
    num_games: int = 50,
):
    """
    Run a standard evaluation: AlphaZero vs Random vs Greedy.
    """
    evaluator = Evaluator(
        game_wrapper_class=game_wrapper_class,
        action_encoder=action_encoder,
        num_players=4,
    )

    # Create agents
    alphazero = AlphaZeroAgent(
        network=network,
        action_encoder=action_encoder,
        mcts_class=mcts_class,
        num_simulations=50,
        use_mcts=True,
    )

    random_agent = RandomAgent()
    greedy_agent = GreedyAgent(action_encoder)

    # AlphaZero vs 3 random agents
    print("\n" + "="*50)
    print("AlphaZero vs 3 Random Agents")
    print("="*50)

    agents = [alphazero, random_agent, random_agent, random_agent]
    results = evaluator.evaluate(agents, num_games=num_games)

    print(f"\nAlphaZero win rate: {results['win_rates'][0]:.1%}")

    return results


if __name__ == "__main__":
    print("Evaluation module loaded successfully")
