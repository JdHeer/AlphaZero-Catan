"""
Monte Carlo Tree Search for AlphaZero Catan.

MCTS uses the neural network to guide search, balancing
exploration and exploitation via UCB formula.
"""
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class MCTSNode:
    """A node in the MCTS tree."""

    state: np.ndarray
    parent: Optional["MCTSNode"] = None
    action: any = None  # Action that led to this node
    player: int = 0  # Player who will move from this node

    # Statistics
    visit_count: int = 0
    value_sum: float = 0.0
    prior: float = 0.0  # Prior probability from policy network

    # Children
    children: dict = field(default_factory=dict)  # action -> MCTSNode
    is_expanded: bool = False
    is_terminal: bool = False
    terminal_value: float = 0.0

    @property
    def value(self) -> float:
        """Average value of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    @property
    def ucb_score(self) -> float:
        """UCB score for selection."""
        if self.parent is None:
            return 0.0

        c_puct = 1.5  # Exploration constant

        # UCB formula from AlphaZero
        exploration = c_puct * self.prior * math.sqrt(self.parent.visit_count) / (1 + self.visit_count)

        # Value from perspective of parent's player
        # (negate if different players, assuming 2-player zero-sum)
        value = self.value

        return value + exploration


class MCTS:
    """
    Monte Carlo Tree Search with neural network guidance.
    """

    def __init__(
        self,
        network,
        action_encoder,
        num_simulations: int = 100,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        exploration_fraction: float = 0.25,
    ):
        self.network = network
        self.action_encoder = action_encoder
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.exploration_fraction = exploration_fraction

    def search(self, game_wrapper, temperature: float = 1.0) -> tuple[any, np.ndarray]:
        """
        Perform MCTS search from current game state.

        Args:
            game_wrapper: CatanGameWrapper with current game state
            temperature: Controls exploration (higher = more random)

        Returns:
            action: Selected action
            action_probs: Visit count distribution over actions
        """
        # Create root node
        root = MCTSNode(
            state=game_wrapper.get_state(),
            player=game_wrapper.get_current_player(),
        )

        # Get valid actions at root
        valid_actions = game_wrapper.get_valid_actions()
        if not valid_actions:
            return None, np.array([])

        # Expand root with network policy
        self._expand_node(root, game_wrapper, valid_actions)

        # Add Dirichlet noise to root for exploration
        self._add_exploration_noise(root)

        # Run simulations
        for _ in range(self.num_simulations):
            # Clone game for simulation
            sim_game = game_wrapper.clone()
            node = root

            # Selection: traverse tree to leaf
            while node.is_expanded and not node.is_terminal:
                action, node = self._select_child(node)
                if action is not None:
                    sim_game.step(action)

            # Expansion and evaluation
            if not node.is_terminal:
                valid = sim_game.get_valid_actions()
                if valid:
                    self._expand_node(node, sim_game, valid)
                    value = self._evaluate(node, sim_game)
                else:
                    # No valid actions - game might be over
                    winner = sim_game.game.winning_color()
                    node.is_terminal = True
                    if winner:
                        node.terminal_value = 1.0 if winner == sim_game.colors[node.player] else -1.0
                    value = node.terminal_value
            else:
                value = node.terminal_value

            # Backpropagation
            self._backpropagate(node, value)

        # Select action based on visit counts
        action, action_probs = self._select_action(root, valid_actions, temperature)

        return action, action_probs

    def _expand_node(self, node: MCTSNode, game_wrapper, valid_actions: list):
        """Expand node by creating children for all valid actions."""
        # Get policy from network
        state = game_wrapper.get_state()
        valid_mask = self.action_encoder.get_valid_action_mask(valid_actions)
        policy, _ = self.network.predict(state, valid_mask)

        # Create child nodes
        for action in valid_actions:
            action_idx = self.action_encoder.encode(action)
            prior = policy[action_idx] if action_idx < len(policy) else 1.0 / len(valid_actions)

            child = MCTSNode(
                state=None,  # Will be set when visited
                parent=node,
                action=action,
                prior=prior,
            )
            node.children[action] = child

        node.is_expanded = True

    def _select_child(self, node: MCTSNode) -> tuple[any, MCTSNode]:
        """Select child with highest UCB score."""
        best_score = float('-inf')
        best_action = None
        best_child = None

        for action, child in node.children.items():
            score = self._ucb_score(node, child)
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def _ucb_score(self, parent: MCTSNode, child: MCTSNode) -> float:
        """Calculate UCB score for a child node."""
        # Prior probability bonus
        pb = self.c_puct * child.prior * math.sqrt(parent.visit_count) / (1 + child.visit_count)

        # Value (from parent's perspective - may need negation in multiplayer)
        if child.visit_count > 0:
            # In multiplayer, value perspective is complex
            # For now, use direct value
            v = child.value
        else:
            v = 0.0

        return v + pb

    def _evaluate(self, node: MCTSNode, game_wrapper) -> float:
        """Get value estimate from neural network."""
        state = game_wrapper.get_state()
        _, value = self.network.predict(state)
        return value

    def _backpropagate(self, node: MCTSNode, value: float):
        """Backpropagate value up the tree."""
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            # In multiplayer, value needs to be negated when changing players
            # Simplified: assuming value is from perspective of node's player
            node = node.parent
            value = -value  # Flip for opponent (simplified 2-player view)

    def _add_exploration_noise(self, root: MCTSNode):
        """Add Dirichlet noise to root priors for exploration."""
        if not root.children:
            return

        actions = list(root.children.keys())
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))

        for i, action in enumerate(actions):
            child = root.children[action]
            child.prior = (
                (1 - self.exploration_fraction) * child.prior +
                self.exploration_fraction * noise[i]
            )

    def _select_action(
        self,
        root: MCTSNode,
        valid_actions: list,
        temperature: float
    ) -> tuple[any, np.ndarray]:
        """Select action based on visit counts."""
        visit_counts = np.array([
            root.children[a].visit_count if a in root.children else 0
            for a in valid_actions
        ], dtype=np.float64)

        if temperature == 0:
            # Greedy selection
            action_idx = np.argmax(visit_counts)
            probs = np.zeros_like(visit_counts)
            probs[action_idx] = 1.0
        else:
            # Temperature-scaled selection
            visit_counts = visit_counts ** (1 / temperature)
            probs = visit_counts / (visit_counts.sum() + 1e-8)
            action_idx = np.random.choice(len(valid_actions), p=probs)

        return valid_actions[action_idx], probs


if __name__ == "__main__":
    print("MCTS module loaded successfully")
    # Full testing requires the game wrapper and network
