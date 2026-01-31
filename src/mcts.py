"""
Monte Carlo Tree Search for AlphaZero Catan.

Optimized implementation that minimizes expensive game cloning.
Uses network policy with UCB exploration.
"""
import numpy as np


class MCTS:
    """
    Fast MCTS with neural network guidance.
    
    Key optimization: Only clones game once per unique action evaluated,
    caches results to avoid redundant cloning.
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

    def search(self, game_wrapper, temperature: float = 1.0) -> tuple:
        """
        Perform MCTS search from current game state.
        
        Optimized: evaluates each action at most once via cloning,
        caches value estimates for subsequent simulations.

        Args:
            game_wrapper: CatanGameWrapper with current game state
            temperature: Controls exploration (higher = more random)

        Returns:
            action: Selected action
            action_probs: Visit count distribution over actions
        """
        valid_actions = game_wrapper.get_valid_actions()
        if not valid_actions:
            return None, np.array([])
        
        num_actions = len(valid_actions)
        current_player = game_wrapper.get_current_player()
        
        # Get network policy for current state
        state = game_wrapper.get_state()
        valid_mask = self.action_encoder.get_valid_action_mask(valid_actions, game_wrapper.game)
        policy, _ = self.network.predict(state, valid_mask)
        
        # Extract priors for valid actions
        priors = np.zeros(num_actions)
        for i, action in enumerate(valid_actions):
            idx = self.action_encoder.encode(action, game_wrapper.game)
            priors[i] = policy[idx] if idx < len(policy) else 1.0 / num_actions
        
        # Normalize priors
        prior_sum = priors.sum()
        if prior_sum > 0:
            priors = priors / prior_sum
        else:
            priors = np.ones(num_actions) / num_actions
        
        # Add Dirichlet noise for exploration
        noise = np.random.dirichlet([self.dirichlet_alpha] * num_actions)
        priors = (1 - self.exploration_fraction) * priors + self.exploration_fraction * noise
        
        # Track statistics
        visit_counts = np.zeros(num_actions)
        value_sums = np.zeros(num_actions)
        value_cache = {}  # Cache evaluated values to avoid re-cloning
        
        # Run simulations
        for _ in range(self.num_simulations):
            # UCB selection
            total_visits = visit_counts.sum() + 1
            
            ucb_scores = np.zeros(num_actions)
            for i in range(num_actions):
                if visit_counts[i] == 0:
                    ucb_scores[i] = float('inf')  # Prioritize unvisited
                else:
                    q_value = value_sums[i] / visit_counts[i]
                    exploration = self.c_puct * priors[i] * np.sqrt(total_visits) / (1 + visit_counts[i])
                    ucb_scores[i] = q_value + exploration
            
            action_idx = np.argmax(ucb_scores)
            
            # Get value (from cache or evaluate)
            if action_idx in value_cache:
                value = value_cache[action_idx]
            else:
                # Clone and evaluate ONCE per action
                sim_game = game_wrapper.clone()
                try:
                    sim_game.step(valid_actions[action_idx])
                    
                    if sim_game.game.winning_color() is not None:
                        winner = sim_game.game.winning_color()
                        winner_idx = sim_game.colors.index(winner)
                        value = 1.0 if winner_idx == current_player else -1.0
                    else:
                        _, value = self.network.predict(sim_game.get_state())
                        # Negate if opponent's turn
                        if sim_game.get_current_player() != current_player:
                            value = -value
                except ValueError:
                    value = -1.0
                
                value_cache[action_idx] = value
            
            visit_counts[action_idx] += 1
            value_sums[action_idx] += value
        
        # Select action
        if temperature == 0:
            action_idx = np.argmax(visit_counts)
            action_probs = np.zeros(num_actions)
            action_probs[action_idx] = 1.0
        else:
            counts_temp = visit_counts ** (1 / temperature)
            action_probs = counts_temp / (counts_temp.sum() + 1e-8)
            action_idx = np.random.choice(num_actions, p=action_probs)
        
        return valid_actions[action_idx], action_probs


if __name__ == "__main__":
    print("MCTS module loaded successfully")
