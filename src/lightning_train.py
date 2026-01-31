"""
LIGHTNING TRAIN - Fast training with proper action encoding
============================================================
Combines speed of hyperspeed with proper learning signal from turbo_train
Target: 10+ games/sec with actual learning
"""

import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from catanatron.game import Game
from catanatron.models.enums import ActionType
from catanatron.models.player import Color


def get_player_key(game, player_idx):
    """Get the state key prefix for a player."""
    color = game.state.colors[player_idx]
    return f"P{color.value}_"


class LightningStateEncoder:
    """Balanced state encoding - enough info to learn, not too much to slow down."""

    def __init__(self):
        # Per player (4 players):
        #   - 5 resources
        #   - 3 building counts
        #   - 1 VP
        #   - 1 dev cards
        # = 10 features per player = 40
        # Plus 5 global features = 45 total
        self.state_size = 45

    def encode(self, game):
        """Encode game state."""
        state = game.state
        features = np.zeros(self.state_size, dtype=np.float32)

        idx = 0
        for p in range(4):
            key = get_player_key(game, p)

            # Resources (5)
            features[idx] = min(state.player_state.get(f"{key}WOOD_IN_HAND", 0), 15) / 15
            features[idx+1] = min(state.player_state.get(f"{key}BRICK_IN_HAND", 0), 15) / 15
            features[idx+2] = min(state.player_state.get(f"{key}SHEEP_IN_HAND", 0), 15) / 15
            features[idx+3] = min(state.player_state.get(f"{key}WHEAT_IN_HAND", 0), 15) / 15
            features[idx+4] = min(state.player_state.get(f"{key}ORE_IN_HAND", 0), 15) / 15
            idx += 5

            # Buildings available (3) - inverted = how many built
            features[idx] = 1.0 - state.player_state.get(f"{key}SETTLEMENTS_AVAILABLE", 5) / 5
            features[idx+1] = 1.0 - state.player_state.get(f"{key}CITIES_AVAILABLE", 4) / 4
            features[idx+2] = 1.0 - state.player_state.get(f"{key}ROADS_AVAILABLE", 15) / 15
            idx += 3

            # VP (1)
            features[idx] = min(state.player_state.get(f"{key}ACTUAL_VICTORY_POINTS", 0), 10) / 10
            idx += 1

            # Dev cards (1)
            dev_cards = sum([
                state.player_state.get(f"{key}KNIGHT_IN_HAND", 0),
                state.player_state.get(f"{key}YEAR_OF_PLENTY_IN_HAND", 0),
                state.player_state.get(f"{key}ROAD_BUILDING_IN_HAND", 0),
                state.player_state.get(f"{key}MONOPOLY_IN_HAND", 0),
            ])
            features[idx] = min(dev_cards, 10) / 10
            idx += 1

        # Global features (5)
        features[40] = state.current_player_index / 3  # Whose turn
        features[41] = len(state.playable_actions) / 50  # Action complexity
        features[42] = 1.0 if state.is_initial_build_phase else 0.0
        features[43] = state.num_turns / 200  # Game progress
        features[44] = len(state.actions) / 1000  # Total actions taken

        return features


class LightningActionEncoder:
    """Proper action encoding with no collisions."""

    def __init__(self):
        # Action categories with proper indexing:
        # 0: ROLL
        # 1: END_TURN
        # 2: BUY_DEVELOPMENT_CARD
        # 3-56: BUILD_SETTLEMENT (54 nodes)
        # 57-128: BUILD_ROAD (72 edges)
        # 129-182: BUILD_CITY (54 nodes)
        # 183-201: MOVE_ROBBER (19 tiles)
        # 202-255: MARITIME_TRADE (54 combos: 6 resources x 9 possible trades)
        # 256-260: PLAY dev cards (5 types)
        # 261-279: DISCARD combinations (simplified)
        # 280+: Other actions
        self.num_actions = 300

        # Pre-compute action type offsets
        self.offsets = {
            ActionType.ROLL: 0,
            ActionType.END_TURN: 1,
            ActionType.BUY_DEVELOPMENT_CARD: 2,
            ActionType.BUILD_SETTLEMENT: 3,
            ActionType.BUILD_ROAD: 57,
            ActionType.BUILD_CITY: 129,
            ActionType.MOVE_ROBBER: 183,
            ActionType.MARITIME_TRADE: 202,
        }

        # Build node index map
        self._node_map = {}
        self._edge_map = {}
        self._tile_map = {}
        self._node_counter = 0
        self._edge_counter = 0
        self._tile_counter = 0

    def _get_node_idx(self, node_id):
        """Get consistent index for a node."""
        if node_id not in self._node_map:
            self._node_map[node_id] = self._node_counter % 54
            self._node_counter += 1
        return self._node_map[node_id]

    def _get_edge_idx(self, edge_id):
        """Get consistent index for an edge."""
        if edge_id not in self._edge_map:
            self._edge_map[edge_id] = self._edge_counter % 72
            self._edge_counter += 1
        return self._edge_map[edge_id]

    def _get_tile_idx(self, tile_id):
        """Get consistent index for a tile."""
        if tile_id not in self._tile_map:
            self._tile_map[tile_id] = self._tile_counter % 19
            self._tile_counter += 1
        return self._tile_map[tile_id]

    def encode(self, action):
        """Encode action to index."""
        action_type = action.action_type

        # Simple actions
        if action_type == ActionType.ROLL:
            return 0
        elif action_type == ActionType.END_TURN:
            return 1
        elif action_type == ActionType.BUY_DEVELOPMENT_CARD:
            return 2

        # Position-based actions
        elif action_type == ActionType.BUILD_SETTLEMENT:
            return 3 + self._get_node_idx(action.value)
        elif action_type == ActionType.BUILD_ROAD:
            return 57 + self._get_edge_idx(action.value)
        elif action_type == ActionType.BUILD_CITY:
            return 129 + self._get_node_idx(action.value)
        elif action_type == ActionType.MOVE_ROBBER:
            if action.value is not None:
                coord = action.value[0] if isinstance(action.value, tuple) else action.value
                return 183 + self._get_tile_idx(coord)
            return 183

        # Maritime trade
        elif action_type == ActionType.MARITIME_TRADE:
            return 202 + (hash(str(action.value)) % 54)

        # Dev card plays
        elif action_type == ActionType.PLAY_KNIGHT_CARD:
            return 256
        elif action_type == ActionType.PLAY_YEAR_OF_PLENTY:
            return 257
        elif action_type == ActionType.PLAY_ROAD_BUILDING:
            return 258
        elif action_type == ActionType.PLAY_MONOPOLY:
            return 259

        # Discard
        elif action_type == ActionType.DISCARD:
            return 261 + (hash(str(action.value)) % 19)

        # Catch-all for other actions
        return 280 + (hash(str(action)) % 20)


class LightningNetwork(nn.Module):
    """Small but capable network."""

    def __init__(self, state_size=45, num_actions=300, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden // 2)
        self.policy = nn.Linear(hidden // 2, num_actions)
        self.value = nn.Linear(hidden // 2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return F.log_softmax(self.policy(x), dim=-1), torch.tanh(self.value(x))


class LightningPlayer:
    """Fast player with smart action selection."""

    def __init__(self, color, network, state_encoder, action_encoder, use_network_prob=0.2):
        self.color = color
        self.network = network
        self.state_encoder = state_encoder
        self.action_encoder = action_encoder
        self.use_network_prob = use_network_prob

    def decide(self, game, playable_actions):
        if not playable_actions:
            return None
        if len(playable_actions) == 1:
            return playable_actions[0]

        # Use network with some probability
        if random.random() < self.use_network_prob and self.network is not None:
            state = self.state_encoder.encode(game)
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

            with torch.no_grad():
                log_probs, _ = self.network(state_tensor)
                probs = torch.exp(log_probs).squeeze().numpy()

            # Score actions and add noise for exploration
            scores = []
            for action in playable_actions:
                action_idx = self.action_encoder.encode(action)
                score = probs[action_idx] + np.random.exponential(0.1)
                scores.append(score)

            # Pick best
            return playable_actions[np.argmax(scores)]
        else:
            # Smart random: prefer building actions
            building_actions = [a for a in playable_actions
                              if a.action_type in (ActionType.BUILD_SETTLEMENT,
                                                   ActionType.BUILD_ROAD,
                                                   ActionType.BUILD_CITY,
                                                   ActionType.BUY_DEVELOPMENT_CARD)]
            if building_actions and random.random() < 0.5:
                return random.choice(building_actions)
            return random.choice(playable_actions)


def play_lightning_game(network, state_encoder, action_encoder, use_network_prob=0.2):
    """Play one game and collect training data."""

    players = [
        LightningPlayer(Color.RED, network, state_encoder, action_encoder, use_network_prob),
        LightningPlayer(Color.BLUE, network, state_encoder, action_encoder, use_network_prob),
        LightningPlayer(Color.WHITE, network, state_encoder, action_encoder, use_network_prob),
        LightningPlayer(Color.ORANGE, network, state_encoder, action_encoder, use_network_prob),
    ]

    game = Game(players)
    examples = []

    while game.winning_color() is None:
        current_idx = game.state.current_player_index
        playable = game.state.playable_actions

        if not playable:
            break

        # Record state before action
        state = state_encoder.encode(game)

        # Get action
        action = players[current_idx].decide(game, playable)
        if action is None:
            break

        action_idx = action_encoder.encode(action)

        # Create soft policy target based on available actions
        policy = np.zeros(action_encoder.num_actions, dtype=np.float32)
        # Give some mass to all legal actions, more to chosen one
        for a in playable:
            policy[action_encoder.encode(a)] = 0.1 / len(playable)
        policy[action_idx] = 0.9  # Most mass on chosen action
        policy = policy / policy.sum()  # Normalize

        examples.append({
            'state': state,
            'policy': policy,
            'player_idx': current_idx
        })

        # Execute
        game.execute(action)

    # Assign values based on winner
    winner = game.winning_color()
    if winner:
        winner_idx = game.state.colors.index(winner)
        for ex in examples:
            if ex['player_idx'] == winner_idx:
                ex['value'] = 1.0
            else:
                ex['value'] = -0.5  # Softer penalty for losers
    else:
        for ex in examples:
            ex['value'] = 0.0

    return examples


def generate_games(network, num_games, state_encoder, action_encoder, use_network_prob=0.2):
    """Generate multiple games."""
    all_examples = []
    for _ in range(num_games):
        examples = play_lightning_game(network, state_encoder, action_encoder, use_network_prob)
        all_examples.extend(examples)
    return all_examples


class LightningTrainer:
    """Fast training with proper learning."""

    def __init__(self, network, lr=0.001, buffer_size=100000):
        self.network = network
        self.optimizer = torch.optim.Adam(network.parameters(), lr=lr)
        self.buffer = deque(maxlen=buffer_size)

    def add_examples(self, examples):
        self.buffer.extend(examples)

    def train_epoch(self, batch_size=256, num_batches=None):
        """Train for one epoch."""
        if len(self.buffer) < batch_size:
            return 0.0

        if num_batches is None:
            num_batches = max(1, len(self.buffer) // batch_size)

        total_loss = 0.0
        for _ in range(num_batches):
            batch = random.sample(list(self.buffer), batch_size)

            states = torch.tensor(np.array([ex['state'] for ex in batch]), dtype=torch.float32)
            policies = torch.tensor(np.array([ex['policy'] for ex in batch]), dtype=torch.float32)
            values = torch.tensor(np.array([[ex['value']] for ex in batch]), dtype=torch.float32)

            # Forward
            log_probs, pred_values = self.network(states)

            # Loss - KL divergence for policy + MSE for value
            policy_loss = F.kl_div(log_probs, policies, reduction='batchmean')
            value_loss = F.mse_loss(pred_values, values)
            loss = policy_loss + 0.5 * value_loss

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / num_batches


def lightning_train(num_iterations=10, games_per_iter=200, epochs_per_iter=5):
    """Main training loop."""

    print("=" * 70)
    print("  ⚡ LIGHTNING TRAINING MODE ⚡")
    print("=" * 70)

    # Initialize
    state_encoder = LightningStateEncoder()
    action_encoder = LightningActionEncoder()
    network = LightningNetwork(state_encoder.state_size, action_encoder.num_actions)
    trainer = LightningTrainer(network)

    total_games = 0
    total_time = 0

    for iteration in range(1, num_iterations + 1):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration}/{num_iterations}")
        print(f"{'='*60}")

        # Increase network usage as training progresses
        network_prob = min(0.4, 0.1 + iteration * 0.03)

        # Generate games
        print(f"\nGenerating {games_per_iter} games (network prob: {network_prob:.2f})...")
        start = time.time()
        examples = generate_games(network, games_per_iter, state_encoder, action_encoder, network_prob)
        gen_time = time.time() - start

        games_per_sec = games_per_iter / gen_time
        moves_per_game = len(examples) / games_per_iter
        total_games += games_per_iter
        total_time += gen_time

        print(f"Generated {len(examples)} examples in {gen_time:.1f}s")
        print(f"Speed: {games_per_sec:.1f} games/sec | Avg moves/game: {moves_per_game:.0f}")

        # Train
        trainer.add_examples(examples)

        print(f"\nTraining on {len(trainer.buffer)} examples...")
        for epoch in range(1, epochs_per_iter + 1):
            loss = trainer.train_epoch(batch_size=256)
            print(f"  Epoch {epoch}: Loss = {loss:.4f}")

    # Final summary
    print(f"\n{'='*70}")
    print("⚡ TRAINING COMPLETE ⚡")
    print(f"{'='*70}")
    print(f"Total games: {total_games}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Average speed: {total_games/total_time:.1f} games/sec")

    # Save
    import os
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        'network': network.state_dict(),
        'state_size': state_encoder.state_size,
        'num_actions': action_encoder.num_actions,
    }, "checkpoints/lightning_model.pt")
    print("\nModel saved to checkpoints/lightning_model.pt")

    return network, state_encoder, action_encoder


def evaluate_model(network, state_encoder, action_encoder, num_games=50):
    """Evaluate trained model vs random players."""

    print(f"\n{'='*60}")
    print("  EVALUATING MODEL")
    print(f"{'='*60}")

    wins = {0: 0, 1: 0, 2: 0, 3: 0}

    for game_num in range(num_games):
        # Player 0 uses network 100%, others play random
        players = [
            LightningPlayer(Color.RED, network, state_encoder, action_encoder, use_network_prob=1.0),
            LightningPlayer(Color.BLUE, None, state_encoder, action_encoder, use_network_prob=0.0),
            LightningPlayer(Color.WHITE, None, state_encoder, action_encoder, use_network_prob=0.0),
            LightningPlayer(Color.ORANGE, None, state_encoder, action_encoder, use_network_prob=0.0),
        ]

        game = Game(players)

        while game.winning_color() is None:
            current_idx = game.state.current_player_index
            playable = game.state.playable_actions
            if not playable:
                break
            action = players[current_idx].decide(game, playable)
            if action is None:
                break
            game.execute(action)

        winner = game.winning_color()
        if winner:
            winner_idx = game.state.colors.index(winner)
            wins[winner_idx] += 1

        if (game_num + 1) % 10 == 0:
            print(f"  Game {game_num + 1}/{num_games}: Trained model wins: {wins[0]}")

    print(f"\nResults over {num_games} games:")
    print(f"  Trained model (P0): {wins[0]} wins ({100*wins[0]/num_games:.1f}%)")
    print(f"  Random players: {wins[1]+wins[2]+wins[3]} wins")

    return wins[0] / num_games


if __name__ == "__main__":
    # Train
    network, state_encoder, action_encoder = lightning_train(
        num_iterations=15,
        games_per_iter=300,
        epochs_per_iter=5
    )

    # Evaluate
    evaluate_model(network, state_encoder, action_encoder, num_games=100)
