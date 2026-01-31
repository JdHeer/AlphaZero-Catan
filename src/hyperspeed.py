"""
HYPERSPEED TRAINING MODULE
===========================
Goal: Absolute maximum training speed - targeting 20+ games/sec

Key optimizations:
1. Minimal state encoding (only essential features)
2. Network called only 10% of the time
3. Tiny network (64 hidden units)
4. No game copying ever - just record transitions
5. Batch everything
6. Use greedy action selection (no sampling)
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


class HyperStateEncoder:
    """Ultra-minimal state encoding - just the essentials."""

    def __init__(self):
        # 4 players x 5 resources = 20
        # 4 players x 2 buildings (settlements, cities) = 8
        # 4 players x 1 (num roads) = 4
        # 4 players x 1 VP = 4
        # Current turn phase = 1
        # Total: 37 features
        self.state_size = 37

    def encode(self, game):
        """Encode game state as minimal feature vector."""
        state = game.state
        features = np.zeros(37, dtype=np.float32)

        idx = 0
        for p in range(4):
            key = get_player_key(game, p)

            # Resources (5) - capped at 10
            features[idx] = min(state.player_state.get(f"{key}WOOD_IN_HAND", 0), 10) / 10
            features[idx+1] = min(state.player_state.get(f"{key}BRICK_IN_HAND", 0), 10) / 10
            features[idx+2] = min(state.player_state.get(f"{key}SHEEP_IN_HAND", 0), 10) / 10
            features[idx+3] = min(state.player_state.get(f"{key}WHEAT_IN_HAND", 0), 10) / 10
            features[idx+4] = min(state.player_state.get(f"{key}ORE_IN_HAND", 0), 10) / 10
            idx += 5

            # Buildings (2)
            features[idx] = state.player_state.get(f"{key}SETTLEMENTS_AVAILABLE", 5) / 5
            features[idx+1] = state.player_state.get(f"{key}CITIES_AVAILABLE", 4) / 4
            idx += 2

            # Roads (1)
            features[idx] = state.player_state.get(f"{key}ROADS_AVAILABLE", 15) / 15
            idx += 1

            # VP (1)
            features[idx] = min(state.player_state.get(f"{key}ACTUAL_VICTORY_POINTS", 0), 10) / 10
            idx += 1

        # Turn phase
        features[36] = 0.5  # Simplified

        return features


class HyperActionEncoder:
    """Ultra-simple action encoding."""

    def __init__(self):
        # Just use action type as category
        # ROLL, BUY_DEVELOPMENT_CARD, BUILD_SETTLEMENT, BUILD_ROAD, BUILD_CITY,
        # MARITIME_TRADE, DISCARD, MOVE_ROBBER, PLAY_KNIGHT_CARD, END_TURN, etc.
        # ~100 action types with simple position encoding
        self.num_actions = 100

    def encode(self, action):
        """Encode action as simple index."""
        action_type = action.action_type

        # Base action type mapping
        type_map = {
            ActionType.ROLL: 0,
            ActionType.BUY_DEVELOPMENT_CARD: 1,
            ActionType.END_TURN: 2,
        }

        if action_type in type_map:
            return type_map[action_type]

        # For position-based actions, use hash
        if action.value is not None:
            h = hash(str(action.value)) % 97  # 97 slots for position-based
            return 3 + h

        return hash(str(action_type)) % 100


class HyperNetwork(nn.Module):
    """Tiny but effective network."""

    def __init__(self, state_size=37, num_actions=100, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.policy = nn.Linear(hidden, num_actions)
        self.value = nn.Linear(hidden, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.policy(x), dim=-1), torch.tanh(self.value(x))


class HyperPlayer:
    """Hyper-fast player that only occasionally uses the network."""

    def __init__(self, color, network, state_encoder, action_encoder, network_prob=0.1):
        self.color = color
        self.network = network
        self.state_encoder = state_encoder
        self.action_encoder = action_encoder
        self.network_prob = network_prob
        self.cached_probs = None

    def decide(self, game, playable_actions):
        if not playable_actions:
            return None
        if len(playable_actions) == 1:
            return playable_actions[0]

        # Only use network occasionally
        if random.random() < self.network_prob and self.network is not None:
            state = self.state_encoder.encode(game)
            state_tensor = torch.tensor(state).unsqueeze(0)

            with torch.no_grad():
                log_probs, _ = self.network(state_tensor)
                probs = torch.exp(log_probs).squeeze().numpy()

            # Score actions
            scores = []
            for action in playable_actions:
                action_idx = self.action_encoder.encode(action)
                scores.append(probs[action_idx])

            # Greedy selection (fastest)
            best_idx = np.argmax(scores)
            return playable_actions[best_idx]
        else:
            # Random selection (super fast)
            return random.choice(playable_actions)


def play_hyper_game(network, state_encoder, action_encoder, network_prob=0.1):
    """Play a single game at hyperspeed and collect training data."""

    players = [
        HyperPlayer(Color.RED, network, state_encoder, action_encoder, network_prob),
        HyperPlayer(Color.BLUE, network, state_encoder, action_encoder, network_prob),
        HyperPlayer(Color.WHITE, network, state_encoder, action_encoder, network_prob),
        HyperPlayer(Color.ORANGE, network, state_encoder, action_encoder, network_prob),
    ]

    game = Game(players)
    examples = []

    while game.winning_color() is None:
        current_idx = game.state.current_player_index
        playable = game.state.playable_actions

        if not playable:
            break

        # Record state
        state = state_encoder.encode(game)

        # Get action
        action = players[current_idx].decide(game, playable)
        if action is None:
            break

        action_idx = action_encoder.encode(action)

        # Create policy target (one-hot)
        policy = np.zeros(action_encoder.num_actions, dtype=np.float32)
        policy[action_idx] = 1.0

        examples.append({
            'state': state,
            'policy': policy,
            'player_idx': current_idx
        })

        # Execute action
        game.execute(action)

    # Get winner and assign rewards
    winner = game.winning_color()
    if winner:
        winner_idx = game.state.colors.index(winner)
        for ex in examples:
            ex['value'] = 1.0 if ex['player_idx'] == winner_idx else -1.0
    else:
        for ex in examples:
            ex['value'] = 0.0

    return examples


def generate_hyper_games(network, num_games, network_prob=0.1):
    """Generate many games at hyperspeed."""
    state_encoder = HyperStateEncoder()
    action_encoder = HyperActionEncoder()

    all_examples = []

    for _ in range(num_games):
        examples = play_hyper_game(network, state_encoder, action_encoder, network_prob)
        all_examples.extend(examples)

    return all_examples


class HyperTrainer:
    """Hyper-fast training loop."""

    def __init__(self, network, lr=0.002, buffer_size=50000):
        self.network = network
        self.optimizer = torch.optim.Adam(network.parameters(), lr=lr)
        self.buffer = deque(maxlen=buffer_size)

    def add_examples(self, examples):
        self.buffer.extend(examples)

    def train_step(self, batch_size=512):
        """Single training step."""
        if len(self.buffer) < batch_size:
            return 0.0

        # Sample batch
        batch = random.sample(list(self.buffer), batch_size)

        states = torch.tensor(np.array([ex['state'] for ex in batch]), dtype=torch.float32)
        policies = torch.tensor(np.array([ex['policy'] for ex in batch]), dtype=torch.float32)
        values = torch.tensor(np.array([[ex['value']] for ex in batch]), dtype=torch.float32)

        # Forward pass
        log_probs, pred_values = self.network(states)

        # Loss
        policy_loss = -torch.mean(torch.sum(policies * log_probs, dim=1))
        value_loss = F.mse_loss(pred_values, values)
        loss = policy_loss + value_loss

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train_epoch(self, batch_size=512):
        """Train for one epoch."""
        num_batches = max(1, len(self.buffer) // batch_size)
        total_loss = 0.0

        for _ in range(num_batches):
            total_loss += self.train_step(batch_size)

        return total_loss / num_batches


def hyperspeed_train(num_iterations=10, games_per_iter=200, epochs_per_iter=3):
    """Main hyperspeed training loop."""

    print("=" * 60)
    print("  HYPERSPEED TRAINING MODE - MAXIMUM VELOCITY")
    print("=" * 60)

    # Initialize
    state_encoder = HyperStateEncoder()
    action_encoder = HyperActionEncoder()
    network = HyperNetwork(state_encoder.state_size, action_encoder.num_actions)
    trainer = HyperTrainer(network)

    total_games = 0
    total_time = 0

    for iteration in range(1, num_iterations + 1):
        print(f"\n{'='*50}")
        print(f"ITERATION {iteration}/{num_iterations}")
        print(f"{'='*50}")

        # Generate games
        print(f"\nGenerating {games_per_iter} games...")
        start = time.time()
        examples = generate_hyper_games(network, games_per_iter, network_prob=0.1)
        gen_time = time.time() - start

        games_per_sec = games_per_iter / gen_time
        total_games += games_per_iter
        total_time += gen_time

        print(f"Generated {len(examples)} examples in {gen_time:.1f}s ({games_per_sec:.1f} games/sec)")

        # Add to buffer and train
        trainer.add_examples(examples)

        print(f"\nTraining on {len(trainer.buffer)} examples...")
        for epoch in range(1, epochs_per_iter + 1):
            loss = trainer.train_epoch(batch_size=512)
            print(f"  Epoch {epoch}: Loss = {loss:.4f}")

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"Total games: {total_games}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Average speed: {total_games/total_time:.1f} games/sec")
    print(f"{'='*60}")

    # Save model
    import os
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(network.state_dict(), "checkpoints/hyper_model.pt")
    print("Model saved to checkpoints/hyper_model.pt")

    return network


def benchmark_hyperspeed(num_games=100):
    """Benchmark pure generation speed without training."""
    print("\n" + "="*60)
    print("  BENCHMARKING HYPERSPEED GENERATION")
    print("="*60)

    state_encoder = HyperStateEncoder()
    action_encoder = HyperActionEncoder()

    # Test with network
    network = HyperNetwork()

    print(f"\nGenerating {num_games} games with network (10% calls)...")
    start = time.time()
    examples = generate_hyper_games(network, num_games, network_prob=0.1)
    elapsed = time.time() - start
    print(f"Speed: {num_games/elapsed:.1f} games/sec, {len(examples)/elapsed:.0f} examples/sec")

    # Test pure random (network_prob=0)
    print(f"\nGenerating {num_games} games pure random...")
    start = time.time()
    examples = generate_hyper_games(network, num_games, network_prob=0.0)
    elapsed = time.time() - start
    print(f"Speed: {num_games/elapsed:.1f} games/sec, {len(examples)/elapsed:.0f} examples/sec")


if __name__ == "__main__":
    # Run benchmark first
    benchmark_hyperspeed(100)

    # Then train
    hyperspeed_train(
        num_iterations=20,
        games_per_iter=500,
        epochs_per_iter=3
    )
