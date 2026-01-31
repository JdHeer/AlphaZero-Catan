"""
SMART LIGHTNING TRAINING
========================
Focus on learning GOOD moves rather than just speed.
Key insight: random play teaches bad habits!

Strategy:
1. Learn from WINNING games only
2. Use simpler state representation focused on what matters
3. Reward building and resource gathering
4. Use policy gradient instead of imitation
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


class SmartStateEncoder:
    """State encoding focused on what matters for winning."""

    def __init__(self):
        # 50 features focused on winning conditions
        self.state_size = 50

    def encode(self, game, player_idx):
        """Encode from perspective of current player."""
        state = game.state
        features = np.zeros(self.state_size, dtype=np.float32)

        key = get_player_key(game, player_idx)

        # My resources (5)
        features[0] = min(state.player_state.get(f"{key}WOOD_IN_HAND", 0), 10) / 10
        features[1] = min(state.player_state.get(f"{key}BRICK_IN_HAND", 0), 10) / 10
        features[2] = min(state.player_state.get(f"{key}SHEEP_IN_HAND", 0), 10) / 10
        features[3] = min(state.player_state.get(f"{key}WHEAT_IN_HAND", 0), 10) / 10
        features[4] = min(state.player_state.get(f"{key}ORE_IN_HAND", 0), 10) / 10

        # My buildings (3)
        my_settlements = 5 - state.player_state.get(f"{key}SETTLEMENTS_AVAILABLE", 5)
        my_cities = 4 - state.player_state.get(f"{key}CITIES_AVAILABLE", 4)
        my_roads = 15 - state.player_state.get(f"{key}ROADS_AVAILABLE", 15)
        features[5] = my_settlements / 5
        features[6] = my_cities / 4
        features[7] = my_roads / 15

        # My VP (1)
        my_vp = state.player_state.get(f"{key}ACTUAL_VICTORY_POINTS", 0)
        features[8] = my_vp / 10

        # My dev cards (1)
        my_dev = sum([
            state.player_state.get(f"{key}KNIGHT_IN_HAND", 0),
            state.player_state.get(f"{key}YEAR_OF_PLENTY_IN_HAND", 0),
            state.player_state.get(f"{key}ROAD_BUILDING_IN_HAND", 0),
            state.player_state.get(f"{key}MONOPOLY_IN_HAND", 0),
        ])
        features[9] = min(my_dev, 10) / 10

        # Can I build stuff? (4)
        wood = state.player_state.get(f"{key}WOOD_IN_HAND", 0)
        brick = state.player_state.get(f"{key}BRICK_IN_HAND", 0)
        sheep = state.player_state.get(f"{key}SHEEP_IN_HAND", 0)
        wheat = state.player_state.get(f"{key}WHEAT_IN_HAND", 0)
        ore = state.player_state.get(f"{key}ORE_IN_HAND", 0)

        features[10] = 1.0 if (wood >= 1 and brick >= 1) else 0.0  # Can build road
        features[11] = 1.0 if (wood >= 1 and brick >= 1 and sheep >= 1 and wheat >= 1) else 0.0  # Settlement
        features[12] = 1.0 if (wheat >= 2 and ore >= 3) else 0.0  # City
        features[13] = 1.0 if (sheep >= 1 and wheat >= 1 and ore >= 1) else 0.0  # Dev card

        # Opponent summary (4 opponents x 4 features = 16)
        idx = 14
        for opp_idx in range(4):
            if opp_idx == player_idx:
                continue
            opp_key = get_player_key(game, opp_idx)

            opp_vp = state.player_state.get(f"{opp_key}ACTUAL_VICTORY_POINTS", 0)
            opp_settlements = 5 - state.player_state.get(f"{opp_key}SETTLEMENTS_AVAILABLE", 5)
            opp_cities = 4 - state.player_state.get(f"{opp_key}CITIES_AVAILABLE", 4)
            opp_roads = 15 - state.player_state.get(f"{opp_key}ROADS_AVAILABLE", 15)

            features[idx] = opp_vp / 10
            features[idx+1] = opp_settlements / 5
            features[idx+2] = opp_cities / 4
            features[idx+3] = opp_roads / 15
            idx += 4

        # Game state (6)
        features[30] = 1.0 if state.is_initial_build_phase else 0.0
        features[31] = len(state.playable_actions) / 30  # Action options
        features[32] = state.num_turns / 200
        features[33] = my_vp / 10  # How close to winning
        features[34] = max(0, (10 - my_vp)) / 10  # How far from winning
        features[35] = 1.0 if my_vp >= 8 else 0.0  # Close to victory

        # Fill remaining with zeros (reserved for future)

        return features


class SmartActionEncoder:
    """Action encoding with categories that make sense."""

    def __init__(self):
        # Categories:
        # 0: ROLL
        # 1: END_TURN
        # 2-55: BUILD_SETTLEMENT (54 positions)
        # 56-127: BUILD_ROAD (72 positions)
        # 128-181: BUILD_CITY (54 positions)
        # 182: BUY_DEVELOPMENT_CARD
        # 183-201: MOVE_ROBBER (19 tiles)
        # 202-255: MARITIME_TRADE (various)
        # 256-259: PLAY dev cards
        # 260-279: DISCARD
        # 280-299: Other
        self.num_actions = 300

        # Position maps built dynamically
        self._node_map = {}
        self._edge_map = {}
        self._node_counter = 0
        self._edge_counter = 0

    def _get_node_idx(self, node_id):
        if node_id not in self._node_map:
            self._node_map[node_id] = self._node_counter % 54
            self._node_counter += 1
        return self._node_map[node_id]

    def _get_edge_idx(self, edge_id):
        if edge_id not in self._edge_map:
            self._edge_map[edge_id] = self._edge_counter % 72
            self._edge_counter += 1
        return self._edge_map[edge_id]

    def encode(self, action):
        at = action.action_type

        if at == ActionType.ROLL:
            return 0
        elif at == ActionType.END_TURN:
            return 1
        elif at == ActionType.BUILD_SETTLEMENT:
            return 2 + self._get_node_idx(action.value)
        elif at == ActionType.BUILD_ROAD:
            return 56 + self._get_edge_idx(action.value)
        elif at == ActionType.BUILD_CITY:
            return 128 + self._get_node_idx(action.value)
        elif at == ActionType.BUY_DEVELOPMENT_CARD:
            return 182
        elif at == ActionType.MOVE_ROBBER:
            return 183 + (hash(str(action.value)) % 19)
        elif at == ActionType.MARITIME_TRADE:
            return 202 + (hash(str(action.value)) % 54)
        elif at == ActionType.PLAY_KNIGHT_CARD:
            return 256
        elif at == ActionType.PLAY_YEAR_OF_PLENTY:
            return 257
        elif at == ActionType.PLAY_ROAD_BUILDING:
            return 258
        elif at == ActionType.PLAY_MONOPOLY:
            return 259
        elif at == ActionType.DISCARD:
            return 260 + (hash(str(action.value)) % 20)
        else:
            return 280 + (hash(str(at)) % 20)


class SmartNetwork(nn.Module):
    """Network with separate heads for action types."""

    def __init__(self, state_size=50, num_actions=300):
        super().__init__()

        # Shared backbone
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)

        # Policy head
        self.policy = nn.Linear(128, num_actions)

        # Value head
        self.value_fc = nn.Linear(128, 64)
        self.value_out = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        policy = F.log_softmax(self.policy(x), dim=-1)
        value = torch.tanh(self.value_out(F.relu(self.value_fc(x))))

        return policy, value


class SmartPlayer:
    """Player that learns good strategies."""

    def __init__(self, color, network, state_encoder, action_encoder, temperature=1.0):
        self.color = color
        self.network = network
        self.state_encoder = state_encoder
        self.action_encoder = action_encoder
        self.temperature = temperature

    def decide(self, game, playable_actions, player_idx):
        if not playable_actions:
            return None
        if len(playable_actions) == 1:
            return playable_actions[0]

        if self.network is None:
            return random.choice(playable_actions)

        # Encode state from current player's perspective
        state = self.state_encoder.encode(game, player_idx)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            log_probs, _ = self.network(state_tensor)
            probs = torch.exp(log_probs).squeeze().numpy()

        # Get scores for legal actions
        action_scores = []
        for action in playable_actions:
            action_idx = self.action_encoder.encode(action)
            action_scores.append(probs[action_idx])

        # Apply temperature
        action_scores = np.array(action_scores)
        if self.temperature > 0:
            action_scores = np.power(action_scores, 1.0 / self.temperature)
            action_scores = action_scores / action_scores.sum()

            # Sample
            try:
                choice = np.random.choice(len(playable_actions), p=action_scores)
            except:
                choice = np.argmax(action_scores)
        else:
            choice = np.argmax(action_scores)

        return playable_actions[choice]


def play_smart_game(network, state_encoder, action_encoder, temperature=1.0):
    """Play a game with all smart players."""

    players = [
        SmartPlayer(Color.RED, network, state_encoder, action_encoder, temperature),
        SmartPlayer(Color.BLUE, network, state_encoder, action_encoder, temperature),
        SmartPlayer(Color.WHITE, network, state_encoder, action_encoder, temperature),
        SmartPlayer(Color.ORANGE, network, state_encoder, action_encoder, temperature),
    ]

    game = Game(players)

    # Track states, actions, and rewards per player
    player_data = {i: [] for i in range(4)}

    while game.winning_color() is None:
        current_idx = game.state.current_player_index
        playable = game.state.playable_actions

        if not playable:
            break

        # Record state
        state = state_encoder.encode(game, current_idx)

        # Get action
        action = players[current_idx].decide(game, playable, current_idx)
        if action is None:
            break

        action_idx = action_encoder.encode(action)

        # Create policy vector for this action
        policy = np.zeros(action_encoder.num_actions, dtype=np.float32)
        policy[action_idx] = 1.0

        player_data[current_idx].append({
            'state': state,
            'policy': policy,
            'action_idx': action_idx
        })

        game.execute(action)

    # Determine winner and assign values
    winner = game.winning_color()
    all_examples = []

    if winner:
        winner_idx = game.state.colors.index(winner)

        # Only learn from winner's moves (key insight!)
        for data in player_data[winner_idx]:
            data['value'] = 1.0
            all_examples.append(data)

        # Optionally, learn from losers with negative reward (but less)
        for player_idx in range(4):
            if player_idx != winner_idx:
                # Only keep last few moves of losers
                loser_data = player_data[player_idx][-10:]
                for data in loser_data:
                    data['value'] = -0.5
                    all_examples.append(data)

    return all_examples, winner is not None


def generate_smart_games(network, num_games, state_encoder, action_encoder, temperature=1.0):
    """Generate games and filter for quality data."""

    all_examples = []
    completed = 0

    for _ in range(num_games):
        examples, finished = play_smart_game(network, state_encoder, action_encoder, temperature)
        if finished:
            all_examples.extend(examples)
            completed += 1

    return all_examples, completed


class SmartTrainer:
    """Training focused on good moves."""

    def __init__(self, network, lr=0.001, buffer_size=50000):
        self.network = network
        self.optimizer = torch.optim.Adam(network.parameters(), lr=lr, weight_decay=1e-4)
        self.buffer = deque(maxlen=buffer_size)

    def add_examples(self, examples):
        self.buffer.extend(examples)

    def train_batch(self, batch_size=256):
        if len(self.buffer) < batch_size:
            return 0.0

        batch = random.sample(list(self.buffer), batch_size)

        states = torch.tensor(np.array([ex['state'] for ex in batch]), dtype=torch.float32)
        action_indices = torch.tensor([ex['action_idx'] for ex in batch], dtype=torch.long)
        values = torch.tensor(np.array([[ex['value']] for ex in batch]), dtype=torch.float32)

        # Forward
        log_probs, pred_values = self.network(states)

        # Policy loss: maximize log prob of good actions, weighted by value
        selected_log_probs = log_probs.gather(1, action_indices.unsqueeze(1))
        policy_loss = -torch.mean(selected_log_probs * values)

        # Value loss
        value_loss = F.mse_loss(pred_values, values)

        # Total loss
        loss = policy_loss + 0.5 * value_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def train_epoch(self, batch_size=256, num_batches=50):
        total_loss = 0.0
        for _ in range(num_batches):
            total_loss += self.train_batch(batch_size)
        return total_loss / num_batches


def smart_train(num_iterations=20, games_per_iter=100, epochs_per_iter=10):
    """Main training loop."""

    print("=" * 70)
    print("  🧠 SMART TRAINING - Learning to WIN 🧠")
    print("=" * 70)

    state_encoder = SmartStateEncoder()
    action_encoder = SmartActionEncoder()
    network = SmartNetwork(state_encoder.state_size, action_encoder.num_actions)
    trainer = SmartTrainer(network)

    total_games = 0
    total_time = 0

    for iteration in range(1, num_iterations + 1):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration}/{num_iterations}")
        print(f"{'='*60}")

        # Temperature: high early (explore), low later (exploit)
        temperature = max(0.5, 2.0 - iteration * 0.1)

        # Generate games
        print(f"\nGenerating {games_per_iter} games (temp={temperature:.2f})...")
        start = time.time()
        examples, completed = generate_smart_games(
            network, games_per_iter, state_encoder, action_encoder, temperature
        )
        gen_time = time.time() - start

        games_per_sec = games_per_iter / gen_time
        total_games += games_per_iter
        total_time += gen_time

        print(f"Completed: {completed}/{games_per_iter} games")
        print(f"Examples from winners: {len(examples)} ({gen_time:.1f}s, {games_per_sec:.1f} g/s)")

        # Train
        trainer.add_examples(examples)

        if len(trainer.buffer) >= 256:
            print(f"\nTraining on {len(trainer.buffer)} examples...")
            for epoch in range(1, epochs_per_iter + 1):
                loss = trainer.train_epoch(batch_size=256, num_batches=20)
                if epoch % 2 == 0:
                    print(f"  Epoch {epoch}: Loss = {loss:.4f}")

    print(f"\n{'='*70}")
    print("🧠 SMART TRAINING COMPLETE 🧠")
    print(f"{'='*70}")
    print(f"Total games: {total_games}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Speed: {total_games/total_time:.1f} games/sec")

    # Save
    import os
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        'network': network.state_dict(),
        'state_size': state_encoder.state_size,
        'num_actions': action_encoder.num_actions,
    }, "checkpoints/smart_model.pt")
    print("\nModel saved to checkpoints/smart_model.pt")

    return network, state_encoder, action_encoder


def evaluate_smart(network, state_encoder, action_encoder, num_games=100):
    """Evaluate the smart player against randoms."""

    print(f"\n{'='*60}")
    print("  EVALUATING SMART MODEL vs RANDOM")
    print(f"{'='*60}")

    wins = 0

    for game_num in range(num_games):
        # P0 is trained (greedy), P1-P3 are random
        trained_player = SmartPlayer(Color.RED, network, state_encoder, action_encoder, temperature=0.0)
        random_players = [
            SmartPlayer(Color.BLUE, None, state_encoder, action_encoder, temperature=1.0),
            SmartPlayer(Color.WHITE, None, state_encoder, action_encoder, temperature=1.0),
            SmartPlayer(Color.ORANGE, None, state_encoder, action_encoder, temperature=1.0),
        ]

        all_players = [trained_player] + random_players
        game = Game(all_players)

        while game.winning_color() is None:
            current_idx = game.state.current_player_index
            playable = game.state.playable_actions
            if not playable:
                break

            action = all_players[current_idx].decide(game, playable, current_idx)
            if action is None:
                break
            game.execute(action)

        winner = game.winning_color()
        if winner:
            winner_idx = game.state.colors.index(winner)
            if winner_idx == 0:
                wins += 1

        if (game_num + 1) % 20 == 0:
            print(f"  Game {game_num+1}/{num_games}: Win rate = {100*wins/(game_num+1):.1f}%")

    win_rate = wins / num_games
    print(f"\nFinal win rate: {wins}/{num_games} = {100*win_rate:.1f}%")
    print("(Random expected: 25%)")

    return win_rate


if __name__ == "__main__":
    # Train
    net, se, ae = smart_train(
        num_iterations=30,
        games_per_iter=150,
        epochs_per_iter=10
    )

    # Evaluate
    evaluate_smart(net, se, ae, num_games=100)
