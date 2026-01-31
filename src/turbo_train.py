"""
TURBO TRAINING - Maximum speed AlphaZero training for Catan.
Run with: uv run python src/turbo_train.py [iterations] [games_per_iter]
"""
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from catanatron import Color, Game, RandomPlayer
from catanatron.models.enums import ActionType
from torch.utils.data import DataLoader, TensorDataset


class TurboNetwork(nn.Module):
    """Smaller, faster network."""

    def __init__(self, state_size=50, action_size=200, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.policy = nn.Linear(hidden, action_size)
        self.value = nn.Linear(hidden, 1)
        self.action_size = action_size

    def forward(self, x):
        h = self.net(x)
        return self.policy(h), torch.tanh(self.value(h))


def encode_state_fast(game_state) -> np.ndarray:
    """Ultra-fast state encoding."""
    ps = game_state.player_state
    features = []

    cp = game_state.current_player_index
    features.extend([1 if i == cp else 0 for i in range(4)])

    for i in range(4):
        features.append(ps.get(f'P{i}_VICTORY_POINTS', 0) / 10.0)
        features.append(ps.get(f'P{i}_WOOD_IN_HAND', 0) / 10.0)
        features.append(ps.get(f'P{i}_BRICK_IN_HAND', 0) / 10.0)
        features.append(ps.get(f'P{i}_SHEEP_IN_HAND', 0) / 10.0)
        features.append(ps.get(f'P{i}_WHEAT_IN_HAND', 0) / 10.0)
        features.append(ps.get(f'P{i}_ORE_IN_HAND', 0) / 10.0)
        features.append((5 - ps.get(f'P{i}_SETTLEMENTS_AVAILABLE', 5)) / 5.0)
        features.append((4 - ps.get(f'P{i}_CITIES_AVAILABLE', 4)) / 4.0)
        features.append((15 - ps.get(f'P{i}_ROADS_AVAILABLE', 15)) / 15.0)

    features.append(1.0 if game_state.is_initial_build_phase else 0.0)
    features.append(game_state.num_turns / 500.0)
    features.extend([0.0] * 8)

    return np.array(features, dtype=np.float32)


def encode_action_fast(action) -> int:
    """Fast action encoding."""
    offsets = {
        ActionType.ROLL: 0, ActionType.END_TURN: 1,
        ActionType.BUILD_SETTLEMENT: 2, ActionType.BUILD_ROAD: 56,
        ActionType.BUILD_CITY: 128, ActionType.BUY_DEVELOPMENT_CARD: 182,
        ActionType.PLAY_KNIGHT_CARD: 183, ActionType.MARITIME_TRADE: 184,
        ActionType.MOVE_ROBBER: 195, ActionType.DISCARD: 196,
        ActionType.PLAY_YEAR_OF_PLENTY: 197, ActionType.PLAY_MONOPOLY: 198,
        ActionType.PLAY_ROAD_BUILDING: 199,
    }
    base = offsets.get(action.action_type, 0)
    if action.value is not None:
        return min(base + hash(str(action.value)) % 50, 199)
    return base


def play_game_turbo(network=None, device='cpu', max_moves=500):
    """Play one game FAST."""
    colors = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]
    players = [RandomPlayer(c) for c in colors]
    game = Game(players)

    states, policies, players_list = [], [], []
    move_count = 0

    while game.winning_color() is None and move_count < max_moves:
        state = game.state
        valid_actions = state.playable_actions
        if not valid_actions:
            break

        state_vec = encode_state_fast(state)
        current_player = state.current_player_index

        if network is not None and move_count % 3 == 0:  # Only use network every 3rd move
            with torch.no_grad():
                state_t = torch.FloatTensor(state_vec).unsqueeze(0).to(device)
                policy_logits, _ = network(state_t)
                policy = F.softmax(policy_logits, dim=-1).cpu().numpy()[0]

            action_probs = np.array([policy[encode_action_fast(a)] for a in valid_actions])
            action_probs = (action_probs + 1e-6)
            action_probs = action_probs / action_probs.sum()
            action = valid_actions[np.random.choice(len(valid_actions), p=action_probs)]
        else:
            valid_list = list(valid_actions)
            action = valid_list[np.random.randint(len(valid_list))]
            action_probs = np.ones(len(valid_list)) / len(valid_list)

        states.append(state_vec)
        policy_vec = np.zeros(200, dtype=np.float32)
        for i, a in enumerate(valid_actions):
            policy_vec[encode_action_fast(a)] = action_probs[i] if i < len(action_probs) else 0
        policies.append(policy_vec)
        players_list.append(current_player)

        game.execute(action)
        move_count += 1

    winner_color = game.winning_color()
    if winner_color:
        winner = colors.index(winner_color)
    else:
        vps = [game.state.player_state.get(f'P{i}_VICTORY_POINTS', 0) for i in range(4)]
        winner = int(np.argmax(vps)) if max(vps) >= 3 else -1

    return states, policies, players_list, winner


def train_turbo(num_iterations=10, games_per_iter=100, epochs=3, device='cpu'):
    """TURBO training loop."""
    print("=" * 60)
    print("  TURBO ALPHAZERO TRAINING")
    print("=" * 60)
    print(f"Device: {device}")

    network = TurboNetwork().to(device)
    optimizer = torch.optim.Adam(network.parameters(), lr=2e-3, weight_decay=1e-4)

    replay_states, replay_policies, replay_values = [], [], []
    max_replay = 50000

    for iteration in range(num_iterations):
        print(f"\n--- Iteration {iteration + 1}/{num_iterations} ---")

        # Generate games
        start = time.time()
        new_states, new_policies, new_values = [], [], []

        for g in range(games_per_iter):
            states, policies, players, winner = play_game_turbo(
                network if iteration > 0 else None, device, max_moves=500
            )
            for s, p, pl in zip(states, policies, players):
                new_states.append(s)
                new_policies.append(p)
                if winner == -1:
                    new_values.append(0.0)
                elif pl == winner:
                    new_values.append(1.0)
                else:
                    new_values.append(-0.3)

            if (g + 1) % 25 == 0:
                print(f"  Games: {g+1}/{games_per_iter}", end="\r")

        elapsed = time.time() - start
        print(f"  Generated {len(new_states)} samples in {elapsed:.1f}s ({games_per_iter/elapsed:.1f} games/sec)")

        # Add to replay buffer
        replay_states.extend(new_states)
        replay_policies.extend(new_policies)
        replay_values.extend(new_values)

        if len(replay_states) > max_replay:
            replay_states = replay_states[-max_replay:]
            replay_policies = replay_policies[-max_replay:]
            replay_values = replay_values[-max_replay:]

        # Train
        dataset = TensorDataset(
            torch.FloatTensor(np.array(replay_states)),
            torch.FloatTensor(np.array(replay_policies)),
            torch.FloatTensor(np.array(replay_values)),
        )
        loader = DataLoader(dataset, batch_size=512, shuffle=True)

        network.train()
        for epoch in range(epochs):
            total_loss = 0
            for bs, bp, bv in loader:
                bs, bp, bv = bs.to(device), bp.to(device), bv.to(device)
                pl, vp = network(bs)

                log_p = F.log_softmax(pl, dim=1)
                p_loss = -torch.sum(bp * log_p, dim=1).mean()
                v_loss = F.mse_loss(vp.squeeze(), bv)
                loss = p_loss + v_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f"  Epoch {epoch+1}: Loss = {total_loss/len(loader):.4f}")

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    return network


if __name__ == "__main__":
    import sys
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    games = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    network = train_turbo(iters, games, epochs=3, device=device)
    torch.save(network.state_dict(), "turbo_model.pt")
    print("Model saved to turbo_model.pt")
