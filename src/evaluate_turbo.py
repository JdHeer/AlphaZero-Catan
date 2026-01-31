"""
Evaluate the turbo-trained model against random players.
"""

import numpy as np
import torch
import torch.nn.functional as F
from catanatron import Color, Game, RandomPlayer

from turbo_train import TurboNetwork, encode_action_fast, encode_state_fast


class TurboPlayer:
    """Player that uses the turbo-trained network."""

    def __init__(self, color, network, device='cpu', temperature=0.5):
        self.color = color
        self.network = network
        self.device = device
        self.temperature = temperature

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]

        state_vec = encode_state_fast(game.state)
        state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy_logits, _ = self.network(state_tensor)
            policy = F.softmax(policy_logits / self.temperature, dim=-1).cpu().numpy()[0]

        action_probs = []
        for action in playable_actions:
            idx = encode_action_fast(action)
            action_probs.append(policy[idx])

        action_probs = np.array(action_probs) + 1e-8
        action_probs = action_probs / action_probs.sum()

        selected_idx = np.random.choice(len(playable_actions), p=action_probs)
        return playable_actions[selected_idx]


def evaluate_vs_random(network, num_games=100, device='cpu'):
    """Evaluate the trained network against random players."""

    colors = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]
    wins = {c: 0 for c in colors}
    total_vps = {c: 0 for c in colors}

    turbo_color = Color.RED

    for game_num in range(num_games):
        players = []
        for c in colors:
            if c == turbo_color:
                players.append(TurboPlayer(c, network, device))
            else:
                players.append(RandomPlayer(c))

        game = Game(players)
        game.play()

        winner = game.winning_color()
        if winner:
            wins[winner] += 1

        for i, c in enumerate(colors):
            vps = game.state.player_state.get(f'P{i}_VICTORY_POINTS', 0)
            total_vps[c] += vps

        if (game_num + 1) % 20 == 0:
            turbo_wins = wins[turbo_color]
            random_wins = sum(wins[c] for c in colors if c != turbo_color)
            print(f"  Games {game_num + 1}/{num_games}: "
                  f"Turbo {turbo_wins} - Random {random_wins} "
                  f"({100*turbo_wins/(game_num+1):.1f}% win rate)")

    return wins, total_vps


def evaluate_random_baseline(num_games=100):
    """Baseline: All random players."""

    colors = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]
    wins = {c: 0 for c in colors}

    for game_num in range(num_games):
        players = [RandomPlayer(c) for c in colors]
        game = Game(players)
        game.play()

        winner = game.winning_color()
        if winner:
            wins[winner] += 1

    return wins


if __name__ == "__main__":
    import sys

    model_path = sys.argv[1] if len(sys.argv) > 1 else "turbo_model.pt"
    num_games = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    print("=" * 60)
    print("  TURBO MODEL EVALUATION")
    print("=" * 60)

    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    network = TurboNetwork()
    network.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    network.to(device)
    network.eval()
    print(f"Loaded model from {model_path}")
    print(f"Device: {device}")
    print()

    # Random baseline
    print("BASELINE: All Random Players")
    print("-" * 40)
    random_wins = evaluate_random_baseline(num_games=50)
    print("  Expected win rate per player: ~25%")
    print(f"  RED wins: {random_wins[Color.RED]} ({100*random_wins[Color.RED]/50:.1f}%)")
    print()

    # Turbo vs Random
    print("TURBO (RED) vs RANDOM (others)")
    print("-" * 40)
    wins, total_vps = evaluate_vs_random(network, num_games=num_games, device=device)

    print()
    print("FINAL RESULTS:")
    print("-" * 40)

    turbo_wins = wins[Color.RED]
    random_wins_total = sum(wins[c] for c in [Color.BLUE, Color.WHITE, Color.ORANGE])

    print(f"  Turbo (RED):   {turbo_wins} wins ({100*turbo_wins/num_games:.1f}%)")
    print(f"  Random (avg):  {random_wins_total/3:.1f} wins each ({100*random_wins_total/3/num_games:.1f}%)")
    print()

    avg_vps_turbo = total_vps[Color.RED] / num_games
    avg_vps_random = sum(total_vps[c] for c in [Color.BLUE, Color.WHITE, Color.ORANGE]) / 3 / num_games

    print(f"  Avg VP - Turbo:  {avg_vps_turbo:.1f}")
    print(f"  Avg VP - Random: {avg_vps_random:.1f}")
    print()

    if turbo_wins > num_games * 0.30:
        print("✓ Model is STRONGER than random (>30% win rate)")
    elif turbo_wins > num_games * 0.25:
        print("~ Model is about EQUAL to random (~25% win rate)")
    else:
        print("✗ Model is WEAKER than random (<25% win rate)")
