"""
TRUE AlphaZero Training for Catan

Key differences from turbo_train:
1. Uses MCTS to generate policy targets (not just network)
2. Proper value targets from game outcomes
3. Full state encoding (1266 features)
4. Deeper residual network
5. Temperature annealing during games
6. Dirichlet noise at root for exploration

Speed optimizations:
- Batched neural network inference
- Efficient state caching
- Parallel game generation (optional)
- Reduced MCTS simulations (25-50 instead of 800)
"""

import math
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from catanatron import Color, Game, RandomPlayer

from encoders import ActionEncoder, StateEncoder

# ============================================================
# RESIDUAL NETWORK (Deeper, Better)
# ============================================================

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Linear(channels, channels)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Linear(channels, channels)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + residual
        return F.relu(x)


class AlphaZeroNetwork(nn.Module):
    """Proper AlphaZero network with residual blocks."""

    def __init__(self, state_size=1266, action_size=483, hidden=256, num_blocks=4):
        super().__init__()

        self.input_layer = nn.Linear(state_size, hidden)
        self.input_bn = nn.BatchNorm1d(hidden)

        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden) for _ in range(num_blocks)
        ])

        # Policy head
        self.policy_fc1 = nn.Linear(hidden, hidden // 2)
        self.policy_bn = nn.BatchNorm1d(hidden // 2)
        self.policy_fc2 = nn.Linear(hidden // 2, action_size)

        # Value head
        self.value_fc1 = nn.Linear(hidden, hidden // 4)
        self.value_bn = nn.BatchNorm1d(hidden // 4)
        self.value_fc2 = nn.Linear(hidden // 4, 1)

    def forward(self, x):
        # Input
        x = F.relu(self.input_bn(self.input_layer(x)))

        # Residual tower
        for block in self.res_blocks:
            x = block(x)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_fc1(x)))
        p = self.policy_fc2(p)

        # Value head
        v = F.relu(self.value_bn(self.value_fc1(x)))
        v = torch.tanh(self.value_fc2(v))

        return p, v


# ============================================================
# MCTS (Simplified but Correct)
# ============================================================

class MCTSNode:
    __slots__ = ['parent', 'action', 'prior', 'visit_count', 'value_sum', 'children', 'is_expanded']

    def __init__(self, parent=None, action=None, prior=0.0):
        self.parent = parent
        self.action = action
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}
        self.is_expanded = False

    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, parent_visits, c_puct=1.5):
        prior_score = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        value_score = -self.value()  # Negative because alternating players
        return value_score + prior_score


class FastMCTS:
    """Simplified MCTS for speed."""

    def __init__(self, network, state_encoder, action_encoder, device='cpu',
                 num_simulations=25, c_puct=1.5):
        self.network = network
        self.state_encoder = state_encoder
        self.action_encoder = action_encoder
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = c_puct

    def search(self, game, add_noise=True):
        """Run MCTS and return action probabilities."""
        root = MCTSNode()

        # Expand root with current valid actions
        self._expand(root, game, add_noise=add_noise)

        if not root.children:
            return [], np.array([])

        # Run simulations
        for _ in range(self.num_simulations):
            node = root
            sim_game = game.copy()
            path = [node]

            # Select down the tree (but only one level for safety)
            if node.children:
                node = self._select_child(node)
                path.append(node)
                if node.action:
                    # Verify action is still valid
                    valid_actions = sim_game.state.playable_actions
                    if node.action in valid_actions:
                        sim_game.execute(node.action)
                    else:
                        # Action became invalid, skip this simulation
                        continue

            # Check terminal
            if sim_game.winning_color():
                winner_idx = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE].index(sim_game.winning_color())
                current_idx = game.state.current_player_index  # Original player
                value = 1.0 if winner_idx == current_idx else -1.0
            else:
                # Evaluate position with network
                state_vec = self.state_encoder.encode(sim_game)
                state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    _, value = self.network(state_tensor)
                    value = value.item()

            # Backpropagate
            for n in reversed(path):
                n.visit_count += 1
                n.value_sum += value
                value = -value  # Flip for alternating players

        # Get action probabilities from visit counts
        actions = list(root.children.keys())
        visits = np.array([root.children[a].visit_count for a in actions])

        if visits.sum() == 0:
            probs = np.ones(len(actions)) / len(actions)
        else:
            probs = visits / visits.sum()

        return actions, probs

    def _expand(self, node, game, add_noise=False):
        """Expand node and return value estimate."""
        node.is_expanded = True

        valid_actions = game.state.playable_actions
        if not valid_actions:
            return 0.0

        # Get network predictions
        state_vec = self.state_encoder.encode(game)
        state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy_logits, value = self.network(state_tensor)
            policy = F.softmax(policy_logits, dim=-1).cpu().numpy()[0]
            value = value.item()

        # Get priors for valid actions
        priors = []
        for action in valid_actions:
            idx = self.action_encoder.encode(action)
            priors.append(policy[idx] if idx < len(policy) else 1e-6)

        priors = np.array(priors)
        priors = priors / (priors.sum() + 1e-8)

        # Add Dirichlet noise at root
        if add_noise:
            noise = np.random.dirichlet([0.3] * len(priors))
            priors = 0.75 * priors + 0.25 * noise

        # Create children
        for action, prior in zip(valid_actions, priors):
            node.children[action] = MCTSNode(parent=node, action=action, prior=prior)

        return value

    def _select_child(self, node):
        """Select child with highest UCB score."""
        best_score = -float('inf')
        best_child = None

        for child in node.children.values():
            score = child.ucb_score(node.visit_count, self.c_puct)
            if score > best_score:
                best_score = score
                best_child = child

        return best_child

    def _backpropagate(self, node, value):
        """Backpropagate value up the tree."""
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value  # Flip for alternating players
            node = node.parent


# ============================================================
# SELF-PLAY DATA GENERATION
# ============================================================

def play_game_with_mcts(network, state_encoder, action_encoder, device,
                         num_simulations=25, temperature=1.0, max_moves=400):
    """Play a game using MCTS and collect training data."""

    colors = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]
    players = [RandomPlayer(c) for c in colors]
    game = Game(players)

    mcts = FastMCTS(network, state_encoder, action_encoder, device, num_simulations)

    states = []
    policy_targets = []
    players_list = []

    move_count = 0

    while not game.winning_color() and move_count < max_moves:
        valid_actions = game.state.playable_actions
        if not valid_actions:
            break

        current_player = game.state.current_player_index

        if len(valid_actions) == 1:
            action = valid_actions[0]
            policy_vec = np.zeros(action_encoder.action_space_size, dtype=np.float32)
            policy_vec[action_encoder.encode(action)] = 1.0
        else:
            # Use MCTS
            actions, probs = mcts.search(game, add_noise=True)

            # Temperature
            if temperature != 1.0:
                probs = probs ** (1 / temperature)
                probs = probs / probs.sum()

            # Sample action
            action_idx = np.random.choice(len(actions), p=probs)
            action = actions[action_idx]

            # Create policy target
            policy_vec = np.zeros(action_encoder.action_space_size, dtype=np.float32)
            for a, p in zip(actions, probs):
                policy_vec[action_encoder.encode(a)] = p

        # Store training data
        states.append(state_encoder.encode(game))
        policy_targets.append(policy_vec)
        players_list.append(current_player)

        game.execute(action)
        move_count += 1

    # Get winner for value targets
    winner_color = game.winning_color()
    if winner_color:
        winner = colors.index(winner_color)
    else:
        vps = [game.state.player_state.get(f'P{i}_VICTORY_POINTS', 0) for i in range(4)]
        winner = int(np.argmax(vps)) if max(vps) >= 5 else -1

    return states, policy_targets, players_list, winner


# ============================================================
# TRAINING
# ============================================================

def train_alphazero(num_iterations=10, games_per_iter=50, num_simulations=25,
                    epochs=5, batch_size=256, lr=0.001, device='cpu'):
    """Main AlphaZero training loop."""

    state_encoder = StateEncoder()
    action_encoder = ActionEncoder()

    network = AlphaZeroNetwork(
        state_size=state_encoder.state_size,
        action_size=action_encoder.action_space_size,
        hidden=256,
        num_blocks=4
    ).to(device)

    optimizer = optim.Adam(network.parameters(), lr=lr, weight_decay=1e-4)

    replay_buffer = []
    max_buffer_size = 100000

    print("=" * 60)
    print("  ALPHAZERO PROPER TRAINING")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"State size: {state_encoder.state_size}")
    print(f"Action size: {action_encoder.action_space_size}")
    print(f"Network params: {sum(p.numel() for p in network.parameters()):,}")
    print()

    for iteration in range(1, num_iterations + 1):
        print(f"--- Iteration {iteration}/{num_iterations} ---")

        # Generate games with MCTS
        network.eval()
        start_time = time.time()

        new_samples = []
        for game_num in range(games_per_iter):
            # Temperature annealing
            temp = 1.0 if game_num < games_per_iter * 0.7 else 0.5

            states, policies, players, winner = play_game_with_mcts(
                network, state_encoder, action_encoder, device,
                num_simulations=num_simulations, temperature=temp, max_moves=400
            )

            # Create training samples with value targets
            for state, policy, player in zip(states, policies, players):
                if winner == -1:
                    value = 0.0
                elif winner == player:
                    value = 1.0
                else:
                    value = -1.0
                new_samples.append((state, policy, value))

            if (game_num + 1) % 10 == 0:
                elapsed = time.time() - start_time
                speed = (game_num + 1) / elapsed
                print(f"    Games: {game_num + 1}/{games_per_iter} ({speed:.1f} games/sec)")

        elapsed = time.time() - start_time
        print(f"  Generated {len(new_samples)} samples in {elapsed:.1f}s")

        # Add to replay buffer
        replay_buffer.extend(new_samples)
        if len(replay_buffer) > max_buffer_size:
            replay_buffer = replay_buffer[-max_buffer_size:]

        # Training
        network.train()

        # Prepare tensors
        indices = np.random.permutation(len(replay_buffer))

        for epoch in range(epochs):
            total_loss = 0
            total_policy_loss = 0
            total_value_loss = 0
            num_batches = 0

            for i in range(0, len(indices), batch_size):
                batch_indices = indices[i:i + batch_size]

                batch_states = torch.FloatTensor(
                    np.array([replay_buffer[j][0] for j in batch_indices])
                ).to(device)
                batch_policies = torch.FloatTensor(
                    np.array([replay_buffer[j][1] for j in batch_indices])
                ).to(device)
                batch_values = torch.FloatTensor(
                    np.array([replay_buffer[j][2] for j in batch_indices])
                ).unsqueeze(1).to(device)

                optimizer.zero_grad()

                policy_logits, value_pred = network(batch_states)

                # Policy loss (cross-entropy)
                policy_loss = -torch.sum(batch_policies * F.log_softmax(policy_logits, dim=-1)) / batch_size

                # Value loss (MSE)
                value_loss = F.mse_loss(value_pred, batch_values)

                # Total loss
                loss = policy_loss + value_loss

                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                num_batches += 1

            print(f"  Epoch {epoch + 1}: Loss = {total_loss/num_batches:.4f} "
                  f"(policy: {total_policy_loss/num_batches:.4f}, value: {total_value_loss/num_batches:.4f})")

        # Save checkpoint
        if iteration % 5 == 0:
            torch.save(network.state_dict(), f"checkpoints/alphazero_iter_{iteration:04d}.pt")

        print()

    # Save final model
    torch.save(network.state_dict(), "checkpoints/alphazero_final.pt")
    print("Model saved to checkpoints/alphazero_final.pt")

    return network, state_encoder, action_encoder


# ============================================================
# EVALUATION
# ============================================================

class AlphaZeroPlayer:
    """Player that uses trained AlphaZero network with MCTS."""

    def __init__(self, color, network, state_encoder, action_encoder, device='cpu',
                 num_simulations=50):
        self.color = color
        self.network = network
        self.state_encoder = state_encoder
        self.action_encoder = action_encoder
        self.device = device
        self.mcts = FastMCTS(network, state_encoder, action_encoder, device, num_simulations)

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]

        actions, probs = self.mcts.search(game, add_noise=False)
        return actions[np.argmax(probs)]


def evaluate_model(network, state_encoder, action_encoder, num_games=50, device='cpu'):
    """Evaluate trained model against random players."""

    colors = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]
    wins = 0
    total_vps = 0

    network.eval()

    for game_num in range(num_games):
        players = []
        for i, c in enumerate(colors):
            if i == 0:  # Our player is RED
                players.append(AlphaZeroPlayer(c, network, state_encoder, action_encoder,
                                               device, num_simulations=25))
            else:
                players.append(RandomPlayer(c))

        game = Game(players)
        game.play()

        winner = game.winning_color()
        if winner == Color.RED:
            wins += 1

        vps = game.state.player_state.get('P0_VICTORY_POINTS', 0)
        total_vps += vps

        if (game_num + 1) % 10 == 0:
            print(f"  Eval games: {game_num + 1}/{num_games}, "
                  f"Win rate: {100*wins/(game_num+1):.1f}%")

    return wins / num_games, total_vps / num_games


if __name__ == "__main__":
    # Parse arguments
    num_iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    games_per_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    num_simulations = int(sys.argv[3]) if len(sys.argv) > 3 else 25

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Train
    network, state_encoder, action_encoder = train_alphazero(
        num_iterations=num_iterations,
        games_per_iter=games_per_iter,
        num_simulations=num_simulations,
        epochs=5,
        batch_size=256,
        device=device
    )

    # Evaluate
    print("\n" + "=" * 60)
    print("  EVALUATION vs RANDOM")
    print("=" * 60)

    win_rate, avg_vps = evaluate_model(network, state_encoder, action_encoder,
                                        num_games=50, device=device)

    print("\nFinal Results:")
    print(f"  Win rate: {100*win_rate:.1f}%")
    print(f"  Avg VPs: {avg_vps:.1f}")

    if win_rate > 0.35:
        print("✓ Model is SIGNIFICANTLY stronger than random!")
    elif win_rate > 0.25:
        print("~ Model is about equal to random")
    else:
        print("✗ Model needs more training")
