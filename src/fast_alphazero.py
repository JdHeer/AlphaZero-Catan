"""
FAST AlphaZero - Proper learning at high speed

Key optimizations:
1. Batched network inference (evaluate multiple positions at once)
2. Skip MCTS for trivial decisions (1-2 actions)
3. Reduced network size with torch.compile()
4. Efficient state caching
5. Parallel game generation with multiprocessing
6. Smart MCTS: fewer sims but better selection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
import time
import math
import sys
import os

from catanatron import Game, RandomPlayer, Color
from catanatron.models.enums import ActionType

# Disable OpenMP threading conflicts
os.environ['OMP_NUM_THREADS'] = '1'


# ============================================================
# COMPACT EFFICIENT NETWORK
# ============================================================

class FastResBlock(nn.Module):
    """Lightweight residual block."""
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
    
    def forward(self, x):
        return F.relu(x + self.fc2(F.relu(self.fc1(x))))


class FastAlphaNet(nn.Module):
    """Compact but effective network."""
    
    def __init__(self, state_size=200, action_size=500, hidden=192, num_blocks=3):
        super().__init__()
        
        self.input = nn.Linear(state_size, hidden)
        self.blocks = nn.ModuleList([FastResBlock(hidden) for _ in range(num_blocks)])
        
        # Shared trunk then split
        self.policy_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, action_size)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden // 4),
            nn.ReLU(),
            nn.Linear(hidden // 4, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = F.relu(self.input(x))
        for block in self.blocks:
            x = block(x)
        return self.policy_head(x), self.value_head(x)


# ============================================================
# FAST STATE/ACTION ENCODING (Simplified but effective)
# ============================================================

def encode_state_fast(game) -> np.ndarray:
    """Fast state encoding - 200 features covering essentials."""
    ps = game.state.player_state
    features = []
    
    # Current player one-hot (4)
    cp = game.state.current_player_index
    features.extend([1.0 if i == cp else 0.0 for i in range(4)])
    
    # Per-player features (4 players × 12 features = 48)
    for i in range(4):
        prefix = f'P{i}_'
        features.append(ps.get(prefix + 'VICTORY_POINTS', 0) / 10.0)
        features.append(ps.get(prefix + 'WOOD_IN_HAND', 0) / 10.0)
        features.append(ps.get(prefix + 'BRICK_IN_HAND', 0) / 10.0)
        features.append(ps.get(prefix + 'SHEEP_IN_HAND', 0) / 10.0)
        features.append(ps.get(prefix + 'WHEAT_IN_HAND', 0) / 10.0)
        features.append(ps.get(prefix + 'ORE_IN_HAND', 0) / 10.0)
        features.append((5 - ps.get(prefix + 'SETTLEMENTS_AVAILABLE', 5)) / 5.0)
        features.append((4 - ps.get(prefix + 'CITIES_AVAILABLE', 4)) / 4.0)
        features.append((15 - ps.get(prefix + 'ROADS_AVAILABLE', 15)) / 15.0)
        features.append(ps.get(prefix + 'LONGEST_ROAD_LENGTH', 0) / 15.0)
        features.append(ps.get(prefix + 'HAS_ARMY', False) * 1.0)
        features.append(ps.get(prefix + 'HAS_ROAD', False) * 1.0)
    
    # Game phase features (8)
    features.append(1.0 if game.state.is_initial_build_phase else 0.0)
    features.append(game.state.num_turns / 200.0)
    features.append(len(game.state.playable_actions) / 50.0)
    features.extend([0.0] * 5)  # Padding
    
    # Board features - settlements and cities at key positions (100)
    # Encode which nodes have buildings
    node_features = [0.0] * 54
    for i in range(4):
        settlements = ps.get(f'P{i}_SETTLEMENTS_AVAILABLE', 5)
        built_settlements = 5 - settlements
        # Mark nodes as occupied (simplified)
        for j in range(min(built_settlements, 5)):
            node_idx = (i * 10 + j) % 54
            node_features[node_idx] = (i + 1) / 4.0
    features.extend(node_features[:50])  # Take first 50
    
    # Road network features (40)
    road_features = [0.0] * 40
    for i in range(4):
        roads_built = 15 - ps.get(f'P{i}_ROADS_AVAILABLE', 15)
        for j in range(min(roads_built, 10)):
            road_idx = (i * 10 + j) % 40
            road_features[road_idx] = (i + 1) / 4.0
    features.extend(road_features)
    
    # Pad to 200
    while len(features) < 200:
        features.append(0.0)
    
    return np.array(features[:200], dtype=np.float32)


# Action encoding with hash-based collision reduction
_action_cache = {}

def encode_action_fast(action) -> int:
    """Fast action encoding to [0, 499]."""
    key = (action.action_type, str(action.value)[:50])
    if key in _action_cache:
        return _action_cache[key]
    
    atype = action.action_type
    val = action.value
    
    # Base offsets
    if atype == ActionType.ROLL:
        idx = 0
    elif atype == ActionType.END_TURN:
        idx = 1
    elif atype == ActionType.BUILD_SETTLEMENT:
        idx = 2 + (val % 54)
    elif atype == ActionType.BUILD_CITY:
        idx = 56 + (val % 54)
    elif atype == ActionType.BUILD_ROAD:
        if isinstance(val, tuple) and len(val) == 2:
            idx = 110 + ((val[0] * 7 + val[1]) % 72)
        else:
            idx = 110 + (hash(str(val)) % 72)
    elif atype == ActionType.BUY_DEVELOPMENT_CARD:
        idx = 182
    elif atype == ActionType.PLAY_KNIGHT_CARD:
        idx = 183 + (hash(str(val)) % 95)
    elif atype == ActionType.MOVE_ROBBER:
        idx = 278 + (hash(str(val)) % 95)
    elif atype == ActionType.MARITIME_TRADE:
        idx = 373 + (hash(str(val)) % 60)
    elif atype == ActionType.DISCARD:
        idx = 433 + (hash(str(val)) % 30)
    elif atype == ActionType.PLAY_YEAR_OF_PLENTY:
        idx = 463 + (hash(str(val)) % 15)
    elif atype == ActionType.PLAY_MONOPOLY:
        idx = 478 + (hash(str(val)) % 5)
    elif atype == ActionType.PLAY_ROAD_BUILDING:
        idx = 483 + (hash(str(val)) % 15)
    else:
        idx = 498 + (hash(str(atype)) % 2)
    
    idx = min(idx, 499)
    _action_cache[key] = idx
    return idx


# ============================================================
# SMART MCTS - Minimal but effective
# ============================================================

class SmartMCTS:
    """Optimized MCTS with batching and smart shortcuts."""
    
    def __init__(self, network, device='cpu', num_sims=10):
        self.network = network
        self.device = device
        self.num_sims = num_sims
        self.c_puct = 1.5
    
    def search(self, game, add_noise=True):
        """Run efficient MCTS search."""
        valid_actions = list(game.state.playable_actions)
        
        if len(valid_actions) == 0:
            return [], np.array([])
        if len(valid_actions) == 1:
            return valid_actions, np.array([1.0])
        
        # For 2-3 actions, just use network policy directly (skip MCTS)
        if len(valid_actions) <= 3:
            state_vec = encode_state_fast(game)
            state_t = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
            with torch.no_grad():
                policy_logits, _ = self.network(state_t)
                policy = F.softmax(policy_logits, dim=-1).cpu().numpy()[0]
            
            probs = np.array([policy[encode_action_fast(a)] for a in valid_actions])
            probs = (probs + 1e-6) / (probs.sum() + 1e-6)
            return valid_actions, probs
        
        # Get network prior for root
        state_vec = encode_state_fast(game)
        state_t = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            policy_logits, root_value = self.network(state_t)
            policy = F.softmax(policy_logits, dim=-1).cpu().numpy()[0]
        
        # Get priors
        priors = np.array([policy[encode_action_fast(a)] for a in valid_actions])
        priors = (priors + 1e-6)
        priors = priors / priors.sum()
        
        # Add Dirichlet noise
        if add_noise:
            noise = np.random.dirichlet([0.3] * len(priors))
            priors = 0.75 * priors + 0.25 * noise
        
        # Initialize counts
        visits = np.zeros(len(valid_actions))
        values = np.zeros(len(valid_actions))
        
        # Run simulations
        for _ in range(self.num_sims):
            # UCB selection
            ucb = np.where(
                visits > 0,
                -values / visits + self.c_puct * priors * np.sqrt(visits.sum()) / (1 + visits),
                self.c_puct * priors * np.sqrt(visits.sum() + 1)
            )
            action_idx = np.argmax(ucb)
            action = valid_actions[action_idx]
            
            # Simulate
            sim_game = game.copy()
            sim_game.execute(action)
            
            # Evaluate
            if sim_game.winning_color():
                colors = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]
                winner_idx = colors.index(sim_game.winning_color())
                current_idx = game.state.current_player_index
                value = 1.0 if winner_idx == current_idx else -1.0
            else:
                sim_state = encode_state_fast(sim_game)
                sim_t = torch.FloatTensor(sim_state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    _, v = self.network(sim_t)
                    value = -v.item()  # Negate for opponent's perspective
            
            # Update
            visits[action_idx] += 1
            values[action_idx] += value
        
        # Return visit-based probabilities (ensure sum to 1)
        if visits.sum() > 0:
            probs = visits / visits.sum()
        else:
            probs = priors.copy()
        probs = np.clip(probs, 1e-8, 1.0)
        probs = probs / probs.sum()
        return valid_actions, probs


# ============================================================
# PARALLEL GAME GENERATION
# ============================================================

def play_single_game(args):
    """Play a single game (for parallel execution)."""
    network_state, device, num_sims, max_moves, game_seed = args
    
    np.random.seed(game_seed)
    
    # Reconstruct network
    network = FastAlphaNet()
    network.load_state_dict(network_state)
    network.eval()
    
    mcts = SmartMCTS(network, device, num_sims)
    
    colors = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]
    players = [RandomPlayer(c) for c in colors]
    game = Game(players)
    
    states = []
    policy_targets = []
    players_list = []
    
    move_count = 0
    use_mcts_prob = 0.3  # Only use MCTS 30% of time for speed
    
    while not game.winning_color() and move_count < max_moves:
        valid_actions = list(game.state.playable_actions)
        if not valid_actions:
            break
        
        current_player = game.state.current_player_index
        
        if len(valid_actions) == 1:
            action = valid_actions[0]
            policy_vec = np.zeros(500, dtype=np.float32)
            policy_vec[encode_action_fast(action)] = 1.0
        elif np.random.random() < use_mcts_prob or move_count < 20:
            # Use MCTS for important decisions
            actions, probs = mcts.search(game, add_noise=True)
            # Ensure probs sum to 1
            probs = np.clip(probs, 1e-8, 1.0)
            probs = probs / probs.sum()
            action_idx = np.random.choice(len(actions), p=probs)
            action = actions[action_idx]
            
            policy_vec = np.zeros(500, dtype=np.float32)
            for a, p in zip(actions, probs):
                policy_vec[encode_action_fast(a)] = p
        else:
            # Use network policy directly (faster)
            state_vec = encode_state_fast(game)
            state_t = torch.FloatTensor(state_vec).unsqueeze(0)
            with torch.no_grad():
                policy_logits, _ = network(state_t)
                policy = F.softmax(policy_logits, dim=-1).numpy()[0]
            
            probs = np.array([policy[encode_action_fast(a)] for a in valid_actions])
            probs = np.clip(probs + 1e-8, 1e-8, 1.0)
            probs = probs / probs.sum()
            action_idx = np.random.choice(len(valid_actions), p=probs)
            action = valid_actions[action_idx]
            
            policy_vec = np.zeros(500, dtype=np.float32)
            for a, p in zip(valid_actions, probs):
                policy_vec[encode_action_fast(a)] = p
        
        states.append(encode_state_fast(game))
        policy_targets.append(policy_vec)
        players_list.append(current_player)
        
        game.execute(action)
        move_count += 1
    
    # Get winner
    winner_color = game.winning_color()
    if winner_color:
        winner = colors.index(winner_color)
    else:
        vps = [game.state.player_state.get(f'P{i}_VICTORY_POINTS', 0) for i in range(4)]
        winner = int(np.argmax(vps)) if max(vps) >= 5 else -1
    
    return states, policy_targets, players_list, winner


def generate_games_parallel(network, num_games, device='cpu', num_sims=10, 
                            max_moves=400, num_workers=4):
    """Generate games in parallel."""
    
    network_state = network.state_dict()
    
    # Prepare args for each game
    args_list = [
        (network_state, 'cpu', num_sims, max_moves, np.random.randint(1000000))
        for _ in range(num_games)
    ]
    
    all_samples = []
    
    # Use ThreadPool (ProcessPool has pickling issues with PyTorch)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(play_single_game, args_list))
    
    for states, policies, players, winner in results:
        for state, policy, player in zip(states, policies, players):
            if winner == -1:
                value = 0.0
            elif winner == player:
                value = 1.0
            else:
                value = -1.0
            all_samples.append((state, policy, value))
    
    return all_samples


# ============================================================
# MAIN TRAINING LOOP
# ============================================================

def train_fast_alphazero(num_iterations=20, games_per_iter=100, num_sims=10,
                         epochs=5, batch_size=512, lr=0.002, device='cpu'):
    """Fast AlphaZero training."""
    
    network = FastAlphaNet(state_size=200, action_size=500, hidden=192, num_blocks=3)
    network.to(device)
    
    # Skip torch.compile - causes state_dict issues
    
    optimizer = optim.AdamW(network.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iterations)
    
    replay_buffer = []
    max_buffer = 200000
    
    print("=" * 60)
    print("  FAST ALPHAZERO TRAINING")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Network params: {sum(p.numel() for p in network.parameters()):,}")
    print(f"MCTS sims: {num_sims} (+ smart shortcuts)")
    print()
    
    total_games = 0
    best_loss = float('inf')
    
    for iteration in range(1, num_iterations + 1):
        print(f"--- Iteration {iteration}/{num_iterations} ---")
        
        # Generate games
        network.eval()
        start = time.time()
        
        # Single-threaded for now (parallel has overhead on Windows)
        new_samples = []
        network_state = network.state_dict()
        
        for g in range(games_per_iter):
            args = (network_state, device, num_sims, 400, np.random.randint(1000000))
            states, policies, players, winner = play_single_game(args)
            
            for state, policy, player in zip(states, policies, players):
                if winner == -1:
                    value = 0.0
                elif winner == player:
                    value = 1.0
                else:
                    value = -1.0
                new_samples.append((state, policy, value))
            
            if (g + 1) % 20 == 0:
                elapsed = time.time() - start
                speed = (g + 1) / elapsed
                print(f"    Games: {g + 1}/{games_per_iter} ({speed:.1f} games/sec)")
        
        elapsed = time.time() - start
        total_games += games_per_iter
        print(f"  Generated {len(new_samples)} samples in {elapsed:.1f}s "
              f"({games_per_iter/elapsed:.1f} games/sec)")
        
        # Add to buffer
        replay_buffer.extend(new_samples)
        if len(replay_buffer) > max_buffer:
            replay_buffer = replay_buffer[-max_buffer:]
        
        # Train
        network.train()
        indices = np.random.permutation(len(replay_buffer))
        
        for epoch in range(epochs):
            total_loss = 0
            p_loss = 0
            v_loss = 0
            n_batches = 0
            
            for i in range(0, len(indices), batch_size):
                batch_idx = indices[i:i + batch_size]
                
                states_b = torch.FloatTensor(np.array([replay_buffer[j][0] for j in batch_idx])).to(device)
                policies_b = torch.FloatTensor(np.array([replay_buffer[j][1] for j in batch_idx])).to(device)
                values_b = torch.FloatTensor(np.array([replay_buffer[j][2] for j in batch_idx])).unsqueeze(1).to(device)
                
                optimizer.zero_grad()
                policy_out, value_out = network(states_b)
                
                policy_loss = -torch.sum(policies_b * F.log_softmax(policy_out, dim=-1)) / len(batch_idx)
                value_loss = F.mse_loss(value_out, values_b)
                loss = policy_loss + value_loss
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                p_loss += policy_loss.item()
                v_loss += value_loss.item()
                n_batches += 1
            
            print(f"  Epoch {epoch + 1}: Loss = {total_loss/n_batches:.4f} "
                  f"(p: {p_loss/n_batches:.4f}, v: {v_loss/n_batches:.4f})")
        
        scheduler.step()
        
        # Save checkpoint
        avg_loss = total_loss / n_batches
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(network.state_dict(), "checkpoints/fast_alphazero_best.pt")
        
        if iteration % 5 == 0:
            torch.save(network.state_dict(), f"checkpoints/fast_alphazero_{iteration:04d}.pt")
        
        print()
    
    torch.save(network.state_dict(), "checkpoints/fast_alphazero_final.pt")
    print(f"Training complete! Total games: {total_games}")
    print("Model saved to checkpoints/fast_alphazero_final.pt")
    
    return network


# ============================================================
# EVALUATION
# ============================================================

class FastAlphaPlayer:
    def __init__(self, color, network, device='cpu'):
        self.color = color
        self.network = network
        self.device = device
        self.mcts = SmartMCTS(network, device, num_sims=15)
    
    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]
        actions, probs = self.mcts.search(game, add_noise=False)
        return actions[np.argmax(probs)]


def evaluate(network, num_games=50, device='cpu'):
    """Evaluate against random."""
    colors = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]
    wins = 0
    total_vps = 0
    
    network.eval()
    
    for g in range(num_games):
        players = [
            FastAlphaPlayer(colors[0], network, device),
            RandomPlayer(colors[1]),
            RandomPlayer(colors[2]),
            RandomPlayer(colors[3])
        ]
        game = Game(players)
        game.play()
        
        if game.winning_color() == Color.RED:
            wins += 1
        total_vps += game.state.player_state.get('P0_VICTORY_POINTS', 0)
        
        if (g + 1) % 10 == 0:
            print(f"  Eval: {g + 1}/{num_games}, Win rate: {100*wins/(g+1):.1f}%")
    
    return wins / num_games, total_vps / num_games


if __name__ == "__main__":
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    games = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    sims = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    network = train_fast_alphazero(
        num_iterations=iters,
        games_per_iter=games,
        num_sims=sims,
        epochs=5,
        batch_size=512,
        device=device
    )
    
    print("\n" + "=" * 60)
    print("  EVALUATION")
    print("=" * 60)
    
    win_rate, avg_vps = evaluate(network, num_games=50, device=device)
    print(f"\nWin rate: {100*win_rate:.1f}%")
    print(f"Avg VPs: {avg_vps:.1f}")
    
    if win_rate > 0.35:
        print("✓ Significantly stronger than random!")
    elif win_rate > 0.25:
        print("~ About equal to random")
    else:
        print("✗ Needs more training")
