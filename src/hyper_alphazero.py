"""
HYPER ALPHAZERO - Maximum Speed Through Async Batch Games

Further optimizations over ULTRA:
1. ASYNC GAME COMPLETION - Don't wait for all games, replace finished ones immediately
2. MOVE LIMIT - Cap games at 300 moves (random games go forever)
3. LARGER BATCHES - 128 games at once
4. EVEN SMALLER NETWORK - 96 hidden
5. SIMPLIFIED VALUE - Train on VP difference, not just win/loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import sys

from catanatron import Game, RandomPlayer, Color
from catanatron.models.enums import ActionType


# ============================================================
# HYPER COMPACT NETWORK  
# ============================================================

class HyperNet(nn.Module):
    def __init__(self, state_size=80, action_size=450, hidden=96):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.policy = nn.Linear(hidden, action_size)
        self.value = nn.Linear(hidden, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.policy(x), torch.tanh(self.value(x))


# ============================================================
# HYPER FAST ENCODING
# ============================================================

def encode_hyper(game) -> np.ndarray:
    """80 features - bare minimum."""
    ps = game.state.player_state
    f = np.zeros(80, dtype=np.float32)
    
    cp = game.state.current_player_index
    f[cp] = 1.0
    
    # 4 players × 9 features = 36
    for i in range(4):
        p = f'P{i}_'
        base = 4 + i * 9
        f[base] = ps.get(p + 'VICTORY_POINTS', 0) / 10.0
        f[base+1] = min(ps.get(p + 'WOOD_IN_HAND', 0), 8) / 8.0
        f[base+2] = min(ps.get(p + 'BRICK_IN_HAND', 0), 8) / 8.0
        f[base+3] = min(ps.get(p + 'SHEEP_IN_HAND', 0), 8) / 8.0
        f[base+4] = min(ps.get(p + 'WHEAT_IN_HAND', 0), 8) / 8.0
        f[base+5] = min(ps.get(p + 'ORE_IN_HAND', 0), 8) / 8.0
        f[base+6] = (5 - ps.get(p + 'SETTLEMENTS_AVAILABLE', 5)) / 5.0
        f[base+7] = (4 - ps.get(p + 'CITIES_AVAILABLE', 4)) / 4.0
        f[base+8] = (15 - ps.get(p + 'ROADS_AVAILABLE', 15)) / 15.0
    
    f[40] = 1.0 if game.state.is_initial_build_phase else 0.0
    f[41] = min(game.state.num_turns, 150) / 150.0
    
    return f


_ACT_CACHE = {}

def encode_act(action) -> int:
    """Cached action encoding."""
    key = id(action.action_type) ^ hash(str(action.value)[:20])
    if key in _ACT_CACHE:
        return _ACT_CACHE[key]
    
    at = action.action_type
    v = action.value
    
    if at == ActionType.ROLL: idx = 0
    elif at == ActionType.END_TURN: idx = 1
    elif at == ActionType.BUILD_SETTLEMENT: idx = 2 + (v % 54 if isinstance(v, int) else 0)
    elif at == ActionType.BUILD_CITY: idx = 56 + (v % 54 if isinstance(v, int) else 0)
    elif at == ActionType.BUILD_ROAD: idx = 110 + (hash(str(v)) % 72)
    elif at == ActionType.BUY_DEVELOPMENT_CARD: idx = 182
    elif at == ActionType.PLAY_KNIGHT_CARD: idx = 183 + (hash(str(v)) % 50)
    elif at == ActionType.MOVE_ROBBER: idx = 233 + (hash(str(v)) % 50)
    elif at == ActionType.MARITIME_TRADE: idx = 283 + (hash(str(v)) % 50)
    elif at == ActionType.DISCARD: idx = 333 + (hash(str(v)) % 50)
    else: idx = 383 + (hash(str(at)) % 50)
    
    idx = min(idx, 449)
    _ACT_CACHE[key] = idx
    return idx


# ============================================================
# HYPER BATCH ENGINE - Async replacement
# ============================================================

class HyperEngine:
    """Run games with immediate replacement of finished ones."""
    
    def __init__(self, network, batch_size=128, device='cpu', max_moves=300):
        self.network = network
        self.batch_size = batch_size
        self.device = device
        self.max_moves = max_moves
        self.colors = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]
        self.states = np.zeros((batch_size, 80), dtype=np.float32)
    
    def new_game(self):
        players = [RandomPlayer(c) for c in self.colors]
        return Game(players)
    
    def generate(self, num_games, temperature=1.0, noise=0.2):
        """Generate exactly num_games worth of data."""
        
        # Initialize batch
        games = [self.new_game() for _ in range(self.batch_size)]
        game_data = [{'states': [], 'policies': [], 'players': []} for _ in range(self.batch_size)]
        move_counts = [0] * self.batch_size
        
        samples = []
        completed = 0
        
        self.network.eval()
        
        while completed < num_games:
            # Encode all current states
            for i, g in enumerate(games):
                self.states[i] = encode_hyper(g)
            
            # Batch forward pass
            with torch.no_grad():
                state_t = torch.from_numpy(self.states).to(self.device)
                policy_logits, _ = self.network(state_t)
                policies = F.softmax(policy_logits / temperature, dim=-1).cpu().numpy()
            
            # Process each game
            for i in range(self.batch_size):
                g = games[i]
                
                # Check if game finished
                done = g.winning_color() or move_counts[i] >= self.max_moves
                
                if done:
                    # Collect samples from this game
                    winner_color = g.winning_color()
                    if winner_color:
                        winner = self.colors.index(winner_color)
                    else:
                        vps = [g.state.player_state.get(f'P{j}_VICTORY_POINTS', 0) for j in range(4)]
                        winner = int(np.argmax(vps)) if max(vps) >= 5 else -1
                    
                    for state, policy, player in zip(
                        game_data[i]['states'],
                        game_data[i]['policies'],
                        game_data[i]['players']
                    ):
                        value = 1.0 if winner == player else (-1.0 if winner >= 0 else 0.0)
                        samples.append((state, policy, value))
                    
                    completed += 1
                    if completed >= num_games:
                        break
                    
                    # Reset this slot
                    games[i] = self.new_game()
                    game_data[i] = {'states': [], 'policies': [], 'players': []}
                    move_counts[i] = 0
                    continue
                
                # Get valid actions
                valid = list(g.state.playable_actions)
                if not valid:
                    games[i] = self.new_game()
                    game_data[i] = {'states': [], 'policies': [], 'players': []}
                    move_counts[i] = 0
                    continue
                
                current_player = g.state.current_player_index
                
                # Compute action probabilities
                policy = policies[i]
                probs = np.array([policy[encode_act(a)] for a in valid])
                
                if len(valid) > 1 and noise > 0:
                    n = np.random.dirichlet([0.3] * len(probs))
                    probs = (1 - noise) * probs + noise * n
                
                probs = np.clip(probs, 1e-8, 1.0)
                probs /= probs.sum()
                
                # Sample and execute
                action = valid[np.random.choice(len(valid), p=probs)]
                
                # Store data
                game_data[i]['states'].append(self.states[i].copy())
                policy_vec = np.zeros(450, dtype=np.float32)
                for a, p in zip(valid, probs):
                    policy_vec[encode_act(a)] = p
                game_data[i]['policies'].append(policy_vec)
                game_data[i]['players'].append(current_player)
                
                g.execute(action)
                move_counts[i] += 1
        
        return samples


# ============================================================
# TRAINING
# ============================================================

def train_hyper(num_iterations=50, games_per_iter=500, epochs=3, lr=0.003, device='cpu'):
    """Hyper-fast training."""
    
    network = HyperNet(state_size=80, action_size=450, hidden=96)
    network.to(device)
    
    optimizer = optim.AdamW(network.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iterations)
    
    engine = HyperEngine(network, batch_size=128, device=device, max_moves=300)
    
    replay_buffer = []
    max_buffer = 500000
    
    print("=" * 60)
    print("  HYPER ALPHAZERO - MAXIMUM SPEED")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Batch: 128 games parallel | Max moves: 300")
    print(f"Network: {sum(p.numel() for p in network.parameters()):,} params")
    print()
    
    total_games = 0
    total_time = 0
    
    for iteration in range(1, num_iterations + 1):
        print(f"--- Iteration {iteration}/{num_iterations} ---")
        
        network.eval()
        start = time.time()
        
        temp = 1.0 if iteration < num_iterations * 0.6 else 0.5
        noise = 0.25 if iteration < num_iterations * 0.7 else 0.1
        
        samples = engine.generate(games_per_iter, temperature=temp, noise=noise)
        
        elapsed = time.time() - start
        total_games += games_per_iter
        total_time += elapsed
        speed = games_per_iter / elapsed
        
        print(f"  {len(samples)} samples from {games_per_iter} games in {elapsed:.1f}s "
              f"({speed:.1f} games/sec)")
        
        # Buffer
        replay_buffer.extend(samples)
        if len(replay_buffer) > max_buffer:
            replay_buffer = replay_buffer[-max_buffer:]
        
        # Train
        network.train()
        indices = np.random.permutation(len(replay_buffer))
        batch_size = 2048
        
        for epoch in range(epochs):
            total_loss = p_loss = v_loss = 0
            n = 0
            
            for i in range(0, min(len(indices), 60000), batch_size):
                idx = indices[i:i + batch_size]
                
                s = torch.FloatTensor(np.array([replay_buffer[j][0] for j in idx])).to(device)
                p = torch.FloatTensor(np.array([replay_buffer[j][1] for j in idx])).to(device)
                v = torch.FloatTensor(np.array([replay_buffer[j][2] for j in idx])).unsqueeze(1).to(device)
                
                optimizer.zero_grad()
                po, vo = network(s)
                
                pl = -torch.sum(p * F.log_softmax(po, dim=-1)) / len(idx)
                vl = F.mse_loss(vo, v)
                loss = pl + vl
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                p_loss += pl.item()
                v_loss += vl.item()
                n += 1
            
            if n > 0:
                print(f"  Epoch {epoch+1}: Loss={total_loss/n:.4f} (p:{p_loss/n:.4f} v:{v_loss/n:.4f})")
        
        scheduler.step()
        
        if iteration % 10 == 0:
            torch.save(network.state_dict(), f"checkpoints/hyper_{iteration:04d}.pt")
        
        avg = total_games / total_time
        print(f"  Avg: {avg:.1f} games/sec | Total: {total_games} games")
        print()
    
    torch.save(network.state_dict(), "checkpoints/hyper_final.pt")
    
    print("=" * 60)
    print(f"  COMPLETE: {total_games} games in {total_time:.0f}s = {total_games/total_time:.1f} games/sec")
    print("=" * 60)
    
    return network


# ============================================================
# EVALUATION  
# ============================================================

class HyperPlayer:
    def __init__(self, color, network, device='cpu'):
        self.color = color
        self.network = network
        self.device = device
    
    def decide(self, game, actions):
        if len(actions) == 1:
            return actions[0]
        
        s = torch.FloatTensor(encode_hyper(game)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            p, _ = self.network(s)
            p = F.softmax(p, dim=-1).cpu().numpy()[0]
        
        probs = np.array([p[encode_act(a)] for a in actions])
        probs = np.clip(probs, 1e-8, 1.0)
        probs /= probs.sum()
        
        return actions[np.argmax(probs)]


def evaluate(network, num_games=100, device='cpu'):
    colors = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]
    wins = 0
    vps = 0
    
    network.eval()
    start = time.time()
    
    for g in range(num_games):
        players = [HyperPlayer(colors[0], network, device)] + [RandomPlayer(c) for c in colors[1:]]
        game = Game(players)
        game.play()
        
        if game.winning_color() == Color.RED:
            wins += 1
        vps += game.state.player_state.get('P0_VICTORY_POINTS', 0)
        
        if (g + 1) % 25 == 0:
            print(f"  Eval {g+1}/{num_games}: {100*wins/(g+1):.1f}% win | "
                  f"{(g+1)/(time.time()-start):.1f} games/sec")
    
    return wins / num_games, vps / num_games


if __name__ == "__main__":
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    games = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    network = train_hyper(num_iterations=iters, games_per_iter=games, epochs=3, device=device)
    
    print("\n" + "=" * 60)
    print("  EVALUATION")
    print("=" * 60)
    
    wr, vp = evaluate(network, 100, device)
    
    print(f"\nWin rate: {100*wr:.1f}% | Avg VP: {vp:.1f}")
    print("✓ STRONG" if wr > 0.35 else ("~ Equal" if wr > 0.25 else "✗ Weak"))
