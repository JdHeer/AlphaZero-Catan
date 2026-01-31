# AlphaZero Catan

Train a super-strong Catan AI using AlphaZero-style reinforcement learning.

## 🎯 Overview

This project implements the AlphaZero algorithm for the board game Catan:

- **Neural Network**: Predicts action probabilities (policy) and win probability (value)
- **Monte Carlo Tree Search (MCTS)**: Uses the network to guide game tree search
- **Self-Play**: Generates training data by playing against itself
- **Iterative Training**: Network improves through repeated self-play and training

## 📦 Installation

```bash
# Install dependencies with UV
uv sync
```

## 🚀 Quick Start

### Run Training

```bash
# Basic training run
uv run python main.py --iterations 10 --games-per-iter 20 --verbose

# Full training with more resources
uv run python main.py \
    --iterations 100 \
    --games-per-iter 50 \
    --mcts-sims 200 \
    --batch-size 512 \
    --epochs 20
```

### Evaluate a Trained Model

```bash
uv run python main.py --eval-only --load-checkpoint checkpoints/checkpoint_0050.pt
```

## 📁 Project Structure

```
├── main.py                 # Main training script
├── src/
│   ├── game_wrapper.py     # Catanatron interface
│   ├── neural_network.py   # PyTorch network architecture
│   ├── mcts.py             # Monte Carlo Tree Search
│   ├── self_play.py        # Self-play data generation
│   ├── trainer.py          # Training loop
│   └── evaluate.py         # Evaluation utilities
├── checkpoints/            # Saved model checkpoints
└── pyproject.toml          # Project dependencies
```

## 🧠 How It Works

### 1. Neural Network Architecture

The network takes a game state and outputs:
- **Policy**: Probability distribution over all possible actions
- **Value**: Expected game outcome from -1 (loss) to +1 (win)

### 2. Monte Carlo Tree Search

MCTS uses the network to guide search:
1. **Selection**: Follow most promising path using UCB formula
2. **Expansion**: Add new nodes when reaching unexplored states
3. **Evaluation**: Use network value head to estimate position
4. **Backpropagation**: Update statistics along the path

### 3. Training Loop

```
For each iteration:
    1. Play N games using MCTS + current network
    2. Collect (state, MCTS_policy, outcome) examples
    3. Train network on collected data
    4. Save checkpoint
```

## ⚙️ Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--iterations` | 100 | Training iterations |
| `--games-per-iter` | 50 | Self-play games per iteration |
| `--mcts-sims` | 100 | MCTS simulations per move |
| `--batch-size` | 256 | Training batch size |
| `--lr` | 0.001 | Learning rate |
| `--hidden-size` | 256 | Network hidden layer size |
| `--num-players` | 4 | Number of players |

## 📊 Training Tips

### Computational Requirements

- **GPU Recommended**: Training is much faster with CUDA
- **Time**: Expect several hours for meaningful results
- **Memory**: ~4GB RAM minimum

### Hyperparameter Tuning

1. **Start small**: Begin with fewer simulations and games
2. **Monitor loss**: Both policy and value loss should decrease
3. **Evaluate regularly**: Check win rate against random baseline

### Improving Results

- Increase MCTS simulations (200-800 for stronger play)
- Train for more iterations (100+)
- Use larger network if you have GPU
- Implement better state encoding (see TODOs in code)

## 🔧 Extending the Project

### Better State Representation

The current state encoding is basic. Consider adding:
- Full board topology (hex positions, numbers)
- Building positions for each player
- Development card information
- Trade history

### Improved Action Encoding

The action encoder needs refinement for:
- Proper mapping of all Catan actions
- Efficient handling of variable action space

### Advanced Features

- Parallel self-play for faster data generation
- Curriculum learning (start with simpler games)
- ELO tracking across training

## 📚 References

- [AlphaZero Paper](https://arxiv.org/abs/1712.01815)
- [Catanatron](https://github.com/bcollazo/catanatron)
- [MCTS Tutorial](https://www.youtube.com/watch?v=UXW2yZndl7U)

## 📝 License

MIT License - Use freely for learning and research!
