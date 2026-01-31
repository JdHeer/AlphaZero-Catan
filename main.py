"""
AlphaZero Catan - Main Training Script

This script orchestrates the full AlphaZero training pipeline:
1. Initialize network and components
2. Run self-play to generate training data
3. Train the neural network
4. Evaluate against baselines
5. Repeat
"""
import argparse
from pathlib import Path

import torch
import numpy as np

from src.game_wrapper import CatanGameWrapper, get_state_size
from src.neural_network import CatanNetwork, ActionEncoder
from src.mcts import MCTS
from src.self_play import SelfPlay
from src.trainer import AlphaZeroTrainer, TrainingPipeline
from src.evaluate import run_evaluation, RandomAgent, AlphaZeroAgent


def parse_args():
    parser = argparse.ArgumentParser(description="AlphaZero Catan Training")
    
    # Training parameters
    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of training iterations")
    parser.add_argument("--games-per-iter", type=int, default=50,
                        help="Self-play games per iteration")
    parser.add_argument("--mcts-sims", type=int, default=100,
                        help="MCTS simulations per move")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Training epochs per iteration")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    
    # Network architecture
    parser.add_argument("--hidden-size", type=int, default=256,
                        help="Hidden layer size")
    parser.add_argument("--num-layers", type=int, default=4,
                        help="Number of hidden layers")
    
    # Game parameters
    parser.add_argument("--num-players", type=int, default=4,
                        help="Number of players (2-4)")
    
    # Checkpoints
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Directory for saving checkpoints")
    parser.add_argument("--load-checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")
    
    # Evaluation
    parser.add_argument("--eval-games", type=int, default=50,
                        help="Number of evaluation games")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only run evaluation, no training")
    
    # Misc
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*60)
    print("  AlphaZero Catan Training")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Iterations: {args.iterations}")
    print(f"  Games per iteration: {args.games_per_iter}")
    print(f"  MCTS simulations: {args.mcts_sims}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Players: {args.num_players}")
    
    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    
    # Initialize components
    print("\nInitializing components...")
    
    # Action encoder
    action_encoder = ActionEncoder()
    
    # Network
    state_size = get_state_size(args.num_players)
    action_size = action_encoder.action_space_size
    
    print(f"  State size: {state_size}")
    print(f"  Action size: {action_size}")
    
    network = CatanNetwork(
        state_size=state_size,
        action_size=action_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
    )
    
    # Trainer
    trainer = AlphaZeroTrainer(
        network=network,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        epochs_per_iteration=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        device=str(device),
    )
    
    # Load checkpoint if specified
    if args.load_checkpoint:
        trainer.load_checkpoint(args.load_checkpoint)
    
    # Evaluation only mode
    if args.eval_only:
        print("\n" + "="*60)
        print("  Running Evaluation Only")
        print("="*60)
        
        results = run_evaluation(
            network=network,
            action_encoder=action_encoder,
            game_wrapper_class=CatanGameWrapper,
            mcts_class=MCTS,
            num_games=args.eval_games,
        )
        return
    
    # Full training pipeline
    pipeline = TrainingPipeline(
        game_wrapper_class=CatanGameWrapper,
        network=network,
        mcts_class=MCTS,
        action_encoder=action_encoder,
        self_play_class=SelfPlay,
        trainer=trainer,
        games_per_iteration=args.games_per_iter,
        mcts_simulations=args.mcts_sims,
        num_players=args.num_players,
    )
    
    # Run training
    print("\n" + "="*60)
    print("  Starting Training")
    print("="*60)
    
    pipeline.run(
        num_iterations=args.iterations,
        verbose=args.verbose,
    )
    
    # Final evaluation
    print("\n" + "="*60)
    print("  Final Evaluation")
    print("="*60)
    
    results = run_evaluation(
        network=network,
        action_encoder=action_encoder,
        game_wrapper_class=CatanGameWrapper,
        mcts_class=MCTS,
        num_games=args.eval_games,
    )
    
    print("\n" + "="*60)
    print("  Training Complete!")
    print("="*60)
    print(f"\nFinal AlphaZero win rate: {results['win_rates'][0]:.1%}")
    print(f"Checkpoint saved to: {args.checkpoint_dir}")


if __name__ == "__main__":
    main()
