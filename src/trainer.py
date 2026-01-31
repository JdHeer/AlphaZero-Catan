"""
Training module for the AlphaZero neural network.

Implements the training loop that:
1. Loads self-play data
2. Trains the network on (state, policy, value) examples
3. Saves checkpoints
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class CatanDataset(Dataset):
    """PyTorch dataset for Catan training examples."""

    def __init__(
        self,
        states: np.ndarray,
        policies: np.ndarray,
        values: np.ndarray,
    ):
        self.states = torch.FloatTensor(states)
        self.policies = torch.FloatTensor(policies)
        self.values = torch.FloatTensor(values)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.policies[idx], self.values[idx]


class AlphaZeroTrainer:
    """
    Trainer for AlphaZero-style learning.
    """

    def __init__(
        self,
        network: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        epochs_per_iteration: int = 10,
        checkpoint_dir: str = "checkpoints",
        device: str = None,
    ):
        self.network = network
        self.batch_size = batch_size
        self.epochs_per_iteration = epochs_per_iteration
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.network.to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            network.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=100,
            gamma=0.9,
        )

        # Training history
        self.history = {
            "policy_loss": [],
            "value_loss": [],
            "total_loss": [],
            "iterations": 0,
        }

    def train(
        self,
        states: np.ndarray,
        policies: np.ndarray,
        values: np.ndarray,
        verbose: bool = True,
    ) -> dict:
        """
        Train the network on a batch of examples.

        Args:
            states: (N, state_size) array of game states
            policies: (N, action_size) array of MCTS policies
            values: (N,) array of game outcomes
            verbose: Print progress

        Returns:
            Dictionary of training metrics
        """
        # Create dataset and loader
        dataset = CatanDataset(states, policies, values)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        self.network.train()

        epoch_metrics = {
            "policy_loss": [],
            "value_loss": [],
            "total_loss": [],
        }

        for epoch in range(self.epochs_per_iteration):
            epoch_policy_loss = 0
            epoch_value_loss = 0
            epoch_total_loss = 0
            num_batches = 0

            for batch_states, batch_policies, batch_values in loader:
                # Move to device
                batch_states = batch_states.to(self.device)
                batch_policies = batch_policies.to(self.device)
                batch_values = batch_values.to(self.device)

                # Forward pass
                policy_logits, value_pred = self.network(batch_states)

                # Policy loss (cross-entropy with MCTS policy)
                # Use KL divergence / cross-entropy
                log_probs = torch.log_softmax(policy_logits, dim=1)
                policy_loss = -torch.sum(batch_policies * log_probs, dim=1).mean()

                # Value loss (MSE)
                value_loss = nn.MSELoss()(value_pred.squeeze(), batch_values)

                # Total loss
                total_loss = policy_loss + value_loss

                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)

                self.optimizer.step()

                # Accumulate metrics
                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                epoch_total_loss += total_loss.item()
                num_batches += 1

            # Average metrics for epoch
            avg_policy = epoch_policy_loss / num_batches
            avg_value = epoch_value_loss / num_batches
            avg_total = epoch_total_loss / num_batches

            epoch_metrics["policy_loss"].append(avg_policy)
            epoch_metrics["value_loss"].append(avg_value)
            epoch_metrics["total_loss"].append(avg_total)

            if verbose:
                print(f"  Epoch {epoch + 1}/{self.epochs_per_iteration}: "
                      f"Policy Loss={avg_policy:.4f}, "
                      f"Value Loss={avg_value:.4f}, "
                      f"Total={avg_total:.4f}")

        # Update scheduler
        self.scheduler.step()

        # Update history
        self.history["policy_loss"].append(np.mean(epoch_metrics["policy_loss"]))
        self.history["value_loss"].append(np.mean(epoch_metrics["value_loss"]))
        self.history["total_loss"].append(np.mean(epoch_metrics["total_loss"]))
        self.history["iterations"] += 1

        return epoch_metrics

    def save_checkpoint(self, name: str = None):
        """Save model checkpoint."""
        if name is None:
            name = f"checkpoint_{self.history['iterations']:04d}"

        checkpoint_path = self.checkpoint_dir / f"{name}.pt"

        torch.save({
            "model_state_dict": self.network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "history": self.history,
        }, checkpoint_path)

        print(f"Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.network.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.history = checkpoint["history"]

        print(f"Loaded checkpoint: {path}")
        print(f"  Iterations: {self.history['iterations']}")

    def save_history(self, path: str = None):
        """Save training history to JSON."""
        if path is None:
            path = self.checkpoint_dir / "training_history.json"

        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)


class TrainingPipeline:
    """
    Full AlphaZero training pipeline.

    Coordinates self-play and training iterations.
    """

    def __init__(
        self,
        game_wrapper_class,
        network,
        mcts_class,
        action_encoder,
        self_play_class,
        trainer: AlphaZeroTrainer,
        games_per_iteration: int = 100,
        mcts_simulations: int = 100,
        num_players: int = 4,
    ):
        self.game_wrapper_class = game_wrapper_class
        self.network = network
        self.mcts_class = mcts_class
        self.action_encoder = action_encoder
        self.self_play_class = self_play_class
        self.trainer = trainer
        self.games_per_iteration = games_per_iteration
        self.mcts_simulations = mcts_simulations
        self.num_players = num_players

        # Data buffer (stores recent games)
        self.replay_buffer = []
        self.max_buffer_size = 10000  # Max training examples to keep

    def run_iteration(self, verbose: bool = True) -> dict:
        """
        Run one iteration of self-play + training.

        Returns:
            Dictionary of metrics
        """
        iteration = self.trainer.history["iterations"] + 1
        print(f"\n{'='*50}")
        print(f"ITERATION {iteration}")
        print(f"{'='*50}")

        # Phase 1: Self-play
        print(f"\nPhase 1: Self-Play ({self.games_per_iteration} games)")
        print("-" * 40)

        self_play = self.self_play_class(
            game_wrapper_class=self.game_wrapper_class,
            network=self.network,
            mcts_class=self.mcts_class,
            action_encoder=self.action_encoder,
            num_simulations=self.mcts_simulations,
            num_players=self.num_players,
        )

        games = self_play.generate_games(
            num_games=self.games_per_iteration,
            verbose=verbose,
            use_mcts=False,  # Fast mode: use network directly
            max_moves=800,   # Limit game length
        )

        # Convert to training data
        states, policies, values = self.self_play_class.games_to_training_data(games)

        # Add to replay buffer
        for i in range(len(states)):
            self.replay_buffer.append((states[i], policies[i], values[i]))

        # Trim buffer if needed
        if len(self.replay_buffer) > self.max_buffer_size:
            self.replay_buffer = self.replay_buffer[-self.max_buffer_size:]

        print(f"\nCollected {len(states)} training examples")
        print(f"Replay buffer size: {len(self.replay_buffer)}")

        # Phase 2: Training
        print("\nPhase 2: Training")
        print("-" * 40)

        # Sample from replay buffer
        buffer_states = np.array([x[0] for x in self.replay_buffer])
        buffer_policies = np.array([x[1] for x in self.replay_buffer])
        buffer_values = np.array([x[2] for x in self.replay_buffer])

        metrics = self.trainer.train(
            buffer_states,
            buffer_policies,
            buffer_values,
            verbose=verbose,
        )

        # Save checkpoint
        self.trainer.save_checkpoint()
        self.trainer.save_history()

        return {
            "games_played": len(games),
            "examples_collected": len(states),
            "buffer_size": len(self.replay_buffer),
            "training_metrics": metrics,
        }

    def run(self, num_iterations: int, verbose: bool = True):
        """
        Run multiple training iterations.
        """
        for i in range(num_iterations):
            metrics = self.run_iteration(verbose=verbose)

            print(f"\nIteration {i + 1} complete:")
            print(f"  Games: {metrics['games_played']}")
            print(f"  Examples: {metrics['examples_collected']}")
            print(f"  Final Loss: {metrics['training_metrics']['total_loss'][-1]:.4f}")


if __name__ == "__main__":
    print("Trainer module loaded successfully")
