"""
Training functions for sequence-to-sequence models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from typing import Dict, List
from tqdm import tqdm

from seq2seq.core import Seq2Seq


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    clip: float,
    device: torch.device,
    teacher_forcing_ratio: float = 0.5,
) -> float:
    """
    Train the model for one epoch.

    Args:
        model: The model to train
        dataloader: DataLoader for training data
        optimizer: Optimizer for model parameters
        criterion: Loss function
        clip: Gradient clipping value
        device: Device to use for training
        teacher_forcing_ratio: Probability of using teacher forcing

    Returns:
        epoch_loss: Average loss for the epoch
    """
    model.train()
    epoch_loss = 0
    total_batches = len(dataloader)

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for batch_idx, batch in enumerate(progress_bar):
        # Get data
        src = batch["src"].to(device)  # [batch_size, src_len]
        tgt = batch["tgt"].to(device)  # [batch_size, tgt_len]
        src_mask = batch["src_mask"].to(device)  # [batch_size, src_len]
        src_lengths = batch["src_lengths"].to(device)  # [batch_size]

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        output_dict = model(
            src=src,
            tgt=tgt,
            src_lengths=src_lengths,
            src_mask=src_mask,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )

        # Calculate loss
        output = output_dict["outputs"]
        loss = criterion(output, tgt)

        # Backward pass
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        # Update parameters
        optimizer.step()

        # Update epoch loss
        batch_loss = loss.item()
        epoch_loss += batch_loss

        # Update progress bar
        progress_bar.set_postfix(
            {
                "batch": f"{batch_idx + 1}/{total_batches}",
                "loss": f"{batch_loss:.4f}",
                "avg_loss": f"{epoch_loss / (batch_idx + 1):.4f}",
            }
        )

        # Print detailed batch info every 10 batches
        if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
            avg_loss = epoch_loss / (batch_idx + 1)
            print(
                f"Batch {batch_idx + 1}/{total_batches} | Loss: {batch_loss:.4f} | Avg Loss: {avg_loss:.4f}"
            )

    avg_epoch_loss = epoch_loss / total_batches
    print(f"Epoch completed | Avg Loss: {avg_epoch_loss:.4f}")
    return avg_epoch_loss


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Evaluate the model on validation or test data.

    Args:
        model: The model to evaluate
        dataloader: DataLoader for evaluation data
        criterion: Loss function
        device: Device to use for evaluation

    Returns:
        epoch_loss: Average loss for the evaluation
    """
    model.eval()
    epoch_loss = 0
    total_batches = len(dataloader)

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)

        for batch_idx, batch in enumerate(progress_bar):
            # Get data
            src = batch["src"].to(device)  # [batch_size, src_len]
            tgt = batch["tgt"].to(device)  # [batch_size, tgt_len]
            src_mask = batch["src_mask"].to(device)  # [batch_size, src_len]
            src_lengths = batch["src_lengths"].to(device)  # [batch_size]

            # Forward pass
            output_dict = model(
                src=src,
                tgt=tgt,
                src_lengths=src_lengths,
                src_mask=src_mask,
                teacher_forcing_ratio=0.0,  # No teacher forcing during evaluation
            )

            # Calculate loss
            output = output_dict["outputs"]
            loss = criterion(output, tgt)

            # Update epoch loss
            batch_loss = loss.item()
            epoch_loss += batch_loss

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "batch": f"{batch_idx + 1}/{total_batches}",
                    "loss": f"{batch_loss:.4f}",
                    "avg_loss": f"{epoch_loss / (batch_idx + 1):.4f}",
                }
            )

    avg_loss = epoch_loss / total_batches
    print(f"Validation | Avg Loss: {avg_loss:.4f}")
    return avg_loss


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    clip: float,
    device: torch.device,
    n_epochs: int,
    patience: int = 5,
    teacher_forcing_ratio: float = 0.5,
    model_path: str = "model.pt",
) -> Dict[str, List[float]]:
    """
    Train the model for multiple epochs.

    Args:
        model: The model to train
        train_dataloader: DataLoader for training data
        val_dataloader: DataLoader for validation data
        optimizer: Optimizer for model parameters
        criterion: Loss function
        clip: Gradient clipping value
        device: Device to use for training
        n_epochs: Number of epochs to train for
        patience: Number of epochs to wait for improvement before early stopping
        teacher_forcing_ratio: Probability of using teacher forcing
        model_path: Path to save the best model

    Returns:
        history: Dictionary containing training and validation losses
    """
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(n_epochs):
        start_time = time.time()

        # Train
        train_loss = train_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            criterion=criterion,
            clip=clip,
            device=device,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )

        # Evaluate
        val_loss = evaluate(
            model=model,
            dataloader=val_dataloader,
            criterion=criterion,
            device=device,
        )

        # Save losses
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # Calculate time
        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

        # Check if this is the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Print progress
        print(f"Epoch: {epoch + 1}/{n_epochs}")
        print(f"Time: {int(epoch_mins)}m {int(epoch_secs)}s")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Best Val Loss: {best_val_loss:.4f}")

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"Early stopping after {epoch + 1} epochs")
            break

    return history
