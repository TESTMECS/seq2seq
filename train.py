"""
Training and evaluation functions for the sequence-to-sequence model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import Seq2Seq, Seq2SeqLoss
from utils import bleu_score, translate_sentence


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
        progress_bar.set_postfix({
            'batch': f'{batch_idx+1}/{total_batches}',
            'loss': f'{batch_loss:.4f}',
            'avg_loss': f'{epoch_loss/(batch_idx+1):.4f}'
        })
        
        # Print detailed batch info every 10 batches
        if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
            avg_loss = epoch_loss / (batch_idx + 1)
            print(f"Batch {batch_idx+1}/{total_batches} | Loss: {batch_loss:.4f} | Avg Loss: {avg_loss:.4f}")

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
            progress_bar.set_postfix({
                'batch': f'{batch_idx+1}/{total_batches}',
                'loss': f'{batch_loss:.4f}',
                'avg_loss': f'{epoch_loss/(batch_idx+1):.4f}'
            })
    
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


def plot_losses(history: Dict[str, List[float]], save_path: Optional[str] = None, show_plot: bool = False) -> None:
    """
    Plot training and validation losses.

    Args:
        history: Dictionary containing training and validation losses
        save_path: Path to save the plot image
        show_plot: Whether to display the plot interactively
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(history["train_loss"]) + 1)
    
    # Plot train and validation loss
    plt.plot(epochs, history["train_loss"], 'bo-', label="Train Loss")
    plt.plot(epochs, history["val_loss"], 'ro-', label="Validation Loss")
    
    # Add epoch numbers to x-axis
    plt.xticks(epochs)
    
    # Add grid and styling
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    # Print losses to console
    print("\nLoss per epoch:")
    print(f"{'Epoch':<6} | {'Train Loss':<12} | {'Val Loss':<12}")
    print(f"{'-'*6}+{'-'*14}+{'-'*14}")
    for i, (train_loss, val_loss) in enumerate(zip(history["train_loss"], history["val_loss"])):
        print(f"{i+1:<6} | {train_loss:<12.4f} | {val_loss:<12.4f}")

    # Save the plot
    if save_path:
        plt.savefig(save_path)
        print(f"\nLoss plot saved to: {save_path}")

    # Show the plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()


def test_model(
    model: nn.Module,
    test_dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    src_tokenizer: Any,
    tgt_tokenizer: Any,
    num_examples: int = 20,
) -> Tuple[float, float]:
    """
    This is not a pytest test function, it's a function to test the model on test data.
    """
    """
    Test the model on test data and compute BLEU score.

    Args:
        model: The model to test
        test_dataloader: DataLoader for test data
        criterion: Loss function
        device: Device to use for testing
        src_tokenizer: Tokenizer for source language
        tgt_tokenizer: Tokenizer for target language
        num_examples: Number of examples to print

    Returns:
        test_loss: Average loss for the test data
        bleu: BLEU score for the test data
    """
    model.eval()
    test_loss = 0

    # Load all test data for BLEU calculation
    all_sources = []
    all_targets = []
    all_predictions = []

    examples = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing"):
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
                teacher_forcing_ratio=0.0,  # No teacher forcing during testing
            )

            # Calculate loss
            output = output_dict["outputs"]
            loss = criterion(output, tgt)

            # Update test loss
            test_loss += loss.item()

            # Translate batch
            max_length = tgt.shape[1]
            bos_idx = tgt_tokenizer.token_to_id("[BOS]")
            eos_idx = tgt_tokenizer.token_to_id("[EOS]")

            translation_dict = model.translate(
                src=src,
                src_lengths=src_lengths,
                src_mask=src_mask,
                max_len=max_length,
                bos_idx=bos_idx,
                eos_idx=eos_idx,
            )

            translations = translation_dict["translations"]

            # Convert token ids to sentences
            for i in range(src.shape[0]):
                # Source text
                src_tokens = src[i].cpu().numpy()
                src_tokens = src_tokens[src_tokens != 0]
                src_text = src_tokenizer.decode(src_tokens.tolist())

                # Target text
                tgt_tokens = tgt[i].cpu().numpy()
                tgt_tokens = tgt_tokens[tgt_tokens != 0]
                tgt_text = tgt_tokenizer.decode(tgt_tokens.tolist())

                # Predicted text
                pred_tokens = translations[i].cpu().numpy()
                # Remove padding, BOS and EOS tokens
                pred_tokens = pred_tokens[pred_tokens != 0]
                pred_tokens = pred_tokens[1:]  # Remove BOS
                if eos_idx in pred_tokens:
                    pred_tokens = pred_tokens[: np.where(pred_tokens == eos_idx)[0][0]]
                pred_text = tgt_tokenizer.decode(pred_tokens.tolist())

                # Add to lists for BLEU calculation
                all_sources.append(src_text)
                all_targets.append(tgt_text)
                all_predictions.append(pred_text)

                # Store example for printing
                if len(examples) < num_examples:
                    examples.append((src_text, tgt_text, pred_text))

    # Calculate BLEU score
    bleu = bleu_score(all_targets, all_predictions)

    # Print examples
    print(f"\n{'=' * 50}")
    print(f"{'Source':<20} | {'Target':<20} | {'Prediction':<20}")
    print(f"{'-' * 20}+{'-' * 22}+{'-' * 22}")

    for src_text, tgt_text, pred_text in examples:
        print(f"{src_text[:18]:<20} | {tgt_text[:18]:<20} | {pred_text[:18]:<20}")

    print(f"{'=' * 50}\n")

    # Average loss
    test_loss = test_loss / len(test_dataloader)

    return test_loss, bleu
