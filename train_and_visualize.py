#!/usr/bin/env python
"""
Interactive script to train and visualize seq2seq models with different parameters.
"""

import os
import argparse
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim

from model import Encoder, Decoder, Attention, Seq2Seq, Seq2SeqLoss

# Import utilities from final_run.py
from final_run import (
    set_seed,
    SimpleDataset,
    collate_fn,
    create_model,
    create_synthetic_data,
    train_epoch,
    evaluate,
    train,
    plot_losses,
)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train and visualize sequence-to-sequence models"
    )

    # Model parameters
    parser.add_argument(
        "--src-vocab-size", type=int, default=100, help="Source vocabulary size"
    )
    parser.add_argument(
        "--tgt-vocab-size", type=int, default=100, help="Target vocabulary size"
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=64, help="Hidden dimension size"
    )
    parser.add_argument(
        "--emb-dim", type=int, default=32, help="Embedding dimension size"
    )
    parser.add_argument("--n-layers", type=int, default=1, help="Number of RNN layers")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")

    # Training parameters
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--n-epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument(
        "--clip", type=float, default=1.0, help="Gradient clipping value"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--patience", type=int, default=5, help="Early stopping patience"
    )
    parser.add_argument(
        "--tf-ratio", type=float, default=0.5, help="Teacher forcing ratio"
    )

    # Data parameters
    parser.add_argument(
        "--num-samples", type=int, default=2000, help="Number of synthetic samples"
    )
    parser.add_argument(
        "--max-len", type=int, default=10, help="Maximum sequence length"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Run options
    parser.add_argument(
        "--no-attention", action="store_true", help="Train model without attention"
    )
    parser.add_argument(
        "--attention", action="store_true", help="Train model with attention"
    )
    parser.add_argument(
        "--compare", action="store_true", help="Train and compare both models"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="models",
        help="Directory to save models and plots",
    )
    parser.add_argument(
        "--cpu", action="store_true", help="Force using CPU even if GPU is available"
    )
    parser.add_argument(
        "--show-plots", action="store_true", help="Show plots during training"
    )

    args = parser.parse_args()

    # If no model type is specified, default to comparing both
    if not (args.no_attention or args.attention or args.compare):
        args.compare = True

    return args


def train_models(args):
    """Train models based on arguments."""
    # Set up device
    device = (
        torch.device("cpu")
        if args.cpu
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # Set seed for reproducibility
    set_seed(args.seed)

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Create synthetic data
    print("Creating synthetic data...")
    train_src, train_tgt, val_src, val_tgt, test_src, test_tgt = create_synthetic_data(
        num_samples=args.num_samples,
        src_vocab_size=args.src_vocab_size,
        tgt_vocab_size=args.tgt_vocab_size,
        max_len=args.max_len,
    )

    print(f"Training examples: {len(train_src)}")
    print(f"Validation examples: {len(val_src)}")
    print(f"Test examples: {len(test_src)}")

    # Create datasets
    train_dataset = SimpleDataset(train_src, train_tgt)
    val_dataset = SimpleDataset(val_src, val_tgt)
    test_dataset = SimpleDataset(test_src, test_tgt)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Define model paths
    model_no_attn_path = os.path.join(args.save_dir, "model_no_attn_custom.pt")
    model_attn_path = os.path.join(args.save_dir, "model_attn_custom.pt")

    # Define pad index for loss function
    pad_idx = 0  # We use 0 as padding index in our synthetic data

    # Create loss function
    criterion = Seq2SeqLoss(pad_idx=pad_idx)

    # Train models
    history_no_attn = None
    history_attn = None

    if args.no_attention or args.compare:
        print("\n" + "=" * 50)
        print("Training model without attention...")
        print("=" * 50 + "\n")

        # Create model without attention
        model_no_attn = create_model(
            src_vocab_size=args.src_vocab_size,
            tgt_vocab_size=args.tgt_vocab_size,
            device=device,
            use_attention=False,
            hidden_dim=args.hidden_dim,
            emb_dim=args.emb_dim,
            n_layers=args.n_layers,
            dropout=args.dropout,
        )
        print(
            f"Model without attention has {sum(p.numel() for p in model_no_attn.parameters() if p.requires_grad):,} trainable parameters"
        )

        # Create optimizer for model without attention
        optimizer_no_attn = optim.Adam(model_no_attn.parameters(), lr=args.lr)

        # Train model without attention
        history_no_attn = train(
            model=model_no_attn,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            optimizer=optimizer_no_attn,
            criterion=criterion,
            clip=args.clip,
            device=device,
            n_epochs=args.n_epochs,
            patience=args.patience,
            teacher_forcing_ratio=args.tf_ratio,
            model_path=model_no_attn_path,
        )

        # Plot losses for model without attention
        plot_losses(
            history=history_no_attn,
            save_path=os.path.join(args.save_dir, "losses_no_attn_custom.png"),
            show_plot=args.show_plots,
        )

    if args.attention or args.compare:
        print("\n" + "=" * 50)
        print("Training model with attention...")
        print("=" * 50 + "\n")

        # Create model with attention
        model_attn = create_model(
            src_vocab_size=args.src_vocab_size,
            tgt_vocab_size=args.tgt_vocab_size,
            device=device,
            use_attention=True,
            hidden_dim=args.hidden_dim,
            emb_dim=args.emb_dim,
            n_layers=args.n_layers,
            dropout=args.dropout,
        )
        print(
            f"Model with attention has {sum(p.numel() for p in model_attn.parameters() if p.requires_grad):,} trainable parameters"
        )

        # Create optimizer for model with attention
        optimizer_attn = optim.Adam(model_attn.parameters(), lr=args.lr)

        # Train model with attention
        history_attn = train(
            model=model_attn,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            optimizer=optimizer_attn,
            criterion=criterion,
            clip=args.clip,
            device=device,
            n_epochs=args.n_epochs,
            patience=args.patience,
            teacher_forcing_ratio=args.tf_ratio,
            model_path=model_attn_path,
        )

        # Plot losses for model with attention
        plot_losses(
            history=history_attn,
            save_path=os.path.join(args.save_dir, "losses_attn_custom.png"),
            show_plot=args.show_plots,
        )

    return history_no_attn, history_attn


def create_comparison_plot(history_no_attn, history_attn, save_dir):
    """Create a comparison plot of losses for models with and without attention."""
    if history_no_attn is None or history_attn is None:
        print("Cannot create comparison plot: missing history data")
        return

    # Create figure
    plt.figure(figsize=(12, 6))
    epochs = range(
        1, max(len(history_no_attn["train_loss"]), len(history_attn["train_loss"])) + 1
    )

    # Plot training losses
    plt.subplot(1, 2, 1)
    plt.plot(
        range(1, len(history_no_attn["train_loss"]) + 1),
        history_no_attn["train_loss"],
        "b-",
        label="Without Attention",
    )
    plt.plot(
        range(1, len(history_attn["train_loss"]) + 1),
        history_attn["train_loss"],
        "r-",
        label="With Attention",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Comparison")
    plt.legend()
    plt.grid(True)

    # Plot validation losses
    plt.subplot(1, 2, 2)
    plt.plot(
        range(1, len(history_no_attn["val_loss"]) + 1),
        history_no_attn["val_loss"],
        "b-",
        label="Without Attention",
    )
    plt.plot(
        range(1, len(history_attn["val_loss"]) + 1),
        history_attn["val_loss"],
        "r-",
        label="With Attention",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss Comparison")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # Save figure
    save_path = os.path.join(save_dir, "comparison_plot_custom.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    print(f"Comparison plot saved to: {save_path}")

    plt.close()


def main():
    """Main function."""
    # Parse arguments
    args = parse_arguments()

    # Train models
    history_no_attn, history_attn = train_models(args)

    # Create comparison plot if both models were trained
    if args.compare and history_no_attn is not None and history_attn is not None:
        create_comparison_plot(history_no_attn, history_attn, args.save_dir)

    print("\nTraining and visualization complete.")
    print(f"Results saved to: {args.save_dir}")
    print("\nTo visualize the attention weights, run:\n")
    print(f"  python visualize_attention.py")


if __name__ == "__main__":
    main()

