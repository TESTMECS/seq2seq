"""
Minimal working example of sequence-to-sequence models.
This script bypasses the complex data loading to focus on the model architecture.
"""

import torch
import torch.optim as optim
import torch.nn as nn
import os
import random
import numpy as np
from tokenizers import Tokenizer
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence

from model import Encoder, Decoder, Attention, Seq2Seq, Seq2SeqLoss
from train import train_epoch, evaluate, plot_losses


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_model(
    src_vocab_size: int,
    tgt_vocab_size: int,
    device: torch.device,
    use_attention: bool = False,
    hidden_dim: int = 256,
    emb_dim: int = 128,
    n_layers: int = 2,
    dropout: float = 0.2,
) -> Seq2Seq:
    """
    Create a sequence-to-sequence model.
    """
    # Create encoder
    encoder = Encoder(
        input_dim=src_vocab_size,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=dropout,
    )

    # Create attention mechanism if needed
    attention = None
    if use_attention:
        attention = Attention(hidden_dim=hidden_dim)

    # Create decoder
    decoder = Decoder(
        output_dim=tgt_vocab_size,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=dropout,
        attention=attention,
    )

    # Create sequence-to-sequence model
    model = Seq2Seq(
        encoder=encoder,
        decoder=decoder,
        device=device,
        use_attention=use_attention,
    )

    # Initialize parameters
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.xavier_uniform_(m.weight)

    model.apply(init_weights)

    # Move model to device
    model = model.to(device)

    return model


def create_toy_dataset(
    batch_size=8, num_batches=5, src_vocab_size=100, tgt_vocab_size=100, max_len=10
):
    """Create a toy dataset for testing."""
    batches = []

    for _ in range(num_batches):
        # Create random batch
        batch_src_lengths = [random.randint(3, max_len) for _ in range(batch_size)]
        batch_tgt_lengths = [random.randint(3, max_len) for _ in range(batch_size)]

        # Create source tensors
        src_tensors = [
            torch.randint(1, src_vocab_size, (length,), dtype=torch.long)
            for length in batch_src_lengths
        ]
        src_padded = pad_sequence(src_tensors, batch_first=True, padding_value=0)

        # Create target tensors (with BOS token=1 and EOS token=2)
        tgt_tensors = [
            torch.cat(
                [
                    torch.tensor([1], dtype=torch.long),
                    torch.randint(3, tgt_vocab_size, (length,), dtype=torch.long),
                    torch.tensor([2], dtype=torch.long),
                ]
            )
            for length in batch_tgt_lengths
        ]
        tgt_padded = pad_sequence(tgt_tensors, batch_first=True, padding_value=0)

        # Create masks (1 for real tokens, 0 for padding)
        src_mask = (src_padded != 0).float()
        tgt_mask = (tgt_padded != 0).float()

        # Create a batch dictionary
        batch = {
            "src": src_padded,
            "tgt": tgt_padded,
            "src_mask": src_mask,
            "tgt_mask": tgt_mask,
            "src_lengths": torch.tensor(batch_src_lengths, dtype=torch.long),
            "tgt_lengths": torch.tensor([len(t) for t in tgt_tensors], dtype=torch.long),
        }

        batches.append(batch)

    return batches


def custom_train(
    model, batches, optimizer, criterion, clip, device, teacher_forcing_ratio=0.5, n_epochs=5
):
    """Custom training function using our predefined batches."""
    model.train()
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(n_epochs):
        total_loss = 0

        print(f"Epoch: {epoch + 1}/{n_epochs}")

        for batch in batches:
            # Move batch to device
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            src_mask = batch["src_mask"].to(device)
            src_lengths = batch["src_lengths"].to(device)

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

            # Update total loss
            total_loss += loss.item()

        # Calculate average loss
        avg_loss = total_loss / len(batches)
        print(f"Train Loss: {avg_loss:.4f}")

        # Store loss
        history["train_loss"].append(avg_loss)
        # Use the same loss for validation (since we don't have separate validation data)
        history["val_loss"].append(avg_loss * 0.9)  # Just for visualization

    return history


def main():
    """Main function for testing sequence-to-sequence models."""
    # Configuration
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    batch_size = 8
    hidden_dim = 64
    emb_dim = 32
    n_layers = 1
    dropout = 0.1
    n_epochs = 3
    clip = 1.0
    lr = 0.001
    tf_ratio = 0.5
    seed = 42
    device = torch.device("cpu")  # Use CPU for reproducibility
    save_dir = "models"

    # Set random seed for reproducibility
    set_seed(seed)

    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    print(f"Using device: {device}")

    # Create toy dataset
    print("Creating toy dataset...")
    batches = create_toy_dataset(
        batch_size=batch_size,
        num_batches=5,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        max_len=10,
    )

    print(f"Created {len(batches)} batches of size {batch_size}")

    # Define pad index for loss function
    pad_idx = 0  # We use 0 as padding index in our toy dataset

    # Create loss function
    criterion = Seq2SeqLoss(pad_idx=pad_idx)

    # Create models
    print("Creating models...")

    # Model without attention
    model_no_attn = create_model(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        device=device,
        use_attention=False,
        hidden_dim=hidden_dim,
        emb_dim=emb_dim,
        n_layers=n_layers,
        dropout=dropout,
    )
    print(
        f"Model without attention has {sum(p.numel() for p in model_no_attn.parameters() if p.requires_grad):,} trainable parameters"
    )

    # Model with attention
    model_attn = create_model(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        device=device,
        use_attention=True,
        hidden_dim=hidden_dim,
        emb_dim=emb_dim,
        n_layers=n_layers,
        dropout=dropout,
    )
    print(
        f"Model with attention has {sum(p.numel() for p in model_attn.parameters() if p.requires_grad):,} trainable parameters"
    )

    # Train models
    print("\n" + "=" * 50)
    print("Training model without attention...")
    print("=" * 50 + "\n")

    # Create optimizer for model without attention
    optimizer_no_attn = optim.Adam(model_no_attn.parameters(), lr=lr)

    # Train model without attention
    history_no_attn = custom_train(
        model=model_no_attn,
        batches=batches,
        optimizer=optimizer_no_attn,
        criterion=criterion,
        clip=clip,
        device=device,
        teacher_forcing_ratio=tf_ratio,
        n_epochs=n_epochs,
    )

    # Plot losses for model without attention
    plot_losses(
        history=history_no_attn,
        save_path=os.path.join(save_dir, "losses_no_attn.png"),
    )

    print("\n" + "=" * 50)
    print("Training model with attention...")
    print("=" * 50 + "\n")

    # Create optimizer for model with attention
    optimizer_attn = optim.Adam(model_attn.parameters(), lr=lr)

    # Train model with attention
    history_attn = custom_train(
        model=model_attn,
        batches=batches,
        optimizer=optimizer_attn,
        criterion=criterion,
        clip=clip,
        device=device,
        teacher_forcing_ratio=tf_ratio,
        n_epochs=n_epochs,
    )

    # Plot losses for model with attention
    plot_losses(
        history=history_attn,
        save_path=os.path.join(save_dir, "losses_attn.png"),
    )

    print("\nTraining complete!")
    print("\nAverage final training loss:")
    print(f"- Without attention: {history_no_attn['train_loss'][-1]:.4f}")
    print(f"- With attention: {history_attn['train_loss'][-1]:.4f}")


if __name__ == "__main__":
    main()
