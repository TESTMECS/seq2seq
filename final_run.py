#!/usr/bin/env python
"""
Final version of the script that demonstrates both models with and without attention.
Uses synthetic data to ensure everything works properly.
"""

import os
import torch
import torch.optim as optim
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, TensorDataset

from model import Encoder, Decoder, Attention, Seq2Seq, Seq2SeqLoss


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SimpleDataset(Dataset):
    """Simple dataset for sequence-to-sequence task."""

    def __init__(self, sources, targets):
        """Initialize dataset with tokenized source and target sentences."""
        self.sources = sources
        self.targets = targets
        assert len(sources) == len(
            targets
        ), "Source and target lists must have the same length"

    def __len__(self) -> int:
        """Return the number of sentence pairs."""
        return len(self.sources)

    def __getitem__(self, idx: int):
        """Return a single source-target pair."""
        return self.sources[idx], self.targets[idx]


def collate_fn(batch):
    """Custom collate function to pad sequences in a batch."""
    # Separate sources and targets
    sources, targets = zip(*batch)

    # Convert to tensors and pad
    src_tensors = [torch.tensor(s, dtype=torch.long) for s in sources]
    tgt_tensors = [torch.tensor(t, dtype=torch.long) for t in targets]

    # Padding
    src_padded = pad_sequence(src_tensors, batch_first=True, padding_value=0)
    tgt_padded = pad_sequence(tgt_tensors, batch_first=True, padding_value=0)

    # Create source and target masks (1 for non-pad, 0 for pad)
    src_mask = (src_padded != 0).float()
    tgt_mask = (tgt_padded != 0).float()

    return {
        "src": src_padded,
        "tgt": tgt_padded,
        "src_mask": src_mask,
        "tgt_mask": tgt_mask,
        "src_lengths": torch.tensor([len(s) for s in sources], dtype=torch.long),
        "tgt_lengths": torch.tensor([len(t) for t in targets], dtype=torch.long),
    }


def create_model(
    src_vocab_size: int,
    tgt_vocab_size: int,
    device: torch.device,
    use_attention: bool = False,
    hidden_dim: int = 64,
    emb_dim: int = 32,
    n_layers: int = 1,
    dropout: float = 0.2,
) -> Seq2Seq:
    """Create a sequence-to-sequence model."""
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


def create_synthetic_data(
    num_samples=1000, src_vocab_size=100, tgt_vocab_size=100, max_len=10
):
    """Create synthetic data for training and testing."""
    src_data = []
    tgt_data = []

    for _ in range(num_samples):
        # Create random source sequence (length 3-10)
        src_len = random.randint(3, max_len)
        src_seq = [random.randint(1, src_vocab_size - 1) for _ in range(src_len)]

        # Create random target sequence with BOS(1) and EOS(2) tokens (length 3-10)
        tgt_len = random.randint(3, max_len)
        tgt_seq = (
            [1] + [random.randint(3, tgt_vocab_size - 1) for _ in range(tgt_len)] + [2]
        )

        src_data.append(src_seq)
        tgt_data.append(tgt_seq)

    # Split into train, val, test
    train_size = int(0.7 * num_samples)
    val_size = int(0.15 * num_samples)

    train_src = src_data[:train_size]
    train_tgt = tgt_data[:train_size]

    val_src = src_data[train_size : train_size + val_size]
    val_tgt = tgt_data[train_size : train_size + val_size]

    test_src = src_data[train_size + val_size :]
    test_tgt = tgt_data[train_size + val_size :]

    return train_src, train_tgt, val_src, val_tgt, test_src, test_tgt


def train_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    clip,
    device,
    teacher_forcing_ratio=0.5,
):
    """Train for one epoch."""
    model.train()
    epoch_loss = 0
    total_batches = len(dataloader)

    for batch_idx, batch in enumerate(dataloader):
        # Get data
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

        # Update epoch loss
        batch_loss = loss.item()
        epoch_loss += batch_loss

        # Print progress every few batches
        if (
            (batch_idx + 1) % 10 == 0
            or batch_idx == 0
            or batch_idx == total_batches - 1
        ):
            avg_loss = epoch_loss / (batch_idx + 1)
            print(
                f"Batch {batch_idx+1}/{total_batches} | Loss: {batch_loss:.4f} | Avg Loss: {avg_loss:.4f}"
            )

    avg_epoch_loss = epoch_loss / total_batches
    print(f"Epoch completed | Avg Loss: {avg_epoch_loss:.4f}")
    return avg_epoch_loss


def evaluate(
    model,
    dataloader,
    criterion,
    device,
):
    """Evaluate the model on the validation set."""
    model.eval()
    epoch_loss = 0
    total_batches = len(dataloader)

    with torch.no_grad():
        for batch in dataloader:
            # Get data
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            src_mask = batch["src_mask"].to(device)
            src_lengths = batch["src_lengths"].to(device)

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
            epoch_loss += loss.item()

    avg_loss = epoch_loss / total_batches
    print(f"Validation | Avg Loss: {avg_loss:.4f}")
    return avg_loss


def train(
    model,
    train_dataloader,
    val_dataloader,
    optimizer,
    criterion,
    clip,
    device,
    n_epochs,
    patience=5,
    teacher_forcing_ratio=0.5,
    model_path=None,
):
    """Train the model for multiple epochs."""
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}/{n_epochs}")

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

        # Check if this is the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if model_path:
                torch.save(model.state_dict(), model_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"Early stopping after {epoch+1} epochs")
            break

    return history


def plot_losses(history, save_path=None, show_plot=False):
    """Plot training and validation losses."""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(history["train_loss"]) + 1)

    # Plot train and validation loss
    plt.plot(epochs, history["train_loss"], "bo-", label="Train Loss")
    plt.plot(epochs, history["val_loss"], "ro-", label="Validation Loss")

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
    for i, (train_loss, val_loss) in enumerate(
        zip(history["train_loss"], history["val_loss"])
    ):
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


def generate_translation(model, sentence, device, max_len=50):
    """Generate a translation for a sentence (just a test function)."""
    # Convert sentence to tensor (simple tokenization by splitting on spaces)
    tokens = [int(random.random() * 100) for _ in range(len(sentence.split()))]
    src_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
    src_mask = torch.ones_like(src_tensor, dtype=torch.float).to(device)
    src_lengths = torch.tensor([len(tokens)], dtype=torch.long).to(device)

    # Generate translation
    model.eval()
    with torch.no_grad():
        bos_idx = 1  # Beginning of sentence token
        eos_idx = 2  # End of sentence token

        # Translate
        translation_dict = model.translate(
            src=src_tensor,
            src_lengths=src_lengths,
            src_mask=src_mask,
            max_len=max_len,
            bos_idx=bos_idx,
            eos_idx=eos_idx,
        )

        # Get translated tokens
        pred_tokens = translation_dict["translations"][0].cpu().numpy()

        # Remove padding, BOS and EOS tokens
        pred_tokens = pred_tokens[pred_tokens != 0]
        pred_tokens = pred_tokens[1:]  # Remove BOS
        if eos_idx in pred_tokens:
            pred_tokens = pred_tokens[: np.where(pred_tokens == eos_idx)[0][0]]

        # This is just a simulation of translation since we're using random data
        return " ".join([f"word_{t}" for t in pred_tokens])


def main():
    # Configuration
    src_vocab_size = 100
    tgt_vocab_size = 100
    batch_size = 32
    hidden_dim = 64
    emb_dim = 32
    n_layers = 1
    dropout = 0.2
    n_epochs = 3
    clip = 1.0
    lr = 0.001
    patience = 5
    tf_ratio = 0.5
    seed = 42
    device = torch.device("cpu")  # Use CPU for consistency
    save_dir = "models"

    # Create directories
    os.makedirs(save_dir, exist_ok=True)

    # Set seed
    set_seed(seed)

    print(f"Using device: {device}")

    # Create synthetic data
    print("Creating synthetic data...")
    train_src, train_tgt, val_src, val_tgt, test_src, test_tgt = create_synthetic_data(
        num_samples=2000,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        max_len=10,
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
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Define model paths
    model_no_attn_path = os.path.join(save_dir, "model_no_attn_final.pt")
    model_attn_path = os.path.join(save_dir, "model_attn_final.pt")

    # Define pad index for loss function
    pad_idx = 0  # We use 0 as padding index in our synthetic data

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
    history_no_attn = train(
        model=model_no_attn,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer_no_attn,
        criterion=criterion,
        clip=clip,
        device=device,
        n_epochs=n_epochs,
        patience=patience,
        teacher_forcing_ratio=tf_ratio,
        model_path=model_no_attn_path,
    )

    # Plot losses for model without attention
    plot_losses(
        history=history_no_attn,
        save_path=os.path.join(save_dir, "losses_no_attn_final.png"),
        show_plot=True,  # Show the plot
    )

    print("\n" + "=" * 50)
    print("Training model with attention...")
    print("=" * 50 + "\n")

    # Create optimizer for model with attention
    optimizer_attn = optim.Adam(model_attn.parameters(), lr=lr)

    # Train model with attention
    history_attn = train(
        model=model_attn,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        optimizer=optimizer_attn,
        criterion=criterion,
        clip=clip,
        device=device,
        n_epochs=n_epochs,
        patience=patience,
        teacher_forcing_ratio=tf_ratio,
        model_path=model_attn_path,
    )

    # Plot losses for model with attention
    plot_losses(
        history=history_attn,
        save_path=os.path.join(save_dir, "losses_attn_final.png"),
        show_plot=True,  # Show the plot
    )

    # Example translations
    print("\n" + "=" * 50)
    print("Example translations...")
    print("=" * 50 + "\n")

    # Some example sentences
    example_sentences = [
        "Hello, how are you?",
        "I love programming in Python.",
        "The weather is nice today.",
        "What time is it?",
        "Can you help me with this translation?",
    ]

    # Translate examples
    print(f"{'Source':<30} | {'No Attention':<30} | {'With Attention':<30}")
    print(f"{'-' * 30}+{'-' * 31}+{'-' * 31}")

    for sentence in example_sentences:
        # Translate with model without attention
        translation_no_attn = generate_translation(
            model=model_no_attn,
            sentence=sentence,
            device=device,
        )

        # Translate with model with attention
        translation_attn = generate_translation(
            model=model_attn,
            sentence=sentence,
            device=device,
        )

        # Print results
        print(
            f"{sentence[:28]:<30} | {translation_no_attn[:28]:<30} | {translation_attn[:28]:<30}"
        )


if __name__ == "__main__":
    main()

