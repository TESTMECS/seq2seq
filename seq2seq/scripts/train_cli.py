"""
Command-line interface for training sequence-to-sequence models.

"""

import argparse
import torch
import torch.optim as optim
import os
import random
import numpy as np

from seq2seq.core import Encoder, Decoder, Attention, Seq2Seq, Seq2SeqLoss
from seq2seq.data import prepare_data
from seq2seq.training import train, plot_losses, test_model


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def create_model(
    src_vocab_size: int,
    tgt_vocab_size: int,
    hidden_dim: int,
    emb_dim: int,
    n_layers: int,
    dropout: float,
    use_attention: bool,
    device: torch.device,
) -> Seq2Seq:
    """Create sequence-to-sequence model."""
    # Create encoder
    encoder = Encoder(
        input_dim=src_vocab_size,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=dropout,
    )

    # Create attention if needed
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

    # Create model
    model = Seq2Seq(
        encoder=encoder,
        decoder=decoder,
        device=device,
        use_attention=use_attention,
    )

    # Print model information
    print(f"Created {'attention-based' if use_attention else 'basic'} Seq2Seq model")
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")

    return model


def main():
    """Main entry point for training."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train a sequence-to-sequence model")
    parser.add_argument("--src_lang", type=str, default="en", help="Source language")
    parser.add_argument("--tgt_lang", type=str, default="fr", help="Target language")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum sequence length")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension size")
    parser.add_argument("--emb_dim", type=int, default=128, help="Embedding dimension size")
    parser.add_argument(
        "--n_layers",
        type=int,
        default=2,
        help="Number of layers in encoder and decoder",
    )
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--clip", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
    parser.add_argument("--tf_ratio", type=float, default=0.5, help="Teacher forcing ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")
    parser.add_argument("--save_dir", type=str, default="models", help="Directory to save models")
    parser.add_argument(
        "--test_only", action="store_true", help="Only test the model (no training)"
    )
    parser.add_argument("--attention", action="store_true", help="Use attention mechanism")
    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Set device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)

    # Prepare data
    print("Preparing data...")
    data_info = prepare_data(
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    # Extract data loaders and information
    train_dataloader = data_info["train_dataloader"]
    val_dataloader = data_info["val_dataloader"]
    test_dataloader = data_info["test_dataloader"]
    src_tokenizer = data_info["src_tokenizer"]
    tgt_tokenizer = data_info["tgt_tokenizer"]
    src_vocab_size = data_info["src_vocab_size"]
    tgt_vocab_size = data_info["tgt_vocab_size"]
    pad_idx = data_info["pad_idx"]

    # Model paths
    model_name = f"model_{'attn' if args.attention else 'no_attn'}.pt"
    model_path = os.path.join(args.save_dir, model_name)

    # Create model
    model = create_model(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        hidden_dim=args.hidden_dim,
        emb_dim=args.emb_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
        use_attention=args.attention,
        device=device,
    )
    model = model.to(device)

    # Create loss function
    # If pad_idx is None, use -100 (PyTorch's default ignore_index)
    if pad_idx is None:
        pad_idx = -100
        print(
            "Warning: [PAD] token not found in target tokenizer. Using default ignore_index (-100)."
        )
    criterion = Seq2SeqLoss(pad_idx=pad_idx)

    # Test-only mode
    if args.test_only:
        print(f"Loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        test_loss, bleu = test_model(
            model=model,
            test_dataloader=test_dataloader,
            criterion=criterion,
            device=device,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
        )
        print(f"Test Loss: {test_loss:.4f}")
        print(f"BLEU Score: {bleu:.4f}")
        return

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Train model
    print("Training model...")
    history = train(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        criterion=criterion,
        clip=args.clip,
        device=device,
        n_epochs=args.n_epochs,
        patience=args.patience,
        teacher_forcing_ratio=args.tf_ratio,
        model_path=model_path,
    )

    # Plot losses
    plot_path = os.path.join(args.save_dir, f"losses_{'attn' if args.attention else 'no_attn'}.png")
    plot_losses(history, save_path=plot_path)

    # Test model
    print("Testing model...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    test_loss, bleu = test_model(
        model=model,
        test_dataloader=test_dataloader,
        criterion=criterion,
        device=device,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
    )
    print(f"Test Loss: {test_loss:.4f}")
    print(f"BLEU Score: {bleu:.4f}")


if __name__ == "__main__":
    main()
