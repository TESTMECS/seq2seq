#!/usr/bin/env python
"""
Script to retrain the seq2seq model with the current 10,000 vocabulary size tokenizers.
This is a simplified version of the main.py script focused on just retraining with the 
large vocabulary tokenizers.
"""

import argparse
import torch
import torch.optim as optim
import os
import random
import numpy as np
from tokenizers import Tokenizer

from data import prepare_data
from model import Encoder, Decoder, Attention, Seq2Seq, Seq2SeqLoss
from train import train, test_model, plot_losses
from utils import translate_sentence


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_model(
    src_vocab_size: int,
    tgt_vocab_size: int,
    device: torch.device,
    use_attention: bool = True,
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
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.Embedding):
            torch.nn.init.xavier_uniform_(m.weight)

    model.apply(init_weights)

    # Move model to device
    model = model.to(device)

    return model


def main():
    """Main function for retraining the model with the 10,000 vocabulary tokenizers."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Retrain sequence-to-sequence model with 10,000 vocabulary tokenizers"
    )

    # Data parameters
    parser.add_argument("--src_lang", type=str, default="en", help="Source language")
    parser.add_argument("--tgt_lang", type=str, default="fr", help="Target language")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum sequence length")

    # Model parameters
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension size")
    parser.add_argument("--emb_dim", type=int, default=128, help="Embedding dimension size")
    parser.add_argument(
        "--n_layers",
        type=int,
        default=2,
        help="Number of layers in encoder and decoder",
    )
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability")

    # Training parameters
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--clip", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--patience", type=int, default=3, help="Patience for early stopping")
    parser.add_argument("--tf_ratio", type=float, default=0.5, help="Teacher forcing ratio")

    # Other parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device",
    )
    parser.add_argument("--save_dir", type=str, default="models", help="Directory to save models")
    parser.add_argument(
        "--cache_dir", type=str, default="a4-data/dataset", help="Directory to cache datasets"
    )
    parser.add_argument(
        "--show_plots", action="store_true", help="Show plots during training (default: False)"
    )

    args = parser.parse_args()

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)

    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Prepare data
    print("Preparing data...")
    data_dict = prepare_data(
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        batch_size=args.batch_size,
        max_length=args.max_length,
        cache_dir=args.cache_dir,
    )

    train_loader = data_dict["train_loader"]
    val_loader = data_dict["val_loader"]
    test_loader = data_dict["test_loader"]
    src_tokenizer = data_dict["src_tokenizer"]
    tgt_tokenizer = data_dict["tgt_tokenizer"]
    src_vocab_size = data_dict["src_vocab_size"]
    tgt_vocab_size = data_dict["tgt_vocab_size"]

    print(f"Source vocabulary size: {src_vocab_size}")
    print(f"Target vocabulary size: {tgt_vocab_size}")

    # Define model paths for the large vocabulary models
    model_attn_large_vocab_path = os.path.join(args.save_dir, "model_attn_large_vocab.pt")

    # Define pad index for loss function
    pad_idx = tgt_tokenizer.token_to_id("[PAD]")
    if pad_idx is None:
        # Use default pad index if not found in tokenizer
        pad_idx = 3
        print(f"Warning: [PAD] token not found in target tokenizer. Using default (3).")

    # Create loss function
    criterion = Seq2SeqLoss(pad_idx=pad_idx)

    # Create model with attention and large vocabulary
    print("Creating model with attention for large vocabulary...")
    model_attn = create_model(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        device=device,
        use_attention=True,
        hidden_dim=args.hidden_dim,
        emb_dim=args.emb_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
    )
    print(
        f"Model with large vocabulary has {sum(p.numel() for p in model_attn.parameters() if p.requires_grad):,} trainable parameters"
    )

    # Train model
    print("\n" + "=" * 50)
    print("Training model with attention for large vocabulary...")
    print("=" * 50 + "\n")

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
        model_path=model_attn_large_vocab_path,
    )

    # Plot losses for model with attention
    plot_losses(
        history=history_attn,
        save_path=os.path.join(args.save_dir, "losses_attn_large_vocab.png"),
        show_plot=args.show_plots,
    )

    # Test model
    print("\n" + "=" * 50)
    print("Testing model with large vocabulary...")
    print("=" * 50 + "\n")

    # Test model with attention
    test_loss_attn, bleu_attn = test_model(
        model=model_attn,
        test_dataloader=test_loader,
        criterion=criterion,
        device=device,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        num_examples=10,
    )

    print(f"Test Loss: {test_loss_attn:.4f}")
    print(f"BLEU Score: {bleu_attn:.4f}")

    # Example translations
    print("\n" + "=" * 50)
    print("Example translations with large vocabulary model...")
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
    print(f"{'Source':<30} | {'Translation':<50}")
    print(f"{'-' * 30}+{'-' * 51}")

    for sentence in example_sentences:
        # Translate with the large vocabulary model
        translation, _ = translate_sentence(
            model=model_attn,
            sentence=sentence,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            device=device,
        )

        # Print results
        print(f"{sentence[:28]:<30} | {translation[:48]:<50}")

    print("\nModel training and testing complete!")
    print(f"Model saved at: {model_attn_large_vocab_path}")


if __name__ == "__main__":
    main()