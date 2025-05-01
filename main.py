"""
Main script for training and evaluating sequence-to-sequence models.
"""

import argparse
import torch
import torch.optim as optim
import torch.nn as nn
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
    use_attention: bool = False,
    hidden_dim: int = 256,
    emb_dim: int = 128,
    n_layers: int = 2,
    dropout: float = 0.2,
) -> Seq2Seq:
    """
    Create a sequence-to-sequence model.

    Args:
        src_vocab_size: Size of source vocabulary
        tgt_vocab_size: Size of target vocabulary
        device: Device to use for model
        use_attention: Whether to use attention mechanism
        hidden_dim: Hidden dimension size
        emb_dim: Embedding dimension size
        n_layers: Number of layers in encoder and decoder
        dropout: Dropout probability

    Returns:
        model: Sequence-to-sequence model
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


def main():
    """Main function for training and evaluating models."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train sequence-to-sequence models for machine translation"
    )

    # Data parameters
    parser.add_argument("--src_lang", type=str, default="en", help="Source language")
    parser.add_argument("--tgt_lang", type=str, default="fr", help="Target language")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
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
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
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
        "--test_only", action="store_true", help="Only test the model (no training)"
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
    try:
        data_dict = prepare_data(
            src_lang=args.src_lang,
            tgt_lang=args.tgt_lang,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
    except Exception as e:
        print(f"Error preparing data: {e}")
        # Set default values for testing
        print("Using small test dataset instead")
        # Create test data
        src_tokenizer = Tokenizer.from_file(f"{args.src_lang}_tokenizer.json")
        tgt_tokenizer = Tokenizer.from_file(f"{args.tgt_lang}_tokenizer.json")

        # Create a small test dataset
        from data import TranslationDataset, collate_fn, BatchSampler
        from torch.utils.data import DataLoader

        # Sample data (5 simple sentences)
        src_data = [[1, 2, 3], [4, 5, 6, 7], [8, 9], [10, 11, 12, 13, 14], [15, 16, 17]]
        tgt_data = [[1, 2, 3, 4], [5, 6], [7, 8, 9, 10], [11, 12], [13, 14, 15, 16, 17]]

        # Create datasets
        test_dataset = TranslationDataset(src_data, tgt_data, args.src_lang, args.tgt_lang)

        # Create batch samplers
        test_sampler = BatchSampler([len(x) for x in src_data], args.batch_size, shuffle=False)

        # Create data loaders
        test_loader = DataLoader(test_dataset, batch_sampler=test_sampler, collate_fn=collate_fn)

        # Return dictionary
        data_dict = {
            "train_loader": test_loader,
            "val_loader": test_loader,
            "test_loader": test_loader,
            "src_tokenizer": src_tokenizer,
            "tgt_tokenizer": tgt_tokenizer,
            "src_vocab_size": src_tokenizer.get_vocab_size(),
            "tgt_vocab_size": tgt_tokenizer.get_vocab_size(),
        }

    train_loader = data_dict["train_loader"]
    val_loader = data_dict["val_loader"]
    test_loader = data_dict["test_loader"]
    src_tokenizer = data_dict["src_tokenizer"]
    tgt_tokenizer = data_dict["tgt_tokenizer"]
    src_vocab_size = data_dict["src_vocab_size"]
    tgt_vocab_size = data_dict["tgt_vocab_size"]

    print(f"Source vocabulary size: {src_vocab_size}")
    print(f"Target vocabulary size: {tgt_vocab_size}")

    # Define model paths
    model_no_attn_path = os.path.join(args.save_dir, "model_no_attn.pt")
    model_attn_path = os.path.join(args.save_dir, "model_attn.pt")

    # Define pad index for loss function
    pad_idx = tgt_tokenizer.token_to_id("[PAD]")

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
        hidden_dim=args.hidden_dim,
        emb_dim=args.emb_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
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
        hidden_dim=args.hidden_dim,
        emb_dim=args.emb_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
    )
    print(
        f"Model with attention has {sum(p.numel() for p in model_attn.parameters() if p.requires_grad):,} trainable parameters"
    )

    if not args.test_only:
        # Train models
        print("\n" + "=" * 50)
        print("Training model without attention...")
        print("=" * 50 + "\n")

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
            save_path=os.path.join(args.save_dir, "losses_no_attn.png"),
        )

        print("\n" + "=" * 50)
        print("Training model with attention...")
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
            model_path=model_attn_path,
        )

        # Plot losses for model with attention
        plot_losses(
            history=history_attn,
            save_path=os.path.join(args.save_dir, "losses_attn.png"),
        )
    else:
        # Load model weights
        print("Loading model weights...")
        model_no_attn.load_state_dict(torch.load(model_no_attn_path, map_location=device))
        model_attn.load_state_dict(torch.load(model_attn_path, map_location=device))

    # Test models
    print("\n" + "=" * 50)
    print("Testing model without attention...")
    print("=" * 50 + "\n")

    # Test model without attention
    test_loss_no_attn, bleu_no_attn = test_model(
        model=model_no_attn,
        test_dataloader=test_loader,
        criterion=criterion,
        device=device,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        num_examples=10,
    )

    print(f"Test Loss: {test_loss_no_attn:.4f}")
    print(f"BLEU Score: {bleu_no_attn:.4f}")

    print("\n" + "=" * 50)
    print("Testing model with attention...")
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

    # Compare results
    print("\n" + "=" * 50)
    print("Comparing results...")
    print("=" * 50 + "\n")

    print(f"{'Model':<20} | {'Test Loss':<15} | {'BLEU Score':<15}")
    print(f"{'-' * 20}+{'-' * 16}+{'-' * 16}")
    print(f"{'Without Attention':<20} | {test_loss_no_attn:<15.4f} | {bleu_no_attn:<15.4f}")
    print(f"{'With Attention':<20} | {test_loss_attn:<15.4f} | {bleu_attn:<15.4f}")

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
        translation_no_attn, _ = translate_sentence(
            model=model_no_attn,
            sentence=sentence,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            device=device,
        )

        # Translate with model with attention
        translation_attn, _ = translate_sentence(
            model=model_attn,
            sentence=sentence,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            device=device,
        )

        # Print results
        print(f"{sentence[:28]:<30} | {translation_no_attn[:28]:<30} | {translation_attn[:28]:<30}")


if __name__ == "__main__":
    main()
