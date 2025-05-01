"""
Simplified script for testing sequence-to-sequence models.
"""

import torch
import torch.optim as optim
import torch.nn as nn
import os
import random
import numpy as np
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from model import Encoder, Decoder, Attention, Seq2Seq, Seq2SeqLoss
from train import train, test_model, plot_losses
from utils import translate_sentence
# Import these directly to avoid potential circular imports
from data import TranslationDataset, collate_fn, BatchSampler


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


def main():
    """Main function for training and evaluating models."""
    # Configuration
    src_lang = "en"
    tgt_lang = "fr"
    batch_size = 5
    max_length = 50
    hidden_dim = 128
    emb_dim = 64
    n_layers = 1
    dropout = 0.2
    n_epochs = 2
    clip = 1.0
    lr = 0.001
    patience = 2
    tf_ratio = 0.5
    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = "models"

    # Set random seed for reproducibility
    set_seed(seed)

    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    print(f"Using device: {device}")

    # Create test data
    print("Creating test dataset...")

    # Load tokenizers (we'll still use the pretrained ones)
    src_tokenizer = Tokenizer.from_file(f"{src_lang}_tokenizer.json")
    tgt_tokenizer = Tokenizer.from_file(f"{tgt_lang}_tokenizer.json")

    # Get vocabulary sizes
    src_vocab_size = src_tokenizer.get_vocab_size()
    tgt_vocab_size = tgt_tokenizer.get_vocab_size()

    # Create sample data with 20 examples
    # These are just random indices within vocab range
    num_examples = 20

    # Generate random source and target sequences
    src_data = []
    tgt_data = []

    for i in range(num_examples):
        # Create random length sequences between 3 and 8 tokens
        src_len = random.randint(3, 8)
        tgt_len = random.randint(3, 8)

        # Create random token IDs (1 to 100 range to avoid special tokens)
        src_tokens = [random.randint(10, 100) for _ in range(src_len)]
        # Add special tokens for target (BOS and EOS)
        tgt_tokens = (
            [tgt_tokenizer.token_to_id("[BOS]")]
            + [random.randint(10, 100) for _ in range(tgt_len)]
            + [tgt_tokenizer.token_to_id("[EOS]")]
        )

        src_data.append(src_tokens)
        tgt_data.append(tgt_tokens)

    # Create dataset
    dataset = TranslationDataset(src_data, tgt_data, src_lang, tgt_lang)

    # Create batch samplers
    batch_sampler = BatchSampler([len(x) for x in src_data], batch_size, shuffle=True)

    # Create data loader
    data_loader = DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=collate_fn)

    # Use the same loader for train, val and test to simplify
    train_loader = data_loader
    val_loader = data_loader
    test_loader = data_loader

    print(f"Source vocabulary size: {src_vocab_size}")
    print(f"Target vocabulary size: {tgt_vocab_size}")
    print(f"Number of examples: {num_examples}")

    # Define model paths
    model_no_attn_path = os.path.join(save_dir, "model_no_attn.pt")
    model_attn_path = os.path.join(save_dir, "model_attn.pt")

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
        save_path=os.path.join(save_dir, "losses_no_attn.png"),
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
        save_path=os.path.join(save_dir, "losses_attn.png"),
    )

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
        num_examples=5,
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
        num_examples=5,
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
