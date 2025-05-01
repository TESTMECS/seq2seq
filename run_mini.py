#!/usr/bin/env python
"""
Mini script to run a small subset of the data and test both models.
"""

import os
import torch
import torch.optim as optim
import torch.nn as nn
import random
import numpy as np
from tokenizers import Tokenizer

from data import load_iwslt_data, load_tokenizers, TranslationDataset, collate_fn, BatchSampler
from torch.utils.data import DataLoader
from model import Encoder, Decoder, Attention, Seq2Seq, Seq2SeqLoss
from train import train, plot_losses
from utils import translate_sentence


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
    hidden_dim: int = 64,
    emb_dim: int = 32,
    n_layers: int = 1,
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
    # Configuration
    src_lang = "en"
    tgt_lang = "fr"
    batch_size = 16
    max_length = 50
    hidden_dim = 64
    emb_dim = 32
    n_layers = 1
    dropout = 0.2
    n_epochs = 2
    clip = 1.0
    lr = 0.001
    patience = 2
    tf_ratio = 0.5
    seed = 42
    device = torch.device("cpu")  # Use CPU for consistency
    save_dir = "models"
    cache_dir = "a4-data/dataset"
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    
    # Set seed
    set_seed(seed)
    
    print(f"Using device: {device}")
    
    # Load tokenizers
    src_tokenizer, tgt_tokenizer = load_tokenizers(
        f"{src_lang}_tokenizer.json", f"{tgt_lang}_tokenizer.json"
    )
    
    # Load a small subset of data
    print("Loading dataset...")
    # Load a limited number of examples
    sample_size = 1000
    train_pairs = load_iwslt_data(src_lang, tgt_lang, "train", cache_dir)[:sample_size]
    val_pairs = load_iwslt_data(src_lang, tgt_lang, "validation", cache_dir)[:100]
    test_pairs = load_iwslt_data(src_lang, tgt_lang, "test", cache_dir)[:100]
    
    # Tokenize data
    print("Tokenizing dataset...")
    
    def tokenize_pairs(pairs):
        src_tokenized = []
        tgt_tokenized = []
        
        for src_text, tgt_text in pairs:
            if src_text is None or tgt_text is None or src_text == "" or tgt_text == "":
                continue
                
            try:
                src_tokens = src_tokenizer.encode(src_text).ids
                tgt_tokens = tgt_tokenizer.encode(tgt_text).ids
                
                if not src_tokens or not tgt_tokens:
                    continue
                    
                if len(src_tokens) > max_length:
                    src_tokens = src_tokens[:max_length]
                if len(tgt_tokens) > max_length:
                    tgt_tokens = tgt_tokens[:max_length]
                    
                src_tokenized.append(src_tokens)
                tgt_tokenized.append(
                    [tgt_tokenizer.token_to_id("[BOS]")]
                    + tgt_tokens
                    + [tgt_tokenizer.token_to_id("[EOS]")]
                )
            except Exception as e:
                print(f"Error tokenizing: {e}")
                
        return src_tokenized, tgt_tokenized
    
    train_src, train_tgt = tokenize_pairs(train_pairs)
    val_src, val_tgt = tokenize_pairs(val_pairs)
    test_src, test_tgt = tokenize_pairs(test_pairs)
    
    print(f"Training examples: {len(train_src)}")
    print(f"Validation examples: {len(val_src)}")
    print(f"Test examples: {len(test_src)}")
    
    # Create datasets
    train_dataset = TranslationDataset(train_src, train_tgt, src_lang, tgt_lang)
    val_dataset = TranslationDataset(val_src, val_tgt, src_lang, tgt_lang)
    test_dataset = TranslationDataset(test_src, test_tgt, src_lang, tgt_lang)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    # Get vocab sizes
    src_vocab_size = src_tokenizer.get_vocab_size()
    tgt_vocab_size = tgt_tokenizer.get_vocab_size()
    
    print(f"Source vocabulary size: {src_vocab_size}")
    print(f"Target vocabulary size: {tgt_vocab_size}")
    
    # Define model paths
    model_no_attn_path = os.path.join(save_dir, "model_no_attn_mini.pt")
    model_attn_path = os.path.join(save_dir, "model_attn_mini.pt")
    
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
        save_path=os.path.join(save_dir, "losses_no_attn_mini.png"),
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
        save_path=os.path.join(save_dir, "losses_attn_mini.png"),
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