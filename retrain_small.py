#!/usr/bin/env python
"""
Script to retrain on a small dataset with the large vocabulary tokenizers.
This script is for demonstration only - it trains on a small subset of data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
import numpy as np
from tokenizers import Tokenizer

from model import Encoder, Decoder, Attention, Seq2Seq, Seq2SeqLoss
from utils import translate_sentence


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    # Parameters
    hidden_dim = 256
    emb_dim = 128
    n_layers = 1
    batch_size = 8
    n_epochs = 3
    device = torch.device("cpu")
    save_dir = "models"
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Load tokenizers
    src_tokenizer = Tokenizer.from_file("en_tokenizer.json")
    tgt_tokenizer = Tokenizer.from_file("fr_tokenizer.json")
    
    # Get vocabulary sizes
    src_vocab_size = src_tokenizer.get_vocab_size()
    tgt_vocab_size = tgt_tokenizer.get_vocab_size()
    
    print(f"Source vocabulary size: {src_vocab_size}")
    print(f"Target vocabulary size: {tgt_vocab_size}")
    
    # Create a small dataset for demonstration
    example_sentences = [
        "Hello, how are you?",
        "I love programming in Python.",
        "The weather is nice today.",
        "What time is it?",
        "Can you help me with this translation?",
        "I'm learning machine translation.",
        "This is a sequence-to-sequence model.",
        "Neural networks are powerful for language tasks.",
        "Attention mechanisms improve translation quality.",
        "Thank you for your help!",
    ]
    
    # French translations (these are just examples, replace with correct translations if needed)
    example_translations = [
        "Bonjour, comment ça va?",
        "J'aime programmer en Python.",
        "Il fait beau aujourd'hui.",
        "Quelle heure est-il?",
        "Pouvez-vous m'aider avec cette traduction?",
        "J'apprends la traduction automatique.",
        "C'est un modèle séquence à séquence.",
        "Les réseaux de neurones sont puissants pour les tâches linguistiques.",
        "Les mécanismes d'attention améliorent la qualité de la traduction.",
        "Merci pour votre aide!",
    ]
    
    # Tokenize the datasets
    print("Tokenizing sentences...")
    src_tokenized = []
    tgt_tokenized = []
    
    for src_text, tgt_text in zip(example_sentences, example_translations):
        src_tokens = src_tokenizer.encode(src_text).ids
        tgt_tokens = tgt_tokenizer.encode(tgt_text).ids
        
        # Get token indices - use defaults if not found in tokenizer
        bos_token = tgt_tokenizer.token_to_id("<s>")
        if bos_token is None:
            bos_token = 0  # Use the actual BOS token ID from your tokenizer
            
        eos_token = tgt_tokenizer.token_to_id("</s>")
        if eos_token is None:
            eos_token = 1  # Use the actual EOS token ID from your tokenizer
            
        # Add special tokens to target
        tgt_tokens_with_special = [bos_token] + tgt_tokens + [eos_token]
        
        src_tokenized.append(torch.tensor(src_tokens))
        tgt_tokenized.append(torch.tensor(tgt_tokens_with_special))
    
    # Create padding function
    def pad_sequence(sequences, padding_value=0):
        max_len = max([seq.size(0) for seq in sequences])
        padded_seqs = []
        
        for seq in sequences:
            padded = torch.full((max_len,), padding_value, dtype=torch.long)
            padded[:seq.size(0)] = seq
            padded_seqs.append(padded)
            
        return torch.stack(padded_seqs)
    
    # Pad sequences
    src_padded = pad_sequence(src_tokenized)
    tgt_padded = pad_sequence(tgt_tokenized)
    
    # Create masks
    src_mask = (src_padded != 0).float()
    src_lengths = torch.tensor([len(s) for s in src_tokenized], dtype=torch.long)
    
    # Create encoder
    encoder = Encoder(
        input_dim=src_vocab_size,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=0.2,
    )
    
    # Create attention mechanism
    attention = Attention(hidden_dim=hidden_dim)
    
    # Create decoder
    decoder = Decoder(
        output_dim=tgt_vocab_size,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=0.2,
        attention=attention,
    )
    
    # Create model
    model = Seq2Seq(
        encoder=encoder,
        decoder=decoder,
        device=device,
        use_attention=True,
    )
    
    # Print model size
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    
    # Move model to device
    model = model.to(device)
    
    # Define pad index for loss function
    pad_idx = 3  # Common default for PAD token
    
    # Create loss function
    criterion = Seq2SeqLoss(pad_idx=pad_idx)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    print("\nTraining model on small dataset...")
    
    # Move data to device
    src_padded = src_padded.to(device)
    tgt_padded = tgt_padded.to(device)
    src_mask = src_mask.to(device)
    src_lengths = src_lengths.to(device)
    
    # Training loop
    for epoch in range(n_epochs):
        model.train()
        
        # Forward pass
        output_dict = model(
            src=src_padded,
            tgt=tgt_padded,
            src_lengths=src_lengths,
            src_mask=src_mask,
            teacher_forcing_ratio=0.5,
        )
        
        # Calculate loss
        output = output_dict["outputs"]
        loss = criterion(output, tgt_padded)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        print(f"Epoch {epoch+1}/{n_epochs} | Loss: {loss.item():.4f}")
    
    # Save model
    model_path = os.path.join(save_dir, "model_small_demo.pt")
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")
    
    # Test translations
    print("\nTesting model with example sentences:")
    
    model.eval()
    
    for sentence in example_sentences:
        with torch.no_grad():
            # Translate
            translation, _ = translate_sentence(
                model=model,
                sentence=sentence,
                src_tokenizer=src_tokenizer,
                tgt_tokenizer=tgt_tokenizer,
                device=device,
            )
            
            # Print results
            print(f"Source: {sentence}")
            print(f"Translation: {translation}")
            print()


if __name__ == "__main__":
    main()