#!/usr/bin/env python
"""
Test script for verifying the translation functionality.
"""

import torch
from tokenizers import Tokenizer
from model import Encoder, Decoder, Attention, Seq2Seq
from utils import translate_sentence
import os


def test_translation():
    """Test basic translation functionality."""
    # Parameters
    model_path = "models/model_attn_final.pt"
    src_tokenizer_path = "en_tokenizer.json"
    tgt_tokenizer_path = "fr_tokenizer.json"
    use_attention = True
    hidden_dim = 256
    emb_dim = 128
    n_layers = 2
    device = torch.device("cpu")  # Use CPU for testing
    test_sentence = "Hello world"
    
    # Check files exist
    print(f"Checking if model exists: {os.path.exists(model_path)}")
    print(f"Checking if source tokenizer exists: {os.path.exists(src_tokenizer_path)}")
    print(f"Checking if target tokenizer exists: {os.path.exists(tgt_tokenizer_path)}")
    
    # Check if the required files exist before proceeding
    if not all([
        os.path.exists(model_path),
        os.path.exists(src_tokenizer_path), 
        os.path.exists(tgt_tokenizer_path)
    ]):
        print("ERROR: Required files not found. Check paths.")
        return
    
    try:
        print("Loading tokenizers...")
        # Load tokenizers
        src_tokenizer = Tokenizer.from_file(src_tokenizer_path)
        tgt_tokenizer = Tokenizer.from_file(tgt_tokenizer_path)
        
        # Get vocabulary sizes
        src_vocab_size = src_tokenizer.get_vocab_size()
        tgt_vocab_size = tgt_tokenizer.get_vocab_size()
        
        print(f"Source vocabulary size: {src_vocab_size}")
        print(f"Target vocabulary size: {tgt_vocab_size}")
        
        print("Creating model components...")
        # Create encoder
        encoder = Encoder(
            input_dim=src_vocab_size,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=0.2,
        )
        
        # Create attention mechanism
        attention = None
        if use_attention:
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
            use_attention=use_attention,
        )
        
        print("Loading model weights...")
        # Load trained weights
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.to(device)
            model.eval()
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return
        
        print(f"Translating: '{test_sentence}'...")
        # Test translation
        translation, attention_weights = translate_sentence(
            model=model,
            sentence=test_sentence,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            device=device,
        )
        
        print(f"Test translation result: '{translation}'")
        print(f"Attention weights available: {attention_weights is not None}")
        
        # Additional verification
        if not translation:
            print("WARNING: Empty translation returned")
        
        print("Translation test completed successfully!")
        
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Starting translation test...")
    test_translation()