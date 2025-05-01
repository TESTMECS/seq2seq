#!/usr/bin/env python
"""
Script to translate sentences using the seq2seq model with large vocabulary.
"""

import torch
import argparse
import os
from tokenizers import Tokenizer
from model import Encoder, Decoder, Attention, Seq2Seq
from utils import translate_sentence


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Translate sentences with large vocabulary model")
    
    # Model parameters
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="models/model_attn_large_vocab.pt",
        help="Path to trained model weights"
    )
    parser.add_argument(
        "--src_lang", 
        type=str, 
        default="en", 
        help="Source language code (e.g., 'en' for English)"
    )
    parser.add_argument(
        "--tgt_lang", 
        type=str, 
        default="fr", 
        help="Target language code (e.g., 'fr' for French)"
    )
    parser.add_argument(
        "--hidden_dim", 
        type=int, 
        default=256, 
        help="Hidden dimension size"
    )
    parser.add_argument(
        "--emb_dim", 
        type=int, 
        default=128, 
        help="Embedding dimension size"
    )
    parser.add_argument(
        "--n_layers", 
        type=int, 
        default=2, 
        help="Number of layers in encoder and decoder"
    )
    parser.add_argument(
        "--sentence", 
        type=str, 
        help="Sentence to translate (if provided, runs in non-interactive mode)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)"
    )
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device)
    
    # Set tokenizer paths based on language flags
    src_tokenizer_path = f"{args.src_lang}_tokenizer.json"
    tgt_tokenizer_path = f"{args.tgt_lang}_tokenizer.json"
    
    # Check if tokenizer files exist
    if not os.path.exists(src_tokenizer_path):
        raise FileNotFoundError(f"Source tokenizer not found: {src_tokenizer_path}")
    if not os.path.exists(tgt_tokenizer_path):
        raise FileNotFoundError(f"Target tokenizer not found: {tgt_tokenizer_path}")
    
    # Load tokenizers
    src_tokenizer = Tokenizer.from_file(src_tokenizer_path)
    tgt_tokenizer = Tokenizer.from_file(tgt_tokenizer_path)
    
    # Get vocabulary sizes
    src_vocab_size = src_tokenizer.get_vocab_size()
    tgt_vocab_size = tgt_tokenizer.get_vocab_size()
    
    print(f"Source vocabulary size: {src_vocab_size}")
    print(f"Target vocabulary size: {tgt_vocab_size}")

    # Create encoder
    encoder = Encoder(
        input_dim=src_vocab_size,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        dropout=0.2,
    )

    # Create attention mechanism
    attention = Attention(hidden_dim=args.hidden_dim)

    # Create decoder
    decoder = Decoder(
        output_dim=tgt_vocab_size,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
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

    print(f"Loading model from {args.model_path}...")
    # Load trained weights
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        model = model.to(device)
        model.eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    print(f"Model loaded on {device}")
    print(f"Translating from {args.src_lang.upper()} to {args.tgt_lang.upper()}")
    
    # If sentence is provided, translate just that sentence and exit
    if args.sentence:
        try:
            print(f"\nTranslating: {args.sentence}")
            translation, _ = translate_sentence(
                model=model,
                sentence=args.sentence,
                src_tokenizer=src_tokenizer,
                tgt_tokenizer=tgt_tokenizer,
                device=device,
            )
            print(f"\nSource ({args.src_lang}): {args.sentence}")
            print(f"Translation ({args.tgt_lang}): {translation}")
        except Exception as e:
            print(f"Translation error: {str(e)}")
            import traceback
            traceback.print_exc()
        return
    
    # Interactive translation mode
    print("\nInteractive Translation Mode")
    print("Enter 'quit' to exit")
    
    while True:
        # Get input sentence
        sentence = input(f"\nEnter {args.src_lang.upper()} sentence: ")
        
        if sentence.lower() == 'quit':
            break
        
        try:    
            # Translate
            translation, _ = translate_sentence(
                model=model,
                sentence=sentence,
                src_tokenizer=src_tokenizer,
                tgt_tokenizer=tgt_tokenizer,
                device=device,
            )
            
            # Print translation
            print(f"{args.tgt_lang.upper()} translation: {translation}")
        except Exception as e:
            print(f"Translation error: {str(e)}")


if __name__ == "__main__":
    main()