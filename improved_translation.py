#!/usr/bin/env python
"""
Improved translation script with better handling of special tokens.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import random
import numpy as np
from tokenizers import Tokenizer

from model import Encoder, Decoder, Attention, Seq2Seq


def improved_translate_sentence(
    model: nn.Module,
    sentence: str,
    src_tokenizer: Tokenizer,
    tgt_tokenizer: Tokenizer,
    device: torch.device,
    max_len: int = 50,
) -> str:
    """
    Translate a sentence using the model with improved token handling.
    """
    model.eval()
    
    # Tokenize the source sentence
    encoding = src_tokenizer.encode(sentence)
    tokens = encoding.ids
    
    # Convert to tensor
    src_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
    src_mask = torch.ones_like(src_tensor, dtype=torch.bool).to(device)
    src_lengths = torch.tensor([len(tokens)], dtype=torch.long).to(device)
    
    # Find special token IDs
    bos_alternatives = ["[BOS]", "<s>", "<bos>", "BOS", "<BOS>"]
    eos_alternatives = ["[EOS]", "</s>", "<eos>", "EOS", "<EOS>"]
    
    # Find BOS token
    bos_idx = None
    for token in bos_alternatives:
        bos_idx = tgt_tokenizer.token_to_id(token)
        if bos_idx is not None:
            break
    
    # If BOS token not found, use first special token
    if bos_idx is None:
        # Try to get the special tokens from the tokenizer vocabulary
        vocab = tgt_tokenizer.get_vocab()
        special_tokens = [
            (token, idx) for token, idx in vocab.items() 
            if token.startswith("<") or token.startswith("[")
        ]
        
        if special_tokens:
            special_tokens.sort(key=lambda x: x[1])  # Sort by index
            bos_idx = special_tokens[0][1]  # Use first special token
        else:
            bos_idx = 0  # Default to 0
    
    # Find EOS token
    eos_idx = None
    for token in eos_alternatives:
        eos_idx = tgt_tokenizer.token_to_id(token)
        if eos_idx is not None:
            break
    
    # If EOS token not found, use second special token or BOS + 1
    if eos_idx is None:
        vocab = tgt_tokenizer.get_vocab()
        special_tokens = [
            (token, idx) for token, idx in vocab.items() 
            if token.startswith("<") or token.startswith("[")
        ]
        
        if len(special_tokens) > 1:
            special_tokens.sort(key=lambda x: x[1])  # Sort by index
            eos_idx = special_tokens[1][1]  # Use second special token
        else:
            eos_idx = 1  # Default to 1 or bos_idx + 1
    
    print(f"Using BOS token ID: {bos_idx}")
    print(f"Using EOS token ID: {eos_idx}")
    
    with torch.no_grad():
        # Encode the source sentence
        encoder_outputs, hidden = model.encoder(src_tensor, src_lengths)
        
        # Initialize tensors for decoding
        batch_size = src_tensor.size(0)
        translations = torch.ones(batch_size, max_len, dtype=torch.long).to(device) * eos_idx
        translations[:, 0] = bos_idx
        
        # Decode one step at a time
        decoder_input = torch.ones(batch_size, dtype=torch.long).fill_(bos_idx).to(device)
        completed = torch.zeros(batch_size, dtype=torch.bool).to(device)
        attention_weights = None
        
        if model.use_attention:
            attention_weights = torch.zeros(batch_size, max_len, src_tensor.size(1)).to(device)
        
        for t in range(1, max_len):
            # Run the decoder for one time step
            output, hidden, attn_weights = model.decoder(
                decoder_input, hidden, encoder_outputs, src_mask
            )
            
            # Get the token with the highest predicted probability
            predicted = output.argmax(1)
            
            # Store the predicted token and attention weights
            translations[:, t] = predicted
            if model.use_attention and attn_weights is not None:
                attention_weights[:, t, :] = attn_weights
            
            # Check for completed sequences
            completed = completed | (predicted == eos_idx)
            if completed.all():
                break
            
            # Set the next decoder input
            decoder_input = predicted
        
        # Extract translation for the first (and only) example in batch
        pred_tokens = translations[0].cpu().numpy()
        
        # Remove padding, BOS and EOS tokens
        pred_tokens = pred_tokens[pred_tokens != 0]
        pred_tokens = pred_tokens[1:]  # Remove BOS
        if eos_idx in pred_tokens:
            # Only remove the first EOS token
            eos_pos = np.where(pred_tokens == eos_idx)[0][0]
            pred_tokens = pred_tokens[:eos_pos]
        
        # Decode the translation
        try:
            # Try standard decoding first
            translation = tgt_tokenizer.decode(pred_tokens.tolist())
        except Exception as e:
            print(f"Error decoding tokens: {e}")
            # Fallback: manually decode by looking up each token in the vocabulary
            vocab = tgt_tokenizer.get_vocab()
            reverse_vocab = {idx: token for token, idx in vocab.items()}
            
            # Map each token ID to its string representation
            tokens_str = []
            for token_id in pred_tokens:
                token = reverse_vocab.get(token_id)
                if token and not (token.startswith("<") or token.startswith("[") or token.startswith("</") or token.startswith("]")):
                    tokens_str.append(token)
                    
            # Join tokens
            translation = "".join(tokens_str).replace("Ġ", " ").strip()
    
    return translation


def main():
    # Parameters
    model_path = "models/model_small_demo.pt"
    src_tokenizer_path = "en_tokenizer.json"
    tgt_tokenizer_path = "fr_tokenizer.json"
    device = torch.device("cpu")
    
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
        emb_dim=128,
        hidden_dim=256,
        n_layers=1,
        dropout=0.2,
    )
    
    # Create attention mechanism
    attention = Attention(hidden_dim=256)
    
    # Create decoder
    decoder = Decoder(
        output_dim=tgt_vocab_size,
        emb_dim=128,
        hidden_dim=256,
        n_layers=1,
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
    
    # Load model
    print(f"Loading model from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print("Model loaded successfully")
    
    # Example sentences
    example_sentences = [
        "Hello, how are you?",
        "I love programming in Python.",
        "The weather is nice today.",
        "What time is it?",
        "Can you help me with this translation?",
    ]
    
    # Test translations
    print("\nTesting model with example sentences:")
    
    for sentence in example_sentences:
        # Translate
        translation = improved_translate_sentence(
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
    
    # Interactive mode
    print("\nInteractive Translation Mode")
    print("Enter 'quit' to exit")
    
    while True:
        # Get input sentence
        sentence = input("\nEnter English sentence: ")
        
        if sentence.lower() == 'quit':
            break
        
        try:    
            # Translate
            translation = improved_translate_sentence(
                model=model,
                sentence=sentence,
                src_tokenizer=src_tokenizer,
                tgt_tokenizer=tgt_tokenizer,
                device=device,
            )
            
            # Print translation
            print(f"French translation: {translation}")
        except Exception as e:
            print(f"Translation error: {str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()