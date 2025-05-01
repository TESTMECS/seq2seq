#!/usr/bin/env python
"""
Script to test tokenizer compatibility with model vocabulary.
"""

from tokenizers import Tokenizer
import torch

def main():
    """Test tokenizers and model compatibility."""
    # Load tokenizers
    print("Loading tokenizers...")
    src_tokenizer_path = "en_tokenizer.json"
    tgt_tokenizer_path = "fr_tokenizer.json"
    
    src_tokenizer = Tokenizer.from_file(src_tokenizer_path)
    tgt_tokenizer = Tokenizer.from_file(tgt_tokenizer_path)
    
    # Get vocabulary size
    src_vocab_size = src_tokenizer.get_vocab_size()
    tgt_vocab_size = tgt_tokenizer.get_vocab_size()
    
    print(f"Source vocabulary size: {src_vocab_size}")
    print(f"Target vocabulary size: {tgt_vocab_size}")
    
    # Print vocabulary details
    src_vocab = src_tokenizer.get_vocab()
    tgt_vocab = tgt_tokenizer.get_vocab()
    
    print("\nSource vocabulary sample (first 20 tokens):")
    for i, (token, idx) in enumerate(sorted(src_vocab.items(), key=lambda x: x[1])[:20]):
        print(f"  {idx}: '{token}'")
    
    print("\nTarget vocabulary sample (first 20 tokens):")
    for i, (token, idx) in enumerate(sorted(tgt_vocab.items(), key=lambda x: x[1])[:20]):
        print(f"  {idx}: '{token}'")
    
    # Test tokenization of example sentences
    test_sentences = [
        "Hello world",
        "How are you?",
        "I love programming",
        "The weather is nice today"
    ]
    
    print("\nTokenization examples:")
    for sentence in test_sentences:
        src_encoding = src_tokenizer.encode(sentence)
        
        print(f"\nInput: '{sentence}'")
        print(f"Tokens: {src_encoding.tokens}")
        print(f"IDs: {src_encoding.ids}")
        
        # Check if any token ID exceeds model vocabulary size (100)
        if max(src_encoding.ids) >= 100:
            print(f"WARNING: Token IDs exceed model vocabulary size of 100")
            over_limit = [
                (token, idx) for token, idx in zip(src_encoding.tokens, src_encoding.ids) 
                if idx >= 100
            ]
            print(f"Tokens outside model vocab: {over_limit}")
    
    # Load the model to check its parameters
    print("\nChecking model parameters:")
    model_path = "models/model_attn_final.pt"
    
    try:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Get source embedding information
        if 'encoder.embedding.weight' in state_dict:
            src_emb_shape = state_dict['encoder.embedding.weight'].shape
            print(f"Source embedding shape: {src_emb_shape}")
            model_src_vocab = src_emb_shape[0]
            
            if model_src_vocab != src_vocab_size:
                print(f"MISMATCH: Model source vocabulary ({model_src_vocab}) ≠ Tokenizer vocabulary ({src_vocab_size})")
        
        # Get target embedding information
        if 'decoder.embedding.weight' in state_dict:
            tgt_emb_shape = state_dict['decoder.embedding.weight'].shape
            print(f"Target embedding shape: {tgt_emb_shape}")
            model_tgt_vocab = tgt_emb_shape[0]
            
            if model_tgt_vocab != tgt_vocab_size:
                print(f"MISMATCH: Model target vocabulary ({model_tgt_vocab}) ≠ Tokenizer vocabulary ({tgt_vocab_size})")
    
        print("\nCONCLUSION:")
        print("The tokenizers and model have different vocabulary sizes.")
        print("To use this model, you would need:")
        print("1. Either use the original tokenizers from when the model was trained")
        print("2. Or retrain the model with the current tokenizers")
        print("3. Or create new tokenizers with vocabulary sizes matching the model")
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")

if __name__ == "__main__":
    main()