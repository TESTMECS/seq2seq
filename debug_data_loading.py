#!/usr/bin/env python
"""
Debug script to analyze data loading issues with the IWSLT dataset.
"""

import os
import torch
from data import load_iwslt_data, load_tokenizers
from tokenizers import Tokenizer
from tqdm import tqdm

def main():
    # Set up cache directory
    cache_dir = os.path.join('a4-data', 'dataset')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Load tokenizers
    src_lang = "en"
    tgt_lang = "fr"
    src_tokenizer, tgt_tokenizer = load_tokenizers(
        f"{src_lang}_tokenizer.json", f"{tgt_lang}_tokenizer.json"
    )
    
    # Load a small sample of the dataset
    sample_size = 100
    print(f"Loading {sample_size} examples from IWSLT dataset...")
    train_pairs = load_iwslt_data(src_lang, tgt_lang, "train", cache_dir)[:sample_size]
    
    # Analyze the pairs
    print("\nAnalyzing dataset pairs...")
    none_count = 0
    empty_count = 0
    too_long_count = 0
    success_count = 0
    
    for i, (src_text, tgt_text) in enumerate(train_pairs):
        print(f"\nPair {i+1}:")
        
        # Check for None values
        if src_text is None or tgt_text is None:
            print(f"  None values detected: src={src_text is None}, tgt={tgt_text is None}")
            none_count += 1
            continue
            
        # Check for empty strings
        if src_text == "" or tgt_text == "":
            print(f"  Empty strings detected: src='{src_text}', tgt='{tgt_text}'")
            empty_count += 1
            continue
            
        # Try tokenizing
        try:
            src_tokens = src_tokenizer.encode(src_text).ids
            tgt_tokens = tgt_tokenizer.encode(tgt_text).ids
            
            # Check token counts
            if len(src_tokens) > 100 or len(tgt_tokens) > 100:
                print(f"  Very long sequence: src={len(src_tokens)} tokens, tgt={len(tgt_tokens)} tokens")
                too_long_count += 1
            
            print(f"  Source text: '{src_text[:50]}...' ({len(src_tokens)} tokens)")
            print(f"  Target text: '{tgt_text[:50]}...' ({len(tgt_tokens)} tokens)")
            success_count += 1
            
        except Exception as e:
            print(f"  Error tokenizing: {e}")
    
    # Print summary
    print("\nSummary:")
    print(f"  Total pairs examined: {sample_size}")
    print(f"  Pairs with None values: {none_count}")
    print(f"  Pairs with empty strings: {empty_count}")
    print(f"  Pairs with very long sequences: {too_long_count}")
    print(f"  Successfully processed pairs: {success_count}")
    
    # Test creating a batch
    print("\nTesting batch creation...")
    
    # Create test tokenized data
    src_tokenized = []
    tgt_tokenized = []
    
    for src_text, tgt_text in tqdm(train_pairs, desc="Tokenizing"):
        if src_text is None or tgt_text is None or src_text == "" or tgt_text == "":
            continue
        
        try:
            src_tokens = src_tokenizer.encode(src_text).ids
            tgt_tokens = tgt_tokenizer.encode(tgt_text).ids
            
            # Add BOS and EOS tokens
            tgt_tokens_with_special = [tgt_tokenizer.token_to_id("[BOS]")] + tgt_tokens + [tgt_tokenizer.token_to_id("[EOS]")]
            
            src_tokenized.append(src_tokens)
            tgt_tokenized.append(tgt_tokens_with_special)
        except Exception as e:
            print(f"Error tokenizing: {e}")
    
    # Create batch
    print(f"Created {len(src_tokenized)} tokenized sequences")
    if len(src_tokenized) > 0:
        # Convert to tensors
        src_tensors = [torch.tensor(s, dtype=torch.long) for s in src_tokenized]
        tgt_tensors = [torch.tensor(t, dtype=torch.long) for t in tgt_tokenized]
        
        # Get shapes
        src_shapes = [s.shape for s in src_tensors]
        tgt_shapes = [t.shape for t in tgt_tensors]
        
        print("Source tensor shapes:")
        for i, shape in enumerate(src_shapes[:5]):
            print(f"  {i}: {shape}")
        
        print("Target tensor shapes:")
        for i, shape in enumerate(tgt_shapes[:5]):
            print(f"  {i}: {shape}")

if __name__ == "__main__":
    main()