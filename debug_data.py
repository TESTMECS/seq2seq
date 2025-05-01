from data import load_iwslt_data
import torch
from tokenizers import Tokenizer

# Load tokenizers
src_tokenizer = Tokenizer.from_file("en_tokenizer.json")
tgt_tokenizer = Tokenizer.from_file("fr_tokenizer.json")

# Load a small sample
pairs = load_iwslt_data("en", "fr", "train")[:5]
print("Sample pairs:")
for src, tgt in pairs:
    print(f"Source: {src}")
    print(f"Target: {tgt}")
    print("---")

# Try tokenizing
src_tokenized = [src_tokenizer.encode(src).ids for src, _ in pairs]
tgt_tokenized = [tgt_tokenizer.encode(tgt).ids for _, tgt in pairs]

# Check if any are None
print("Any None in source?", None in src_tokenized)
print("Any None in target?", None in tgt_tokenized)

# Check the shapes
print("Source shapes:", [len(s) for s in src_tokenized])
print("Target shapes:", [len(t) for t in tgt_tokenized])

# Test the collate_fn
from data import collate_fn

# Prepare a batch with the tokenized data
batch = []
for i in range(len(pairs)):
    src_tokens = src_tokenizer.encode(pairs[i][0]).ids
    tgt_tokens = tgt_tokenizer.encode(pairs[i][1]).ids
    # Add BOS and EOS to target
    tgt_tokens = (
        [tgt_tokenizer.token_to_id("[BOS]")] + tgt_tokens + [tgt_tokenizer.token_to_id("[EOS]")]
    )
    batch.append((src_tokens, tgt_tokens))

try:
    # Try to collate the batch
    collated = collate_fn(batch)
    print("Collate success!")
    print("Batch shapes:")
    for k, v in collated.items():
        print(f"{k}: {v.shape}")
except Exception as e:
    print(f"Collate error: {e}")
