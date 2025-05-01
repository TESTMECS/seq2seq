"""
Script to download IWSLT data and create tokenizers for the source and target languages.
"""

import argparse
import os
from datasets import load_dataset
from utils import create_tokenizer_json


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Create tokenizers for machine translation")
    parser.add_argument("--src_lang", type=str, default="en", help="Source language")
    parser.add_argument("--tgt_lang", type=str, default="fr", help="Target language")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size")
    args = parser.parse_args()

    print(f"Creating tokenizers for {args.src_lang}-{args.tgt_lang} translation...")

    # Download IWSLT dataset
    print("Downloading IWSLT dataset...")
    train_dataset = load_dataset("iwslt2017", f"{args.src_lang}-{args.tgt_lang}", split="train")

    # Extract texts for both languages
    src_texts = [item["translation"][args.src_lang] for item in train_dataset]
    tgt_texts = [item["translation"][args.tgt_lang] for item in train_dataset]

    print(f"Loaded {len(src_texts)} sentence pairs.")

    # Create tokenizers
    print(f"Creating source tokenizer ({args.src_lang})...")
    create_tokenizer_json(
        texts=src_texts,
        vocab_size=args.vocab_size,
        output_path=f"{args.src_lang}_tokenizer.json",
    )

    print(f"Creating target tokenizer ({args.tgt_lang})...")
    create_tokenizer_json(
        texts=tgt_texts,
        vocab_size=args.vocab_size,
        output_path=f"{args.tgt_lang}_tokenizer.json",
    )

    print("Tokenizers created successfully!")


if __name__ == "__main__":
    main()
