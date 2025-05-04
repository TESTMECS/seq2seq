"""
Command-line interface for preparing tokenizers.
"""

import argparse
import os
from datasets import load_dataset
from seq2seq.utils import create_tokenizer_json


def main():
    """Main entry point for preparing tokenizers."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Prepare tokenizers for sequence-to-sequence models"
    )
    parser.add_argument("--src_lang", type=str, default="en", help="Source language")
    parser.add_argument("--tgt_lang", type=str, default="fr", help="Target language")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size")
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory")
    args = parser.parse_args()

    # Load dataset
    print(f"Loading IWSLT2017 {args.src_lang}-{args.tgt_lang} dataset...")
    dataset = load_dataset("iwslt2017", f"iwslt2017-{args.src_lang}-{args.tgt_lang}", split="train")

    # Extract texts
    src_texts = [item["translation"][args.src_lang] for item in dataset]
    tgt_texts = [item["translation"][args.tgt_lang] for item in dataset]

    print(f"Creating tokenizer for {args.src_lang} language...")
    src_tokenizer_path = os.path.join(args.output_dir, f"{args.src_lang}_tokenizer.json")
    create_tokenizer_json(
        texts=src_texts,
        vocab_size=args.vocab_size,
        output_path=src_tokenizer_path,
    )

    print(f"Creating tokenizer for {args.tgt_lang} language...")
    tgt_tokenizer_path = os.path.join(args.output_dir, f"{args.tgt_lang}_tokenizer.json")
    create_tokenizer_json(
        texts=tgt_texts,
        vocab_size=args.vocab_size,
        output_path=tgt_tokenizer_path,
    )

    print("Tokenizers created successfully.")
    print(f"Source tokenizer saved to: {src_tokenizer_path}")
    print(f"Target tokenizer saved to: {tgt_tokenizer_path}")


if __name__ == "__main__":
    main()
