"""
Tokenizer utilities for sequence-to-sequence models.
"""

from typing import List
import os


def create_tokenizer_json(texts: List[str], vocab_size: int, output_path: str) -> None:
    """
    Create a tokenizer and save it to a JSON file.

    Args:
        texts: List of texts to train the tokenizer on
        vocab_size: Size of the vocabulary
        output_path: Path to save the tokenizer
    """
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing

    # Initialize tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
    )

    # Set pre-tokenizer
    tokenizer.pre_tokenizer = Whitespace()

    # Train tokenizer
    tokenizer.train_from_iterator(texts, trainer)

    # Set post-processor
    tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[
            ("[BOS]", tokenizer.token_to_id("[BOS]")),
            ("[EOS]", tokenizer.token_to_id("[EOS]")),
        ],
    )

    # Save tokenizer
    tokenizer.save(output_path)