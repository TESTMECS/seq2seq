"""
Translation utilities for sequence-to-sequence models.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional, Any


def translate_sentence(
    model: nn.Module,
    sentence: str,
    src_tokenizer: Any,
    tgt_tokenizer: Any,
    device: torch.device,
    max_len: int = 50,
) -> Tuple[str, Optional[np.ndarray]]:
    """
    Translate a sentence using the model.

    Args:
        model: The trained model
        sentence: Source language sentence
        src_tokenizer: Tokenizer for source language
        tgt_tokenizer: Tokenizer for target language
        device: Device to use for translation
        max_len: Maximum length of translation

    Returns:
        translation: Translated sentence
        attention: Attention weights (if available)
    """
    model.eval()

    # Tokenize the source sentence
    tokens = src_tokenizer.encode(sentence).ids

    # Convert to tensor
    src_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
    src_mask = torch.ones_like(src_tensor, dtype=torch.float).to(device)
    src_lengths = torch.tensor([len(tokens)], dtype=torch.long).to(device)

    with torch.no_grad():
        # Translate
        bos_idx = tgt_tokenizer.token_to_id("[BOS]")
        eos_idx = tgt_tokenizer.token_to_id("[EOS]")

        translation_dict = model.translate(
            src=src_tensor,
            src_lengths=src_lengths,
            src_mask=src_mask,
            max_len=max_len,
            bos_idx=bos_idx,
            eos_idx=eos_idx,
        )

        # Extract translation and attention
        pred_tokens = translation_dict["translations"][0].cpu().numpy()
        attention = None
        if translation_dict["attention_weights"] is not None:
            attention = translation_dict["attention_weights"][0].cpu().numpy()

        # Remove padding, BOS and EOS tokens
        pred_tokens = pred_tokens[pred_tokens != 0]
        pred_tokens = pred_tokens[1:]  # Remove BOS
        if eos_idx in pred_tokens:
            pred_tokens = pred_tokens[: np.where(pred_tokens == eos_idx)[0][0]]

        # Decode the translation
        translation = tgt_tokenizer.decode(pred_tokens.tolist())

    return translation, attention


def compare_translations(
    source_sentences: List[str],
    reference_translations: List[str],
    model_translations: Dict[str, List[str]],
) -> None:
    """
    Compare translations from different models.

    Args:
        source_sentences: Source language sentences
        reference_translations: Reference translations
        model_translations: Dictionary mapping model names to lists of translations
    """
    print(f"\n{'=' * 80}")
    print(f"{'Source':<20} | {'Reference':<20}", end="")

    for model_name in model_translations.keys():
        print(f" | {model_name:<20}", end="")

    print(f"\n{'-' * 20}+{'-' * 21}", end="")

    for _ in model_translations.keys():
        print(f"+{'-' * 21}", end="")

    print()

    for i, (src, ref) in enumerate(zip(source_sentences, reference_translations)):
        print(f"{src[:18]:<20} | {ref[:18]:<20}", end="")

        for model_name in model_translations.keys():
            print(f" | {model_translations[model_name][i][:18]:<20}", end="")

        print()

    print(f"{'=' * 80}\n")