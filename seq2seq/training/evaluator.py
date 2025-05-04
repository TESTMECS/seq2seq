"""
Evaluation functions for sequence-to-sequence models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple, Any
from tqdm import tqdm

from seq2seq.utils.metrics import bleu_score


def test_model(
    model: nn.Module,
    test_dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    src_tokenizer: Any,
    tgt_tokenizer: Any,
    num_examples: int = 20,
) -> Tuple[float, float]:
    """
    Test the model on test data and compute BLEU score.

    Args:
        model: The model to test
        test_dataloader: DataLoader for test data
        criterion: Loss function
        device: Device to use for testing
        src_tokenizer: Tokenizer for source language
        tgt_tokenizer: Tokenizer for target language
        num_examples: Number of examples to print

    Returns:
        test_loss: Average loss for the test data
        bleu: BLEU score for the test data
    """
    model.eval()
    test_loss = 0

    # Load all test data for BLEU calculation
    all_sources = []
    all_targets = []
    all_predictions = []

    examples = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing"):
            # Get data
            src = batch["src"].to(device)  # [batch_size, src_len]
            tgt = batch["tgt"].to(device)  # [batch_size, tgt_len]
            src_mask = batch["src_mask"].to(device)  # [batch_size, src_len]
            src_lengths = batch["src_lengths"].to(device)  # [batch_size]

            # Forward pass
            output_dict = model(
                src=src,
                tgt=tgt,
                src_lengths=src_lengths,
                src_mask=src_mask,
                teacher_forcing_ratio=0.0,  # No teacher forcing during testing
            )

            # Calculate loss
            output = output_dict["outputs"]
            loss = criterion(output, tgt)

            # Update test loss
            test_loss += loss.item()

            # Translate batch
            max_length = tgt.shape[1]
            bos_idx = tgt_tokenizer.token_to_id("[BOS]")
            eos_idx = tgt_tokenizer.token_to_id("[EOS]")

            translation_dict = model.translate(
                src=src,
                src_lengths=src_lengths,
                src_mask=src_mask,
                max_len=max_length,
                bos_idx=bos_idx,
                eos_idx=eos_idx,
            )

            translations = translation_dict["translations"]

            # Convert token ids to sentences
            for i in range(src.shape[0]):
                # Source text
                src_tokens = src[i].cpu().numpy()
                src_tokens = src_tokens[src_tokens != 0]
                src_text = src_tokenizer.decode(src_tokens.tolist())

                # Target text
                tgt_tokens = tgt[i].cpu().numpy()
                tgt_tokens = tgt_tokens[tgt_tokens != 0]
                tgt_text = tgt_tokenizer.decode(tgt_tokens.tolist())

                # Predicted text
                pred_tokens = translations[i].cpu().numpy()
                # Remove padding, BOS and EOS tokens
                pred_tokens = pred_tokens[pred_tokens != 0]
                pred_tokens = pred_tokens[1:]  # Remove BOS
                if eos_idx in pred_tokens:
                    pred_tokens = pred_tokens[: np.where(pred_tokens == eos_idx)[0][0]]
                pred_text = tgt_tokenizer.decode(pred_tokens.tolist())

                # Add to lists for BLEU calculation
                all_sources.append(src_text)
                all_targets.append(tgt_text)
                all_predictions.append(pred_text)

                # Store example for printing
                if len(examples) < num_examples:
                    examples.append((src_text, tgt_text, pred_text))

    # Calculate BLEU score
    bleu = bleu_score(all_targets, all_predictions)

    # Print examples
    print(f"\n{'=' * 50}")
    print(f"{'Source':<20} | {'Target':<20} | {'Prediction':<20}")
    print(f"{'-' * 20}+{'-' * 22}+{'-' * 22}")

    for src_text, tgt_text, pred_text in examples:
        print(f"{src_text[:18]:<20} | {tgt_text[:18]:<20} | {pred_text[:18]:<20}")

    print(f"{'=' * 50}\n")

    # Average loss
    test_loss = test_loss / len(test_dataloader)

    return test_loss, bleu
