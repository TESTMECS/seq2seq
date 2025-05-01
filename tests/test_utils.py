"""
Tests for utility functions in the sequence-to-sequence model.
"""

import torch
import pytest
import sys
import os
import numpy as np
from unittest.mock import patch, MagicMock

# Add parent directory to path to import from modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import bleu_score, translate_sentence, compare_translations


class TestBleuScore:
    """Tests for the BLEU score calculation function."""

    def test_perfect_match(self):
        """Test BLEU score calculation with perfect match."""
        references = ["this is a test", "another test sentence"]
        hypotheses = ["this is a test", "another test sentence"]

        # Perfect match should give BLEU score of 1.0
        score = bleu_score(references, hypotheses)
        assert score == 1.0

    def test_no_match(self):
        """Test BLEU score calculation with no match."""
        references = ["this is a test", "another test sentence"]
        hypotheses = ["completely different text", "nothing in common"]

        # No match should give a low BLEU score
        score = bleu_score(references, hypotheses)
        assert score < 0.1  # Very low score, but not exactly 0 due to smoothing


class TestTranslateSentence:
    """Tests for the translate_sentence function."""

    def test_translate_sentence(self):
        """Test the translate_sentence function."""
        # Mock model and tokenizers
        model = MagicMock()
        src_tokenizer = MagicMock()
        tgt_tokenizer = MagicMock()
        device = torch.device("cpu")

        # Configure mock tokenizers
        src_tokenizer.encode.return_value.ids = [1, 2, 3]
        tgt_tokenizer.token_to_id.side_effect = lambda token: 1 if token == "[BOS]" else 2
        tgt_tokenizer.decode.return_value = "mock translation"

        # Configure mock model
        model.translate.return_value = {
            "translations": torch.tensor([[1, 3, 4, 2, 0]]),
            "attention_weights": torch.randn(1, 5, 3),
        }

        # Call function
        translation, attention = translate_sentence(
            model=model,
            sentence="test sentence",
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            device=device,
        )

        # Check results
        assert translation == "mock translation"
        assert attention is not None
        assert isinstance(attention, np.ndarray)

        # Verify correct tokenizer and model calls
        src_tokenizer.encode.assert_called_once_with("test sentence")
        model.translate.assert_called_once()


class TestCompareTranslations:
    """Tests for the compare_translations function."""

    def test_compare_translations(self, capsys):
        """Test the compare_translations function using stdout capture."""
        # Test data
        sources = ["Source 1", "Source 2"]
        references = ["Reference 1", "Reference 2"]
        model_translations = {
            "Model 1": ["Translation 1A", "Translation 1B"],
            "Model 2": ["Translation 2A", "Translation 2B"],
        }

        # Call function (outputs to stdout)
        compare_translations(sources, references, model_translations)

        # Capture the output
        captured = capsys.readouterr()

        # Check that each item appears in the output
        for item in (
            sources + references + model_translations["Model 1"] + model_translations["Model 2"]
        ):
            assert item in captured.out
