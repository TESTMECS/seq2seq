"""
Tests for data processing functions.
"""

import torch
import pytest
import sys
import os
import numpy as np

# Add parent directory to path to import from modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import TranslationDataset, BatchSampler, collate_fn


class TestTranslationDataset:
    """Tests for the TranslationDataset class."""

    def test_dataset_creation(self):
        """Test creation of the dataset."""
        # Create sample data
        sources = [[1, 2, 3], [4, 5, 6, 7], [8, 9]]
        targets = [[10, 11, 12], [13, 14, 15, 16], [17, 18]]

        # Create dataset
        dataset = TranslationDataset(sources, targets)

        # Check length
        assert len(dataset) == 3

        # Check item access
        src, tgt = dataset[0]
        assert src == [1, 2, 3]
        assert tgt == [10, 11, 12]

        src, tgt = dataset[1]
        assert src == [4, 5, 6, 7]
        assert tgt == [13, 14, 15, 16]

        src, tgt = dataset[2]
        assert src == [8, 9]
        assert tgt == [17, 18]

    def test_dataset_with_unequal_lengths(self):
        """Test dataset creation with unequal source and target lengths."""
        # Create sample data
        sources = [[1, 2, 3], [4, 5, 6, 7]]
        targets = [[10, 11, 12], [13, 14, 15, 16], [17, 18]]

        # Check that assertion is raised
        with pytest.raises(AssertionError):
            TranslationDataset(sources, targets)


class TestBatchSampler:
    """Tests for the BatchSampler class."""

    def test_batch_sampler_deterministic(self):
        """Test batch sampler with deterministic behavior (no shuffle)."""
        # Create sample lengths
        lengths = [5, 2, 3, 6, 2, 3, 6]

        # Create batch sampler
        batch_size = 2
        sampler = BatchSampler(lengths, batch_size, shuffle=False)

        # Check number of batches
        assert len(sampler) == 4  # ceil(7/2) = 4

        # Check batches
        batches = list(sampler)

        # Lengths should be sorted
        expected_indices = [1, 4, 2, 5, 0, 3, 6]  # Sorted by length
        expected_batches = [
            expected_indices[0:2],  # [1, 4] (lengths 2, 2)
            expected_indices[2:4],  # [2, 5] (lengths 3, 3)
            expected_indices[4:6],  # [0, 3] (lengths 5, 6)
            expected_indices[6:],  # [6]    (length 6)
        ]

        assert len(batches) == len(expected_batches)

        # Check batch contents
        for batch, expected in zip(batches, expected_batches):
            assert sorted(batch) == sorted(expected)

    def test_batch_sampler_with_shuffle(self):
        """Test batch sampler with shuffling."""
        # Create sample lengths
        lengths = [5, 2, 3, 6, 2, 3, 6]

        # Create batch sampler
        batch_size = 2
        sampler = BatchSampler(lengths, batch_size, shuffle=True)

        # Set a specific seed for reproducibility
        np.random.seed(42)

        # Check number of batches
        assert len(sampler) == 4  # ceil(7/2) = 4

        # Check batches
        batches = list(sampler)
        assert len(batches) == 4

        # Each index should appear exactly once
        all_indices = []
        for batch in batches:
            all_indices.extend(batch)

        assert sorted(all_indices) == list(range(len(lengths)))


class TestCollateFunction:
    """Tests for the collate_fn function."""

    def test_collate_function(self):
        """Test the collate function."""
        # Create sample batch
        batch = [
            ([1, 2, 3], [10, 11, 12, 13]),
            ([4, 5], [14, 15, 16]),
            ([6, 7, 8, 9], [17, 18, 19]),
        ]

        # Apply collate function
        result = collate_fn(batch)

        # Check keys
        assert set(result.keys()) == {
            "src",
            "tgt",
            "src_mask",
            "tgt_mask",
            "src_lengths",
            "tgt_lengths",
        }

        # Check shapes
        assert result["src"].shape == (3, 4)  # batch_size=3, max_src_len=4
        assert result["tgt"].shape == (3, 4)  # batch_size=3, max_tgt_len=4
        assert result["src_mask"].shape == (3, 4)
        assert result["tgt_mask"].shape == (3, 4)
        assert result["src_lengths"].shape == (3,)
        assert result["tgt_lengths"].shape == (3,)

        # Check values
        expected_src = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0], [6, 7, 8, 9]], dtype=torch.long)

        expected_tgt = torch.tensor(
            [[10, 11, 12, 13], [14, 15, 16, 0], [17, 18, 19, 0]], dtype=torch.long
        )

        expected_src_mask = torch.tensor(
            [[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]],
            dtype=torch.float,
        )

        expected_tgt_mask = torch.tensor(
            [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.0]],
            dtype=torch.float,
        )

        expected_src_lengths = torch.tensor([3, 2, 4], dtype=torch.long)
        expected_tgt_lengths = torch.tensor([4, 3, 3], dtype=torch.long)

        assert torch.all(result["src"] == expected_src)
        assert torch.all(result["tgt"] == expected_tgt)
        assert torch.all(result["src_mask"] == expected_src_mask)
        assert torch.all(result["tgt_mask"] == expected_tgt_mask)
        assert torch.all(result["src_lengths"] == expected_src_lengths)
        assert torch.all(result["tgt_lengths"] == expected_tgt_lengths)
