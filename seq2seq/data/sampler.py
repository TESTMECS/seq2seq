"""
Custom batch sampler for efficient training.
"""

import torch
import numpy as np
from typing import Iterator, List


class BatchSampler:
    """
    Custom batch sampler for sequence-to-sequence model training.
    Creates batches of similar lengths to minimize padding.
    """

    def __init__(
        self,
        dataset_length: int,
        lens: List[int],
        batch_size: int,
        shuffle: bool = True,
    ):
        """
        Initialize the batch sampler.

        Args:
            dataset_length: Number of samples in the dataset
            lens: List of sequence lengths for each sample
            batch_size: Batch size
            shuffle: Whether to shuffle the dataset
        """
        self.dataset_length = dataset_length
        self.lens = np.array(lens)
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[List[int]]:
        """Generate batches of indices."""
        # Create indices and sort by sequence length
        indices = np.arange(self.dataset_length)
        sort_indices = np.argsort(self.lens)
        indices = indices[sort_indices]

        # Shuffle the sorted indices (maintains grouping by length but randomizes order)
        if self.shuffle:
            # Shuffle in chunks to keep similar lengths together
            chunks = len(indices) // self.batch_size
            if chunks > 1:
                for i in range(chunks):
                    start_idx = i * self.batch_size
                    end_idx = min((i + 1) * self.batch_size, len(indices))
                    np.random.shuffle(indices[start_idx:end_idx])

        # Create batches
        batches = []
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            if len(batch_indices) > 0:  # Ensure we don't add empty batches
                batches.append(batch_indices.tolist())

        # Shuffle the order of batches
        if self.shuffle:
            np.random.shuffle(batches)

        # Return batches
        for batch in batches:
            yield batch

    def __len__(self) -> int:
        """Return the number of batches."""
        return (self.dataset_length + self.batch_size - 1) // self.batch_size
