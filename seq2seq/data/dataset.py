"""
Dataset implementation for sequence-to-sequence translation.
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Any


class TranslationDataset(Dataset):
    """Dataset for sequence-to-sequence translation."""

    def __init__(self, src_texts: List[List[int]], tgt_texts: List[List[int]]):
        """
        Initialize the dataset.

        Args:
            src_texts: List of source language tokenized texts
            tgt_texts: List of target language tokenized texts
        """
        super().__init__()
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.src_texts)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        """
        Return a sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Dictionary containing source and target texts
        """
        return {
            "src": self.src_texts[idx],
            "tgt": self.tgt_texts[idx],
        }