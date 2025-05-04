"""
Collate functions for data batching.
"""

import torch
from typing import Dict, List


def collate_fn(batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.
    Pads sequences within a batch to the same length.

    Args:
        batch: List of dictionaries containing source and target texts

    Returns:
        Dictionary containing batched tensors
    """
    # Separate source and target sequences
    src_seqs = [torch.tensor(item["src"]) for item in batch]
    tgt_seqs = [torch.tensor(item["tgt"]) for item in batch]

    # Get lengths of source sequences for packing
    src_lens = torch.tensor([len(s) for s in src_seqs])

    # Pad sequences
    src_padded = torch.nn.utils.rnn.pad_sequence(src_seqs, batch_first=True, padding_value=0)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_seqs, batch_first=True, padding_value=0)

    # Create mask for source sequences (1 for tokens, 0 for padding)
    src_mask = (src_padded != 0).float()

    # Return batch
    return {
        "src": src_padded,
        "tgt": tgt_padded,
        "src_lengths": src_lens,
        "src_mask": src_mask,
    }
