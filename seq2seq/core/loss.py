"""
Loss function for sequence-to-sequence model.
"""

import torch
import torch.nn as nn


class Seq2SeqLoss(nn.Module):
    """Loss function for sequence-to-sequence model."""

    def __init__(self, pad_idx: int):
        """Initialize the loss function."""
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate the loss for the sequence-to-sequence model.

        Args:
            outputs: Model outputs [batch_size, tgt_len, output_dim]
            targets: Target token ids [batch_size, tgt_len]

        Returns:
            loss: The loss value
        """
        batch_size = outputs.shape[0]
        tgt_len = outputs.shape[1]
        output_dim = outputs.shape[2]

        # Reshape outputs and targets for the loss function
        outputs = outputs.contiguous().view(-1, output_dim)
        targets = targets.contiguous().view(-1)

        # Calculate the loss
        loss = self.criterion(outputs, targets)

        return loss