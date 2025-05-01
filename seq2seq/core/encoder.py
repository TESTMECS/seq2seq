"""
Encoder module for the sequence-to-sequence model.
"""

import torch
import torch.nn as nn
from typing import Tuple


class Encoder(nn.Module):
    """
    Encoder module for the sequence-to-sequence model.
    Encodes the input sequence into a context vector.
    """

    def __init__(
        self,
        input_dim: int,
        emb_dim: int,
        hidden_dim: int,
        n_layers: int,
        dropout: float,
    ):
        """Initialize the encoder."""
        super().__init__()

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.GRU(
            emb_dim,
            hidden_dim,
            n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, src: torch.Tensor, src_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the encoder.

        Args:
            src: Source sequence [batch_size, src_len]
            src_lengths: Length of each source sequence [batch_size]

        Returns:
            outputs: Encoder outputs [batch_size, src_len, hidden_dim]
            hidden: Final hidden state [batch_size, hidden_dim]
        """
        # Embedding layer
        embedded = self.dropout(self.embedding(src))  # [batch_size, src_len, emb_dim]

        # Pack padded sequences
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # Run through RNN
        outputs, hidden = self.rnn(packed)  # hidden: [n_layers*2, batch_size, hidden_dim]

        # Unpack sequences
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True
        )  # [batch_size, src_len, hidden_dim*2]

        # Combine bidirectional outputs
        hidden = torch.cat(
            (hidden[-2, :, :], hidden[-1, :, :]), dim=1
        )  # [batch_size, hidden_dim*2]
        hidden = torch.tanh(self.fc(hidden))  # [batch_size, hidden_dim]

        return outputs, hidden