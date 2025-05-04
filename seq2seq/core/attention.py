"""
Attention mechanism for the sequence-to-sequence model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """
    Attention mechanism for the sequence-to-sequence model.
    """

    def __init__(self, hidden_dim: int):
        """Initialize the attention mechanism."""
        super().__init__()

        self.attn = nn.Linear(hidden_dim * 3, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        # Store the input feature size for later dimension checking
        self.in_features = hidden_dim * 3

    def forward(
        self,
        hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the attention mechanism.

        Args:
            hidden: Decoder hidden state [batch_size, hidden_dim]
            encoder_outputs: Encoder outputs [batch_size, src_len, hidden_dim*2 for bidirectional or hidden_dim]
            mask: Source mask [batch_size, src_len]

        Returns:
            attention: Attention weights [batch_size, src_len]
        """
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]

        # Repeat decoder hidden state across time dimension
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)  # [batch_size, src_len, hidden_dim]

        # Ensure encoder_outputs has the expected dimension for the concatenation
        encoder_dim = encoder_outputs.shape[2]
        hidden_dim = hidden.shape[2]

        # Adapt to the actual encoder output dimensions
        # If encoder outputs are already the right dimension, use them as is
        # This handles both bidirectional (hidden_dim*2) and unidirectional cases
        if encoder_dim + hidden_dim != self.attn.in_features:
            # Project encoder outputs to match expected dimension if needed
            encoder_outputs = encoder_outputs[:, :, :hidden_dim]

        # Concatenate decoder hidden state with encoder outputs
        energy = torch.cat(
            (hidden, encoder_outputs), dim=2
        )  # [batch_size, src_len, hidden_dim+encoder_dim]

        # Calculate attention scores
        energy = torch.tanh(self.attn(energy))  # [batch_size, src_len, hidden_dim]
        attention = self.v(energy).squeeze(2)  # [batch_size, src_len]

        # Mask out padding tokens
        attention = attention.masked_fill(mask == 0, -1e10)

        # Return softmaxed attention scores
        return F.softmax(attention, dim=1)  # [batch_size, src_len]
