"""
Decoder module for the sequence-to-sequence model.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

from seq2seq.core.attention import Attention


class Decoder(nn.Module):
    """
    Decoder module for the sequence-to-sequence model.
    """

    def __init__(
        self,
        output_dim: int,
        emb_dim: int,
        hidden_dim: int,
        n_layers: int,
        dropout: float,
        attention: Optional[Attention] = None,
    ):
        """Initialize the decoder."""
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)

        # If attention is used, we need to increase the input dimension
        # For bidirectional encoder, encoder_outputs have 2x hidden_dim
        rnn_input_dim = emb_dim + hidden_dim * 2 if attention else emb_dim

        self.rnn = nn.GRU(
            rnn_input_dim,
            hidden_dim,
            n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )

        # For attention, we concatenate: output (hidden_dim) + hidden (hidden_dim) + context (hidden_dim*2)
        fc_input_dim = hidden_dim + hidden_dim + hidden_dim * 2 if attention else hidden_dim
        self.fc_out = nn.Linear(fc_input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input: torch.Tensor,
        hidden: torch.Tensor,
        encoder_outputs: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the decoder for a single time step.

        Args:
            input: Input token ids [batch_size]
            hidden: Hidden state [batch_size, hidden_dim]
            encoder_outputs: Encoder outputs [batch_size, src_len, hidden_dim*2]
            mask: Source mask [batch_size, src_len]

        Returns:
            prediction: Predicted logits [batch_size, output_dim]
            hidden: Updated hidden state [batch_size, hidden_dim]
            attention_weights: Attention weights [batch_size, src_len]
        """
        # Make sure input is a 2D tensor [batch_size, 1]
        if input.dim() == 1:
            input = input.unsqueeze(1)  # [batch_size, 1]

        # Embedding
        embedded = self.dropout(self.embedding(input))  # [batch_size, 1, emb_dim]

        # Apply attention if available
        attention_weights = None
        if self.attention and encoder_outputs is not None and mask is not None:
            attention_weights = self.attention(
                hidden, encoder_outputs, mask
            )  # [batch_size, src_len]
            context = torch.bmm(
                attention_weights.unsqueeze(1), encoder_outputs
            )  # [batch_size, 1, hidden_dim*2]

            # Concatenate embedding and context
            rnn_input = torch.cat(
                (embedded, context), dim=2
            )  # [batch_size, 1, emb_dim + hidden_dim*2]
        else:
            rnn_input = embedded

        # GRU expects hidden state to be [n_layers, batch_size, hidden_dim]
        # If hidden is [batch_size, hidden_dim], reshape it
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0).contiguous()
            if self.rnn.num_layers > 1:
                # For multi-layer GRU, repeat the hidden state for each layer
                hidden = hidden.repeat(self.rnn.num_layers, 1, 1)

        # Run through RNN
        output, hidden = self.rnn(rnn_input, hidden)
        # output: [batch_size, 1, hidden_dim], hidden: [n_layers, batch_size, hidden_dim]

        # Get final layer hidden state
        last_hidden = hidden[-1]  # [batch_size, hidden_dim]

        # Calculate prediction
        if self.attention and encoder_outputs is not None:
            # Concatenate embedding, output and context for prediction
            prediction = self.fc_out(
                torch.cat((output.squeeze(1), last_hidden, context.squeeze(1)), dim=1)
            )  # [batch_size, output_dim]
        else:
            # Use only hidden state for prediction
            prediction = self.fc_out(last_hidden)  # [batch_size, output_dim]

        return prediction, last_hidden, attention_weights
