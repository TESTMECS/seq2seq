"""
Sequence-to-sequence model architecture for machine translation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


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


class Seq2Seq(nn.Module):
    """
    Sequence-to-sequence model for machine translation.
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        device: torch.device,
        use_attention: bool = False,
    ):
        """Initialize the sequence-to-sequence model."""
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.use_attention = use_attention

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_lengths: torch.Tensor,
        src_mask: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the sequence-to-sequence model.

        Args:
            src: Source sequence [batch_size, src_len]
            tgt: Target sequence [batch_size, tgt_len]
            src_lengths: Length of each source sequence [batch_size]
            src_mask: Source mask [batch_size, src_len]
            teacher_forcing_ratio: Probability of using teacher forcing

        Returns:
            Dictionary containing:
                outputs: Predicted token ids [batch_size, tgt_len, output_dim]
                attention_weights: Attention weights [batch_size, tgt_len, src_len] (if using attention)
        """
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        tgt_vocab_size = self.decoder.output_dim

        # Initialize outputs tensor
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        # Initialize attention weights tensor if using attention
        attention_weights = None
        if self.use_attention:
            attention_weights = torch.zeros(batch_size, tgt_len, src.shape[1]).to(self.device)

        # Encode the source sequence
        encoder_outputs, hidden = self.encoder(src, src_lengths)

        # First input to the decoder is the <BOS> token
        decoder_input = tgt[:, 0]

        # Decode one step at a time
        for t in range(1, tgt_len):
            # Run the decoder for one time step
            output, hidden, attn_weights = self.decoder(
                decoder_input, hidden, encoder_outputs, src_mask
            )

            # Store the output and attention weights
            outputs[:, t, :] = output
            if self.use_attention and attn_weights is not None:
                attention_weights[:, t, :] = attn_weights

            # Determine the next input to the decoder
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio

            # If teacher forcing, use the ground-truth token
            # Otherwise, use the token with the highest predicted probability
            decoder_input = tgt[:, t] if teacher_force else output.argmax(1)

        return {"outputs": outputs, "attention_weights": attention_weights}

    def translate(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
        src_mask: torch.Tensor,
        max_len: int,
        bos_idx: int,
        eos_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Translate a source sequence to target language.

        Args:
            src: Source sequence [batch_size, src_len]
            src_lengths: Length of each source sequence [batch_size]
            src_mask: Source mask [batch_size, src_len]
            max_len: Maximum target sequence length
            bos_idx: Index of the beginning-of-sequence token
            eos_idx: Index of the end-of-sequence token

        Returns:
            Dictionary containing:
                translations: Translated token ids [batch_size, max_len]
                attention_weights: Attention weights [batch_size, max_len, src_len] (if using attention)
        """
        batch_size = src.shape[0]

        # Initialize translations tensor
        translations = torch.ones(batch_size, max_len, dtype=torch.long).to(self.device) * eos_idx
        translations[:, 0] = bos_idx

        # Initialize attention weights tensor if using attention
        attention_weights = None
        if self.use_attention:
            attention_weights = torch.zeros(batch_size, max_len, src.shape[1]).to(self.device)

        # Set to evaluation mode
        self.eval()

        with torch.no_grad():
            # Encode the source sequence
            encoder_outputs, hidden = self.encoder(src, src_lengths)

            # First input to the decoder is the <BOS> token
            decoder_input = torch.ones(batch_size, dtype=torch.long).fill_(bos_idx).to(self.device)

            # Store completed sequences
            completed = torch.zeros(batch_size, dtype=torch.bool).to(self.device)

            # Decode one step at a time
            for t in range(1, max_len):
                # Run the decoder for one time step
                output, hidden, attn_weights = self.decoder(
                    decoder_input, hidden, encoder_outputs, src_mask
                )

                # Get the token with the highest predicted probability
                predicted = output.argmax(1)

                # Store the predicted token and attention weights
                translations[:, t] = predicted
                if self.use_attention and attn_weights is not None:
                    attention_weights[:, t, :] = attn_weights

                # Check for completed sequences
                completed = completed | (predicted == eos_idx)
                if completed.all():
                    break

                # Set the next decoder input
                decoder_input = predicted

        return {"translations": translations, "attention_weights": attention_weights}


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
