"""
Sequence-to-sequence model for machine translation.
"""

import torch
import torch.nn as nn
from typing import Dict

from seq2seq.core.encoder import Encoder
from seq2seq.core.decoder import Decoder


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