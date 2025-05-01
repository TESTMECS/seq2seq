"""
Integration tests for the sequence-to-sequence model pipeline.
"""

import torch
import pytest
import sys
import os

# Add parent directory to path to import from modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import Encoder, Decoder, Attention, Seq2Seq, Seq2SeqLoss


class TestModelIntegration:
    """Integration tests for the model pipeline."""

    @pytest.mark.slow  # Mark this test as slow-running
    def test_end_to_end_no_attention(self):
        """Test model end-to-end without attention mechanism."""
        # Parameters
        input_dim = 100  # Source vocabulary size
        output_dim = 120  # Target vocabulary size
        emb_dim = 32  # Embedding dimension
        hidden_dim = 64  # Hidden dimension
        n_layers = 2  # Number of layers
        dropout = 0.1  # Dropout probability
        batch_size = 2  # Batch size
        src_len = 5  # Source sequence length
        tgt_len = 6  # Target sequence length

        # Create encoder
        encoder = Encoder(
            input_dim=input_dim,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
        )

        # Create decoder
        decoder = Decoder(
            output_dim=output_dim,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            attention=None,
        )

        # Create seq2seq model
        device = torch.device("cpu")
        model = Seq2Seq(
            encoder=encoder,
            decoder=decoder,
            device=device,
            use_attention=False,
        )

        # Create inputs
        src = torch.randint(1, input_dim, (batch_size, src_len))
        tgt = torch.randint(1, output_dim, (batch_size, tgt_len))
        src_lengths = torch.full((batch_size,), src_len)
        src_mask = torch.ones_like(src).float()

        # Forward pass
        output_dict = model(
            src=src,
            tgt=tgt,
            src_lengths=src_lengths,
            src_mask=src_mask,
            teacher_forcing_ratio=0.5,
        )

        # Check outputs
        assert output_dict["outputs"].shape == (batch_size, tgt_len, output_dim)
        assert output_dict["attention_weights"] is None

        # Test translation
        bos_idx = 1
        eos_idx = 2
        max_len = 10

        translation_dict = model.translate(
            src=src,
            src_lengths=src_lengths,
            src_mask=src_mask,
            max_len=max_len,
            bos_idx=bos_idx,
            eos_idx=eos_idx,
        )

        # Check translation outputs
        assert translation_dict["translations"].shape == (batch_size, max_len)
        assert translation_dict["attention_weights"] is None

        # Test loss calculation
        criterion = Seq2SeqLoss(pad_idx=0)
        loss = criterion(output_dict["outputs"], tgt)

        # Check loss
        assert loss.shape == torch.Size([])  # Scalar
        assert loss.item() > 0  # Loss should be positive

    @pytest.mark.slow  # Mark this test as slow-running
    def test_end_to_end_with_attention(self):
        """Test model end-to-end with attention mechanism."""
        # Parameters
        input_dim = 100  # Source vocabulary size
        output_dim = 120  # Target vocabulary size
        emb_dim = 32  # Embedding dimension
        hidden_dim = 64  # Hidden dimension
        n_layers = 2  # Number of layers
        dropout = 0.1  # Dropout probability
        batch_size = 2  # Batch size
        src_len = 5  # Source sequence length
        tgt_len = 6  # Target sequence length

        # Create encoder
        encoder = Encoder(
            input_dim=input_dim,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
        )

        # Create attention
        attention = Attention(hidden_dim=hidden_dim)

        # Create decoder
        decoder = Decoder(
            output_dim=output_dim,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            attention=attention,
        )

        # Create seq2seq model
        device = torch.device("cpu")
        model = Seq2Seq(
            encoder=encoder,
            decoder=decoder,
            device=device,
            use_attention=True,
        )

        # Create inputs
        src = torch.randint(1, input_dim, (batch_size, src_len))
        tgt = torch.randint(1, output_dim, (batch_size, tgt_len))
        src_lengths = torch.full((batch_size,), src_len)
        src_mask = torch.ones_like(src).float()

        # Forward pass
        output_dict = model(
            src=src,
            tgt=tgt,
            src_lengths=src_lengths,
            src_mask=src_mask,
            teacher_forcing_ratio=0.5,
        )

        # Check outputs
        assert output_dict["outputs"].shape == (batch_size, tgt_len, output_dim)
        assert output_dict["attention_weights"].shape == (batch_size, tgt_len, src_len)

        # Test translation
        bos_idx = 1
        eos_idx = 2
        max_len = 10

        translation_dict = model.translate(
            src=src,
            src_lengths=src_lengths,
            src_mask=src_mask,
            max_len=max_len,
            bos_idx=bos_idx,
            eos_idx=eos_idx,
        )

        # Check translation outputs
        assert translation_dict["translations"].shape == (batch_size, max_len)
        assert translation_dict["attention_weights"].shape == (
            batch_size,
            max_len,
            src_len,
        )

        # Test loss calculation
        criterion = Seq2SeqLoss(pad_idx=0)
        loss = criterion(output_dict["outputs"], tgt)

        # Check loss
        assert loss.shape == torch.Size([])  # Scalar
        assert loss.item() > 0  # Loss should be positive


# Skipped because it requires actual tokenizer files
@pytest.mark.skip(reason="Requires actual tokenizer files")
def test_translate_sentence_integration():
    """Test translate_sentence function with actual model and tokenizers."""
    # This would be an integration test that loads actual tokenizers and model
    # It's skipped because it requires the tokenizer files to exist
    pass
