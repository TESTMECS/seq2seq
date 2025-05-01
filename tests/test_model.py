"""
Tests for the sequence-to-sequence model.
"""

import torch
import pytest
import sys
import os

# Add parent directory to path to import from modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import Encoder, Decoder, Attention, Seq2Seq, Seq2SeqLoss


class TestEncoder:
    """Tests for the Encoder class."""

    def test_encoder_forward(self):
        """Test forward pass of the encoder."""
        # Define parameters
        batch_size = 5
        seq_len = 10
        input_dim = 100
        emb_dim = 32
        hidden_dim = 64
        n_layers = 2
        dropout = 0.1

        # Create encoder
        encoder = Encoder(
            input_dim=input_dim,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
        )

        # Create input tensors
        src = torch.randint(0, input_dim, (batch_size, seq_len))
        src_lengths = torch.tensor([seq_len] * batch_size)

        # Forward pass
        outputs, hidden = encoder(src, src_lengths)

        # Check shapes
        assert outputs.shape == (batch_size, seq_len, hidden_dim * 2)  # Bidirectional
        assert hidden.shape == (batch_size, hidden_dim)


class TestAttention:
    """Tests for the Attention class."""

    def test_attention_forward(self):
        """Test forward pass of the attention mechanism."""
        # Define parameters
        batch_size = 5
        seq_len = 10
        hidden_dim = 64

        # Create attention mechanism
        attention = Attention(hidden_dim=hidden_dim)

        # Create input tensors
        hidden = torch.randn(batch_size, hidden_dim)
        # For bidirectional encoder, encoder outputs have 2x hidden_dim
        encoder_outputs = torch.randn(batch_size, seq_len, hidden_dim * 2)
        mask = torch.ones(batch_size, seq_len)

        # Forward pass
        attention_weights = attention(hidden, encoder_outputs, mask)

        # Check shapes
        assert attention_weights.shape == (batch_size, seq_len)

        # Check that attention weights sum to 1 for each batch
        for i in range(batch_size):
            assert torch.isclose(attention_weights[i].sum(), torch.tensor(1.0))


class TestDecoder:
    """Tests for the Decoder class."""

    def test_decoder_forward_no_attention(self):
        """Test forward pass of the decoder without attention."""
        # Define parameters
        batch_size = 5
        output_dim = 100
        emb_dim = 32
        hidden_dim = 64
        n_layers = 2
        dropout = 0.1

        # Create decoder
        decoder = Decoder(
            output_dim=output_dim,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            attention=None,
        )

        # Create input tensors
        input = torch.randint(0, output_dim, (batch_size,))
        # In the decoder's forward method, we expand hidden to [1, batch_size, hidden_dim]
        # for the GRU, and then squeeze it back to [batch_size, hidden_dim]
        hidden = torch.randn(batch_size, hidden_dim)

        # Forward pass
        output, next_hidden, attention_weights = decoder(input, hidden)

        # Check shapes
        assert output.shape == (batch_size, output_dim)
        assert next_hidden.shape == (batch_size, hidden_dim)
        assert attention_weights is None

    def test_decoder_forward_with_attention(self):
        """Test forward pass of the decoder with attention."""
        # Define parameters
        batch_size = 5
        seq_len = 10
        output_dim = 100
        emb_dim = 32
        hidden_dim = 64
        n_layers = 2
        dropout = 0.1

        # Create attention mechanism
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

        # Create input tensors
        input = torch.randint(0, output_dim, (batch_size,))
        hidden = torch.randn(batch_size, hidden_dim)
        # For bidirectional encoder, encoder outputs have 2x hidden_dim
        encoder_outputs = torch.randn(batch_size, seq_len, hidden_dim * 2)
        mask = torch.ones(batch_size, seq_len)

        # Forward pass
        output, next_hidden, attention_weights = decoder(input, hidden, encoder_outputs, mask)

        # Check shapes
        assert output.shape == (batch_size, output_dim)
        assert next_hidden.shape == (batch_size, hidden_dim)
        assert attention_weights.shape == (batch_size, seq_len)


class TestSeq2Seq:
    """Tests for the Seq2Seq class."""

    def test_seq2seq_forward_no_attention(self):
        """Test forward pass of the seq2seq model without attention."""
        # Define parameters
        batch_size = 5
        src_len = 10
        tgt_len = 8
        src_vocab_size = 200
        tgt_vocab_size = 300
        emb_dim = 32
        hidden_dim = 64
        n_layers = 2
        dropout = 0.1

        # Create encoder
        encoder = Encoder(
            input_dim=src_vocab_size,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
        )

        # Create decoder
        decoder = Decoder(
            output_dim=tgt_vocab_size,
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

        # Create input tensors
        src = torch.randint(1, src_vocab_size, (batch_size, src_len))  # Avoid 0 (padding)
        tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_len))  # Avoid 0 (padding)
        src_lengths = torch.tensor([src_len] * batch_size)
        src_mask = torch.ones(batch_size, src_len)

        # Forward pass
        output_dict = model(
            src=src,
            tgt=tgt,
            src_lengths=src_lengths,
            src_mask=src_mask,
            teacher_forcing_ratio=0.5,
        )

        # Check shapes
        outputs = output_dict["outputs"]
        assert outputs.shape == (batch_size, tgt_len, tgt_vocab_size)
        assert output_dict["attention_weights"] is None

    def test_seq2seq_forward_with_attention(self):
        """Test forward pass of the seq2seq model with attention."""
        # Define parameters
        batch_size = 5
        src_len = 10
        tgt_len = 8
        src_vocab_size = 200
        tgt_vocab_size = 300
        emb_dim = 32
        hidden_dim = 64
        n_layers = 2
        dropout = 0.1

        # Create encoder
        encoder = Encoder(
            input_dim=src_vocab_size,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
        )

        # Create attention mechanism
        attention = Attention(hidden_dim=hidden_dim)

        # Create decoder
        decoder = Decoder(
            output_dim=tgt_vocab_size,
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

        # Create input tensors
        src = torch.randint(1, src_vocab_size, (batch_size, src_len))  # Avoid 0 (padding)
        tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_len))  # Avoid 0 (padding)
        src_lengths = torch.tensor([src_len] * batch_size)
        src_mask = torch.ones(batch_size, src_len)

        # Forward pass
        output_dict = model(
            src=src,
            tgt=tgt,
            src_lengths=src_lengths,
            src_mask=src_mask,
            teacher_forcing_ratio=0.5,
        )

        # Check shapes
        outputs = output_dict["outputs"]
        attention_weights = output_dict["attention_weights"]
        assert outputs.shape == (batch_size, tgt_len, tgt_vocab_size)
        assert attention_weights.shape == (batch_size, tgt_len, src_len)

    def test_seq2seq_translate(self):
        """Test translation function of the seq2seq model."""
        # Define parameters
        batch_size = 5
        src_len = 10
        max_len = 15
        src_vocab_size = 200
        tgt_vocab_size = 300
        emb_dim = 32
        hidden_dim = 64
        n_layers = 2
        dropout = 0.1
        bos_idx = 1
        eos_idx = 2

        # Create encoder
        encoder = Encoder(
            input_dim=src_vocab_size,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
        )

        # Create attention mechanism
        attention = Attention(hidden_dim=hidden_dim)

        # Create decoder
        decoder = Decoder(
            output_dim=tgt_vocab_size,
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

        # Create input tensors
        src = torch.randint(1, src_vocab_size, (batch_size, src_len))  # Avoid 0 (padding)
        src_lengths = torch.tensor([src_len] * batch_size)
        src_mask = torch.ones(batch_size, src_len)

        # Translate
        translation_dict = model.translate(
            src=src,
            src_lengths=src_lengths,
            src_mask=src_mask,
            max_len=max_len,
            bos_idx=bos_idx,
            eos_idx=eos_idx,
        )

        # Check shapes
        translations = translation_dict["translations"]
        attention_weights = translation_dict["attention_weights"]
        assert translations.shape == (batch_size, max_len)
        assert attention_weights.shape == (batch_size, max_len, src_len)


class TestSeq2SeqLoss:
    """Tests for the Seq2SeqLoss class."""

    def test_seq2seq_loss(self):
        """Test the seq2seq loss function."""
        # Define parameters
        batch_size = 5
        tgt_len = 8
        tgt_vocab_size = 300
        pad_idx = 0

        # Create loss function
        criterion = Seq2SeqLoss(pad_idx=pad_idx)

        # Create input tensors
        outputs = torch.randn(batch_size, tgt_len, tgt_vocab_size)
        targets = torch.randint(0, tgt_vocab_size, (batch_size, tgt_len))

        # Set some targets to padding
        targets[:, -2:] = pad_idx

        # Calculate loss
        loss = criterion(outputs, targets)

        # Check that loss is a scalar
        assert loss.dim() == 0
        assert loss.item() > 0
