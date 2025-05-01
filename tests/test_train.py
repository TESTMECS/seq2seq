"""
Tests for training functions in the sequence-to-sequence model.
"""

import torch
import torch.nn.functional as F
import pytest
import sys
import os
import numpy as np
from unittest.mock import patch, MagicMock, call
import tempfile

# Add parent directory to path to import from modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import train_epoch, evaluate, train

# Import test_model with a name that won't be treated as a test
from train import test_model as model_test_function
from model import Seq2Seq, Seq2SeqLoss


class TestTrainEpoch:
    """Tests for the train_epoch function."""

    def test_train_epoch(self):
        """Test the train_epoch function with mock objects."""
        # Mock model, dataloader, optimizer, criterion, device
        model = MagicMock(spec=Seq2Seq)
        dataloader = MagicMock()
        optimizer = MagicMock()
        criterion = MagicMock(spec=Seq2SeqLoss)
        device = torch.device("cpu")

        # Create a batch
        batch = {
            "src": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "tgt": torch.tensor([[10, 11, 12], [13, 14, 15]]),
            "src_mask": torch.ones(2, 3),
            "tgt_mask": torch.ones(2, 3),
            "src_lengths": torch.tensor([3, 3]),
            "tgt_lengths": torch.tensor([3, 3]),
        }

        # Make dataloader yield the batch
        dataloader.__iter__.return_value = [batch]
        dataloader.__len__.return_value = 1

        # Make model return outputs
        model.return_value = {"outputs": torch.randn(2, 3, 10)}

        # Skip the backward call to avoid the need for a proper loss tensor with grad_fn
        # We'll patch the loss.backward() method to do nothing
        loss_tensor = torch.tensor(0.5, requires_grad=False)
        criterion.return_value = loss_tensor

        # Patch the backward method to avoid the backward call error
        with patch.object(torch.Tensor, "backward", return_value=None):
            # Call function
            loss = train_epoch(
                model=model,
                dataloader=dataloader,
                optimizer=optimizer,
                criterion=criterion,
                clip=1.0,
                device=device,
                teacher_forcing_ratio=0.5,
            )

        # Check result
        assert abs(loss - 0.5) < 1e-5  # Use floating-point comparison with tolerance

        # Verify correct calls
        model.train.assert_called_once()
        optimizer.zero_grad.assert_called_once()
        model.assert_called_once()
        criterion.assert_called_once()
        optimizer.step.assert_called_once()


class TestEvaluate:
    """Tests for the evaluate function."""

    def test_evaluate(self):
        """Test the evaluate function with mock objects."""
        # Mock model, dataloader, criterion, device
        model = MagicMock(spec=Seq2Seq)
        dataloader = MagicMock()
        criterion = MagicMock(spec=Seq2SeqLoss)
        device = torch.device("cpu")

        # Create a batch
        batch = {
            "src": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "tgt": torch.tensor([[10, 11, 12], [13, 14, 15]]),
            "src_mask": torch.ones(2, 3),
            "tgt_mask": torch.ones(2, 3),
            "src_lengths": torch.tensor([3, 3]),
            "tgt_lengths": torch.tensor([3, 3]),
        }

        # Make dataloader yield the batch
        dataloader.__iter__.return_value = [batch]
        dataloader.__len__.return_value = 1

        # Make model return outputs
        model.return_value = {"outputs": torch.randn(2, 3, 10)}

        # Make criterion return loss
        # Create a tensor with a mocked item method that returns a scalar
        loss_tensor = torch.tensor(0.3)
        # MagicMock doesn't work directly with tensor.item(), so we need a different approach
        criterion.return_value = loss_tensor

        # Call function
        loss = evaluate(model=model, dataloader=dataloader, criterion=criterion, device=device)

        # Check result
        assert abs(loss - 0.3) < 1e-5  # Use floating-point comparison with tolerance

        # Verify correct calls
        model.eval.assert_called_once()
        model.assert_called_once()
        criterion.assert_called_once()


class TestTrain:
    """Tests for the train function."""

    def test_train(self):
        """Test the train function with mock objects."""
        # Mock model, dataloaders, optimizer, criterion, device
        model = MagicMock(spec=Seq2Seq)
        train_dataloader = MagicMock()
        val_dataloader = MagicMock()
        optimizer = MagicMock()
        criterion = MagicMock(spec=Seq2SeqLoss)
        device = torch.device("cpu")

        # Use tempfile for model saving
        with tempfile.NamedTemporaryFile() as tmp:
            # Use patching for train_epoch and evaluate
            with (
                patch("train.train_epoch") as mock_train_epoch,
                patch("train.evaluate") as mock_evaluate,
                patch("torch.save") as mock_save,
            ):
                # Set return values for train_epoch and evaluate
                # First epoch: train_loss=0.5, val_loss=0.4
                # Second epoch: train_loss=0.3, val_loss=0.2
                mock_train_epoch.side_effect = [0.5, 0.3]
                mock_evaluate.side_effect = [0.4, 0.2]

                # Call function
                history = train(
                    model=model,
                    train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader,
                    optimizer=optimizer,
                    criterion=criterion,
                    clip=1.0,
                    device=device,
                    n_epochs=2,
                    patience=5,
                    teacher_forcing_ratio=0.5,
                    model_path=tmp.name,
                )

                # Check result
                assert history["train_loss"] == [0.5, 0.3]
                assert history["val_loss"] == [0.4, 0.2]

                # Verify correct calls
                assert mock_train_epoch.call_count == 2
                assert mock_evaluate.call_count == 2
                assert (
                    mock_save.call_count == 2
                )  # Save model twice (once per epoch since val_loss improves)


class TestTestModel:
    """Tests for the test_model function."""

    def test_test_model(self):
        """Test the test_model function with mock objects."""
        # Mock model, dataloader, criterion, device, tokenizers
        model = MagicMock(spec=Seq2Seq)
        test_dataloader = MagicMock()
        criterion = MagicMock(spec=Seq2SeqLoss)
        device = torch.device("cpu")
        src_tokenizer = MagicMock()
        tgt_tokenizer = MagicMock()

        # Create a batch
        batch = {
            "src": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "tgt": torch.tensor([[10, 11, 12], [13, 14, 15]]),
            "src_mask": torch.ones(2, 3),
            "tgt_mask": torch.ones(2, 3),
            "src_lengths": torch.tensor([3, 3]),
            "tgt_lengths": torch.tensor([3, 3]),
        }

        # Make dataloader yield the batch
        test_dataloader.__iter__.return_value = [batch]
        test_dataloader.__len__.return_value = 1

        # Make model return outputs for forward pass
        model.return_value = {"outputs": torch.randn(2, 3, 10)}

        # Make model return translations for translate method
        model.translate.return_value = {
            "translations": torch.tensor([[1, 3, 4, 2, 0], [1, 5, 6, 2, 0]]),
            "attention_weights": torch.randn(2, 5, 3),
        }

        # Configure tokenizers
        src_tokenizer.decode.side_effect = ["source text 1", "source text 2"]
        tgt_tokenizer.decode.side_effect = [
            "target text 1",
            "target text 2",
            "predicted text 1",
            "predicted text 2",
        ]
        tgt_tokenizer.token_to_id.side_effect = lambda token: 1 if token == "[BOS]" else 2

        # Make criterion return loss
        # Create a tensor with a mocked item method that returns a scalar
        loss_tensor = torch.tensor(0.3)
        # MagicMock doesn't work directly with tensor.item(), so we need a different approach
        criterion.return_value = loss_tensor

        # Patch bleu_score
        with patch("train.bleu_score", return_value=0.7):
            # Call function
            test_loss, bleu = model_test_function(
                model=model,
                test_dataloader=test_dataloader,
                criterion=criterion,
                device=device,
                src_tokenizer=src_tokenizer,
                tgt_tokenizer=tgt_tokenizer,
                num_examples=2,
            )

            # Check results
            assert abs(test_loss - 0.3) < 1e-5  # Use floating-point comparison with tolerance
            assert bleu == 0.7

            # Verify correct calls
            model.eval.assert_called_once()
            model.assert_called_once()
            criterion.assert_called_once()
            model.translate.assert_called_once()
