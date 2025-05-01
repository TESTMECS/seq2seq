"""
Tests for the translation chart functionality.
"""

import os
import tempfile
import pytest
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for testing
import matplotlib.pyplot as plt

from translation_chart import create_translation_chart


def test_create_translation_chart():
    """Test the creation of a translation chart."""
    # Sample data
    sources = [
        "Hello, how are you?",
        "I love programming.",
        "The weather is nice today.",
    ]
    translations = [
        "Bonjour, comment ça va?",
        "J'adore la programmation.",
        "Le temps est beau aujourd'hui.",
    ]

    # Create a temporary file for the chart
    with tempfile.NamedTemporaryFile(suffix=".png") as temp_file:
        save_path = temp_file.name

        # Test with show_plot=False to avoid opening windows during tests
        create_translation_chart(
            sources=sources,
            translations=translations,
            save_path=save_path,
            show_plot=False,
        )

        # Check if the file was created
        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 0


def test_create_translation_chart_with_max_length():
    """Test the creation of a translation chart with max_length parameter."""
    # Sample data with long sentences
    sources = [
        "This is a very long sentence that should be truncated in the visualization."
    ]
    translations = [
        "C'est une phrase très longue qui devrait être tronquée dans la visualisation."
    ]

    # Create a temporary file for the chart
    with tempfile.NamedTemporaryFile(suffix=".png") as temp_file:
        save_path = temp_file.name

        # Test with max_length=20
        create_translation_chart(
            sources=sources,
            translations=translations,
            save_path=save_path,
            show_plot=False,
            max_length=20,
        )

        # Check if the file was created
        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 0


def test_create_translation_chart_empty_input():
    """Test the creation of a translation chart with empty input."""
    # Empty data
    sources = []
    translations = []

    # Create a temporary file for the chart
    with tempfile.NamedTemporaryFile(suffix=".png") as temp_file:
        save_path = temp_file.name

        # Test with empty inputs
        create_translation_chart(
            sources=sources,
            translations=translations,
            save_path=save_path,
            show_plot=False,
        )

        # Check if the file was created
        assert os.path.exists(save_path)
        # File should still be created even with empty inputs
        assert os.path.getsize(save_path) > 0

