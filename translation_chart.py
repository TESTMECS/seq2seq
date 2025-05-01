#!/usr/bin/env python
"""
Script to display English sentences and their French translations side by side in a chart.
"""

import os
import torch
import matplotlib.pyplot as plt
import argparse
from tokenizers import Tokenizer

from model import Encoder, Decoder, Attention, Seq2Seq
from utils import translate_sentence


def load_model(
    model_path,
    src_vocab_size,
    tgt_vocab_size,
    device,
    use_attention=True,
    hidden_dim=256,
    emb_dim=128,
    n_layers=2,
    dropout=0.2,
):
    """Load a trained sequence-to-sequence model."""
    # Create encoder
    encoder = Encoder(
        input_dim=src_vocab_size,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=dropout,
    )

    # Create attention mechanism if needed
    attention = None
    if use_attention:
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

    # Create model
    model = Seq2Seq(
        encoder=encoder,
        decoder=decoder,
        device=device,
        use_attention=use_attention,
    )

    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    return model


def display_translations(
    sentences,
    model,
    src_tokenizer,
    tgt_tokenizer,
    device,
    save_path=None,
    show_plot=True,
    max_length=None,
):
    """Display translations for a list of sentences."""
    translations = []

    # Get translations for each sentence
    for sentence in sentences:
        translation, _ = translate_sentence(
            model=model,
            sentence=sentence,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            device=device,
        )
        translations.append(translation)

    # Display as a table
    create_translation_chart(sentences, translations, save_path, show_plot, max_length)

    return translations


def create_translation_chart(
    sources, translations, save_path=None, show_plot=True, max_length=None
):
    """Create a chart displaying source sentences and their translations side by side."""
    # Check for empty inputs
    if not sources or not translations:
        # Create an empty chart with a message
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No translation data available",
            ha="center",
            va="center",
            fontsize=14,
        )

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)
            print(f"Empty chart saved to {save_path}")

        if not show_plot:
            plt.close()
        return

    # Prepare data
    if max_length:
        sources = [
            s[:max_length] + "..." if len(s) > max_length else s for s in sources
        ]
        translations = [
            t[:max_length] + "..." if len(t) > max_length else t for t in translations
        ]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, max(len(sources) * 0.5 + 1.5, 3)))

    # Hide axes
    ax.axis("off")

    # Create table
    table_data = [[src, trans] for src, trans in zip(sources, translations)]

    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=["English Source", "French Translation"],
        loc="center",
        cellLoc="left",
        colWidths=[0.5, 0.5],
        colColours=["#e6f2ff", "#ffe6e6"],
    )

    # Set font size
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # Adjust cell height
    for key, cell in table.get_celld().items():
        if key[0] == 0:  # Header row
            cell.set_height(0.1)
            cell.set_text_props(weight="bold")
        else:  # Data rows
            cell.set_height(0.08)
            # Alternate row colors
            if key[0] % 2 == 1:
                cell.set_facecolor("#f5f5f5")
            else:
                cell.set_facecolor("white")

    # Add title
    plt.suptitle("English to French Translation Results", fontsize=14)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # Save if requested
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Chart saved to {save_path}")

    # Show if requested
    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    """Main function to parse arguments and display translation chart."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Display English to French translations in a chart"
    )

    # Model parameters
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/model_attn_final.pt",
        help="Path to trained model",
    )
    parser.add_argument(
        "--src_tokenizer",
        type=str,
        default="en_tokenizer.json",
        help="Path to source tokenizer",
    )
    parser.add_argument(
        "--tgt_tokenizer",
        type=str,
        default="fr_tokenizer.json",
        help="Path to target tokenizer",
    )
    parser.add_argument(
        "--use_attention",
        action="store_true",
        default=True,
        help="Whether the model uses attention",
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=256, help="Hidden dimension size"
    )
    parser.add_argument(
        "--emb_dim", type=int, default=128, help="Embedding dimension size"
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=2,
        help="Number of layers in encoder and decoder",
    )

    # Display parameters
    parser.add_argument(
        "--save_path",
        type=str,
        default="models/translation_chart.png",
        help="Path to save the visualization",
    )
    parser.add_argument(
        "--show_plot",
        action="store_true",
        default=True,
        help="Whether to display the plot",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Maximum length of sentences to display",
    )

    # Other parameters
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for model",
    )

    args = parser.parse_args()

    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load tokenizers
    src_tokenizer = Tokenizer.from_file(args.src_tokenizer)
    tgt_tokenizer = Tokenizer.from_file(args.tgt_tokenizer)

    # Get vocabulary sizes
    src_vocab_size = src_tokenizer.get_vocab_size()
    tgt_vocab_size = tgt_tokenizer.get_vocab_size()

    # Load model
    model = load_model(
        model_path=args.model_path,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        device=device,
        use_attention=args.use_attention,
        hidden_dim=args.hidden_dim,
        emb_dim=args.emb_dim,
        n_layers=args.n_layers,
    )

    # Example sentences
    sentences = [
        "Hello, how are you?",
        "I love programming in Python.",
        "The weather is nice today.",
        "What time is it?",
        "Can you help me with this translation?",
        "I'm learning machine translation.",
        "This is a sequence-to-sequence model.",
        "Neural networks are powerful for language tasks.",
        "Attention mechanisms improve translation quality.",
        "Thank you for your help!",
    ]

    # Display translations
    print("Translating sentences...")
    display_translations(
        sentences=sentences,
        model=model,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        device=device,
        save_path=args.save_path,
        show_plot=args.show_plot,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()
