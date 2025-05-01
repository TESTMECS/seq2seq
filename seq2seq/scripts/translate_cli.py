"""
Command-line interface for translating using trained models.
"""

import argparse
import torch
import os

from seq2seq.core import Encoder, Decoder, Attention, Seq2Seq
from seq2seq.data import load_tokenizers
from seq2seq.utils import translate_sentence


def main():
    """Main entry point for translation."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Translate text using a trained model")
    parser.add_argument("--src_lang", type=str, default="en", help="Source language")
    parser.add_argument("--tgt_lang", type=str, default="fr", help="Target language")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension size")
    parser.add_argument("--emb_dim", type=int, default=128, help="Embedding dimension size")
    parser.add_argument("--n_layers", type=int, default=2, help="Number of layers in encoder and decoder")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--attention", action="store_true", help="Use attention mechanism")
    parser.add_argument("--sentence", type=str, help="Sentence to translate")
    parser.add_argument("--max_len", type=int, default=100, help="Maximum translation length")
    parser.add_argument("--show_attention", action="store_true", help="Display attention weights")
    args = parser.parse_args()

    # Set device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load tokenizers
    src_tokenizer, tgt_tokenizer = load_tokenizers(args.src_lang, args.tgt_lang)
    src_vocab_size = src_tokenizer.get_vocab_size()
    tgt_vocab_size = tgt_tokenizer.get_vocab_size()

    # Create model
    encoder = Encoder(
        input_dim=src_vocab_size,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
    )

    attention = None
    if args.attention:
        attention = Attention(hidden_dim=args.hidden_dim)

    decoder = Decoder(
        output_dim=tgt_vocab_size,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
        attention=attention,
    )

    model = Seq2Seq(
        encoder=encoder,
        decoder=decoder,
        device=device,
        use_attention=args.attention,
    )

    # Load model weights
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Get sentence
    sentence = args.sentence
    if not sentence:
        print("Please enter a sentence to translate:")
        sentence = input("> ")

    # Translate
    translation, attention_weights = translate_sentence(
        model=model,
        sentence=sentence,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        device=device,
        max_len=args.max_len,
    )

    # Print results
    print(f"\nSource: {sentence}")
    print(f"Translation: {translation}")

    # Display attention if requested and available
    if args.show_attention and attention_weights is not None:
        import matplotlib.pyplot as plt
        import numpy as np

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot attention heatmap
        src_tokens = src_tokenizer.encode(sentence).tokens
        tgt_tokens = tgt_tokenizer.encode(translation).tokens

        # Remove special tokens
        src_tokens = [t for t in src_tokens if t not in ["[BOS]", "[EOS]", "[PAD]", "[UNK]"]]
        tgt_tokens = [t for t in tgt_tokens if t not in ["[BOS]", "[EOS]", "[PAD]", "[UNK]"]]

        # Trim attention weights to actual tokens
        attention_weights = attention_weights[:len(tgt_tokens), :len(src_tokens)]

        # Plot heatmap
        im = ax.imshow(attention_weights, cmap="viridis")

        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)

        # Set tick labels
        ax.set_xticks(np.arange(len(src_tokens)))
        ax.set_yticks(np.arange(len(tgt_tokens)))
        ax.set_xticklabels(src_tokens, rotation=45, ha="right")
        ax.set_yticklabels(tgt_tokens)

        # Add grid
        for edge, spine in ax.spines.items():
            spine.set_visible(False)
        ax.set_xticks(np.arange(attention_weights.shape[1] + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(attention_weights.shape[0] + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

        # Add title and labels
        plt.title("Attention Weights")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()