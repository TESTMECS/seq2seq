"""
Main entry point for the seq2seq package.
This allows running the package directly with python -m seq2seq
"""

import sys
import argparse


def main():
    """Main entry point for the seq2seq package."""
    parser = argparse.ArgumentParser(
        description="Sequence-to-sequence translation model CLI",
        prog="seq2seq",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a sequence-to-sequence model")
    train_parser.add_argument("--src_lang", type=str, default="en", help="Source language")
    train_parser.add_argument("--tgt_lang", type=str, default="fr", help="Target language")
    train_parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    train_parser.add_argument("--max_length", type=int, default=100, help="Maximum sequence length")
    train_parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension size")
    train_parser.add_argument("--emb_dim", type=int, default=128, help="Embedding dimension size")
    train_parser.add_argument(
        "--n_layers",
        type=int,
        default=2,
        help="Number of layers in encoder and decoder",
    )
    train_parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability")
    train_parser.add_argument("--n_epochs", type=int, default=10, help="Number of epochs")
    train_parser.add_argument("--clip", type=float, default=1.0, help="Gradient clipping value")
    train_parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    train_parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
    train_parser.add_argument("--tf_ratio", type=float, default=0.5, help="Teacher forcing ratio")
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    train_parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")
    train_parser.add_argument(
        "--save_dir", type=str, default="models", help="Directory to save models"
    )
    train_parser.add_argument(
        "--test_only", action="store_true", help="Only test the model (no training)"
    )
    train_parser.add_argument("--attention", action="store_true", help="Use attention mechanism")

    # Translate command
    translate_parser = subparsers.add_parser(
        "translate", help="Translate text using a trained model"
    )
    translate_parser.add_argument("--src_lang", type=str, default="en", help="Source language")
    translate_parser.add_argument("--tgt_lang", type=str, default="fr", help="Target language")
    translate_parser.add_argument(
        "--hidden_dim", type=int, default=256, help="Hidden dimension size"
    )
    translate_parser.add_argument(
        "--emb_dim", type=int, default=128, help="Embedding dimension size"
    )
    translate_parser.add_argument(
        "--n_layers",
        type=int,
        default=2,
        help="Number of layers in encoder and decoder",
    )
    translate_parser.add_argument("--dropout", type=float, default=0.2, help="Dropout probability")
    translate_parser.add_argument(
        "--device", type=str, default=None, help="Device to use (cuda/cpu)"
    )
    translate_parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    translate_parser.add_argument(
        "--attention", action="store_true", help="Use attention mechanism"
    )
    translate_parser.add_argument("--sentence", type=str, help="Sentence to translate")
    translate_parser.add_argument(
        "--max_len", type=int, default=100, help="Maximum translation length"
    )
    translate_parser.add_argument(
        "--show_attention", action="store_true", help="Display attention weights"
    )

    # Prepare command
    prepare_parser = subparsers.add_parser(
        "prepare", help="Prepare tokenizers for sequence-to-sequence models"
    )
    prepare_parser.add_argument("--src_lang", type=str, default="en", help="Source language")
    prepare_parser.add_argument("--tgt_lang", type=str, default="fr", help="Target language")
    prepare_parser.add_argument("--vocab_size", type=int, default=10000, help="Vocabulary size")
    prepare_parser.add_argument("--output_dir", type=str, default=".", help="Output directory")

    # Parse arguments
    args = parser.parse_args()

    if args.command == "train":
        from seq2seq.scripts.train_cli import main as train_main

        sys.argv = [sys.argv[0]] + [
            f"--{k}={v}" if not isinstance(v, bool) else f"--{k}" if v else ""
            for k, v in vars(args).items()
            if k != "command" and v is not None
        ]
        sys.argv = [arg for arg in sys.argv if arg]  # Remove empty args
        train_main()
    elif args.command == "translate":
        from seq2seq.scripts.translate_cli import main as translate_main

        sys.argv = [sys.argv[0]] + [
            f"--{k}={v}" if not isinstance(v, bool) else f"--{k}" if v else ""
            for k, v in vars(args).items()
            if k != "command" and v is not None
        ]
        sys.argv = [arg for arg in sys.argv if arg]  # Remove empty args
        translate_main()
    elif args.command == "prepare":
        from seq2seq.scripts.prepare_cli import main as prepare_main

        sys.argv = [sys.argv[0]] + [
            f"--{k}={v}" if not isinstance(v, bool) else f"--{k}" if v else ""
            for k, v in vars(args).items()
            if k != "command" and v is not None
        ]
        sys.argv = [arg for arg in sys.argv if arg]  # Remove empty args
        prepare_main()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
