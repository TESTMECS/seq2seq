#!/usr/bin/env python
"""
Run script for sequence-to-sequence translation model.
This script sets up the necessary directories and runs the model with the dataset.
"""

import os
import subprocess
import argparse

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run sequence-to-sequence model training")
    parser.add_argument("--n_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension size")
    parser.add_argument("--emb_dim", type=int, default=64, help="Embedding dimension size")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage")
    args = parser.parse_args()

    # Create directories for dataset cache
    cache_dir = os.path.join('a4-data', 'dataset')
    os.makedirs(cache_dir, exist_ok=True)

    # Build the command
    cmd = [
        "uv", "run", "main.py",
        "--n_epochs", str(args.n_epochs),
        "--batch_size", str(args.batch_size),
        "--hidden_dim", str(args.hidden_dim),
        "--emb_dim", str(args.emb_dim),
        "--cache_dir", cache_dir,
        "--show_plots",  # Always show plots
    ]
    
    # Force CPU if requested
    if args.cpu:
        cmd.extend(["--device", "cpu"])
    
    # Run the model training
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd)

if __name__ == "__main__":
    main()