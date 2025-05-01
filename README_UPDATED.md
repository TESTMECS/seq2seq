# Sequence-to-Sequence Translation Model

This repository contains an implementation of a sequence-to-sequence model for machine translation, with and without attention mechanism.

## Features

- Data loading and preprocessing for IWSLT 2017 dataset (English-French by default)
- Custom batch sampler for efficient training
- Seq2Seq model with:
  - Encoder (bidirectional GRU)
  - Decoder (GRU)
  - Optional attention mechanism
- Training and evaluation utilities
- BLEU score calculation
- Visualization of training progress

## Getting Started

### Requirements

- Python 3.12 or higher
- PyTorch 2.0 or higher
- Other dependencies listed in `pyproject.toml`

### Installation

Install dependencies:
```
uv add torch numpy tqdm matplotlib datasets tokenizers nltk ruff pytest
```

### Running the Code

We provide several implementations with varying complexity:

#### 1. Minimal Example (Recommended for Understanding)

For a minimal working example focusing on the model architecture:

```bash
uv run run_minimal.py
```

This version uses a completely synthetic dataset and simplified training loop.

#### 2. Hybrid Implementation (Recommended for Testing)

A balanced approach that creates synthetic data but uses the original model structure:

```bash
uv run main_hybrid.py
```

#### 3. Simplified Implementation

If you want to use the pre-trained tokenizers with synthetic data:

```bash
uv run main_simplified.py
```

#### 4. Full Implementation

To use the full implementation with IWSLT dataset (requires datasets package):

```bash
uv run main.py --n_epochs 5 --batch_size 32
```

Custom training settings:

```bash
uv run main.py --batch_size 64 --hidden_dim 512 --emb_dim 256
```

### Command-line Arguments

Main training parameters:
- `--batch_size`: Batch size (default: 32)
- `--hidden_dim`: Hidden dimension size (default: 256)
- `--emb_dim`: Embedding dimension size (default: 128)
- `--n_epochs`: Number of epochs (default: 10)
- `--lr`: Learning rate (default: 0.001)

For a full list of parameters, see the help:
```
uv run main.py --help
```

## Project Structure

- `main.py`: Main script for training and evaluation
- `main_simplified.py`: Simplified version using synthetic data
- `run_minimal.py`: Minimal working example
- `main_hybrid.py`: Hybrid implementation
- `data.py`: Data loading and preprocessing
- `model.py`: Seq2Seq model architecture
- `train.py`: Training and evaluation functions
- `utils.py`: Utility functions
- `prepare_tokenizers.py`: Script to create tokenizers

## Implementation Details

The implementation includes two models:
1. Basic Seq2Seq model without attention
2. Seq2Seq model with attention mechanism

Both models are trained and evaluated on the same data, allowing for direct comparison.

### Attention Mechanism

The attention mechanism allows the decoder to focus on different parts of the input sequence at each decoding step, typically improving translation quality for longer sentences.

## Troubleshooting

If you encounter issues, please see the `FIXES.md` and `SOLUTIONS.md` files, which document common issues and their solutions.

The most common issues include:
- Dataset loading configuration format
- Missing imports
- Model initialization for embedding layers
- Handling None values in data processing

## License

MIT