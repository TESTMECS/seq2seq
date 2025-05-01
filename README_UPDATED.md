# Sequence-to-Sequence Translation Model

This repository contains an implementation of a sequence-to-sequence model for machine translation, with and without attention mechanism. The implementation uses PyTorch and includes comprehensive visualization tools.

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
- Attention weight visualization
- Model comparison tools

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

#### 2. Comprehensive Implementation with Visualizations

The most complete implementation with excellent visualizations:

```bash
uv run final_run.py
```

This runs both models (with and without attention) and generates detailed comparison visualizations.

#### 3. Custom Training and Visualization

Train and visualize models with custom parameters:

```bash
uv run train_and_visualize.py --compare --hidden-dim 128 --n-epochs 5 --num-samples 2000
```

#### 4. Simplified Implementation

If you want to use the pre-trained tokenizers with synthetic data:

```bash
uv run main_simplified.py
```

#### 5. Full Implementation

To use the full implementation with IWSLT dataset (requires datasets package):

```bash
uv run main.py --n_epochs 5 --batch_size 32
```

### Visualizing Results

To visualize attention weights:

```bash
uv run visualize_attention.py
```

To generate comprehensive model architecture and comparison visualizations:

```bash
uv run visualize_results.py
```

### Command-line Arguments for train_and_visualize.py

- `--src-vocab-size`: Source vocabulary size (default: 100)
- `--tgt-vocab-size`: Target vocabulary size (default: 100)
- `--hidden-dim`: Hidden dimension size (default: 64)
- `--emb-dim`: Embedding dimension size (default: 32)
- `--n-layers`: Number of RNN layers (default: 1)
- `--dropout`: Dropout rate (default: 0.2)
- `--batch-size`: Batch size (default: 32)
- `--n-epochs`: Number of epochs (default: 3)
- `--clip`: Gradient clipping value (default: 1.0)
- `--lr`: Learning rate (default: 0.001)
- `--patience`: Early stopping patience (default: 5)
- `--tf-ratio`: Teacher forcing ratio (default: 0.5)
- `--num-samples`: Number of synthetic samples (default: 2000)
- `--max-len`: Maximum sequence length (default: 10)
- `--seed`: Random seed (default: 42)
- `--no-attention`: Train model without attention
- `--attention`: Train model with attention
- `--compare`: Train and compare both models (default if no model type specified)
- `--save-dir`: Directory to save models and plots (default: "models")
- `--cpu`: Force using CPU even if GPU is available
- `--show-plots`: Show plots during training

## Project Structure

- `main.py`: Main script for training and evaluation (fixed version)
- `main_simplified.py`: Simplified version using the original structure
- `final_run.py`: Complete implementation with synthetic data
- `data.py`: Data loading and preprocessing
- `model.py`: Seq2Seq model architecture
- `train.py`: Training and evaluation functions
- `utils.py`: Utility functions
- `prepare_tokenizers.py`: Script to create tokenizers
- `train_and_visualize.py`: Interactive script for training and visualization
- `visualize_results.py`: Script for comprehensive visualizations
- `visualize_attention.py`: Script for attention weight visualization
- `tests/`: Unit and integration tests
- `RESULTS_ANALYSIS.md`: Detailed analysis of model results
- `FIXES.md`: Documentation of issues and fixes
- `SOLUTIONS.md`: Alternative solutions implemented
- `SUMMARY.md`: Summary of the project and findings

## Implementation Details

The implementation includes two models:
1. Basic Seq2Seq model without attention
2. Seq2Seq model with attention mechanism

Both models can be trained on the IWSLT 2017 dataset for English-French translation or on synthetic data for testing and demonstration purposes.

### Attention Mechanism

The attention mechanism allows the decoder to focus on different parts of the input sequence at each decoding step, which typically improves translation quality, especially for longer sentences. The visualization tools help understand how attention works in practice.

### Visualization Tools

The repository includes several visualization tools:
- Training and validation loss plots
- Model architecture visualization
- Attention weight visualizations
- Model comparison plots

These tools help understand the model's behavior and the impact of different architectural choices.

## Troubleshooting

If you encounter issues, please see the `FIXES.md` and `SOLUTIONS.md` files, which document common issues and their solutions.

The most common issues include:
- Dataset loading configuration format
- Missing imports
- Model initialization for embedding layers
- Handling None values in data processing

## License

MIT