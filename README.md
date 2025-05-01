# Sequence-to-Sequence Translation Model

This repository contains an implementation of a sequence-to-sequence model for machine translation, with and without attention mechanism. The implementation uses PyTorch.

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

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd seq2seq

# Install dependencies
uv add torch numpy tqdm matplotlib datasets tokenizers nltk ruff pytest
```

## Quick Start

### Preparing Tokenizers

```bash
python -m seq2seq prepare --src_lang en --tgt_lang fr --vocab_size 10000
```

### Training

```bash
python -m seq2seq train --n_epochs 10 --attention
```

### Translation

```bash
python -m seq2seq translate --model_path models/model_attn.pt --attention --sentence "Hello, how are you?"
```

## Documentation

For detailed documentation, please see:

- [User Guide](docs/USAGE.md): Detailed usage instructions
- [API Reference](docs/API.md): API documentation
- [Contributing Guide](docs/CONTRIBUTING.md): How to contribute to the project

## Package Structure

- `seq2seq/`: Main package
  - `core/`: Core model architecture
  - `data/`: Data loading and processing
  - `training/`: Training and evaluation
  - `utils/`: Utility functions
  - `scripts/`: Command-line interfaces
  - `visualization/`: Visualization utilities

## License

MIT