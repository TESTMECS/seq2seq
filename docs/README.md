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

## Requirements

- Python 3.12 or higher
- PyTorch 2.0 or higher
- Other dependencies listed in `pyproject.toml`

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd seq2seq

# Install dependencies
uv add torch numpy tqdm matplotlib datasets tokenizers nltk ruff pytest
```

## Usage

### Preparing Tokenizers

Before training, you need to create tokenizers:

```bash
seq2seq-prepare --src_lang en --tgt_lang fr --vocab_size 10000
```

Or using the Python module:

```bash
python -m seq2seq prepare --src_lang en --tgt_lang fr --vocab_size 10000
```

### Training

Train the models with default parameters:

```bash
seq2seq-train
```

Custom training settings:

```bash
seq2seq-train --batch_size 64 --n_epochs 20 --hidden_dim 512 --emb_dim 256 --attention
```

### Translation

Translate using a trained model:

```bash
seq2seq-translate --model_path models/model_attn.pt --attention --sentence "Hello, how are you?"
```

### Package Structure

- `seq2seq/`: Main package
  - `core/`: Core model architecture
  - `data/`: Data loading and processing
  - `training/`: Training and evaluation
  - `utils/`: Utility functions
  - `scripts/`: Command-line interfaces
  - `visualization/`: Visualization utilities

## Implementation Details

The implementation includes two models:
1. Basic Seq2Seq model without attention
2. Seq2Seq model with attention mechanism

Both models are trained and evaluated on the IWSLT 2017 dataset for English-French translation by default, but can be configured for other language pairs.

### Custom Batch Sampler

The custom batch sampler groups sentences with similar lengths to minimize padding and improve training efficiency. This is particularly important for sequence-to-sequence models, as it reduces computational waste.

### Attention Mechanism

The attention mechanism allows the decoder to focus on different parts of the input sequence at each decoding step, which typically improves translation quality, especially for longer sentences.

## License

MIT