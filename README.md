# Sequence-to-Sequence Translation Model

This repository contains an implementation of a sequence-to-sequence model for machine translation, with and without attention mechanism. The implementation follows the steps outlined in the assignment instructions and uses PyTorch.

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

## Setup

1. Clone this repository
2. Install dependencies:
   ```
   uv add torch numpy tqdm matplotlib datasets tokenizers nltk ruff pytest
   ```
3. Create tokenizers:
   ```
   uv run prepare_tokenizers.py --src_lang en --tgt_lang fr --vocab_size 10000
   ```

## Usage

### Training

Train the models with default parameters:

```
uv run main.py
```

Custom training settings:

```
uv run main.py --batch_size 64 --n_epochs 20 --hidden_dim 512 --emb_dim 256
```

### Testing

Test pre-trained models:

```
uv run main.py --test_only
```

### Running Tests

The project includes comprehensive unit and integration tests. To run the tests:

```
# Run all tests
uv run run_tests.py --all

# Run only unit tests
uv run run_tests.py --unit

# Run only integration tests
uv run run_tests.py --integration

# Add verbose output
uv run run_tests.py --verbose
```

Alternatively, you can use pytest directly:

```
uv run -m pytest tests/
```

### Code Formatting

Format code with ruff:

```
uv run -m ruff format .
```

### Command-line Arguments

- `--src_lang`: Source language (default: "en")
- `--tgt_lang`: Target language (default: "fr")
- `--batch_size`: Batch size (default: 32)
- `--max_length`: Maximum sequence length (default: 100)
- `--hidden_dim`: Hidden dimension size (default: 256)
- `--emb_dim`: Embedding dimension size (default: 128)
- `--n_layers`: Number of layers in encoder and decoder (default: 2)
- `--dropout`: Dropout probability (default: 0.2)
- `--n_epochs`: Number of epochs (default: 10)
- `--clip`: Gradient clipping value (default: 1.0)
- `--lr`: Learning rate (default: 0.001)
- `--patience`: Patience for early stopping (default: 5)
- `--tf_ratio`: Teacher forcing ratio (default: 0.5)
- `--seed`: Random seed (default: 42)
- `--device`: Device to use (default: "cuda" if available, otherwise "cpu")
- `--save_dir`: Directory to save models (default: "models")
- `--test_only`: Only test the model (no training)

## Project Structure

- `main.py`: Main script for training and evaluation
- `data.py`: Data loading and preprocessing
- `model.py`: Seq2Seq model architecture
- `train.py`: Training and evaluation functions
- `utils.py`: Utility functions
- `prepare_tokenizers.py`: Script to create tokenizers
- `tests/`: Unit and integration tests
- `pytest.ini`: Configuration for pytest
- `run_tests.py`: Script to run tests

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