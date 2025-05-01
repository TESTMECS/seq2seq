# Usage Guide

This guide provides detailed instructions on how to use the seq2seq package for machine translation.

## Installation

Install the package using pip:

```bash
pip install seq2seq
```

Or from source:

```bash
git clone <repository-url>
cd seq2seq
pip install -e .
```

## Preparing Tokenizers

Before training models, you need to create tokenizers for the source and target languages:

```bash
# Using the command-line interface
seq2seq-prepare --src_lang en --tgt_lang fr --vocab_size 10000

# Using the Python module
python -m seq2seq prepare --src_lang en --tgt_lang fr --vocab_size 10000
```

This creates two tokenizer files in the current directory: `en_tokenizer.json` and `fr_tokenizer.json`.

## Training Models

### Basic Training

Train a basic sequence-to-sequence model:

```bash
# Using the command-line interface
seq2seq-train --n_epochs 10 --batch_size 32 --hidden_dim 256 --emb_dim 128

# Using the Python module
python -m seq2seq train --n_epochs 10 --batch_size 32 --hidden_dim 256 --emb_dim 128
```

### Training with Attention

Add the `--attention` flag to train a model with attention:

```bash
seq2seq-train --n_epochs 10 --attention
```

### Available Training Options

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
- `--attention`: Use attention mechanism

### Testing Models

Test a trained model without retraining:

```bash
seq2seq-train --test_only --save_dir models
```

This loads the model from the specified directory and evaluates it on the test set.

## Translating Text

### Basic Translation

Translate a sentence using a trained model:

```bash
# Using the command-line interface
seq2seq-translate --model_path models/model_attn.pt --attention --sentence "Hello, how are you?"

# Using the Python module
python -m seq2seq translate --model_path models/model_attn.pt --attention --sentence "Hello, how are you?"
```

### Interactive Translation

If you don't provide a sentence, the program enters interactive mode:

```bash
seq2seq-translate --model_path models/model_attn.pt --attention
```

### Visualizing Attention

To visualize attention weights, add the `--show_attention` flag:

```bash
seq2seq-translate --model_path models/model_attn.pt --attention --sentence "Hello, how are you?" --show_attention
```

This displays a heatmap of attention weights, showing which source words the model focused on when generating each target word.

### Available Translation Options

- `--src_lang`: Source language (default: "en")
- `--tgt_lang`: Target language (default: "fr")
- `--hidden_dim`: Hidden dimension size (default: 256)
- `--emb_dim`: Embedding dimension size (default: 128)
- `--n_layers`: Number of layers in encoder and decoder (default: 2)
- `--dropout`: Dropout probability (default: 0.2)
- `--device`: Device to use (default: "cuda" if available, otherwise "cpu")
- `--model_path`: Path to the model (required)
- `--attention`: Use attention mechanism
- `--sentence`: Sentence to translate
- `--max_len`: Maximum translation length (default: 100)
- `--show_attention`: Display attention weights

## Using the Python API

### Loading Data

```python
from seq2seq.data import prepare_data

data_info = prepare_data(
    src_lang="en",
    tgt_lang="fr",
    batch_size=32,
    max_length=100,
)
```

### Creating a Model

```python
import torch
from seq2seq.core import Encoder, Decoder, Attention, Seq2Seq

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create encoder
encoder = Encoder(
    input_dim=data_info["src_vocab_size"],
    emb_dim=128,
    hidden_dim=256,
    n_layers=2,
    dropout=0.2,
)

# Create attention
attention = Attention(hidden_dim=256)

# Create decoder
decoder = Decoder(
    output_dim=data_info["tgt_vocab_size"],
    emb_dim=128,
    hidden_dim=256,
    n_layers=2,
    dropout=0.2,
    attention=attention,
)

# Create model
model = Seq2Seq(
    encoder=encoder,
    decoder=decoder,
    device=device,
    use_attention=True,
)
model = model.to(device)
```

### Training a Model

```python
import torch.optim as optim
from seq2seq.core import Seq2SeqLoss
from seq2seq.training import train

# Create loss function
criterion = Seq2SeqLoss(pad_idx=data_info["pad_idx"])

# Create optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train model
history = train(
    model=model,
    train_dataloader=data_info["train_dataloader"],
    val_dataloader=data_info["val_dataloader"],
    optimizer=optimizer,
    criterion=criterion,
    clip=1.0,
    device=device,
    n_epochs=10,
    patience=5,
    teacher_forcing_ratio=0.5,
    model_path="models/model_attn.pt",
)

# Plot losses
from seq2seq.training import plot_losses
plot_losses(history, save_path="models/losses.png", show_plot=True)
```

### Translating Text

```python
from seq2seq.utils import translate_sentence

# Load tokenizers
from seq2seq.data import load_tokenizers
src_tokenizer, tgt_tokenizer = load_tokenizers("en", "fr")

# Translate a sentence
translation, attention_weights = translate_sentence(
    model=model,
    sentence="Hello, how are you?",
    src_tokenizer=src_tokenizer,
    tgt_tokenizer=tgt_tokenizer,
    device=device,
    max_len=50,
)

print(f"Translation: {translation}")
```

### Comparing Models

```python
from seq2seq.utils import compare_translations

# List of source sentences
source_sentences = [
    "Hello, how are you?",
    "I love programming.",
    "Machine learning is fascinating.",
]

# Reference translations
reference_translations = [
    "Bonjour, comment ça va ?",
    "J'adore la programmation.",
    "L'apprentissage automatique est fascinant.",
]

# Model translations
model_translations = {
    "No Attention": ["Translation 1", "Translation 2", "Translation 3"],
    "With Attention": ["Translation 1", "Translation 2", "Translation 3"],
}

# Compare translations
compare_translations(source_sentences, reference_translations, model_translations)
```