# Seq2Seq Assignment 4

A PyTorch implementation of sequence-to-sequence models for machine translation, supporting both basic and attention-based architectures.
## Description

The package implements two main model variants:
1. Basic Seq2Seq Model:
   - Encoder: Bidirectional GRU
   - Decoder: Unidirectional GRU
   - No attention mechanism
   - Simple architecture suitable for basic translation tasks
2. Attention-based Seq2Seq Model:
   - Encoder: Bidirectional GRU
   - Decoder: Unidirectional GRU with attention
   - Additive attention mechanism
   - Better performance on longer sequences
   - Visualizable attention weights
## File Structure

The package is structured as follows:

```
seq2seq/
├── core/          # Encoder, Decoder, Attention, Loss classes
├── data/          # Data loader and preprocessing. 
├── training/      # Training and Visualization
├── utils/         # Utility functions
├── scripts/       # Command-line interfaces also in pyproject.toml
└── visualization/ # Visualization utilities
```

## Installation
Create the virtual environment and install dependencies:
```bash
uv sync 
```
or 
With virtual env:
```bash
python -m venv .venv
source .venv/bin/activate # venv\Scripts\activate on windows
pip install -e .
```
## Prepare Tokenizers
```bash
# Create tokenizers for English to French
seq2seq-prepare --src_lang en --tgt_lang fr --vocab_size 10000
```
## Train 
Train with attention:
```bash
seq2seq-train --src_lang en --tgt_lang fr --batch_size 32 --hidden_dim 256 --attention
```
Train without attention:
```bash
seq2seq-train --src_lang en --tgt_lang fr --batch_size 32 --hidden_dim 256
```
## Translate
Translate with attention:
```bash
seq2seq-translate --src_lang en --tgt_lang fr --hidden_dim 256 --attention
```
Translate without attention:
```bash
seq2seq-translate --src_lang en --tgt_lang fr --hidden_dim 256
```
## Analyze model architecture 
```bash
seq2seq-inspect --model_path models/model_no_attn.pt
```

