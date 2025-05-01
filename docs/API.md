# API Reference

This document provides an overview of the main classes and functions in the seq2seq package.

## Core Module

The `seq2seq.core` module provides the neural network architecture components.

### Encoder

```python
class Encoder(nn.Module):
    """Encoder module for the sequence-to-sequence model."""
    
    def __init__(
        self,
        input_dim: int,
        emb_dim: int,
        hidden_dim: int,
        n_layers: int,
        dropout: float,
    ):
        # Initialize encoder with embedding, bidirectional GRU, and output projection
        
    def forward(
        self, src: torch.Tensor, src_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Forward pass through the encoder
        # Returns encoder outputs and final hidden state
```

### Attention

```python
class Attention(nn.Module):
    """Attention mechanism for the sequence-to-sequence model."""
    
    def __init__(self, hidden_dim: int):
        # Initialize attention mechanism
        
    def forward(
        self,
        hidden: torch.Tensor,
        encoder_outputs: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        # Calculate attention weights
        # Returns attention weights
```

### Decoder

```python
class Decoder(nn.Module):
    """Decoder module for the sequence-to-sequence model."""
    
    def __init__(
        self,
        output_dim: int,
        emb_dim: int,
        hidden_dim: int,
        n_layers: int,
        dropout: float,
        attention: Optional[Attention] = None,
    ):
        # Initialize decoder with embedding, GRU, and output projection
        
    def forward(
        self,
        input: torch.Tensor,
        hidden: torch.Tensor,
        encoder_outputs: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Forward pass through the decoder for a single time step
        # Returns prediction, updated hidden state, and attention weights
```

### Seq2Seq

```python
class Seq2Seq(nn.Module):
    """Sequence-to-sequence model for machine translation."""
    
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        device: torch.device,
        use_attention: bool = False,
    ):
        # Initialize sequence-to-sequence model
        
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_lengths: torch.Tensor,
        src_mask: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        # Forward pass through the sequence-to-sequence model
        # Returns dictionary with outputs and attention weights
        
    def translate(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
        src_mask: torch.Tensor,
        max_len: int,
        bos_idx: int,
        eos_idx: int,
    ) -> Dict[str, torch.Tensor]:
        # Translate a source sequence to target language
        # Returns dictionary with translations and attention weights
```

### Seq2SeqLoss

```python
class Seq2SeqLoss(nn.Module):
    """Loss function for sequence-to-sequence model."""
    
    def __init__(self, pad_idx: int):
        # Initialize loss function with padding index
        
    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Calculate loss
        # Returns loss value
```

## Data Module

The `seq2seq.data` module provides data loading and processing functionality.

### TranslationDataset

```python
class TranslationDataset(Dataset):
    """Dataset for sequence-to-sequence translation."""
    
    def __init__(self, src_texts: List[List[int]], tgt_texts: List[List[int]]):
        # Initialize dataset with tokenized texts
        
    def __len__(self) -> int:
        # Return number of samples
        
    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        # Return a sample from the dataset
```

### BatchSampler

```python
class BatchSampler:
    """Custom batch sampler for sequence-to-sequence model training."""
    
    def __init__(
        self,
        dataset_length: int,
        lens: List[int],
        batch_size: int,
        shuffle: bool = True,
    ):
        # Initialize batch sampler
        
    def __iter__(self) -> Iterator[List[int]]:
        # Generate batches of indices
        
    def __len__(self) -> int:
        # Return number of batches
```

### Data Loading Functions

```python
def collate_fn(batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader."""
    # Pad sequences and create masks
    
def load_tokenizers(src_lang: str, tgt_lang: str) -> Tuple[Tokenizer, Tokenizer]:
    """Load tokenizers for source and target languages."""
    # Load tokenizers from files
    
def load_iwslt_data(
    src_lang: str, tgt_lang: str, split: str
) -> Tuple[List[str], List[str]]:
    """Load IWSLT dataset for specified languages and split."""
    # Load and return texts
    
def prepare_data(
    src_lang: str,
    tgt_lang: str,
    batch_size: int,
    max_length: int,
    data_cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Prepare data for training and evaluation."""
    # Load, tokenize, and batch data
    # Return dataloaders and tokenizer information
```

## Training Module

The `seq2seq.training` module provides training and evaluation functionality.

### Training Functions

```python
def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    clip: float,
    device: torch.device,
    teacher_forcing_ratio: float = 0.5,
) -> float:
    """Train the model for one epoch."""
    # Train for one epoch
    # Return average loss
    
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Evaluate the model on validation or test data."""
    # Evaluate model
    # Return average loss
    
def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    clip: float,
    device: torch.device,
    n_epochs: int,
    patience: int = 5,
    teacher_forcing_ratio: float = 0.5,
    model_path: str = "model.pt",
) -> Dict[str, List[float]]:
    """Train the model for multiple epochs."""
    # Train for multiple epochs with early stopping
    # Return history with training and validation losses
```

### Evaluation Functions

```python
def test_model(
    model: nn.Module,
    test_dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    src_tokenizer: Any,
    tgt_tokenizer: Any,
    num_examples: int = 20,
) -> Tuple[float, float]:
    """Test the model on test data and compute BLEU score."""
    # Test model and calculate BLEU score
    # Return test loss and BLEU score
```

### Visualization Functions

```python
def plot_losses(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    show_plot: bool = False,
) -> None:
    """Plot training and validation losses."""
    # Create and save plot
```

## Utils Module

The `seq2seq.utils` module provides utility functions.

### Metrics

```python
def bleu_score(references: List[str], hypotheses: List[str]) -> float:
    """Compute BLEU score for translated sentences."""
    # Calculate BLEU score
```

### Translation Functions

```python
def translate_sentence(
    model: nn.Module,
    sentence: str,
    src_tokenizer: Any,
    tgt_tokenizer: Any,
    device: torch.device,
    max_len: int = 50,
) -> Tuple[str, Optional[np.ndarray]]:
    """Translate a sentence using the model."""
    # Translate sentence
    # Return translation and attention weights
    
def compare_translations(
    source_sentences: List[str],
    reference_translations: List[str],
    model_translations: Dict[str, List[str]],
) -> None:
    """Compare translations from different models."""
    # Print comparison table
```

### Tokenizer Functions

```python
def create_tokenizer_json(texts: List[str], vocab_size: int, output_path: str) -> None:
    """Create a tokenizer and save it to a JSON file."""
    # Create and save tokenizer
```