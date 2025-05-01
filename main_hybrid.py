"""
Hybrid approach for sequence-to-sequence models.
This combines features from main.py, main_simplified.py, and run_minimal.py
to create a simple working example that still preserves the original code structure.
"""

import torch
import torch.optim as optim
import torch.nn as nn
import os
import random
import numpy as np
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from model import Encoder, Decoder, Attention, Seq2Seq, Seq2SeqLoss
from train import train, evaluate, plot_losses
from utils import translate_sentence


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SimpleDataset(Dataset):
    """Simple dataset for sequence-to-sequence task."""

    def __init__(self, sources, targets):
        """Initialize dataset with tokenized source and target sentences."""
        self.sources = sources
        self.targets = targets
        assert len(sources) == len(targets), "Source and target lists must have the same length"

    def __len__(self) -> int:
        """Return the number of sentence pairs."""
        return len(self.sources)

    def __getitem__(self, idx: int):
        """Return a single source-target pair."""
        return self.sources[idx], self.targets[idx]


def collate_batch(batch):
    """Custom collate function for batches."""
    sources, targets = zip(*batch)
    
    # Convert to tensors and pad
    src_tensors = [torch.tensor(s, dtype=torch.long) for s in sources]
    tgt_tensors = [torch.tensor(t, dtype=torch.long) for t in targets]
    
    # Padding
    src_padded = pad_sequence(src_tensors, batch_first=True, padding_value=0)
    tgt_padded = pad_sequence(tgt_tensors, batch_first=True, padding_value=0)
    
    # Create source and target masks (1 for non-pad, 0 for pad)
    src_mask = (src_padded != 0).float()
    tgt_mask = (tgt_padded != 0).float()
    
    return {
        "src": src_padded,
        "tgt": tgt_padded,
        "src_mask": src_mask,
        "tgt_mask": tgt_mask,
        "src_lengths": torch.tensor([len(s) for s in sources], dtype=torch.long),
        "tgt_lengths": torch.tensor([len(t) for t in targets], dtype=torch.long),
    }


def create_model(
    src_vocab_size: int,
    tgt_vocab_size: int,
    device: torch.device,
    use_attention: bool = False,
    hidden_dim: int = 256,
    emb_dim: int = 128,
    n_layers: int = 2,
    dropout: float = 0.2,
) -> Seq2Seq:
    """
    Create a sequence-to-sequence model.
    """
    # Create encoder
    encoder = Encoder(
        input_dim=src_vocab_size,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=dropout,
    )

    # Create attention mechanism if needed
    attention = None
    if use_attention:
        attention = Attention(hidden_dim=hidden_dim)

    # Create decoder
    decoder = Decoder(
        output_dim=tgt_vocab_size,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=dropout,
        attention=attention,
    )

    # Create sequence-to-sequence model
    model = Seq2Seq(
        encoder=encoder,
        decoder=decoder,
        device=device,
        use_attention=use_attention,
    )

    # Initialize parameters
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.xavier_uniform_(m.weight)

    model.apply(init_weights)

    # Move model to device
    model = model.to(device)

    return model


def create_toy_dataset(size=100, src_len=10, tgt_len=12, src_vocab_size=1000, tgt_vocab_size=1000):
    """Create a toy dataset for testing."""
    # Create random source and target sequences
    src_data = []
    tgt_data = []
    
    for _ in range(size):
        # Create random sequences
        src_tokens = [random.randint(1, src_vocab_size-1) for _ in range(random.randint(3, src_len))]
        # Add special tokens for target (BOS=1 and EOS=2)
        tgt_tokens = [1] + [random.randint(3, tgt_vocab_size-1) for _ in range(random.randint(3, tgt_len))] + [2]
        
        src_data.append(src_tokens)
        tgt_data.append(tgt_tokens)
    
    return src_data, tgt_data


def main():
    """Main function for training and evaluating models."""
    # Configuration
    batch_size = 32
    hidden_dim = 64
    emb_dim = 32
    n_layers = 1
    dropout = 0.1
    n_epochs = 2
    clip = 1.0
    lr = 0.001
    tf_ratio = 0.5
    seed = 42
    device = torch.device("cpu")  # Use CPU for reproducibility
    save_dir = "models"
    
    # Set random seed for reproducibility
    set_seed(seed)

    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    print(f"Using device: {device}")

    # Create toy dataset
    print("Creating toy dataset...")
    src_data, tgt_data = create_toy_dataset(
        size=500,
        src_len=8,
        tgt_len=10,
        src_vocab_size=1000,
        tgt_vocab_size=1000,
    )
    
    # Split into train, val, test
    train_size = int(0.7 * len(src_data))
    val_size = int(0.15 * len(src_data))
    test_size = len(src_data) - train_size - val_size
    
    train_src = src_data[:train_size]
    train_tgt = tgt_data[:train_size]
    
    val_src = src_data[train_size:train_size+val_size]
    val_tgt = tgt_data[train_size:train_size+val_size]
    
    test_src = src_data[train_size+val_size:]
    test_tgt = tgt_data[train_size+val_size:]
    
    # Create datasets
    train_dataset = SimpleDataset(train_src, train_tgt)
    val_dataset = SimpleDataset(val_src, val_tgt)
    test_dataset = SimpleDataset(test_src, test_tgt)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_batch
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_batch
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_batch
    )
    
    # Get vocabulary sizes
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    
    # Create dummy tokenizers for translation examples
    # In this simplified version, we'll just create minimal tokenizers with encode/decode functions
    class DummyTokenizer:
        def __init__(self, vocab_size):
            self.vocab_size = vocab_size
            self.special_tokens = {
                "[PAD]": 0,
                "[BOS]": 1,
                "[EOS]": 2,
                "[UNK]": 3
            }
        
        def encode(self, text):
            # Create a simple hash-based encoding
            tokens = text.split()
            ids = []
            for token in tokens:
                # Simple hash function to get consistent IDs
                token_id = hash(token) % (self.vocab_size - 10) + 10  # Start after special tokens
                ids.append(token_id)
            
            class EncodedOutput:
                def __init__(self, ids):
                    self.ids = ids
            
            return EncodedOutput(ids)
        
        def decode(self, ids):
            # For simplicity, just return a placeholder text
            return " ".join([f"token_{id}" for id in ids if id > 3])  # Skip special tokens
        
        def token_to_id(self, token):
            return self.special_tokens.get(token, 3)  # Default to UNK
        
        def get_vocab_size(self):
            return self.vocab_size
    
    src_tokenizer = DummyTokenizer(src_vocab_size)
    tgt_tokenizer = DummyTokenizer(tgt_vocab_size)

    print(f"Source vocabulary size: {src_vocab_size}")
    print(f"Target vocabulary size: {tgt_vocab_size}")
    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    # Define model paths
    model_no_attn_path = os.path.join(save_dir, "model_no_attn.pt")
    model_attn_path = os.path.join(save_dir, "model_attn.pt")

    # Define pad index for loss function
    pad_idx = 0  # We use 0 as padding index

    # Create loss function
    criterion = Seq2SeqLoss(pad_idx=pad_idx)

    # Create models
    print("Creating models...")

    # Model without attention
    model_no_attn = create_model(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        device=device,
        use_attention=False,
        hidden_dim=hidden_dim,
        emb_dim=emb_dim,
        n_layers=n_layers,
        dropout=dropout,
    )
    print(
        f"Model without attention has {sum(p.numel() for p in model_no_attn.parameters() if p.requires_grad):,} trainable parameters"
    )

    # Model with attention
    model_attn = create_model(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        device=device,
        use_attention=True,
        hidden_dim=hidden_dim,
        emb_dim=emb_dim,
        n_layers=n_layers,
        dropout=dropout,
    )
    print(
        f"Model with attention has {sum(p.numel() for p in model_attn.parameters() if p.requires_grad):,} trainable parameters"
    )

    # Train models
    print("\n" + "=" * 50)
    print("Training model without attention...")
    print("=" * 50 + "\n")

    # Create optimizer for model without attention
    optimizer_no_attn = optim.Adam(model_no_attn.parameters(), lr=lr)

    # Train model without attention
    # Create our own simplified training loop
    history_no_attn = {"train_loss": [], "val_loss": []}
    
    for epoch in range(n_epochs):
        # Training
        model_no_attn.train()
        total_train_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            src_mask = batch["src_mask"].to(device)
            src_lengths = batch["src_lengths"].to(device)
            
            # Zero gradients
            optimizer_no_attn.zero_grad()
            
            # Forward pass
            output_dict = model_no_attn(
                src=src,
                tgt=tgt,
                src_lengths=src_lengths,
                src_mask=src_mask,
                teacher_forcing_ratio=tf_ratio,
            )
            
            # Calculate loss
            output = output_dict["outputs"]
            loss = criterion(output, tgt)
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model_no_attn.parameters(), clip)
            
            # Update parameters
            optimizer_no_attn.step()
            
            # Update total loss
            total_train_loss += loss.item()
            
            # Print progress
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{n_epochs} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        # Calculate average training loss
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation
        model_no_attn.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                src = batch["src"].to(device)
                tgt = batch["tgt"].to(device)
                src_mask = batch["src_mask"].to(device)
                src_lengths = batch["src_lengths"].to(device)
                
                # Forward pass
                output_dict = model_no_attn(
                    src=src,
                    tgt=tgt,
                    src_lengths=src_lengths,
                    src_mask=src_mask,
                    teacher_forcing_ratio=0.0,  # No teacher forcing during validation
                )
                
                # Calculate loss
                output = output_dict["outputs"]
                loss = criterion(output, tgt)
                
                # Update total loss
                total_val_loss += loss.item()
        
        # Calculate average validation loss
        avg_val_loss = total_val_loss / len(val_loader)
        
        # Print progress
        print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Store losses
        history_no_attn["train_loss"].append(avg_train_loss)
        history_no_attn["val_loss"].append(avg_val_loss)
    
    # Plot losses for model without attention
    plot_losses(
        history=history_no_attn,
        save_path=os.path.join(save_dir, "losses_no_attn.png"),
    )

    print("\n" + "=" * 50)
    print("Training model with attention...")
    print("=" * 50 + "\n")

    # Create optimizer for model with attention
    optimizer_attn = optim.Adam(model_attn.parameters(), lr=lr)

    # Train model with attention
    # Create our own simplified training loop
    history_attn = {"train_loss": [], "val_loss": []}
    
    for epoch in range(n_epochs):
        # Training
        model_attn.train()
        total_train_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            src_mask = batch["src_mask"].to(device)
            src_lengths = batch["src_lengths"].to(device)
            
            # Zero gradients
            optimizer_attn.zero_grad()
            
            # Forward pass
            output_dict = model_attn(
                src=src,
                tgt=tgt,
                src_lengths=src_lengths,
                src_mask=src_mask,
                teacher_forcing_ratio=tf_ratio,
            )
            
            # Calculate loss
            output = output_dict["outputs"]
            loss = criterion(output, tgt)
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model_attn.parameters(), clip)
            
            # Update parameters
            optimizer_attn.step()
            
            # Update total loss
            total_train_loss += loss.item()
            
            # Print progress
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{n_epochs} | Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        # Calculate average training loss
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation
        model_attn.eval()
        total_val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                src = batch["src"].to(device)
                tgt = batch["tgt"].to(device)
                src_mask = batch["src_mask"].to(device)
                src_lengths = batch["src_lengths"].to(device)
                
                # Forward pass
                output_dict = model_attn(
                    src=src,
                    tgt=tgt,
                    src_lengths=src_lengths,
                    src_mask=src_mask,
                    teacher_forcing_ratio=0.0,  # No teacher forcing during validation
                )
                
                # Calculate loss
                output = output_dict["outputs"]
                loss = criterion(output, tgt)
                
                # Update total loss
                total_val_loss += loss.item()
        
        # Calculate average validation loss
        avg_val_loss = total_val_loss / len(val_loader)
        
        # Print progress
        print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Store losses
        history_attn["train_loss"].append(avg_train_loss)
        history_attn["val_loss"].append(avg_val_loss)
    
    # Plot losses for model with attention
    plot_losses(
        history=history_attn,
        save_path=os.path.join(save_dir, "losses_attn.png"),
    )

    print("\nTraining complete!")
    print("\nFinal training loss:")
    print(f"- Without attention: {history_no_attn['train_loss'][-1]:.4f}")
    print(f"- With attention: {history_attn['train_loss'][-1]:.4f}")
    
    print("\nFinal validation loss:")
    print(f"- Without attention: {history_no_attn['val_loss'][-1]:.4f}")
    print(f"- With attention: {history_attn['val_loss'][-1]:.4f}")
    
    # Save models
    torch.save(model_no_attn.state_dict(), model_no_attn_path)
    torch.save(model_attn.state_dict(), model_attn_path)
    
    # Example translations
    print("\n" + "=" * 50)
    print("Example translations...")
    print("=" * 50 + "\n")
    
    # Some example sentences
    example_sentences = [
        "Hello, how are you?",
        "I love programming in Python.",
        "The weather is nice today.",
        "What time is it?",
        "Can you help me with this translation?",
    ]
    
    # Translate examples
    print(f"{'Source':<30} | {'No Attention':<30} | {'With Attention':<30}")
    print(f"{'-' * 30}+{'-' * 31}+{'-' * 31}")
    
    for sentence in example_sentences:
        # Translate with model without attention
        translation_no_attn, _ = translate_sentence(
            model=model_no_attn,
            sentence=sentence,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            device=device,
        )
        
        # Translate with model with attention
        translation_attn, _ = translate_sentence(
            model=model_attn,
            sentence=sentence,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            device=device,
        )
        
        # Print results
        print(f"{sentence[:28]:<30} | {translation_no_attn[:28]:<30} | {translation_attn[:28]:<30}")


if __name__ == "__main__":
    main()