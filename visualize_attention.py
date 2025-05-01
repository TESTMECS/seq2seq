#!/usr/bin/env python
"""
Visualize attention weights for sequence-to-sequence model translation.
"""

import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.colors import LinearSegmentedColormap

from model import Encoder, Decoder, Attention, Seq2Seq


def load_model(model_path, src_vocab_size=100, tgt_vocab_size=100, device=torch.device("cpu")):
    """Load a trained model from a checkpoint."""
    # Create attention mechanism
    attention = Attention(hidden_dim=64)
    
    # Create encoder
    encoder = Encoder(
        input_dim=src_vocab_size,
        emb_dim=32,
        hidden_dim=64,
        n_layers=1,
        dropout=0.2,
    )
    
    # Create decoder with attention
    decoder = Decoder(
        output_dim=tgt_vocab_size,
        emb_dim=32,
        hidden_dim=64,
        n_layers=1,
        dropout=0.2,
        attention=attention,
    )
    
    # Create sequence-to-sequence model
    model = Seq2Seq(
        encoder=encoder,
        decoder=decoder,
        device=device,
        use_attention=True,
    )
    
    # Load state dict
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    
    return model


def create_sample_data(seq_len=10, vocab_size=100):
    """Create synthetic data for attention visualization."""
    # Create source and target sequences
    src_seq = [random.randint(1, vocab_size-1) for _ in range(seq_len)]
    tgt_seq = [1] + [random.randint(3, vocab_size-1) for _ in range(seq_len)] + [2]  # BOS(1) and EOS(2)
    
    # Convert to tensors
    src_tensor = torch.tensor([src_seq], dtype=torch.long)
    src_mask = torch.ones_like(src_tensor, dtype=torch.float)
    src_lengths = torch.tensor([len(src_seq)], dtype=torch.long)
    
    return src_tensor, src_mask, src_lengths, src_seq, tgt_seq


def generate_attention_matrix(model, src_tensor, src_mask, src_lengths, max_len=12):
    """Generate attention weights for a sample input sequence."""
    model.eval()
    with torch.no_grad():
        # Encode the source sequence
        encoder_outputs, hidden = model.encoder(src_tensor, src_lengths)
        
        # First input to the decoder is the <BOS> token
        bos_idx = 1
        decoder_input = torch.ones(src_tensor.size(0), dtype=torch.long).fill_(bos_idx).to(src_tensor.device)
        
        # Initialize attention weights matrix
        attention_matrix = torch.zeros(max_len, src_tensor.size(1))
        
        # Generate translation and collect attention weights
        for t in range(max_len):
            # Run the decoder for one time step
            _, hidden, attn_weights = model.decoder(
                decoder_input, hidden, encoder_outputs, src_mask
            )
            
            # Store attention weights
            if attn_weights is not None:
                attention_matrix[t] = attn_weights
            
            # Get the token with the highest predicted probability
            # We don't actually use the prediction; this is just to collect attention weights
            output, _, _ = model.decoder(
                decoder_input, hidden, encoder_outputs, src_mask
            )
            predicted = output.argmax(1)
            
            # Use the predicted token as the next decoder input
            decoder_input = predicted
    
    return attention_matrix


def visualize_attention_weights(attention_matrix, src_tokens, save_path=None):
    """Visualize attention weights as a heatmap."""
    # Create a custom colormap (white to blue)
    cmap = LinearSegmentedColormap.from_list("custom_blue", ["#ffffff", "#0063B1"])
    
    # Create the figure
    plt.figure(figsize=(10, 8))
    
    # Plot the attention matrix
    ax = plt.gca()
    ax.matshow(attention_matrix.cpu().numpy(), cmap=cmap)
    
    # Set the labels
    ax.set_xticks(range(len(src_tokens)))
    ax.set_xticklabels([f"w_{i+1}" for i, token in enumerate(src_tokens)], rotation=45)
    
    ax.set_yticks(range(attention_matrix.size(0)))
    ax.set_yticklabels([f"output_{i+1}" for i in range(attention_matrix.size(0))])
    
    # Add colorbar
    cbar = plt.colorbar(ax.matshow(attention_matrix.cpu().numpy(), cmap=cmap))
    cbar.set_label("Attention Weight")
    
    # Add title and labels
    plt.title("Attention Weights Visualization", fontsize=14)
    plt.xlabel("Source Tokens", fontsize=12)
    plt.ylabel("Decoder Steps", fontsize=12)
    
    # Add a note about the visualization
    textstr = "\n".join([
        "Visualization Details:",
        "• Each row represents one step of the decoder",
        "• Each column represents one token in the source sequence",
        "• Darker color indicates higher attention weight",
        "• This shows which source tokens the model focuses on at each decoding step"
    ])
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.4)
    plt.figtext(0.5, 0.02, textstr, fontsize=10, wrap=True, horizontalalignment='center', bbox=props)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save figure
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Attention visualization saved to: {save_path}")
    
    plt.close()


def generate_multiple_attention_visualizations(model_path, save_dir="models", num_samples=3):
    """Generate multiple attention visualizations for different sample inputs."""
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Load model
    device = torch.device("cpu")
    model = load_model(model_path, device=device)
    
    # Generate multiple visualizations
    for i in range(num_samples):
        # Create sample data with varying sequence lengths
        seq_len = random.randint(5, 10)
        src_tensor, src_mask, src_lengths, src_seq, _ = create_sample_data(seq_len=seq_len)
        
        # Move to device
        src_tensor = src_tensor.to(device)
        src_mask = src_mask.to(device)
        src_lengths = src_lengths.to(device)
        
        # Generate attention matrix
        attention_matrix = generate_attention_matrix(
            model, src_tensor, src_mask, src_lengths, max_len=seq_len+2
        )
        
        # Visualize attention weights
        save_path = os.path.join(save_dir, f"attention_viz_{i+1}.png")
        visualize_attention_weights(attention_matrix, src_seq, save_path=save_path)


def main():
    """Main function to create attention visualizations."""
    # Path to the trained model with attention
    model_path = "models/model_attn_final.pt"
    
    # Generate attention visualizations
    print(f"Generating attention visualizations using model: {model_path}")
    generate_multiple_attention_visualizations(model_path, save_dir="models", num_samples=3)
    
    print("Attention visualizations complete. Check the models directory for the output files.")


if __name__ == "__main__":
    main()