#!/usr/bin/env python
"""
Script to inspect the structure of the saved model.
"""

import torch
import pprint

def inspect_model(model_path):
    """Inspect the structure of a saved PyTorch model."""
    print(f"Loading model from: {model_path}")
    
    try:
        # Load the state dict
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Print model keys and shapes
        print("\nModel Structure:")
        print("================")
        
        # Get the keys and their shapes
        model_structure = {key: tuple(state_dict[key].shape) for key in state_dict}
        
        # Pretty print the model structure
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(model_structure)
        
        # Analyze parameter shapes for architecture reconstruction
        encoder_params = {k: v for k, v in model_structure.items() if k.startswith('encoder')}
        decoder_params = {k: v for k, v in model_structure.items() if k.startswith('decoder')}
        
        # Extract key dimensions
        if 'encoder.embedding.weight' in model_structure:
            src_vocab_size = model_structure['encoder.embedding.weight'][0]
            emb_dim = model_structure['encoder.embedding.weight'][1]
            print(f"\nSource vocabulary size: {src_vocab_size}")
            print(f"Embedding dimension: {emb_dim}")
        
        if 'encoder.rnn.weight_ih_l0' in model_structure:
            rnn_input_dim = model_structure['encoder.rnn.weight_ih_l0'][1]
            hidden_size = model_structure['encoder.rnn.weight_ih_l0'][0] // 3  # For GRU (3 gates)
            print(f"RNN input dimension: {rnn_input_dim}")
            print(f"Hidden dimension: {hidden_size}")
        
        # Determine number of layers
        n_layers = 1
        i = 1
        while f'encoder.rnn.weight_ih_l{i}' in model_structure:
            n_layers += 1
            i += 1
        print(f"Number of layers: {n_layers}")
        
        # Determine if using attention
        has_attention = any(k.startswith('decoder.attention') for k in model_structure)
        print(f"Uses attention: {has_attention}")
        
        if 'decoder.fc_out.weight' in model_structure:
            tgt_vocab_size = model_structure['decoder.fc_out.weight'][0]
            print(f"Target vocabulary size: {tgt_vocab_size}")
            
        print("\nRecommended model configuration:")
        print("===============================")
        print(f"src_vocab_size = {src_vocab_size}")
        print(f"tgt_vocab_size = {tgt_vocab_size}")
        print(f"emb_dim = {emb_dim}")
        print(f"hidden_dim = {hidden_size}")
        print(f"n_layers = {n_layers}")
        print(f"use_attention = {has_attention}")
        
    except Exception as e:
        print(f"Error inspecting model: {str(e)}")


if __name__ == "__main__":
    model_path = "models/model_attn_final.pt"
    inspect_model(model_path)