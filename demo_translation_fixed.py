#!/usr/bin/env python
"""
Script to translate sentences using the seq2seq model with proper configuration.
"""

import torch
import argparse
import os
import numpy as np
from tokenizers import Tokenizer
from model import Encoder, Decoder, Attention, Seq2Seq

def translate_sentence(model, sentence, device):
    """
    Translate a sentence using the model with fixed vocabulary.
    
    This version doesn't rely on the tokenizers but uses a simple approach
    since we know the model has a very small vocabulary.
    """
    model.eval()
    
    # Simple tokenization by space
    tokens = ["<s>"] + sentence.lower().split() + ["</s>"]
    
    # Map to IDs, using special values for unknown words
    # Note we limit to the model's vocabulary size of 100
    token_map = {"<s>": 0, "</s>": 1, "<unk>": 2}
    ids = [token_map.get(token, 2) for token in tokens]  # Map unknown words to <unk> (ID 2)
    
    print(f"Input: '{sentence}'")
    print(f"Tokenized: {tokens}")
    print(f"IDs: {ids}")
    
    # Convert to tensor
    src_tensor = torch.tensor([ids], dtype=torch.long).to(device)
    src_mask = torch.ones_like(src_tensor, dtype=torch.bool).to(device)
    src_lengths = torch.tensor([len(ids)], dtype=torch.long).to(device)
    
    # Define BOS and EOS indices
    bos_idx = 0  # <s>
    eos_idx = 1  # </s>
    
    with torch.no_grad():
        # Encode the source sentence
        encoder_outputs, hidden = model.encoder(src_tensor, src_lengths)
        
        # Initialize tensors for decoding
        batch_size = src_tensor.size(0)
        max_len = 50  # Maximum translation length
        translations = torch.ones(batch_size, max_len, dtype=torch.long).to(device) * eos_idx
        translations[:, 0] = bos_idx
        
        # Initialize attention weights tensor if using attention
        attention_weights = None
        if model.use_attention:
            attention_weights = torch.zeros(batch_size, max_len, src_tensor.size(1)).to(device)
        
        # Decode one step at a time
        decoder_input = torch.ones(batch_size, dtype=torch.long).fill_(bos_idx).to(device)
        completed = torch.zeros(batch_size, dtype=torch.bool).to(device)
        
        for t in range(1, max_len):
            # Run the decoder for one time step
            output, hidden, attn_weights = model.decoder(
                decoder_input, hidden, encoder_outputs, src_mask
            )
            
            # Get the token with the highest predicted probability
            predicted = output.argmax(1)
            
            # Store the predicted token and attention weights
            translations[:, t] = predicted
            if model.use_attention and attn_weights is not None:
                attention_weights[:, t, :] = attn_weights
            
            # Check for completed sequences
            completed = completed | (predicted == eos_idx)
            if completed.all():
                break
            
            # Set the next decoder input
            decoder_input = predicted
        
        # Extract translation for the first (and only) example in batch
        pred_tokens = translations[0].cpu().numpy()
        
        # Remove padding, BOS and EOS tokens
        pred_tokens = pred_tokens[pred_tokens != 0]  # Remove padding
        pred_tokens = pred_tokens[1:]  # Remove BOS
        if eos_idx in pred_tokens:
            pred_tokens = pred_tokens[: np.where(pred_tokens == eos_idx)[0][0]]
        
        # Convert back to words - since this is a fixed small vocabulary model,
        # we can't do proper detokenization
        translation = " ".join([str(idx) for idx in pred_tokens])
        
        # Since we can't properly detokenize without the original vocabulary,
        # we just return the numerical indices
        return translation, attention_weights


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Translate sentences between languages")
    
    # Model parameters
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="models/model_attn_final.pt",
        help="Path to trained model weights"
    )
    parser.add_argument(
        "--sentence", 
        type=str, 
        help="Sentence to translate (if provided, runs in non-interactive mode)"
    )
    
    args = parser.parse_args()
    
    # Set device - use CPU to avoid CUDA errors
    device = torch.device("cpu")
    
    # Use fixed parameters from the saved model
    hidden_dim = 64
    emb_dim = 32
    n_layers = 1
    src_vocab_size = 100
    tgt_vocab_size = 100
    use_attention = True
    
    print(f"Using device: {device}")
    print(f"Model parameters: emb_dim={emb_dim}, hidden_dim={hidden_dim}, n_layers={n_layers}")
    print(f"Source vocabulary size: {src_vocab_size}, Target vocabulary size: {tgt_vocab_size}")
    
    # Create encoder
    encoder = Encoder(
        input_dim=src_vocab_size,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=0.2,
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
        dropout=0.2,
        attention=attention,
    )

    # Create model
    model = Seq2Seq(
        encoder=encoder,
        decoder=decoder,
        device=device,
        use_attention=use_attention,
    )

    print(f"Loading model from {args.model_path}...")
    # Load trained weights
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print("Model loaded successfully")
    
    print("\nNote: Since we don't have the original small vocabulary, translations")
    print("will be displayed as numeric token IDs rather than proper text.")
    print("To use this model properly, you would need the original tokenizers")
    print("that match the vocabulary size of 100.")
    
    # If sentence is provided, translate just that sentence and exit
    if args.sentence:
        try:
            print(f"\nTranslating: {args.sentence}")
            translation, _ = translate_sentence(
                model=model,
                sentence=args.sentence,
                device=device,
            )
            print(f"\nSource: {args.sentence}")
            print(f"Translation (token IDs): {translation}")
        except Exception as e:
            print(f"Translation error: {str(e)}")
            import traceback
            traceback.print_exc()
        return
    
    # Interactive translation mode
    print("\nInteractive Translation Mode")
    print("Enter 'quit' to exit")
    
    while True:
        # Get input sentence
        sentence = input("\nEnter English sentence: ")
        
        if sentence.lower() == 'quit':
            break
        
        try:    
            # Translate
            translation, _ = translate_sentence(
                model=model,
                sentence=sentence,
                device=device,
            )
            
            # Print translation
            print(f"Translation (token IDs): {translation}")
        except Exception as e:
            print(f"Translation error: {str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()