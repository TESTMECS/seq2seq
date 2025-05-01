#!/usr/bin/env python
"""
Simple translation script with fixed parameters for the saved model.
"""

import torch
import numpy as np
from tokenizers import Tokenizer
from model import Encoder, Decoder, Attention, Seq2Seq

def translate(sentence, model, src_tokenizer, tgt_tokenizer, device, max_len=50):
    """
    Customized translate function to handle special token issues.
    """
    model.eval()
    
    # Tokenize the source sentence
    encoding = src_tokenizer.encode(sentence)
    tokens = encoding.ids
    
    # Print tokenization for debugging
    print(f"Input: '{sentence}'")
    print(f"Tokenized: {encoding.tokens}")
    
    # Convert to tensor
    src_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
    src_mask = torch.ones_like(src_tensor, dtype=torch.bool).to(device)
    src_lengths = torch.tensor([len(tokens)], dtype=torch.long).to(device)
    
    # Get token indices - use defaults if not found in tokenizer
    bos_idx = tgt_tokenizer.token_to_id("[BOS]")
    if bos_idx is None:
        bos_idx = 2  # Common default for BOS
        print("Warning: Using default BOS token index (2)")
    
    eos_idx = tgt_tokenizer.token_to_id("[EOS]")
    if eos_idx is None:
        eos_idx = 3  # Common default for EOS
        print("Warning: Using default EOS token index (3)")
    
    with torch.no_grad():
        # Encode the source sentence
        encoder_outputs, hidden = model.encoder(src_tensor, src_lengths)
        
        # Initialize tensors for decoding
        batch_size = src_tensor.size(0)
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
        pred_tokens = pred_tokens[pred_tokens != 0]
        pred_tokens = pred_tokens[1:]  # Remove BOS
        if eos_idx in pred_tokens:
            pred_tokens = pred_tokens[: np.where(pred_tokens == eos_idx)[0][0]]
        
        # Decode the translation
        # Try both decoding methods in case one fails
        try:
            translation = tgt_tokenizer.decode(pred_tokens.tolist())
        except:
            # Fallback decoding, try to handle each token individually
            translation_tokens = []
            for token_id in pred_tokens:
                try:
                    token = tgt_tokenizer.id_to_token(int(token_id))
                    if token and token not in ("[PAD]", "[BOS]", "[EOS]"):
                        translation_tokens.append(token)
                except:
                    pass
            translation = " ".join(translation_tokens)
    
    return translation


def main():
    # Parameters
    model_path = "models/model_attn_final.pt"
    src_tokenizer_path = "en_tokenizer.json"
    tgt_tokenizer_path = "fr_tokenizer.json"
    use_attention = True
    hidden_dim = 64
    emb_dim = 32
    n_layers = 1
    src_vocab_size = 100
    tgt_vocab_size = 100
    device = torch.device("cpu")  # Force CPU to avoid CUDA errors
    
    print(f"Using device: {device}")
    
    # Load tokenizers
    print("Loading tokenizers...")
    src_tokenizer = Tokenizer.from_file(src_tokenizer_path)
    tgt_tokenizer = Tokenizer.from_file(tgt_tokenizer_path)
    
    # Create encoder
    encoder = Encoder(
        input_dim=src_vocab_size,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=0.2,
    )
    
    # Create attention mechanism
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
    
    # Load trained weights
    print(f"Loading model from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    print("Model loaded successfully")
    
    # Check special tokens and set defaults if needed
    print("\nChecking and setting up tokenizer special tokens:")
    
    # Known special token maps from different tokenizers
    bos_alternatives = ["[BOS]", "<s>", "<bos>", "BOS", "<BOS>"]
    eos_alternatives = ["[EOS]", "</s>", "<eos>", "EOS", "<EOS>"]
    pad_alternatives = ["[PAD]", "<pad>", "PAD", "<PAD>"]
    
    # Find special token IDs
    def find_special_token_id(tokenizer, alternatives):
        for token in alternatives:
            token_id = tokenizer.token_to_id(token)
            if token_id is not None:
                return token, token_id
        return alternatives[0], None
    
    # Get special token IDs for both tokenizers
    for tokenizer_name, tokenizer in [("Source", src_tokenizer), ("Target", tgt_tokenizer)]:
        print(f"{tokenizer_name} tokenizer:")
        
        # Find BOS
        bos_token, bos_id = find_special_token_id(tokenizer, bos_alternatives)
        if bos_id is not None:
            print(f"  BOS token: '{bos_token}' → ID: {bos_id}")
        else:
            print(f"  BOS token not found, will use default ID")
            
        # Find EOS
        eos_token, eos_id = find_special_token_id(tokenizer, eos_alternatives)
        if eos_id is not None:
            print(f"  EOS token: '{eos_token}' → ID: {eos_id}")
        else:
            print(f"  EOS token not found, will use default ID")
            
        # Find PAD
        pad_token, pad_id = find_special_token_id(tokenizer, pad_alternatives)
        if pad_id is not None:
            print(f"  PAD token: '{pad_token}' → ID: {pad_id}")
        else:
            print(f"  PAD token not found, will use default ID")
    
    # Interactive translation
    print("\nEnglish to French Translation")
    print("Enter 'quit' to exit")
    
    # Test examples
    test_examples = [
        "Hello world",
        "How are you?",
        "I love programming",
        "The weather is nice today"
    ]
    
    print("\nTest examples:")
    for example in test_examples:
        try:
            translation = translate(
                sentence=example,
                model=model,
                src_tokenizer=src_tokenizer,
                tgt_tokenizer=tgt_tokenizer,
                device=device,
            )
            print(f"EN: {example}")
            print(f"FR: {translation}")
            print()
        except Exception as e:
            print(f"Error translating '{example}': {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Interactive mode
    print("\nInteractive mode:")
    while True:
        sentence = input("\nEnter English sentence (or 'quit' to exit): ")
        
        if sentence.lower() == 'quit':
            break
        
        try:
            translation = translate(
                sentence=sentence,
                model=model,
                src_tokenizer=src_tokenizer,
                tgt_tokenizer=tgt_tokenizer,
                device=device,
            )
            print(f"French translation: {translation}")
        except Exception as e:
            print(f"Translation error: {str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()