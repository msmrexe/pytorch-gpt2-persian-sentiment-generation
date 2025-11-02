"""
Text generation script for the sentiment-controlled GPT-2 model.

This script loads a trained model checkpoint and generates text based on
a given sentiment and other generation parameters.
"""

import argparse
import logging
import os
import sys
import torch

# Add src to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils import setup_logging
from src.dataset import get_tokenizer
from src.config import GPT2Config
from src.model import GPT2

# Setup logger
logger = setup_logging()

def get_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate text with a trained sentiment-controlled GPT-2 model."
    )
    
    # Model
    parser.add_argument(
        '--model_path', 
        type=str, 
        default="models/best_gpt2_model.pt",
        help="Path to the saved model checkpoint (.pt file)."
    )
    
    # Generation
    parser.add_argument(
        '--sentiment', 
        type=str, 
        required=True, 
        choices=['positive', 'negative'],
        help="The desired sentiment for generation."
    )
    parser.add_argument(
        '--num_samples', 
        type=int, 
        default=5,
        help="Number of samples to generate."
    )
    parser.add_argument(
        '--max_new_tokens', 
        type=int, 
        default=50,
        help="Maximum number of new tokens to generate per sample."
    )
    parser.add_argument(
        '--temperature', 
        type=float, 
        default=0.8,
        help="Softmax temperature for sampling."
    )
    parser.add_argument(
        '--top_k', 
        type=int, 
        default=50,
        help="Top-k sampling (0 to disable)."
    )
    parser.add_argument(
        '--top_p', 
        type=float, 
        default=0.9,
        help="Nucleus (top-p) sampling (1.0 to disable)."
    )
    
    # System
    parser.add_argument(
        '--device', 
        type=str, 
        default=None,
        help="Device to use (e.g., 'cuda', 'cpu'). Autodetects if None."
    )
    
    return parser.parse_args()

def load_model(model_path: str, device: torch.device) -> tuple[GPT2, AutoTokenizer]:
    """Loads the model and tokenizer from a checkpoint."""
    if not os.path.exists(model_path):
        logger.error(f"Model checkpoint not found at: {model_path}")
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    logger.info(f"Loading checkpoint from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Load config and tokenizer
    config = checkpoint['config']
    tokenizer_name = checkpoint['tokenizer_name']
    
    logger.info(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = get_tokenizer(tokenizer_name)
    
    # Re-initialize model with saved config
    # We must update vocab_size in case tokenizer was updated
    config.vocab_size = len(tokenizer)
    model = GPT2(config)
    
    # Load model state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info("Model and tokenizer loaded successfully.")
    return model, tokenizer

def generate_text(
    model: GPT2,
    tokenizer: AutoTokenizer,
    sentiment: str,
    num_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    device: torch.device
) -> list[str]:
    """Generates a list of text samples."""
    
    # 1. Create the appropriate prefix
    if sentiment.lower() == "positive":
        prefix = "<POSITIVE>"
    elif sentiment.lower() == "negative":
        prefix = "<NEGATIVE>"
    else:
        raise ValueError("Sentiment must be 'positive' or 'negative'")

    # 2. Tokenize the prefix
    # We repeat the prompt for batch generation
    input_ids = tokenizer.encode(prefix, return_tensors='pt').to(device)
    input_ids = input_ids.repeat(num_samples, 1) # (num_samples, T_prompt)
    
    logger.info(f"Generating {num_samples} sample(s) for sentiment: '{sentiment}'")
    logger.info(
        f"Params: max_new={max_new_tokens}, temp={temperature}, "
        f"top_k={top_k}, top_p={top_p}"
    )
    
    # 3. Generate text
    generated_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p
    )
    
    # 4. Decode the generated text
    generated_texts = tokenizer.batch_decode(
        generated_ids, 
        skip_special_tokens=True
    )
    
    # 5. Clean the prefix from the output
    cleaned_texts = []
    for text in generated_texts:
        if text.startswith("<POSITIVE>"):
            cleaned_texts.append(text[10:].strip())
        elif text.startswith("<NEGATIVE>"):
            cleaned_texts.append(text[10:].strip())
        else:
            cleaned_texts.append(text.strip())
            
    return cleaned_texts

def main():
    """Main generation function."""
    args = get_args()
    logger.info(f"Arguments: {vars(args)}")
    
    # --- 1. Setup Device ---
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    try:
        # --- 2. Load Model and Tokenizer ---
        model, tokenizer = load_model(args.model_path, device)
        
        # --- 3. Generate Text ---
        texts = generate_text(
            model=model,
            tokenizer=tokenizer,
            sentiment=args.sentiment,
            num_samples=args.num_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device
        )
        
        # --- 4. Print Results ---
        print("\n" + "=" * 60)
        print(f"Generated {args.sentiment.upper()} Comments")
        print("=" * 60)
        for i, text in enumerate(texts):
            print(f"{i+1:2d}. {text}\n")
        print("=" * 60)

    except FileNotFoundError as e:
        logger.fatal(str(e))
        logger.fatal("Please run the training script first (`scripts/train.py`)")
    except Exception as e:
        logger.fatal(f"An unexpected error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()
