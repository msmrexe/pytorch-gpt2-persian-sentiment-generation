"""
Custom PyTorch Dataset and DataLoader setup for the Snappfood comments.
"""

import logging
from typing import Dict
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

class CommentDataset(Dataset):
    """
    Custom PyTorch Dataset for Snappfood comments.
    
    It formats each comment with a sentiment prefix (e.g., "<POSITIVE> text")
    and tokenizes it for language modeling.
    """
    def __init__(self, dataframe: pd.DataFrame, tokenizer: AutoTokenizer, max_length: int = 128):
        """
        Args:
            dataframe (pd.DataFrame): DataFrame containing 'comment_cleaned' and 'label'.
            tokenizer (AutoTokenizer): The tokenizer instance.
            max_length (int): Maximum sequence length for padding/truncation.
        """
        self.tokenizer = tokenizer
        self.comments = dataframe['comment_cleaned'].tolist()
        self.labels = dataframe['label'].tolist()
        self.max_length = max_length
        self.pos_token = "<POSITIVE>"
        self.neg_token = "<NEGATIVE>"

    def __len__(self) -> int:
        """Returns the total number of samples."""
        return len(self.comments)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Retrieves and formats a single sample.

        Args:
            idx (int): The index of the sample.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing 'input_ids', 
                                     'attention_mask', and 'labels'.
        """
        try:
            comment = str(self.comments[idx])
            label = self.labels[idx]
            
            # Create a prefix based on the sentiment (1 = positive, 0 = negative)
            sentiment_prefix = self.pos_token if label == 1 else self.neg_token
            
            # Combine prefix with comment
            text = f"{sentiment_prefix} {comment}"
            
            # Tokenize the text
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].squeeze(0) # Remove batch dim
            attention_mask = encoding['attention_mask'].squeeze(0)
            
            # For language modeling, labels are the same as input_ids.
            # The model's forward pass will handle shifting.
            labels = input_ids.clone()
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }
        except Exception as e:
            logger.error(f"Error processing item at index {idx}: {e}")
            # Return a dummy item to avoid crashing the loader
            return self.__getitem__(0) # Be careful with recursion, but 0 should be safe

def get_tokenizer(model_name: str) -> AutoTokenizer:
    """
    Loads a tokenizer and adds custom sentiment tokens.

    Args:
        model_name (str): The Hugging Face model identifier for the tokenizer.

    Returns:
        AutoTokenizer: The configured tokenizer.
    """
    logger.info(f"Loading tokenizer: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        logger.error(f"Failed to load tokenizer '{model_name}'.")
        logger.error("Please ensure you are logged in to Hugging Face CLI.")
        logger.error("Run `huggingface-cli login` with your token.")
        raise e

    # Set pad token if not available
    if tokenizer.pad_token is None:
        logger.warning("Tokenizer has no pad_token. Using eos_token as pad_token.")
        tokenizer.pad_token = tokenizer.eos_token

    # Add custom special tokens for sentiment
    special_tokens = {
        'additional_special_tokens': ['<POSITIVE>', '<NEGATIVE>']
    }
    num_added_tokens = tokenizer.add_special_tokens(special_tokens)
    
    if num_added_tokens > 0:
        logger.info(f"Added {num_added_tokens} special tokens: {special_tokens['additional_special_tokens']}")
    else:
        logger.info("Special sentiment tokens already exist in tokenizer.")

    logger.info(f"Tokenizer loaded. New vocabulary size: {len(tokenizer)}")
    return tokenizer

def create_dataloaders(
    raw_corpus: pd.DataFrame, 
    tokenizer: AutoTokenizer, 
    max_length: int, 
    batch_size: int, 
    test_size: float = 0.2, 
    random_state: int = 42
) -> tuple[DataLoader, DataLoader]:
    """
    Splits data and creates training and validation DataLoaders.

    Args:
        raw_corpus (pd.DataFrame): The preprocessed data.
        tokenizer (AutoTokenizer): The tokenizer.
        max_length (int): Max sequence length.
        batch_size (int): Batch size.
        test_size (float): Proportion of data for validation.
        random_state (int): Random seed for split.

    Returns:
        tuple[DataLoader, DataLoader]: Train and validation dataloaders.
    """
    logger.info(f"Splitting data... (test_size={test_size})")
    
    # Split the data into train and validation sets
    try:
        train_df, val_df = train_test_split(
            raw_corpus,
            test_size=test_size,
            random_state=random_state,
            stratify=raw_corpus['label']  # Ensure balanced distribution
        )
    except ValueError as e:
        logger.warning(f"Stratified split failed: {e}. Using non-stratified split.")
        train_df, val_df = train_test_split(
            raw_corpus,
            test_size=test_size,
            random_state=random_state
        )

    logger.info(f"Training samples: {len(train_df)}")
    logger.info(f"Validation samples: {len(val_df)}")

    # Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    # Create dataset instances
    train_dataset = CommentDataset(train_df, tokenizer, max_length=max_length)
    val_dataset = CommentDataset(val_df, tokenizer, max_length=max_length)

    # Create DataLoader instances
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    logger.info(f"Created {len(train_loader)} training batches.")
    logger.info(f"Created {len(val_loader)} validation batches.")
    
    return train_loader, val_loader
