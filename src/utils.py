"""
Utility functions for logging, plotting, and data setup.
"""

import logging
import sys
import os
import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

def setup_logging(log_file: str = "logs/gpt2.log") -> logging.Logger:
    """
    Configures logging to both file and console.

    Args:
        log_file (str): Path to the log file.

    Returns:
        logging.Logger: The configured logger instance.
    """
    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create file handler
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Silence overly verbose loggers
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

    return logger

def download_data(dataset_name: str, output_path: str) -> str:
    """
    Downloads the dataset from KaggleHub.

    Args:
        dataset_name (str): The name of the Kaggle dataset.
        output_path (str): The directory to save the data.

    Returns:
        str: The path to the downloaded dataset file.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Downloading dataset '{dataset_name}' from KaggleHub...")
    try:
        os.makedirs(output_path, exist_ok=True)
        dataset_path = kagglehub.dataset_download(
            dataset_name, 
            path=output_path, 
            force_download=False # Avoid re-downloading if it exists
        )
        
        # kagglehub.dataset_download returns the path to the directory
        file_path = os.path.join(dataset_path, "cleaned_snappfood.csv")
        
        if not os.path.exists(file_path):
            logger.error(f"Downloaded dataset but file not found at {file_path}")
            # Try to find the csv in the dir
            for root, _, files in os.walk(dataset_path):
                for f in files:
                    if f.endswith('.csv'):
                        file_path = os.path.join(root, f)
                        logger.info(f"Found dataset file at: {file_path}")
                        return file_path
            raise FileNotFoundError("Could not find the .csv file in the downloaded dataset.")

        logger.info(f"Dataset downloaded and available at: {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Failed to download or locate dataset: {e}")
        logger.error("Please ensure you are authenticated with Kaggle.")
        logger.error("Run `kaggle config set -n path -v /path/to/kaggle/json`")
        raise

def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """
    Loads and preprocesses the raw data from the CSV file.

    Args:
        file_path (str): Path to the .csv file.

    Returns:
        pd.DataFrame: A DataFrame with 'comment_cleaned' and 'label' columns.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading raw data from {file_path}...")
    try:
        raw_corpus = pd.read_csv(file_path)
        raw_corpus = raw_corpus[["comment_cleaned", "label"]]
        logger.info(f"Data loaded. Total samples: {len(raw_corpus)}")
        logger.info(f"Label distribution:\n{raw_corpus['label'].value_counts()}")
        return raw_corpus
    except FileNotFoundError:
        logger.error(f"Data file not found at {file_path}")
        raise
    except KeyError:
        logger.error("Data file must contain 'comment_cleaned' and 'label' columns.")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def plot_losses(
    epochs: int,
    epoch_train_losses: List[float],
    epoch_val_losses: List[float],
    all_train_step_nums: List[int],
    all_train_step_losses: List[float],
    all_val_step_nums: List[int],
    all_val_step_losses: List[float],
    save_path: str = "models/training_loss_plots.png"
) -> None:
    """
    Visualizes and saves the training and validation loss curves.

    Args:
        epochs (int): Total number of epochs.
        epoch_train_losses (List[float]): Avg training loss per epoch.
        epoch_val_losses (List[float]): Avg validation loss per epoch.
        all_train_step_nums (List[int]): Training step numbers.
        all_train_step_losses (List[float]): Training loss at each step.
        all_val_step_nums (List[int]): Validation step numbers.
        all_val_step_losses (List[float]): Validation loss at each step.
        save_path (str): Path to save the plot image.
    """
    logger = logging.getLogger(__name__)
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))

    try:
        # a. Plot epoch-level training and validation losses
        ax = axes[0]
        ax.plot(range(1, epochs + 1), epoch_train_losses, 'b-o', label='Training Loss', linewidth=2, markersize=8)
        ax.plot(range(1, epochs + 1), epoch_val_losses, 'r-o', label='Validation Loss', linewidth=2, markersize=8)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training and Validation Loss per Epoch', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(1, epochs + 1))

        # b. Plot step-level training losses
        ax = axes[1]
        if all_train_step_nums and all_train_step_losses:
            ax.plot(all_train_step_nums, all_train_step_losses, 'b-', alpha=0.7, linewidth=1)
            ax.set_xlabel('Training Step', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.set_title('Training Loss per Step', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No step-level training data available', 
                    horizontalalignment='center', verticalalignment='center', 
                    transform=ax.transAxes, color='gray')

        # c. Plot step-level validation losses
        ax = axes[2]
        if all_val_step_nums and all_val_step_losses:
            ax.plot(all_val_step_nums, all_val_step_losses, 'r-', alpha=0.7, linewidth=1)
            ax.set_xlabel('Validation Step', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.set_title('Validation Loss per Step', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No step-level validation data available', 
                    horizontalalignment='center', verticalalignment='center', 
                    transform=ax.transAxes, color='gray')

        plt.tight_layout(pad=3.0)
        
        # Ensure models directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        logger.info(f"Loss plots saved to {save_path}")
        
    except Exception as e:
        logger.error(f"Failed to create or save plots: {e}")
    finally:
        plt.close(fig) # Close the figure to free memory
