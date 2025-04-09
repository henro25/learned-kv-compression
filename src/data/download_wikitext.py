"""
Module Name: download_wikitext.py
Description: Script to download and prepare the WikiText dataset.
Author: Ben Choi
Date: 2025-04-08
"""

import os
from datasets import load_dataset
from transformers import AutoTokenizer

def download_wikitext():
    """Download and prepare the WikiText dataset."""
    print("Downloading WikiText dataset...")
    
    # Create data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Download WikiText dataset
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    
    # Save the dataset
    dataset.save_to_disk(os.path.join(data_dir, "wikitext"))
    
    print("WikiText dataset downloaded and saved successfully!")
    print(f"Dataset saved to: {os.path.join(data_dir, 'wikitext')}")

if __name__ == "__main__":
    download_wikitext() 