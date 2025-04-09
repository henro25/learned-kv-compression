"""
Module for evaluation metrics.
Author: Ben Choi, Henry Huang
Date: 2025-04-08
"""

import torch
import numpy as np

def calculate_mse(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """
    Calculate mean squared error between original and reconstructed tensors.
    
    Args:
        original: Original tensor
        reconstructed: Reconstructed tensor
        
    Returns:
        Mean squared error
    """
    return torch.mean((original - reconstructed) ** 2).item() 