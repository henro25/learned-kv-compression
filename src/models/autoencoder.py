"""
Module Name: autoencoder.py
Description: This module defines an Autoencoder model, which consists of an encoder that reduces the dimensionality of the input vector to a latent representation, and a decoder that reconstructs the original input from the latent representation.
Author: Henry Huang
Date: 2025-03-13
"""

import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, dtype=None):
        super(Autoencoder, self).__init__()
        # Encoder: compress from input_dim to latent_dim.
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU()
        )
        # Decoder: reconstruct from latent_dim to input_dim.
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim)
        )
        
        # Convert model parameters to specified dtype if provided
        if dtype is not None:
            self.to(dtype)
    
    def forward(self, x):
        # Ensure input and model parameters use the same dtype
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        x = x.to(device=device, dtype=dtype)
        
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

if __name__ == "__main__":
    # Quick test
    model = Autoencoder(input_dim=64, latent_dim=16)
    sample = torch.randn(10, 64)
    recon, z = model(sample)
    print("Reconstruction shape:", recon.shape)
