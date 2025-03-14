"""
Module Name: autoencoder.py
Description: This module defines an Autoencoder model, which consists of an encoder that reduces the dimensionality of the input vector to a latent representation, and a decoder that reconstructs the original input from the latent representation using a mean squared error reconstruction loss.
Author: Henry Huang
Date: 2025-03-13
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
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
    
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        loss = torch.mean((x_recon - x) ** 2)
        return loss, x_recon, z

if __name__ == "__main__":
    # Quick test
    model = Autoencoder(input_dim=64, latent_dim=16)
    sample = torch.randn(10, 64)
    loss, recon, z = model(sample)
    print("Reconstruction shape:", recon.shape)
