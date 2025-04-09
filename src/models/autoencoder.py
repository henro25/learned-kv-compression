"""
Module Name: autoencoder.py
Description: This module defines an Autoencoder model, which consists of an encoder that reduces the dimensionality of the input vector to a latent representation, and a decoder that reconstructs the original input from the latent representation.
Author: Henry Huang
Date: 2025-03-13
"""

import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, encoder_depth=1, decoder_depth=1, hidden_dim=None):
        super(Autoencoder, self).__init__()
        
        # If hidden_dim not specified, use input_dim
        if hidden_dim is None:
            hidden_dim = input_dim
            
        # Build encoder layers
        encoder_layers = []
        if encoder_depth == 1:
            encoder_layers.append(nn.Linear(input_dim, latent_dim))
        else:
            # First layer
            encoder_layers.append(nn.Linear(input_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            # Middle layers
            for _ in range(encoder_depth - 2):
                encoder_layers.append(nn.Linear(hidden_dim, hidden_dim))
                encoder_layers.append(nn.ReLU())
            # Final layer to latent space
            encoder_layers.append(nn.Linear(hidden_dim, latent_dim))
            
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder layers
        decoder_layers = []
        if decoder_depth == 1:
            decoder_layers.append(nn.Linear(latent_dim, input_dim))
        else:
            # First layer
            decoder_layers.append(nn.Linear(latent_dim, hidden_dim))
            decoder_layers.append(nn.ReLU())
            # Middle layers
            for _ in range(decoder_depth - 2):
                decoder_layers.append(nn.Linear(hidden_dim, hidden_dim))
                decoder_layers.append(nn.ReLU())
            # Final layer to input space
            decoder_layers.append(nn.Linear(hidden_dim, input_dim))
            
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

if __name__ == "__main__":
    # Quick test
    # Symmetric autoencoder
    model_sym = Autoencoder(input_dim=64, latent_dim=16, encoder_depth=2, decoder_depth=2)
    # Asymmetric autoencoder
    model_asym = Autoencoder(input_dim=64, latent_dim=16, encoder_depth=4, decoder_depth=1)
    
    sample = torch.randn(10, 64)
    recon_sym, z_sym = model_sym(sample)
    recon_asym, z_asym = model_asym(sample)
    print("Symmetric reconstruction shape:", recon_sym.shape)
    print("Asymmetric reconstruction shape:", recon_asym.shape)
