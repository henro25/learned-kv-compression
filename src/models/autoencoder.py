"""
Module Name: autoencoder.py
Description: This module defines an Autoencoder model for KV cache compression.
Author: Ben Choi, Henry Huang
Date: 2025-04-08
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

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
    
    def train(self, num_epochs=10, learning_rate=1e-3, batch_size=1024):
        """
        Train the autoencoder.
        
        Args:
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimization
            batch_size: Batch size for training
            
        Returns:
            List of training losses
        """
        device = next(self.parameters()).device
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Generate synthetic training data
        # For now, we'll use random data. In practice, you'd use real KV cache data
        num_samples = 10000
        train_data = torch.randn(num_samples, self.encoder[0].in_features).to(device)
        
        losses = []
        for epoch in range(num_epochs):
            self.train()
            total_loss = 0
            num_batches = num_samples // batch_size
            
            for i in range(num_batches):
                # Get batch
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                batch = train_data[start_idx:end_idx]
                
                # Forward pass
                recon, _ = self(batch)
                loss = criterion(recon, batch)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / num_batches
            losses.append(avg_loss)
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")
        
        return losses
    
    def get_num_parameters(self):
        """Get total number of parameters in the model."""
        return sum(p.numel() for p in self.parameters())

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
