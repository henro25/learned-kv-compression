"""
Module Name: hybrid_press.py
Description: Implementation of a hybrid compression approach combining KVPress quantization 
with autoencoder-based dimensionality reduction.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Any
import logging

# Import our autoencoder
from src.models.autoencoder import Autoencoder

# This would be equivalent to importing from kvpress.presses.base_press import BasePress
# You'll need to adapt based on how you install/import KVPress
class HybridKVPress:
    """
    Hybrid KV Cache Compression that combines:
    1. Autoencoder-based dimensionality reduction
    2. Quantization from KVPress
    
    This approach first uses our autoencoder to reduce dimensions, then applies
    quantization to further compress the latent representation.
    """
    
    def __init__(
        self, 
        autoencoder_path: str,
        quantization_bits: int = 8,
        device: str = "cuda"
    ):
        """
        Initialize the hybrid compression.
        
        Args:
            autoencoder_path: Path to the trained autoencoder model
            quantization_bits: Number of bits for quantization (4-8)
            device: Computation device
        """
        self.device = torch.device(device)
        self.quantization_bits = quantization_bits
        
        # Load the autoencoder
        checkpoint = torch.load(autoencoder_path, map_location=device)
        
        # Extract metadata
        if isinstance(checkpoint, dict) and "input_dim" in checkpoint and "latent_dim" in checkpoint:
            # Newer format with metadata
            self.input_dim = checkpoint["input_dim"]
            self.latent_dim = checkpoint["latent_dim"]
            autoencoder_state = checkpoint["state_dict"]
        else:
            # Older format, just the state dict
            # We'll need to infer dimensions from the state dict keys
            first_layer_key = [k for k in checkpoint.keys() if "encoder.0.weight" in k][0]
            last_layer_key = [k for k in checkpoint.keys() if "decoder.0.weight" in k][0]
            self.input_dim = checkpoint[last_layer_key].size(0)
            self.latent_dim = checkpoint[first_layer_key].size(0)
            autoencoder_state = checkpoint
        
        # Initialize autoencoder
        self.autoencoder = Autoencoder(
            input_dim=self.input_dim,
            latent_dim=self.latent_dim
        ).to(self.device)
        
        # Load weights
        self.autoencoder.load_state_dict(autoencoder_state)
        self.autoencoder.eval()
        
        # Set up quantization parameters
        self.scale = 2.0**(quantization_bits-1) - 1  # Scale for quantization
        
        print(f"Initialized Hybrid KV Press with:")
        print(f"  - Autoencoder: input_dim={self.input_dim}, latent_dim={self.latent_dim}")
        print(f"  - Quantization: {quantization_bits}-bit")
    
    def __call__(self, model):
        """Register hooks to compress KV cache during generation"""
        # Store original attention modules to restore later
        self.original_modules = {}
        
        # Register hooks for each attention layer
        for name, module in model.named_modules():
            # This pattern will need to be adjusted based on your model architecture
            if "self_attn" in name and hasattr(module, "forward") and "rotary" not in name.lower():
                # Save reference to original module
                self.original_modules[name] = module
                
                # Register forward hook
                module.register_forward_hook(self.forward_hook)
        
        # Return self for context manager usage
        return self
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Clean up by removing hooks if needed
        pass
    
    def quantize(self, x):
        """Quantize tensor to specified bit depth"""
        # Scale to use full range of quantization bits
        x_scaled = x * self.scale
        
        # Clamp to valid range
        x_clamped = torch.clamp(x_scaled, -self.scale, self.scale - 1)
        
        # Quantize by rounding to integers
        x_quantized = torch.round(x_clamped)
        
        # Scale back to original range
        return x_quantized / self.scale
    
    def compress_kv(self, keys, values):
        """Compress KV pairs using autoencoder + quantization"""
        with torch.no_grad():
            # Reshape for autoencoder
            batch_size, num_heads, seq_len, head_dim = keys.shape
            keys_flat = keys.reshape(-1, head_dim)
            values_flat = values.reshape(-1, head_dim)
            
            # Step 1: Encode to latent space
            _, k_latent = self.autoencoder.encoder(keys_flat)
            _, v_latent = self.autoencoder.encoder(values_flat)
            
            # Step 2: Quantize latent representation
            k_latent_quantized = self.quantize(k_latent)
            v_latent_quantized = self.quantize(v_latent)
            
            # Save compressed representation
            self.cached_k_latent = k_latent_quantized
            self.cached_v_latent = v_latent_quantized
            
            # Step 3: Decompress for immediate use
            # First dequantize (already handled by returning quantized values)
            # Then decode
            k_decoded = self.autoencoder.decoder(k_latent_quantized)
            v_decoded = self.autoencoder.decoder(v_latent_quantized)
            
            # Reshape back to original dimensions
            k_decoded = k_decoded.reshape(batch_size, num_heads, seq_len, head_dim)
            v_decoded = v_decoded.reshape(batch_size, num_heads, seq_len, head_dim)
            
            return k_decoded, v_decoded
    
    def forward_hook(self, module, inputs, outputs):
        """Hook to compress KV cache during forward pass"""
        # This implementation will depend on your model's specific structure
        # Assuming outputs includes past_key_values or cache components
        
        # Example for models where outputs include past_key_values
        if hasattr(outputs, "past_key_values") and outputs.past_key_values is not None:
            compressed_past_kv = []
            
            for layer_idx, (k, v) in enumerate(outputs.past_key_values):
                # Apply compression
                k_comp, v_comp = self.compress_kv(k, v)
                compressed_past_kv.append((k_comp, v_comp))
            
            # Replace with compressed version
            outputs.past_key_values = tuple(compressed_past_kv)
        
        return outputs
    
    @property
    def compression_ratio(self):
        """Calculate the total compression ratio achieved"""
        # Original size: input_dim * sequence_length * 2 (keys + values) * 4 bytes (float32)
        # Compressed size: latent_dim * sequence_length * 2 * (quantization_bits/8) bytes
        return (self.input_dim * 4) / (self.latent_dim * (self.quantization_bits/8))

# Example usage
if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Example usage
    model_name = "distilgpt2"
    autoencoder_path = "models/autoencoder_final.pth"
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create HybridKVPress with 4-bit quantization
    hybrid_press = HybridKVPress(
        autoencoder_path=autoencoder_path,
        quantization_bits=4
    )
    
    # Input text
    text = "Hello, world! This is a test."
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    
    # Generate with compression
    with torch.no_grad(), hybrid_press(model):
        outputs = model.generate(
            input_ids, 
            max_length=50,
            do_sample=True
        )
    
    # Print result
    print(tokenizer.decode(outputs[0]))
    print(f"Compression ratio: {hybrid_press.compression_ratio:.2f}x") 