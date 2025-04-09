"""
Module Name: autoencoder_tradeoffs.py
Description: This module contains experiments to measure the tradeoffs between symmetric and asymmetric autoencoders
for KV cache compression, specifically focusing on:
1. Decompression speed (decoder-only inference time)
2. Compression quality vs symmetric autoencoder using real KV cache data
Author: Ben Choi (1), Henry Huang (2)
(1) Harvard College
(2) Harvard College
"""

import torch
import torch.nn as nn
import time
import numpy as np
from typing import Tuple, Dict
import json
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from src.models.autoencoder import Autoencoder
from src.utils.buffer import Buffer

def measure_decoder_speed(
    model: nn.Module,
    input_dim: int,
    latent_dim: int,
    batch_size: int = 32,
    num_runs: int = 1000  # Increased for better statistics
) -> Tuple[float, float]:
    """
    Measure decoder-only inference speed (decompression time).
    
    Args:
        model: The autoencoder model to test
        input_dim: Dimension of input vectors
        latent_dim: Dimension of latent space
        batch_size: Size of input batch
        num_runs: Number of runs to average over
        
    Returns:
        Tuple of (mean_time, std_time) in milliseconds
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Generate random latent vectors (simulating compressed data)
    z = torch.randn(batch_size, latent_dim).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(50):  # More warmup runs
            _ = model.decoder(z)
    
    # Ensure CUDA sync before timing
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Measure decoder-only inference time
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()  # More precise timer
            _ = model.decoder(z)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to ms
    
    # Remove outliers (times > 3 std from mean)
    times = np.array(times)
    mean = np.mean(times)
    std = np.std(times)
    valid_times = times[np.abs(times - mean) <= 3 * std]
    
    return np.mean(valid_times), np.std(valid_times)

def measure_reconstruction_quality(
    model: nn.Module,
    buffer: Buffer,
    num_batches: int = 1  # Reduced number of batches
) -> float:
    """
    Measure reconstruction quality using MSE on real KV cache data.
    
    Args:
        model: The autoencoder model to test
        buffer: Buffer containing real KV cache data
        num_batches: Number of batches to test
        
    Returns:
        Mean squared error between input and reconstruction
    """
    model.eval()
    total_mse = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for _ in range(num_batches):
            # Get a batch of real KV pairs
            kvs, _ = buffer.next()
            keys, values = kvs
            
            # Flatten for processing
            keys_flat = keys.reshape(-1, keys.size(-1))
            values_flat = values.reshape(-1, values.size(-1))
            
            # Process keys and values
            k_recon, _ = model(keys_flat)
            v_recon, _ = model(values_flat)
            
            # Compute MSE
            k_mse = nn.MSELoss()(k_recon, keys_flat)
            v_mse = nn.MSELoss()(v_recon, values_flat)
            
            # Average MSE for keys and values
            batch_mse = (k_mse + v_mse) / 2
            total_mse += batch_mse.item() * (keys_flat.size(0) + values_flat.size(0))
            total_samples += keys_flat.size(0) + values_flat.size(0)
    
    return total_mse / total_samples

def train_model(
    model: nn.Module,
    buffer: Buffer,
    config_name: str,
    epochs: int = 1,
    learning_rate: float = 1e-3,
    batch_size: int = 32
) -> nn.Module:
    """
    Train the model for the specified number of epochs.
    
    Args:
        model: The autoencoder model to train
        buffer: Buffer containing training data
        config_name: Name of the model configuration
        epochs: Number of epochs to train
        learning_rate: Learning rate for optimization
        batch_size: Batch size for training
        
    Returns:
        Trained model
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    print(f"\nTraining {config_name} for {epochs} epochs...")
    model.train()
    
    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0.0
        batches = 0
        
        # Progress bar for each epoch - reduced to 20 batches per epoch
        with tqdm(total=20, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for i in range(20):  # Process only 20 batches per epoch (reduced from 100)
                # Get a batch of real KV pairs
                kvs, _ = buffer.next()
                keys, values = kvs
                
                # Flatten for processing
                keys_flat = keys.reshape(-1, keys.size(-1))
                values_flat = values.reshape(-1, values.size(-1))
                
                # Train on keys
                optimizer.zero_grad()
                k_recon, _ = model(keys_flat)
                k_loss = criterion(k_recon, keys_flat)
                k_loss.backward()
                optimizer.step()
                
                # Train on values
                optimizer.zero_grad()
                v_recon, _ = model(values_flat)
                v_loss = criterion(v_recon, values_flat)
                v_loss.backward()
                optimizer.step()
                
                # Track loss
                batch_loss = (k_loss.item() + v_loss.item()) / 2
                epoch_loss += batch_loss
                batches += 1
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix(loss=batch_loss)
        
        # Print epoch results
        print(f"Epoch {epoch+1} loss: {epoch_loss/batches:.6f}")
    
    return model

def run_tradeoff_experiments(
    model_name: str = "distilgpt2",
    latent_dim: int = 192,  # 4x compression for distilgpt2
    hidden_dim: int = 512,
    batch_size: int = 32,
    num_runs: int = 100,  # Reduced from 1000
    num_train_texts: int = 10,  # Reduced from 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict:
    """
    Run experiments comparing symmetric and asymmetric autoencoders using real KV cache data.
    """
    results = {}
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lm = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map={"": device},
        output_hidden_states=True,
        output_attentions=True,
        use_cache=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    lm.eval()
    
    # Get model dimensions
    head_dim = lm.config.hidden_size // lm.config.num_attention_heads
    input_dim = head_dim  # Each KV vector has dimension head_dim
    
    # Load WikiText dataset - use only 10 texts
    wiki_dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    texts = [text for text in wiki_dataset["train"]["text"][:num_train_texts] if text.strip()]
    
    # Create config for buffer with reduced sequence length
    cfg = {
        "name": model_name,
        "batch_size": batch_size,
        "buffer_mult": 2,  # Reduced from 4
        "lm_batch_size": 1,
        "num_hidden_layers": lm.config.num_hidden_layers,
        "num_key_value_heads": lm.config.num_attention_heads,
        "head_dim": head_dim,
        "max_seq_len": 64,  # Reduced from 128
        "device": device
    }
    
    # Initialize buffer with real KV cache data
    buffer = Buffer(cfg, lm, tokenizer, texts)
    
    # Define model configurations - 3 symmetric and 2 asymmetric
    configs = []
    
    # Symmetric configurations (depths 1, 2, 3)
    configs.append({
        "name": f"sym_1",
        "encoder_depth": 1,
        "decoder_depth": 1,
        "train_epochs": 1
    })
    configs.append({
        "name": f"sym_2",
        "encoder_depth": 2,
        "decoder_depth": 2,
        "train_epochs": 1  # Reduced from 3
    })

    
    # Asymmetric configurations (deep encoder, shallow decoder)
    for encoder_depth in [2, 3, 4, 5]:
        configs.append({
            "name": f"asym_{encoder_depth}_1",
            "encoder_depth": encoder_depth,
            "decoder_depth": 1,  # Fixed shallow decoder
            "train_epochs": 1
        })
    
    # Run experiments with progress bar
    for config in tqdm(configs, desc="Testing configurations"):
        print(f"\nTesting configuration: {config['name']}")
        
        # Create model
        model = Autoencoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            encoder_depth=config["encoder_depth"],
            decoder_depth=config["decoder_depth"],
            hidden_dim=hidden_dim
        ).to(device)
        
        # Train the model if specified
        if config.get("train_epochs", 0) > 0:
            model = train_model(
                model, 
                buffer, 
                config["name"],
                epochs=config["train_epochs"],
                learning_rate=1e-3,
                batch_size=batch_size
            )
        
        # Measure decoder-only speed (decompression time)
        mean_time, std_time = measure_decoder_speed(
            model, input_dim, latent_dim, batch_size, num_runs
        )
        
        # Measure reconstruction quality on real data
        mse = measure_reconstruction_quality(
            model, buffer, num_batches=1  # Reduced from 10
        )
        
        # Store results
        results[config["name"]] = {
            "mean_inference_time_ms": mean_time,
            "std_inference_time_ms": std_time,
            "reconstruction_mse": mse,
            "compression_ratio": input_dim / latent_dim,
            "total_params": sum(p.numel() for p in model.parameters()),
            "encoder_params": sum(p.numel() for p in model.encoder.parameters()),
            "decoder_params": sum(p.numel() for p in model.decoder.parameters()),
            "train_epochs": config.get("train_epochs", 0)
        }
        
        # Print progress
        print(f"  Decompression time: {mean_time:.3f} ± {std_time:.3f} ms")
        print(f"  Reconstruction MSE: {mse:.6f}")
        print(f"  Parameters: {results[config['name']]['total_params']:,}")
    
    return results

def main():
    # Run experiments
    results = run_tradeoff_experiments()
    
    # Save results
    output_dir = Path("experiment_results")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "autoencoder_tradeoffs.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\nExperiment Results Summary:")
    print("=" * 80)
    
    # Group results by type
    symmetric = {k: v for k, v in results.items() if k.startswith("sym")}
    asymmetric = {k: v for k, v in results.items() if k.startswith("asym")}
    
    print("\nSymmetric Models:")
    for name, metrics in sorted(symmetric.items()):
        print(f"\n{name}:")
        print(f"  Trained for {metrics['train_epochs']} epochs")
        print(f"  Decompression Time: {metrics['mean_inference_time_ms']:.3f} ± {metrics['std_inference_time_ms']:.3f} ms")
        print(f"  Reconstruction MSE: {metrics['reconstruction_mse']:.6f}")
        print(f"  Decoder Parameters: {metrics['decoder_params']:,}")
    
    print("\nAsymmetric Models:")
    for name, metrics in sorted(asymmetric.items()):
        print(f"\n{name}:")
        print(f"  Trained for {metrics['train_epochs']} epochs")
        print(f"  Decompression Time: {metrics['mean_inference_time_ms']:.3f} ± {metrics['std_inference_time_ms']:.3f} ms")
        print(f"  Reconstruction MSE: {metrics['reconstruction_mse']:.6f}")
        print(f"  Decoder Parameters: {metrics['decoder_params']:,}")
    
    # Verify decoder consistency
    print("\nVerifying Decoder Consistency:")
    asym_times = [m["mean_inference_time_ms"] for m in asymmetric.values()]
    asym_std = [m["std_inference_time_ms"] for m in asymmetric.values()]
    time_variation = np.std(asym_times)
    print(f"Asymmetric model decompression time variation: {time_variation:.6f} ms")
    if time_variation < 0.1:  # Less than 0.1ms variation
        print("✓ All asymmetric models have consistent decompression times (as expected)")
    else:
        print("⚠ Warning: Unexpected variation in asymmetric model decompression times")

if __name__ == "__main__":
    main() 