"""
Module Name: autoencoder_tradeoffs.py
Description: This module contains experiments to measure the tradeoffs between symmetric and asymmetric autoencoders
for KV cache compression, specifically focusing on:
1. Decompression speed (decoder-only inference time)
2. Compression quality vs symmetric autoencoder using real KV cache data
"""

import torch
import torch.nn as nn
import time
import numpy as np
from typing import Tuple, Dict, List
import json
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os

from src.models.autoencoder import Autoencoder
from src.utils.buffer import Buffer
from src.evaluation.metrics import calculate_mse
from src.inference.benchmark import run_benchmark

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
    num_batches: int = 10  # Number of batches to test
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

def run_tradeoff_experiments(
    model_name: str = "meta-llama/Llama-2-7b-hf",
    latent_dim: int = 16,
    batch_size: int = 1024,
    num_runs: int = 5,
    output_dir: str = "experiment_results"
) -> Dict:
    """
    Run experiments to evaluate tradeoffs between different autoencoder architectures.
    
    Args:
        model_name: Name of the base model to use
        latent_dim: Dimension of the latent space
        batch_size: Batch size for inference
        num_runs: Number of runs for each configuration
        output_dir: Directory to save results
        
    Returns:
        Dictionary containing experiment results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Test different symmetric configurations (1, 2, 3)
    symmetric_configs = [
        {"name": "sym1", "encoder_depth": 1, "decoder_depth": 1, "num_epochs": 5, "learning_rate": 1e-3},
        {"name": "sym2", "encoder_depth": 2, "decoder_depth": 2, "num_epochs": 10, "learning_rate": 1e-3},
        {"name": "sym3", "encoder_depth": 3, "decoder_depth": 3, "num_epochs": 15, "learning_rate": 1e-3}
    ]
    
    # Test different asymmetric configurations (encoder 2-7, decoder fixed at 1)
    asymmetric_configs = [
        {"name": "asym2", "encoder_depth": 2, "decoder_depth": 1, "num_epochs": 10, "learning_rate": 1e-3},
        {"name": "asym3", "encoder_depth": 3, "decoder_depth": 1, "num_epochs": 10, "learning_rate": 1e-3},
        {"name": "asym4", "encoder_depth": 4, "decoder_depth": 1, "num_epochs": 10, "learning_rate": 1e-3},
        {"name": "asym5", "encoder_depth": 5, "decoder_depth": 1, "num_epochs": 10, "learning_rate": 1e-3},
        {"name": "asym6", "encoder_depth": 6, "decoder_depth": 1, "num_epochs": 10, "learning_rate": 1e-3},
        {"name": "asym7", "encoder_depth": 7, "decoder_depth": 1, "num_epochs": 10, "learning_rate": 1e-3}
    ]
    
    results = {
        "symmetric": [],
        "asymmetric": []
    }
    
    # Test all configurations with progress bar
    all_configs = symmetric_configs + asymmetric_configs
    for config in tqdm(all_configs, desc="Testing configurations"):
        # Create autoencoder with current configuration
        autoencoder = Autoencoder(
            input_dim=4096,  # Llama-2-7b hidden size
            latent_dim=latent_dim,
            encoder_depth=config["encoder_depth"],
            decoder_depth=config["decoder_depth"]
        )
        
        # Train autoencoder with adjusted parameters
        autoencoder.train(
            num_epochs=config["num_epochs"],
            learning_rate=config["learning_rate"],
            batch_size=batch_size
        )
        
        # Measure decompression time and reconstruction quality
        decompression_times = []
        reconstruction_qualities = []
        
        for _ in range(num_runs):
            # Run benchmark with current configuration
            benchmark_results = run_benchmark(
                model_name=model_name,
                autoencoder=autoencoder,
                batch_size=batch_size
            )
            
            decompression_times.append(benchmark_results["mean_decompression_time"])
            reconstruction_qualities.append(benchmark_results["mean_reconstruction_quality"])
        
        # Calculate mean and std for metrics
        mean_decompression_time = np.mean(decompression_times)
        std_decompression_time = np.std(decompression_times)
        mean_reconstruction_quality = np.mean(reconstruction_qualities)
        std_reconstruction_quality = np.std(reconstruction_qualities)
        
        # Store results
        result = {
            "name": config["name"],
            "encoder_depth": config["encoder_depth"],
            "decoder_depth": config["decoder_depth"],
            "num_parameters": autoencoder.get_num_parameters(),
            "mean_decompression_time": mean_decompression_time,
            "std_decompression_time": std_decompression_time,
            "mean_reconstruction_quality": mean_reconstruction_quality,
            "std_reconstruction_quality": std_reconstruction_quality
        }
        
        if config in symmetric_configs:
            results["symmetric"].append(result)
        else:
            results["asymmetric"].append(result)
    
    # Save results
    output_path = os.path.join(output_dir, "autoencoder_tradeoffs.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
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
        print(f"  Decompression Time: {metrics['mean_decompression_time']:.3f} ± {metrics['std_decompression_time']:.3f} ms")
        print(f"  Reconstruction MSE: {metrics['mean_reconstruction_quality']:.6f}")
        print(f"  Parameters: {metrics['num_parameters']:,}")
    
    print("\nAsymmetric Models:")
    for name, metrics in sorted(asymmetric.items()):
        print(f"\n{name}:")
        print(f"  Decompression Time: {metrics['mean_decompression_time']:.3f} ± {metrics['std_decompression_time']:.3f} ms")
        print(f"  Reconstruction MSE: {metrics['mean_reconstruction_quality']:.6f}")
        print(f"  Parameters: {metrics['num_parameters']:,}")
    
    # Verify decoder consistency
    print("\nVerifying Decoder Consistency:")
    asym_times = [m["mean_decompression_time"] for m in asymmetric.values()]
    asym_std = [m["std_decompression_time"] for m in asymmetric.values()]
    time_variation = np.std(asym_times)
    print(f"Asymmetric model decompression time variation: {time_variation:.6f} ms")
    if time_variation < 0.1:  # Less than 0.1ms variation
        print("✓ All asymmetric models have consistent decompression times (as expected)")
    else:
        print("⚠ Warning: Unexpected variation in asymmetric model decompression times")

if __name__ == "__main__":
    main() 