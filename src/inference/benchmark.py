"""
Module Name: benchmark.py
Description: Benchmark script for KV Cache compression using actual KV cache data.
Author: Ben Choi, Henry Huang
Date: 2025-04-08
"""

import argparse
import json
import os
import time
from datasets import load_dataset
from src.inference.inference import KVCacheInference
import torch
import numpy as np
from typing import Dict, Any
from tqdm import tqdm

def run_benchmark(
    model_name: str,
    autoencoder: Any,
    batch_size: int = 1024,
    num_runs: int = 10
) -> Dict[str, float]:
    """
    Run compression benchmark with the given autoencoder.
    
    Args:
        model_name: Name of the language model (for reporting)
        autoencoder: Trained autoencoder model
        batch_size: Batch size for testing
        num_runs: Number of runs for averaging
        
    Returns:
        Dictionary with benchmark results
    """
    device = next(autoencoder.parameters()).device
    
    # Generate synthetic test data (simulate KV cache)
    input_dim = autoencoder.encoder[0].in_features
    test_data = torch.randn(batch_size, input_dim).to(device)
    
    # Measure baseline performance
    baseline_times = []
    with torch.no_grad():
        # Warmup
        for _ in range(5):
            _ = test_data  # Just access the data
            
        # Benchmark
        for _ in range(num_runs):
            start_time = time.time()
            _ = test_data  # Baseline just accesses the data
            end_time = time.time()
            baseline_times.append((end_time - start_time) * 1000)  # ms
    
    # Measure compression and decompression
    compression_times = []
    decompression_times = []
    reconstruction_qualities = []
    
    with torch.no_grad():
        # Warmup
        for _ in range(5):
            compressed, _ = autoencoder.encoder(test_data)
            _ = autoencoder.decoder(compressed)
            
        # Benchmark
        for _ in range(num_runs):
            # Compression
            start_time = time.time()
            compressed, _ = autoencoder.encoder(test_data)
            end_time = time.time()
            compression_times.append((end_time - start_time) * 1000)  # ms
            
            # Decompression
            start_time = time.time()
            reconstructed = autoencoder.decoder(compressed)
            end_time = time.time()
            decompression_times.append((end_time - start_time) * 1000)  # ms
            
            # Quality
            mse = torch.mean((test_data - reconstructed) ** 2).item()
            reconstruction_qualities.append(mse)
    
    # Calculate metrics
    mean_baseline = np.mean(baseline_times)
    mean_compression = np.mean(compression_times)
    mean_decompression = np.mean(decompression_times)
    mean_reconstruction_quality = np.mean(reconstruction_qualities)
    
    # Calculate compression ratio
    latent_dim = autoencoder.decoder[0].in_features
    compression_ratio = input_dim / latent_dim
    
    # Calculate speedup
    baseline_latency = mean_baseline
    compressed_latency = mean_compression + mean_decompression
    speedup = baseline_latency / compressed_latency if compressed_latency > 0 else float('inf')
    
    return {
        "model_name": model_name,
        "encoder_depth": autoencoder.encoder.parameters,
        "decoder_depth": autoencoder.decoder.parameters,
        "compression_ratio": compression_ratio,
        "mean_baseline_time": mean_baseline,
        "mean_compression_time": mean_compression,
        "mean_decompression_time": mean_decompression,
        "mean_reconstruction_quality": mean_reconstruction_quality,
        "speedup": speedup
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run KV Cache compression benchmark")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--autoencoder", type=str, required=True, help="Path to autoencoder model")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--num_runs", type=int, default=10, help="Number of runs for averaging")
    
    args = parser.parse_args()
    
    run_benchmark(
        model_name=args.model,
        autoencoder=torch.load(args.autoencoder),
        batch_size=args.batch_size,
        num_runs=args.num_runs
    ) 