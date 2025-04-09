"""
Module Name: benchmark.py
Description: Benchmark script for KV Cache compression using actual KV cache data.
Author: Ben Choi
Date: 2025-04-08
"""

import argparse
import json
import os
import time
from datasets import load_dataset
from src.inference.inference import KVCacheInference
import torch

def run_benchmark(
    model_name,
    autoencoder_path,
    latent_dim,
    cache_sizes,
    batch_size=1024,
    num_runs=5,
    output_dir="experiment_results"
):
    """Run benchmark with actual KV cache data."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inference = KVCacheInference(
        model_name=model_name,
        device=device,
        autoencoder_path=autoencoder_path,
        latent_dim=latent_dim,
        batch_size=batch_size
    )
    
    # Load WikiText dataset
    print("Loading WikiText dataset...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    text = dataset["train"]["text"][0]  # Use first training example
    
    results = []
    for size_mb in cache_sizes:
        print(f"\nTesting cache size: {size_mb}MB")
        
        # Generate KV cache
        print("Generating KV cache...")
        past_key_values, generated_text, actual_size_mb = inference.generate_kv_cache(
            text, target_size_mb=size_mb
        )
        print(f"Generated KV cache of size: {actual_size_mb:.2f}MB")
        
        # Measure baseline performance
        print("Measuring baseline performance...")
        baseline_times = []
        for _ in range(num_runs):
            start_time = time.time()
            inference.measure_time_to_first_token(
                text, past_key_values, use_compression=False
            )
            baseline_times.append(time.time() - start_time)
        
        # Measure compressed performance
        print("Measuring compressed performance...")
        compressed_times = []
        for _ in range(num_runs):
            start_time = time.time()
            inference.measure_time_to_first_token(
                text, past_key_values, use_compression=True
            )
            compressed_times.append(time.time() - start_time)
        
        # Calculate metrics
        mean_baseline_time = sum(baseline_times) / len(baseline_times)
        mean_compressed_time = sum(compressed_times) / len(compressed_times)
        speedup = mean_baseline_time / mean_compressed_time
        
        # Store results
        results.append({
            "cache_size_mb": actual_size_mb,
            "mean_baseline_time": mean_baseline_time,
            "mean_compressed_time": mean_compressed_time,
            "speedup": speedup,
            "baseline_times": baseline_times,
            "compressed_times": compressed_times
        })
    
    # Save results
    output_path = os.path.join(output_dir, "benchmark_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run KV Cache compression benchmark")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--autoencoder", type=str, required=True, help="Path to autoencoder model")
    parser.add_argument("--latent_dim", type=int, required=True, help="Latent dimension")
    parser.add_argument("--cache_sizes", type=int, nargs="+", required=True, help="Cache sizes to test (MB)")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--num_runs", type=int, default=5, help="Number of runs per test")
    parser.add_argument("--output_dir", type=str, default="experiment_results", help="Output directory")
    
    args = parser.parse_args()
    
    run_benchmark(
        model_name=args.model,
        autoencoder_path=args.autoencoder,
        latent_dim=args.latent_dim,
        cache_sizes=args.cache_sizes,
        batch_size=args.batch_size,
        num_runs=args.num_runs,
        output_dir=args.output_dir
    ) 