"""
Module Name: benchmark.py
Description: Benchmark script for KV Cache compression. This script runs experiments
for different KV cache sizes with and without compression, and generates visualizations
of the results.
Author: AI Assistant
Date: 2023-04-03
"""

import os
import json
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.inference.inference import KVCacheInference

def run_benchmark(model_name, autoencoder_path=None, latent_dim=16, sizes=None, output_dir="results"):
    """
    Run benchmarks for different KV cache sizes.
    
    Args:
        model_name (str): Name of the model to use
        autoencoder_path (str, optional): Path to trained autoencoder
        latent_dim (int): Latent dimension of autoencoder
        sizes (list): List of KV cache sizes in MB to benchmark
        output_dir (str): Directory to save results
    """
    if sizes is None:
        sizes = [1, 10, 100, 1000, 3000]  # Default sizes in MB (up to 3GB)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize inference
    inference = KVCacheInference(
        model_name=model_name,
        autoencoder_path=autoencoder_path,
        latent_dim=latent_dim
    )
    
    results = {
        "model": model_name,
        "autoencoder": autoencoder_path,
        "latent_dim": latent_dim,
        "benchmarks": {}
    }
    
    # Run benchmark for each size
    for size_mb in tqdm(sizes, desc="Benchmarking different cache sizes"):
        print(f"\nRunning benchmark for {size_mb}MB KV cache")
        
        # Generate prompt and KV cache
        prompt = "Once upon a time in a land far away, there was a kingdom of"
        past_key_values, generated_text, actual_size_mb = inference.generate_kv_cache(prompt, size_mb)
        
        # Store the actual size
        size_result = {
            "target_size_mb": size_mb,
            "actual_size_mb": actual_size_mb,
            "generated_text": generated_text,
            "times": {}
        }
        
        # Measure time to first token without compression (baseline)
        print("Measuring baseline (no compression)...")
        cpu_kv_cache = tuple((k.cpu(), v.cpu()) for k, v in past_key_values)
        time_baseline, _ = inference.measure_time_to_first_token(
            generated_text, 
            cpu_kv_cache=cpu_kv_cache, 
            use_compression=False
        )
        size_result["times"]["baseline"] = time_baseline
        
        # If autoencoder is provided, measure with compression
        if autoencoder_path:
            print("Measuring with compression...")
            # Compress KV cache
            compressed_kv = inference.compress_kv_cache(past_key_values)
            cpu_compressed_kv = inference.move_to_cpu(compressed_kv)
            
            # Measure time to first token with compression
            time_compressed, _ = inference.measure_time_to_first_token(
                generated_text, 
                cpu_kv_cache=cpu_compressed_kv, 
                use_compression=True
            )
            size_result["times"]["compressed"] = time_compressed
            
            # Calculate compression ratio
            original_size = sum(k.nelement() * k.element_size() + v.nelement() * v.element_size() 
                               for k, v in past_key_values)
            compressed_size = sum(k.nelement() * k.element_size() + v.nelement() * v.element_size() 
                                 for k, v, _, _ in compressed_kv)
            compression_ratio = original_size / compressed_size
            size_result["compression_ratio"] = compression_ratio
            
            print(f"Compression ratio: {compression_ratio:.2f}x")
        
        print(f"Time to first token (baseline): {time_baseline:.4f}s")
        if autoencoder_path:
            print(f"Time to first token (compressed): {time_compressed:.4f}s")
        
        # Add to results
        results["benchmarks"][str(size_mb)] = size_result
    
    # Save full results
    results_file = os.path.join(output_dir, "benchmark_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_file}")
    
    # Generate visualizations
    visualize_results(results, output_dir)
    
    return results

def visualize_results(results, output_dir):
    """
    Generate visualizations of benchmark results.
    
    Args:
        results (dict): Benchmark results
        output_dir (str): Directory to save visualizations
    """
    # Extract data for plotting
    sizes = []
    baseline_times = []
    compressed_times = []
    compression_ratios = []
    
    for size_mb, data in results["benchmarks"].items():
        sizes.append(float(size_mb))
        baseline_times.append(data["times"]["baseline"])
        if "compressed" in data["times"]:
            compressed_times.append(data["times"]["compressed"])
        if "compression_ratio" in data:
            compression_ratios.append(data["compression_ratio"])
    
    # Sort by size
    sorted_indices = np.argsort(sizes)
    sizes = [sizes[i] for i in sorted_indices]
    baseline_times = [baseline_times[i] for i in sorted_indices]
    if compressed_times:
        compressed_times = [compressed_times[i] for i in sorted_indices]
    if compression_ratios:
        compression_ratios = [compression_ratios[i] for i in sorted_indices]
    
    # Plot time to first token comparison
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, baseline_times, 'o-', label='Baseline (No Compression)')
    if compressed_times:
        plt.plot(sizes, compressed_times, 's-', label='With Compression')
    plt.xscale('log')
    plt.xlabel('KV Cache Size (MB)')
    plt.ylabel('Time to First Token (s)')
    plt.title('KV Cache Loading Time Comparison')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_comparison.png'), dpi=300)
    
    # Plot compression ratio if available
    if compression_ratios:
        plt.figure(figsize=(10, 6))
        plt.plot(sizes, compression_ratios, 'o-', color='green')
        plt.xscale('log')
        plt.xlabel('KV Cache Size (MB)')
        plt.ylabel('Compression Ratio')
        plt.title('KV Cache Compression Ratio')
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'compression_ratio.png'), dpi=300)
    
    # Plot speedup if available
    if compressed_times:
        speedups = [b/c for b, c in zip(baseline_times, compressed_times)]
        plt.figure(figsize=(10, 6))
        plt.plot(sizes, speedups, 'o-', color='purple')
        plt.xscale('log')
        plt.axhline(y=1.0, color='r', linestyle='--')
        plt.xlabel('KV Cache Size (MB)')
        plt.ylabel('Speedup Factor (Baseline / Compressed)')
        plt.title('Speedup from Compression')
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'speedup.png'), dpi=300)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KV Cache Compression Benchmark")
    parser.add_argument("--model", type=str, default="distilgpt2", help="Model name")
    parser.add_argument("--autoencoder", type=str, help="Path to trained autoencoder")
    parser.add_argument("--latent_dim", type=int, default=16, help="Latent dimension of autoencoder")
    parser.add_argument("--sizes", type=float, nargs="+", default=[1, 10, 100, 1000, 3000], 
                        help="KV cache sizes in MB to benchmark")
    parser.add_argument("--output", type=str, default="results", help="Output directory for results")
    
    args = parser.parse_args()
    
    # Run benchmark
    results = run_benchmark(
        model_name=args.model,
        autoencoder_path=args.autoencoder,
        latent_dim=args.latent_dim,
        sizes=args.sizes,
        output_dir=args.output
    ) 