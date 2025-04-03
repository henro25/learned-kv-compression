#!/usr/bin/env python3
"""
Module Name: run_experiments.py
Description: Script to run the complete KV Cache compression workflow.
This script performs:
1. Training autoencoders with different latent dimensions
2. Running benchmarks on various KV cache sizes
3. Creating visualizations of results
Author: AI Assistant
Date: 2023-04-03
"""

import os
import argparse
import subprocess
import time
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Run KV Cache Compression Experiments")
    parser.add_argument("--model", type=str, default="distilgpt2", 
                        help="Model name to use for experiments")
    parser.add_argument("--latent_dims", type=int, nargs="+", default=[8, 16, 32], 
                        help="Latent dimensions to test")
    parser.add_argument("--cache_sizes", type=float, nargs="+", 
                        default=[1, 10, 100, 1000, 3000],
                        help="KV cache sizes in MB to test")
    parser.add_argument("--num_epochs", type=int, default=5, 
                        help="Number of epochs for training")
    parser.add_argument("--num_train_texts", type=int, default=1000, 
                        help="Number of training texts to use")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip training and use existing models")
    parser.add_argument("--output_dir", type=str, default="experiment_results",
                        help="Directory to save results")
    return parser.parse_args()

def train_autoencoder(model_name, latent_dim, num_epochs, num_train_texts, output_dir):
    """Train an autoencoder with the specified parameters"""
    model_dir = os.path.join(output_dir, f"{model_name}_latent{latent_dim}")
    os.makedirs(model_dir, exist_ok=True)
    
    cmd = [
        "python", "-m", "src.dictionary_learning.train",
        "--name", model_name,
        "--latent_dim", str(latent_dim),
        "--num_epochs", str(num_epochs),
        "--num_train_texts", str(num_train_texts),
        "--output_dir", model_dir
    ]
    
    print(f"\n{'='*80}")
    print(f"Training autoencoder with latent_dim={latent_dim}")
    print(f"{'='*80}")
    print(" ".join(cmd))
    
    subprocess.run(cmd)
    
    return os.path.join(model_dir, "autoencoder_best.pth")

def run_benchmark(model_name, autoencoder_path, latent_dim, cache_sizes, output_dir):
    """Run benchmarks with the trained autoencoder"""
    result_dir = os.path.join(output_dir, f"benchmark_{model_name}_latent{latent_dim}")
    os.makedirs(result_dir, exist_ok=True)
    
    cmd = [
        "python", "-m", "src.inference.benchmark",
        "--model", model_name,
        "--autoencoder", autoencoder_path,
        "--latent_dim", str(latent_dim),
        "--sizes"
    ] + [str(s) for s in cache_sizes] + [
        "--output", result_dir
    ]
    
    print(f"\n{'='*80}")
    print(f"Running benchmark with latent_dim={latent_dim}")
    print(f"{'='*80}")
    print(" ".join(cmd))
    
    subprocess.run(cmd)
    
    return result_dir

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Record experiment start time
    start_time = time.time()
    
    # Keep track of results for different latent dimensions
    model_results = []
    
    for latent_dim in args.latent_dims:
        # Train autoencoder (unless skipped)
        if not args.skip_training:
            model_path = train_autoencoder(
                model_name=args.model,
                latent_dim=latent_dim,
                num_epochs=args.num_epochs,
                num_train_texts=args.num_train_texts,
                output_dir=args.output_dir
            )
        else:
            # Use existing model if skipping training
            model_dir = os.path.join(args.output_dir, f"{args.model}_latent{latent_dim}")
            model_path = os.path.join(model_dir, "autoencoder_best.pth")
            if not os.path.exists(model_path):
                print(f"Warning: Model {model_path} not found. Skipping latent_dim={latent_dim}.")
                continue
        
        # Run benchmarks
        result_dir = run_benchmark(
            model_name=args.model,
            autoencoder_path=model_path,
            latent_dim=latent_dim,
            cache_sizes=args.cache_sizes,
            output_dir=args.output_dir
        )
        
        model_results.append((latent_dim, result_dir))
    
    # Calculate total runtime
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Print summary of experiments
    print(f"\n{'='*80}")
    print(f"Experiment Summary")
    print(f"{'='*80}")
    print(f"Model: {args.model}")
    print(f"Latent dimensions tested: {args.latent_dims}")
    print(f"KV cache sizes tested: {args.cache_sizes} MB")
    print(f"Total runtime: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print(f"Results saved to: {args.output_dir}")
    
    for latent_dim, result_dir in model_results:
        print(f"- Latent dim {latent_dim}: {result_dir}")
    
    print(f"{'='*80}")

if __name__ == "__main__":
    main() 