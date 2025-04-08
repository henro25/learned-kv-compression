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
import json
from pathlib import Path
from typing import List, Dict, Any

def parse_args():
    parser = argparse.ArgumentParser(description="Run KV Cache Compression Experiments")
    parser.add_argument("--model", type=str, default="distilgpt2", 
                        help="Model name to use for experiments")
    parser.add_argument("--latent_dims", type=int, nargs="+", default=[8, 16, 32], 
                        help="Latent dimensions to test")
    parser.add_argument("--cache_sizes", type=float, nargs="+", 
                        default=[1, 10, 100, 1000],
                        help="KV cache sizes in MB to test")
    parser.add_argument("--num_epochs", type=int, default=5, 
                        help="Number of epochs for training")
    parser.add_argument("--num_train_texts", type=int, default=100, 
                        help="Number of training texts to use")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for compression operations")
    parser.add_argument("--num_runs", type=int, default=5,
                        help="Number of runs for timing statistics")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip training and use existing models")
    parser.add_argument("--output_dir", type=str, default="experiment_results",
                        help="Directory to save results")
    parser.add_argument("--config", type=str, default="src/configs/default_config.json",
                        help="Path to config file")
    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)

def train_autoencoder(model_name: str, latent_dim: int, num_epochs: int, 
                     num_train_texts: int, output_dir: str, cfg: Dict[str, Any]) -> str:
    """Train an autoencoder with the specified parameters."""
    model_dir = os.path.join(output_dir, f"{model_name}_latent{latent_dim}")
    os.makedirs(model_dir, exist_ok=True)
    
    print("MODEL DIR: ", model_dir)
    
    # Update config with training parameters
    train_cfg = cfg.copy()
    train_cfg.update({
        "latent_dim": latent_dim,
        "num_epochs": num_epochs,
        "num_train_texts": num_train_texts,
        "output_dir": model_dir
    })
    
    # Save updated config
    config_path = os.path.join(model_dir, "train_config.json")
    with open(config_path, "w") as f:
        json.dump(train_cfg, f, indent=2)
    
    cmd = [
        "python", "-m", "src.dictionary_learning.train",
        "--config", config_path,
        "--output_dir", model_dir
    ]
    
    print(f"\n{'='*80}")
    print(f"Training autoencoder with latent_dim={latent_dim}")
    print(f"{'='*80}")
    print(" ".join(cmd))
    
    subprocess.run(cmd)
    
    return os.path.join(model_dir, "autoencoder_final.pth")

def run_benchmark(model_name: str, autoencoder_path: str, latent_dim: int, 
                 cache_sizes: List[float], batch_size: int, num_runs: int, 
                 output_dir: str, cfg: Dict[str, Any]) -> str:
    """Run benchmarks with the trained autoencoder."""
    result_dir = os.path.join(output_dir, f"benchmark_{model_name}_latent{latent_dim}")
    os.makedirs(result_dir, exist_ok=True)
    
    # Update config with benchmark parameters
    benchmark_cfg = cfg.copy()
    benchmark_cfg.update({
        "model_name": model_name,
        "latent_dim": latent_dim,
        "batch_size": batch_size,
        "num_runs": num_runs,
        "cache_sizes": cache_sizes
    })
    
    # Save updated config
    config_path = os.path.join(result_dir, "benchmark_config.json")
    with open(config_path, "w") as f:
        json.dump(benchmark_cfg, f, indent=2)
    
    cmd = [
        "python", "-m", "src.inference.benchmark",
        "--model", model_name,
        "--autoencoder", autoencoder_path,
        "--latent_dim", str(latent_dim),
        "--cache_sizes"
    ] + [str(s) for s in cache_sizes] + [
        "--batch_size", str(batch_size),
        "--num_runs", str(num_runs),
        "--output", result_dir,
        "--config", config_path
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
    
    # Load base configuration
    cfg = load_config(args.config)
    
    # Record experiment start time
    start_time = time.time()
    
    # Keep track of results for different latent dimensions
    model_results = []
    
    for latent_dim in args.latent_dims:
        
        print("Output directory for training is: ", args.output_dir)
        
        # Train autoencoder (unless skipped)
        if not args.skip_training:
            model_path = train_autoencoder(
                model_name=args.model,
                latent_dim=latent_dim,
                num_epochs=args.num_epochs,
                num_train_texts=args.num_train_texts,
                output_dir=args.output_dir,
                cfg=cfg
            )
        else:
            # Use existing model if skipping training
            model_path = os.path.join(args.output_dir, "autoencoder_final.pth")
            if not os.path.exists(model_path):
                print(f"Warning: Model {model_path} not found. Skipping latent_dim={latent_dim}.")
                continue
        
        # Run benchmarks
        result_dir = run_benchmark(
            model_name=args.model,
            autoencoder_path=model_path,
            latent_dim=latent_dim,
            cache_sizes=args.cache_sizes,
            batch_size=args.batch_size,
            num_runs=args.num_runs,
            output_dir=args.output_dir,
            cfg=cfg
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
    print(f"Batch size: {args.batch_size}")
    print(f"Number of runs for timing: {args.num_runs}")
    print(f"Total runtime: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print(f"Results saved to: {args.output_dir}")
    
    for latent_dim, result_dir in model_results:
        print(f"- Latent dim {latent_dim}: {result_dir}")
    
    print(f"{'='*80}")

if __name__ == "__main__":
    main() 
