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
import re
from pathlib import Path
from typing import List, Dict, Any, Union

def parse_args():
    parser = argparse.ArgumentParser(description="Run KV Cache Compression Experiments")
    parser.add_argument("--model", type=str, nargs="+", default=["distilgpt2"], 
                        help="Model name(s) to use for experiments")
    parser.add_argument("--latent_dims", type=int, nargs="+", default=[8, 16, 32], 
                        help="Latent dimensions to test")
    parser.add_argument("--cache_sizes", type=float, nargs="+", 
                        default=[1, 10, 100, 1000],
                        help="KV cache sizes in MB to test")
    parser.add_argument("--num_epochs", type=int, nargs="+", default=[5], 
                        help="Number of epochs for training")
    parser.add_argument("--num_train_texts", type=int, nargs="+", default=[100], 
                        help="Number of training texts to use")
    parser.add_argument("--batch_size", type=int, nargs="+", default=[64],
                        help="Batch size for compression operations")
    parser.add_argument("--num_runs", type=int, nargs="+", default=[5],
                        help="Number of runs for timing statistics")
    parser.add_argument("--dtype", type=str, nargs="+", default=["f32"],
                        help="Data type for training and inference")
    parser.add_argument("--buffer_size", type=int, default=512,
                        help="Maximum sequence length to use in the buffer (to reduce memory usage)")
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
                     num_train_texts: int, output_dir: str, cfg: Dict[str, Any], data_type: str) -> str:
    """Train an autoencoder with the specified parameters."""
    safe_model_name = model_name.replace("/", "_")
    model_dir = os.path.join(output_dir, f"{safe_model_name}_latent{latent_dim}")
    os.makedirs(model_dir, exist_ok=True)
    
    print("MODEL DIR: ", model_dir)
    print("CFG: ", cfg)
    # Update config with training parameters
    train_cfg = cfg.copy()
    train_cfg.update({
        "name": model_name,  # Ensure model name is passed to trainer
        "latent_dim": latent_dim,
        "num_epochs": num_epochs,
        "num_train_texts": num_train_texts,
        "output_dir": model_dir,
        "dtype": data_type,  # Use the data_type parameter instead of cfg["dtype"]
        "buffer_size": cfg["buffer_size"]  # Pass buffer_size from config
    })
    
    # Save updated config
    config_path = os.path.join(model_dir, "train_config.json")
    with open(config_path, "w") as f:
        json.dump(train_cfg, f, indent=2)
    
    cmd = [
        "python", "-m", "src.dictionary_learning.train",
        "--config", config_path,
        "--name", model_name,
        "--latent_dim", str(latent_dim),
        "--num_epochs", str(num_epochs),
        "--num_train_texts", str(num_train_texts),
        "--output_dir", model_dir,
        "--dtype", data_type,  # Use the data_type parameter instead of cfg["dtype"]
        "--buffer_size", str(cfg["buffer_size"])  # Pass buffer_size from config
    ]
    
    print(f"\n{'='*80}")
    print(f"Training autoencoder with model={model_name}, latent_dim={latent_dim}, epochs={num_epochs}, train_texts={num_train_texts}, dtype={data_type}")
    print(f"{'='*80}")
    print(" ".join(cmd))
    
    subprocess.run(cmd)
    
    return os.path.join(model_dir, "autoencoder_final.pth")

def run_benchmark(model_name: str, autoencoder_path: str, latent_dim: int, 
                 cache_sizes: List[float], batch_size: int, num_runs: int, 
                 output_dir: str, cfg: Dict[str, Any], data_type: str, buffer_size: int) -> str:
    """Run benchmarks with the trained autoencoder."""
    safe_model_name = model_name.replace("/", "_")
    result_dir = os.path.join(output_dir, f"benchmark_{safe_model_name}_latent{latent_dim}_batch{batch_size}_runs{num_runs}")
    os.makedirs(result_dir, exist_ok=True)
    
    # Update config with benchmark parameters
    benchmark_cfg = cfg.copy()
    benchmark_cfg.update({
        "model_name": model_name,
        "latent_dim": latent_dim,
        "batch_size": batch_size,
        "num_runs": num_runs,
        "cache_sizes": cache_sizes,
        "data_type": data_type,
        "buffer_size": buffer_size  # Pass buffer_size from config
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
        "--config", config_path,
        "--dtype", data_type,
        "--buffer_size", str(buffer_size)  # Pass buffer_size as command-line arg
    ]
    
    print(f"\n{'='*80}")
    print(f"Running benchmark with model={model_name}, latent_dim={latent_dim}, batch_size={batch_size}, num_runs={num_runs}, data_type={data_type}")
    print(f"{'='*80}")
    print(" ".join(cmd))
    
    subprocess.run(cmd)
    
    return result_dir

def ensure_list(value):
    """Convert a single value to a list if it's not already a list."""
    if not isinstance(value, list):
        return [value]
    return value

def parse_space_separated_values(values, target_type=str):
    """Parse space-separated values from command line to a list of the target type."""
    if isinstance(values, list) and len(values) == 1 and isinstance(values[0], str) and " " in values[0]:
        # Handle space-separated string as list
        return [target_type(v) for v in values[0].split()]
    return ensure_list(values)

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Handle different input formats (single value, list, space-separated string)
    models = parse_space_separated_values(args.model)
    latent_dims = parse_space_separated_values(args.latent_dims, int)
    cache_sizes = parse_space_separated_values(args.cache_sizes, float)
    num_epochs_list = parse_space_separated_values(args.num_epochs, int)
    num_train_texts_list = parse_space_separated_values(args.num_train_texts, int)
    batch_sizes = parse_space_separated_values(args.batch_size, int)
    num_runs_list = parse_space_separated_values(args.num_runs, int)
    data_types = parse_space_separated_values(args.dtype, str)
    
    # Print parsed arguments for debugging
    print("Parsed arguments:")
    print(f"Models: {models}")
    print(f"Latent dimensions: {latent_dims}")
    print(f"Cache sizes: {cache_sizes}")
    print(f"Epochs: {num_epochs_list}")
    print(f"Training texts: {num_train_texts_list}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Number of runs: {num_runs_list}")
    print(f"Data types: {data_types}")
    print(f"Buffer size: {args.buffer_size}")
    
    # Load base configuration
    cfg = load_config(args.config)
    
    # Add buffer_size to config
    cfg["buffer_size"] = args.buffer_size
    
    # Record experiment start time
    start_time = time.time()
    
    # Keep track of all experiment results
    all_results = []
    
    # Run experiments for all combinations of parameters
    for model_name in models:
        for latent_dim in latent_dims:
            for num_epochs in num_epochs_list:
                for num_train_texts in num_train_texts_list:
                    # For each trained model, create a unique identifier
                    safe_model_name = model_name.replace("/", "_")
                    model_id = f"{safe_model_name}_latent{latent_dim}_epochs{num_epochs}_texts{num_train_texts}"
                    model_output_dir = os.path.join(args.output_dir, model_id)
                    os.makedirs(model_output_dir, exist_ok=True)
                    
                    # Train autoencoder (unless skipped)
                    if not args.skip_training:
                        model_path = train_autoencoder(
                            model_name=model_name,
                            latent_dim=latent_dim,
                            num_epochs=num_epochs,
                            num_train_texts=num_train_texts,
                            output_dir=args.output_dir,
                            cfg=cfg,
                            data_type=data_types[0]
                        )
                    else:
                        # Use existing model if skipping training
                        safe_model_name = model_name.replace("/", "_")
                        model_dir = os.path.join(args.output_dir, f"{safe_model_name}_latent{latent_dim}")
                        model_path = os.path.join(model_dir, "autoencoder_final.pth")
                        if not os.path.exists(model_path):
                            print(f"Warning: Model {model_path} not found. Skipping this configuration.")
                            continue
                    
                    # Run benchmarks with various batch sizes and run counts
                    for batch_size in batch_sizes:
                        for num_runs in num_runs_list:
                            result_dir = run_benchmark(
                                model_name=model_name,
                                autoencoder_path=model_path,
                                latent_dim=latent_dim,
                                cache_sizes=cache_sizes,
                                batch_size=batch_size,
                                num_runs=num_runs,
                                output_dir=args.output_dir,
                                cfg=cfg,
                                data_type=data_types[0],
                                buffer_size=args.buffer_size
                            )
                            
                            all_results.append({
                                "model": model_name,
                                "latent_dim": latent_dim,
                                "num_epochs": num_epochs,
                                "num_train_texts": num_train_texts,
                                "batch_size": batch_size,
                                "num_runs": num_runs,
                                "result_dir": result_dir,
                                "data_type": data_types[0]
                            })
    
    # Calculate total runtime
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Save overall experiment summary
    summary_path = os.path.join(args.output_dir, "experiment_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "total_runtime_seconds": total_time,
            "models": models,
            "latent_dims": latent_dims,
            "num_epochs": num_epochs_list,
            "num_train_texts": num_train_texts_list,
            "batch_sizes": batch_sizes,
            "num_runs": num_runs_list,
            "cache_sizes": cache_sizes,
            "results": all_results,
            "data_type": data_types[0]
        }, f, indent=2)
    
    # Print summary of experiments
    print(f"\n{'='*80}")
    print(f"Experiment Summary")
    print(f"{'='*80}")
    print(f"Models tested: {models}")
    print(f"Data type: {data_types[0]}")
    print(f"Latent dimensions tested: {latent_dims}")
    print(f"Epochs tested: {num_epochs_list}")
    print(f"Training texts tested: {num_train_texts_list}")
    print(f"Batch sizes tested: {batch_sizes}")
    print(f"Number of runs tested: {num_runs_list}")
    print(f"KV cache sizes tested: {cache_sizes} MB")
    print(f"Total runtime: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print(f"Results saved to: {args.output_dir}")
    print(f"Total experiments: {len(all_results)}")
    print(f"{'='*80}")
    print("Experiment Results:")
    for result in all_results:
        print(f"- Model: {result['model']}, Latent dim: {result['latent_dim']}, " +
              f"Epochs: {result['num_epochs']}, Texts: {result['num_train_texts']}, " +
              f"Batch: {result['batch_size']}, Runs: {result['num_runs']}")
        print(f"  Result dir: {result['result_dir']}")
    
    print(f"{'='*80}")

if __name__ == "__main__":
    main() 
