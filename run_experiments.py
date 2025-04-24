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
from typing import List, Dict, Any, Union, Optional

def parse_args():
    parser = argparse.ArgumentParser(description="Run KV Cache Compression Experiments")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to experiment configuration file")
    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)

def train_autoencoder(model_name: str, latent_dim: int, learning_rate: float, num_epochs: int, 
                     num_train_texts: int, output_dir: str, cfg: Dict[str, Any]) -> str:
    """Train an autoencoder with the specified parameters."""
    safe_model_name = model_name.replace("/", "_")
    model_dir = os.path.join(output_dir, f"{safe_model_name}_latent{latent_dim}_lr{learning_rate}")
    os.makedirs(model_dir, exist_ok=True)
    
    print("MODEL DIR: ", model_dir)
    
    # Update config with training parameters
    train_cfg = cfg.copy()
    train_cfg.update({
        "name": model_name,  # Ensure model name is passed to trainer
        "latent_dim": latent_dim,
        "lr": learning_rate,
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
        "--config", config_path
    ]
    
    print(f"\n{'='*80}")
    print(f"Training autoencoder with model={model_name}, latent_dim={latent_dim}, lr={learning_rate}, epochs={num_epochs}, train_texts={num_train_texts}")
    print(f"{'='*80}")
    print(" ".join(cmd))
    
    subprocess.run(cmd)
    
    return os.path.join(model_dir, "autoencoders_final.pth")

def run_benchmark(model_name: str, autoencoder_path: str, latent_dim: int, 
                 cache_sizes: List[float], batch_size: int, num_runs: int, 
                 output_dir: str, cfg: Dict[str, Any], learning_rate: Optional[float] = None, quantization_bits: Optional[int] = None) -> str:
    """Run benchmarks with the trained autoencoder."""
    safe_model_name = model_name.replace("/", "_")
    lr_suffix = f"_lr{learning_rate}" if learning_rate is not None else ""
    quant_suffix = f"_quant{quantization_bits}" if quantization_bits is not None else ""
    result_dir = os.path.join(output_dir, f"benchmark_{safe_model_name}_latent{latent_dim}{lr_suffix}{quant_suffix}_batch{batch_size}_runs{num_runs}")
    os.makedirs(result_dir, exist_ok=True)
    
    # Update config with benchmark parameters
    # NOTE: THIS IS A HACK TO GET IT TO WORK
    benchmark_cfg = cfg.copy()
    benchmark_cfg.update({
        "model_name": model_name,
        "name": model_name,  # Include both for compatibility
        "latent_dim": latent_dim,
        "batch_size": batch_size,
        "num_runs": num_runs,
        "cache_sizes": cache_sizes,
        "autoencoder_path": autoencoder_path,
        "output_dir": result_dir,  # Set result_dir as output_dir
        "learning_rate": learning_rate,
        "quantization_bits": quantization_bits
    })

    # Save updated config
    config_path = os.path.join(result_dir, "benchmark_config.json")
    with open(config_path, "w") as f:
        json.dump(benchmark_cfg, f, indent=2)
    
    cmd = [
        "python", "-m", "src.inference.benchmark",
        "--config", config_path
    ]
    
    print(f"\n{'='*80}")
    print(f"Running benchmark with model={model_name}, latent_dim={latent_dim}, batch_size={batch_size}, num_runs={num_runs}")
    print(f"{'='*80}")
    print(" ".join(cmd))
    
    subprocess.run(cmd)
    
    return result_dir

def ensure_list(value):
    """Convert a single value to a list if it's not already a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]

def main():
    args = parse_args()
    
    # Load experiment configuration
    experiment_cfg = load_config(args.config)
    
    # Extract experiment parameters
    models = ensure_list(experiment_cfg.get("models", ["distilgpt2"]))
    latent_dims = ensure_list(experiment_cfg.get("latent_dims", [8, 16, 32]))
    learning_rates = ensure_list(experiment_cfg.get("learning_rates", [1e-3]))
    cache_sizes = ensure_list(experiment_cfg.get("cache_sizes", [1, 10, 100, 1000]))
    quant_bits_list = ensure_list(experiment_cfg.get("quantization_bits", [8]))
    num_epochs_list = ensure_list(experiment_cfg.get("num_epochs", [5]))
    num_train_texts_list = ensure_list(experiment_cfg.get("num_train_texts", [100]))
    batch_sizes = ensure_list(experiment_cfg.get("batch_sizes", [64]))
    num_runs_list = ensure_list(experiment_cfg.get("num_runs", [5]))
    output_dir = experiment_cfg.get("output_dir", "experiment_results")
    
    # Print memory optimization parameters if present
    if "gradient_accumulation_steps" in experiment_cfg:
        print(f"Gradient accumulation steps: {experiment_cfg['gradient_accumulation_steps']}")
    if "memory_efficient" in experiment_cfg:
        print(f"Memory efficient mode: {experiment_cfg['memory_efficient']}")
    if "chunk_size_mb" in experiment_cfg:
        print(f"Chunk size (MB): {experiment_cfg['chunk_size_mb']}")
    if "buffer_size" in experiment_cfg:
        print(f"Buffer size: {experiment_cfg['buffer_size']}")
    if "buffer_mult" in experiment_cfg:
        print(f"Buffer multiplier: {experiment_cfg['buffer_mult']}")
    if "max_seq_len" in experiment_cfg:
        print(f"Max sequence length: {experiment_cfg['max_seq_len']}")
    
    # Convert output_dir to absolute path and create it
    output_dir = os.path.abspath(output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Print experiment configuration for debugging
    print("Running experiments with the following configuration:")
    print(f"Models: {models}")
    print(f"Latent dimensions: {latent_dims}")
    print(f"Learning rates: {learning_rates}")
    print(f"Cache sizes: {cache_sizes}")
    print(f"Quantization bits: {quant_bits_list}")
    print(f"Epochs: {num_epochs_list}")
    print(f"Training texts: {num_train_texts_list}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Number of runs: {num_runs_list}")
    print(f"Output directory: {output_dir}")
    
    # Load base configuration
    cfg = load_config(args.config)
    
    # Record experiment start time
    start_time = time.time()
    
    # Keep track of all experiment results
    all_results = []
    
    # Run experiments for all combinations of parameters
    for model_name in models:
        # Create a model-specific output directory
        model_output_dir = os.path.join(output_dir, model_name.replace("/", "_"))
        os.makedirs(model_output_dir, exist_ok=True)
        
        for latent_dim in latent_dims:
            for learning_rate in learning_rates:
                for num_epochs in num_epochs_list:
                    for num_train_texts in num_train_texts_list:
                        # For each trained model, create a unique identifier
                        safe_model_name = model_name.replace("/", "_")
                        model_id = f"{safe_model_name}_latent{latent_dim}_lr{learning_rate}_epochs{num_epochs}_texts{num_train_texts}"
                        model_output_path = os.path.join(model_output_dir, model_id)
                        os.makedirs(model_output_path, exist_ok=True)
                        
                        # Train autoencoder (unless skipped)
                        if not experiment_cfg.get("skip_training", False):
                            model_path = train_autoencoder(
                                model_name=model_name,
                                latent_dim=latent_dim,
                                learning_rate=learning_rate,
                                num_epochs=num_epochs,
                                num_train_texts=num_train_texts,
                                output_dir=model_output_dir,
                                cfg=experiment_cfg
                            )
                            print(f"Training finished for {model_id}. Checking for model file...")
                            time.sleep(5) # Add a pause
                            check_cmd = f"ls -l {model_path}"
                            print(f"Running check: {check_cmd}")
                            subprocess.run(check_cmd, shell=True) # Check if file exists
                        else:
                            # Use existing model if skipping training
                            model_dir = os.path.join(model_output_dir, f"{safe_model_name}_latent{latent_dim}_lr{learning_rate}")
                            model_path = os.path.join(model_dir, "autoencoders_final.pth")
                            if not os.path.exists(model_path):
                                print(f"Warning: Model {model_path} not found. Skipping this configuration.")
                                continue
                        
                        # Run benchmarks for each quantization bit depth
                        for quant_bits in quant_bits_list:
                            for batch_size in batch_sizes:
                                for num_runs in num_runs_list:
                                    # Run benchmark for the specified cache sizes
                                    result_dir = run_benchmark(
                                        model_name=model_name,
                                        autoencoder_path=model_path,
                                        latent_dim=latent_dim,
                                        cache_sizes=cache_sizes,
                                        batch_size=batch_size,
                                        num_runs=num_runs,
                                        output_dir=model_output_dir,
                                        cfg=experiment_cfg,
                                        learning_rate=learning_rate,
                                        quantization_bits=quant_bits
                                    )
                                    # Store absolute path for result_dir
                                    result_dir = os.path.abspath(result_dir)
                                    # Append experiment result with cache sizes
                                    all_results.append({
                                        "model": model_name,
                                        "latent_dim": latent_dim,
                                        "learning_rate": learning_rate,
                                        "num_epochs": num_epochs,
                                        "num_train_texts": num_train_texts,
                                        "quantization_bits": quant_bits,
                                        "batch_size": batch_size,
                                        "num_runs": num_runs,
                                        "cache_sizes": cache_sizes,
                                        "result_dir": result_dir
                                    })
    
    # Calculate total runtime
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Save overall experiment summary
    summary_path = os.path.join(output_dir, "experiment_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "total_runtime_seconds": total_time,
            "config_file": args.config,
            "models": models,
            "latent_dims": latent_dims,
            "learning_rates": learning_rates,
            "quantization_bits": quant_bits_list,
            "num_epochs": num_epochs_list,
            "num_train_texts": num_train_texts_list,
            "batch_sizes": batch_sizes,
            "num_runs": num_runs_list,
            "cache_sizes": cache_sizes,
            "results": all_results
        }, f, indent=2)
    
    # Print summary of experiments
    print(f"\n{'='*80}")
    print(f"Experiment Summary")
    print(f"{'='*80}")
    print(f"Models tested: {models}")
    print(f"Latent dimensions tested: {latent_dims}")
    print(f"Learning rates tested: {learning_rates}")
    print(f"Quantization bits tested: {quant_bits_list}")
    print(f"Epochs tested: {num_epochs_list}")
    print(f"Training texts tested: {num_train_texts_list}")
    print(f"Batch sizes tested: {batch_sizes}")
    print(f"Number of runs tested: {num_runs_list}")
    print(f"KV cache sizes tested: {cache_sizes} MB")
    print(f"Total runtime: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print(f"Results saved to: {output_dir}")
    print(f"Total experiments: {len(all_results)}")
    print(f"{'='*80}")
    print("Experiment Results:")
    for result in all_results:
        print(
            f"- Model: {result['model']}, Latent dim: {result['latent_dim']}, Cache sizes: {result['cache_sizes']} MB, "
            f"LR: {result['learning_rate']}, Epochs: {result['num_epochs']}, Texts: {result['num_train_texts']}, "
            f"Quantization bits: {result['quantization_bits']}, Batch: {result['batch_size']}, Runs: {result['num_runs']}"
        )
        print(f"  Result dir: {result['result_dir']}")
    
    print(f"{'='*80}")

if __name__ == "__main__":
    main() 
