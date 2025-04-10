#!/usr/bin/env python3
"""
Module Name: benchmark_hybrid.py
Description: Benchmark script for Hybrid KV compression (autoencoder + quantization)
"""

import os
import argparse
import json
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from src.compression.hybrid_press import HybridKVPress

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Hybrid KV Cache Compression")
    parser.add_argument("--model", type=str, default="distilgpt2", help="Model name")
    parser.add_argument("--autoencoder", type=str, required=True, help="Path to autoencoder checkpoint")
    parser.add_argument("--quant_bits", type=int, default=8, help="Quantization bits (4-8)")
    parser.add_argument("--prompt_length", type=int, default=512, help="Length of input prompt")
    parser.add_argument("--gen_length", type=int, default=128, help="Number of tokens to generate")
    parser.add_argument("--output", type=str, default="hybrid_results", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    return parser.parse_args()

def measure_memory_and_time(func, *args, **kwargs):
    """Measure memory usage and execution time of a function"""
    # Record starting memory
    torch.cuda.reset_peak_memory_stats()
    start_mem = torch.cuda.memory_allocated()
    
    # Time the function
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    
    # Record peak memory
    peak_mem = torch.cuda.max_memory_allocated()
    mem_used = peak_mem - start_mem
    
    return result, end_time - start_time, mem_used

def generate_text(model, tokenizer, input_ids, gen_length, use_hybrid=False, hybrid_press=None):
    """Generate text with or without hybrid compression"""
    if use_hybrid:
        with torch.no_grad(), hybrid_press(model):
            outputs = model.generate(
                input_ids, 
                max_new_tokens=gen_length,
                do_sample=True,
                use_cache=True
            )
    else:
        with torch.no_grad():
            outputs = model.generate(
                input_ids, 
                max_new_tokens=gen_length,
                do_sample=True,
                use_cache=True
            )
    
    return outputs

def run_benchmark(args):
    """Run the benchmark comparing baseline vs. hybrid compression"""
    # Prepare output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load model and tokenizer
    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(args.model).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Create hybrid compression
    hybrid_press = HybridKVPress(
        autoencoder_path=args.autoencoder,
        quantization_bits=args.quant_bits,
        device=args.device
    )
    
    # Load benchmark dataset
    print("Loading benchmark dataset")
    wiki_dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    benchmark_texts = [text for text in wiki_dataset["test"]["text"] if len(text.split()) > 50][:10]
    
    results = {
        "model": args.model,
        "autoencoder": args.autoencoder,
        "quant_bits": args.quant_bits,
        "prompt_length": args.prompt_length,
        "gen_length": args.gen_length,
        "compression_ratio": hybrid_press.compression_ratio,
        "benchmarks": []
    }
    
    for i, text in enumerate(benchmark_texts):
        print(f"Running benchmark {i+1}/{len(benchmark_texts)}")
        
        # Prepare inputs
        input_text = text[:args.prompt_length]
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(args.device)
        
        # Baseline generation
        print("  Baseline generation...")
        baseline_outputs, baseline_time, baseline_mem = measure_memory_and_time(
            generate_text, model, tokenizer, input_ids, args.gen_length
        )
        
        # Hybrid compressed generation
        print("  Hybrid compressed generation...")
        hybrid_outputs, hybrid_time, hybrid_mem = measure_memory_and_time(
            generate_text, model, tokenizer, input_ids, args.gen_length, 
            use_hybrid=True, hybrid_press=hybrid_press
        )
        
        # Process outputs
        baseline_text = tokenizer.decode(baseline_outputs[0])
        hybrid_text = tokenizer.decode(hybrid_outputs[0])
        
        # Calculate speedup and memory reduction
        speedup = baseline_time / hybrid_time
        mem_reduction = baseline_mem / hybrid_mem
        
        # Save results
        result = {
            "baseline_time": baseline_time,
            "hybrid_time": hybrid_time,
            "baseline_mem_mb": baseline_mem / (1024 * 1024),
            "hybrid_mem_mb": hybrid_mem / (1024 * 1024),
            "speedup": speedup,
            "memory_reduction": mem_reduction,
            "baseline_text": baseline_text,
            "hybrid_text": hybrid_text,
        }
        
        results["benchmarks"].append(result)
        
        print(f"  Speedup: {speedup:.2f}x, Memory reduction: {mem_reduction:.2f}x")
    
    # Calculate averages
    avg_speedup = np.mean([b["speedup"] for b in results["benchmarks"]])
    avg_mem_reduction = np.mean([b["memory_reduction"] for b in results["benchmarks"]])
    
    results["avg_speedup"] = avg_speedup
    results["avg_mem_reduction"] = avg_mem_reduction
    
    print(f"Average speedup: {avg_speedup:.2f}x, Average memory reduction: {avg_mem_reduction:.2f}x")
    
    # Save results
    results_file = output_dir / "hybrid_benchmark_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create plots
    create_plots(results, output_dir)
    
    return results

def create_plots(results, output_dir):
    """Create visualizations from benchmark results"""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Extract data
    speedups = [b["speedup"] for b in results["benchmarks"]]
    mem_reductions = [b["memory_reduction"] for b in results["benchmarks"]]
    
    # Bar plot of speedup and memory reduction
    plt.figure(figsize=(10, 6))
    plt.bar([0], [np.mean(speedups)], width=0.4, label="Speedup")
    plt.bar([0.5], [np.mean(mem_reductions)], width=0.4, label="Memory Reduction")
    
    plt.errorbar([0], [np.mean(speedups)], yerr=[np.std(speedups)], fmt='o', color='black')
    plt.errorbar([0.5], [np.mean(mem_reductions)], yerr=[np.std(mem_reductions)], fmt='o', color='black')
    
    plt.title(f"Hybrid Compression Performance\n"
              f"({results['quant_bits']}-bit quantization, {results['compression_ratio']:.2f}x compression ratio)")
    plt.xticks([0.25], ["Improvement Factor"])
    plt.ylabel("Factor (higher is better)")
    plt.ylim(bottom=0)
    plt.legend()
    plt.savefig(plots_dir / "hybrid_performance.png", dpi=300)
    
    # Memory usage comparison
    baseline_mem = [b["baseline_mem_mb"] for b in results["benchmarks"]]
    hybrid_mem = [b["hybrid_mem_mb"] for b in results["benchmarks"]]
    
    plt.figure(figsize=(8, 6))
    plt.boxplot([baseline_mem, hybrid_mem], labels=["Baseline", "Hybrid"])
    plt.title("GPU Memory Usage")
    plt.ylabel("Memory Usage (MB)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(plots_dir / "memory_usage.png", dpi=300)
    
    # Timing comparison
    baseline_time = [b["baseline_time"] for b in results["benchmarks"]]
    hybrid_time = [b["hybrid_time"] for b in results["benchmarks"]]
    
    plt.figure(figsize=(8, 6))
    plt.boxplot([baseline_time, hybrid_time], labels=["Baseline", "Hybrid"])
    plt.title("Generation Time")
    plt.ylabel("Time (seconds)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(plots_dir / "generation_time.png", dpi=300)

if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args) 