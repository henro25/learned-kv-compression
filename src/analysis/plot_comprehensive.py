"""
Module Name: plot_comprehensive.py
Description: This module generates comprehensive plots for the autoencoder tradeoff experiments,
including plots that align with the original proposal.
Author: Ben Choi
Date: 2024-04-09
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def load_results():
    """Load experiment results from JSON file."""
    with open("experiment_results/autoencoder_tradeoffs.json", "r") as f:
        return json.load(f)

def plot_decompression_time_vs_depth(results):
    """Plot decompression time vs model depth as a bar chart."""
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    # Extract data
    sym_depths = []
    sym_times = []
    sym_stds = []
    asym_depths = []
    asym_times = []
    asym_stds = []
    
    for name, data in results.items():
        if name.startswith("sym_"):
            depth = int(name.split("_")[1])
            sym_depths.append(depth)
            sym_times.append(data["mean_inference_time_ms"])
            sym_stds.append(data["std_inference_time_ms"])
        elif name.startswith("asym_"):
            depth = int(name.split("_")[1])
            asym_depths.append(depth)
            asym_times.append(data["mean_inference_time_ms"])
            asym_stds.append(data["std_inference_time_ms"])
    
    # Plot
    x = np.arange(len(sym_depths) + len(asym_depths))
    width = 0.35
    
    # Plot symmetric models
    plt.bar(x[:len(sym_depths)], sym_times, width, yerr=sym_stds, 
            capsize=5, label='Symmetric', color='skyblue')
    
    # Plot asymmetric models
    plt.bar(x[len(sym_depths):], asym_times, width, yerr=asym_stds,
            capsize=5, label='Asymmetric', color='salmon')
    
    # Add labels
    labels = [f'sym_{d}' for d in sym_depths] + [f'asym_{d}_1' for d in asym_depths]
    plt.xticks(x, labels, rotation=45)
    
    plt.xlabel("Model Configuration", fontsize=12)
    plt.ylabel("Decompression Time (ms)", fontsize=12)
    plt.title("Decompression Time vs Model Depth", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("experiment_results/decompression_time_vs_depth.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_parameter_distribution(results):
    """Plot parameter distribution between encoder and decoder with logarithmic scale."""
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    # Extract data
    names = []
    encoder_params = []
    decoder_params = []
    
    for name, data in results.items():
        names.append(name)
        encoder_params.append(data["encoder_params"])
        decoder_params.append(data["decoder_params"])
    
    x = np.arange(len(names))
    width = 0.35
    
    plt.bar(x - width/2, encoder_params, width, label='Encoder', color='skyblue')
    plt.bar(x + width/2, decoder_params, width, label='Decoder', color='salmon')
    
    plt.xlabel("Model Configuration", fontsize=12)
    plt.ylabel("Number of Parameters (log scale)", fontsize=12)
    plt.title("Parameter Distribution", fontsize=14)
    plt.xticks(x, names, rotation=45)
    plt.yscale('log')  # Set logarithmic scale
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("experiment_results/parameter_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_mse_comparison(results):
    """Plot MSE comparison across all models."""
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    # Extract data
    names = []
    mse_values = []
    
    for name, data in results.items():
        names.append(name)
        mse_values.append(data["reconstruction_mse"])
    
    # Sort by MSE
    sorted_indices = np.argsort(mse_values)
    names = [names[i] for i in sorted_indices]
    mse_values = [mse_values[i] for i in sorted_indices]
    
    # Plot
    x = np.arange(len(names))
    plt.bar(x, mse_values, color='skyblue')
    
    plt.xlabel("Model Configuration", fontsize=12)
    plt.ylabel("Reconstruction MSE", fontsize=12)
    plt.title("Reconstruction MSE Comparison", fontsize=14)
    plt.xticks(x, names, rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("experiment_results/mse_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_compression_ratio_vs_quality(results):
    """Plot compression ratio vs reconstruction quality."""
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Extract data
    sym_ratios = []
    sym_quality = []
    asym_ratios = []
    asym_quality = []
    
    for name, data in results.items():
        ratio = data["compression_ratio"]
        quality = data["reconstruction_mse"]
        if name.startswith("sym_"):
            sym_ratios.append(ratio)
            sym_quality.append(quality)
        else:
            asym_ratios.append(ratio)
            asym_quality.append(quality)
    
    # Plot
    plt.scatter(sym_ratios, sym_quality, s=100, label='Symmetric', marker='o')
    plt.scatter(asym_ratios, asym_quality, s=100, label='Asymmetric', marker='s')
    
    # Add labels
    for i, (x, y) in enumerate(zip(sym_ratios, sym_quality)):
        plt.annotate(f'sym_{i+1}', (x, y), xytext=(5, 5), textcoords='offset points')
    for i, (x, y) in enumerate(zip(asym_ratios, asym_quality)):
        plt.annotate(f'asym_{i+1}_1', (x, y), xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel("Compression Ratio", fontsize=12)
    plt.ylabel("Reconstruction MSE", fontsize=12)
    plt.title("Compression Ratio vs Reconstruction Quality", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("experiment_results/compression_ratio_vs_quality.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_encoder_size_vs_decompression_time(results):
    """Plot encoder size vs decompression time."""
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Extract data
    sym_sizes = []
    sym_times = []
    asym_sizes = []
    asym_times = []
    
    for name, data in results.items():
        size = data["encoder_params"]
        time = data["mean_inference_time_ms"]
        if name.startswith("sym_"):
            sym_sizes.append(size)
            sym_times.append(time)
        else:
            asym_sizes.append(size)
            asym_times.append(time)
    
    # Plot
    plt.scatter(sym_sizes, sym_times, s=100, label='Symmetric', marker='o')
    plt.scatter(asym_sizes, asym_times, s=100, label='Asymmetric', marker='s')
    
    # Add labels
    for i, (x, y) in enumerate(zip(sym_sizes, sym_times)):
        plt.annotate(f'sym_{i+1}', (x, y), xytext=(5, 5), textcoords='offset points')
    for i, (x, y) in enumerate(zip(asym_sizes, asym_times)):
        plt.annotate(f'asym_{i+1}_1', (x, y), xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel("Encoder Parameters", fontsize=12)
    plt.ylabel("Decompression Time (ms)", fontsize=12)
    plt.title("Encoder Size vs Decompression Time", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("experiment_results/encoder_size_vs_decompression_time.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_encoder_size_vs_quality(results):
    """Plot encoder size vs reconstruction quality."""
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Extract data
    sym_sizes = []
    sym_quality = []
    asym_sizes = []
    asym_quality = []
    
    for name, data in results.items():
        size = data["encoder_params"]
        quality = data["reconstruction_mse"]
        if name.startswith("sym_"):
            sym_sizes.append(size)
            sym_quality.append(quality)
        else:
            asym_sizes.append(size)
            asym_quality.append(quality)
    
    # Plot
    plt.scatter(sym_sizes, sym_quality, s=100, label='Symmetric', marker='o')
    plt.scatter(asym_sizes, asym_quality, s=100, label='Asymmetric', marker='s')
    
    # Add labels
    for i, (x, y) in enumerate(zip(sym_sizes, sym_quality)):
        plt.annotate(f'sym_{i+1}', (x, y), xytext=(5, 5), textcoords='offset points')
    for i, (x, y) in enumerate(zip(asym_sizes, asym_quality)):
        plt.annotate(f'asym_{i+1}_1', (x, y), xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel("Encoder Parameters", fontsize=12)
    plt.ylabel("Reconstruction MSE", fontsize=12)
    plt.title("Encoder Size vs Reconstruction Quality", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("experiment_results/encoder_size_vs_quality.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_speed_quality_tradeoff(results):
    """Plot speed vs quality tradeoff."""
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Extract data
    sym_times = []
    sym_quality = []
    asym_times = []
    asym_quality = []
    
    for name, data in results.items():
        time = data["mean_inference_time_ms"]
        quality = data["reconstruction_mse"]
        if name.startswith("sym_"):
            sym_times.append(time)
            sym_quality.append(quality)
        else:
            asym_times.append(time)
            asym_quality.append(quality)
    
    # Plot
    plt.scatter(sym_times, sym_quality, s=100, label='Symmetric', marker='o')
    plt.scatter(asym_times, asym_quality, s=100, label='Asymmetric', marker='s')
    
    # Add labels
    for i, (x, y) in enumerate(zip(sym_times, sym_quality)):
        plt.annotate(f'sym_{i+1}', (x, y), xytext=(5, 5), textcoords='offset points')
    for i, (x, y) in enumerate(zip(asym_times, asym_quality)):
        plt.annotate(f'asym_{i+1}_1', (x, y), xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel("Decompression Time (ms)", fontsize=12)
    plt.ylabel("Reconstruction MSE", fontsize=12)
    plt.title("Speed vs Quality Tradeoff", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("experiment_results/speed_quality_tradeoff.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_quality_vs_size(results):
    """Plot quality vs model size."""
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Extract data
    sym_sizes = []
    sym_quality = []
    asym_sizes = []
    asym_quality = []
    
    for name, data in results.items():
        size = data["total_params"]
        quality = data["reconstruction_mse"]
        if name.startswith("sym_"):
            sym_sizes.append(size)
            sym_quality.append(quality)
        else:
            asym_sizes.append(size)
            asym_quality.append(quality)
    
    # Plot
    plt.scatter(sym_sizes, sym_quality, s=100, label='Symmetric', marker='o')
    plt.scatter(asym_sizes, asym_quality, s=100, label='Asymmetric', marker='s')
    
    # Add labels
    for i, (x, y) in enumerate(zip(sym_sizes, sym_quality)):
        plt.annotate(f'sym_{i+1}', (x, y), xytext=(5, 5), textcoords='offset points')
    for i, (x, y) in enumerate(zip(asym_sizes, asym_quality)):
        plt.annotate(f'asym_{i+1}_1', (x, y), xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel("Total Model Parameters", fontsize=12)
    plt.ylabel("Reconstruction MSE", fontsize=12)
    plt.title("Quality vs Model Size", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("experiment_results/quality_vs_size.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_inference_time_comparison(results):
    """Plot comparison of inference times."""
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    # Extract data
    names = []
    times = []
    stds = []
    
    for name, data in results.items():
        names.append(name)
        times.append(data["mean_inference_time_ms"])
        stds.append(data["std_inference_time_ms"])
    
    # Plot
    x = np.arange(len(names))
    plt.bar(x, times, yerr=stds, capsize=5)
    
    plt.xlabel("Model Configuration", fontsize=12)
    plt.ylabel("Decompression Time (ms)", fontsize=12)
    plt.title("Inference Time Comparison", fontsize=14)
    plt.xticks(x, names, rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("experiment_results/inference_time_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Generate all plots."""
    results = load_results()
    
    # Create plots
    plot_decompression_time_vs_depth(results)
    plot_parameter_distribution(results)
    plot_mse_comparison(results)
    plot_compression_ratio_vs_quality(results)
    plot_encoder_size_vs_decompression_time(results)
    plot_encoder_size_vs_quality(results)
    plot_speed_quality_tradeoff(results)
    plot_quality_vs_size(results)
    plot_inference_time_comparison(results)
    
    print("All plots generated successfully!")

if __name__ == "__main__":
    main() 