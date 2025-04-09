"""
Module Name: plot_comprehensive.py
Description: This module generates comprehensive plots for the autoencoder tradeoff experiments,
including plots that align with the original proposal.
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
        depth = i+2  # Since we removed asym_1_1, we start from 2
        plt.annotate(f'asym_{depth}_1', (x, y), xytext=(5, 5), textcoords='offset points')
    
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
        depth = i+2  # Since we removed asym_1_1, we start from 2
        plt.annotate(f'asym_{depth}_1', (x, y), xytext=(5, 5), textcoords='offset points')
    
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
        depth = i+2  # Since we removed asym_1_1, we start from 2
        plt.annotate(f'asym_{depth}_1', (x, y), xytext=(5, 5), textcoords='offset points')
    
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
        depth = i+2  # Since we removed asym_1_1, we start from 2
        plt.annotate(f'asym_{depth}_1', (x, y), xytext=(5, 5), textcoords='offset points')
    
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
        depth = i+2  # Since we removed asym_1_1, we start from 2
        plt.annotate(f'asym_{depth}_1', (x, y), xytext=(5, 5), textcoords='offset points')
    
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
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Extract data
    names = []
    times = []
    stds = []
    
    for name, data in sorted(results.items()):
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

def plot_training_effect(results):
    """Plot the effect of training epochs on reconstruction quality."""
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Extract data for symmetric models
    depths = []
    epochs = []
    quality = []
    
    for name, data in results.items():
        if name.startswith("sym_"):
            depth = int(name.split("_")[1])
            depths.append(depth)
            epochs.append(data["train_epochs"])
            quality.append(data["reconstruction_mse"])
    
    # Create a colormap based on training epochs
    cmap = plt.cm.viridis
    norm = plt.Normalize(min(epochs), max(epochs))
    colors = cmap(norm(epochs))
    
    # Plot
    scatter = plt.scatter(depths, quality, s=150, c=colors, marker='o')
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Training Epochs', fontsize=12)
    
    # Add labels
    for i, (x, y, e) in enumerate(zip(depths, quality, epochs)):
        plt.annotate(f'sym_{x} ({e} epochs)', (x, y), xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel("Model Depth", fontsize=12)
    plt.ylabel("Reconstruction MSE", fontsize=12)
    plt.title("Effect of Training on Reconstruction Quality", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("experiment_results/training_effect.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Generate all plots."""
    results = load_results()
    
    # Create plots
    plot_compression_ratio_vs_quality(results)
    plot_encoder_size_vs_decompression_time(results)
    plot_encoder_size_vs_quality(results)
    plot_speed_quality_tradeoff(results)
    plot_quality_vs_size(results)
    plot_inference_time_comparison(results)
    plot_training_effect(results)
    
    print("All plots generated successfully!")

if __name__ == "__main__":
    main() 
