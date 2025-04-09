"""
Module Name: plot_tradeoffs.py
Description: This module generates plots for the autoencoder tradeoff experiments
Author: Ben Choi
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
    """Plot decompression time vs model depth."""
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Extract symmetric model data
    sym_depths = []
    sym_times = []
    sym_stds = []
    for name, data in results.items():
        if name.startswith("sym_"):
            depth = int(name.split("_")[1])
            sym_depths.append(depth)
            sym_times.append(data["mean_inference_time_ms"])
            sym_stds.append(data["std_inference_time_ms"])
    
    # Extract asymmetric model data
    asym_depths = []
    asym_times = []
    asym_stds = []
    for name, data in results.items():
        if name.startswith("asym_"):
            depth = int(name.split("_")[1])
            asym_depths.append(depth)
            asym_times.append(data["mean_inference_time_ms"])
            asym_stds.append(data["std_inference_time_ms"])
    
    # Plot symmetric models
    plt.errorbar(sym_depths, sym_times, yerr=sym_stds, 
                fmt='o-', label='Symmetric', linewidth=2, markersize=8)
    
    # Plot asymmetric models
    plt.errorbar(asym_depths, asym_times, yerr=asym_stds,
                fmt='s--', label='Asymmetric', linewidth=2, markersize=8)
    
    plt.xlabel("Model Depth", fontsize=12)
    plt.ylabel("Decompression Time (ms)", fontsize=12)
    plt.title("Decompression Time vs Model Depth", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("experiment_results/decompression_time_vs_depth.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_quality_vs_depth(results):
    """Plot reconstruction quality vs model depth."""
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Extract symmetric model data
    sym_depths = []
    sym_mse = []
    for name, data in results.items():
        if name.startswith("sym_"):
            depth = int(name.split("_")[1])
            sym_depths.append(depth)
            sym_mse.append(data["reconstruction_mse"])
    
    # Extract asymmetric model data
    asym_depths = []
    asym_mse = []
    for name, data in results.items():
        if name.startswith("asym_"):
            depth = int(name.split("_")[1])
            asym_depths.append(depth)
            asym_mse.append(data["reconstruction_mse"])
    
    # Plot symmetric models
    plt.plot(sym_depths, sym_mse, 'o-', label='Symmetric', linewidth=2, markersize=8)
    
    # Plot asymmetric models
    plt.plot(asym_depths, asym_mse, 's--', label='Asymmetric', linewidth=2, markersize=8)
    
    plt.xlabel("Model Depth", fontsize=12)
    plt.ylabel("Reconstruction MSE", fontsize=12)
    plt.title("Reconstruction Quality vs Model Depth", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("experiment_results/quality_vs_depth.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_parameter_distribution(results):
    """Plot parameter distribution between encoder and decoder."""
    plt.figure(figsize=(10, 6))
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
    
    plt.bar(x - width/2, encoder_params, width, label='Encoder')
    plt.bar(x + width/2, decoder_params, width, label='Decoder')
    
    plt.xlabel("Model Configuration", fontsize=12)
    plt.ylabel("Number of Parameters", fontsize=12)
    plt.title("Parameter Distribution", fontsize=14)
    plt.xticks(x, names, rotation=45)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("experiment_results/parameter_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_quality_vs_decompression_time(results):
    """Plot reconstruction quality vs decompression time."""
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Extract symmetric model data
    sym_times = []
    sym_mse = []
    for name, data in results.items():
        if name.startswith("sym_"):
            sym_times.append(data["mean_inference_time_ms"])
            sym_mse.append(data["reconstruction_mse"])
    
    # Extract asymmetric model data
    asym_times = []
    asym_mse = []
    for name, data in results.items():
        if name.startswith("asym_"):
            asym_times.append(data["mean_inference_time_ms"])
            asym_mse.append(data["reconstruction_mse"])
    
    # Plot symmetric models
    plt.scatter(sym_times, sym_mse, s=100, label='Symmetric')
    for i, (x, y) in enumerate(zip(sym_times, sym_mse)):
        plt.annotate(f'sym_{i+1}', (x, y), xytext=(5, 5), textcoords='offset points')
    
    # Plot asymmetric models
    plt.scatter(asym_times, asym_mse, s=100, label='Asymmetric')
    for i, (x, y) in enumerate(zip(asym_times, asym_mse)):
        plt.annotate(f'asym_{i+1}_1', (x, y), xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel("Decompression Time (ms)", fontsize=12)
    plt.ylabel("Reconstruction MSE", fontsize=12)
    plt.title("Quality vs Decompression Time Tradeoff", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("experiment_results/quality_vs_decompression_time.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Generate all plots."""
    results = load_results()
    
    # Create plots
    plot_decompression_time_vs_depth(results)
    plot_quality_vs_depth(results)
    plot_parameter_distribution(results)
    plot_quality_vs_decompression_time(results)
    
    print("Plots generated successfully!")

if __name__ == "__main__":
    main() 