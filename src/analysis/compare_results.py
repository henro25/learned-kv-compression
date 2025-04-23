"""
Module Name: compare_results.py
Description: Script to compare and visualize results from multiple KV cache compression experiments.
This generates comparative plots of time to first token, compression ratio, and speedup
across different latent dimensions and cache sizes.
Author: AI Assistant
Date: 2023-04-03
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_results(result_dirs):
    """
    Load benchmark results from multiple directories.
    
    Args:
        result_dirs (list): List of result directory paths
        
    Returns:
        pandas.DataFrame: Combined results for analysis
    """
    all_data = []
    
    for result_dir in result_dirs:
        dir_path = Path(result_dir)
        result_file = dir_path / "benchmark_results.json"
        
        if not result_file.exists():
            print(f"Warning: Results file not found in {result_dir}")
            continue
        
        with open(result_file, 'r') as f:
            results = json.load(f)
        
        # Skip files that do not have the expected benchmarks section
        if "benchmarks" not in results:
            print(f"Warning: 'benchmarks' not found in {result_file}; skipping this directory.")
            continue
        
        # Extract model info (fallback to config if not directly present)
        model_name = results.get("model") or results.get("config", {}).get("model_name") or results.get("config", {}).get("name")
        latent_dim = results.get("latent_dim") or results.get("config", {}).get("latent_dim")
        # Skip if essential metadata is missing
        if model_name is None or latent_dim is None:
            print(f"Warning: 'model' or 'latent_dim' missing in {result_file}; skipping this directory.")
            continue
        
        # Process each benchmark
        for size_mb, data in results["benchmarks"].items():
            record = {
                "model": model_name,
                "latent_dim": latent_dim,
                "cache_size_mb": float(size_mb),
                "actual_size_mb": data["actual_size_mb"]
            }
            
            # Handle new results format with avg and std_dev
            if isinstance(data["times"]["baseline"], dict):
                record["baseline_time"] = data["times"]["baseline"]["avg"]
                record["baseline_std"] = data["times"]["baseline"]["std_dev"]
            else:
                # Handle older format for backward compatibility
                record["baseline_time"] = data["times"]["baseline"]
                record["baseline_std"] = 0.0
            
            # Add compressed data if available
            if "compressed" in data["times"]:
                if isinstance(data["times"]["compressed"], dict):
                    record["compressed_time"] = data["times"]["compressed"]["avg"]
                    record["compressed_std"] = data["times"]["compressed"]["std_dev"]
                else:
                    # Handle older format for backward compatibility
                    record["compressed_time"] = data["times"]["compressed"]
                    record["compressed_std"] = 0.0
                
                # Calculate speedup if not already provided
                if "speedup" in data:
                    record["speedup"] = data["speedup"]
                else:
                    record["speedup"] = record["baseline_time"] / record["compressed_time"]
            
            if "compression_ratio" in data:
                record["compression_ratio"] = data["compression_ratio"]
            
            all_data.append(record)
    
    # Convert to DataFrame and sort
    if not all_data:
        raise ValueError("No valid results found in the provided directories")
    
    df = pd.DataFrame(all_data)
    df = df.sort_values(["latent_dim", "cache_size_mb"])
    
    return df

def plot_time_comparison(df, output_dir):
    """Plot time to first token comparison across different latent dimensions"""
    plt.figure(figsize=(12, 8))
    
    # Plot baseline as reference
    baseline_data = df.drop_duplicates("cache_size_mb")[["cache_size_mb", "baseline_time", "baseline_std"]]
    plt.errorbar(baseline_data["cache_size_mb"], baseline_data["baseline_time"], 
             yerr=baseline_data["baseline_std"],
             fmt='k--', linewidth=2, label='Baseline (No Compression)', capsize=4)
    
    # Plot compressed times for each latent dimension
    for latent_dim in sorted(df["latent_dim"].unique()):
        subset = df[df["latent_dim"] == latent_dim]
        if "compressed_time" in subset.columns:
            plt.errorbar(subset["cache_size_mb"], subset["compressed_time"], 
                     yerr=subset["compressed_std"],
                     fmt='o-', linewidth=2, label=f'Latent Dim = {latent_dim}', capsize=4)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('KV Cache Size (MB)', fontsize=14)
    plt.ylabel('Time to First Token (s)', fontsize=14)
    plt.title('KV Cache Loading Time Comparison', fontsize=16)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'time_comparison.png'), dpi=300)
    
    # Also create a bar chart for the largest cache size
    largest_size = df["cache_size_mb"].max()
    largest_df = df[df["cache_size_mb"] == largest_size]
    
    plt.figure(figsize=(10, 6))
    
    # Create bars for baseline and each latent dimension
    bar_positions = np.arange(len(largest_df["latent_dim"].unique()) + 1)
    bar_width = 0.6
    
    # Add baseline bar
    baseline_row = largest_df.iloc[0]
    plt.bar(bar_positions[0], baseline_row["baseline_time"], bar_width, 
            label='Baseline', color='gray', yerr=baseline_row["baseline_std"], capsize=4)
    
    # Add bars for each latent dimension
    for i, latent_dim in enumerate(sorted(largest_df["latent_dim"].unique())):
        subset = largest_df[largest_df["latent_dim"] == latent_dim]
        if "compressed_time" in subset.columns:
            plt.bar(bar_positions[i+1], subset["compressed_time"].iloc[0], bar_width,
                   label=f'Latent Dim = {latent_dim}', yerr=subset["compressed_std"].iloc[0], capsize=4)
    
    plt.xlabel('Method', fontsize=14)
    plt.ylabel('Time to First Token (s)', fontsize=14)
    plt.title(f'Loading Time for {largest_size}MB KV Cache', fontsize=16)
    plt.xticks(bar_positions, ['Baseline'] + [f'Dim={dim}' for dim in sorted(largest_df["latent_dim"].unique())])
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bar_comparison_largest.png'), dpi=300)

def plot_compression_ratio(df, output_dir):
    """Plot compression ratio across different latent dimensions"""
    if "compression_ratio" not in df.columns:
        print("Warning: Compression ratio data not available")
        return
    
    plt.figure(figsize=(12, 8))
    
    for latent_dim in sorted(df["latent_dim"].unique()):
        subset = df[df["latent_dim"] == latent_dim]
        plt.plot(subset["cache_size_mb"], subset["compression_ratio"], 
                 'o-', linewidth=2, label=f'Latent Dim = {latent_dim}')
    
    plt.xscale('log')
    plt.xlabel('KV Cache Size (MB)', fontsize=14)
    plt.ylabel('Compression Ratio', fontsize=14)
    plt.title('KV Cache Compression Ratio', fontsize=16)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'compression_ratio.png'), dpi=300)
    
    # Create a bar chart of compression ratios for the middle cache size
    sizes = sorted(df["cache_size_mb"].unique())
    middle_size = sizes[len(sizes)//2]
    middle_df = df[df["cache_size_mb"] == middle_size]
    
    plt.figure(figsize=(10, 6))
    bar_positions = np.arange(len(middle_df["latent_dim"].unique()))
    bar_width = 0.6
    
    for i, latent_dim in enumerate(sorted(middle_df["latent_dim"].unique())):
        subset = middle_df[middle_df["latent_dim"] == latent_dim]
        plt.bar(bar_positions[i], subset["compression_ratio"].iloc[0], bar_width,
               label=f'Latent Dim = {latent_dim}')
    
    plt.xlabel('Latent Dimension', fontsize=14)
    plt.ylabel('Compression Ratio', fontsize=14)
    plt.title(f'Compression Ratio for {middle_size}MB KV Cache', fontsize=16)
    plt.xticks(bar_positions, [f'Dim={dim}' for dim in sorted(middle_df["latent_dim"].unique())])
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bar_compression_ratio.png'), dpi=300)

def plot_speedup(df, output_dir):
    """Plot speedup across different latent dimensions"""
    if "speedup" not in df.columns:
        print("Warning: Speedup data not available")
        return
    
    plt.figure(figsize=(12, 8))
    
    for latent_dim in sorted(df["latent_dim"].unique()):
        subset = df[df["latent_dim"] == latent_dim]
        plt.plot(subset["cache_size_mb"], subset["speedup"], 
                 'o-', linewidth=2, label=f'Latent Dim = {latent_dim}')
    
    plt.xscale('log')
    plt.axhline(y=1.0, color='r', linestyle='--', label='Break-even')
    plt.xlabel('KV Cache Size (MB)', fontsize=14)
    plt.ylabel('Speedup Factor (Baseline / Compressed)', fontsize=14)
    plt.title('Speedup from Compression', fontsize=16)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'speedup.png'), dpi=300)

def plot_tradeoff(df, output_dir):
    """Plot the tradeoff between compression ratio and speedup"""
    if "speedup" not in df.columns or "compression_ratio" not in df.columns:
        print("Warning: Speedup or compression ratio data not available")
        return
    
    # Create a new figure
    plt.figure(figsize=(12, 8))
    
    # Use different markers for different cache sizes
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*']
    
    # Plot each cache size with a different marker
    for i, size in enumerate(sorted(df["cache_size_mb"].unique())):
        subset = df[df["cache_size_mb"] == size]
        marker = markers[i % len(markers)]
        
        scatter = plt.scatter(subset["compression_ratio"], subset["speedup"], 
                 s=100, marker=marker, label=f'{size} MB')
        
        # Add latent dimension annotations
        for _, row in subset.iterrows():
            plt.annotate(f'dim={int(row["latent_dim"])}', 
                        (row["compression_ratio"], row["speedup"]),
                        xytext=(5, 5), textcoords='offset points')
    
    plt.axhline(y=1.0, color='r', linestyle='--', label='Break-even')
    plt.xlabel('Compression Ratio', fontsize=14)
    plt.ylabel('Speedup Factor', fontsize=14)
    plt.title('Compression vs. Speedup Tradeoff', fontsize=16)
    plt.grid(True, ls="--", alpha=0.5)
    plt.legend(fontsize=12, title="Cache Size")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tradeoff.png'), dpi=300)
    
    # Create a heatmap of speedup for different latent dims and cache sizes
    pivot_df = df.pivot_table(
        index="latent_dim", 
        columns="cache_size_mb", 
        values="speedup"
    )
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_df, annot=True, fmt=".2f", cmap="viridis", 
                cbar_kws={'label': 'Speedup Factor'})
    plt.title('Speedup by Latent Dimension and Cache Size', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'speedup_heatmap.png'), dpi=300)

def generate_report(df, output_dir):
    """Generate a summary report of the results"""
    report_path = os.path.join(output_dir, "comparison_report.md")
    
    with open(report_path, 'w') as f:
        f.write("# KV Cache Compression Benchmark Report\n\n")
        
        f.write("## Overview\n\n")
        f.write(f"- Model: {df['model'].iloc[0]}\n")
        f.write(f"- Latent dimensions tested: {sorted(df['latent_dim'].unique())}\n")
        f.write(f"- Cache sizes tested: {sorted(df['cache_size_mb'].unique())} MB\n\n")
        
        f.write("## Key Findings\n\n")
        
        # Find best configurations
        if "speedup" in df.columns:
            best_speedup = df.loc[df["speedup"].idxmax()]
            f.write(f"- Best speedup: **{best_speedup['speedup']:.2f}x** (Latent dim={best_speedup['latent_dim']}, Cache size={best_speedup['cache_size_mb']} MB)\n")
        
        if "compression_ratio" in df.columns:
            best_compression = df.loc[df["compression_ratio"].idxmax()]
            f.write(f"- Best compression ratio: **{best_compression['compression_ratio']:.2f}x** (Latent dim={best_compression['latent_dim']}, Cache size={best_compression['cache_size_mb']} MB)\n\n")
        
        # Add table of results for the largest cache size
        largest_size = df["cache_size_mb"].max()
        largest_df = df[df["cache_size_mb"] == largest_size].sort_values("latent_dim")
        
        f.write(f"## Results for {largest_size} MB Cache\n\n")
        f.write("| Latent Dim | Baseline Time (s) | Baseline Std | Compressed Time (s) | Compressed Std | Speedup | Compression Ratio |\n")
        f.write("|------------|------------------|--------------|---------------------|----------------|---------|-------------------|\n")
        
        for _, row in largest_df.iterrows():
            baseline = row["baseline_time"]
            baseline_std = row.get("baseline_std", "N/A")
            compressed = row.get("compressed_time", "N/A")
            compressed_std = row.get("compressed_std", "N/A") 
            speedup = row.get("speedup", "N/A")
            compression = row.get("compression_ratio", "N/A")
            
            f.write(f"| {int(row['latent_dim'])} | {baseline:.4f} | {baseline_std if isinstance(baseline_std, str) else baseline_std:.4f} | {compressed if isinstance(compressed, str) else compressed:.4f} | {compressed_std if isinstance(compressed_std, str) else compressed_std:.4f} | {speedup if isinstance(speedup, str) else speedup:.2f} | {compression if isinstance(compression, str) else compression:.2f} |\n")
        
        f.write("\n## Conclusions\n\n")
        f.write("- The optimal latent dimension depends on the size of the KV cache and the importance of speed vs. compression.\n")
        
        # Add specific conclusions based on the data
        if "speedup" in df.columns and "compression_ratio" in df.columns:
            # Check if any configuration is both fast and compresses well
            good_configs = df[(df["speedup"] > 1) & (df["compression_ratio"] > 3)]
            if not good_configs.empty:
                best_overall = good_configs.loc[good_configs["speedup"].idxmax()]
                f.write(f"- **Recommended configuration:** Latent dim={int(best_overall['latent_dim'])} provides good balance with {best_overall['compression_ratio']:.2f}x compression and {best_overall['speedup']:.2f}x speedup.\n")
        
        f.write("\n## Visualizations\n\n")
        f.write("See the generated PNG files for detailed comparisons:\n")
        f.write("- `time_comparison.png`: Comparison of time to first token\n")
        f.write("- `compression_ratio.png`: Achieved compression ratios\n")
        f.write("- `speedup.png`: Speedup factors\n")
        f.write("- `tradeoff.png`: Compression vs. speedup tradeoff\n")
    
    print(f"Report generated at {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Compare KV Cache Compression Results")
    parser.add_argument("--results", type=str, nargs="+", required=True,
                        help="Directories containing benchmark results")
    parser.add_argument("--output", type=str, default="comparison_results",
                        help="Directory to save comparison results")
    
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    
    # Load and combine results
    print("Loading results from:", args.results)
    df = load_results(args.results)
    
    # Generate plots
    print("Generating comparison plots...")
    plot_time_comparison(df, args.output)
    plot_compression_ratio(df, args.output)
    plot_speedup(df, args.output)
    plot_tradeoff(df, args.output)
    
    # Generate report
    print("Generating summary report...")
    generate_report(df, args.output)
    
    print(f"Comparison complete. Results saved to {args.output}")

if __name__ == "__main__":
    main() 