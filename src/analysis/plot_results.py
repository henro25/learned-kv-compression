"""
Module Name: plot_results.py
Description: This module generates plots to visualize the autoencoder experiment results
for KV cache compression using DistilGPT-2 data.
"""

import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
import numpy as np

def plot_tradeoffs(results_file="experiment_results/autoencoder_tradeoffs.json"):
    """Generate plots from the experiment results."""
    
    # Load results from file
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Convert results to a pandas DataFrame for easier plotting
    df = pd.DataFrame.from_dict(results, orient='index')
    
    # Add model type column and sort by configuration
    df['type'] = ['Symmetric' if 'sym' in idx else 'Asymmetric' for idx in df.index]
    df['config_name'] = df.index
    df['depth'] = df.index.map(lambda x: int(x.split('_')[1]))  # Extract depth number
    df = df.sort_values(['type', 'depth'])
    
    # Set the style
    plt.style.use('default')
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = [12, 6]
    plt.rcParams['figure.dpi'] = 300
    
    # Create output directory
    output_dir = Path("experiment_results")
    output_dir.mkdir(exist_ok=True)

    # --- Plot 1: Decompression Time Comparison ---
    plt.figure()
    ax = sns.barplot(data=df, x='config_name', y='mean_inference_time_ms', 
                    hue='type', dodge=False, alpha=0.7)
    plt.errorbar(x=range(len(df)), y=df['mean_inference_time_ms'], 
                yerr=df['std_inference_time_ms'], fmt='none', c='black', capsize=5)
    plt.title('Decompression Time by Model Configuration (DistilGPT-2)', fontsize=14, pad=20)
    plt.xlabel('Model Configuration', fontsize=12)
    plt.ylabel('Mean Decompression Time (ms)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Architecture Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_dir / "decompression_time_comparison.png")
    plt.close()

    # --- Plot 2: Decompression Time vs Model Depth ---
    plt.figure()
    g = sns.lineplot(data=df, x='depth', y='mean_inference_time_ms',
                     hue='type', style='type', markers=True, dashes=False)
    
    # Add error bands
    for t in df['type'].unique():
        type_data = df[df['type'] == t]
        plt.fill_between(type_data['depth'], 
                        type_data['mean_inference_time_ms'] - type_data['std_inference_time_ms'],
                        type_data['mean_inference_time_ms'] + type_data['std_inference_time_ms'],
                        alpha=0.2)
    
    plt.title('Decompression Time vs Model Depth (DistilGPT-2)', fontsize=14, pad=20)
    plt.xlabel('Model Depth', fontsize=12)
    plt.ylabel('Mean Decompression Time (ms)', fontsize=12)
    plt.legend(title='Architecture Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_dir / "decompression_time_vs_depth.png")
    plt.close()

    # --- Plot 3: Parameter Distribution ---
    plt.figure()
    df_params = pd.DataFrame({
        'Encoder': df['encoder_params'],
        'Decoder': df['decoder_params']
    }, index=df.index)
    
    ax = df_params.plot(kind='bar', stacked=True, alpha=0.7)
    plt.title('Parameter Distribution between Encoder and Decoder', fontsize=14, pad=20)
    plt.xlabel('Model Configuration', fontsize=12)
    plt.ylabel('Number of Parameters', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "parameter_distribution.png")
    plt.close()

    # --- Plot 4: Quality vs Depth ---
    plt.figure()
    g = sns.lineplot(data=df, x='depth', y='reconstruction_mse',
                     hue='type', style='type', markers=True, dashes=False)
    
    plt.title('Reconstruction Quality vs Model Depth (DistilGPT-2)', fontsize=14, pad=20)
    plt.xlabel('Model Depth', fontsize=12)
    plt.ylabel('Reconstruction MSE', fontsize=12)
    plt.legend(title='Architecture Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_dir / "quality_vs_depth.png")
    plt.close()

    # --- Plot 5: Decompression Time vs Quality ---
    plt.figure()
    g = sns.scatterplot(data=df, x='mean_inference_time_ms', y='reconstruction_mse',
                        hue='type', style='type', s=200)
    
    # Add labels for each point
    for idx, row in df.iterrows():
        plt.annotate(idx, (row['mean_inference_time_ms'], row['reconstruction_mse']),
                    xytext=(5, 5), textcoords='offset points')
    
    plt.title('Reconstruction Quality vs Decompression Time (DistilGPT-2)', fontsize=14, pad=20)
    plt.xlabel('Mean Decompression Time (ms)', fontsize=12)
    plt.ylabel('Reconstruction MSE', fontsize=12)
    plt.legend(title='Architecture Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_dir / "quality_vs_decompression_time.png")
    plt.close()

    print("Generated plots have been saved to the experiment_results directory:")
    print("1. decompression_time_comparison.png")
    print("2. decompression_time_vs_depth.png")
    print("3. parameter_distribution.png")
    print("4. quality_vs_depth.png")
    print("5. quality_vs_decompression_time.png")

if __name__ == "__main__":
    plot_tradeoffs() 
