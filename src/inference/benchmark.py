"""
Module Name: benchmark.py
Description: Benchmark script for KV Cache compression. This script runs experiments
for different KV cache sizes with and without compression, and generates visualizations
of the results.
Author: AI Assistant
Date: 2023-04-03
"""

import os
import json
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from src.models.autoencoder import Autoencoder
import torch.nn as nn
import seaborn as sns
import pandas as pd
from typing import List, Dict

from src.inference.inference import KVCacheInference

def calculate_perplexity(model, tokenizer, texts, max_length=1024):
    """Calculate perplexity for a list of texts."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for text in tqdm(texts, desc="Calculating perplexity"):
            # Tokenize and prepare input
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs["attention_mask"].to(model.device)
            
            # Get model outputs
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Calculate loss for each token
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Mask out padding tokens
            mask = attention_mask[..., 1:].contiguous().view(-1)
            loss = loss * mask
            
            total_loss += loss.sum().item()
            total_tokens += mask.sum().item()
    
    # Calculate perplexity
    perplexity = torch.exp(torch.tensor(total_loss / total_tokens)).item()
    return perplexity

def compress_kv_cache(past_key_values, autoencoder):
    """Compress the KV cache using the autoencoder."""
    if past_key_values is None:
        return None
        
    compressed_cache = []
    for layer in past_key_values:
        keys, values = layer
        # Compress keys and values
        k_compressed, _ = autoencoder(keys.reshape(-1, keys.size(-1)))
        v_compressed, _ = autoencoder(values.reshape(-1, values.size(-1)))
        
        # Reshape back to original dimensions
        k_compressed = k_compressed.reshape(keys.shape)
        v_compressed = v_compressed.reshape(values.shape)
        compressed_cache.append((k_compressed, v_compressed))
    return compressed_cache

def evaluate_with_compressed_cache(model, tokenizer, autoencoder, texts, max_length=1024):
    """Evaluate model using compressed KV cache."""
    model.eval()
    autoencoder.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for text in tqdm(texts, desc="Evaluating with compressed cache"):
            # Tokenize and prepare input
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs["attention_mask"].to(model.device)
            
            # Initialize KV cache
            past_key_values = None
            
            # Process one token at a time
            for i in range(input_ids.size(1) - 1):
                # Skip if this is a padding token
                if not attention_mask[0, i+1].item():
                    continue
                    
                # Get current token and attention mask
                current_input = input_ids[:, i:i+1]
                current_mask = attention_mask[:, i:i+1]
                
                # Get model outputs with compressed cache
                outputs = model(
                    current_input,
                    attention_mask=current_mask,
                    past_key_values=past_key_values,
                    use_cache=True
                )
                
                # Compress and store KV cache
                past_key_values = compress_kv_cache(outputs.past_key_values, autoencoder)
                
                # Calculate loss for next token
                next_token_logits = outputs.logits[..., -1, :]
                next_token_id = input_ids[:, i+1]
                loss_fct = nn.CrossEntropyLoss(reduction='none')
                loss = loss_fct(next_token_logits, next_token_id)
                
                total_loss += loss.item()
                total_tokens += 1
    
    # Calculate perplexity
    perplexity = torch.exp(torch.tensor(total_loss / total_tokens)).item()
    return perplexity

def evaluate_longbench(model, tokenizer, autoencoder, cfg):
    """Evaluate on LongBench dataset."""
    # Define the LongBench subsets to evaluate
    longbench_subsets = [
        'narrativeqa',
        'hotpotqa',
        '2wikimqa',
        'musique',
        'dureader'
    ]
    
    results = {
        "baseline": {},
        "compressed": {}
    }
    
    # Evaluate on each subset
    for subset in longbench_subsets:
        print(f"\nEvaluating on {subset}...")
        try:
            # Load the specific subset
            dataset = load_dataset("THUDM/LongBench", subset)
            texts = dataset["test"]["input"]
            
            # Calculate baseline perplexity
            baseline_ppl = calculate_perplexity(model, tokenizer, texts, cfg["max_seq_len"])
            results["baseline"][subset] = baseline_ppl
            
            # Calculate perplexity with compressed cache
            compressed_ppl = evaluate_with_compressed_cache(model, tokenizer, autoencoder, texts, cfg["max_seq_len"])
            results["compressed"][subset] = compressed_ppl
            
            print(f"{subset} - Baseline PPL: {baseline_ppl:.2f}, Compressed PPL: {compressed_ppl:.2f}")
        except Exception as e:
            print(f"Error evaluating {subset}: {str(e)}")
            results["baseline"][subset] = None
            results["compressed"][subset] = None
    
    return results

def plot_results(results, output_dir):
    """Create and save visualization plots."""
    # Create output directory for plots
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Perplexity comparison bar plot
    plt.figure(figsize=(12, 6))
    subsets = list(results["longbench_results"]["baseline"].keys())
    baseline_ppl = [results["longbench_results"]["baseline"][s] for s in subsets]
    compressed_ppl = [results["longbench_results"]["compressed"][s] for s in subsets]
    
    x = np.arange(len(subsets))
    width = 0.35
    
    plt.bar(x - width/2, baseline_ppl, width, label='Baseline')
    plt.bar(x + width/2, compressed_ppl, width, label='Compressed Cache')
    
    plt.xlabel('Dataset Subset')
    plt.ylabel('Perplexity')
    plt.title('Perplexity Comparison: Baseline vs Compressed Cache')
    plt.xticks(x, subsets, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "perplexity_comparison.png"))
    plt.close()
    
    # 2. Perplexity ratio heatmap
    perplexity_ratio = np.array(compressed_ppl) / np.array(baseline_ppl)
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        perplexity_ratio.reshape(1, -1),
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        xticklabels=subsets,
        yticklabels=["Ratio"],
        cbar_kws={'label': 'Compressed/Baseline Perplexity Ratio'}
    )
    plt.title("Perplexity Ratio Across Datasets")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "perplexity_ratio_heatmap.png"))
    plt.close()
    
    # 3. Performance degradation plot
    degradation = (np.array(compressed_ppl) - np.array(baseline_ppl)) / np.array(baseline_ppl) * 100
    plt.figure(figsize=(12, 6))
    plt.bar(subsets, degradation)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.xlabel('Dataset Subset')
    plt.ylabel('Performance Degradation (%)')
    plt.title('Performance Degradation with Compressed Cache')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "performance_degradation.png"))
    plt.close()

def save_results(results, output_dir):
    """Save evaluation results to files."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results as JSON
    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Save results as CSV
    df = pd.DataFrame({
        'Dataset': list(results["longbench_results"]["baseline"].keys()) + ['WikiText'],
        'Baseline_PPL': list(results["longbench_results"]["baseline"].values()) + [results["baseline_perplexity"]],
        'Compressed_PPL': list(results["longbench_results"]["compressed"].values()) + [results["compressed_perplexity"]]
    })
    df['Ratio'] = df['Compressed_PPL'] / df['Baseline_PPL']
    df['Degradation (%)'] = (df['Compressed_PPL'] - df['Baseline_PPL']) / df['Baseline_PPL'] * 100
    
    csv_path = os.path.join(output_dir, "evaluation_results.csv")
    df.to_csv(csv_path, index=False)
    
    return results_path, csv_path

def run_benchmark(
    model_name: str,
    autoencoder_path: str,
    latent_dim: int,
    cache_sizes: List[float],
    batch_size: int,
    num_runs: int,
    output_dir: str,
    cfg: Dict,
    data_type: str,
    buffer_size: int = 512
) -> str:
    """Run benchmarks with the trained autoencoder."""
    safe_model_name = model_name.replace("/", "_")
    result_dir = os.path.join(output_dir, f"benchmark_{safe_model_name}_latent{latent_dim}")
    os.makedirs(result_dir, exist_ok=True)
    
    # Update buffer size in config
    cfg["buffer_size"] = buffer_size
    
    # Set device
    device = torch.device(cfg["device"])
    
    # Convert data_type string to torch dtype
    if data_type == "bf16":
        dtype = torch.bfloat16
    elif data_type == "fp16" or data_type == "f16":
        dtype = torch.float16
    else:
        dtype = torch.float32
    
    # Update config with the dtype we're using
    cfg["dtype"] = data_type
    
    print(f"Using dtype: {dtype}, buffer size: {buffer_size}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map={"": device},
        output_hidden_states=True,
        output_attentions=True,
        use_cache=True
    )
    
    # Load trained autoencoder
    autoencoder = Autoencoder(input_dim=cfg["head_dim"], latent_dim=latent_dim).to(device)
    autoencoder.load_state_dict(torch.load(autoencoder_path))
    
    # Load WikiText evaluation texts
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    eval_texts = [text for text in dataset["test"]["text"] if text.strip()][:cfg["num_eval_texts"]]
    
    # Calculate WikiText perplexity
    print("\nCalculating WikiText perplexity...")
    baseline_ppl = calculate_perplexity(model, tokenizer, eval_texts, cfg["max_seq_len"])
    print(f"Baseline perplexity: {baseline_ppl:.2f}")
    
    compressed_ppl = evaluate_with_compressed_cache(model, tokenizer, autoencoder, eval_texts, cfg["max_seq_len"])
    print(f"Compressed cache perplexity: {compressed_ppl:.2f}")
    
    # Evaluate on LongBench
    print("\nEvaluating on LongBench...")
    longbench_results = evaluate_longbench(model, tokenizer, autoencoder, cfg)
    
    # Prepare results
    results = {
        "baseline_perplexity": baseline_ppl,
        "compressed_perplexity": compressed_ppl,
        "longbench_results": longbench_results,
        "config": cfg,
        "buffer_size": buffer_size
    }
    
    # Save results and create visualizations
    results_path, csv_path = save_results(results, result_dir)
    plot_results(results, result_dir)
    
    print("\nBenchmarking complete!")
    print(f"Results saved in {result_dir}")
    print(f"- Evaluation results (JSON): {results_path}")
    print(f"- Evaluation results (CSV): {csv_path}")
    print(f"- Plots saved in {os.path.join(result_dir, 'plots')}")
    
    return result_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--autoencoder", type=str, required=True, help="Path to autoencoder model")
    parser.add_argument("--latent_dim", type=int, required=True, help="Latent dimension")
    parser.add_argument("--cache_sizes", type=float, nargs="+", required=True, help="Cache sizes to test")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--num_runs", type=int, default=5, help="Number of runs for timing")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--dtype", type=str, default="f32", help="Data type for training and inference")
    parser.add_argument("--buffer_size", type=int, default=512, help="Buffer size for training")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        cfg = json.load(f)
    
    run_benchmark(
        args.model,
        args.autoencoder,
        args.latent_dim,
        args.cache_sizes,
        args.batch_size,
        args.num_runs,
        args.output,
        cfg,
        args.dtype,
        args.buffer_size
    ) 