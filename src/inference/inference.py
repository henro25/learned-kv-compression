"""
Module Name: inference.py
Description: Inference module for KV Cache compression benchmarking. This module contains utilities
for running inference with LLMs, generating KV cache of different sizes, compressing and decompressing
the KV cache, and measuring performance metrics like time to first token.
Author: AI Assistant
Date: 2023-04-03
"""

import os
import time
import torch
import numpy as np
import json
import argparse
import statistics
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.models.autoencoder import Autoencoder

# Use high-performance timer
try:
    # Use time.perf_counter for higher precision timing
    timer = time.perf_counter
except AttributeError:
    # Fallback for older Python versions
    timer = time.time

class KVCacheInference:
    def __init__(self, model_name, device, autoencoder_path, latent_dim, batch_size=1024):
        self.device = device
        self.batch_size = batch_size
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map={"": device},
            output_hidden_states=True,
            output_attentions=True,
            use_cache=True
        )
        
        # Get model config
        self.num_heads = self.model.config.num_attention_heads
        self.head_dim = self.model.config.hidden_size // self.num_heads
        self.num_layers = self.model.config.num_hidden_layers
        
        # Load autoencoder
        self.autoencoder = Autoencoder(
            input_dim=self.head_dim,
            latent_dim=latent_dim
        ).to(device)
        self.autoencoder.load_state_dict(torch.load(autoencoder_path))
        self.autoencoder.eval()
    
    def generate_kv_cache(self, text, target_size_mb=1):
        """
        Generate KV Cache of approximately target_size_mb from actual text.
        Returns:
            - past_key_values: The generated KV cache
            - generated_text: The text used to generate the cache
            - actual_size_mb: The actual size of the generated cache
        """
        # Calculate how many tokens we need to reach target_size_mb
        bytes_per_token = 2 * self.num_heads * self.head_dim * 2 * self.num_layers
        target_tokens = int((target_size_mb * 1024 * 1024) / bytes_per_token)
        
        # Tokenize input text
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # Generate tokens one by one to create KV cache
        with torch.no_grad():
            past_key_values = None
            generated_tokens = 0
            generated_text = text
            
            while generated_tokens < target_tokens:
                # Get model outputs
                outputs = self.model(
                    input_ids=input_ids[:, -1:] if past_key_values else input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True
                )
                
                # Update KV cache
                past_key_values = outputs.past_key_values
                generated_tokens += 1
                
                # Generate next token
                next_token = torch.argmax(outputs.logits[:, -1], dim=-1)
                generated_text += self.tokenizer.decode(next_token[0])
                
                # Update input for next iteration
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                attention_mask = torch.cat([attention_mask, torch.ones(1, 1, device=self.device)], dim=1)
        
        # Calculate actual size
        actual_size_bytes = sum(
            k.nelement() * k.element_size() + v.nelement() * v.element_size()
            for k, v in past_key_values
        )
        actual_size_mb = actual_size_bytes / (1024 * 1024)
        
        return past_key_values, generated_text, actual_size_mb
    
    def compress_kv_cache(self, past_key_values):
        """Compress the KV cache using the autoencoder."""
        if past_key_values is None:
            return None
            
        compressed_cache = []
        with torch.no_grad():
            for layer in past_key_values:
                keys, values = layer
                # Compress keys and values
                k_compressed, _ = self.autoencoder(keys.reshape(-1, keys.size(-1)))
                v_compressed, _ = self.autoencoder(values.reshape(-1, values.size(-1)))
                
                # Reshape back to original dimensions
                k_compressed = k_compressed.reshape(keys.shape)
                v_compressed = v_compressed.reshape(values.shape)
                compressed_cache.append((k_compressed, v_compressed, keys, values))
        return compressed_cache
    
    def move_to_cpu(self, cache):
        """Move cache to CPU."""
        if cache is None:
            return None
        return tuple((k.cpu(), v.cpu(), k_orig.cpu(), v_orig.cpu()) 
                    for k, v, k_orig, v_orig in cache)
    
    def measure_time_to_first_token(self, text, cpu_kv_cache, use_compression, num_runs=5):
        """
        Measure time to generate first token with and without compression.
        Returns:
            - mean_time: Average time in seconds
            - std_time: Standard deviation of times
            - generated_text: The generated text
        """
        times = []
        generated_text = ""
        
        for _ in range(num_runs):
            start_time = time.time()
            
            # Move cache to device
            if use_compression:
                cache = tuple((k.to(self.device), v.to(self.device), k_orig.to(self.device), v_orig.to(self.device))
                            for k, v, k_orig, v_orig in cpu_kv_cache)
            else:
                cache = tuple((k.to(self.device), v.to(self.device)) for k, v in cpu_kv_cache)
            
            # Tokenize input
            inputs = self.tokenizer(text, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            
            # Generate first token
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=True
                )
                next_token = torch.argmax(outputs.logits[:, -1], dim=-1)
                generated_text = self.tokenizer.decode(next_token[0])
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        return sum(times) / len(times), torch.std(torch.tensor(times)).item(), generated_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KV Cache Compression Benchmark")
    parser.add_argument("--model", type=str, default="distilgpt2", help="Model name")
    parser.add_argument("--size", type=float, default=1, help="Target KV cache size in MB")
    parser.add_argument("--autoencoder", type=str, help="Path to trained autoencoder")
    parser.add_argument("--latent_dim", type=int, default=16, help="Latent dimension of autoencoder")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for compression operations")
    parser.add_argument("--num_runs", type=int, default=5, help="Number of runs for timing statistics")
    parser.add_argument("--output", type=str, default="results.json", help="Output file for results")
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = KVCacheInference(
        model_name=args.model,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        autoencoder_path=args.autoencoder,
        latent_dim=args.latent_dim,
        batch_size=args.batch_size
    )
    
    # Generate prompt and KV cache
    prompt = "Once upon a time in a land far away, there was a kingdom of"
    past_key_values, generated_text, actual_size_mb = inference.generate_kv_cache(prompt, args.size)
    
    results = {
        "model": args.model,
        "target_size_mb": args.size,
        "actual_size_mb": actual_size_mb,
        "generated_text": generated_text,
        "benchmarks": {}
    }
    
    # Measure time to first token without compression (baseline)
    print("\nMeasuring baseline (no compression)...")
    cpu_kv_cache = tuple((k.cpu(), v.cpu()) for k, v in past_key_values)
    time_baseline_avg, time_baseline_std, _ = inference.measure_time_to_first_token(
        generated_text, 
        cpu_kv_cache=cpu_kv_cache, 
        use_compression=False,
        num_runs=args.num_runs
    )
    results["benchmarks"]["baseline"] = {
        "avg_time": time_baseline_avg,
        "std_dev": time_baseline_std
    }
    
    # If autoencoder is provided, measure with compression
    if args.autoencoder:
        print("\nCompressing KV cache...")
        # Compress KV cache
        compressed_kv = inference.compress_kv_cache(past_key_values)
        cpu_compressed_kv = inference.move_to_cpu(compressed_kv)
        
        print("\nMeasuring with compression...")
        # Measure time to first token with compression
        time_compressed_avg, time_compressed_std, _ = inference.measure_time_to_first_token(
            generated_text, 
            cpu_kv_cache=cpu_compressed_kv, 
            use_compression=True,
            num_runs=args.num_runs
        )
        results["benchmarks"]["compressed"] = {
            "avg_time": time_compressed_avg,
            "std_dev": time_compressed_std
        }
        
        # Calculate compression ratio
        original_size = sum(k.nelement() * k.element_size() + v.nelement() * v.element_size() 
                           for k, v in past_key_values)
        compressed_size = sum(k.nelement() * k.element_size() + v.nelement() * v.element_size() 
                             for k, v, _, _ in compressed_kv)
        compression_ratio = original_size / compressed_size
        results["compression_ratio"] = compression_ratio
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output}")
    print(f"Time to first token (baseline): {time_baseline_avg:.4f}s ± {time_baseline_std:.4f}s")
    if args.autoencoder:
        print(f"Time to first token (compressed): {time_compressed_avg:.4f}s ± {time_compressed_std:.4f}s")
        print(f"Compression ratio: {compression_ratio:.2f}x")
        
        # Calculate speedup or slowdown
        speedup = time_baseline_avg / time_compressed_avg
        print(f"Speedup factor: {speedup:.2f}x")
        
        if speedup > 1.0:
            print(f"Compression provides {(speedup-1)*100:.1f}% faster time to first token")
        else:
            print(f"Compression adds {(1-speedup)*100:.1f}% overhead to time to first token")
