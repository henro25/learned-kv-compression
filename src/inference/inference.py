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
    def __init__(self, model_name, device=None, autoencoder_path=None, latent_dim=16, batch_size=1, quantization_bits=None):
        """
        Initialize inference with a model and optional autoencoder for compression.
        
        Args:
            model_name (str): Name of the HuggingFace model to use
            device (str, optional): Device to run inference on. Defaults to CUDA if available.
            autoencoder_path (str, optional): Path to the trained autoencoder model
            latent_dim (int): Dimension of the latent space for the autoencoder
            batch_size (int): Batch size for compression operations
            quantization_bits (int, optional): Quantization bits for KV and latent representations
        """
        self.model_name = model_name
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map={"": self.device}
        )
        
        # Get model configuration
        self.num_layers = self.model.config.num_hidden_layers
        self.num_heads = self.model.config.num_attention_heads
        self.head_dim = self.model.config.hidden_size // self.num_heads
        
        # Load autoencoder if provided
        self.autoencoder = None
        if autoencoder_path:
            if not os.path.exists(autoencoder_path):
                raise FileNotFoundError(f"Autoencoder model not found at {autoencoder_path}")
            
            self.autoencoder = Autoencoder(input_dim=self.head_dim, latent_dim=latent_dim).to(self.device)
            self.autoencoder.load_state_dict(torch.load(autoencoder_path, map_location=self.device))
            self.autoencoder.eval()
            print(f"Loaded autoencoder from {autoencoder_path}")
        
        self.quantization_bits = quantization_bits
        # Set up quantization scale if bits provided
        if quantization_bits:
            self.quant_scale = 2.0 ** (quantization_bits - 1) - 1
        else:
            self.quant_scale = None
    
    def generate_kv_cache(self, text, target_size_mb=1):
        """
        Generate KV Cache of approximately target_size_mb.
        
        Args:
            text (str): Input text to generate from
            target_size_mb (float): Target size of KV cache in MB
            
        Returns:
            tuple: (past_key_values, generated_text, actual_size_mb)
        """
        # Calculate how many tokens we need to reach target_size_mb
        # Size per token per layer: 2 * num_heads * head_dim * 2 bytes (fp16)
        bytes_per_token = 2 * self.num_heads * self.head_dim * 2 * self.num_layers
        target_tokens = int((target_size_mb * 1024 * 1024) / bytes_per_token)
        
        # Add a maximum sequence length check based on model's context window
        max_sequence_length = self.model.config.max_position_embeddings
        if target_tokens > max_sequence_length:
            print(f"Warning: Target tokens ({target_tokens}) exceeds model's max sequence length ({max_sequence_length})")
            target_tokens = max_sequence_length
        
        # Encode the input
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        generated_tokens = input_ids.shape[1]
        
        # Generate tokens one by one to create KV cache
        generated_text = text
        with torch.no_grad():
            past_key_values = None
            pbar = tqdm(total=target_tokens, desc=f"Generating {target_size_mb}MB KV cache")
            
            while generated_tokens < target_tokens:
                outputs = self.model(
                    input_ids=input_ids[:, -1:] if past_key_values else input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True
                )
                
                past_key_values = outputs.past_key_values
                next_token = outputs.logits[:, -1, :].argmax(dim=-1)
                
                # Append to input_ids and attention_mask
                input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
                attention_mask = torch.cat([attention_mask, torch.ones_like(next_token.unsqueeze(-1))], dim=-1)
                generated_tokens += 1
                
                # Update progress bar every 10 tokens
                if generated_tokens % 10 == 0:
                    pbar.update(10)
                    # Decode the new text
                    new_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
                    generated_text = new_text
            
            pbar.close()
        
        # Calculate actual size of the KV cache
        actual_size_mb = (generated_tokens * bytes_per_token) / (1024 * 1024)
        print(f"Generated KV cache of size: {actual_size_mb:.2f}MB ({generated_tokens} tokens)")
        
        return past_key_values, generated_text, actual_size_mb
    
    def compress_kv_cache(self, past_key_values):
        """
        Compress KV cache using the trained autoencoder.
        Optimized with batched processing for faster compression.
        
        Args:
            past_key_values: Past key values from the model
            
        Returns:
            compressed_kv: Compressed representation of the KV cache
        """
        if not self.autoencoder:
            raise ValueError("Autoencoder is not loaded for compression")
        
        compressed_kv = []
        with torch.no_grad():
            for layer_idx, (k, v) in enumerate(past_key_values):
                # Process keys: shape (batch, num_heads, seq_len, head_dim)
                k_flat = k.view(-1, self.head_dim)
                k_latent = self._process_in_batches(k_flat, self.autoencoder.encoder)
                
                # Process values: shape (batch, num_heads, seq_len, head_dim)
                v_flat = v.view(-1, self.head_dim)
                v_latent = self._process_in_batches(v_flat, self.autoencoder.encoder)
                
                # Store the shapes for reconstruction
                k_shape = k.shape
                v_shape = v.shape
                
                # Quantize latent if specified
                if self.quantization_bits:
                    k_latent = self.quantize(k_latent)
                    v_latent = self.quantize(v_latent)
                compressed_kv.append((k_latent, v_latent, k_shape, v_shape))
        
        return compressed_kv
    
    def _process_in_batches(self, tensor, model_func):
        """Process a tensor in batches to avoid OOM errors"""
        result_chunks = []
        num_items = tensor.size(0)
        
        for i in range(0, num_items, self.batch_size):
            end_idx = min(i + self.batch_size, num_items)
            batch = tensor[i:end_idx]
            result_chunks.append(model_func(batch))
        
        return torch.cat(result_chunks, dim=0)
    
    def decompress_kv_cache(self, compressed_kv):
        """
        Decompress KV cache using the trained autoencoder.
        Optimized with batched processing for faster decompression.
        
        Args:
            compressed_kv: Compressed representation of the KV cache
            
        Returns:
            past_key_values: Decompressed past key values for the model
        """
        if not self.autoencoder:
            raise ValueError("Autoencoder is not loaded for decompression")
        
        past_key_values = []
        with torch.no_grad():
            for k_latent, v_latent, k_shape, v_shape in compressed_kv:
                # Decompress keys
                k_flat = self._process_in_batches(k_latent, self.autoencoder.decoder)
                k = k_flat.view(k_shape)
                
                # Decompress values
                v_flat = self._process_in_batches(v_latent, self.autoencoder.decoder)
                v = v_flat.view(v_shape)
                
                past_key_values.append((k, v))
        
        return tuple(past_key_values)
    
    def move_to_cpu(self, compressed_kv):
        """Move compressed KV cache to CPU memory"""
        cpu_compressed_kv = []
        for k_latent, v_latent, k_shape, v_shape in compressed_kv:
            cpu_compressed_kv.append((
                k_latent.cpu(), 
                v_latent.cpu(), 
                k_shape, 
                v_shape
            ))
        return cpu_compressed_kv
    
    def move_to_gpu(self, cpu_compressed_kv):
        """Move compressed KV cache from CPU to GPU memory"""
        gpu_compressed_kv = []
        for k_latent, v_latent, k_shape, v_shape in cpu_compressed_kv:
            gpu_compressed_kv.append((
                k_latent.to(self.device), 
                v_latent.to(self.device), 
                k_shape, 
                v_shape
            ))
        return gpu_compressed_kv
    
    def quantize(self, x):
        """Quantize tensor to specified bit depth"""
        if not self.quantization_bits:
            return x
        # Scale to integer range
        x_scaled = x * self.quant_scale
        # Clamp
        x_clamped = torch.clamp(x_scaled, -self.quant_scale, self.quant_scale - 1)
        # Round
        x_quant = torch.round(x_clamped)
        # Scale back
        return x_quant / self.quant_scale
    
    def measure_time_to_first_token(self, prompt, cpu_kv_cache=None, use_compression=False, num_runs=5):
        """
        Measure time to first token when continuing generation from a stored KV cache.
        Runs multiple times and returns average and standard deviation.
        
        Args:
            prompt (str): Prompt to continue from
            cpu_kv_cache: KV cache stored on CPU (compressed or not)
            use_compression (bool): Whether to use compression/decompression
            num_runs (int): Number of times to run the measurement for statistical significance
            
        Returns:
            tuple: (avg_time, std_dev, generated_text)
        """
        # Encode the prompt (do this once outside the timing loop)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        
        times = []
        generated_text = None
        
        for run in range(num_runs):
            # Clear CUDA cache between runs for more consistent timing
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            start_time = timer()
            
            past_key_values = None
            if cpu_kv_cache is not None:
                if use_compression:
                    # Move compressed KV cache to GPU and decompress
                    gpu_compressed_kv = self.move_to_gpu(cpu_kv_cache)
                    past_key_values = self.decompress_kv_cache(gpu_compressed_kv)
                else:
                    # Baseline: optionally quantize raw KV before use
                    cpu_quant_kv = []
                    for k, v in cpu_kv_cache:
                        if self.quantization_bits:
                            k_q = self.quantize(k)
                            v_q = self.quantize(v)
                        else:
                            k_q, v_q = k, v
                        cpu_quant_kv.append((k_q.to(self.device), v_q.to(self.device)))
                    past_key_values = tuple(cpu_quant_kv)
            
            # Generate first token
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids[:, -1:] if past_key_values else input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True
                )
                
                next_token = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(-1)
                current_input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            end_time = timer()
            times.append(end_time - start_time)
            
            # Only need to decode the text once
            if generated_text is None:
                generated_text = self.tokenizer.decode(current_input_ids[0], skip_special_tokens=True)
        
        # Calculate statistics
        avg_time = statistics.mean(times)
        std_dev = statistics.stdev(times) if len(times) > 1 else 0.0
        
        return avg_time, std_dev, generated_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KV Cache Compression Benchmark")
    parser.add_argument("--model", type=str, default="distilgpt2", help="Model name")
    parser.add_argument("--size", type=float, default=1, help="Target KV cache size in MB")
    parser.add_argument("--autoencoder", type=str, help="Path to trained autoencoder")
    parser.add_argument("--latent_dim", type=int, default=16, help="Latent dimension of autoencoder")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for compression operations")
    parser.add_argument("--num_runs", type=int, default=5, help="Number of runs for timing statistics")
    parser.add_argument("--output", type=str, default="results.json", help="Output file for results")
    parser.add_argument("--quantization_bits", type=int, default=None, help="Number of bits for quantizing KV tensors before measurement")
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = KVCacheInference(
        model_name=args.model,
        device=args.device if hasattr(args, 'device') else None,
        autoencoder_path=args.autoencoder,
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        quantization_bits=args.quantization_bits
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
    
    # Measure time to first token without compression (baseline), with optional quantization
    print("\nMeasuring baseline (no compression) with quantization_bits=", inference.quantization_bits)
    baseline_kv = []
    for k, v in past_key_values:
        if inference.quantization_bits:
            k = inference.quantize(k)
            v = inference.quantize(v)
        baseline_kv.append((k.cpu(), v.cpu()))
    cpu_kv_cache = tuple(baseline_kv)
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
    
    # If autoencoder is provided, measure with compression (autoencoder + optional quantization)
    if args.autoencoder:
        print("\nCompressing KV cache with quantization_bits=", inference.quantization_bits)
        # Compress KV cache (latent + quantization)
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
        
        # Calculate compression ratio using quantized bit-depth for both baseline and compressed
        if inference.quantization_bits:
            bytes_per_element = inference.quantization_bits / 8
        else:
            # fallback to native float32 size
            bytes_per_element = baseline_kv[0][0].element_size()  # assume float32
        # original size in bytes
        original_size = sum(
            k.nelement() * bytes_per_element + v.nelement() * bytes_per_element
            for k, v in cpu_kv_cache
        )
        # compressed size in bytes (latent representation)
        compressed_size = sum(
            k_latent.nelement() * bytes_per_element + v_latent.nelement() * bytes_per_element
            for k_latent, v_latent, _, _ in compressed_kv
        )
        compression_ratio = original_size / compressed_size if compressed_size > 0 else float('inf')
        results["compression_ratio"] = compression_ratio
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output}")
    print(f"Time to first token (baseline): {time_baseline_avg:.4f}s ± {time_baseline_std:.4f}s")
    if args.autoencoder:
        print(f"Time to first token (compressed): {time_compressed_avg:.4f}s ± {time_compressed_std:.4f}s")
        print(f"Compression ratio: {compression_ratio:.2f}x (using {inference.quantization_bits or 32}-bit quantization)")
        
        # Calculate speedup or slowdown
        speedup = time_baseline_avg / time_compressed_avg
        print(f"Speedup factor: {speedup:.2f}x")
        
        if speedup > 1.0:
            print(f"Compression provides {(speedup-1)*100:.1f}% faster time to first token")
        else:
            print(f"Compression adds {(1-speedup)*100:.1f}% overhead to time to first token")
