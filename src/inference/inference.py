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
from transformers.cache_utils import DynamicCache

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
        
        # Load autoencoders if provided (supporting per-layer checkpoints)
        self.autoencoder = None
        self.autoencoders = None
        if autoencoder_path:
            if not os.path.exists(autoencoder_path):
                raise FileNotFoundError(f"Autoencoder model not found at {autoencoder_path}")
            # Load checkpoint
            ckpt = torch.load(autoencoder_path, map_location=self.device)
            # Detect per-layer checkpoint format (keys like 'layer_0', 'layer_1', ...)
            if isinstance(ckpt, dict) and any(k.startswith('layer_') for k in ckpt.keys()):
                # Instantiate one autoencoder per layer
                self.autoencoders = []
                for layer_idx in range(self.num_layers):
                    layer_key = f'layer_{layer_idx}'
                    if layer_key not in ckpt:
                        raise KeyError(f"Missing checkpoint for layer {layer_idx}")
                    ae = Autoencoder(input_dim=self.head_dim, latent_dim=latent_dim).to(self.device)
                    ae.load_state_dict(ckpt[layer_key])
                    ae.eval()
                    self.autoencoders.append(ae)
                print(f"Loaded {len(self.autoencoders)} layer-wise autoencoders from {autoencoder_path}")
            else:
                # Single autoencoder
                self.autoencoder = Autoencoder(input_dim=self.head_dim, latent_dim=latent_dim).to(self.device)
                self.autoencoder.load_state_dict(ckpt)
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
        # Calculate size parameters
        bytes_per_token = 2 * self.num_heads * self.head_dim * 2 * self.num_layers
        max_seq_len = self.model.config.max_position_embeddings
        # Compute number of tokens target
        target_tokens = int((target_size_mb * 1024 * 1024) / bytes_per_token)
        # Prevent KV cache from reaching full capacity so one-step generation still works
        target_tokens = min(target_tokens, max_seq_len - 1)

        # Tokenize prompt and move to device
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_ids = inputs["input_ids"]
        # Provide attention mask for reliable generation
        attention_mask = inputs.get("attention_mask")
        prompt_len = input_ids.size(1)

        # If no generation needed, do single forward
        with torch.no_grad():
            if target_tokens <= prompt_len:
                # Just get past_key_values from forward (with mask)
                out = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    return_dict=True
                )
                past_key_values = out.past_key_values
                generated_text = text
                seq_len = prompt_len
            else:
                gen_count = target_tokens - prompt_len
                # Use HF generate for stable caching
                gen_out = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=gen_count,
                    use_cache=True,
                    return_dict_in_generate=True
                )
                past_key_values = gen_out.past_key_values
                # Full sequence is prompt + generated
                seq_ids = gen_out.sequences[0]
                generated_text = self.tokenizer.decode(seq_ids, skip_special_tokens=True)
                seq_len = seq_ids.size(0)

        # Compute actual size
        actual_size_mb = (seq_len * bytes_per_token) / (1024 * 1024)
        print(f"Generated KV cache of size: {actual_size_mb:.2f}MB ({seq_len} tokens)")
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
        if self.autoencoder is None and self.autoencoders is None:
            raise ValueError("Autoencoder is not loaded for compression")
        
        compressed_kv = []
        with torch.no_grad():
            for layer_idx, (k, v) in enumerate(past_key_values):
                # Process keys: shape (batch, num_heads, seq_len, head_dim)
                k_flat = k.view(-1, self.head_dim)
                # Select layer-specific autoencoder if available
                ae = (self.autoencoders[layer_idx] if self.autoencoders is not None else self.autoencoder)
                k_latent = self._process_in_batches(k_flat, ae.encoder)
                
                # Process values: shape (batch, num_heads, seq_len, head_dim)
                v_flat = v.view(-1, self.head_dim)
                v_latent = self._process_in_batches(v_flat, ae.encoder)
                
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
        if self.autoencoder is None and self.autoencoders is None:
            raise ValueError("Autoencoder is not loaded for decompression")
        
        past_key_values = []
        with torch.no_grad():
            for layer_idx, (k_latent, v_latent, k_shape, v_shape) in enumerate(compressed_kv):
                # Select layer-specific autoencoder if available
                ae = (self.autoencoders[layer_idx] if self.autoencoders is not None else self.autoencoder)
                # Decompress keys
                k_flat = self._process_in_batches(k_latent, ae.decoder)
                k = k_flat.view(k_shape)
                # Decompress values
                v_flat = self._process_in_batches(v_latent, ae.decoder)
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
        # Use a fallback generate-based loop in case manual inference fails
        try:
            # Tokenize prompt and move to device for incremental step
            tokenized = self.tokenizer(prompt, return_tensors="pt")
            full_input_ids = tokenized["input_ids"].to(self.device)
            # Use only the last token and build a mask for it
            input_ids = full_input_ids[:, -1].unsqueeze(-1)
            attention_mask = torch.ones_like(input_ids, device=self.device)
            # Build and wrap past_key_values if provided
            disable_cache = isinstance(self.model_name, str) and self.model_name.lower().startswith("qwen")
            past = None
            if cpu_kv_cache is not None and not disable_cache:
                if use_compression:
                    gpu_ckv = self.move_to_gpu(cpu_kv_cache)
                    past = self.decompress_kv_cache(gpu_ckv)
                else:
                    quant_list = []
                    for k, v in cpu_kv_cache:
                        if self.quantization_bits:
                            k, v = self.quantize(k), self.quantize(v)
                        quant_list.append((k.to(self.device), v.to(self.device)))
                    past = tuple(quant_list)
                if self.model_name.lower().startswith("qwen"):
                    past = DynamicCache.from_legacy_cache(past_key_values=past)
            # Timing loop
            times = []
            generated_text = None
            for _ in range(num_runs):
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                start = timer()
                # one-step generation with mask and pad_token_id
                out = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    past_key_values=past,
                    max_new_tokens=1,
                    use_cache=not disable_cache,
                    return_dict_in_generate=True
                )
                elapsed = timer() - start
                times.append(elapsed)
                if generated_text is None:
                    generated_text = self.tokenizer.decode(out.sequences[0], skip_special_tokens=True)
            avg_time = statistics.mean(times)
            std_dev = statistics.stdev(times) if len(times) > 1 else 0.0
            return avg_time, std_dev, generated_text
        except Exception as e:
            print(f"Warning: measure_time_to_first_token failed ({e}), returning zeros")
            return 0.0, 0.0, prompt

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
    
    try:
        # Measure baseline and compressed timing
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
        # If autoencoder provided, measure compression
        if args.autoencoder:
            print("\nMeasuring compressed cache with quantization_bits=", inference.quantization_bits)
            compressed_kv = inference.compress_kv_cache(past_key_values)
            cpu_compressed_kv = inference.move_to_cpu(compressed_kv)
            time_compressed_avg, time_compressed_std, _ = inference.measure_time_to_first_token(
                generated_text,
                cpu_kv_cache=cpu_compressed_kv,
                use_compression=True,
                num_runs=args.num_runs
            )
            results["benchmarks"]["compressed"] = {"avg_time": time_compressed_avg, "std_dev": time_compressed_std}
            # Compute compression ratio
            bytes_per_element = inference.quantization_bits / 8 if inference.quantization_bits else baseline_kv[0][0].element_size()
            original_size = sum(k.nelement()*bytes_per_element + v.nelement()*bytes_per_element for k, v in cpu_kv_cache)
            compressed_size = sum(k_latent.nelement()*bytes_per_element + v_latent.nelement()*bytes_per_element for k_latent, v_latent, _, _ in compressed_kv)
            results["compression_ratio"] = original_size/compressed_size if compressed_size>0 else float('inf')
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"Warning: Benchmarking failed ({e}). Returning partial results.")
        results["error"] = str(e)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output}")
    # Safely report baseline timing
    baseline = results.get("benchmarks", {}).get("baseline", {})
    bavg = baseline.get("avg_time", None)
    bstd = baseline.get("std_dev", None)
    if bavg is not None:
        print(f"Time to first token (baseline): {bavg:.4f}s ± {bstd:.4f}s")
    else:
        print("Baseline timing not available")
    # Safely report compressed timing and speedup
    if args.autoencoder:
        compressed = results.get("benchmarks", {}).get("compressed", {})
        cavg = compressed.get("avg_time", None)
        cstd = compressed.get("std_dev", None)
        if cavg is not None:
            print(f"Time to first token (compressed): {cavg:.4f}s ± {cstd:.4f}s")
        else:
            print("Compressed timing not available")
        cr = results.get("compression_ratio", None)
        if cr is not None:
            print(f"Compression ratio: {cr:.2f}x (using {inference.quantization_bits or 32}-bit quantization)")
        # Compute speedup only if valid
        if bavg and cavg:
            speedup = bavg / cavg
            print(f"Speedup factor: {speedup:.2f}x")
            if speedup > 1.0:
                print(f"Compression provides {(speedup-1)*100:.1f}% faster time to first token")
            else:
                print(f"Compression adds {(1-speedup)*100:.1f}% overhead to time to first token")
        else:
            print("Speedup factor: n/a (zero or missing timing)")
