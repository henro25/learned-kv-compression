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
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.models.autoencoder import Autoencoder

class KVCacheInference:
    def __init__(self, model_name, device=None, autoencoder_path=None, latent_dim=16):
        """
        Initialize inference with a model and optional autoencoder for compression.
        
        Args:
            model_name (str): Name of the HuggingFace model to use
            device (str, optional): Device to run inference on. Defaults to CUDA if available.
            autoencoder_path (str, optional): Path to the trained autoencoder model
            latent_dim (int): Dimension of the latent space for the autoencoder
        """
        self.model_name = model_name
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map={"": self.device}
        )
        
        # Get model configuration
        self.num_layers = self.model.config.num_hidden_layers
        self.num_heads = self.model.config.num_attention_heads
        self.head_dim = self.model.config.hidden_size // self.num_heads
        
        # Load autoencoder if provided
        self.autoencoder = None
        if autoencoder_path:
            self.autoencoder = Autoencoder(input_dim=self.head_dim, latent_dim=latent_dim).to(self.device)
            self.autoencoder.load_state_dict(torch.load(autoencoder_path, map_location=self.device))
            self.autoencoder.eval()
            print(f"Loaded autoencoder from {autoencoder_path}")
    
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
                k_latent = self.autoencoder.encoder(k_flat)
                
                # Process values: shape (batch, num_heads, seq_len, head_dim)
                v_flat = v.view(-1, self.head_dim)
                v_latent = self.autoencoder.encoder(v_flat)
                
                # Store the shapes for reconstruction
                k_shape = k.shape
                v_shape = v.shape
                
                compressed_kv.append((k_latent, v_latent, k_shape, v_shape))
        
        return compressed_kv
    
    def decompress_kv_cache(self, compressed_kv):
        """
        Decompress KV cache using the trained autoencoder.
        
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
                k_flat = self.autoencoder.decoder(k_latent)
                k = k_flat.view(k_shape)
                
                # Decompress values
                v_flat = self.autoencoder.decoder(v_latent)
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
    
    def measure_time_to_first_token(self, prompt, cpu_kv_cache=None, use_compression=False):
        """
        Measure time to first token when continuing generation from a stored KV cache.
        
        Args:
            prompt (str): Prompt to continue from
            cpu_kv_cache: KV cache stored on CPU (compressed or not)
            use_compression (bool): Whether to use compression/decompression
            
        Returns:
            tuple: (time_to_first_token, generated_text)
        """
        # Encode the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        
        start_time = time.time()
        
        past_key_values = None
        if cpu_kv_cache is not None:
            if use_compression:
                # Move compressed KV cache to GPU and decompress
                gpu_compressed_kv = self.move_to_gpu(cpu_kv_cache)
                past_key_values = self.decompress_kv_cache(gpu_compressed_kv)
            else:
                # Just move the uncompressed KV cache to GPU
                past_key_values = tuple((k.to(self.device), v.to(self.device)) for k, v in cpu_kv_cache)
        
        # Generate first token
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids[:, -1:] if past_key_values else input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True
            )
            
            next_token = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(-1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        time_to_first_token = time.time() - start_time
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        return time_to_first_token, generated_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KV Cache Compression Benchmark")
    parser.add_argument("--model", type=str, default="distilgpt2", help="Model name")
    parser.add_argument("--size", type=float, default=1, help="Target KV cache size in MB")
    parser.add_argument("--autoencoder", type=str, help="Path to trained autoencoder")
    parser.add_argument("--latent_dim", type=int, default=16, help="Latent dimension of autoencoder")
    parser.add_argument("--output", type=str, default="results.json", help="Output file for results")
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = KVCacheInference(
        model_name=args.model,
        autoencoder_path=args.autoencoder,
        latent_dim=args.latent_dim
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
    cpu_kv_cache = tuple((k.cpu(), v.cpu()) for k, v in past_key_values)
    time_baseline, _ = inference.measure_time_to_first_token(
        generated_text, 
        cpu_kv_cache=cpu_kv_cache, 
        use_compression=False
    )
    results["benchmarks"]["baseline"] = time_baseline
    
    # If autoencoder is provided, measure with compression
    if args.autoencoder:
        # Compress KV cache
        compressed_kv = inference.compress_kv_cache(past_key_values)
        cpu_compressed_kv = inference.move_to_cpu(compressed_kv)
        
        # Measure time to first token with compression
        time_compressed, _ = inference.measure_time_to_first_token(
            generated_text, 
            cpu_kv_cache=cpu_compressed_kv, 
            use_compression=True
        )
        results["benchmarks"]["compressed"] = time_compressed
        
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
    
    print(f"Results saved to {args.output}")
    print(f"Time to first token (baseline): {time_baseline:.4f}s")
    if args.autoencoder:
        print(f"Time to first token (compressed): {time_compressed:.4f}s")
        print(f"Compression ratio: {compression_ratio:.2f}x")
