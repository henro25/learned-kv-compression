"""
Module Name: buffer.py
Description: Buffer Module for KV Cache Extraction. This module defines a Buffer class designed to extract and store key-value (KV) vectors and queries from a transformer model's caching mechanism for the purpose of training an autoencoder for KV cache compression. It operates by processing a small batch of texts, encoding them using a pretrained language model, extracting the keys, values, and queries from each transformer layer, and then storing them in preallocated buffers for later use during training.
Author: Henry Huang
Date: 2025-03-13
"""

import random
import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, logging

# Suppress specific warnings
logging.set_verbosity_error()
warnings.filterwarnings("ignore", message=".*return_dict_in_generate.*")
warnings.filterwarnings("ignore", message=".*scaled_dot_product_attention.*")

class Buffer():
    """
    Buffer for storing KV vectors and queries for training the autoencoder.
    This optimized version uses less memory by reducing buffer sizes and adding explicit memory management.
    """
    def __init__(self, cfg, model, tokenizer, texts):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        # Ensure the tokenizer has a pad token.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        # Get actual model configuration
        if cfg.get("name").split("/")[0] == "Qwen":
            self.num_heads = model.config.num_attention_heads
            self.hidden_size = model.config.hidden_size
            # Get the actual number of hidden layers from the model
            actual_layers = model.config.num_hidden_layers
            
            # Update config with actual model values if they don't match
            if self.cfg["num_attention_heads"] != self.num_heads:
                print(f"Warning: num_attention_heads in config ({self.cfg['num_attention_heads']}) does not match the model ({self.num_heads})")
                self.cfg["num_attention_heads"] = self.num_heads
                
            if self.cfg["hidden_size"] != self.hidden_size:
                print(f"Warning: hidden_size in config ({self.cfg['hidden_size']}) does not match the model ({self.hidden_size})")
                self.cfg["hidden_size"] = self.hidden_size
                
            if self.cfg["num_hidden_layers"] != actual_layers:
                print(f"Warning: num_hidden_layers in config ({self.cfg['num_hidden_layers']}) does not match the model ({actual_layers})")
                self.cfg["num_hidden_layers"] = actual_layers
        else:
            # Default handling for other models
            self.num_heads = model.config.n_head
            self.hidden_size = model.config.hidden_size
            
            # Get actual number of layers
            if hasattr(model.config, 'num_hidden_layers'):
                actual_layers = model.config.num_hidden_layers
                if self.cfg.get("num_hidden_layers") != actual_layers:
                    print(f"Warning: num_hidden_layers in config ({self.cfg.get('num_hidden_layers')}) does not match the model ({actual_layers})")
                    self.cfg["num_hidden_layers"] = actual_layers
        
        self.head_dim = self.hidden_size // self.num_heads
        
        
        
        # Filter out empty texts and ensure minimum length
        self.texts = [text for text in texts if text.strip() and len(text.split()) >= 3]
        if not self.texts:
            raise ValueError("No valid texts found after filtering. Please provide non-empty texts with at least 3 words.")
            
        self.device = cfg["device"]
        
        # Get data type from config and convert to torch dtype
        if "dtype" in cfg:
            if isinstance(cfg["dtype"], str):
                if cfg["dtype"] == "bf16":
                    self.dtype = torch.bfloat16
                elif cfg["dtype"] == "fp16" or cfg["dtype"] == "f16":
                    self.dtype = torch.float16
                else:
                    self.dtype = torch.float32
            else:
                # If already a torch dtype (converted by train.py)
                self.dtype = cfg["dtype"]
        else:
            self.dtype = torch.float32
            
        # print(f"Buffer using data type: {self.dtype}")
        
        # Reduce buffer sequence length significantly to save memory
        buffer_size = cfg.get("buffer_size", 128)  # Default to 256 tokens instead of 512
        self.buffer_seq_len = min(cfg.get("max_seq_len", 1024), buffer_size)  # Use smaller of max_seq_len or buffer_size
        
        # Reduce buffer multiplier even more
        buffer_mult = min(cfg.get("buffer_mult", 1), 1)  # Cap at 1x instead of 2x
        
        # Get batch size with a smaller default
        try:
            batch_size = cfg.get("batch_size")
            if batch_size is None:
                raise ValueError("batch_size not specified in configuration")
        except Exception as e:
            print(f"Error: {e}")
            batch_size = 8  # Default to 8 instead of 64
            print(f"Using default batch_size: {batch_size}")
        
        # Pre-allocate buffers with reduced size to save memory
        # Shape: (batch_size * buffer_mult, num_layers, num_heads, buffer_seq_len, head_dim)
        buffer_size = (batch_size * buffer_mult, cfg["num_hidden_layers"], 
                      self.num_heads, self.buffer_seq_len, self.head_dim)
        
        # print(f"Allocating buffers with shape {buffer_size}, total elements: {torch.prod(torch.tensor(buffer_size))}")
        
        try:
            # Try allocating tensors with specified size
            self.keys_buffer = torch.zeros(buffer_size, device=self.device, dtype=self.dtype)
            self.values_buffer = torch.zeros_like(self.keys_buffer)
            self.queries_buffer = torch.zeros_like(self.keys_buffer)
        except RuntimeError as e:
            # If allocation fails, try with progressively smaller buffers
            print(f"Failed to allocate buffers on GPU: {e}")
            print("Trying with smaller buffers...")
            
            # First reduction - try with smaller batch size
            try:
                buffer_size = (max(1, batch_size // 2) * buffer_mult, cfg["num_hidden_layers"],
                              self.num_heads, self.buffer_seq_len, self.head_dim)
                print(f"Retrying with buffer shape {buffer_size}")
                self.keys_buffer = torch.zeros(buffer_size, device=self.device, dtype=self.dtype)
                self.values_buffer = torch.zeros_like(self.keys_buffer)
                self.queries_buffer = torch.zeros_like(self.keys_buffer)
            except RuntimeError:
                # Second reduction - reduce sequence length too
                try:
                    self.buffer_seq_len = min(self.buffer_seq_len, 64)  # Reduce to 64 tokens
                    buffer_size = (max(1, batch_size // 2) * buffer_mult, cfg["num_hidden_layers"],
                                  self.num_heads, self.buffer_seq_len, self.head_dim)
                    print(f"Retrying with buffer shape {buffer_size}")
                    self.keys_buffer = torch.zeros(buffer_size, device=self.device, dtype=self.dtype)
                    self.values_buffer = torch.zeros_like(self.keys_buffer)
                    self.queries_buffer = torch.zeros_like(self.keys_buffer)
                except RuntimeError:
                    # Last resort - allocate minimum viable buffers
                    print("Using minimum viable buffer size")
                    self.buffer_seq_len = 32
                    buffer_size = (2, cfg["num_hidden_layers"],
                                 self.num_heads, self.buffer_seq_len, self.head_dim)
                    self.keys_buffer = torch.zeros(buffer_size, device=self.device, dtype=self.dtype)
                    self.values_buffer = torch.zeros_like(self.keys_buffer)
                    self.queries_buffer = torch.zeros_like(self.keys_buffer)
        
        self.text_pointer = 0
        random.shuffle(self.texts)
        self.refresh()
    
    @torch.no_grad()
    def refresh(self):
        self.pointer = 0
        
        while self.pointer < self.keys_buffer.shape[0]:
            # Get lm_batch_size with default
            lm_batch_size = self.cfg.get("lm_batch_size", 1)  # Default to 1 if not specified
            
            # Grab a mini-batch of texts.
            batch_texts = self.texts[self.text_pointer:self.text_pointer + lm_batch_size]
            if not batch_texts:
                self.text_pointer = 0
                random.shuffle(self.texts)
                continue
                
            encoded_inputs = self.tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=self.buffer_seq_len,  # Use reduced sequence length
                return_tensors="pt"
            )
            
            if not encoded_inputs["input_ids"].numel():
                raise ValueError("Empty encoded inputs")
                
            input_ids = encoded_inputs["input_ids"].to(self.device)
            attention_mask = encoded_inputs["attention_mask"].to(self.device)
            seq_len = input_ids.size(1)  # Get actual sequence length

            # Forward pass with caching enabled and output_attentions=True to get queries
            outputs = self.model(
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                use_cache=True, 
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True
            )
            past_key_values = outputs.past_key_values
            hidden_states = outputs.hidden_states  # Contains query vectors for each layer

            if hidden_states is None:
                raise ValueError("Model did not return hidden states. Make sure output_hidden_states=True is set in model config.")

            # Extract keys, values, and queries from each layer
            for l in range(self.cfg["num_hidden_layers"]):
                # Get keys and values from past_key_values
                keys, values = past_key_values[l]
                # Get query vectors from hidden states
                queries = hidden_states[l]  # Shape: (batch, seq_len, hidden_dim)
                
                # Reshape queries to match key/value shape
                queries = queries.reshape(queries.size(0), seq_len, self.num_heads, self.head_dim)
                queries = queries.transpose(1, 2)  # Shape: (batch, num_heads, seq_len, head_dim)
                
                # Ensure shapes match
                batch_size = keys.size(0)
                num_heads = keys.size(1)
                head_dim = keys.size(3)
                
                # Store in buffers - only up to actual sequence length
                buffer_slice_size = min(self.keys_buffer.shape[0] - self.pointer, batch_size)
                actual_seq_len = min(seq_len, self.buffer_seq_len)
                
                # Store in buffers
                self.keys_buffer[self.pointer:self.pointer + buffer_slice_size, l, :num_heads, :actual_seq_len, :head_dim] = keys[:buffer_slice_size, :num_heads, :actual_seq_len, :head_dim]
                self.values_buffer[self.pointer:self.pointer + buffer_slice_size, l, :num_heads, :actual_seq_len, :head_dim] = values[:buffer_slice_size, :num_heads, :actual_seq_len, :head_dim]
                self.queries_buffer[self.pointer:self.pointer + buffer_slice_size, l, :num_heads, :actual_seq_len, :head_dim] = queries[:buffer_slice_size, :num_heads, :actual_seq_len, :head_dim]

            self.pointer += buffer_slice_size
            self.text_pointer += lm_batch_size
            
            if self.text_pointer > len(self.texts) - lm_batch_size:
                self.text_pointer = 0
                random.shuffle(self.texts)

            # Release some memory after each batch
            torch.cuda.empty_cache()

        # Shuffle the buffers
        self.pointer = 0
        indices = torch.randperm(self.keys_buffer.shape[0], device=self.device)
        self.keys_buffer = self.keys_buffer[indices]
        self.values_buffer = self.values_buffer[indices]
        self.queries_buffer = self.queries_buffer[indices]

    def next(self):
        """
        Returns a batch of key-value pairs and queries.
        
        Returns:
            tuple: (kvs, queries) where:
                - kvs is a tuple of (keys, values) tensors
                - queries is a tensor
                All tensors have shape (batch_size, num_layers, num_heads, seq_len, head_dim)
        """
        # Get batch size with default
        batch_size = self.cfg.get("batch_size", 2)  # Default to 2 if not specified
        
        # Get the next batch
        keys = self.keys_buffer[self.pointer:self.pointer + batch_size]
        values = self.values_buffer[self.pointer:self.pointer + batch_size]
        queries = self.queries_buffer[self.pointer:self.pointer + batch_size]
        
        self.pointer += batch_size
        if self.pointer > self.keys_buffer.shape[0] - batch_size:
            self.refresh()
            
        return (keys, values), queries

# Minimal test code for CPU using distilgpt2.
if __name__ == "__main__":
    # Configuration for testing (small values for quick CPU runs)
    cfg = {
        "batch_size": 2,
        "buffer_mult": 2,
        "lm_batch_size": 1,
        "num_hidden_layers": 6,       # distilgpt2 has 6 layers
        "num_attention_heads": 12,    # as in the model config
        "head_dim": 64,               # 768/12 = 64
        "max_seq_len": 128,           # maximum sequence length
        "device": torch.device("cpu")
    }
    
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        output_hidden_states=True,
        output_attentions=True,
        use_cache=True
    ).to(cfg["device"])
    
    # Use a couple of short texts
    texts = [
        "Hello, world! This is a test sentence.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a fascinating field of study.",
        "Natural language processing helps computers understand human language."
    ]
    
    try:
        buf = Buffer(cfg, model, tokenizer, texts)
        (keys, values), queries = buf.next()
        print("Extracted KV shapes:")
        print("Keys:", keys.shape)
        print("Values:", values.shape)
        print("Queries:", queries.shape)
    except Exception as e:
        print(f"Error during buffer initialization: {e}")
