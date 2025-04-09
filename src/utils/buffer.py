"""
Module Name: buffer.py
Description: Buffer Module for KV Cache Extraction. This module defines a Buffer class designed to extract and store key-value (KV) vectors and queries from a transformer model's caching mechanism for the purpose of training an autoencoder for KV cache compression. It operates by processing a small batch of texts, encoding them using a pretrained language model, extracting the keys, values, and queries from each transformer layer, and then storing them in preallocated buffers for later use during training.
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
    This version is adapted for quick CPU testing using "distilgpt2".
    """
    def __init__(self, cfg, model, tokenizer, texts):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        # Ensure the tokenizer has a pad token.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        # Get actual model configuration
        self.num_heads = model.config.n_head
        self.hidden_size = model.config.hidden_size
        self.head_dim = self.hidden_size // self.num_heads
        
        # Update config with actual model values
        self.cfg["num_key_value_heads"] = self.num_heads
        self.cfg["head_dim"] = self.head_dim
        
        # Filter out empty texts and ensure minimum length
        self.texts = [text for text in texts if text.strip() and len(text.split()) >= 3]
        if not self.texts:
            raise ValueError("No valid texts found after filtering. Please provide non-empty texts with at least 3 words.")
            
        self.device = cfg["device"]
        
        # Pre-allocate buffers for keys, values, and queries
        # Shape: (batch_size * buffer_mult, num_layers, num_heads, seq_len, head_dim)
        self.keys_buffer = torch.zeros(
            (cfg["batch_size"] * cfg["buffer_mult"], cfg["num_hidden_layers"], self.num_heads, cfg["max_seq_len"], self.head_dim),
            device=self.device
        )
        self.values_buffer = torch.zeros_like(self.keys_buffer)
        self.queries_buffer = torch.zeros_like(self.keys_buffer)
        
        self.text_pointer = 0
        random.shuffle(self.texts)
        self.refresh()
    
    @torch.no_grad()
    def refresh(self):
        self.pointer = 0
        
        while self.pointer < self.keys_buffer.shape[0]:
            # Grab a mini-batch of texts.
            batch_texts = self.texts[self.text_pointer:self.text_pointer + self.cfg["lm_batch_size"]]
            if not batch_texts:
                self.text_pointer = 0
                random.shuffle(self.texts)
                continue
                
            encoded_inputs = self.tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=self.cfg["max_seq_len"], 
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
                
                # Store in buffers
                self.keys_buffer[self.pointer:self.pointer + buffer_slice_size, l, :num_heads, :seq_len, :head_dim] = keys[:buffer_slice_size, :num_heads, :seq_len, :head_dim]
                self.values_buffer[self.pointer:self.pointer + buffer_slice_size, l, :num_heads, :seq_len, :head_dim] = values[:buffer_slice_size, :num_heads, :seq_len, :head_dim]
                self.queries_buffer[self.pointer:self.pointer + buffer_slice_size, l, :num_heads, :seq_len, :head_dim] = queries[:buffer_slice_size, :num_heads, :seq_len, :head_dim]

            self.pointer += buffer_slice_size
            self.text_pointer += self.cfg["lm_batch_size"]
            
            if self.text_pointer > len(self.texts) - self.cfg["lm_batch_size"]:
                self.text_pointer = 0
                random.shuffle(self.texts)

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
        # Get the next batch
        keys = self.keys_buffer[self.pointer:self.pointer + self.cfg["batch_size"]]
        values = self.values_buffer[self.pointer:self.pointer + self.cfg["batch_size"]]
        queries = self.queries_buffer[self.pointer:self.pointer + self.cfg["batch_size"]]
        
        self.pointer += self.cfg["batch_size"]
        if self.pointer > self.keys_buffer.shape[0] - self.cfg["batch_size"]:
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
        "num_key_value_heads": 12,    # as in the model config
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
