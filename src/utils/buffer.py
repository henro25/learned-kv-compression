"""
Module Name: buffer.py
Description: Buffer Module for KV Cache Extraction. This module defines a simple Buffer class designed to extract and store key-value (KV) vectors from a transformer model's caching mechanism for the purpose of training an autoencoder for KV cache compression. It operates by processing a small batch of texts, encoding them using a pretrained language model (e.g. distilgpt2), extracting the keys and values from each transformer layer (across all layers), and then storing them in a preallocated buffer for later use during training.
Author: Henry Huang
Date: 2025-03-13
"""

import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class Buffer():
    """
    Buffer for storing KV vectors for training the autoencoder.
    This version is adapted for quick CPU testing using "distilgpt2".
    """
    def __init__(self, cfg, model, tokenizer, texts):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        # Ensure the tokenizer has a pad token.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.texts = texts
        self.device = cfg["device"]
        # Pre-allocate a buffer: shape = (total_kv_vectors, num_layers*2, head_dim)
        self.buffer = torch.zeros(
            (cfg["batch_size"] * cfg["buffer_mult"], cfg["num_hidden_layers"] * 2, cfg["head_dim"]),
            device=self.device
        )
        self.text_pointer = 0
        random.shuffle(self.texts)
        self.refresh()
    
    @torch.no_grad()
    def refresh(self):
        self.pointer = 0
        while self.pointer < self.buffer.shape[0]:
            try:
                # Grab a mini-batch of texts.
                batch_texts = self.texts[self.text_pointer:self.text_pointer + self.cfg["lm_batch_size"]]
                encoded_inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
                input_ids = encoded_inputs["input_ids"].to(self.device)
                attention_mask = encoded_inputs["attention_mask"].to(self.device)

                # Forward pass with caching enabled.
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, return_dict=True)
                past_key_values = outputs.past_key_values

                # Extract keys and values from each layer.
                kvs = []
                for l in range(self.cfg["num_hidden_layers"]):
                    keys, values = past_key_values[l]
                    kvs.append(keys)
                    kvs.append(values)
                # Stack and rearrange:
                # After stacking, shape: (num_layers*2, batch, num_heads, seq_len, head_dim)
                # Permute to shape: (batch, seq_len, num_heads, num_layers*2, head_dim)
                # Then flatten (batch, seq_len, num_heads) into one dimension.
                kvs = torch.stack(kvs).permute(1, 3, 2, 0, 4)  \
                    .reshape(-1, self.cfg["num_hidden_layers"] * 2, self.cfg["head_dim"])

                # Create a flat mask based on the attention mask.
                # The mask is repeated for each key-value head.
                mask = attention_mask.view(-1, 1).repeat(1, self.cfg["num_key_value_heads"]).view(-1)
                kvs = kvs[mask.bool()]

                # If the KV tensor is empty, skip this batch.
                if kvs.numel() == 0:
                    print("Warning: Empty KV batch encountered. Skipping this batch.")
                    self.text_pointer += self.cfg["lm_batch_size"]
                    continue

                # Fill the remaining buffer.
                buffer_slice_size = min(self.buffer.shape[0] - self.pointer, kvs.size(0))
                self.buffer[self.pointer:self.pointer + buffer_slice_size] = kvs[:buffer_slice_size]
                self.pointer += buffer_slice_size
                self.text_pointer += self.cfg["lm_batch_size"]
                if self.text_pointer > len(self.texts) - self.cfg["lm_batch_size"]:
                    self.text_pointer = 0
                    random.shuffle(self.texts)
            except RuntimeError as e:
                print(f"Error encountered: {e}. Skipping this batch.")
                self.text_pointer += self.cfg["lm_batch_size"]

        # Shuffle the entire buffer.
        self.pointer = 0
        self.buffer = self.buffer[torch.randperm(self.buffer.shape[0], device=self.device)]

    def next(self):
        out = self.buffer[self.pointer:self.pointer + self.cfg["batch_size"]]
        self.pointer += self.cfg["batch_size"]
        if self.pointer > self.buffer.shape[0] - self.cfg["batch_size"]:
            self.refresh()
        return out

# Minimal test code for CPU using distilgpt2.
if __name__ == "__main__":
    # Configuration for testing (small values for quick CPU runs)
    cfg = {
        "batch_size": 2,
        "buffer_mult": 2,
        "lm_batch_size": 1,
        "num_hidden_layers": 6,       # distilgpt2 has 6 layers.
        "head_dim": 64,               # 768/12 = 64.
        "num_key_value_heads": 12,    # as in the model config.
        "device": torch.device("cpu")
    }
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        output_hidden_states=True,
        use_cache=True
    ).to(cfg["device"])
    
    # Use a couple of short texts.
    texts = [
        "Hello, world! This is a test sentence.",
        "The quick brown fox jumps over the lazy dog."
    ]
    
    buf = Buffer(cfg, model, tokenizer, texts)
    batch = buf.next()
    print("Extracted KV batch shape:", batch.shape)
