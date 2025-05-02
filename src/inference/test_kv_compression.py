#!/usr/bin/env python3
"""
test_kv_compression.py

Given a DistilGPT2 model and a set of trained layer-wise autoencoders,
compare generation with full KV cache vs. compress→decompress KV cache.
"""

import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.models.autoencoder import Autoencoder

def load_model(model_name: str, device: torch.device, dtype: torch.dtype):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map={"": device},
        output_hidden_states=False,
        output_attentions=False,
        use_cache=True
    )
    model.eval()
    return tokenizer, model

def load_autoencoders(path: str, num_layers: int, head_dim: int, latent_dim: int, device: torch.device, dtype: torch.dtype):
    chk = torch.load(path, map_location=device)
    aes = []
    for i in range(num_layers):
        ae = Autoencoder(input_dim=head_dim, latent_dim=latent_dim, dtype=dtype).to(device)
        ae.load_state_dict(chk[f"layer_{i}"])
        ae.eval()
        aes.append(ae)
    return aes

def compress_kv(past, autoencoders):
    compressed = []
    for (k, v), ae in zip(past, autoencoders):
        B, H, S, D = k.shape
        k_flat = k.view(-1, D)
        v_flat = v.view(-1, D)
        with torch.no_grad():
            k_lat = ae.encoder(k_flat)
            v_lat = ae.encoder(v_flat)
            k_rec = ae.decoder(k_lat).view(B, H, S, D)
            v_rec = ae.decoder(v_lat).view(B, H, S, D)
        compressed.append((k_rec, v_rec))
    return tuple(compressed)

def generate_with_cache(tokenizer, model, prompt, cache=None, gen_length=50, device=None):
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_ids = encoded.input_ids[0].tolist()

    if cache is None:
        out = model(input_ids=encoded.input_ids, use_cache=True)
        cache = out.past_key_values
    last_id = prompt_ids[-1]

    generated = []
    for _ in range(gen_length):
        input_id = torch.tensor([[last_id]], device=model.device)
        with torch.no_grad():
            out = model(input_ids=input_id, past_key_values=cache, use_cache=True)
            
        # apply top-k top-p filtering
        # logits = out.logits[:, -1, :]
        
        # original argmax
        next_id = out.logits[:, -1, :].argmax(dim=-1).item()
        
        generated.append(next_id)
        cache = out.past_key_values
        last_id = next_id

    full_ids = prompt_ids + generated
    return tokenizer.decode(full_ids, skip_special_tokens=True)

def build_kv_cache(tokenizer, model, prompt, target_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    ids = inputs.input_ids
    cache = None

    for _ in range(target_tokens):
        out = model(
            input_ids=ids[:, -1:],
            past_key_values=cache,
            use_cache=True,
        )
        cache = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1).unsqueeze(-1)
        ids = torch.cat([ids, next_token], dim=-1)

    return cache

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name",      type=str, default="distilgpt2")
    p.add_argument("--ae_path",         type=str, required=True)
    p.add_argument("--prompt",          type=str, default="Once upon a time in a land far away,")
    p.add_argument("--gen_length",      type=int, default=50, help="Tokens to generate")
    p.add_argument("--latent_dim",      type=int, default=32)
    p.add_argument("--dtype",           type=str, choices=["fp16","bf16","fp32"], default="fp16")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"fp16":torch.float16,"bf16":torch.bfloat16,"fp32":torch.float32}[args.dtype]

    tokenizer, model = load_model(args.model_name, device, dtype)

    # derive dims
    num_layers = model.config.n_layer
    head_dim    = model.config.hidden_size // model.config.num_attention_heads

    # load autoencoders
    autoencoders = load_autoencoders(
        args.ae_path,
        num_layers=num_layers,
        head_dim=head_dim,
        latent_dim=args.latent_dim,
        device=device,
        dtype=dtype
    )

    # build a KV cache of roughly gen_length tokens
    print("⏳ building KV cache...")
    cache = build_kv_cache(tokenizer, model, args.prompt, target_tokens=args.gen_length)

    # baseline continuation
    print("▶️  Generating baseline continuation...")
    baseline = generate_with_cache(
        tokenizer, model, args.prompt,
        cache=cache,
        gen_length=args.gen_length,
        device=device
    )

    # compress → decompress cache
    print("🔧 compressing & decompressing KV cache...")
    comp_cache = compress_kv(cache, autoencoders)

    print("▶️  Generating compressed continuation...")
    compressed = generate_with_cache(
        tokenizer, model, args.prompt,
        cache=comp_cache,
        gen_length=args.gen_length,
        device=device
    )

    # display
    print("\n=== BASELINE OUTPUT ===\n")
    print(baseline)
    print("\n=== COMPRESSED OUTPUT ===\n")
    print(compressed)

if __name__=="__main__":
    main()