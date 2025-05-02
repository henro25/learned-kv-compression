#!/usr/bin/env python3
"""
test_kv_compression.py  –  clean + correct KV‑cache comparison

• Silences TF / XLA / cuDNN registration noise
• Works with GPT‑2 style models *and* Qwen2 family
• Builds a KV cache from the prompt *once* (no dummy generation)
• Continues generation step‑by‑step, updating:
      - attention_mask  (1 → 1 → 1 …)
      - position_ids    (0‑based incremental positions)
      - past_key_values (from model)
• Uses top‑k sampling (k=50, temperature=1) so we never get “all spaces”
"""

# ─── silence nearly everything *before* any heavy libs are imported ─────────────
import os, warnings, logging, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["ABSL_MIN_LOG_LEVEL"]  = "3"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
# Hugging‑Face logging
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from src.models.autoencoder import Autoencoder

# ────────────────────────────────────────────────────────────────────────────────
def top_k_logits(logits: torch.Tensor, k: int = 50) -> torch.Tensor:
    if k <= 0 or k >= logits.size(-1):
        return logits
    v, idx = torch.topk(logits, k, dim=-1)
    out = logits.new_full(logits.shape, float("-inf"))
    out.scatter_(-1, idx, v)
    return out

# ─── helper: get layer / head counts (GPT‑2, Qwen, …) ──────────────────────────
def resolve_dims(cfg):
    def first(*names):
        for n in names:
            if hasattr(cfg, n): return getattr(cfg, n)
        raise ValueError(f"Config missing any of {names}")
    n_layers = first("num_hidden_layers", "n_layer")
    n_heads  = first("num_attention_heads", "n_head")
    hid_dim  = first("hidden_size", "n_embd")
    head_dim = hid_dim // n_heads
    return n_layers, head_dim

# ─── load LLM & tokenizer ───────────────────────────────────────────────────────
def load_model(name, device, dtype):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=dtype,
        device_map={"": device},
        use_cache=True
    )
    model.eval()
    return tok, model

# ─── load per‑layer autoencoders ────────────────────────────────────────────────
def load_aes(path, n_layers, head_dim, lat, device, dtype):
    ckpt = torch.load(path, map_location=device)
    aes  = []
    for i in range(n_layers):
        ae = Autoencoder(input_dim=head_dim, latent_dim=lat, dtype=dtype).to(device)
        ae.load_state_dict(ckpt[f"layer_{i}"])
        ae.eval()
        aes.append(ae)
    return aes

# ─── KV cache of the *prompt only* (no token growth) ───────────────────────────
def prompt_cache(tok, model, prompt):
    enc = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**enc, use_cache=True)
    return enc, out.past_key_values   # (input_ids+mask, past)

# ─── single‑step generation with proper masks & position_ids ───────────────────
def step(model, last_token, past, pos_id):
    attn_mask = torch.ones((1,1), device=model.device)
    with torch.no_grad():
        out = model(
            input_ids=last_token,
            attention_mask=attn_mask,
            position_ids=pos_id,
            past_key_values=past,
            use_cache=True,
        )
    return out.logits[:, -1, :], out.past_key_values

# ─── manual top‑k sampler loop ─────────────────────────────────────────────────
def continue_text(tok, model, enc, past, n_tokens=50, k=50, temp=1.0):
    ids   = enc.input_ids.clone()
    seq_l = ids.size(1)              # length so far
    last  = ids[0, -1].unsqueeze(0).unsqueeze(0)

    for _ in range(n_tokens):
        pos  = torch.tensor([[seq_l]], device=model.device)
        logits, past = step(model, last, past, pos)
        logits = logits / temp
        nxt = torch.multinomial(torch.softmax(top_k_logits(logits, k), dim=-1), 1)
        ids   = torch.cat([ids, nxt], dim=-1)
        last  = nxt
        seq_l += 1

    return tok.decode(ids[0], skip_special_tokens=True)

# ─── compress→decompress KV cache via AEs and wrap in DynamicCache ─────────────
def reconstruct_cache(past, aes):
    rebuilt = []
    for (k, v), ae in zip(past, aes):
        B,H,S,D = k.shape
        k_flat, v_flat = k.view(-1,D), v.view(-1,D)
        with torch.no_grad():
            k_rec = ae.decoder(ae.encoder(k_flat)).view(B,H,S,D)
            v_rec = ae.decoder(ae.encoder(v_flat)).view(B,H,S,D)
        rebuilt.append((k_rec, v_rec))
    return DynamicCache.from_legacy_cache(tuple(rebuilt))

# ─── CLI ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="distilgpt2")
    ap.add_argument("--ae_path", required=True)
    ap.add_argument("--prompt",  default="Once upon a time in a land far away,")
    ap.add_argument("--gen_len", type=int, default=50)
    ap.add_argument("--latent_dim", type=int, default=32)
    ap.add_argument("--dtype", choices=["fp32","fp16","bf16"], default="fp16")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = {"fp32":torch.float32, "fp16":torch.float16, "bf16":torch.bfloat16}[args.dtype]

    tok, model = load_model(args.model_name, device, dtype)
    n_layers, head_dim = resolve_dims(model.config)
    aes = load_aes(args.ae_path, n_layers, head_dim, args.latent_dim, device, dtype)

    print("⏳ caching prompt …")
    enc, prv_cache = prompt_cache(tok, model, args.prompt)

    print("▶️  baseline continuation")
    baseline = continue_text(tok, model, enc, prv_cache, n_tokens=args.gen_len)

    print("🔧 AE compress→decompress cache")
    cmp_cache = reconstruct_cache(prv_cache, aes)

    print("▶️  compressed continuation")
    compressed = continue_text(tok, model, enc, cmp_cache, n_tokens=args.gen_len)

    print("\n=== BASELINE OUTPUT ===\n", baseline)
    print("\n=== COMPRESSED OUTPUT ===\n", compressed)

if __name__ == "__main__":
    main()