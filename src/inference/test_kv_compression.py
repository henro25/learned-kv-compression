#!/usr/bin/env python3
"""
test_kv_compression.py

Given a HuggingFace causal-LM (e.g. distilgpt2, Qwen2.5-0.5B) plus a set of
trained layer-wise autoencoders, this script:

  1) Builds a KV cache from a prompt
  2) Continues generation in two modes:
     a) Baseline (raw KV)
     b) Compressed→Decompressed KV
  3) Prints both continuations side by side
  4) Uses top-k sampling to avoid the “always space” trap
  5) Suppresses noisy TF / XLA / cuDNN warnings
"""

import os
# ─── 1) Silence TF / XLA / cuDNN logs ────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"       # FATAL only
os.environ["ABSL_MIN_LOG_LEVEL"] = "3"         # absl logging
import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from src.models.autoencoder import Autoencoder

# ─── Helper: top-k filter ────────────────────────────────────────────────────────
def top_k_logits(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Zero out all but the top k logits (set others to -inf)."""
    if k <= 0:
        return logits
    values, indices = torch.topk(logits, k, dim=-1)
    mask = torch.full_like(logits, float("-inf"))
    mask.scatter_(-1, indices, values)
    return mask

# ─── Model loader & config resolver ─────────────────────────────────────────────
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

def resolve_dims(cfg):
    # layers
    if hasattr(cfg, "num_hidden_layers"):
        L = cfg.num_hidden_layers
    elif hasattr(cfg, "n_layer"):
        L = cfg.n_layer
    else:
        raise ValueError("Unknown layer count in config")
    # hidden dim
    if hasattr(cfg, "hidden_size"):
        H = cfg.hidden_size
    elif hasattr(cfg, "n_embd"):
        H = cfg.n_embd
    else:
        raise ValueError("Unknown hidden_size in config")
    # heads
    if hasattr(cfg, "num_attention_heads"):
        A = cfg.num_attention_heads
    elif hasattr(cfg, "n_head"):
        A = cfg.n_head
    else:
        raise ValueError("Unknown num_attention_heads in config")
    return L, A, H // A

# ─── Load autoencoders ───────────────────────────────────────────────────────────
def load_autoencoders(path, num_layers, head_dim, latent_dim, device, dtype):
    chk = torch.load(path, map_location=device)
    aes = []
    for i in range(num_layers):
        ae = Autoencoder(input_dim=head_dim, latent_dim=latent_dim, dtype=dtype).to(device)
        ae.load_state_dict(chk[f"layer_{i}"])
        ae.eval()
        aes.append(ae)
    return aes

# ─── Build KV cache with mask ───────────────────────────────────────────────────
def build_kv_cache(tok, model, prompt, target_tokens=50):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    ids = inputs.input_ids
    mask = inputs.attention_mask
    cache = None

    for _ in range(target_tokens):
        out = model(input_ids=ids[:, -1:], attention_mask=mask[:, -1:], past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=-1)
        mask = torch.cat([mask, torch.ones_like(next_id)], dim=-1)

    return cache

# ─── Generate with top-k sampling ────────────────────────────────────────────────
def generate_with_cache(tok, model, prompt, cache=None, gen_len=50, top_k=50, temp=1.0):
    # encode prompt + mask
    enc = tok(prompt, return_tensors="pt").to(model.device)
    prompt_ids = enc.input_ids[0].tolist()
    mask = enc.attention_mask

    # if no cache, run prompt through model to populate it
    if cache is None:
        out = model(input_ids=enc.input_ids, attention_mask=mask, use_cache=True)
        cache = out.past_key_values

    last_id = prompt_ids[-1]
    generated = []

    for _ in range(gen_len):
        inp = torch.tensor([[last_id]], device=model.device)
        # we need a dummy mask of ones for this single token
        dummy_mask = torch.ones_like(inp)
        with torch.no_grad():
            out = model(input_ids=inp, attention_mask=dummy_mask, past_key_values=cache, use_cache=True)
        logits = out.logits[:, -1, :] / temp
        filt   = top_k_logits(logits, k=top_k)
        probs  = torch.softmax(filt, dim=-1)
        nxt    = torch.multinomial(probs, num_samples=1).item()
        generated.append(nxt)
        cache = out.past_key_values
        last_id = nxt

    full = prompt_ids + generated
    return tok.decode(full, skip_special_tokens=True)

# ─── Main CLI ────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="distilgpt2")
    p.add_argument("--ae_path",     type=str, required=True)
    p.add_argument("--prompt",      type=str, default="Once upon a time in a land far away,")
    p.add_argument("--gen_len",     type=int, default=50)
    p.add_argument("--latent_dim",  type=int, default=32)
    p.add_argument("--dtype",       choices=["fp32","fp16","bf16"], default="fp32")
    args = p.parse_args()

    # device & dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dt_map = {"fp32":torch.float32, "fp16":torch.float16, "bf16":torch.bfloat16}
    dtype  = dt_map[args.dtype]

    # load model/tokenizer
    tok, model = load_model(args.model_name, device, dtype)
    L, A, head_dim = resolve_dims(model.config)

    # load autoencoders
    aes = load_autoencoders(args.ae_path, L, head_dim, args.latent_dim, device, dtype)

    # 1) build KV
    print("⏳ Building KV cache...")
    cache = build_kv_cache(tok, model, args.prompt, target_tokens=args.gen_len)

    # 2) baseline continuation
    print("▶️  Baseline continuation (top-k sampling)...")
    base_out = generate_with_cache(tok, model, args.prompt, cache=cache, gen_len=args.gen_len)

    # 3) compress & decompress
    print("🔧 Compressing → Decompressing KV cache...")
    rec = []
    for (k,v), ae in zip(cache, aes):
        B,H,S,D = k.shape
        k_flat = k.view(-1,D); v_flat = v.view(-1,D)
        with torch.no_grad():
            k_lat = ae.encoder(k_flat); v_lat = ae.encoder(v_flat)
            k_rec = ae.decoder(k_lat).view(B,H,S,D)
            v_rec = ae.decoder(v_lat).view(B,H,S,D)
        rec.append((k_rec, v_rec))
    comp_cache = DynamicCache.from_legacy_cache(past_key_values=tuple(rec))

    # 4) compressed continuation
    print("▶️  Compressed continuation (top-k sampling)...")
    comp_out = generate_with_cache(tok, model, args.prompt, cache=comp_cache, gen_len=args.gen_len)

    # display
    print("\n=== BASELINE OUTPUT ===\n", base_out)
    print("\n=== COMPRESSED OUTPUT ===\n", comp_out)

if __name__=="__main__":
    main()