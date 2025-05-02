#!/usr/bin/env python3
"""
test_kv_compression.py  –  clean + correct KV‑cache comparison

• Silences TF / XLA / cuDNN warnings
• Works with GPT‑2 style models *and* Qwen2 family
• Builds a KV cache from the prompt once
• Continues generation step‑by‑step, updating attention_mask / position_ids / cache
• Uses top‑k sampling so we never get “all spaces”
"""

# ─── silence nearly everything *before* heavy libs are imported ────────────────
import os, warnings, logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["ABSL_MIN_LOG_LEVEL"]  = "3"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# silence absl‑cpp backend warnings
import absl.logging as absl_log
absl_log.set_verbosity(absl_log.ERROR)
absl_log.set_stderrthreshold("fatal")

from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

# ───────────────────────────────────────────────────────────────────────────────
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from src.models.autoencoder import Autoencoder

# ---- helper -------------------------------------------------------------------
def top_k_logits(logits: torch.Tensor, k: int = 50) -> torch.Tensor:
    if k <= 0 or k >= logits.size(-1):
        return logits
    v, idx = torch.topk(logits, k, dim=-1)
    out = logits.new_full(logits.shape, float("-inf"))
    out.scatter_(-1, idx, v)
    return out

def resolve_dims(cfg):
    def grab(*names):
        for n in names:
            if hasattr(cfg, n):
                return getattr(cfg, n)
        raise ValueError(f"Missing any of {names}")
    n_layers = grab("num_hidden_layers", "n_layer")
    n_heads  = grab("num_attention_heads", "n_head")
    hidden   = grab("hidden_size", "n_embd")
    head_dim = hidden // n_heads
    return n_layers, head_dim

# ---- model / tokenizer --------------------------------------------------------
def load_model(name, device, dtype):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=dtype, device_map={"": device}, use_cache=True
    )
    model.eval()
    return tok, model

# ---- autoencoders -------------------------------------------------------------
def load_aes(path, n_layers, head_dim, lat, device, dtype):
    ckpt = torch.load(path, map_location=device)
    aes  = []
    for i in range(n_layers):
        ae = Autoencoder(input_dim=head_dim, latent_dim=lat, dtype=dtype).to(device)
        ae.load_state_dict(ckpt[f"layer_{i}"])
        ae.eval()
        aes.append(ae)
    return aes

# ---- prompt cache -------------------------------------------------------------
def prompt_cache(tok, model, prompt):
    enc = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**enc, use_cache=True)
    return enc, out.past_key_values

# ---- single‑step helper -------------------------------------------------------
def step(model, last_token, past, pos_id):
    mask = torch.ones_like(last_token)
    with torch.no_grad():
        out = model(
            input_ids=last_token,
            attention_mask=mask,
            position_ids=pos_id,
            past_key_values=past,
            use_cache=True,
        )
    return out.logits[:, -1, :], out.past_key_values

# ---- top‑k sampling loop ------------------------------------------------------
def continue_text(tok, model, enc, past, n_tokens=50, k=50, temp=1.0):
    ids   = enc.input_ids.clone()
    pos   = ids.size(1)
    last  = ids[:, -1:]

    for _ in range(n_tokens):
        logits, past = step(model, last, past, torch.tensor([[pos]], device=ids.device))
        probs  = torch.softmax(top_k_logits(logits / temp, k), dim=-1)
        next_t = torch.multinomial(probs, 1)
        ids    = torch.cat([ids, next_t], dim=-1)
        last   = next_t
        pos   += 1

    return tok.decode(ids[0], skip_special_tokens=True)

# ---- reconstruct cache with contiguous/reshape fix ---------------------------
def reconstruct_cache(past, aes):
    rebuilt = []
    for (k, v), ae in zip(past, aes):
        B, H, S, D = k.shape
        k_flat = k.contiguous().view(-1, D)
        v_flat = v.contiguous().view(-1, D)
        with torch.no_grad():
            k_rec = ae.decoder(ae.encoder(k_flat)).view(B, H, S, D)
            v_rec = ae.decoder(ae.encoder(v_flat)).view(B, H, S, D)
        rebuilt.append((k_rec, v_rec))
    return DynamicCache.from_legacy_cache(tuple(rebuilt))

# ---- main ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="distilgpt2")
    ap.add_argument("--ae_path",    required=True)
    ap.add_argument("--prompt",     default="Once upon a time in a land far away,")
    ap.add_argument("--gen_len",    type=int, default=50)
    ap.add_argument("--latent_dim", type=int, default=32)
    ap.add_argument("--dtype", choices=["fp32","fp16","bf16"], default="fp16")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = {"fp32":torch.float32,"fp16":torch.float16,"bf16":torch.bfloat16}[args.dtype]

    tok, model = load_model(args.model_name, device, dtype)
    n_layers, head_dim = resolve_dims(model.config)
    aes = load_aes(args.ae_path, n_layers, head_dim, args.latent_dim, device, dtype)

    print("⏳ caching prompt …")
    enc, cache_prompt = prompt_cache(tok, model, args.prompt)

    print("▶️  baseline continuation")
    baseline = continue_text(tok, model, enc, cache_prompt, n_tokens=args.gen_len)

    print("🔧 AE compress→decompress cache")
    comp_cache = reconstruct_cache(cache_prompt, aes)

    print("▶️  compressed continuation")
    compressed = continue_text(tok, model, enc, comp_cache, n_tokens=args.gen_len)

    print("\n=== BASELINE OUTPUT ===\n", baseline)
    print("\n=== COMPRESSED OUTPUT ===\n", compressed)

if __name__ == "__main__":
    main()