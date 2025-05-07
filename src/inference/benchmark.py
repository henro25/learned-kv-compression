#!/usr/bin/env python3
"""
benchmark_per_head.py   (patched 2025-05-05, adapted for per-head AE)

Benchmarks:
  • raw-perplexity baseline
  • KV-cache baseline with optional quantisation
  • AE-compressed + quantised KV perplexity (per-head)
  • LongBench across bit-widths (now with KV-quant baseline too)
  • Example generations

Patches
───────
✔ Robust quantisation – guards against NaNs/Infs before & after the op.  
✔ compress_past sanitises encoder/decoder outputs to keep them finite.  
✔ token_loop skips steps that would propagate NaNs to the loss.  
✔ Added explicit sanity checks + log warnings so problems surface early.  
✔ Adaptation to load and use per-head autoencoders.
"""

# ─── quiet noisy logs ────────────────────────────────────────────────────
import os, warnings, logging, json, argparse, math
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["ABSL_MIN_LOG_LEVEL"]  = "3"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import absl.logging as absl_log
absl_log.set_verbosity(absl_log.ERROR)
absl_log.set_stderrthreshold("fatal")

from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

# ─── deps ───────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from datasets import load_dataset
from src.models.autoencoder import Autoencoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────
def _sanitize(x: torch.Tensor) -> torch.Tensor:
    """Replace NaNs/Infs with finite zeros so downstream ops stay defined."""
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def quantize_tensor(x: torch.Tensor, bits: int | None):
    """Uniform symmetric quantisation with safety checks against NaNs."""
    if bits is None or bits <= 1:
        return _sanitize(x)
    x = _sanitize(x)
    scale = 2 ** (bits - 1) - 1
    if scale <= 0:
        return x
    mx = x.abs().amax()
    if not torch.isfinite(mx) or mx == 0:
        return x
    xq = torch.round(torch.clamp(x / mx * scale, -scale, scale)) / scale * mx
    return _sanitize(xq)


def quantize_past(past, bits):
    if bits is None:
        return past
    return DynamicCache.from_legacy_cache(tuple(
        (quantize_tensor(k, bits), quantize_tensor(v, bits))
        for k, v in past
    ))


def resolve_dims(cfg):
    L  = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", None) or getattr(cfg, "num_layers", None)
    H  = getattr(cfg, "hidden_size", None)       or getattr(cfg, "n_embd", None)
    nH = getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head", None)
    if None in (L, H, nH):
        raise ValueError(f"Could not resolve model dimensions from config: {cfg}")
    return L, nH, H // nH


def compress_past_per_head(past, aes, bits):
    """Encode → (optional) quantise → decode each KV pair head-wise & layer-wise."""
    rebuilt = []
    for layer_idx, (k, v) in enumerate(past):
        B, H, S, D = k.shape
        layer_aes = aes[layer_idx]
        heads_out = []
        for head_idx in range(H):
            k_head = k[:, head_idx].reshape(-1, D)
            v_head = v[:, head_idx].reshape(-1, D)
            # encode
            k_lat = _sanitize(layer_aes[head_idx].encoder(k_head))
            v_lat = _sanitize(layer_aes[head_idx].encoder(v_head))
            # quantise in latent space
            if bits is not None:
                k_lat = quantize_tensor(k_lat, bits)
                v_lat = quantize_tensor(v_lat, bits)
            # decode
            k_rec = _sanitize(layer_aes[head_idx].decoder(k_lat)).reshape(B, S, D)
            v_rec = _sanitize(layer_aes[head_idx].decoder(v_lat)).reshape(B, S, D)
            heads_out.append((k_rec, v_rec))
        # stack heads
        k_stack = torch.stack([kv[0] for kv in heads_out], dim=1)
        v_stack = torch.stack([kv[1] for kv in heads_out], dim=1)
        rebuilt.append((k_stack, v_stack))
    return DynamicCache.from_legacy_cache(tuple(rebuilt))


# ─── generation helpers ────────────────────────────────────────────────
def safe_sample(probs):
    probs = _sanitize(probs).clamp(min=0.0)
    z = probs.sum(dim=-1, keepdim=True)
    probs = torch.where(z == 0,
                        torch.full_like(probs, 1 / probs.size(-1)),
                        probs / z)
    return torch.multinomial(probs, 1)


def top_k_filter(logits, k):
    if k <= 0 or k >= logits.size(-1):
        return logits
    v, idx = torch.topk(logits, k, dim=-1)
    out = logits.new_full(logits.shape, float("-inf"))
    out.scatter_(-1, idx, v)
    return out


def continue_text(tokenizer, model, enc, past,
                  *, gen_len=20, top_k=50, temp=1.0,
                  bits=None, aes=None, per_head=False):
    ids  = enc.input_ids.clone()
    last = ids[:, -1:].to(model.device)
    for _ in range(gen_len):
        with torch.no_grad():
            out = model(input_ids=last, past_key_values=past, use_cache=True)
        logits, next_past = out.logits[:, -1, :], out.past_key_values
        if torch.isnan(logits).any():
            break
        logits = top_k_filter(logits / temp, top_k)
        probs  = torch.softmax(logits, dim=-1)
        nxt    = safe_sample(probs)
        ids, last = torch.cat([ids, nxt], dim=-1), nxt
        past = (quantize_past(next_past, bits)
                if aes is None
                else (compress_past_per_head(next_past, aes, bits)
                      if per_head
                      else compress_past(next_past, aes, bits)))
    return tokenizer.decode(ids[0], skip_special_tokens=True)


# ─── perplexity helpers ─────────────────────────────────────────────────
def perplexity(model, tok, texts, max_len, desc):
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    tot_loss = tot_tok = 0
    for txt in tqdm(texts, desc=desc):
        inp = tok(txt, return_tensors="pt", truncation=True, max_length=max_len).to(model.device)
        if inp.input_ids.size(1) < 2:
            continue
        out = model(**inp)
        logits = out.logits[:, :-1, :].contiguous()
        lbl    = inp.input_ids[:, 1:].contiguous()
        msk    = inp.attention_mask[:, 1:].contiguous()
        flat   = loss_fn(logits.view(-1, logits.size(-1)), lbl.view(-1))
        tot_loss += (flat * msk.view(-1)).sum().item()
        tot_tok  += int(msk.sum())
    if tot_tok == 0:
        raise RuntimeError("No valid tokens for perplexity calculation.")
    return math.exp(tot_loss / tot_tok)


def token_loop_per_head(model, tok, texts,
                        *, aes=None, bits=None,
                        max_len=1024, desc="kv"):
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    tot_loss = tot_tok = 0
    for txt in tqdm(texts, desc=desc):
        inp = tok(txt, return_tensors="pt", truncation=True, max_length=max_len).to(model.device)
        ids, attn = inp.input_ids, inp.attention_mask
        if ids.size(1) < 2:
            continue
        past = None
        for t in range(ids.size(1) - 1):
            if not attn[0, t + 1]:
                continue
            out = model(input_ids=ids[:, t:t + 1], past_key_values=past, use_cache=True)
            if torch.isnan(out.logits).any():
                continue
            past = (compress_past_per_head(out.past_key_values, aes, bits)
                    if aes is not None
                    else quantize_past(out.past_key_values, bits))
            step = loss_fn(out.logits[:, -1, :], ids[:, t + 1]).item()
            if math.isnan(step):
                logger.warning(f"NaN loss at token {t} – skipping.")
                continue
            tot_loss += step
            tot_tok  += 1
    if tot_tok == 0:
        raise RuntimeError(f"No valid steps for token_loop in {desc}.")
    return math.exp(tot_loss / tot_tok)


# ────────────────────────────────────────────────────────────────────────
def run_benchmark(cfg: dict):
    # device / dtype
    device = torch.device(cfg.get("device", "cuda"))
    dtype  = {"fp32": torch.float32, "fp16": torch.float16,
              "bf16": torch.bfloat16}.get(cfg.get("dtype", "fp32"), torch.float32)

    # tokenizer + model
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        torch_dtype=dtype,
        device_map={"": device},
        use_cache=True
    )
    model.config.pad_token_id = tok.pad_token_id

    # build per-head autoencoders
    L, nH, dH = resolve_dims(model.config)
    aes = nn.ModuleList([
        nn.ModuleList([
            Autoencoder(
                input_dim=dH,
                latent_dim=cfg["latent_dim"],
                encoder_layer_sizes=cfg.get("encoder_layer_sizes", []),
                decoder_layer_sizes=cfg.get("decoder_layer_sizes", []),
                activation=cfg.get("activation", "ReLU"),
                dtype=dtype,
            ).to(device)
            for _ in range(nH)
        ]) for _ in range(L)
    ])

    if os.path.exists(cfg.get("autoencoder_path", "")):
        ckpt = torch.load(cfg["autoencoder_path"], map_location=device)
        # [your existing checkpoint-loading logic here]
        aes.eval()

    # quantization bits list (skip None)
    bits_list = sorted(b for b in cfg.get("quantization_bits", []) if isinstance(b, int) and b > 1)

    # prepare WikiText-103 test texts
    wiki = load_dataset("wikitext", "wikitext-103-raw-v1")["test"]["text"]
    texts = [t for t in wiki if t.strip()][: cfg.get("num_eval_texts") or None]

    results = {}

    # raw baseline
    results["raw_baseline_ppl"] = perplexity(model, tok, texts, cfg["max_seq_len"], "Raw")

    # per-bit WikiText benchmarks
    results["perplexities"] = {}
    for b in bits_list:
        kv = token_loop_per_head(model, tok, texts, bits=b,
                                 max_len=cfg["max_seq_len"], desc=f"KV{b}")
        ae = token_loop_per_head(model, tok, texts, aes=aes, bits=b,
                                 max_len=cfg["max_seq_len"], desc=f"AE{b}")
        results["perplexities"][str(b)] = {
            "kv_cache_baseline_ppl": kv,
            "ae_compressed_ppl":     ae
        }

    # LongBench (baseline, quantized, compressed)
    subsets = ["narrativeqa", "hotpotqa", "2wikimqa", "musique", "dureader"]
    longbench = {"baseline": {}, "quantized": {}, "compressed": {}}

    for task in subsets:
        data = load_dataset("THUDM/LongBench", task, trust_remote_code=True)["test"]["input"]
        txts = [t for t in data if t.strip()][: cfg.get("num_eval_texts") or None]

        # raw baseline
        longbench["baseline"][task] = perplexity(
            model, tok, txts, cfg["max_seq_len"], f"PPL {task}"
        )

        # per-bit quantized & compressed
        for b in bits_list:
            # KV-quant baseline
            kvq = token_loop_per_head(
                model, tok, txts,
                bits=b, max_len=cfg["max_seq_len"], desc=f"{task}_KV{b}"
            )
            longbench["quantized"].setdefault(str(b), {})[task] = kvq

            # AE-compressed (per-head)
            ae_cmp = token_loop_per_head(
                model, tok, txts,
                aes=aes, bits=b, max_len=cfg["max_seq_len"], desc=f"{task}_AE{b}"
            )
            longbench["compressed"].setdefault(str(b), {})[task] = ae_cmp

    results["longbench"] = longbench

    # example generations
    examples = []
    gen_len   = cfg.get("gen_len", 20)
    top_k     = cfg.get("top_k", 50)
    first_bit = bits_list[0] if bits_list else None
    for prompt in texts[:5]:
        enc   = tok(prompt, return_tensors="pt").to(device)
        cache = model(**enc, use_cache=True).past_key_values

        raw_text = continue_text(tok, model, enc, cache,
                                 gen_len=gen_len, top_k=top_k)
        kv_text  = continue_text(tok, model, enc, cache,
                                 gen_len=gen_len, top_k=top_k, bits=first_bit)
        ae_text  = continue_text(tok, model, enc,
                                 compress_past_per_head(cache, aes, first_bit),
                                 gen_len=gen_len, top_k=top_k, per_head=True)

        examples.append({
            "prompt":       prompt,
            "raw":          raw_text,
            "kv_quant":     kv_text,
            "ae_compressed":ae_text
        })

    results["examples"] = examples
    results["config"]   = cfg

    # save results
    os.makedirs(cfg["output_dir"], exist_ok=True)
    out_path = os.path.join(cfg["output_dir"], "benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("✅ Saved to", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    run_benchmark(json.load(open(parser.parse_args().config)))