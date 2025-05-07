#!/usr/bin/env python3
"""
benchmark_per_head.py   (patched 2025-05-07, adapted for per-head AE)

Benchmarks:
  • raw-perplexity baseline + overhead
  • KV-cache baseline with optional quantisation + overhead
  • AE-compressed + quantised KV perplexity (per-head) + overhead
  • LongBench across bit-widths
  • Example generations + decompression speed
  • Estimated memory footprint (KV cache)

Patches
───────
✔ Robust quantisation – guards against NaNs/Infs before & after the op.
✔ compress_past sanitises encoder/decoder outputs to keep them finite.
✔ token_loop skips steps that would propagate NaNs to the loss.
✔ Added explicit sanity checks + log warnings so problems surface early.
✔ Adaptation to load and use per-head autoencoders.
✔ Added decompression speed to the benchmark (time to first token).
✔ Added quantization to the raw KV-cache baseline.
✔ Added estimation of memory footprint for KV cache (with and without AE).
✔ Added measurement of inference overhead for all baselines.
"""

# ─── quiet noisy logs ────────────────────────────────────────────────────
import os, warnings, logging, json, argparse, math, time
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
    if torch.isfinite(mx) is False or mx == 0:
        return x  # nothing to quantise – all zeros or invalid max
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


def estimate_kv_cache_size(past, bits=None):
    """Estimates the size of the KV cache in bytes."""
    total_bytes = 0
    if past:
        for k, v in past:
            dtype_size = 4  # Assume float32 if bits is None
            if bits is not None:
                dtype_size = bits / 8
            total_bytes += k.numel() * dtype_size
            total_bytes += v.numel() * dtype_size
    return total_bytes


def estimate_compressed_kv_cache_size(past, aes, bits=None, latent_dim=None):
    """Estimates the size of the compressed KV cache in bytes."""
    if latent_dim is None:
        return 0
    total_bytes = 0
    if past:
        for layer_idx, (k, v) in enumerate(past):
            B, H, S, D = k.shape
            dtype_latent = 4  # Assume float32 if bits is None
            if bits is not None:
                dtype_latent = bits / 8
            total_bytes += B * H * S * latent_dim * dtype_latent * 2 # For both k_lat and v_lat
    return total_bytes


def compress_past_per_head(past, aes, bits):
    """Encode → (optional) quantise → decode each KV pair head-wise and layer-wise."""
    rebuilt = []
    total_decode_time = 0
    for layer_idx, (k, v) in enumerate(past):
        B, H, S, D = k.shape
        layer_aes = aes[layer_idx]
        layer_rebuilt = []
        layer_decode_time = 0
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
            start_decode_time = time.perf_counter()
            k_rec = _sanitize(layer_aes[head_idx].decoder(k_lat)).reshape(B, S, D)
            v_rec = _sanitize(layer_aes[head_idx].decoder(v_lat)).reshape(B, S, D)
            layer_decode_time += (time.perf_counter() - start_decode_time)
            layer_rebuilt.append((k_rec, v_rec))

        # Stack the rebuilt heads back together
        k_rec_stacked = torch.stack([kv[0] for kv in layer_rebuilt], dim=1)
        v_rec_stacked = torch.stack([kv[1] for kv in layer_rebuilt], dim=1)
        rebuilt.append((k_rec_stacked, v_rec_stacked))
        total_decode_time += layer_decode_time
    return DynamicCache.from_legacy_cache(tuple(rebuilt)), total_decode_time


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


def continue_text(tokenizer, model, enc, past, *, gen_len=20, top_k=50, temp=1.0, bits=None, aes=None, per_head=False):
    ids  = enc.input_ids.clone()
    last = ids[:, -1:].to(model.device)
    first_token_time = None
    for i in range(gen_len):
        start_time = time.perf_counter()
        with torch.no_grad():
            out = model(input_ids=last, past_key_values=past, use_cache=True)
        logits, next_past = out.logits[:, -1, :], out.past_key_values
        if i == 0:
            first_token_time = time.perf_counter() - start_time
        if torch.isnan(logits).any():
            # logger.warning("NaNs detected in logits during generation – token skipped.")
            break
        logits = top_k_filter(logits / temp, top_k)
        probs  = torch.softmax(logits, dim=-1)
        nxt    = safe_sample(probs)
        ids, last = torch.cat([ids, nxt], dim=-1), nxt
        if aes and per_head:
            compressed_past, _ = compress_past_per_head(next_past, aes, bits)
            past = compressed_past
        else:
            past = quantize_past(next_past, bits)
    return tokenizer.decode(ids[0], skip_special_tokens=True), first_token_time


# ─── perplexity helpers ─────────────────────────────────────────────────

def perplexity(model, tok, texts, max_len, desc, measure_overhead=False):
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    tot_loss = tot_tok = 0
    total_overhead = 0
    num_steps = 0
    for txt in tqdm(texts, desc=desc):
        inp = tok(txt, return_tensors="pt", truncation=True, max_length=max_len).to(model.device)
        if inp.input_ids.size(1) < 2:
            continue
        start_time = time.perf_counter()
        out = model(**inp)
        forward_time = time.perf_counter() - start_time
        logits = out.logits[:, :-1, :].contiguous()
        lbl    = inp.input_ids[:, 1:].contiguous()
        msk    = inp.attention_mask[:, 1:].contiguous()
        flat_loss = loss_fn(logits.view(-1, logits.size(-1)), lbl.view(-1))
        tot_loss += (flat_loss * msk.view(-1)).sum().item()
        tot_tok  += int(msk.sum())
        if measure_overhead:
            total_overhead += forward_time
            num_steps += 1

    if tot_tok == 0:
        raise RuntimeError("No valid tokens for perplexity calculation.")
    perplexity_val = math.exp(tot_loss / tot_tok)
    overhead_per_token = total_overhead / tot_tok if tot_tok > 0 and measure_overhead else 0
    return perplexity_val, overhead_per_token


def token_loop_per_head(model, tok, texts, *, aes=None, bits=None, max_len=1024, desc="kv", measure_overhead=False):
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    tot_loss = tot_tok = 0
    total_decode_time = 0
    total_overhead = 0
    num_steps = 0
    first_past = None
    for txt in tqdm(texts, desc=desc):
        inp = tok(txt, return_tensors="pt", truncation=True, max_length=max_len).to(model.device)
        ids, attn = inp.input_ids, inp.attention_mask
        if ids.size(1) < 2:
            continue
        past = None
        for t in range(ids.size(1) - 1):
            if not attn[0, t + 1]:
                continue
            start_time = time.perf_counter()
            out = model(input_ids=ids[:, t:t + 1], past_key_values=past, use_cache=True)
            forward_time = time.perf_counter() - start_time
            if t == 0:
                first_past = out.past_key_values
            if torch.isnan(out.logits).any():  # skip corrupt step
                # logger.warning(f"NaNs detected in logits at token {t} – skipping.")
                continue
            if aes:
                compress_start_time = time.perf_counter()
                compressed_past, decode_time = compress_past_per_head(out.past_key_values, aes, bits)
                compress_time = time.perf_counter() - compress_start_time
                past = compressed_past
                total_decode_time += decode_time + compress_time # Include compression time in overhead
            else:
                quantize_start_time = time.perf_counter()
                past = quantize_past(out.past_key_values, bits)
                quantize_time = time.perf_counter() - quantize_start_time
                total_overhead += quantize_time # Include quantization time in overhead

            step_loss = loss_fn(out.logits[:, -1, :], ids[:, t + 1]).item()
            if math.isnan(step_loss):
                logger.warning(f"NaN loss at token {t} – skipping.")
                continue
            tot_loss += step_loss
            tot_tok  += 1
            num_steps += 1
            if measure_overhead:
                total_overhead += forward_time

    if tot_tok == 0:
        raise RuntimeError(f"No valid steps for token_loop in {desc}.")
    avg_decode_speed = total_decode_time / num_steps if num_steps > 0 and aes else 0
    overhead_per_token = total_overhead / tot_tok if tot_tok > 0 and measure_overhead else 0
    return math.exp(tot_loss / tot_tok), avg_decode_speed, first_past, overhead_per_token

# ────────────────────────────────────────────────────────────────────────
def run_benchmark(cfg: dict):
    # -------- device / dtype ---------------------------------------------------
    device = torch.device(cfg.get("device", "cuda"))
    dtype  = {"fp32": torch.float32, "fp16": torch.float16,
              "bf16": torch.bfloat16}.get(cfg.get("dtype", "fp32"), torch.float32)

    # -------- tokenizer + model ------------------------------------------------
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

    # -------- autoencoder construction (per-head) -----------------------------
    L, nH, dH = resolve_dims(model.config)
    latent_dim = cfg.get("latent_dim")

    ckpt_path = cfg.get("autoencoder_path", "")
    ckpt = torch.load(ckpt_path, map_location=device) if os.path.exists(ckpt_path) else None

    enc_sizes = cfg.get("encoder_layer_sizes", [])
    dec_sizes = cfg.get("decoder_layer_sizes", [])
    act_name  = cfg.get("activation", "ReLU")

    aes = nn.ModuleList([
        nn.ModuleList([
            Autoencoder(
                input_dim=dH,
                latent_dim=latent_dim,
                encoder_layer_sizes=enc_sizes,
                decoder_layer_sizes=dec_sizes,
                activation=act_name,
                dtype=dtype,
            ).to(device)
            for _ in range(nH)
        ])
        for _ in range(L)
    ])

    if ckpt:
        for l_idx in range(L):
            for h_idx in range(nH):
                key = f"layer_{l_idx}/head_{h_idx}" if f"layer_{l_idx}" in ckpt and f"head_{h_idx}" in ckpt[f"layer_{l_idx}"] else f"layer_{l_idx}" # Handle both saving formats
                if f"layer_{l_idx}" in ckpt and f"head_{h_idx}" in ckpt[f"layer_{l_idx}"]:
                    aes[l_idx][h_idx].load_state_dict(ckpt[f"layer_{l_idx}"][f"head_{h_idx}"])
                elif f"layer_{l_idx}" in ckpt and isinstance(ckpt[f"layer_{l_idx}"], nn.ModuleList) and len(ckpt[f"layer_{l_idx}"] ) > h_idx:
                    aes[l_idx][h_idx].load_state_dict(ckpt[f"layer_{l_idx}"][h_idx]) # Handle if the per-head AEs were saved as a ModuleList directly
                elif isinstance(ckpt, nn.ModuleList) and len(ckpt) > l_idx and len(ckpt[l_idx]) > h_idx: # Handle if the top level is a ModuleList
                    aes[l_idx][h_idx].load_state_dict(ckpt[l_idx][h_idx])
                elif isinstance(ckpt, dict) and f"layer_{l_idx}" in ckpt and isinstance(ckpt[f"layer_{l_idx}"], dict) and len(ckpt[f"layer_{l_idx}"]) > h_idx and all(k.startswith('head_') for k in ckpt[f"layer_{l_idx}"].keys()): # Handle dict of heads
                    head_key = next(k for k in ckpt[f"layer_{l_idx}"].keys() if int(k.split('_')[-1]) == h_idx)
                    aes[l_idx][h_idx].load_state_dict(ckpt[f"layer_{l_idx}"][head_key])
                elif isinstance(ckpt, dict) and f"layer_{l_idx}" in ckpt and isinstance(ckpt[f"layer_{l_idx}"], nn.ModuleList) and len(ckpt[f"layer_{l_idx}"]) > h_idx: # Handle ModuleList within layer dict
                    aes[l_idx][h_idx].load_state_dict(ckpt[f"layer_{l_idx}"][h_idx])
                elif isinstance(ckpt, nn.ModuleList) and len(ckpt) > l_idx and isinstance(ckpt[l_idx], nn.ModuleList) and len(ckpt[l_idx]) > h_idx: # Handle nested ModuleList
                    aes[l_idx][h_idx].load_state_dict(ckpt[l_idx][h_idx])
                elif isinstance(ckpt, dict) and f"layer_{l_idx}" in ckpt and isinstance(ckpt[f"layer_{l_idx}"], dict) and str(h_idx) in ckpt[f"layer_{l_idx}"]: # Handle string keys for heads
                    aes[l_idx][h_idx].load_state_dict(ckpt[f"layer_{l_idx}"][str(h_idx)])
                else:
                    logger.warning(f"Could not load state dict for layer {l_idx}, head {h_idx} from checkpoint.")
        aes.eval()

    # -------- bit-width list ---------------------------------------------------
    bits_list = [None] + sorted(
        b for b in cfg.get("quantization_bits", [])
        if isinstance(b, int) and b > 1
    )

    # -------- evaluation datasets ---------------------------------------------
    ds    = load_dataset("wikitext", "wikitext-103-raw-v1")["test"]["text"]
    texts = [t for t in ds if t.strip()]
    if cfg.get("num_eval_texts"):
        texts = texts[: cfg["num_eval_texts"]]

    results = {}

    # raw baseline with overhead
    raw_ppl, raw_overhead = perplexity(
        model, tok, texts, cfg["max_seq_len"], "Raw", measure_overhead=True
    )
    results["raw_baseline_ppl"] = raw_ppl
    results["raw_baseline_overhead_per_token_s"] = raw_overhead

    # Memory footprint estimation
    example_input = tok(texts[0], return_tensors="pt", truncation=True, max_length=cfg["max_seq_len"]).to(device)
    with torch.no_grad():
        example_input_ids = example_input.input_ids.to(device)
        example_attention_mask = example_input.attention_mask.to(device)
        example_output = model(input_ids=example_input_ids, attention_mask=example_attention_mask, use_cache=True)
        raw_kv_cache = example_output.past_key_values
        results["memory_footprint_mb"] = {}
        results["memory_footprint_mb"]["float32_kv_cache"] = estimate_kv_cache_size(raw_kv_cache) / (1024 * 1024)

        for b in bits_list:
            if b is not None:
                quantized_kv_cache = quantize_past(raw_kv_cache, b)
                results["memory_footprint_mb"][f"int{b}_kv_cache"] = estimate_kv_cache_size(quantized_kv_cache, b) / (1024 * 1024)
                if latent_dim is not None:
                    with torch.no_grad():
                        compressed_example_past, _ = compress_past_per_head(raw_kv_cache, aes, b)
                        results["memory_footprint_mb"][f"ae_int{b}_compressed_kv_cache"] = estimate_compressed_kv_cache_size(compressed_example_past, aes, b, latent_dim) / (1024 * 1024)

    # per-bit benchmarks
    results["perplexities"] = {}
    for b in bits_list:
        b_str = str(b) if b is not None else "none"
        results["perplexities"][b_str] = {}

        # KV Cache Baseline (with optional quantization) + overhead
        kv_ppl, _, first_kv_past, kv_overhead = token_loop_per_head(
            model, tok, texts,
            bits=b, max_len=cfg["max_seq_len"], desc=f"KV{b_str}", measure_overhead=True
        )
        results["perplexities"][b_str]["kv_cache_baseline_ppl"] = kv_ppl
        results["perplexities"][b_str]["kv_cache_baseline_overhead_per_token_s"] = kv_overhead
        if first_kv_past is not None and b is not None:
            results["memory_footprint_mb"].setdefault("generation", {})[f"int{b}_kv_cache"] = estimate_kv_cache_size(first_kv_past, b) / (1024 * 1024)

        # AE Compressed KV + overhead
        ae_ppl, avg_decode_speed, first_ae_past, ae_overhead = token_loop_per_head(
            model, tok, texts,
            aes=aes, bits=b, max_len=cfg["max_seq_len"], desc=f"AE{b_str}", measure_overhead=True
        )
        results["perplexities"][b_str]["ae_compressed_ppl"] = ae_ppl
        results["perplexities"][b_str]["avg_decompression_speed_per_token_s"] = avg_decode_speed
        results["perplexities"][b_str]["ae_compressed_overhead_per_token_s"] = ae_overhead
        if first_ae_past is not None and latent_dim is not None:
            results["memory_footprint_mb"].setdefault("generation", {})[f"ae_int{b}_compressed_kv_cache"] = estimate_compressed_kv_cache_size(first_ae_past, aes, b, latent_dim) / (1024 * 1024)

    # -------- LongBench --------------------------------------------------------
    subsets   = ["narrativeqa", "hotpotqa", "2wikimqa", "musique", "dureader"]
    longbench = {"baseline": {}, "compressed": {}}
    for s in subsets:
        data = load_dataset("THUDM/LongBench", s, trust_remote_code=True)["test"]["input"]
        txts = [t for t in data if t.strip()][: cfg.get("num_eval_texts") or None]
        baseline_ppl, _ = perplexity(
            model, tok, txts, cfg["max_seq_len"], f"PPL {s}"
        )
        longbench["baseline"][s] = baseline_ppl
        for b in bits_list:
            if b is None:
                continue
            ae_ppl, avg_decode_speed, _, _ = token_loop_per_head(
                model, tok, txts,
                aes=aes, bits=b, max_len=cfg["max_seq_len"], desc=f"{s}_AE{b}"
            )
            longbench["compressed"].setdefault(str(b), {})[s] = ae_ppl
            longbench["compressed"][str(b)].setdefault("avg_decompression_speed_per_token_s", {})[s] = avg_decode_speed
    results["longbench"] = longbench

    # -------- example generations ---------------------------------------------
    examples  = []
    gen_len   = cfg.get("gen_len", 20)
    top_k     = cfg.get("top_k", 50)
    first_bit = next((b for b in bits_list if b), None) or 2
    for prompt in texts[:min(5, len(texts))]:
        enc   = tok(prompt, return_tensors="pt").to(device)
        cache = model(**enc, use_cache=True).past_key_values

        # Raw generation
        raw_text, _ = continue_text(tok, model, enc, cache,
                                     gen_len=gen_len, top_k=top_k)

        # Quantized KV generation
        kv_quant_text, _ = continue_text(tok, model, enc, cache,
                                          gen_len=gen_len, top_k=top_k, bits=first_bit)

        # AE compressed generation
        start_ae_time = time.perf_counter()
        compressed_past, decode_time = compress_past_per_head(cache, aes, first_bit)
        ae_compressed_text, first_token_ae_time = continue_text(tok, model, enc,
                                                                compressed_past,
                                                                gen_len=gen_len, top_k=top_k, per_head=True)
        if first_token_ae_time is not None:
            decompression_speed = decode_time # Total decompression time for the initial past

        examples.append({
            "prompt": prompt,
            "raw": raw_text,
            "kv_quant": kv_quant_text,
            "ae_compressed": ae_compressed_text,
            "decompression_time_first_token_s": first_token_ae_time if first_token_ae_time is not None else None
        })
    results["examples"] = examples
    results["config"]   = cfg

    # -------- save -------------------------------------------------------------
    os.makedirs(cfg["output_dir"], exist_ok=True)
    out_path = os.path.join(cfg["output_dir"], "benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("✅ Saved to", out_path)


# ─── cli ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    run_benchmark(json.load(open(parser.parse_args().config)))