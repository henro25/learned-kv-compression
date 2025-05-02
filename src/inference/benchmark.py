#!/usr/bin/env python3
"""
benchmark.py

Run perplexity + cache-size + timing benchmarks for baseline vs. AE-compressed KV caches.
"""

# ─── suppress noisy backend logs ────────────────────────────────────────────────
import os, warnings, logging
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

# ─── standard imports ───────────────────────────────────────────────────────────
import json, time, argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from datasets import load_dataset
from src.models.autoencoder import Autoencoder

# ─── helper: quantization -------------------------------------------------------
def quantize_tensor(x: torch.Tensor, bits: int):
    scale = 2 ** (bits - 1) - 1
    max_val = x.abs().amax()
    if max_val == 0:
        return x
    x_q = torch.round(torch.clamp(x / max_val * scale, -scale, scale)) / scale
    return x_q * max_val

# ─── helper: robust config resolution (GPT-2, Qwen, etc) ───────────────────────
def resolve_dims(cfg):
    # layers
    if hasattr(cfg, "num_hidden_layers"):
        n_layers = cfg.num_hidden_layers
    elif hasattr(cfg, "n_layer"):
        n_layers = cfg.n_layer
    else:
        raise ValueError(f"Cannot find number of layers in config {cfg}")
    # hidden size
    if hasattr(cfg, "hidden_size"):
        hidden = cfg.hidden_size
    elif hasattr(cfg, "n_embd"):
        hidden = cfg.n_embd
    else:
        raise ValueError(f"Cannot find hidden_size in config {cfg}")
    # heads
    if hasattr(cfg, "num_attention_heads"):
        heads = cfg.num_attention_heads
    elif hasattr(cfg, "n_head"):
        heads = cfg.n_head
    else:
        raise ValueError(f"Cannot find number of heads in config {cfg}")
    head_dim = hidden // heads
    return n_layers, heads, head_dim

# ─── compress & reconstruct KV cache ------------------------------------------
def compress_kv_cache(past_key_values, autoencoders, quantization_bits=None):
    reconstructed = []
    for (k, v), ae in zip(past_key_values, autoencoders):
        B, H, S, D = k.shape
        k_flat = k.contiguous().view(-1, D)
        v_flat = v.contiguous().view(-1, D)
        with torch.no_grad():
            k_lat = ae.encoder(k_flat)
            v_lat = ae.encoder(v_flat)
            if quantization_bits is not None:
                k_lat = quantize_tensor(k_lat, quantization_bits)
                v_lat = quantize_tensor(v_lat, quantization_bits)
            k_rec = ae.decoder(k_lat).view(B, H, S, D)
            v_rec = ae.decoder(v_lat).view(B, H, S, D)
        reconstructed.append((k_rec, v_rec))
    return DynamicCache.from_legacy_cache(past_key_values=tuple(reconstructed))

# ─── perplexity on raw text ----------------------------------------------------
def calculate_perplexity(model, tokenizer, texts, max_length=1024, desc="Calculating PPL"):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    with torch.no_grad():
        for text in tqdm(texts, desc=desc):
            inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
            out = model(**inp)
            logits = out.logits[..., :-1, :].contiguous()
            labels = inp.input_ids[..., 1:].contiguous()
            mask   = inp.attention_mask[..., 1:].contiguous()
            loss   = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss  += (loss * mask.view(-1)).sum().item()
            total_tokens += mask.view(-1).sum().item()
    return torch.exp(torch.tensor(total_loss / total_tokens)).item()

# ─── evaluate baseline with KV quantization only --------------------------------
def evaluate_with_kv_quant(model, tokenizer, texts, max_length=1024, quantization_bits=None):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    with torch.no_grad():
        for text in tqdm(texts, desc="Eval quant KV baseline"):
            inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
            ids  = inp.input_ids
            mask = inp.attention_mask
            past = None
            for t in range(ids.size(1) - 1):
                if not mask[0, t+1]:
                    continue
                out = model(input_ids=ids[:, t:t+1], attention_mask=mask[:, t:t+1],
                            past_key_values=past, use_cache=True)
                new_past = []
                for kk, vv in out.past_key_values:
                    kk = quantize_tensor(kk, quantization_bits) if quantization_bits is not None else kk
                    vv = quantize_tensor(vv, quantization_bits) if quantization_bits is not None else vv
                    new_past.append((kk, vv))
                past = DynamicCache.from_legacy_cache(past_key_values=tuple(new_past))
                logits = out.logits[..., -1, :]
                loss   = loss_fct(logits, ids[:, t+1])
                total_loss  += loss.item()
                total_tokens += 1
    return torch.exp(torch.tensor(total_loss / total_tokens)).item()

# ─── evaluate with full AE compression on the fly ------------------------------
def evaluate_with_compressed_cache(model, tokenizer, autoencoders, texts, max_length=1024, quantization_bits=None):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    with torch.no_grad():
        for text in tqdm(texts, desc="Eval AE-compressed"):
            inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
            ids  = inp.input_ids
            mask = inp.attention_mask
            past = None
            for t in range(ids.size(1) - 1):
                if not mask[0, t+1]:
                    continue
                out = model(input_ids=ids[:, t:t+1], attention_mask=mask[:, t:t+1],
                            past_key_values=past, use_cache=True)
                past = compress_kv_cache(out.past_key_values, autoencoders, quantization_bits)
                logits = out.logits[..., -1, :]
                loss   = loss_fct(logits, ids[:, t+1])
                total_loss  += loss.item()
                total_tokens += 1
    return torch.exp(torch.tensor(total_loss / total_tokens)).item()

# ─── evaluate LongBench subsets ------------------------------------------------
def evaluate_longbench(model, tokenizer, autoencoders, cfg, quantization_bits=None):
    subsets = ['narrativeqa','hotpotqa','2wikimqa','musique','dureader']
    results = {"baseline": {}, "compressed": {}}
    num_eval = cfg.get("num_eval_texts")
    for s in subsets:
        data = load_dataset("THUDM/LongBench", s, trust_remote_code=True)["test"]["input"]
        texts = [t for t in data if t.strip()][:num_eval or None]
        results["baseline"][s]   = calculate_perplexity(model, tokenizer, texts, cfg["max_seq_len"], f"PPL {s}")
        results["compressed"][s] = evaluate_with_compressed_cache(model, tokenizer,
                                       autoencoders, texts, cfg["max_seq_len"], quantization_bits)
    return results

# ─── main benchmarking function -----------------------------------------------
def run_benchmark(model_name, autoencoder_path, latent_dim, output_dir, cfg):
    device = torch.device(cfg.get("device","cuda"))
    dtype_str = cfg.get("dtype","fp32")
    dtype = torch.bfloat16 if dtype_str=="bf16" else torch.float16 if dtype_str in ("fp16","f16") else torch.float32

    # load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token; tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map={"":device}, use_cache=True
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # resolve dims
    n_layers, n_heads, head_dim = resolve_dims(model.config)

    # load AE checkpoints
    ckpt = torch.load(autoencoder_path) if os.path.exists(autoencoder_path) else None
    autoencoders = []
    for i in range(n_layers):
        ae = Autoencoder(input_dim=head_dim, latent_dim=latent_dim, dtype=dtype).to(device)
        if ckpt:
            ae.load_state_dict(ckpt[f"layer_{i}"])
        ae.eval(); autoencoders.append(ae)

    # prepare texts
    ds = load_dataset("wikitext","wikitext-103-raw-v1")
    texts = [t for t in ds["test"]["text"] if t.strip()]
    if cfg.get("num_eval_texts"):
        texts = texts[:cfg["num_eval_texts"]]

    # loop over quantization bits
    bits_list = cfg.get("quantization_bits", [None])
    summary = {}
    for bits in bits_list:
        b_ppl = evaluate_with_kv_quant(model, tokenizer, texts, cfg["max_seq_len"], bits)
        c_ppl = evaluate_with_compressed_cache(model, tokenizer, autoencoders, texts, cfg["max_seq_len"], bits)
        summary[str(bits)] = {
            "baseline_perplexity": b_ppl,
            "compressed_perplexity": c_ppl
        }

    # optional LongBench
    longbench = evaluate_longbench(model, tokenizer, autoencoders, cfg, bits_list[0] if bits_list else None)

    # write results
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "perplexities": summary,
            "longbench": longbench,
            "config": cfg
        }, f, indent=2)

    print(f"Saved benchmark results to {out_path}")

# ─── CLI entrypoint ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KV cache benchmark")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config")
    args = parser.parse_args()
    cfg = json.load(open(args.config))
    run_benchmark(cfg["model_name"], cfg["autoencoder_path"], cfg["latent_dim"], cfg["output_dir"], cfg)