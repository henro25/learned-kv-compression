#!/usr/bin/env python3
"""
benchmark_kv_compression.py
Run perplexity + speed benchmarks for baseline vs. AE‑compressed KV caches.
"""

# ─── suppress noisy backend logs (cuDNN, cuBLAS, XLA, HF INFO) ────────────────
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

# ─── std libs / torch / hf ─────────────────────────────────────────────────────
import json, time, argparse
import numpy as np
import torch, torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from datasets import load_dataset
from src.models.autoencoder import Autoencoder

# ─── utility: quantize helper ---------------------------------------------------
def quantize_tensor(x, bits: int):
    if bits is None:
        return x
    scale = 2 ** (bits - 1) - 1
    max_val = x.abs().amax()
    if max_val == 0:
        return x
    x_q = torch.round(torch.clamp(x / max_val * scale, -scale, scale)) / scale
    return x_q * max_val

# ─── compress & reconstruct KV cache -------------------------------------------
def compress_kv_cache(past, aes, bits=None):
    rec = []
    for (k, v), ae in zip(past, aes):
        B, H, S, D = k.shape
        k_f = k.contiguous().view(-1, D)
        v_f = v.contiguous().view(-1, D)

        with torch.no_grad():
            k_lat = ae.encoder(k_f)
            v_lat = ae.encoder(v_f)
            if bits:
                k_lat = quantize_tensor(k_lat, bits)
                v_lat = quantize_tensor(v_lat, bits)
            k_rec = ae.decoder(k_lat).view(B, H, S, D)
            v_rec = ae.decoder(v_lat).view(B, H, S, D)
        rec.append((k_rec, v_rec))

    return DynamicCache.from_legacy_cache(tuple(rec))

# ─── perplexity helpers (unchanged except for compress_kv_cache call) ---------- 
def calculate_perplexity(model, tok, texts, max_len, desc):
    model.eval()
    tot_loss, tot_tok = 0.0, 0
    with torch.no_grad():
        for t in tqdm(texts, desc=desc):
            inp = tok(t, return_tensors="pt", max_length=max_len, truncation=True).to(model.device)
            out = model(**inp)
            logits = out.logits[:, :-1].contiguous()
            labels = inp.input_ids[:, 1:].contiguous()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), reduction="none")
            mask = inp.attention_mask[:, 1:].contiguous().view(-1)
            tot_loss += (loss * mask).sum().item()
            tot_tok  += mask.sum().item()
    return torch.exp(torch.tensor(tot_loss / tot_tok)).item()

def evaluate_with_compressed_cache(model, tok, aes, texts, max_len, bits=None):
    model.eval()
    for ae in aes: ae.eval()
    tot_loss, tot_tok = 0.0, 0
    with torch.no_grad():
        for txt in tqdm(texts, desc="Eval compressed"):
            inp = tok(txt, return_tensors="pt", truncation=True, max_length=max_len).to(model.device)
            ids, mask = inp.input_ids, inp.attention_mask
            past = None
            for t in range(ids.size(1) - 1):
                if not mask[0, t + 1]:
                    continue
                out = model(ids[:, t:t+1], attention_mask=mask[:, t:t+1], past_key_values=past, use_cache=True)
                past = compress_kv_cache(out.past_key_values, aes, bits)
                logit = out.logits[:, -1, :]
                loss  = F.cross_entropy(logit, ids[:, t+1], reduction="none")
                tot_loss += loss.item(); tot_tok += 1
    return torch.exp(torch.tensor(tot_loss / tot_tok)).item()

def evaluate_with_kv_quant(model, tok, texts, max_len, bits=None):
    model.eval()
    tot_loss, tot_tok = 0.0, 0
    with torch.no_grad():
        for txt in tqdm(texts, desc="Eval quantized baseline"):
            inp = tok(txt, return_tensors="pt", truncation=True, max_length=max_len).to(model.device)
            ids, mask = inp.input_ids, inp.attention_mask
            past = None
            for t in range(ids.size(1) - 1):
                if not mask[0, t + 1]:
                    continue
                out = model(ids[:, t:t+1], attention_mask=mask[:, t:t+1], past_key_values=past, use_cache=True)
                new_past = []
                for k, v in out.past_key_values:
                    kq = quantize_tensor(k, bits) if bits else k
                    vq = quantize_tensor(v, bits) if bits else v
                    new_past.append((kq, vq))
                past = DynamicCache.from_legacy_cache(tuple(new_past))
                loss = F.cross_entropy(out.logits[:, -1, :], ids[:, t+1], reduction="none")
                tot_loss += loss.item(); tot_tok += 1
    return torch.exp(torch.tensor(tot_loss / tot_tok)).item()

# ─── longbench wrapper (unchanged except compress fn) --------------------------
def evaluate_longbench(model, tok, aes, cfg, bits=None):
    subsets = ['narrativeqa','hotpotqa','2wikimqa','musique','dureader']
    res = {"baseline":{}, "compressed":{}}
    n_eval = cfg.get("num_eval_texts")
    for s in subsets:
        data = load_dataset("THUDM/LongBench", s, trust_remote_code=True)["test"]["input"]
        texts = [t for t in data if t.strip()][:n_eval or None]
        res["baseline"][s]   = calculate_perplexity(model, tok, texts, cfg["max_seq_len"], f"PPL {s}")
        res["compressed"][s] = evaluate_with_compressed_cache(model, tok, aes, texts, cfg["max_seq_len"], bits)
    return res

# ─── main benchmark runner ------------------------------------------------------
def run_benchmark(model_name, ae_path, lat_dim, out_dir, cfg):
    device = torch.device(cfg.get("device","cuda"))
    dtype  = torch.bfloat16 if cfg["dtype"]=="bf16" else torch.float16 if cfg["dtype"] in ("fp16","f16") else torch.float32

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token; tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map={"":device}, use_cache=True)
    model.config.pad_token_id = tok.pad_token_id

    # derive dims robustly
    num_layers = getattr(model.config, "num_hidden_layers", getattr(model.config,"n_layer"))
    head_dim   = getattr(model.config, "hidden_size", getattr(model.config,"n_embd")) // \
                 getattr(model.config, "num_attention_heads", getattr(model.config,"n_head"))

    ckpt = torch.load(ae_path) if os.path.exists(ae_path) else None
    aes  = []
    for i in range(num_layers):
        ae = Autoencoder(input_dim=head_dim, latent_dim=lat_dim, dtype=dtype).to(device)
        if ckpt: ae.load_state_dict(ckpt[f"layer_{i}"])
        ae.eval(); aes.append(ae)

    # dataset slice
    texts = [t for t in load_dataset("wikitext","wikitext-103-raw-v1")["test"]["text"] if t.strip()]
    if cfg.get("num_eval_texts"): texts = texts[:cfg["num_eval_texts"]]

    bits = cfg.get("quantization_bits")
    base_ppl = evaluate_with_kv_quant(model, tok, texts, cfg["max_seq_len"], bits)
    comp_ppl = evaluate_with_compressed_cache(model, tok, aes, texts, cfg["max_seq_len"], bits)
    longbench = evaluate_longbench(model, tok, aes, cfg, bits)

    # (speed / size benchmarks unchanged – still use compress_kv_cache)
    # ...

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir,"benchmark_results.json"),"w") as f:
        json.dump({
            "baseline_perplexity":base_ppl,
            "compressed_perplexity":comp_ppl,
            "longbench_results":longbench,
            "config":cfg
        }, f, indent=2)
    print(f"Baseline PPL: {base_ppl:.2f} | Compressed PPL: {comp_ppl:.2f}")

# ─── CLI -----------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = json.load(open(args.config))
    run_benchmark(cfg["model_name"], cfg["autoencoder_path"], cfg["latent_dim"], cfg["output_dir"], cfg)