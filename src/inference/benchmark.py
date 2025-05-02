#!/usr/bin/env python3
"""
benchmark.py

Run perplexity + cache-size + timing benchmarks for baseline vs. AE-compressed KV caches,
and include a few example continuations in the output JSON.
"""

# ─── Suppress noisy logs ─────────────────────────────────────────────────────────
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

# ─── Standard imports ───────────────────────────────────────────────────────────
import json, argparse, time
import numpy as np
import torch, torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from datasets import load_dataset
from src.models.autoencoder import Autoencoder

# ─── Helpers ────────────────────────────────────────────────────────────────────

def quantize_tensor(x: torch.Tensor, bits: int):
    scale = 2 ** (bits - 1) - 1
    m = x.abs().amax()
    if m == 0: return x
    x_q = torch.round(torch.clamp(x / m * scale, -scale, scale)) / scale
    return x_q * m

def resolve_dims(cfg):
    if hasattr(cfg, "num_hidden_layers"):
        n_layers = cfg.num_hidden_layers
    elif hasattr(cfg, "n_layer"):
        n_layers = cfg.n_layer
    else:
        raise ValueError("Missing layer count in config")
    if hasattr(cfg, "hidden_size"):
        hidden = cfg.hidden_size
    elif hasattr(cfg, "n_embd"):
        hidden = cfg.n_embd
    else:
        raise ValueError("Missing hidden_size in config")
    if hasattr(cfg, "num_attention_heads"):
        heads = cfg.num_attention_heads
    elif hasattr(cfg, "n_head"):
        heads = cfg.n_head
    else:
        raise ValueError("Missing num_attention_heads in config")
    return n_layers, heads, hidden // heads

def compress_kv_cache(past, aes, bits=None):
    rebuilt = []
    for (k, v), ae in zip(past, aes):
        B,H,S,D = k.shape
        k_flat = k.contiguous().view(-1, D)
        v_flat = v.contiguous().view(-1, D)
        with torch.no_grad():
            k_lat = ae.encoder(k_flat)
            v_lat = ae.encoder(v_flat)
            if bits is not None:
                k_lat = quantize_tensor(k_lat, bits)
                v_lat = quantize_tensor(v_lat, bits)
            k_rec = ae.decoder(k_lat).view(B,H,S,D)
            v_rec = ae.decoder(v_lat).view(B,H,S,D)
        rebuilt.append((k_rec, v_rec))
    return DynamicCache.from_legacy_cache(tuple(rebuilt))

# Prompt‐only cache builder
def prompt_cache(tokenizer, model, prompt: str):
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**enc, use_cache=True)
    return enc, out.past_key_values

# One‐step generation + top‐k sampling, updating mask & position_ids
def continue_text(tokenizer, model, enc, past, gen_len=20, top_k=50, temp=1.0):
    ids   = enc.input_ids.clone()
    pos   = ids.size(1)
    last  = ids[:, -1:]

    for _ in range(gen_len):
        mask     = torch.ones_like(last)
        pos_ids  = torch.tensor([[pos]], device=model.device)
        with torch.no_grad():
            out = model(
                input_ids=last,
                attention_mask=mask,
                position_ids=pos_ids,
                past_key_values=past,
                use_cache=True
            )
        logits = out.logits[:, -1, :] / temp
        # top-k filter
        vals, idx = torch.topk(logits, top_k, dim=-1)
        filt = logits.new_full(logits.shape, float("-inf"))
        filt.scatter_(-1, idx, vals)
        probs = torch.softmax(filt, dim=-1)
        nxt   = torch.multinomial(probs, num_samples=1)
        ids   = torch.cat([ids, nxt], dim=-1)
        last  = nxt
        past  = out.past_key_values
        pos  += 1

    return tokenizer.decode(ids[0], skip_special_tokens=True)

# ─── Perplexity routines ────────────────────────────────────────────────────────

def calculate_perplexity(model, tok, texts, max_len, desc):
    model.eval()
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    tot_loss, tot_tok = 0.0, 0
    with torch.no_grad():
        for t in tqdm(texts, desc=desc):
            inp = tok(t, return_tensors="pt", truncation=True, max_length=max_len).to(model.device)
            out = model(**inp)
            logits = out.logits[..., :-1, :].contiguous()
            labels = inp.input_ids[..., 1:].contiguous()
            mask   = inp.attention_mask[..., 1:].contiguous()
            loss   = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            tot_loss += (loss * mask.view(-1)).sum().item()
            tot_tok  += mask.view(-1).sum().item()
    return torch.exp(torch.tensor(tot_loss / tot_tok)).item()

def evaluate_with_kv_quant(model, tok, texts, max_len, bits=None):
    model.eval()
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    tot_loss, tot_tok = 0.0, 0
    with torch.no_grad():
        for t in tqdm(texts, desc="Eval quantized KV baseline"):
            inp = tok(t, return_tensors="pt", truncation=True, max_length=max_len).to(model.device)
            ids, mask = inp.input_ids, inp.attention_mask
            past = None
            for i in range(ids.size(1)-1):
                if not mask[0,i+1]: continue
                out = model(ids[:,i:i+1], attention_mask=mask[:,i:i+1],
                            past_key_values=past, use_cache=True)
                # optionally quantize raw KV
                new_past = []
                for kk,vv in out.past_key_values:
                    kk = quantize_tensor(kk, bits) if bits is not None else kk
                    vv = quantize_tensor(vv, bits) if bits is not None else vv
                    new_past.append((kk,vv))
                past = DynamicCache.from_legacy_cache(tuple(new_past))
                loss = loss_fct(out.logits[..., -1, :], ids[:,i+1])
                tot_loss += loss.item()
                tot_tok  += 1
    return torch.exp(torch.tensor(tot_loss / tot_tok)).item()

def evaluate_with_compressed_cache(model, tok, aes, texts, max_len, bits=None):
    model.eval()
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    tot_loss, tot_tok = 0.0, 0
    with torch.no_grad():
        for t in tqdm(texts, desc="Eval AE-compressed KV"):
            inp = tok(t, return_tensors="pt", truncation=True, max_length=max_len).to(model.device)
            ids, mask = inp.input_ids, inp.attention_mask
            past = None
            for i in range(ids.size(1)-1):
                if not mask[0,i+1]: continue
                out = model(ids[:,i:i+1], attention_mask=mask[:,i:i+1],
                            past_key_values=past, use_cache=True)
                past = compress_kv_cache(out.past_key_values, aes, bits)
                loss = loss_fct(out.logits[..., -1, :], ids[:,i+1])
                tot_loss += loss.item()
                tot_tok  += 1
    return torch.exp(torch.tensor(tot_loss / tot_tok)).item()

def evaluate_longbench(model, tok, aes, cfg, bits=None):
    subsets = ['narrativeqa','hotpotqa','2wikimqa','musique','dureader']
    res = {"baseline":{}, "compressed":{}}
    n_eval = cfg.get("num_eval_texts")
    for s in subsets:
        data = load_dataset("THUDM/LongBench", s, trust_remote_code=True)["test"]["input"]
        texts = [x for x in data if x.strip()][:n_eval or None]
        res["baseline"][s]   = calculate_perplexity(model, tok, texts, cfg["max_seq_len"], f"PPL {s}")
        res["compressed"][s] = evaluate_with_compressed_cache(model, tok, aes, texts, cfg["max_seq_len"], bits)
    return res

# ─── Main runner ───────────────────────────────────────────────────────────────
def run_benchmark(model_name, ae_path, lat_dim, out_dir, cfg):
    device = torch.device(cfg.get("device","cuda"))
    dtype = {"fp32":torch.float32,"fp16":torch.float16,"bf16":torch.bfloat16}[cfg.get("dtype","fp32")]

    # load model
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token; tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map={"":device}, use_cache=True
    )
    model.config.pad_token_id = tok.pad_token_id

    # resolve dims & load AEs
    n_layers, _, head_dim = resolve_dims(model.config)
    ckpt = torch.load(ae_path, map_location=device) if os.path.exists(ae_path) else None
    aes = []
    for i in range(n_layers):
        ae = Autoencoder(input_dim=head_dim, latent_dim=lat_dim, dtype=dtype).to(device)
        if ckpt: ae.load_state_dict(ckpt[f"layer_{i}"])
        ae.eval(); aes.append(ae)

    # prepare dataset
    ds = load_dataset("wikitext","wikitext-103-raw-v1")["test"]["text"]
    texts = [x for x in ds if x.strip()]
    if cfg.get("num_eval_texts"):
        texts = texts[:cfg["num_eval_texts"]]

    # perplexities per bit
    bits_list = cfg.get("quantization_bits", [None])
    ppl_summary = {}
    for bits in bits_list:
        b = evaluate_with_kv_quant(model, tok, texts, cfg["max_seq_len"], bits)
        c = evaluate_with_compressed_cache(model, tok, aes, texts, cfg["max_seq_len"], bits)
        ppl_summary[str(bits)] = {"baseline_ppl": b, "compressed_ppl": c}

    # longbench
    longbench = evaluate_longbench(model, tok, aes, cfg, bits_list[0])

    # --- new: example continuations ---
    gen_len = cfg.get("gen_len", 20)
    examples = []
    for prompt in texts[:5]:
        enc, cache = prompt_cache(tok, model, prompt)
        base_out = continue_text(tok, model, enc, cache, gen_len=gen_len)
        comp_cache = compress_kv_cache(cache, aes, bits_list[0])
        comp_out   = continue_text(tok, model, enc, comp_cache, gen_len=gen_len)
        examples.append({
            "prompt": prompt,
            "baseline": base_out,
            "compressed": comp_out
        })

    # write everything
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir,"benchmark_results.json"),"w") as f:
        json.dump({
            "perplexities": ppl_summary,
            "longbench": longbench,
            "examples": examples,
            "config": cfg
        }, f, indent=2)

    print(f"✅ Results and examples saved to {out_dir}/benchmark_results.json")

if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = json.load(open(args.config))
    run_benchmark(cfg["model_name"], cfg["autoencoder_path"], cfg["latent_dim"], cfg["output_dir"], cfg)