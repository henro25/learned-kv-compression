#!/usr/bin/env python3
"""
benchmark.py   (2025‑05‑05)

Benchmarks raw, KV‑cache (quantised) and AE‑compressed (quantised) perplexity,
plus LongBench across multiple quantisation bit‑widths.

Changes vs. previous version
────────────────────────────
• filters out bits == 1
• LongBench now evaluated for every (baseline, compressed, bits) combo
"""

# ─── Quiet noisy logs ───────────────────────────────────────────────────────────
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

# ─── Std imports ────────────────────────────────────────────────────────────────
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from datasets import load_dataset
from src.models.autoencoder import Autoencoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ───── Helper functions ────────────────────────────────────────────────────────
def _max_positions(model):
    return getattr(model.config, "n_positions",
           getattr(model.config, "max_position_embeddings", 1024))

def quantize_tensor(x: torch.Tensor, bits: int):
    if bits is None:
        return x
    scale = 2 ** (bits - 1) - 1
    m = x.abs().amax()
    if m == 0:
        return x
    return torch.round(torch.clamp(x / m * scale, -scale, scale)) / scale * m

def quantize_past(past, bits):
    if bits is None:
        return past
    return DynamicCache.from_legacy_cache(tuple(
        (quantize_tensor(k, bits), quantize_tensor(v, bits)) for k, v in past))

def resolve_dims(cfg):
    L = getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer"))
    H = getattr(cfg, "hidden_size", getattr(cfg, "n_embd"))
    nH = getattr(cfg, "num_attention_heads", getattr(cfg, "n_head"))
    return L, nH, H // nH

def compress_past(past, aes, bits):
    rebuilt = []
    for (k, v), ae in zip(past, aes):
        B, H, S, D = k.shape
        k_lat = ae.encoder(k.reshape(-1, D))
        v_lat = ae.encoder(v.reshape(-1, D))
        if bits is not None:
            k_lat, v_lat = quantize_tensor(k_lat, bits), quantize_tensor(v_lat, bits)
        k_rec = ae.decoder(k_lat).reshape(B, H, S, D)
        v_rec = ae.decoder(v_lat).reshape(B, H, S, D)
        rebuilt.append((k_rec, v_rec))
    return DynamicCache.from_legacy_cache(tuple(rebuilt))

def safe_sample(probs):
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
    z = probs.sum(dim=-1, keepdim=True)
    probs = torch.where(z == 0, torch.full_like(probs, 1/probs.size(-1)), probs / z)
    return torch.multinomial(probs, 1)

def top_k_filter(logits, k):
    if k <= 0 or k >= logits.size(-1):
        return logits
    v, idx = torch.topk(logits, k, dim=-1)
    out = logits.new_full(logits.shape, float("-inf"))
    out.scatter_(-1, idx, v)
    return out


# ───── Evaluation primitives ───────────────────────────────────────────────────
def perplexity(model, tok, texts, max_len, desc):
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    tot_loss = tot_tok = 0
    for txt in tqdm(texts, desc=desc):
        inp = tok(txt, return_tensors="pt", truncation=True, max_length=max_len).to(model.device)
        out = model(**inp)
        logits = out.logits[:, :-1, :].contiguous()
        lbl = inp.input_ids[:, 1:].contiguous()
        msk = inp.attention_mask[:, 1:].contiguous()
        loss = loss_fn(logits.view(-1, logits.size(-1)), lbl.view(-1))
        tot_loss += (loss * msk.view(-1)).sum().item()
        tot_tok  += int(msk.sum())
    return math.exp(tot_loss / tot_tok) if tot_tok else float("nan")

def token_loop(model, tok, texts, *, aes=None, bits=None, max_len=1024, desc="kv"):
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    tot_loss = tot_tok = 0
    for txt in tqdm(texts, desc=desc):
        inp = tok(txt, return_tensors="pt", truncation=True, max_length=max_len).to(model.device)
        ids, attn = inp.input_ids, inp.attention_mask
        past = None
        for t in range(ids.size(1)-1):
            if not attn[0, t+1]:
                continue
            out = model(input_ids=ids[:, t:t+1], past_key_values=past, use_cache=True)
            past = compress_past(out.past_key_values, aes, bits) if aes else quantize_past(out.past_key_values, bits)
            tot_loss += loss_fn(out.logits[:, -1, :], ids[:, t+1]).item()
            tot_tok  += 1
    return math.exp(tot_loss / tot_tok) if tot_tok else float("nan")


# ───── Main benchmark ──────────────────────────────────────────────────────────
def run_benchmark(cfg):
    device = torch.device(cfg.get("device", "cuda"))
    dtype  = {"fp32":torch.float32,"fp16":torch.float16,"bf16":torch.bfloat16}[cfg.get("dtype","fp32")]

    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg["model_name"], torch_dtype=dtype,
                                                 device_map={"":device}, use_cache=True)
    model.config.pad_token_id = tok.pad_token_id

    # autoencoders
    L, _, dH = resolve_dims(model.config)
    ckpt = torch.load(cfg["autoencoder_path"], map_location=device) if os.path.exists(cfg["autoencoder_path"]) else None
    aes = []
    for i in range(L):
        ae = Autoencoder(dH, cfg["latent_dim"], dtype=dtype).to(device)
        if ckpt: ae.load_state_dict(ckpt[f"layer_{i}"])
        ae.eval(); aes.append(ae)

    # texts
    ds = load_dataset("wikitext","wikitext-103-raw-v1")["test"]["text"]
    texts = [t for t in ds if t.strip()]
    if cfg.get("num_eval_texts"): texts = texts[:cfg["num_eval_texts"]]

    # bit‑widths (skip 1)
    bits_list = [b for b in cfg.get("quantization_bits",[None]) if b not in (1,)]
    results = {"raw_baseline_ppl": perplexity(model, tok, texts, cfg["max_seq_len"], "Raw") , "perplexities": {}}

    for b in bits_list:
        kv_ppl = token_loop(model, tok, texts, bits=b, max_len=cfg["max_seq_len"], desc=f"KV{b}")
        ae_ppl = token_loop(model, tok, texts, aes=aes, bits=b, max_len=cfg["max_seq_len"], desc=f"AE{b}")
        results["perplexities"][str(b)] = {"kv_cache_baseline_ppl": kv_ppl, "ae_compressed_ppl": ae_ppl}

    # LongBench: baseline once, compressed per bit‑width
    subsets = ["narrativeqa","hotpotqa","2wikimqa","musique","dureader"]
    longbench = {"baseline": {}, "compressed": {str(b):{} for b in bits_list}}
    for s in subsets:
        data = load_dataset("THUDM/LongBench", s, trust_remote_code=True)["test"]["input"]
        txts = [t for t in data if t.strip()][: cfg.get("num_eval_texts") or None]
        longbench["baseline"][s] = perplexity(model, tok, txts, cfg["max_seq_len"], f"PPL {s}")
        for b in bits_list:
            longbench["compressed"][str(b)][s] = token_loop(model, tok, txts, aes=aes, bits=b,
                                                            max_len=cfg["max_seq_len"], desc=f"{s} AE{b}")
    results["longbench"] = longbench

    # examples
    examples=[]
    gen_len, top_k = cfg.get("gen_len",20), cfg.get("top_k",50)
    for prompt in texts[:5]:
        enc, cache = tok(prompt, return_tensors="pt").to(device), None
        cache = model(**enc, use_cache=True).past_key_values
        raw  = continue_text(tok, model, enc, cache, gen_len=gen_len, top_k=top_k)
        kv16 = continue_text(tok, model, enc, cache, gen_len=gen_len, top_k=top_k, bits=bits_list[0])
        ae16 = continue_text(tok, model, enc,
                             compress_past(cache, aes, bits_list[0]),
                             gen_len=gen_len, top_k=top_k)
        examples.append({"prompt":prompt,"raw":raw,"kv_quant":kv16,"ae_compressed":ae16})
    results["examples"]=examples
    results["config"]=cfg

    os.makedirs(cfg["output_dir"],exist_ok=True)
    with open(os.path.join(cfg["output_dir"],"benchmark_results.json"),"w") as f:
        json.dump(results,f,indent=2)
    print("✅  Saved to",cfg["output_dir"]+"/benchmark_results.json")


# ───── CLI ─────────────────────────────────────────────────────────────────────
if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True)
    run_benchmark(json.load(open(p.parse_args().config)))