#!/usr/bin/env python3
"""
benchmark.py

Run:
  • raw-perplexity baseline
  • quantized-KV baseline vs. AE-compressed KV PPL
  • a few example top-k generations
for both GPT-2 style and Qwen2 family models.
"""

# ─── Suppress noisy logs *before* imports ────────────────────────────────────────
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
import json, argparse
import torch, torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from datasets import load_dataset
from src.models.autoencoder import Autoencoder

# ─── Logger ─────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Helpers ────────────────────────────────────────────────────────────────────

def quantize_tensor(x: torch.Tensor, bits: int):
    scale = 2 ** (bits - 1) - 1
    m = x.abs().amax()
    if m == 0:
        return x
    x_q = torch.round(torch.clamp(x / m * scale, -scale, scale)) / scale
    return x_q * m

def resolve_dims(cfg):
    # number of layers
    if hasattr(cfg, "num_hidden_layers"):
        n = cfg.num_hidden_layers
    elif hasattr(cfg, "n_layer"):
        n = cfg.n_layer
    else:
        raise ValueError("Cannot find layer count in config")
    # hidden size
    if hasattr(cfg, "hidden_size"):
        h = cfg.hidden_size
    elif hasattr(cfg, "n_embd"):
        h = cfg.n_embd
    else:
        raise ValueError("Cannot find hidden_size in config")
    # heads
    if hasattr(cfg, "num_attention_heads"):
        heads = cfg.num_attention_heads
    elif hasattr(cfg, "n_head"):
        heads = cfg.n_head
    else:
        raise ValueError("Cannot find num_attention_heads in config")
    return n, heads, h // heads

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
            k_rec = ae.decoder(k_lat).view(B, H, S, D)
            v_rec = ae.decoder(v_lat).view(B, H, S, D)
        rebuilt.append((k_rec, v_rec))
    return DynamicCache.from_legacy_cache(tuple(rebuilt))

def prompt_cache(tokenizer, model, prompt: str):
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**enc, use_cache=True)
    return enc, out.past_key_values

def top_k_logits(logits: torch.Tensor, k: int = 50) -> torch.Tensor:
    if k <= 0 or k >= logits.size(-1):
        return logits
    v, idx = torch.topk(logits, k, dim=-1)
    out = logits.new_full(logits.shape, float("-inf"))
    out.scatter_(-1, idx, v)
    return out

def continue_text(tokenizer, model, enc, past, gen_len=20, top_k=50, temp=1.0):
    ids  = enc.input_ids.clone()
    pos  = ids.size(1)
    last = ids[:, -1:].to(model.device)

    for _ in range(gen_len):
        mask    = torch.ones_like(last)
        pos_ids = torch.tensor([[pos]], device=model.device)
        with torch.no_grad():
            out = model(
                input_ids=last,
                attention_mask=mask,
                position_ids=pos_ids,
                past_key_values=past,
                use_cache=True,
            )
        logits, past = out.logits[:, -1, :], out.past_key_values
        filt   = top_k_logits(logits / temp, top_k)
        probs  = torch.softmax(filt, dim=-1)
        nxt    = torch.multinomial(probs, num_samples=1)
        ids    = torch.cat([ids, nxt], dim=-1)
        last   = nxt
        pos   += 1

    return tokenizer.decode(ids[0], skip_special_tokens=True)

# ─── Perplexity routines ────────────────────────────────────────────────────────

def calculate_perplexity(model, tokenizer, texts, max_length=1024, desc="Raw baseline"):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    with torch.no_grad():
        for text in tqdm(texts, desc=desc):
            inp = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=max_length).to(model.device)
            out = model(**inp)
            logits = out.logits[..., :-1, :].contiguous()
            labels = inp.input_ids[..., 1:].contiguous()
            mask   = inp.attention_mask[..., 1:].contiguous()

            loss = loss_fct(logits.view(-1, logits.size(-1)),
                            labels.view(-1))
            valid = mask.view(-1).sum().item()
            total_loss  += (loss * mask.view(-1)).sum().item()
            total_tokens += valid

    logger.info(f"[{desc}] valid tokens = {total_tokens}")
    if total_tokens == 0:
        logger.warning(f"[{desc}] no valid tokens, returning NaN")
        return float("nan")
    return torch.exp(torch.tensor(total_loss / total_tokens)).item()

def evaluate_with_kv_quant(model, tokenizer, texts, max_length=1024, bits=None):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    with torch.no_grad():
        for text in tqdm(texts, desc="Quantized-KV baseline"):
            inp = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=max_length).to(model.device)
            ids, mask = inp.input_ids, inp.attention_mask
            past = None
            for t in range(ids.size(1)-1):
                if not mask[0, t+1]:
                    continue
                out = model(
                    input_ids=ids[:, t:t+1],
                    attention_mask=mask[:, t:t+1],
                    past_key_values=past,
                    use_cache=True,
                )
                # quantize nothing here—this is baseline
                loss_val = loss_fct(out.logits[:, -1, :], ids[:, t+1])
                total_loss  += loss_val.item()
                total_tokens += 1

    logger.info(f"[Quantized-KV baseline] valid tokens = {total_tokens}")
    if total_tokens == 0:
        logger.warning("[Quantized-KV baseline] no valid tokens, returning NaN")
        return float("nan")
    return torch.exp(torch.tensor(total_loss / total_tokens)).item()

def evaluate_with_compressed_cache(model, tokenizer, aes, texts, max_length=1024, bits=None):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    with torch.no_grad():
        for text in tqdm(texts, desc="AE-compressed KV"):
            inp = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=max_length).to(model.device)
            ids, mask = inp.input_ids, inp.attention_mask
            past = None
            for t in range(ids.size(1)-1):
                if not mask[0, t+1]:
                    continue
                out = model(
                    input_ids=ids[:, t:t+1],
                    attention_mask=mask[:, t:t+1],
                    past_key_values=past,
                    use_cache=True,
                )
                past      = compress_kv_cache(out.past_key_values, aes, bits)
                loss_val  = loss_fct(out.logits[:, -1, :], ids[:, t+1])
                total_loss  += loss_val.item()
                total_tokens += 1

    logger.info(f"[AE-compressed KV] valid tokens = {total_tokens}")
    if total_tokens == 0:
        logger.warning("[AE-compressed KV] no valid tokens, returning NaN")
        return float("nan")
    return torch.exp(torch.tensor(total_loss / total_tokens)).item()

def evaluate_longbench(model, tokenizer, aes, cfg, bits=None):
    subsets = ['narrativeqa','hotpotqa','2wikimqa','musique','dureader']
    out = {"baseline":{}, "compressed":{}}
    n_eval = cfg.get("num_eval_texts")
    for s in subsets:
        data  = load_dataset("THUDM/LongBench", s, trust_remote_code=True)["test"]["input"]
        texts = [x for x in data if x.strip()][:n_eval or None]
        out["baseline"][s]   = calculate_perplexity(model, tokenizer, texts,
                                                   cfg["max_seq_len"], f"PPL {s}")
        out["compressed"][s] = evaluate_with_compressed_cache(
                                    model, tokenizer, aes, texts,
                                    cfg["max_seq_len"], bits)
    return out

# ─── Main runner ───────────────────────────────────────────────────────────────
def run_benchmark(model_name, ae_path, lat_dim, out_dir, cfg):
    device = torch.device(cfg.get("device","cuda"))
    dtype  = {"fp32":torch.float32,"fp16":torch.float16,"bf16":torch.bfloat16}[cfg.get("dtype","fp32")]

    # load model & tokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token; tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map={"":device}, use_cache=True
    )
    model.config.pad_token_id = tok.pad_token_id

    # load autoencoders
    n_layers, _, head_dim = resolve_dims(model.config)
    ckpt = torch.load(ae_path, map_location=device) if os.path.exists(ae_path) else None
    aes = []
    for i in range(n_layers):
        ae = Autoencoder(input_dim=head_dim, latent_dim=lat_dim, dtype=dtype).to(device)
        if ckpt: ae.load_state_dict(ckpt[f"layer_{i}"])
        ae.eval(); aes.append(ae)

    # get test texts
    ds = load_dataset("wikitext","wikitext-103-raw-v1")["test"]["text"]
    texts = [x for x in ds if x.strip()]
    if cfg.get("num_eval_texts"):
        texts = texts[:cfg["num_eval_texts"]]

    # 1) raw baseline PPL
    raw_ppl = calculate_perplexity(model, tok, texts, cfg["max_seq_len"], "Raw baseline")

    # 2) quantized-KV vs. AE-compressed
    bits_list = cfg.get("quantization_bits",[None])
    ppl_summary = {}
    for bits in bits_list:
        q_ppl = evaluate_with_kv_quant(model, tok, texts, cfg["max_seq_len"], bits)
        c_ppl = evaluate_with_compressed_cache(model, tok, aes, texts, cfg["max_seq_len"], bits)
        ppl_summary[str(bits)] = {
            "quantized_baseline_ppl": q_ppl,
            "ae_compressed_ppl":     c_ppl
        }

    # 3) LongBench
    longbench = evaluate_longbench(model, tok, aes, cfg, bits_list[0])

    # 4) examples
    gen_len = cfg.get("gen_len",20)
    examples = []
    for prompt in texts[:5]:
        enc, cache = prompt_cache(tok, model, prompt)
        base_out_q = continue_text(tok, model, enc, cache, gen_len, top_k=cfg.get("top_k",50))
        comp_cache = compress_kv_cache(cache, aes, bits_list[0])
        comp_out   = continue_text(tok, model, enc, comp_cache, gen_len, top_k=cfg.get("top_k",50))
        examples.append({
            "prompt":    prompt,
            "raw_baseline":  None,  # too expensive to generate full with attention each step
            "quant_baseline": base_out_q,
            "ae_compressed":  comp_out
        })

    # write results
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/benchmark_results.json","w") as f:
        json.dump({
            "raw_baseline_ppl": raw_ppl,
            "perplexities":    ppl_summary,
            "longbench":       longbench,
            "examples":        examples,
            "config":          cfg
        }, f, indent=2)

    print(f"✅ Saved results to {out_dir}/benchmark_results.json")

if __name__=="__main__":
    p = argparse.ArgumentParser(description="KV-cache benchmark")
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = json.load(open(args.config))
    run_benchmark(
      cfg["model_name"],
      cfg["autoencoder_path"],
      cfg["latent_dim"],
      cfg["output_dir"],
      cfg
    )