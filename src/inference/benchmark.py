#!/usr/bin/env python3
"""
benchmark.py

Measures:
  • raw‑perplexity baseline (full attention)
  • KV‑cache baseline with optional quantisation
  • AE‑compressed + quantised KV perplexity
  • Example generations (raw, KV‑quant, AE+quant)

Works with GPT‑2‑style and Qwen2‑family models.
"""

# ─── Silence noisy logs *before* imports ────────────────────────────────────────
import os, warnings, logging, json, argparse
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
import torch, torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from datasets import load_dataset
from src.models.autoencoder import Autoencoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─────────────────────────── Utility helpers ───────────────────────────────────

def _max_positions(model):
    """Return the model's max position IDs (GPT‑2, OPT, Qwen all differ)."""
    if hasattr(model.config, "n_positions"):
        return model.config.n_positions
    if hasattr(model.config, "max_position_embeddings"):
        return model.config.max_position_embeddings
    return 1024  # sensible default


def quantize_tensor(x: torch.Tensor, bits: int):
    """Uniform symmetric quantisation to the given bit‑width."""
    scale = 2 ** (bits - 1) - 1
    m = x.abs().amax()
    if m == 0:
        return x
    x_q = torch.round(torch.clamp(x / m * scale, -scale, scale)) / scale
    return x_q * m


def quantize_past_key_values(past, bits: int):
    """Quantise each (key,value) pair in a past‑KV tuple."""
    if bits is None:
        return past
    rebuilt = []
    for k, v in past:
        rebuilt.append((quantize_tensor(k, bits), quantize_tensor(v, bits)))
    return DynamicCache.from_legacy_cache(tuple(rebuilt))


def resolve_dims(cfg):
    """Infer (n_layers, n_heads, head_dim) from a model config."""
    n = getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", None))
    h = getattr(cfg, "hidden_size", getattr(cfg, "n_embd",  None))
    heads = getattr(cfg, "num_attention_heads", getattr(cfg, "n_head", None))
    if None in (n, h, heads):
        raise ValueError("Could not resolve dimensions from config")
    return n, heads, h // heads


def compress_kv_cache(past, aes, bits=None):
    """Apply layer‑wise AE compression (and optional quantisation) to past‑KV."""
    rebuilt = []
    for (k, v), ae in zip(past, aes):
        B, H, S, D = k.shape
        k_flat, v_flat = k.view(-1, D), v.view(-1, D)
        with torch.no_grad():
            k_lat, v_lat = ae.encoder(k_flat), ae.encoder(v_flat)
            if bits is not None:
                k_lat, v_lat = quantize_tensor(k_lat, bits), quantize_tensor(v_lat, bits)
            k_rec = ae.decoder(k_lat).view(B, H, S, D)
            v_rec = ae.decoder(v_lat).view(B, H, S, D)
        rebuilt.append((k_rec, v_rec))
    return DynamicCache.from_legacy_cache(tuple(rebuilt))


def prompt_cache(tokenizer, model, prompt: str):
    """Encode prompt (truncated to model max length) and return cache."""
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=_max_positions(model),
    ).to(model.device)
    with torch.no_grad():
        out = model(**enc, use_cache=True)
    return enc, out.past_key_values


def top_k_logits(logits: torch.Tensor, k: int = 50) -> torch.Tensor:
    """Top‑k filtering (–inf elsewhere)."""
    if k <= 0 or k >= logits.size(-1):
        return logits
    v, idx = torch.topk(logits, k, dim=-1)
    mask = logits.new_full(logits.shape, float("-inf"))
    mask.scatter_(-1, idx, v)
    return mask


def safe_sample(probs: torch.Tensor):
    """Sample 1 token, guarding against NaN/Inf or negative probs."""
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    probs = torch.clamp(probs, min=0.0)
    probs_sum = probs.sum(dim=-1, keepdim=True)
    probs = torch.where(probs_sum == 0, torch.full_like(probs, 1.0 / probs.size(-1)), probs / probs_sum)
    return torch.multinomial(probs, num_samples=1)


def continue_text(tokenizer, model, enc, past, *, gen_len=20, top_k=50, temp=1.0, bits=None):
    """Incremental generation with optional KV quantisation."""
    ids = enc.input_ids.clone()
    last = ids[:, -1:].to(model.device)

    for _ in range(gen_len):
        with torch.no_grad():
            out = model(input_ids=last, past_key_values=past, use_cache=True)

        logits, past = out.logits[:, -1, :], out.past_key_values
        past = quantize_past_key_values(past, bits)

        filt = top_k_logits(logits / temp, top_k)
        probs = torch.softmax(filt, dim=-1)
        nxt = safe_sample(probs)

        ids, last = torch.cat([ids, nxt], dim=-1), nxt

    return tokenizer.decode(ids[0], skip_special_tokens=True)


# ─────────────────── Perplexity evaluation routines ────────────────────────────

def calculate_perplexity(model, tokenizer, texts, max_length=1024, desc="Raw baseline"):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    with torch.no_grad():
        for text in tqdm(texts, desc=desc):
            inp = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            ).to(model.device)
            out = model(**inp)
            logits = out.logits[..., :-1, :].contiguous()
            labels = inp.input_ids[..., 1:].contiguous()
            mask = inp.attention_mask[..., 1:].contiguous()

            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += (loss * mask.view(-1)).sum().item()
            total_tokens += int(mask.sum())

    logger.info(f"[{desc}] valid tokens = {total_tokens}")
    return float("nan") if total_tokens == 0 else torch.exp(torch.tensor(total_loss / total_tokens)).item()


def evaluate_with_kv_quant(model, tokenizer, texts, *, max_length=1024, bits=None):
    """Incremental decoding + quantisation‑only baseline."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    with torch.no_grad():
        for text in tqdm(texts, desc="KV‑cache baseline"):
            inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
            ids, mask = inp.input_ids, inp.attention_mask
            past = None

            for t in range(ids.size(1) - 1):
                if not mask[0, t + 1]:
                    continue
                out = model(input_ids=ids[:, t:t + 1], past_key_values=past, use_cache=True)
                loss_val = loss_fct(out.logits[:, -1, :], ids[:, t + 1])
                total_loss += loss_val.item()
                total_tokens += 1
                past = quantize_past_key_values(out.past_key_values, bits)

    logger.info(f"[KV‑cache baseline] valid tokens = {total_tokens}")
    return float("nan") if total_tokens == 0 else torch.exp(torch.tensor(total_loss / total_tokens)).item()


def evaluate_with_compressed_cache(model, tokenizer, aes, texts, *, max_length=1024, bits=None):
    """Incremental decoding + AE compression (+ optional quantisation)."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    with torch.no_grad():
        for text in tqdm(texts, desc="AE‑compressed KV"):
            inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
            ids, mask = inp.input_ids, inp.attention_mask
            past = None

            for t in range(ids.size(1) - 1):
                if not mask[0, t + 1]:
                    continue
                out = model(input_ids=ids[:, t:t + 1], past_key_values=past, use_cache=True)
                past = compress_kv_cache(out.past_key_values, aes, bits)
                loss_val = loss_fct(out.logits[:, -1, :], ids[:, t + 1])
                total_loss += loss_val.item()
                total_tokens += 1

    logger.info(f"[AE‑compressed KV] valid tokens = {total_tokens}")
    return float("nan") if total_tokens == 0 else torch.exp(torch.tensor(total_loss / total_tokens)).item()


def evaluate_longbench(model, tokenizer, aes, cfg, bits=None):
    subsets = ["narrativeqa", "hotpotqa", "2wikimqa", "musique", "dureader"]
    out = {"baseline": {}, "compressed": {}}
    n_eval = cfg.get("num_eval_texts")

    for s in subsets:
        data = load_dataset("THUDM/LongBench", s, trust_remote_code=True)["test"]["input"]
        texts = [x for x in data if x.strip()][: n_eval or None]
        out["baseline"][s] = calculate_perplexity(model, tokenizer, texts, cfg["max_seq_len"], f"PPL {s}")
        out["compressed"][s] = evaluate_with_compressed_cache(
            model, tokenizer, aes, texts, max_length=cfg["max_seq_len"], bits=bits
        )
    return out


# ─────────────────────────── Main benchmark driver ─────────────────────────────

def run_benchmark(cfg: dict):
    device = torch.device(cfg.get("device", "cuda"))
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[cfg.get("dtype", "fp32")]

    # --- model & tokenizer ------------------------------------------------------
    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        tok.pad_token_id = tok.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"], torch_dtype=dtype, device_map={"": device}, use_cache=True
    )
    model.config.pad_token_id = tok.pad_token_id

    # --- autoencoders ----------------------------------------------------------
    n_layers, _, head_dim = resolve_dims(model.config)
    ckpt = torch.load(cfg["autoencoder_path"], map_location=device) if os.path.exists(cfg["autoencoder_path"]) else None
    aes = []
    for i in range(n_layers):
        ae = Autoencoder(input_dim=head_dim, latent_dim=cfg["latent_dim"], dtype=dtype).to(device)
        if ckpt:
            ae.load_state_dict(ckpt[f"layer_{i}"])
        ae.eval()
        aes.append(ae)

    # --- evaluation texts ------------------------------------------------------
    ds = load_dataset("wikitext", "wikitext-103-raw-v1")["test"]["text"]
    texts = [x for x in ds if x.strip()]
    if cfg.get("num_eval_texts"):
        texts = texts[: cfg["num_eval_texts"]]

    # 1) raw PPL
    raw_ppl = calculate_perplexity(model, tok, texts, cfg["max_seq_len"], "Raw baseline")

    # 2) KV‑cache vs. AE‑compressed
    bits_list = cfg.get("quantization_bits", [None])
    ppl_summary = {}
    for bits in bits_list:
        kv_ppl = evaluate_with_kv_quant(model, tok, texts, max_length=cfg["max_seq_len"], bits=bits)
        ae_ppl = evaluate_with_compressed_cache(model, tok, aes, texts, max_length=cfg["max_seq_len"], bits=bits)
        ppl_summary[str(bits)] = {"kv_cache_baseline_ppl": kv_ppl, "ae_compressed_ppl": ae_ppl}

    # 3) LongBench (optional)
    longbench = evaluate_longbench(model, tok, aes, cfg, bits=bits_list[0])

    # 4) example generations -----------------------------------------------------
    gen_len, top_k = cfg.get("gen_len", 20), cfg.get("top_k", 50)
    examples = []
    for prompt in texts[:5]:
        enc, cache = prompt_cache(tok, model, prompt)

        raw_out = continue_text(tok, model, enc, cache, gen_len=gen_len, top_k=top_k, bits=None)
        kv_out = continue_text(tok, model, enc, cache, gen_len=gen_len, top_k=top_k, bits=bits_list[0])
        ae_cache = compress_kv_cache(cache, aes, bits_list[0])
        ae_out = continue_text(tok, model, enc, ae_cache, gen_len=gen_len, top_k=top_k, bits=None)

        examples.append({"prompt": prompt, "raw_baseline": raw_out, "kv_baseline": kv_out, "ae_compressed": ae_out})

    # --- write results ---------------------------------------------------------
    os.makedirs(cfg["output_dir"], exist_ok=True)
    with open(os.path.join(cfg["output_dir"], "benchmark_results.json"), "w") as f:
        json.dump(
            {
                "raw_baseline_ppl": raw_ppl,
                "perplexities": ppl_summary,
                "longbench": longbench,
                "examples": examples,
                "config": cfg,
            },
            f,
            indent=2,
        )

    print(f"✅ Saved results to {cfg['output_dir']}/benchmark_results.json")


# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KV‑cache benchmark")
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    args = parser.parse_args()
    run_benchmark(json.load(open(args.config)))