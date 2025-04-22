"""
Module Name: benchmark.py
Description: Benchmark with layer-wise autoencoders
"""

import os
import json
import argparse
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from src.models.autoencoder import Autoencoder
from transformers.cache_utils import DynamicCache

def calculate_perplexity(model, tokenizer, texts, max_length=1024):
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for text in tqdm(texts, desc="Calculating perplexity"):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs["attention_mask"].to(model.device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            mask = attention_mask[..., 1:].contiguous().view(-1)
            loss = loss * mask
            total_loss += loss.sum().item()
            total_tokens += mask.sum().item()
    perplexity = torch.exp(torch.tensor(total_loss / total_tokens)).item()
    return perplexity

def compress_kv_cache(past_key_values, autoencoders):
    if past_key_values is None:
        return None
    reconstructed = []
    for i, (keys, values) in enumerate(past_key_values):
        ae = autoencoders[i]
        B, H, S, D = keys.shape
        k_flat = keys.reshape(-1, D)
        v_flat = values.reshape(-1, D)
        k_rec_flat, _ = ae(k_flat)
        v_rec_flat, _ = ae(v_flat)
        k_rec = k_rec_flat.reshape(B, H, S, D)
        v_rec = v_rec_flat.reshape(B, H, S, D)
        reconstructed.append((k_rec, v_rec))
    return DynamicCache.from_legacy_cache(past_key_values=tuple(reconstructed))

def evaluate_with_compressed_cache(model, tokenizer, autoencoders, texts, max_length=1024):
    model.eval()
    for ae in autoencoders:
        ae.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for text in tqdm(texts, desc="Eval compressed"):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = inputs["input_ids"].to(model.device)
            mask = inputs["attention_mask"].to(model.device)
            past = None
            for t in range(input_ids.size(1) - 1):
                if not mask[0, t+1]:
                    continue
                out = model(
                    input_ids[:, t:t+1],
                    attention_mask=mask[:, t:t+1],
                    past_key_values=past,
                    use_cache=True
                )
                past = compress_kv_cache(out.past_key_values, autoencoders)
                logits = out.logits[..., -1, :]
                loss = F.cross_entropy(logits, input_ids[:, t+1], reduction='none')
                total_loss += loss.item()
                total_tokens += 1
    perplexity = torch.exp(torch.tensor(total_loss / total_tokens)).item()
    return perplexity

def evaluate_longbench(model, tokenizer, autoencoders, cfg):
    subsets = ['narrativeqa', 'hotpotqa', '2wikimqa', 'musique', 'dureader']
    results = {"baseline": {}, "compressed": {}}
    # Number of evaluation texts for LongBench from config
    num_eval = cfg.get("num_eval_texts")
    for s in subsets:
        all_inputs = load_dataset("THUDM/LongBench", s)["test"]["input"]
        # Filter out empty strings and limit to num_eval texts
        texts = [t for t in all_inputs if t.strip()]
        if num_eval is not None:
            texts = texts[:num_eval]
        # Calculate baseline and compressed perplexities on this subset
        base_ppl = calculate_perplexity(model, tokenizer, texts, cfg["max_seq_len"])
        comp_ppl = evaluate_with_compressed_cache(model, tokenizer, autoencoders, texts, cfg["max_seq_len"])
        results["baseline"][s] = base_ppl
        results["compressed"][s] = comp_ppl
    return results

def run_benchmark(model_name, autoencoder_path, latent_dim, output_dir, cfg):
    head_dim = cfg["hidden_size"] // cfg["num_attention_heads"]
    device = torch.device(cfg.get("device", "cuda"))
    dtype = (
        torch.bfloat16 if cfg.get("dtype") == "bf16"
        else torch.float16 if cfg.get("dtype") in ("fp16", "f16")
        else torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map={"": device},
        output_hidden_states=True,
        output_attentions=True,
        use_cache=True
    )
    # Check if the autoencoder file exists before attempting to load
    if not os.path.exists(autoencoder_path):
        raise FileNotFoundError(
            f"Autoencoder file not found at the specified path: {autoencoder_path}. "
            "Ensure the training step for this configuration completed successfully and saved the model."
        )
    chk = torch.load(autoencoder_path)
    autoencoders = []
    for i in range(cfg["num_hidden_layers"]):
        ae = Autoencoder(input_dim=head_dim, latent_dim=latent_dim, dtype=dtype).to(device)
        ae.load_state_dict(chk[f"layer_{i}"])
        autoencoders.append(ae)
    ds = load_dataset("wikitext", "wikitext-103-raw-v1")
    # Limit WikiText evaluation to num_eval_texts from config
    num_eval = cfg.get("num_eval_texts")
    all_texts = [t for t in ds["test"]["text"] if t.strip()]
    txts = all_texts[:num_eval] if num_eval is not None else all_texts
    base_ppl = calculate_perplexity(model, tokenizer, txts, cfg["max_seq_len"])
    comp_ppl = evaluate_with_compressed_cache(model, tokenizer, autoencoders, txts, cfg["max_seq_len"])
    longbench = evaluate_longbench(model, tokenizer, autoencoders, cfg)
    results = {
        "baseline_perplexity": base_ppl,
        "compressed_perplexity": comp_ppl,
        "longbench_results": longbench,
        "config": cfg
    }
    # Save results to JSON file in the output directory
    os.makedirs(output_dir, exist_ok=True)
    result_file = os.path.join(output_dir, "benchmark_results.json")
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)
    # Print out where results are saved and a summary
    print(f"Saved benchmark results to {result_file}")
    print(f"Baseline perplexity: {base_ppl:.2f}")
    print(f"Compressed perplexity: {comp_ppl:.2f}")
    print("LongBench results:")
    for subset in longbench["baseline"]:
        bp = longbench["baseline"][subset]
        cp = longbench["compressed"][subset]
        print(f"  {subset}: baseline={bp:.2f}, compressed={cp:.2f}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run layer-wise KV cache benchmark")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    cfg = json.load(open(args.config))
    run_benchmark(
        cfg["model_name"],
        cfg["autoencoder_path"],
        cfg["latent_dim"],
        cfg.get("output_dir"),
        cfg
    )