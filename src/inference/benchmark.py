import os
import json
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from src.models.autoencoder import Autoencoder
from transformers.cache_utils import DynamicCache


def calculate_perplexity(model, tokenizer, texts, max_length=1024, desc="Calculating perplexity"):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for text in tqdm(texts, desc=desc):
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
    return torch.exp(torch.tensor(total_loss / total_tokens)).item()


def compress_kv_cache(past_key_values, autoencoders, quantization_bits=None):
    reconstructed = []
    def quantize_tensor(x, bits):
        scale = 2 ** (bits - 1) - 1
        max_val = x.abs().max()
        if max_val == 0:
            return x
        x_norm = x / max_val
        x_q = torch.round(torch.clamp(x_norm * scale, -scale, scale)) / scale
        return x_q * max_val

    for i, (k, v) in enumerate(past_key_values):
        ae = autoencoders[i]
        B, H, S, D = k.shape
        k_flat = k.reshape(-1, D)
        v_flat = v.reshape(-1, D)
        k_latent = ae.encoder(k_flat)
        v_latent = ae.encoder(v_flat)
        if quantization_bits:
            k_latent = quantize_tensor(k_latent, quantization_bits)
            v_latent = quantize_tensor(v_latent, quantization_bits)
        k_rec = ae.decoder(k_latent).reshape(B, H, S, D)
        v_rec = ae.decoder(v_latent).reshape(B, H, S, D)
        reconstructed.append((k_rec, v_rec))
    return DynamicCache.from_legacy_cache(past_key_values=tuple(reconstructed))


def evaluate_with_compressed_cache(model, tokenizer, autoencoders, texts, max_length=1024, quantization_bits=None):
    model.eval()
    for ae in autoencoders:
        ae.eval()
    total_loss = 0.0
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
                past = compress_kv_cache(out.past_key_values, autoencoders, quantization_bits)
                logits = out.logits[..., -1, :]
                loss = F.cross_entropy(logits, input_ids[:, t+1], reduction='none')
                total_loss += loss.item()
                total_tokens += 1
    return torch.exp(torch.tensor(total_loss / total_tokens)).item()


def evaluate_with_kv_quantization(model, tokenizer, texts, max_length=1024, quantization_bits=None):
    model.eval()
    def quantize_tensor(x, bits):
        scale = 2 ** (bits - 1) - 1
        max_val = x.abs().amax()
        if max_val == 0:
            return x
        x_norm = x / max_val
        x_q = torch.round(torch.clamp(x_norm * scale, -scale, scale)) / scale
        return x_q * max_val

    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for text in tqdm(texts, desc="Eval quantized KV baseline"):
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
                new_past = []
                for k, v in out.past_key_values:
                    if quantization_bits:
                        k = quantize_tensor(k, quantization_bits)
                        v = quantize_tensor(v, quantization_bits)
                    new_past.append((k, v))
                past = DynamicCache.from_legacy_cache(past_key_values=tuple(new_past))
                logits = out.logits[..., -1, :]
                loss = F.cross_entropy(logits, input_ids[:, t+1], reduction='none')
                total_loss += loss.item()
                total_tokens += 1
    return torch.exp(torch.tensor(total_loss / total_tokens)).item()


def evaluate_longbench(model, tokenizer, autoencoders, cfg, quantization_bits=None):
    subsets = ['narrativeqa', 'hotpotqa', '2wikimqa', 'musique', 'dureader']
    results = {"baseline": {}, "compressed": {}}
    num_eval = cfg.get("num_eval_texts")
    for s in subsets:
        all_inputs = load_dataset("THUDM/LongBench", s, trust_remote_code=True)["test"]["input"]
        texts = [t for t in all_inputs if t.strip()]
        if num_eval:
            texts = texts[:num_eval]
        base_ppl = calculate_perplexity(model, tokenizer, texts, cfg["max_seq_len"], desc=f"PPL {s}")
        comp_ppl = evaluate_with_compressed_cache(model, tokenizer, autoencoders, texts, cfg["max_seq_len"], quantization_bits)
        results["baseline"][s] = base_ppl
        results["compressed"][s] = comp_ppl
    return results


def run_benchmark(model_name, autoencoder_path, latent_dim, output_dir, cfg):
    # Model + AE setup
    head_dim = cfg["hidden_size"] // cfg["num_attention_heads"]
    device = torch.device(cfg.get("device", "cuda"))
    dtype = (
        torch.bfloat16 if cfg.get("dtype") == 'bf16' else
        torch.float16 if cfg.get("dtype") in ('fp16','f16') else torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map={"": device},
        output_hidden_states=True,
        output_attentions=True,
        use_cache=True
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    chk = torch.load(autoencoder_path) if os.path.exists(autoencoder_path) else None
    autoencoders = []
    for i in range(cfg["num_hidden_layers"]):
        ae = Autoencoder(input_dim=head_dim, latent_dim=latent_dim, dtype=dtype).to(device)
        if chk:
            ae.load_state_dict(chk[f"layer_{i}"])
        ae.eval()
        autoencoders.append(ae)

    # Load texts
    ds = load_dataset("wikitext","wikitext-103-raw-v1")
    all_texts = [t for t in ds['test']['text'] if t.strip()]
    num_eval = cfg.get("num_eval_texts")
    txts = all_texts[:num_eval] if num_eval else all_texts

    # 1) Perplexity benchmarks
    quant_bits = cfg.get("quantization_bits")
    base_ppl = evaluate_with_kv_quantization(model, tokenizer, txts, cfg["max_seq_len"], quant_bits)
    comp_ppl = evaluate_with_compressed_cache(model, tokenizer, autoencoders, txts, cfg["max_seq_len"], quant_bits)
    longbench = evaluate_longbench(model, tokenizer, autoencoders, cfg, quant_bits)

    # 2) Cache-size timing benchmarks (unchanged)
    cache_sizes = cfg.get("cache_sizes", [])
    num_runs = cfg.get("num_runs", 1)
    batch_size = cfg.get("batch_size", 1)
    dtype_bytes = 4 if dtype == torch.float32 else 2

    benchmarks = {}
    for size_mb in cache_sizes:
        tokens = int((size_mb*1024**2)/(batch_size*cfg["num_attention_heads"]*head_dim*dtype_bytes*2))
        seq_len = min(tokens, cfg.get("max_seq_len", tokens))
        texts = txts[:batch_size]
        inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=seq_len)
        input_ids = inputs['input_ids'].to(device)
        mask = inputs['attention_mask'].to(device)

        orig_bytes = batch_size*cfg["num_attention_heads"]*seq_len*head_dim*dtype_bytes*2
        comp_bytes = batch_size*cfg["num_attention_heads"]*seq_len*latent_dim*dtype_bytes*2
        actual_mb = orig_bytes/(1024**2)
        compression = orig_bytes/comp_bytes if comp_bytes>0 else None

        base_times=[]
        for _ in range(num_runs):
            torch.cuda.synchronize()
            start=time.time()
            _=model(input_ids[:,:1], attention_mask=mask[:,:1], use_cache=True)
            torch.cuda.synchronize()
            base_times.append(time.time()-start)
        b_avg,b_std = float(np.mean(base_times)), float(np.std(base_times))

        comp_times=[]
        for _ in range(num_runs):
            out0 = model(input_ids[:,:1], attention_mask=mask[:,:1], use_cache=True)
            past = compress_kv_cache(out0.past_key_values, autoencoders, quant_bits)
            torch.cuda.synchronize()
            start=time.time()
            _=model(input_ids[:,1:2], attention_mask=mask[:,1:2], past_key_values=past, use_cache=True)
            torch.cuda.synchronize()
            comp_times.append(time.time()-start)
        c_avg,c_std = float(np.mean(comp_times)), float(np.std(comp_times))

        benchmarks[str(size_mb)] = {
            "actual_size_mb": actual_mb,
            "compression_ratio": compression,
            "times": {"baseline": {"avg":b_avg,"std_dev":b_std}, "compressed": {"avg":c_avg,"std_dev":c_std}},
            "speedup": b_avg/c_avg if c_avg>0 else None
        }

    results = {
        "baseline_perplexity": base_ppl,
        "compressed_perplexity": comp_ppl,
        "longbench_results": longbench,
        "benchmarks": benchmarks,
        "config": cfg
    }

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "benchmark_results.json")
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Saved benchmark results to {out_file}")
    print(f"Baseline PPL: {base_ppl:.2f}, Compressed PPL: {comp_ppl:.2f}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run KV cache benchmarks (perplexity + cache sizes)")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    cfg = json.load(open(args.config))
    run_benchmark(
        cfg["model_name"], cfg["autoencoder_path"], cfg["latent_dim"], cfg.get("output_dir"), cfg
    )
