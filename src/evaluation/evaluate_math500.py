#!/usr/bin/env python3
"""
evaluate_math500.py   –   Evaluate baseline and AE-compressed KV on MATH-500 dataset
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from datasets import load_dataset
import json
from pathlib import Path
import argparse
import re
import gc
from tqdm import tqdm
from src.models.autoencoder import Autoencoder  # Assumes Autoencoder is defined in your project

# Utility Functions

def quantize_tensor(x: torch.Tensor, bits: int):
    """Quantize a tensor to the specified number of bits."""
    if bits is None or bits <= 1:
        return x
    scale = 2 ** (bits - 1) - 1
    if scale <= 0:
        return x
    m = x.abs().amax()
    if m == 0:
        return x
    return torch.round(torch.clamp(x / m * scale, -scale, scale)) / scale * m

def resolve_dims(cfg):
    """Resolve model dimensions from config."""
    L = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layer", None) or getattr(cfg, "num_layers", None)
    H = getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd", None)
    nH = getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_head", None)
    if None in (L, H, nH):
        raise ValueError(f"Could not resolve model dimensions from config: {cfg}")
    return L, nH, H // nH

def compress_past(past, aes, bits):
    """Compress past key values using autoencoders and optional quantization."""
    rebuilt = []
    for (k, v), ae in zip(past, aes):
        B, H, S, D = k.shape
        k_lat = ae.encoder(k.reshape(-1, D))
        v_lat = ae.encoder(v.reshape(-1, D))
        if bits is not None:
            k_lat = quantize_tensor(k_lat, bits)
            v_lat = quantize_tensor(v_lat, bits)
        k_rec = ae.decoder(k_lat).reshape(B, H, S, D)
        v_rec = ae.decoder(v_lat).reshape(B, H, S, D)
        rebuilt.append((k_rec, v_rec))
    return DynamicCache.from_legacy_cache(tuple(rebuilt))

def extract_boxed_answer(text):
    """Extract the last boxed answer from a string."""
    pattern = r'\\boxed\{(.*?)\}'
    matches = re.findall(pattern, text)
    return matches[-1] if matches else None

def top_k_filter(logits, k):
    """Apply top-k filtering to logits."""
    if k <= 0 or k >= logits.size(-1):
        return logits
    v, idx = torch.topk(logits, k, dim=-1)
    out = logits.new_full(logits.shape, float("-inf"))
    out.scatter_(-1, idx, v)
    return out

def safe_sample(probs):
    """Safely sample from probabilities, handling NaNs and zero sums."""
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
    z = probs.sum(dim=-1, keepdim=True)
    probs = torch.where(z == 0, torch.full_like(probs, 1 / probs.size(-1)), probs / z)
    return torch.multinomial(probs, 1)

# Generation Function

def generate_with_compression(model, tokenizer, prompt, compressor=None, max_length=512, top_k=50, temperature=1.0):
    """Generate text for a single prompt with optional KV cache compression, passing only the latest token after the first step."""
    device = model.device
    inputs = tokenizer([prompt], return_tensors="pt").to(device)
    generated_ids = inputs["input_ids"]
    past_key_values = None
    output_text = ""
    for step in range(max_length):
        # Use full sequence on first step, last token on subsequent steps
        if past_key_values is not None:
            input_ids = generated_ids[:, -1:]  # Last token only
        else:
            input_ids = generated_ids  # Full prompt initially
        with torch.no_grad():
            out = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
        logits = out.logits[:, -1, :] / temperature
        logits = top_k_filter(logits, top_k)
        probs = torch.softmax(logits, dim=-1)
        next_token = safe_sample(probs)
        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        if compressor:
            past_key_values = compressor(out.past_key_values)
        else:
            past_key_values = out.past_key_values
        # Decode and check for boxed answer
        output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        if extract_boxed_answer(output_text):
            break
    return output_text

# Evaluation Function

def evaluate_model(model, tokenizer, problems, solutions, compressor=None, max_length=512, top_k=50, temperature=1.0):
    """Evaluate the model on MATH-500 problems and compute exact-match accuracy with progress bar."""
    outputs = []
    for prompt in tqdm(problems, desc="Evaluating prompts", unit="prompt"):
        prompt_text = f"Problem: {prompt}\nSolution:"
        output = generate_with_compression(model, tokenizer, prompt_text, compressor, max_length, top_k, temperature)
        outputs.append(output)
        # Clear memory
        torch.cuda.empty_cache()
        gc.collect()
    pred_answers = [extract_boxed_answer(output) for output in outputs]
    true_answers = [extract_boxed_answer(sol) for sol in solutions]
    correct = [p == t for p, t in zip(pred_answers, true_answers) if t is not None]
    accuracy = sum(correct) / len(correct) if correct else 0.0
    return accuracy

# Main Function

def main():
    parser = argparse.ArgumentParser(description="Evaluate a model on MATH-500 dataset.")
    parser.add_argument("--config", required=True, help="Path to the JSON configuration file.")
    args = parser.parse_args()
    cfg = json.load(open(args.config))

    # Configuration parameters
    model_name = cfg["model_name"]
    ae_ckpt = cfg["autoencoder_path"]
    latent_dim = cfg["latent_dim"]
    bits_list = cfg["quantization_bits"]
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    device = cfg.get("device", "cuda")
    max_length = cfg.get("max_length", 512)
    top_k = cfg.get("top_k", 50)
    temperature = cfg.get("temperature", 1.0)

    # Load the MATH-500 dataset
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    problems = dataset["problem"][:50]
    solutions = dataset["solution"][:50]

    # Load tokenizer and model with mixed precision
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map={"": device}, torch_dtype=torch.float16, use_cache=True
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # Load autoencoders
    ckpt = torch.load(ae_ckpt, map_location=device)
    L, _, dH = resolve_dims(model.config)
    aes = []
    for i in range(L):
        ae = Autoencoder(dH, latent_dim, dtype=torch.float16).to(device)
        ae.load_state_dict(ckpt[f"layer_{i}"])
        ae.eval()
        aes.append(ae)

    # Baseline evaluation
    print(f"\n▶ Baseline: {model_name}")
    base_acc = evaluate_model(model, tokenizer, problems, solutions, max_length=max_length, top_k=top_k, temperature=temperature)
    print(f"   Exact-match accuracy: {base_acc:.3f}")
    json.dump({"accuracy": base_acc}, open(out_dir / "math500_baseline.json", "w"), indent=2)

    # Compressed evaluations
    for bits in bits_list:
        tag = f"lat{latent_dim}_bits{bits}"
        print(f"\n▶ Compressed: latent={latent_dim} bits={bits}")
        def compressor(past):
            return compress_past(past, aes, bits)
        acc = evaluate_model(model, tokenizer, problems, solutions, compressor=compressor, max_length=max_length, top_k=top_k, temperature=temperature)
        ratio = (64 / latent_dim) * (16 / bits)
        print(f"   Accuracy: {acc:.3f} (compression ×{ratio:.1f})")
        json.dump({"accuracy": acc}, open(out_dir / f"math500_{tag}.json", "w"), indent=2)

if __name__ == "__main__":
    main()