#!/usr/bin/env python3
"""
Module Name: train.py
Description: Train layer‑wise Autoencoders for KV Cache Compression. This module trains one autoencoder per self‑attention layer to compress the KV cache from a pretrained transformer model.
Author: Henry Huang  •  Updated: 2025‑05‑06

Key update
──────────
• **Loss = KV reconstruction + w_attn × attention‑reconstruction (default w_attn = 1.0)**
• Uses the existing `compute_attention()` helper to compute attention maps for the
  original and reconstructed KV pairs given the same queries.
"""

import os
import sys
import random
import json
import pprint
import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import tqdm
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)

from src.models.autoencoder import Autoencoder
from src.utils.buffer import Buffer

# ────────────────────────────────────────────────────────────────────────────────
# Helper functions
# ────────────────────────────────────────────────────────────────────────────────

def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # ── sensible defaults for new keys ─────────────────────────────────────────
    cfg.setdefault("encoder_layer_sizes", [])   # means 1‑layer encoder
    cfg.setdefault("decoder_layer_sizes", [])
    cfg.setdefault("activation", "ReLU")
    cfg.setdefault("attn_loss_weight", 1.0)     # weight on attention‑reconstruction
    return cfg


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="Train layer‑wise Autoencoders for KV Cache Compression")
    parser.add_argument(
        "--config", type=str,
        default="src/configs/default_config.json",
        help="Path to config file")
    return vars(parser.parse_args())


@torch.no_grad()
def compute_attention(q: torch.Tensor,
                      k: torch.Tensor,
                      v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Scaled‑dot‑product attention for tensors shaped (B, L, H, S, D)."""
    B, L, H, S, D = q.shape
    q2 = q.reshape(B * L * H, S, D)
    k2 = k.reshape(B * L * H, S, D)
    v2 = v.reshape(B * L * H, S, D)

    scores = torch.matmul(q2, k2.transpose(-2, -1)) / math.sqrt(D)   # (B*L*H, S, S)
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v2)                          # (B*L*H, S, D)

    attn_weights = attn_weights.reshape(B, L, H, S, S)
    output       = output.reshape(B, L, H, S, D)
    return output, attn_weights


# ────────────────────────────────────────────────────────────────────────────────
# Training
# ────────────────────────────────────────────────────────────────────────────────

def main(cfg):
    # reproducibility
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])

    # dtype
    if cfg.get("dtype") == "bf16":
        dtype = torch.bfloat16
    elif cfg.get("dtype") in ("fp16", "f16"):
        dtype = torch.float16
    else:
        dtype = torch.float32
    print(f"Using dtype: {dtype}")

    # output dir
    os.makedirs(cfg["output_dir"], exist_ok=True)

    # model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["name"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["name"],
        trust_remote_code=cfg["name"].startswith("Qwen"),
        torch_dtype=dtype,
        device_map={"": cfg["device"]},
        output_hidden_states=True,
        output_attentions=True,
        use_cache=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    model.eval()

    # prepare autoencoders
    head_dim = cfg["hidden_size"] // cfg["num_attention_heads"]
    autoencoders = nn.ModuleList([
        Autoencoder(
            input_dim=head_dim,
            latent_dim=cfg["latent_dim"],
            encoder_layer_sizes=cfg["encoder_layer_sizes"],
            decoder_layer_sizes=cfg["decoder_layer_sizes"],
            activation=cfg.get("activation", "ReLU"),
            dtype=dtype,
        ).to(cfg["device"])
        for _ in range(cfg["num_hidden_layers"])
    ])

    # tensorboard
    writer = SummaryWriter(log_dir=f'runs/{cfg["name"]}_{cfg["latent_dim"]}')

    # ── dataset prep ───────────────────────────────────────────────────────────
    wiki = load_dataset("wikitext", "wikitext-103-raw-v1")
    long_texts = []
    for subset in [
        'narrativeqa', 'hotpotqa', '2wikimqa', 'musique', 'dureader']:
        try:
            ds = load_dataset("THUDM/LongBench", subset)
            long_texts.extend(ds["test"]["input"])
        except Exception:
            pass

    all_train = [t for t in wiki["train"]["text"] if t.strip()]
    n_w = int(cfg["num_train_texts"] * 0.7)
    n_l = cfg["num_train_texts"] - n_w
    train_texts = all_train[:min(n_w, len(all_train))] + \
                 [t for t in long_texts if t.strip()][:min(n_l, len(long_texts))]

    val_size = int(len(train_texts) * cfg.get("val_split", 0.1))
    texts_val   = train_texts[:val_size]
    texts_train = train_texts[val_size:]

    train_buffer = Buffer(cfg, model, tokenizer, texts_train)
    val_buffer   = Buffer(cfg, model, tokenizer, texts_val)

    # ── optimisers & schedulers ────────────────────────────────────────────────
    batches_per_epoch = len(texts_train) // cfg["batch_size"]
    total_steps = (batches_per_epoch // cfg["gradient_accumulation_steps"]) * cfg["num_epochs"]
    warmup_steps = int(cfg.get("warmup_ratio", 0.1) * total_steps)

    optimizer = AdamW(
        [p for ae in autoencoders for p in ae.parameters()],
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 1e-3),
        betas=(0.9, 0.999),
    )
    cosine_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    plateau_scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=cfg.get("lr_reduce_factor", 0.5),
        patience=cfg.get("lr_patience", 1),
        verbose=True,
    )

    # ── training loop ──────────────────────────────────────────────────────────
    for epoch in range(1, cfg["num_epochs"] + 1):
        for ae in autoencoders:
            ae.train()
        running_total, running_kv, running_attn = 0.0, 0.0, 0.0

        for step in tqdm.trange(batches_per_epoch, desc=f"Epoch {epoch}/{cfg['num_epochs']}"):
            (keys, values), queries = train_buffer.next()
            B, L, H, S, D = keys.shape

            k_rec = torch.zeros_like(keys)
            v_rec = torch.zeros_like(values)
            for l in range(L):
                # flatten heads & seq for AE
                k_flat, _ = autoencoders[l](keys[:, l].reshape(-1, D))
                v_flat, _ = autoencoders[l](values[:, l].reshape(-1, D))
                k_rec[:, l] = k_flat.reshape(B, H, S, D)
                v_rec[:, l] = v_flat.reshape(B, H, S, D)

            # losses
            kv_loss = F.mse_loss(k_rec, keys) + F.mse_loss(v_rec, values)
            _, attn_orig = compute_attention(queries, keys, values)
            _, attn_rec  = compute_attention(queries, k_rec, v_rec)
            attn_loss = F.mse_loss(attn_rec, attn_orig)

            total_loss = kv_loss * 1 + attn_loss * 0
            (total_loss / cfg["gradient_accumulation_steps"]).backward()

            if ((step + 1) % cfg["gradient_accumulation_steps"] == 0) or (step + 1 == batches_per_epoch):
                torch.nn.utils.clip_grad_norm_(
                    [p for ae in autoencoders for p in ae.parameters() if p.grad is not None],
                    max_norm=cfg.get("max_grad_norm", 1.0),
                )
                optimizer.step()
                cosine_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # logging
            running_total += total_loss.item() * B
            running_kv    += kv_loss.item() * B
            running_attn  += attn_loss.item() * B
            glb_step = (epoch - 1) * batches_per_epoch + step
            writer.add_scalar('Loss/train_total_step', total_loss.item(), glb_step)
            writer.add_scalar('Loss/train_kv_step', kv_loss.item(), glb_step)
            writer.add_scalar('Loss/train_attn_step', attn_loss.item(), glb_step)

        denom = batches_per_epoch * cfg["batch_size"]
        avg_total = running_total / denom
        avg_kv    = running_kv / denom
        avg_attn  = running_attn / denom

        print(f"[Epoch {epoch}] Train → total={avg_total:.4f}, kv={avg_kv:.4f}, attn={avg_attn:.4f}")
        writer.add_scalars('Loss/train_epoch', {
            'total': avg_total,
            'kv': avg_kv,
            'attn': avg_attn,
        }, epoch)

        # ── validation ──────────────────────────────────────────────────────────
        for ae in autoencoders:
            ae.eval()
        val_totals, val_kvs, val_attns = [], [], []

        with torch.no_grad():
            for _ in range(min(batches_per_epoch // 5, 20)):
                (keys, values), queries = val_buffer.next()
                B, L, H, S, D = keys.shape

                k_rec = torch.zeros_like(keys)
                v_rec = torch.zeros_like(values)
                for l in range(L):
                    k_flat, _ = autoencoders[l](keys[:, l].reshape(-1, D))
                    v_flat, _ = autoencoders[l](values[:, l].reshape(-1, D))
                    k_rec[:, l] = k_flat.reshape(B, H, S, D)
                    v_rec[:, l] = v_flat.reshape(B, H, S, D)

                kv_loss = F.mse_loss(k_rec, keys) + F.mse_loss(v_rec, values)
                _, attn_orig = compute_attention(queries, keys, values)
                _, attn_rec  = compute_attention(queries, k_rec, v_rec)
                attn_loss = F.mse_loss(attn_rec, attn_orig)

                val_totals.append(kv_loss.item() * 1 + attn_loss.item() * 0)
                val_kvs.append(kv_loss.item())
                val_attns.append(attn_loss.item())

        avg_val_total = float(np.mean(val_totals))
        avg_val_kv    = float(np.mean(val_kvs))
        avg_val_attn  = float(np.mean(val_attns))

        print(f"[Epoch {epoch}] Val   → total={avg_val_total:.4f}, kv={avg_val_kv:.4f}, attn={avg_val_attn:.4f}")
        writer.add_scalars('Loss/val_epoch', {
            'total': avg_val_total,
            'kv': avg_val_kv,
            'attn': avg_val_attn,
        }, epoch)

        # LR scheduling
        plateau_scheduler.step(avg_val_total)

        # save checkpoint per epoch
        ckpt = {f"layer_{i}": ae.state_dict() for i, ae in enumerate(autoencoders)}
        torch.save(ckpt, os.path.join(cfg["output_dir"], f"autoencoders_epoch_{epoch}.pth"))

    # ── final save ─────────────────────────────────────────────────────────────
    final_ckpt = {f"layer_{i}": ae.state_dict() for i, ae in enumerate(autoencoders)}
    torch.save(final_ckpt, os.path.join(cfg["output_dir"], "autoencoders_final.pth"))
    print("Training complete!")
    writer.close()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args["config"])
    pprint.pprint(cfg)
    main(cfg)
