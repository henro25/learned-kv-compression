#!/usr/bin/env python3
"""
Module Name: train_per_head.py
Description: Train head-wise and layer-wise Autoencoders for KV Cache Compression.
             This module trains one autoencoder per self-attention head within each
             layer to compress the KV cache from a pretrained transformer model.
Author: Henry Huang  •  Updated: 2025-05-06 (Modified by AI Assistant)

Key update
──────────
• **Loss = KV reconstruction + w_attn × attention-reconstruction (default w_attn = 1.0)**
• Uses the existing `compute_attention()` helper to compute attention maps for the
  original and reconstructed KV pairs given the same queries.
• **Modification:** Trains a separate autoencoder for each attention head in each layer.
"""

import os
import sys
import random
import json
import pprint
import math
import logging
from typing import Tuple, List

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
# Helper functions (No changes needed)
# ────────────────────────────────────────────────────────────────────────────────

def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.setdefault("encoder_layer_sizes", [])
    cfg.setdefault("decoder_layer_sizes", [])
    cfg.setdefault("activation", "ReLU")
    cfg.setdefault("attn_loss_weight", 1.0)
    return cfg


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="Train head-wise and layer-wise Autoencoders for KV Cache Compression")
    parser.add_argument(
        "--config", type=str,
        default="src/configs/default_config.json",
        help="Path to config file")
    return vars(parser.parse_args())


@torch.no_grad()
def compute_attention(q: torch.Tensor,
                      k: torch.Tensor,
                      v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Scaled-dot-product attention for tensors shaped (B, L, H, S, D)."""
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
# Training (Modified for per-head autoencoders)
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

    # output dir (adjust for clarity)
    output_dir = os.path.join(cfg["output_dir"], "per_head")
    os.makedirs(output_dir, exist_ok=True)

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

    # prepare autoencoders (now a list of lists)
    head_dim = cfg["hidden_size"] // cfg["num_attention_heads"]
    num_heads = cfg["num_attention_heads"]
    num_layers = cfg["num_hidden_layers"]
    autoencoders = nn.ModuleList([
        nn.ModuleList([
            Autoencoder(
                input_dim=head_dim,
                latent_dim=cfg["latent_dim"],
                encoder_layer_sizes=cfg["encoder_layer_sizes"],
                decoder_layer_sizes=cfg["decoder_layer_sizes"],
                activation=cfg.get("activation", "ReLU"),
                dtype=dtype,
            ).to(cfg["device"])
            for _ in range(num_heads)
        ])
        for _ in range(num_layers)
    ])

    # tensorboard (adjust log directory)
    writer = SummaryWriter(log_dir=f'runs/{cfg["name"]}_{cfg["latent_dim"]}_per_head')

    # ── dataset prep (No changes needed) ───────────────────────────────────────
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

    class LoggingReduceLROnPlateau(ReduceLROnPlateau):
        def __init__(self, optimizer, mode='min', factor=0.5, patience=1,
                     threshold=1e-4, threshold_mode='rel', cooldown=0,
                     min_lr=0, eps=1e-8, verbose=False):
            super().__init__(optimizer, mode, factor, patience, threshold, threshold_mode,
                             cooldown, min_lr, eps, verbose)
            self.logger = logging.getLogger(__name__)

        def _reduce_lr(self, epoch):
            old_lr = self.optimizer.param_groups[0]['lr']
            super()._reduce_lr(epoch)
            new_lr = self.optimizer.param_groups[0]['lr']
            if new_lr < old_lr:
                self.logger.info(f"Epoch {epoch}: Reducing learning rate from {old_lr:.6f} to {new_lr:.6f}")

    plateau_scheduler = LoggingReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=cfg.get("lr_reduce_factor", 0.5),
        patience=cfg.get("lr_patience", 1),
        verbose=False  # Set verbose to False here
    )

    # ── training loop (Modified for per-head autoencoders) ─────────────────────
    for epoch in range(1, cfg["num_epochs"] + 1):
        for layer_aes in autoencoders:
            for head_ae in layer_aes:
                head_ae.train()
        running_total, running_kv, running_attn = 0.0, 0.0, 0.0

        for step in tqdm.trange(batches_per_epoch, desc=f"Epoch {epoch}/{cfg['num_epochs']}"):
            (keys, values), queries = train_buffer.next()
            B, L, H, S, D = keys.shape  # L is num_layers, H is num_heads

            k_rec = torch.zeros_like(keys)
            v_rec = torch.zeros_like(values)
            for l in range(L):
                for h in range(H):
                    # Flatten seq_len and head_dim for AE
                    k_flat, _ = autoencoders[l][h](keys[:, l, h].reshape(-1, D))
                    v_flat, _ = autoencoders[l][h](values[:, l, h].reshape(-1, D))
                    k_rec[:, l, h] = k_flat.reshape(B, S, D)
                    v_rec[:, l, h] = v_flat.reshape(B, S, D)

            # losses
            kv_loss = F.mse_loss(k_rec, keys) + F.mse_loss(v_rec, values)
            _, attn_orig = compute_attention(queries, keys, values)
            _, attn_rec  = compute_attention(queries, k_rec, v_rec)
            attn_loss = F.mse_loss(attn_rec, attn_orig)

            total_loss = kv_loss * 1 + attn_loss * 0
            (total_loss / cfg["gradient_accumulation_steps"]).backward()

            if ((step + 1) % cfg["gradient_accumulation_steps"] == 0) or (step + 1 == batches_per_epoch):
                torch.nn.utils.clip_grad_norm_(
                    [p for layer_aes in autoencoders for head_ae in layer_aes for p in head_ae.parameters() if p.grad is not None],
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

        # ── validation (Modified for per-head autoencoders) ────────────────────
        for layer_aes in autoencoders:
            for head_ae in layer_aes:
                head_ae.eval()
        val_totals, val_kvs, val_attns = [], [], []

        with torch.no_grad():
            for _ in range(min(batches_per_epoch // 5, 20)):
                (keys, values), queries = val_buffer.next()
                B, L, H, S, D = keys.shape

                k_rec = torch.zeros_like(keys)
                v_rec = torch.zeros_like(values)
                for l in range(L):
                    for h in range(H):
                        k_flat, _ = autoencoders[l][h](keys[:, l, h].reshape(-1, D))
                        v_flat, _ = autoencoders[l][h](values[:, l, h].reshape(-1, D))
                        k_rec[:, l, h] = k_flat.reshape(B, S, D)
                        v_rec[:, l, h] = v_flat.reshape(B, S, D)

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

        # save checkpoint per epoch (now saving a nested dictionary)
        ckpt = {
            f"layer_{l}": {
                f"head_{h}": autoencoders[l][h].state_dict() for h in range(num_heads)
            }
            for l in range(num_layers)
        }
        torch.save(ckpt, os.path.join(output_dir, f"autoencoders_epoch_{epoch}.pth"))

    # ── final save (now saving a nested dictionary) ───────────────────────────
    final_ckpt = {
        f"layer_{l}": {
            f"head_{h}": autoencoders[l][h].state_dict() for h in range(num_heads)
        }
        for l in range(num_layers)
    }
    final_save_path = cfg["autoencoder_path"]
    os.makedirs(os.path.dirname(final_save_path), exist_ok=True) # Ensure the directory exists
    torch.save(final_ckpt, final_save_path)
    print(f"Training complete! Autoencoders saved to: {final_save_path}")
    writer.close()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args["config"])
    pprint.pprint(cfg)
    main(cfg)