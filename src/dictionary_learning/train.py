#!/usr/bin/env python3
"""
Module Name: train.py
Description: Train layer-wise Autoencoders for KV Cache Compression. This module trains one autoencoder per self-attention layer to compress the KV cache from a pretrained transformer model.
Author: Henry Huang
Date: 2025-03-13
"""

import os
import sys
import random
import json
import pprint

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

def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # ── set sensible defaults for new keys ───────────────
    cfg.setdefault("encoder_layer_sizes", [])   # means 1‑layer encoder
    cfg.setdefault("decoder_layer_sizes", [])
    cfg.setdefault("activation", "ReLU")
    return cfg

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Train layer-wise Autoencoders for KV Cache Compression")
    parser.add_argument(
        "--config", type=str,
        default="src/configs/default_config.json",
        help="Path to config file"
    )
    return vars(parser.parse_args())

def compute_attention(q, k, v):
    B, L, H, S, D = q.shape
    q2 = q.reshape(-1, H, S, D)
    k2 = k.reshape(-1, H, S, D)
    v2 = v.reshape(-1, H, S, D)
    scores = torch.matmul(q2, k2.transpose(-2, -1)) / (D ** 0.5)
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v2)
    out = output.reshape(B, L, H, S, D)
    return out, attn_weights.reshape(B, L, H, S, S)

def main(cfg):
    # reproducibility
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])

    # dtype
    if cfg.get("dtype") == "bf16":
        dtype = torch.bfloat16
    elif cfg.get("dtype") in ("fp16","f16"):
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
        use_cache=True
    )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token':'[PAD]'})
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

    # load datasets
    wiki = load_dataset("wikitext", "wikitext-103-raw-v1")
    long_texts = []
    for subset in ['narrativeqa','hotpotqa','2wikimqa','musique','dureader']:
        try:
            ds = load_dataset("THUDM/LongBench", subset)
            long_texts.extend(ds["test"]["input"])
        except:
            pass

    # assemble and shuffle
    all_train = [t for t in wiki["train"]["text"] if t.strip()]
    n_w = int(cfg["num_train_texts"] * 0.7)
    n_l = cfg["num_train_texts"] - n_w
    train_texts = random.sample(all_train, min(n_w, len(all_train))) \
                  + random.sample([t for t in long_texts if t.strip()], min(n_l, len(long_texts)))
    random.shuffle(train_texts)

    # split out validation
    val_size = int(len(train_texts) * cfg.get("val_split", 0.1))
    texts_val   = train_texts[:val_size]
    texts_train = train_texts[val_size:]

    train_buffer = Buffer(cfg, model, tokenizer, texts_train)
    val_buffer   = Buffer(cfg, model, tokenizer, texts_val)

    # optimizer + schedulers
    batches_per_epoch = len(texts_train) // cfg["batch_size"]
    total_steps       = (batches_per_epoch // cfg["gradient_accumulation_steps"]) * cfg["num_epochs"]
    warmup_steps      = int(cfg.get("warmup_ratio",0.1) * total_steps)

    optimizer = AdamW(
        [p for ae in autoencoders for p in ae.parameters()],
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay",1e-3),
        betas=(0.9,0.999)
    )
    cosine_scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    plateau_scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=cfg.get("lr_reduce_factor", 0.5),
        patience=cfg.get("lr_patience", 2),
        verbose=True
    )

    # training loop
    for epoch in range(1, cfg["num_epochs"]+1):
        for ae in autoencoders: ae.train()
        running_loss = 0.0

        for step in tqdm.trange(batches_per_epoch, desc=f"Epoch {epoch}/{cfg['num_epochs']}"):
            (keys, values), queries = train_buffer.next()
            B, L, H, S, D = keys.shape

            # reconstruction
            k_rec = torch.zeros_like(keys)
            v_rec = torch.zeros_like(values)
            for l in range(L):
                k_flat, _ = autoencoders[l](keys[:,l].reshape(-1, D))
                v_flat, _ = autoencoders[l](values[:,l].reshape(-1, D))
                k_rec[:,l] = k_flat.reshape(B, H, S, D)
                v_rec[:,l] = v_flat.reshape(B, H, S, D)

            kv_loss = F.mse_loss(k_rec, keys) + F.mse_loss(v_rec, values)
            loss    = kv_loss / cfg["gradient_accumulation_steps"]
            loss.backward()

            if (step+1) % cfg["gradient_accumulation_steps"] == 0 or step+1 == batches_per_epoch:
                torch.nn.utils.clip_grad_norm_(
                    [p for ae in autoencoders for p in ae.parameters() if p.grad is not None],
                    max_norm=cfg.get("max_grad_norm",1.0)
                )
                optimizer.step()
                cosine_scheduler.step()
                optimizer.zero_grad()

            running_loss += loss.item() * B * cfg["gradient_accumulation_steps"]
            writer.add_scalar('Loss/train_step', loss.item(), (epoch-1)*batches_per_epoch + step)

        avg_train_loss = running_loss / (batches_per_epoch * cfg["batch_size"])
        print(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.4f}")
        writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch)

        # validation
        for ae in autoencoders: ae.eval()
        val_losses = []
        with torch.no_grad():
            for _ in range(min(batches_per_epoch//5, 20)):
                (keys, values), queries = val_buffer.next()
                B, L, H, S, D = keys.shape
                k_rec = torch.zeros_like(keys)
                v_rec = torch.zeros_like(values)
                for l in range(L):
                    k_flat, _ = autoencoders[l](keys[:,l].reshape(-1, D))
                    v_flat, _ = autoencoders[l](values[:,l].reshape(-1, D))
                    k_rec[:,l] = k_flat.reshape(B, H, S, D)
                    v_rec[:,l] = v_flat.reshape(B, H, S, D)

                kv_loss = F.mse_loss(k_rec, keys) + F.mse_loss(v_rec, values)
                # **raw** loss, no division here
                val_losses.append(kv_loss.item())

        avg_val_loss = np.mean(val_losses)
        print(f"[Epoch {epoch}] Val Loss:   {avg_val_loss:.4f}")
        writer.add_scalar('Loss/val_epoch', avg_val_loss, epoch)

        # LR reduction on plateau
        plateau_scheduler.step(avg_val_loss)

        # save checkpoint
        ckpt = {f"layer_{i}": ae.state_dict() for i, ae in enumerate(autoencoders)}
        path = os.path.join(cfg["output_dir"], f"autoencoders_epoch_{epoch}.pth")
        torch.save(ckpt, path)

    # final save
    final_ckpt = {f"layer_{i}": ae.state_dict() for i, ae in enumerate(autoencoders)}
    final_path = os.path.join(cfg["output_dir"], "autoencoders_final.pth")
    torch.save(final_ckpt, final_path)
    print(f"Saved final checkpoint to: {final_path}")
    writer.close()
    print("Training complete!")

if __name__ == "__main__":
    args = parse_args()
    cfg  = load_config(args["config"])
    pprint.pprint(cfg)
    main(cfg)