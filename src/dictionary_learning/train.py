#!/usr/bin/env python3
"""
Module Name: train.py
Description: Train layer-wise Autoencoders for KV Cache Compression. This module trains one autoencoder per self-attention layer to compress the KV cache from a pretrained transformer model.
Author: Henry Huang
Date: 2025-03-13
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import argparse
import json
import pprint
import random
import numpy as np
import tqdm
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from src.models.autoencoder import Autoencoder
from src.utils.buffer import Buffer

class MMapDataset(Dataset):
    """A torch Dataset that reads K/V/Q from disk via np.memmap."""
    def __init__(self, key_path, value_path, query_path, shape, dtype, device):
        self.shape = shape  # (N, L, H, S, D)
        self.device = device
        self.dtype = dtype

        # Load the three memmaps in read-only mode:
        self.keys_mmap    = np.memmap(key_path,   mode='r', dtype=np.float32, shape=shape)
        self.values_mmap  = np.memmap(value_path, mode='r', dtype=np.float32, shape=shape)
        self.queries_mmap = np.memmap(query_path, mode='r', dtype=np.float32, shape=shape)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):
        # Grab the slice from each memmap, wrap in torch, send to device
        k = torch.from_numpy(self.keys_mmap[idx]).to(self.device, dtype=self.dtype)
        v = torch.from_numpy(self.values_mmap[idx]).to(self.device, dtype=self.dtype)
        q = torch.from_numpy(self.queries_mmap[idx]).to(self.device, dtype=self.dtype)
        return k, v, q

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def parse_args():
    parser = argparse.ArgumentParser(description="Train layer-wise Autoencoders for KV Cache Compression")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    return vars(parser.parse_args())

def visualize_attention_differences(original_attn, recon_attn, layer_idx, head_idx, save_path=None):
    original = original_attn[0].detach().cpu().numpy()
    recon    = recon_attn[0].detach().cpu().numpy()
    diff     = np.abs(original - recon)
    fig, axes = plt.subplots(1, 3, figsize=(15,5))
    axes[0].imshow(original, cmap='viridis'); axes[0].set_title('Original')
    axes[1].imshow(recon,    cmap='viridis'); axes[1].set_title('Reconstructed')
    axes[2].imshow(diff,     cmap='hot');    axes[2].set_title('Absolute Difference')
    fig.suptitle(f'Layer {layer_idx}, Head {head_idx}')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path); plt.close()
    else:
        plt.show()

def compute_attention(q, k, v):
    batch_size, num_layers, num_heads, seq_len, head_dim = q.shape
    q = q.reshape(-1, num_heads, seq_len, head_dim)
    k = k.reshape(-1, num_heads, seq_len, head_dim)
    v = v.reshape(-1, num_heads, seq_len, head_dim)
    scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
    attn_weights = F.softmax(scores, dim=-1)
    out = torch.matmul(attn_weights, v)
    attn_out = out.reshape(batch_size, num_layers, num_heads, seq_len, head_dim)
    attn_w   = attn_weights.reshape(batch_size, num_layers, num_heads, seq_len, seq_len)
    return attn_out, attn_w

def main(cfg):
    # reproducibility
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])

    # dtype setup
    if cfg.get("dtype") == "bf16":
        dtype = torch.bfloat16
    elif cfg.get("dtype") in ("fp16","f16"):
        dtype = torch.float16
    else:
        dtype = torch.float32
    cfg["dtype"] = dtype
    print(f"Using dtype: {dtype}")

    os.makedirs(cfg["output_dir"], exist_ok=True)
    os.makedirs(os.path.join(cfg["output_dir"], "attention_viz"), exist_ok=True)

    # load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(cfg["name"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["name"],
        trust_remote_code=(cfg["name"].startswith("Qwen")),
        torch_dtype=dtype,
        device_map={"": cfg["device"]},
        output_hidden_states=True,
        output_attentions=True,
        use_cache=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.resize_token_embeddings(len(tokenizer))
    model.eval()

    # build one autoencoder per layer
    head_dim = cfg["hidden_size"] // cfg["num_attention_heads"]
    autoencoders = nn.ModuleList([
        Autoencoder(input_dim=head_dim, latent_dim=cfg["latent_dim"], dtype=dtype).to(cfg["device"])
        for _ in range(cfg["num_hidden_layers"])
    ])
    # COMPILE each for speed
    for i, ae in enumerate(autoencoders):
        autoencoders[i] = torch.compile(ae)

    writer = SummaryWriter(log_dir=f'runs/{cfg["name"]}_{cfg["latent_dim"]}')

    # prepare text pools (unchanged) …
    wiki = load_dataset("wikitext", "wikitext-103-raw-v1")["train"]["text"]
    longbench_subsets = ['narrativeqa','hotpotqa','2wikimqa','musique','dureader']
    longbench_texts = []
    for subset in longbench_subsets:
        try:
            longbench_texts.extend(load_dataset("THUDM/LongBench", subset)["test"]["input"])
        except:
            pass
    wiki = [t for t in wiki if t.strip()]
    num_wiki = int(cfg["num_train_texts"] * 0.7)
    wiki_train = random.sample(wiki, min(num_wiki, len(wiki)))
    long_train = random.sample([t for t in longbench_texts if t.strip()],
                               min(cfg["num_train_texts"]-len(wiki_train), len(longbench_texts)))
    texts_train = wiki_train + long_train
    random.shuffle(texts_train)
    texts_test = [t for t in load_dataset("wikitext", "wikitext-103-raw-v1")["test"]["text"] if t.strip()][:cfg["num_eval_texts"]]
    texts_test += longbench_texts[cfg["num_train_texts"]:
                                  cfg["num_train_texts"]+cfg["num_eval_texts"]]

    # build buffers
    train_buffer = Buffer(cfg, model, tokenizer, texts_train)
    eval_buffer  = Buffer(cfg, model, tokenizer, texts_test)

    # compute batches per epoch
    batches_per_epoch = len(texts_train) // cfg["batch_size"]

    #
    # ─── PRECOMPUTE ALL KV/Q TO DISK ───────────────────────────────────────────
    #
    model.eval()
    # decide full cache size:
    N = batches_per_epoch * cfg["batch_size"]
    L = cfg["num_hidden_layers"]
    H = cfg["num_attention_heads"]
    S = train_buffer.buffer_seq_len
    D = head_dim

    # file paths
    key_path = os.path.join(cfg["output_dir"], "keys.dat")
    val_path = os.path.join(cfg["output_dir"], "values.dat")
    qry_path = os.path.join(cfg["output_dir"], "queries.dat")

    # create memmaps (float32 on disk)
    keys_mmap    = np.memmap(key_path, mode='w+', dtype=np.float32, shape=(N, L, H, S, D))
    values_mmap  = np.memmap(val_path, mode='w+', dtype=np.float32, shape=(N, L, H, S, D))
    queries_mmap = np.memmap(qry_path, mode='w+', dtype=np.float32, shape=(N, L, H, S, D))

    pos = 0
    with torch.no_grad():
        for _ in range(batches_per_epoch):
            (keys, values), queries = train_buffer.next()
            bsz = keys.size(0)

            # copy into memmap
            keys_mmap   [pos:pos+bsz] = keys.cpu().numpy()
            values_mmap [pos:pos+bsz] = values.cpu().numpy()
            queries_mmap[pos:pos+bsz] = queries.cpu().numpy()
            pos += bsz

    # flush and delete to free host memory
    del keys_mmap, values_mmap, queries_mmap
    del train_buffer
    torch.cuda.empty_cache()

    # ─── BUILD DATASET & DATALOADER FROM DISK ─────────────────────────────────
    ds = MMapDataset(
        key_path, val_path, qry_path,
        shape=(N, L, H, S, D),
        dtype=dtype,
        device=cfg["device"]
    )
    train_loader = DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True
    )
    batches_per_epoch = len(train_loader)

    # optimizer & scheduler (unchanged) …
    total_steps  = (batches_per_epoch // cfg["gradient_accumulation_steps"]) * cfg["num_epochs"]
    warmup_steps = int(cfg.get("warmup_ratio", 0.1) * total_steps)
    optimizer = AdamW(
        [p for ae in autoencoders for p in ae.parameters()],
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 1e-3),
        betas=(0.9, 0.999)
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # logging setup
    log_interval = cfg.get("log_interval", 50)
    global_step = 0

    # placeholders for reconstructions
    k_recon = v_recon = None

    # ─── TRAINING LOOP ───────────────────────────────────────────────────────
    for epoch in range(cfg["num_epochs"]):
        for ae in autoencoders:
            ae.train()

        epoch_loss_total = 0.0
        running_loss     = 0.0

        for i, (keys, values, queries) in enumerate(tqdm.trange(
                batches_per_epoch, desc=f"Epoch {epoch+1}/{cfg['num_epochs']}")):
            keys    = keys.to(cfg["device"])
            values  = values.to(cfg["device"])
            queries = queries.to(cfg["device"])

            B, L, H, S, D = keys.shape

            # pre-allocate or clear
            if k_recon is None or k_recon.shape != keys.shape:
                k_recon = torch.empty_like(keys)
                v_recon = torch.empty_like(values)
            else:
                k_recon.zero_()
                v_recon.zero_()

            # layer-wise AE
            for l in range(L):
                k_l = keys[:, l].reshape(-1, D)
                v_l = values[:, l].reshape(-1, D)
                k_flat, _ = autoencoders[l](k_l)
                v_flat, _ = autoencoders[l](v_l)
                k_recon[:, l] = k_flat.reshape(B, H, S, D)
                v_recon[:, l] = v_flat.reshape(B, H, S, D)

            # losses
            orig_attn, _  = compute_attention(queries, keys, values)
            recon_attn, _ = compute_attention(queries, k_recon, v_recon)
            kv_loss = F.mse_loss(k_recon, keys) + F.mse_loss(v_recon, values)
            loss    = kv_loss / cfg["gradient_accumulation_steps"]
            loss.backward()

            if (i+1) % cfg["gradient_accumulation_steps"] == 0 or (i+1) == batches_per_epoch:
                torch.nn.utils.clip_grad_norm_(
                    [p for ae in autoencoders for p in ae.parameters() if p.grad is not None],
                    max_norm=cfg.get("max_grad_norm", 1.0)
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # logging
            epoch_loss_total += loss.item() * cfg["batch_size"] * cfg["gradient_accumulation_steps"]
            running_loss     += loss.item()
            global_step      += 1

            if global_step % log_interval == 0:
                writer.add_scalar('Loss/train', running_loss / log_interval, global_step)
                running_loss = 0.0

        # epoch end
        epoch_loss = epoch_loss_total / (batches_per_epoch * cfg["batch_size"])
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

        # checkpoints & eval (unchanged) …
        if (epoch+1) % 5 == 0:
            ckpt = {f"layer_{i}": ae.state_dict() for i, ae in enumerate(autoencoders)}
            path = os.path.join(cfg["output_dir"], f"autoencoders_epoch_{epoch+1}.pth")
            torch.save(ckpt, path)
            print(f"Saved {path}")

        if (epoch+1) % cfg["eval_interval"] == 0:
            pass  # your existing eval_buffer-based eval

    # final save
    final_ckpt = {f"layer_{i}": ae.state_dict() for i, ae in enumerate(autoencoders)}
    final_path = os.path.join(cfg["output_dir"], "autoencoders_final.pth")
    torch.save(final_ckpt, final_path)
    print(f"Saved final autoencoder checkpoint to: {final_path}")

    writer.close()
    print("Training complete!")

if __name__ == "__main__":
    args = parse_args()
    cfg  = load_config(args["config"])
    pprint.pprint(cfg)
    main(cfg)