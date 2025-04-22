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
from torch.utils.tensorboard import SummaryWriter
import argparse
import json
import pprint
import random
import numpy as np
import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW, get_cosine_schedule_with_warmup
from src.models.autoencoder import Autoencoder
from src.utils.buffer import Buffer

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config

def parse_args():
    parser = argparse.ArgumentParser(description="Train layer-wise Autoencoders for KV Cache Compression")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    return vars(parser.parse_args())

def visualize_attention_differences(original_attn, recon_attn, layer_idx, head_idx, save_path=None):
    """
    Visualize the differences between original and reconstructed attention matrices.
    """
    original = original_attn[0].detach().cpu().float().numpy()
    recon = recon_attn[0].detach().cpu().float().numpy()
    diff = np.abs(original - recon)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    im1 = axes[0].imshow(original, cmap='viridis')
    axes[0].set_title('Original Attention')
    axes[0].set_xlabel('Key Position')
    axes[0].set_ylabel('Query Position')
    plt.colorbar(im1, ax=axes[0])
    im2 = axes[1].imshow(recon, cmap='viridis')
    axes[1].set_title('Reconstructed Attention')
    axes[1].set_xlabel('Key Position')
    axes[1].set_ylabel('Query Position')
    plt.colorbar(im2, ax=axes[1])
    im3 = axes[2].imshow(diff, cmap='hot')
    axes[2].set_title('Absolute Difference')
    axes[2].set_xlabel('Key Position')
    axes[2].set_ylabel('Query Position')
    plt.colorbar(im3, ax=axes[2])
    fig.suptitle(f'Layer {layer_idx}, Head {head_idx}')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def compute_attention(q, k, v):
    batch_size, num_layers, num_heads, seq_len, head_dim = q.shape
    q_reshaped = q.reshape(-1, num_heads, seq_len, head_dim)
    k_reshaped = k.reshape(-1, num_heads, seq_len, head_dim)
    v_reshaped = v.reshape(-1, num_heads, seq_len, head_dim)
    scores = torch.matmul(q_reshaped, k_reshaped.transpose(-2, -1)) / (head_dim ** 0.5)
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, v_reshaped)
    attention_output = output.reshape(batch_size, num_layers, num_heads, seq_len, head_dim)
    attention_weights = attn_weights.reshape(batch_size, num_layers, num_heads, seq_len, seq_len)
    return attention_output, attention_weights

def main(cfg):
    SEED = cfg["seed"]
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    head_dim = cfg["hidden_size"] // cfg["num_attention_heads"]

    if cfg.get("dtype") == "bf16":
        cfg["dtype"] = torch.bfloat16
    elif cfg.get("dtype") in ("fp16","f16"):
        cfg["dtype"] = torch.float16
    else:
        cfg["dtype"] = torch.float32
    print(f"Using dtype: {cfg['dtype']}")

    os.makedirs(cfg["output_dir"], exist_ok=True)
    os.makedirs(os.path.join(cfg["output_dir"], "attention_viz"), exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg["name"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["name"],
        trust_remote_code=(cfg["name"].startswith("Qwen")),
        torch_dtype=cfg["dtype"],
        device_map={"": cfg["device"]},
        output_hidden_states=True,
        output_attentions=True,
        use_cache=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.add_special_tokens({'pad_token':'[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    model.eval()

    autoencoders = nn.ModuleList()
    for _ in range(cfg["num_hidden_layers"]):
        ae = Autoencoder(input_dim=head_dim, latent_dim=cfg["latent_dim"], dtype=cfg["dtype"])
        autoencoders.append(ae.to(cfg["device"]))

    writer = SummaryWriter(log_dir=f'runs/{cfg["name"]}_{cfg["latent_dim"]}')

    wiki_dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    longbench_subsets = ['narrativeqa','hotpotqa','2wikimqa','musique','dureader']
    longbench_texts = []
    for subset in longbench_subsets:
        try:
            dataset = load_dataset("THUDM/LongBench", subset)
            longbench_texts.extend(dataset["test"]["input"])
        except:
            pass

    wiki_train = [t for t in wiki_dataset["train"]["text"] if t.strip()]
    num_wiki = int(cfg["num_train_texts"] * 0.7)
    num_long = cfg["num_train_texts"] - num_wiki
    wiki_train = random.sample(wiki_train, min(num_wiki, len(wiki_train)))
    long_train = random.sample([t for t in longbench_texts if t.strip()], min(num_long, len(longbench_texts)))
    texts_train = wiki_train + long_train
    random.shuffle(texts_train)
    texts_test = [t for t in wiki_dataset["test"]["text"] if t.strip()][:cfg["num_eval_texts"]]
    texts_test.extend(longbench_texts[cfg["num_train_texts"]:cfg["num_train_texts"]+cfg["num_eval_texts"]])

    train_buffer = Buffer(cfg, model, tokenizer, texts_train)
    eval_buffer  = Buffer(cfg, model, tokenizer, texts_test)

    # Compute number of updates per epoch
    batches_per_epoch = len(texts_train) // cfg["batch_size"]
    # Setup AdamW optimizer with weight decay and cosine scheduler with warmup
    total_steps = (batches_per_epoch // cfg["gradient_accumulation_steps"]) * cfg["num_epochs"]
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

    for epoch in range(cfg["num_epochs"]):
        for ae in autoencoders:
            ae.train()
        epoch_loss = 0.0
        for i in tqdm.trange(batches_per_epoch, desc=f"Epoch {epoch+1}/{cfg['num_epochs']}"):
            (keys, values), queries = train_buffer.next()
            B, L, H, S, D = keys.shape
            k_recon = torch.zeros_like(keys)
            v_recon = torch.zeros_like(values)
            for l in range(L):
                k_l = keys[:,l].reshape(-1, D)
                v_l = values[:,l].reshape(-1, D)
                k_flat, _ = autoencoders[l](k_l)
                v_flat, _ = autoencoders[l](v_l)
                k_rec = k_flat.reshape(B, H, S, D)
                v_rec = v_flat.reshape(B, H, S, D)
                k_recon[:,l] = k_rec
                v_recon[:,l] = v_rec

            orig_attn, _  = compute_attention(queries, keys, values)
            recon_attn, _ = compute_attention(queries, k_recon, v_recon)
            loss = F.mse_loss(recon_attn, orig_attn) / cfg["gradient_accumulation_steps"]
            loss.backward()
            if (i+1) % cfg["gradient_accumulation_steps"] == 0 or (i+1) == batches_per_epoch:
                # Clip gradients before stepping
                torch.nn.utils.clip_grad_norm_(
                    [p for ae in autoencoders for p in ae.parameters() if p.grad is not None],
                    max_norm=cfg.get("max_grad_norm", 1.0)
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()
            epoch_loss += loss.item() * B * cfg["gradient_accumulation_steps"]
            writer.add_scalar('Loss/train', loss.item(), epoch*batches_per_epoch + i)

        epoch_loss /= (batches_per_epoch * cfg["batch_size"])
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")
        scheduler.step()
        ckpt = {f"layer_{i}": ae.state_dict() for i, ae in enumerate(autoencoders)}
        path = os.path.join(cfg["output_dir"], f"autoencoders_epoch_{epoch+1}.pth")
        torch.save(ckpt, path)
        print(f"Saved {path}")

        if (epoch+1) % cfg["eval_interval"] == 0:
            # Periodic evaluation (no best-model checkpoint saved)
            for ae in autoencoders:
                ae.eval()
            eval_losses = []
            for _ in range(min(batches_per_epoch//5, 20)):
                (keys, values), queries = eval_buffer.next()
                B, L, H, S, D = keys.shape
                k_recon = torch.zeros_like(keys)
                v_recon = torch.zeros_like(values)
                for l in range(L):
                    k_l = keys[:,l].reshape(-1, D)
                    v_l = values[:,l].reshape(-1, D)
                    k_flat, _ = autoencoders[l](k_l)
                    v_flat, _ = autoencoders[l](v_l)
                    k_rec = k_flat.reshape(B, H, S, D)
                    v_rec = v_flat.reshape(B, H, S, D)
                    k_recon[:,l] = k_rec
                    v_recon[:,l] = v_rec
                orig_attn, _  = compute_attention(queries, keys, values)
                recon_attn, _ = compute_attention(queries, k_recon, v_recon)
                eval_losses.append(F.mse_loss(recon_attn, orig_attn).item())
            avg_eval = np.mean(eval_losses)
            writer.add_scalar('Loss/eval', avg_eval, epoch+1)

    final_ckpt = {f"layer_{i}": ae.state_dict() for i, ae in enumerate(autoencoders)}
    torch.save(final_ckpt, os.path.join(cfg["output_dir"], "autoencoders_final.pth"))
    writer.close()
    print("Training complete!")

if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args["config"])
    pprint.pprint(cfg)
    main(cfg)