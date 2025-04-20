"""
Module Name: train.py
Description: Train Autoencoder for KV Cache Compression. This module trains an autoencoder to compress the key-value (KV) cache extracted from a pretrained transformer model. It uses the WikiText-103 dataset to provide input texts, and a custom Buffer class to extract batches of KV vectors from the model's caching mechanism. The autoencoder is then optimized using a mean squared error reconstruction loss. Training progress is logged via TensorBoard, and model checkpoints are saved at the end of each epoch.
Author: Henry Huang
Date: 2025-03-13
"""

import os
import sys
import json
import torch
import random
import pprint
import numpy as np
import argparse
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import trange
from datasets import load_dataset
from torch import optim
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.autoencoder import Autoencoder
from src.utils.buffer import Buffer


def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return vars(parser.parse_args())


def visualize_attention_differences(original_attn, recon_attn, layer_idx, head_idx, save_path=None):
    original = original_attn[0].detach().cpu().float().numpy()
    recon = recon_attn[0].detach().cpu().float().numpy()
    diff = np.abs(original - recon)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    im1 = axes[0].imshow(original, cmap='viridis')
    axes[0].set_title('Original Attention')
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(recon, cmap='viridis')
    axes[1].set_title('Reconstructed Attention')
    plt.colorbar(im2, ax=axes[1])

    im3 = axes[2].imshow(diff, cmap='hot')
    axes[2].set_title('Absolute Difference')
    plt.colorbar(im3, ax=axes[2])

    fig.suptitle(f'Layer {layer_idx}, Head {head_idx}')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def compute_attention(q, k, v):
    bsz, n_layers, n_heads, seq_len, head_dim = q.shape
    q = q.reshape(-1, n_heads, seq_len, head_dim)
    k = k.reshape(-1, n_heads, seq_len, head_dim)
    v = v.reshape(-1, n_heads, seq_len, head_dim)

    scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
    weights = F.softmax(scores, dim=-1)
    output = torch.matmul(weights, v)

    attn_output = output.reshape(bsz, n_layers, n_heads, seq_len, head_dim)
    attn_weights = weights.reshape(bsz, n_layers, n_heads, seq_len, seq_len)
    return attn_output, attn_weights


def main(cfg):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])

    head_dim = cfg["hidden_size"] // cfg["num_attention_heads"]

    # Convert dtype string
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "float32": torch.float32}
    cfg["dtype"] = dtype_map.get(cfg.get("dtype", "float32"), torch.float32)
    print(f"Using dtype: {cfg['dtype']}")

    os.makedirs(cfg["output_dir"], exist_ok=True)
    if cfg.get("enable_visualizations", True):
        os.makedirs(os.path.join(cfg["output_dir"], "attention_viz"), exist_ok=True)

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(cfg["name"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["name"],
        trust_remote_code=True if "Qwen" in cfg["name"] else False,
        torch_dtype=cfg["dtype"],
        device_map={"": cfg["device"]},
        output_hidden_states=True,
        output_attentions=True,
        use_cache=True
    ).eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    autoencoder = Autoencoder(
        input_dim=head_dim,
        latent_dim=cfg["latent_dim"],
        dtype=cfg["dtype"]
    ).to(cfg["device"])

    if cfg.get("use_compile", True):
        try:
            autoencoder = torch.compile(autoencoder)
        except Exception as e:
            print(f"torch.compile() failed: {e}")

    writer = SummaryWriter(log_dir=f'runs/{cfg["name"]}_{cfg["latent_dim"]}')

    # Load dataset
    wiki = load_dataset("wikitext", "wikitext-103-raw-v1")
    longbench = []
    for subset in ['narrativeqa', 'hotpotqa', '2wikimqa', 'musique', 'dureader']:
        try:
            d = load_dataset("THUDM/LongBench", subset)["test"]["input"]
            longbench += [x for x in d if x.strip()]
        except:
            continue

    wiki_train = random.sample([t for t in wiki["train"]["text"] if t.strip()], int(cfg["num_train_texts"] * 0.7))
    long_train = random.sample(longbench, cfg["num_train_texts"] - len(wiki_train))
    texts_train = wiki_train + long_train
    random.shuffle(texts_train)

    texts_test = [t for t in wiki["test"]["text"] if t.strip()][:cfg["num_eval_texts"]]
    texts_test += longbench[cfg["num_train_texts"]:cfg["num_train_texts"] + cfg["num_eval_texts"]]

    train_buffer = Buffer(cfg, model, tokenizer, texts_train)
    eval_buffer = Buffer(cfg, model, tokenizer, texts_test)

    optimizer = optim.Adam(autoencoder.parameters(), lr=cfg["lr"])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["num_epochs"])
    batches_per_epoch = len(texts_train) // cfg["batch_size"]

    best_eval_loss = float("inf")

    for epoch in range(cfg["num_epochs"]):
        autoencoder.train()
        epoch_loss = 0.0

        for i in trange(batches_per_epoch, desc=f"Epoch {epoch+1}", disable=cfg.get("fast_mode", False)):
            with torch.no_grad():
                kvs, queries = train_buffer.next()
            keys, values = kvs

            keys_flat = keys.reshape(-1, head_dim)
            values_flat = values.reshape(-1, head_dim)

            k_recon_flat, _ = autoencoder(keys_flat)
            v_recon_flat, _ = autoencoder(values_flat)

            k_recon = k_recon_flat.reshape(keys.shape)
            v_recon = v_recon_flat.reshape(values.shape)

            attn_orig, weights_orig = compute_attention(queries, keys, values)
            attn_recon, weights_recon = compute_attention(queries, k_recon, v_recon)

            loss = F.mse_loss(attn_recon, attn_orig) / cfg["gradient_accumulation_steps"]
            loss.backward()

            if (i + 1) % cfg["gradient_accumulation_steps"] == 0 or (i + 1) == batches_per_epoch:
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * keys.size(0) * cfg["gradient_accumulation_steps"]
            writer.add_scalar('Loss/train', loss.item(), epoch * batches_per_epoch + i)

            if not cfg.get("fast_mode") and cfg.get("enable_visualizations", True) and i % 1000 == 0:
                with torch.no_grad():
                    attn_diff = torch.norm(attn_orig - attn_recon, p='fro')
                    writer.add_scalar('Attention/difference', attn_diff.item(), epoch * batches_per_epoch + i)
                    try:
                        corr = torch.corrcoef(torch.stack([weights_orig.view(-1), weights_recon.view(-1)]))[0, 1]
                        writer.add_scalar('Attention/correlation', corr.item(), epoch * batches_per_epoch + i)
                    except:
                        pass

                    visualize_attention_differences(
                        weights_orig[:, 0, 0],
                        weights_recon[:, 0, 0],
                        0, 0,
                        os.path.join(cfg["output_dir"], "attention_viz",
                                     f"epoch{epoch+1}_batch{i}.png")
                    )

        avg_loss = epoch_loss / (batches_per_epoch * cfg["batch_size"])
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
        scheduler.step()

        torch.save(autoencoder.state_dict(), os.path.join(cfg["output_dir"], f"autoencoder_epoch_{epoch+1}.pth"))

        if (epoch + 1) % cfg["eval_interval"] == 0:
            autoencoder.eval()
            losses, diffs = [], []
            with torch.no_grad():
                for _ in range(min(20, batches_per_epoch // 5)):
                    kvs, queries = eval_buffer.next()
                    keys, values = kvs
                    k_recon = autoencoder(keys.reshape(-1, head_dim))[0].reshape(keys.shape)
                    v_recon = autoencoder(values.reshape(-1, head_dim))[0].reshape(values.shape)
                    attn_o, _ = compute_attention(queries, keys, values)
                    attn_r, _ = compute_attention(queries, k_recon, v_recon)
                    losses.append(F.mse_loss(attn_r, attn_o).item())
                    diffs.append(torch.norm(attn_o - attn_r, p='fro').item())

            mean_loss, mean_diff = np.mean(losses), np.mean(diffs)
            writer.add_scalar('Loss/eval', mean_loss, epoch + 1)
            writer.add_scalar('Attention/difference_eval', mean_diff, epoch + 1)

            if mean_loss < best_eval_loss:
                best_eval_loss = mean_loss
                torch.save(autoencoder.state_dict(), os.path.join(cfg["output_dir"], "autoencoder_final.pth"))

    torch.save(autoencoder.state_dict(), os.path.join(cfg["output_dir"], "autoencoder_final.pth"))
    with open(os.path.join(cfg["output_dir"], "autoencoder_config.json"), 'w') as f:
        json.dump(cfg, f, indent=2)
    writer.close()

    print("Training complete!")
    print(f"Best evaluation loss: {best_eval_loss:.4f}")


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args["config"])
    print("Training with configuration:")
    pprint.pprint(config)
    main(config)
    