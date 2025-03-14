"""
Module Name: train.py
Description: Train Autoencoder for KV Cache Compression. This module trains an autoencoder to compress the key-value (KV) cache extracted from a pretrained transformer model. It uses the WikiText-103 dataset to provide input texts, and a custom Buffer class to extract batches of KV vectors from the model's caching mechanism. The autoencoder is then optimized using a mean squared error reconstruction loss. Training progress is logged via TensorBoard, and model checkpoints are saved at the end of each epoch.
Author: Henry Huang
Date: 2025-03-13
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import argparse
import json
import os
import pprint
import random
import numpy as np
import tqdm

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# Import your autoencoder and Buffer.
from src.models.autoencoder import Autoencoder
from src.utils.buffer import Buffer  # Ensure you have a Buffer class in src/utils/buffer.py

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config

def parse_args():
    parser = argparse.ArgumentParser(description="Train Autoencoder for KV Cache Compression")
    parser.add_argument("--config", type=str, default="src/configs/default_config.json", help="Path to config file")
    # Optional overrides:
    parser.add_argument("--name", type=str, help="Pretrained model name or path")
    parser.add_argument("--input_dim", type=int, help="Dimension of KV vectors")
    parser.add_argument("--latent_dim", type=int, help="Dimension of latent code")
    parser.add_argument("--batch_size", type=int, help="Training batch size")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs")
    parser.add_argument("--head_dim", type=int, help="Dimension of each head")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--lr", type=float, help="Learning rate")
    return vars(parser.parse_args())

def main(cfg):
    # Set random seeds.
    SEED = cfg["seed"]
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    cfg["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the pretrained model and tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(cfg["name"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["name"],
        torch_dtype=torch.float16,
        device_map={"": cfg["device"]}  # assign all layers to the chosen device
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    model.eval()

    # Initialize the autoencoder.
    # Here we assume each KV vector has dimension equal to the head dimension.
    autoencoder = Autoencoder(input_dim=cfg["head_dim"], latent_dim=cfg["latent_dim"]).to(cfg["device"])

    writer = SummaryWriter(log_dir=f'runs/{cfg["name"]}')

    # Load WikiText dataset via Hugging Face datasets.
    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    texts_train = dataset["train"]["text"][:6]
    texts_test = dataset["test"]["text"][:6]

    # Initialize Buffers for training and evaluation.
    # The Buffer class should feed batches of KV pairs extracted from the model for given texts.
    train_buffer = Buffer(cfg, model, tokenizer, texts_train)
    eval_buffer = Buffer(cfg, model, tokenizer, texts_test)

    optimizer = optim.Adam(autoencoder.parameters(), lr=cfg["lr"])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["num_epochs"], eta_min=0)
    batches_per_epoch = len(texts_train) // cfg["batch_size"]

    for epoch in range(cfg["num_epochs"]):
        autoencoder.train()
        epoch_loss = 0.0
        for i in tqdm.trange(batches_per_epoch, desc=f"Epoch {epoch+1}/{cfg['num_epochs']}"):
            # Get a batch of KV pairs from the training buffer.
            kvs = train_buffer.next()  # Expected shape: (batch_size, num_kv, head_dim)
            loss, recon, _ = autoencoder(kvs)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item() * kvs.size(0)
            writer.add_scalar('Loss/train', loss.item(), epoch * batches_per_epoch + i)
        epoch_loss /= (batches_per_epoch * cfg["batch_size"])
        print(f"Epoch {epoch+1}/{cfg['num_epochs']}, Average Loss: {epoch_loss:.4f}")
        scheduler.step()

        # Save a checkpoint at the end of each epoch.
        torch.save(autoencoder.state_dict(), f"autoencoder_epoch_{epoch+1}.pth")
        print(f"Checkpoint saved at the end of epoch {epoch+1}")

        # Evaluate every eval_interval epochs.
        if (epoch + 1) % cfg["eval_interval"] == 0:
            autoencoder.eval()
            with torch.no_grad():
                kvs = eval_buffer.next()
                eval_loss, _, _ = autoencoder(kvs)
                writer.add_scalar('Loss/eval', eval_loss.item(), epoch+1)
                print(f"Evaluation Loss at epoch {epoch+1}: {eval_loss.item():.4f}")

    # Save the final autoencoder model.
    torch.save(autoencoder.state_dict(), "autoencoder.pth")
    print("Autoencoder model saved as autoencoder.pth")
    writer.close()

if __name__ == "__main__":
    args = parse_args()
    args = {key: value for key, value in args.items() if value is not None}
    config_from_file = load_config(args["config"])
    config_from_file.update(args)
    pprint.pprint(config_from_file)
    main(config_from_file)
    