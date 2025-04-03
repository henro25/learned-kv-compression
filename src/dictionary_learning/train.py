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
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--num_train_texts", type=int, default=1000, help="Number of training texts to use")
    parser.add_argument("--num_eval_texts", type=int, default=100, help="Number of evaluation texts to use")
    return vars(parser.parse_args())

def main(cfg):
    # Set random seeds.
    SEED = cfg["seed"]
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    cfg["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create output directory
    os.makedirs(cfg["output_dir"], exist_ok=True)
    
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

    writer = SummaryWriter(log_dir=f'runs/{cfg["name"]}_{cfg["latent_dim"]}')

    # Load WikiText dataset via Hugging Face datasets.
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    
    # Filter out empty texts and take a subset for training and evaluation
    texts_train = [text for text in dataset["train"]["text"] if text.strip()][:cfg["num_train_texts"]]
    texts_test = [text for text in dataset["test"]["text"] if text.strip()][:cfg["num_eval_texts"]]
    
    print(f"Using {len(texts_train)} texts for training and {len(texts_test)} texts for evaluation")

    # Initialize Buffers for training and evaluation.
    # The Buffer class should feed batches of KV pairs extracted from the model for given texts.
    train_buffer = Buffer(cfg, model, tokenizer, texts_train)
    eval_buffer = Buffer(cfg, model, tokenizer, texts_test)

    optimizer = optim.Adam(autoencoder.parameters(), lr=cfg["lr"])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["num_epochs"], eta_min=0)
    batches_per_epoch = len(texts_train) // cfg["batch_size"]

    best_eval_loss = float('inf')
    
    for epoch in range(cfg["num_epochs"]):
        autoencoder.train()
        epoch_loss = 0.0
        
        for i in tqdm.trange(batches_per_epoch, desc=f"Epoch {epoch+1}/{cfg['num_epochs']}"):
            # Get a batch of KV pairs from the training buffer.
            kvs = train_buffer.next()  # Expected shape: (batch_size, num_kv, head_dim)
            loss, recon, latent = autoencoder(kvs)
            
            # Compute compression ratio
            original_size = kvs.numel() * kvs.element_size()  # in bytes
            compressed_size = latent.numel() * latent.element_size()  # in bytes
            compression_ratio = original_size / compressed_size
            
            # Log the compression ratio
            writer.add_scalar('Compression/ratio', compression_ratio, epoch * batches_per_epoch + i)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item() * kvs.size(0)
            writer.add_scalar('Loss/train', loss.item(), epoch * batches_per_epoch + i)
            
            # Log MSE and the reconstruction/original correlation every 100 batches
            if i % 100 == 0:
                mse = torch.mean((recon - kvs) ** 2)
                writer.add_scalar('MSE/train', mse.item(), epoch * batches_per_epoch + i)
                
                # Calculate correlation between original and reconstructed values
                recon_flat = recon.view(-1).detach()
                kvs_flat = kvs.view(-1).detach()
                correlation = torch.corrcoef(torch.stack([recon_flat, kvs_flat]))[0, 1]
                writer.add_scalar('Correlation/train', correlation.item(), epoch * batches_per_epoch + i)
        
        epoch_loss /= (batches_per_epoch * cfg["batch_size"])
        print(f"Epoch {epoch+1}/{cfg['num_epochs']}, Average Loss: {epoch_loss:.4f}")
        scheduler.step()

        # Save a checkpoint at the end of each epoch.
        checkpoint_path = os.path.join(cfg["output_dir"], f"autoencoder_epoch_{epoch+1}.pth")
        torch.save(autoencoder.state_dict(), checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

        # Evaluate every eval_interval epochs.
        if (epoch + 1) % cfg["eval_interval"] == 0:
            autoencoder.eval()
            eval_losses = []
            eval_correlations = []
            num_eval_batches = min(batches_per_epoch // 5, 20)  # Limit evaluation to a reasonable number of batches
            
            with torch.no_grad():
                for i in range(num_eval_batches):
                    kvs = eval_buffer.next()
                    eval_loss, recon, _ = autoencoder(kvs)
                    eval_losses.append(eval_loss.item())
                    
                    # Calculate correlation between original and reconstructed values
                    recon_flat = recon.view(-1).detach()
                    kvs_flat = kvs.view(-1).detach()
                    correlation = torch.corrcoef(torch.stack([recon_flat, kvs_flat]))[0, 1]
                    eval_correlations.append(correlation.item())
                
                avg_eval_loss = np.mean(eval_losses)
                avg_eval_correlation = np.mean(eval_correlations)
                
                writer.add_scalar('Loss/eval', avg_eval_loss, epoch+1)
                writer.add_scalar('Correlation/eval', avg_eval_correlation, epoch+1)
                print(f"Evaluation Loss at epoch {epoch+1}: {avg_eval_loss:.4f}, Correlation: {avg_eval_correlation:.4f}")
                
                # Save the best model based on evaluation loss
                if avg_eval_loss < best_eval_loss:
                    best_eval_loss = avg_eval_loss
                    best_model_path = os.path.join(cfg["output_dir"], "autoencoder_best.pth")
                    torch.save(autoencoder.state_dict(), best_model_path)
                    print(f"New best model saved at {best_model_path}")

    # Save the final autoencoder model.
    final_model_path = os.path.join(cfg["output_dir"], "autoencoder_final.pth")
    torch.save(autoencoder.state_dict(), final_model_path)
    print(f"Final model saved at {final_model_path}")
    
    # Save the configuration used for training alongside the model
    config_path = os.path.join(cfg["output_dir"], "autoencoder_config.json")
    with open(config_path, 'w') as f:
        json.dump(cfg, f, indent=2)
    
    writer.close()
    
    print("Training complete!")
    print(f"Best evaluation loss: {best_eval_loss:.4f}")
    print(f"Models saved in {cfg['output_dir']}")

if __name__ == "__main__":
    args = parse_args()
    args = {key: value for key, value in args.items() if value is not None}
    config_from_file = load_config(args["config"])
    config_from_file.update(args)
    pprint.pprint(config_from_file)
    main(config_from_file)
    