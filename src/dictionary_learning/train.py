"""
Module Name: train.py
Description: Train Autoencoder for KV Cache Compression. This module trains an autoencoder to compress the key-value (KV) cache extracted from a pretrained transformer model. It uses the WikiText-103 dataset to provide input texts, and a custom Buffer class to extract batches of KV vectors from the model's caching mechanism. The autoencoder is then optimized using a mean squared error reconstruction loss. Training progress is logged via TensorBoard, and model checkpoints are saved at the end of each epoch.
Author: Henry Huang
Date: 2025-03-13
"""

import os
import sys
# Add the root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
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
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    return vars(parser.parse_args())

def visualize_attention_differences(original_attn, recon_attn, layer_idx, head_idx, save_path=None):
    """
    Visualize the differences between original and reconstructed attention matrices.
    
    Args:
        original_attn: Original attention matrix of shape (batch_size, seq_len, seq_len)
        recon_attn: Reconstructed attention matrix of shape (batch_size, seq_len, seq_len)
        layer_idx: Layer index for the title
        head_idx: Head index for the title
        save_path: Path to save the visualization (optional)
    """
    # Take the first sample in the batch
    original = original_attn[0].detach().cpu().float().numpy()  # Convert BFloat16 to float32 first
    recon = recon_attn[0].detach().cpu().float().numpy()
    diff = np.abs(original - recon)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot original attention
    im1 = axes[0].imshow(original, cmap='viridis')
    axes[0].set_title('Original Attention')
    axes[0].set_xlabel('Key Position')
    axes[0].set_ylabel('Query Position')
    plt.colorbar(im1, ax=axes[0])
    
    # Plot reconstructed attention
    im2 = axes[1].imshow(recon, cmap='viridis')
    axes[1].set_title('Reconstructed Attention')
    axes[1].set_xlabel('Key Position')
    axes[1].set_ylabel('Query Position')
    plt.colorbar(im2, ax=axes[1])
    
    # Plot absolute difference
    im3 = axes[2].imshow(diff, cmap='hot')
    axes[2].set_title('Absolute Difference')
    axes[2].set_xlabel('Key Position')
    axes[2].set_ylabel('Query Position')
    plt.colorbar(im3, ax=axes[2])
    
    # Add overall title
    fig.suptitle(f'Layer {layer_idx}, Head {head_idx}')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def compute_attention(q, k, v):
    """
    Compute attention output using scaled dot-product attention.
    
    Args:
        q: Query tensor of shape (batch_size, num_layers, num_heads, seq_len, head_dim)
        k: Key tensor of shape (batch_size, num_layers, num_heads, seq_len, head_dim)
        v: Value tensor of shape (batch_size, num_layers, num_heads, seq_len, head_dim)
        
    Returns:
        tuple: (attention_output, attention_weights) where:
            - attention_output has shape (batch_size, num_layers, num_heads, seq_len, head_dim)
            - attention_weights has shape (batch_size, num_layers, num_heads, seq_len, seq_len)
    """
    # Get dimensions
    batch_size, num_layers, num_heads, seq_len, head_dim = q.shape
    
    # Reshape tensors to combine batch and layer dimensions for parallel processing
    # New shape: (batch_size * num_layers, num_heads, seq_len, head_dim)
    q_reshaped = q.reshape(-1, num_heads, seq_len, head_dim)
    k_reshaped = k.reshape(-1, num_heads, seq_len, head_dim)
    v_reshaped = v.reshape(-1, num_heads, seq_len, head_dim)
    
    # Compute attention scores in parallel for all layers
    # Shape: (batch_size * num_layers, num_heads, seq_len, seq_len)
    scores = torch.matmul(q_reshaped, k_reshaped.transpose(-2, -1)) / (head_dim ** 0.5)
    
    # Apply softmax
    attn_weights = F.softmax(scores, dim=-1)
    
    # Compute attention output in parallel
    # Shape: (batch_size * num_layers, num_heads, seq_len, head_dim)
    output = torch.matmul(attn_weights, v_reshaped)
    
    # Reshape back to original dimensions
    attention_output = output.reshape(batch_size, num_layers, num_heads, seq_len, head_dim)
    attention_weights = attn_weights.reshape(batch_size, num_layers, num_heads, seq_len, seq_len)
    
    return attention_output, attention_weights

def main(cfg):
    # Set random seeds.
    SEED = cfg["seed"]
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    head_dim = cfg["hidden_size"] // cfg["num_attention_heads"]
    
    # Convert dtype string to torch dtype
    if "dtype" not in cfg:
        print("Warning: dtype not found in config, defaulting to float32")
        cfg["dtype"] = torch.float32
    else:
        if cfg["dtype"] == "bf16":
            cfg["dtype"] = torch.bfloat16
        elif cfg["dtype"] == "fp16" or cfg["dtype"] == "f16":
            cfg["dtype"] = torch.float16
        else:
            cfg["dtype"] = torch.float32
            
    print(f"Using dtype: {cfg['dtype']}")
    
    # Create output direct
    os.makedirs(cfg["output_dir"], exist_ok=True)
    os.makedirs(os.path.join(cfg["output_dir"], "attention_viz"), exist_ok=True)
    
    # Load the pretrained model and tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(cfg["name"])
    if cfg["name"].split("/")[0] == "Qwen":
        model = AutoModelForCausalLM.from_pretrained(
            cfg["name"],
            trust_remote_code=True,
            torch_dtype=cfg["dtype"],
            device_map={"": cfg["device"]},
            output_hidden_states=True,  # Enable hidden states output
            output_attentions=True,     # Enable attention output
            use_cache=True              # Enable KV cache
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["name"],
            torch_dtype=cfg["dtype"],
            device_map={"": cfg["device"]},
            output_hidden_states=True,  # Enable hidden states output
            output_attentions=True,     # Enable attention output
            use_cache=True              # Enable KV cache
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
    model.eval()

    # Initialize the autoencoder with the proper dtype
    autoencoder = Autoencoder(
        input_dim=head_dim, 
        latent_dim=cfg["latent_dim"],
        dtype=cfg["dtype"]
    ).to(cfg["device"])

    writer = SummaryWriter(log_dir=f'runs/{cfg["name"]}_{cfg["latent_dim"]}')

    # Load WikiText dataset via Hugging Face datasets.
    wiki_dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    
    # Load LongBench datasets
    longbench_subsets = ['narrativeqa', 'hotpotqa', '2wikimqa', 'musique', 'dureader']
    longbench_texts = []
    for subset in longbench_subsets:
        try:
            dataset = load_dataset("THUDM/LongBench", subset)
            texts = dataset["test"]["input"]
            longbench_texts.extend(texts)
        except Exception as e:
            print(f"Warning: Could not load {subset}: {str(e)}")
    
    # Filter out empty texts and prepare training data
    wiki_train = [text for text in wiki_dataset["train"]["text"] if text.strip()]
    longbench_train = [text for text in longbench_texts if text.strip()]
    
    # Calculate how many texts to take from each dataset
    num_wiki = int(cfg["num_train_texts"] * 0.7)  # 70% from WikiText
    num_longbench = cfg["num_train_texts"] - num_wiki  # 30% from LongBench
    
    # Randomly sample from each dataset
    wiki_train = random.sample(wiki_train, min(num_wiki, len(wiki_train)))
    longbench_train = random.sample(longbench_train, min(num_longbench, len(longbench_train)))
    
    # Combine and shuffle the training data
    texts_train = wiki_train + longbench_train
    random.shuffle(texts_train)
    
    texts_test = [text for text in wiki_dataset["test"]["text"] if text.strip()][:cfg["num_eval_texts"]]
    texts_test.extend(longbench_texts[cfg["num_train_texts"]:cfg["num_train_texts"] + cfg["num_eval_texts"]])  # Add LongBench texts
    
    print(f"Using {len(texts_train)} texts for training and {len(texts_test)} texts for evaluation")
    print(f"Training data includes {len(texts_train) - cfg['num_train_texts']} LongBench texts")
    print(f"Evaluation data includes {len(texts_test) - cfg['num_eval_texts']} LongBench texts")

    # Initialize Buffers for training and evaluation.
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
            # Get a batch of KV pairs and queries from the training buffer.
            kvs, queries = train_buffer.next()
            keys, values = kvs
            
            # Flatten for processing through autoencoder
            keys_flat = keys.reshape(-1, head_dim)
            values_flat = values.reshape(-1, head_dim)
            
            # Forward pass through autoencoder
            k_recon_flat, k_latent = autoencoder(keys_flat)
            v_recon_flat, v_latent = autoencoder(values_flat)
            
            # Reshape back to original dimensions
            k_recon = k_recon_flat.reshape(keys.shape)
            v_recon = v_recon_flat.reshape(values.shape)
            
            # Compute original and reconstructed attention
            original_attn, original_weights = compute_attention(queries, keys, values)
            recon_attn, recon_weights = compute_attention(queries, k_recon, v_recon)
            
            # Compute attention-preserving loss
            loss = F.mse_loss(recon_attn, original_attn)
            
            # Scale the loss by gradient accumulation steps
            loss = loss / cfg["gradient_accumulation_steps"]
            
            # Backward pass (accumulate gradients)
            loss.backward()
            
            # Only update weights after accumulating gradients for specified steps
            if (i + 1) % cfg["gradient_accumulation_steps"] == 0 or (i + 1) == batches_per_epoch:
                optimizer.step()
                optimizer.zero_grad()
                
                # Free up memory
                torch.cuda.empty_cache()
            
            epoch_loss += loss.item() * keys.size(0) * cfg["gradient_accumulation_steps"]
            writer.add_scalar('Loss/train', loss.item(), epoch * batches_per_epoch + i)
            
            # Log attention matrix difference every 100 batches
            if i % 100 == 0:
                with torch.no_grad():
                    # Calculate Frobenius norm of difference
                    attn_diff = torch.norm(original_attn - recon_attn, p='fro')
                    writer.add_scalar('Attention/difference', attn_diff.item(), epoch * batches_per_epoch + i)
                    
                    # Calculate correlation between original and reconstructed attention weights
                    correlation = torch.corrcoef(torch.stack([original_weights.view(-1), recon_weights.view(-1)]))[0, 1]
                    writer.add_scalar('Attention/correlation', correlation.item(), epoch * batches_per_epoch + i)
                    
                    # Visualize attention differences for a specific layer and head
                    layer_idx = 0  # First layer
                    head_idx = 0   # First head
                    save_path = os.path.join(
                        cfg["output_dir"], 
                        "attention_viz", 
                        f"epoch_{epoch+1}_batch_{i}_layer_{layer_idx}_head_{head_idx}.png"
                    )
                    visualize_attention_differences(
                        original_weights[:, layer_idx, head_idx],
                        recon_weights[:, layer_idx, head_idx],
                        layer_idx,
                        head_idx,
                        save_path
                    )
        
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
            eval_attn_diffs = []
            num_eval_batches = min(batches_per_epoch // 5, 20)
            
            with torch.no_grad():
                for i in range(num_eval_batches):
                    kvs, queries = eval_buffer.next()
                    keys, values = kvs
                    
                    # Forward pass through autoencoder
                    keys_flat = keys.reshape(-1, head_dim)
                    values_flat = values.reshape(-1, head_dim)
                    k_recon_flat, _ = autoencoder(keys_flat)
                    v_recon_flat, _ = autoencoder(values_flat)
                    
                    # Reshape back to original dimensions
                    k_recon = k_recon_flat.reshape(keys.shape)
                    v_recon = v_recon_flat.reshape(values.shape)
                    
                    # Compute attention matrices
                    original_attn, original_weights = compute_attention(queries, keys, values)
                    recon_attn, recon_weights = compute_attention(queries, k_recon, v_recon)
                    
                    # Compute loss and attention difference
                    eval_loss = F.mse_loss(recon_attn, original_attn)
                    eval_losses.append(eval_loss.item())
                    
                    attn_diff = torch.norm(original_attn - recon_attn, p='fro')
                    eval_attn_diffs.append(attn_diff.item())
                    
                    # Visualize attention differences for evaluation
                    if i == 0:  # Only visualize first batch
                        layer_idx = 0
                        head_idx = 0
                        save_path = os.path.join(
                            cfg["output_dir"],
                            "attention_viz",
                            f"eval_epoch_{epoch+1}_layer_{layer_idx}_head_{head_idx}.png"
                        )
                        visualize_attention_differences(
                            original_weights[:, layer_idx, head_idx],
                            recon_weights[:, layer_idx, head_idx],
                            layer_idx,
                            head_idx,
                            save_path
                        )
                
                avg_eval_loss = np.mean(eval_losses)
                avg_eval_attn_diff = np.mean(eval_attn_diffs)
                
                writer.add_scalar('Loss/eval', avg_eval_loss, epoch+1)
                writer.add_scalar('Attention/difference_eval', avg_eval_attn_diff, epoch+1)
                print(f"Evaluation Loss at epoch {epoch+1}: {avg_eval_loss:.4f}, Attention Difference: {avg_eval_attn_diff:.4f}")
                
                # Save the best model based on evaluation loss
                if avg_eval_loss < best_eval_loss:
                    best_eval_loss = avg_eval_loss
                    best_model_path = os.path.join(cfg["output_dir"], "autoencoder_final.pth")
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
    config = load_config(args["config"])
    print("Training with configuration:")
    pprint.pprint(config)
    main(config)
    