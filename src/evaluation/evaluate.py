import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import os
from src.models.autoencoder import Autoencoder
from src.utils.buffer import Buffer

def calculate_perplexity(model, tokenizer, texts, device, max_length=2048):
    """
    Calculate perplexity for a list of texts.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        texts: List of texts to evaluate
        device: Device to run on
        max_length: Maximum sequence length
        
    Returns:
        float: Average perplexity
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for text in tqdm(texts, desc="Calculating perplexity"):
            # Tokenize the text
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            
            # Get model outputs
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            
            # Accumulate loss and token count
            total_loss += loss.item() * input_ids.size(1)
            total_tokens += input_ids.size(1)
    
    # Calculate average perplexity
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    return perplexity

def evaluate_with_compressed_cache(model, tokenizer, autoencoder, texts, device, cfg):
    """
    Evaluate model using compressed KV cache.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        autoencoder: The trained autoencoder
        texts: List of texts to evaluate
        device: Device to run on
        cfg: Configuration dictionary
        
    Returns:
        float: Average perplexity
    """
    model.eval()
    autoencoder.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for text in tqdm(texts, desc="Evaluating with compressed cache"):
            # Tokenize the text
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=cfg["max_seq_len"])
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            
            # Get model outputs with caching
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True
            )
            
            # Compress and decompress KV cache
            past_key_values = outputs.past_key_values
            compressed_cache = []
            
            for layer in past_key_values:
                keys, values = layer
                # Compress keys and values
                k_compressed, _ = autoencoder(keys.reshape(-1, cfg["head_dim"]))
                v_compressed, _ = autoencoder(values.reshape(-1, cfg["head_dim"]))
                
                # Reshape back to original dimensions
                k_compressed = k_compressed.reshape(keys.shape)
                v_compressed = v_compressed.reshape(values.shape)
                compressed_cache.append((k_compressed, v_compressed))
            
            # Use compressed cache for next token prediction
            next_token_logits = model(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=compressed_cache,
                return_dict=True
            ).logits
            
            # Calculate loss
            shift_logits = next_token_logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            total_loss += loss.item() * (input_ids.size(1) - 1)
            total_tokens += (input_ids.size(1) - 1)
    
    # Calculate average perplexity
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    return perplexity

def evaluate_longbench(model, tokenizer, autoencoder, device, cfg):
    """
    Evaluate on LongBench dataset.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        autoencoder: The trained autoencoder
        device: Device to run on
        cfg: Configuration dictionary
        
    Returns:
        dict: Evaluation results
    """
    # Load LongBench dataset
    dataset = load_dataset("THUDM/LongBench")
    
    results = {
        "baseline": {},
        "compressed": {}
    }
    
    # Evaluate on each subset
    for subset in ["narrative_qa", "hotpotqa", "2wikimqa", "musique", "dureader"]:
        print(f"\nEvaluating on {subset}...")
        texts = dataset[subset]["input"]
        
        # Calculate baseline perplexity
        baseline_ppl = calculate_perplexity(model, tokenizer, texts, device)
        results["baseline"][subset] = baseline_ppl
        
        # Calculate perplexity with compressed cache
        compressed_ppl = evaluate_with_compressed_cache(model, tokenizer, autoencoder, texts, device, cfg)
        results["compressed"][subset] = compressed_ppl
        
        print(f"{subset} - Baseline PPL: {baseline_ppl:.2f}, Compressed PPL: {compressed_ppl:.2f}")
    
    return results

def plot_perplexity_comparison(results, save_path=None):
    """
    Plot comparison of perplexity between baseline and compressed models.
    
    Args:
        results: Dictionary containing evaluation results
        save_path: Path to save the plot (optional)
    """
    subsets = list(results["baseline"].keys())
    baseline_ppl = [results["baseline"][s] for s in subsets]
    compressed_ppl = [results["compressed"][s] for s in subsets]
    
    x = np.arange(len(subsets))
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, baseline_ppl, width, label='Baseline')
    plt.bar(x + width/2, compressed_ppl, width, label='Compressed Cache')
    
    plt.xlabel('Dataset Subset')
    plt.ylabel('Perplexity')
    plt.title('Perplexity Comparison: Baseline vs Compressed Cache')
    plt.xticks(x, subsets, rotation=45)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def main(cfg):
    # Set device
    device = torch.device(cfg["device"])
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["name"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["name"],
        torch_dtype=torch.float32,
        device_map={"": device},
        output_hidden_states=True,
        output_attentions=True,
        use_cache=True
    )
    
    # Load trained autoencoder
    autoencoder = Autoencoder(input_dim=cfg["head_dim"], latent_dim=cfg["latent_dim"]).to(device)
    autoencoder.load_state_dict(torch.load(os.path.join(cfg["output_dir"], "autoencoder_final.pth")))
    
    # Load evaluation texts
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
    eval_texts = [text for text in dataset["test"]["text"] if text.strip()][:cfg["num_eval_texts"]]
    
    # Calculate perplexity
    print("\nCalculating baseline perplexity...")
    baseline_ppl = calculate_perplexity(model, tokenizer, eval_texts, device)
    print(f"Baseline perplexity: {baseline_ppl:.2f}")
    
    print("\nCalculating perplexity with compressed cache...")
    compressed_ppl = evaluate_with_compressed_cache(model, tokenizer, autoencoder, eval_texts, device, cfg)
    print(f"Compressed cache perplexity: {compressed_ppl:.2f}")
    
    # Evaluate on LongBench
    print("\nEvaluating on LongBench...")
    longbench_results = evaluate_longbench(model, tokenizer, autoencoder, device, cfg)
    
    # Plot results
    plot_path = os.path.join(cfg["output_dir"], "perplexity_comparison.png")
    plot_perplexity_comparison(longbench_results, plot_path)
    
    # Save results
    results = {
        "baseline_perplexity": baseline_ppl,
        "compressed_perplexity": compressed_ppl,
        "longbench_results": longbench_results
    }
    
    with open(os.path.join(cfg["output_dir"], "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nEvaluation complete!")
    print(f"Results saved in {cfg['output_dir']}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        cfg = json.load(f)
    
    main(cfg) 