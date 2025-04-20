# Learned KV Cache Compression

This repository contains code for training and evaluating autoencoder-based compression of key-value (KV) caches for Large Language Models (LLMs).

## Project Overview

When serving LLMs, a significant memory bottleneck is storing and transferring KV caches between CPU and GPU memory, especially for long contexts. This project implements an autoencoder-based compression technique for KV caches to:

1. Reduce the memory footprint of KV caches
2. Speed up the transfer of KV caches between CPU and GPU
3. Minimize the time to first token when resuming generation from a stored context

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/learned-kv-compression.git
cd learned-kv-compression
pip install -r requirements.txt
```

## To Run

The program is designed to run the experiments using a single config file. Under the hood, we are loading the model, training it in src/dictionary_learning/train.py, and then running benchmarks afterwards in src/inference/benchmark.

Below are the provided, supported config files and ways to run our experiment:

```bash
<<<<<<< HEAD
python -m src.dictionary_learning.train \
    --name distilgpt2 \
    --latent_dim 16 \
    --num_epochs 10 \
    --batch_size 32 \
    --output_dir models/distilgpt2_16 \
    --num_train_texts 1000
```

This trains an autoencoder that compresses each KV vector to a 16-dimensional latent representation using 1000 texts from WikiText-103.

## Running Benchmarks

After training an autoencoder, you can benchmark its performance with different KV cache sizes:

```bash
python -m src.inference.benchmark \
    --model distilgpt2 \
    --autoencoder models/distilgpt2_16/autoencoder_final.pth \
    --sizes 1 10 100 1000 3000 \
    --output results/distilgpt2_16
```

This generates KV caches of different sizes (in MB) and measures:
- Time to first token with and without compression
- Compression ratio
- Speedup from using compression

## Running a Single Inference Test

To test a specific KV cache size:

```bash
python -m src.inference.inference \
    --model distilgpt2 \
    --size 100 \
    --autoencoder models/distilgpt2_16/autoencoder_final.pth \
    --output results.json
```

## Experiment Workflow

The full experimental workflow involves:

1. Training autoencoders with different latent dimensions on WikiText
2. Running benchmarks with varying KV cache sizes (1KB, 1MB, 100MB, 1GB, 3GB, 10GB, 20GB)
3. Comparing time to first token with and without compression
4. Plotting results and analyzing the tradeoff between compression ratio and quality

## Results Visualization

After running benchmarks, visualization files will be saved to the output directory:
- `time_comparison.png`: Comparison of time to first token with and without compression
- `compression_ratio.png`: Achieved compression ratio for different cache sizes
- `speedup.png`: Speedup factor (baseline time / compressed time)

## Future Directions

Possible improvements to explore:
- Non-linear autoencoders with more layers for better compression
- Quantization on top of autoencoder compression
- Per-layer specialized autoencoders to capture layer-specific patterns
- Pruning techniques to identify and compress only the most important KV vectors

## Citation

If you use this code in your research, please cite:

```
@misc{huang2025kvcompression,
  author = {Huang, Henry},
  title = {Learned KV Cache Compression},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/learned-kv-compression}}
}
```
=======
./run_experiments.sh /n/holylabs/LABS/meng_lab/Lab/learned-kv-compression/configs/qwen25_7b_experiment.json
./run_experiments.sh /n/holylabs/LABS/meng_lab/Lab/learned-kv-compression/configs/qwen25_3b_experiment.json
./run_experiments.sh /n/holylabs/LABS/meng_lab/Lab/learned-kv-compression/configs/qwen25_1.5b_experiment.json
./run_experiments.sh /n/holylabs/LABS/meng_lab/Lab/learned-kv-compression/configs/qwen25_0.5b_experiment.json
```
>>>>>>> main
