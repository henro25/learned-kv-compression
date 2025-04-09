# Learned KV Cache Compression

This repository contains code for compressing the KV cache in transformer-based language models using learned autoencoders. The key insight is that we can use asymmetric autoencoders (deep encoder, shallow decoder) to achieve better compression quality while maintaining fast decompression speeds.

## Authors

- Ben Choi (1)
- Henry Huang (2)

(1) Harvard College  
(2) Harvard College

## Project Overview

When serving LLMs, a significant memory bottleneck is storing and transferring KV caches between CPU and GPU memory, especially for long contexts. This project implements an autoencoder-based compression technique for KV caches to:

1. Reduce the memory footprint of KV caches
2. Speed up the transfer of KV caches between CPU and GPU
3. Minimize the time to first token when resuming generation from a stored context

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/bjpchoi/lkvc.git
cd lkvc
pip install -r requirements.txt
```

## Project Structure

- `src/models/autoencoder.py`: Autoencoder implementation for compressing KV cache vectors
- `src/utils/buffer.py`: Buffer for extracting KV cache examples from a pre-trained model
- `src/dictionary_learning/train.py`: Script for training the autoencoder on WikiText data
- `src/inference/inference.py`: Utilities for inference with compressed KV caches
- `src/inference/benchmark.py`: Benchmarking script to measure time to first token and other metrics
- `src/configs/`: Configuration files for different models and experiments

## Training an Autoencoder

To train an autoencoder for KV cache compression:

```bash
python -m src.dictionary_learning.train \
    --name distilgpt2 \
    --latent_dim 16 \
    --num_epochs 10 \
    --batch_size 32 \
    --output_dir models/distilgpt2_16 \
    --num_train_texts 1000
```

This trains an autoencoder that compresses each KV vector to a 16-dimensional latent representation using 1000 texts from WikiText-103.

## Running Experiments

To test the tradeoffs between symmetric and asymmetric autoencoder architectures:

```bash
python -m src.experiments.autoencoder_tradeoffs
```

This will train and evaluate 3 symmetric and 2 asymmetric architectures, with varying training times, and generate plots in the `experiment_results` directory.

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

After running experiments, visualization files will be saved to the output directory:
- `compression_ratio_vs_quality.png`: Shows how compression ratio affects reconstruction quality
- `encoder_size_vs_decompression_time.png`: Shows the relationship between encoder size and decompression speed
- `encoder_size_vs_quality.png`: Shows how encoder size affects reconstruction quality
- `speed_quality_tradeoff.png`: Shows the tradeoff between decompression speed and reconstruction quality
- `quality_vs_size.png`: Shows how total model size affects reconstruction quality
- `training_effect.png`: Shows the effect of training epochs on reconstruction quality

## Future Directions

Possible improvements to explore:
- Non-linear autoencoders with more layers for better compression
- Quantization on top of autoencoder compression
- Per-layer specialized autoencoders to capture layer-specific patterns
- Pruning techniques to identify and compress only the most important KV vectors

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{learned-kv-compression,
  author = {Ben Choi and Henry Huang},
  title = {Learned KV Cache Compression},
  howpublished = {\url{https://github.com/bjpchoi/lkvc}},
  note = {Harvard College}
}
```
