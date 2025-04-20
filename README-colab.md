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
./run_experiments.sh configs/qwen25_7b_experiment.json
./run_experiments.sh configs/qwen25_3b_experiment.json
./run_experiments.sh configs/qwen25_1.5b_experiment.json
./run_experiments.sh configs/qwen25_0.5b_experiment.json
```
