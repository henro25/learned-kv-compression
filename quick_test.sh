#!/bin/bash
# Quick test script for KV Cache compression with minimal parameters

# Set small values for quick testing
MODEL="distilgpt2"
LATENT_DIM="16"
NUM_EPOCHS="1"
NUM_TRAIN_TEXTS="10"
CACHE_SIZE="1"  # 1MB cache size for quick testing
BATCH_SIZE="512"  # Smaller batch size for faster testing
NUM_RUNS="3"  # Fewer runs for faster testing

# Create output directories
TEST_MODEL_DIR="test_models/${MODEL}_latent${LATENT_DIM}"
TEST_RESULTS_DIR="test_results"

mkdir -p $TEST_MODEL_DIR
mkdir -p $TEST_RESULTS_DIR

echo "==== Quick Test: KV Cache Compression ===="
echo "Model: $MODEL"
echo "Latent dimension: $LATENT_DIM"
echo "Number of epochs: $NUM_EPOCHS"
echo "Number of training texts: $NUM_TRAIN_TEXTS"
echo "Cache size: $CACHE_SIZE MB"
echo "Batch size: $BATCH_SIZE"
echo "Number of runs: $NUM_RUNS"
echo "========================================"

# Step 1: Train a small autoencoder
echo "Step 1: Training autoencoder..."
python -m src.dictionary_learning.train \
    --name $MODEL \
    --latent_dim $LATENT_DIM \
    --num_epochs $NUM_EPOCHS \
    --num_train_texts $NUM_TRAIN_TEXTS \
    --output_dir $TEST_MODEL_DIR

# Step 2: Run a small benchmark test
echo "Step 2: Running benchmark test..."
python -m src.inference.benchmark \
    --model $MODEL \
    --autoencoder "${TEST_MODEL_DIR}/autoencoders_final.pth" \
    --latent_dim $LATENT_DIM \
    --batch_size $BATCH_SIZE \
    --num_runs $NUM_RUNS \
    --sizes $CACHE_SIZE \
    --output $TEST_RESULTS_DIR

echo "Quick test completed!"
echo "Check ${TEST_RESULTS_DIR} for results and visualizations." 