#!/bin/bash
#SBATCH --job-name=hybrid_benchmark
#SBATCH --output=logs/hybrid_benchmark_%j.out
#SBATCH --error=logs/hybrid_benchmark_%j.err
#SBATCH --time=8:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# Usage: sbatch run_hybrid_benchmark.sh MODEL_NAME AUTOENCODER_PATH [QUANT_BITS=8]
# Example: sbatch run_hybrid_benchmark.sh distilgpt2 ./outputs/autoencoder_ckpt.pt 6

# Ensure logs directory exists
mkdir -p logs

# Default values
MODEL_NAME=${1:-"distilgpt2"}
AUTOENCODER_PATH=${2:-""}
QUANT_BITS=${3:-8}
OUTPUT_DIR="results/hybrid_${MODEL_NAME}_${QUANT_BITS}bits"

# Validate arguments
if [ -z "$AUTOENCODER_PATH" ]; then
    echo "Error: Autoencoder path is required."
    echo "Usage: sbatch run_hybrid_benchmark.sh MODEL_NAME AUTOENCODER_PATH [QUANT_BITS=8]"
    exit 1
fi

# Print configuration
echo "======== Hybrid Compression Benchmark ========"
echo "Model: $MODEL_NAME"
echo "Autoencoder: $AUTOENCODER_PATH"
echo "Quantization bits: $QUANT_BITS"
echo "Output directory: $OUTPUT_DIR"
echo "=============================================="

# Log start time
echo "Starting benchmark at $(date)"

# Run the benchmark
python benchmark_hybrid.py \
    --model $MODEL_NAME \
    --autoencoder $AUTOENCODER_PATH \
    --quant_bits $QUANT_BITS \
    --output $OUTPUT_DIR

# Log end time
echo "Benchmark completed at $(date)"

# Print summary from results
if [ -f "${OUTPUT_DIR}/hybrid_benchmark_results.json" ]; then
    echo "======== Benchmark Summary ========"
    python -c "
import json
with open('${OUTPUT_DIR}/hybrid_benchmark_results.json', 'r') as f:
    results = json.load(f)
    print(f'Average speedup: {results[\"avg_speedup\"]:.2f}x')
    print(f'Average memory reduction: {results[\"avg_mem_reduction\"]:.2f}x')
    print(f'Compression ratio: {results[\"compression_ratio\"]:.2f}x')
"
    echo "Results and plots saved to ${OUTPUT_DIR}/"
else
    echo "No results file found. Benchmark may have failed."
fi 