#!/bin/bash
#SBATCH --job-name=kv_compression
#SBATCH --output=kv_compression_%j.log
#SBATCH --error=kv_compression_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# Usage: sbatch run_experiments.sh [model_name] [latent_dims] [cache_sizes] [num_epochs] [num_train_texts] [batch_size] [num_runs]
# Example: sbatch run_experiments.sh "Qwen/Qwen2.5-7B"  "1 2 3" "10 100 150" "1 2 3" "100 1000 1500" "64 65 66" "1 2 3"
# run_experiments.sh "Qwen/Qwen2.5-7B" "8 16" "1000" "5" "10000" "64" "2" "bf16" "./src/configs/qwen25_7b_config.json"
MODEL=${1:-distilgpt2}
LATENT_DIMS=${2:-"8 16 32"}
CACHE_SIZES=${3:-"1 10 100 1000"}
NUM_EPOCHS=${4:-5}
NUM_TRAIN_TEXTS=${5:-10000}
BATCH_SIZE=${6:-64}
NUM_RUNS=${7:-5}
DATA_TYPE=${8:-"f32"}
CONFIG_FILE=${9:-"default_config.json"}
OUTPUT_DIR="experiment_results_${MODEL}"

# Print configuration
echo "==== KV Cache Compression Experiment ===="
echo "Model: $MODEL"
echo "Latent dimensions: $LATENT_DIMS"
echo "Cache sizes (MB): $CACHE_SIZES"
echo "Number of epochs: $NUM_EPOCHS"
echo "Number of training texts: $NUM_TRAIN_TEXTS"
echo "Batch size: $BATCH_SIZE"
echo "Number of runs for timing: $NUM_RUNS"
echo "Output directory: $OUTPUT_DIR"
echo "========================================"

# Load necessary modules (adjust for your cluster)
# module load cuda/11.7
# module load python/3.9

# Run the experiment
echo "Starting experiment at $(date)"
python run_experiments.py \
    --model $MODEL \
    --latent_dims $LATENT_DIMS \
    --cache_sizes $CACHE_SIZES \
    --num_epochs $NUM_EPOCHS \
    --num_train_texts $NUM_TRAIN_TEXTS \
    --batch_size $BATCH_SIZE \
    --num_runs $NUM_RUNS \
    --dtype $DATA_TYPE \
    --config $CONFIG_FILE \
    --output_dir $OUTPUT_DIR 
echo "Experiment completed at $(date)"

# Run comparison analysis
echo "Generating comparison report..."
mkdir -p $OUTPUT_DIR/comparison

# Use the experiment_summary.json file to get result directories
if [ -f "$OUTPUT_DIR/experiment_summary.json" ]; then
    echo "Using experiment summary to find result directories"
    # Extract result directories from the experiment summary using jq if available
    if command -v jq &> /dev/null; then
        RESULT_DIRS=$(jq -r '.results[].result_dir' "$OUTPUT_DIR/experiment_summary.json" | tr '\n' ' ')
    else
        # Fallback for when jq is not available - collect directories based on directory structure
        RESULT_DIRS=""
        for latent_dim in $LATENT_DIMS; do
            # Find all benchmark directories matching the pattern
            for dir in "$OUTPUT_DIR"/benchmark_"${MODEL}"_latent"${latent_dim}"*; do
                if [ -d "$dir" ]; then
                    RESULT_DIRS="$RESULT_DIRS $dir"
                fi
            done
        done
    fi
else
    # Fallback to previous logic but adapted for new directory structure
    echo "Experiment summary not found, using pattern matching to find result directories"
    RESULT_DIRS=""
    for latent_dim in $LATENT_DIMS; do
        # Find all benchmark directories matching the pattern
        for dir in "$OUTPUT_DIR"/benchmark_"${MODEL}"_latent"${latent_dim}"*; do
            if [ -d "$dir" ]; then
                RESULT_DIRS="$RESULT_DIRS $dir"
            fi
        done
    done
fi

echo "Found result directories: $RESULT_DIRS"

# Run comparison if we have result directories
if [ -n "$RESULT_DIRS" ]; then
    python -m src.analysis.compare_results \
        --results $RESULT_DIRS \
        --output $OUTPUT_DIR/comparison
    echo "Comparison analysis complete!"
else
    echo "No result directories found for comparison analysis!"
fi

echo "Experiment and analysis complete!" 
