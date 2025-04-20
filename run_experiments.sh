#!/bin/bash
#SBATCH --job-name=kv_compression
#SBATCH --output=kv_compression_%j.log
#SBATCH --error=kv_compression_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

<<<<<<< HEAD
# Usage: sbatch run_experiment.sh [model_name] [latent_dims] [cache_sizes] [num_epochs] [num_train_texts] [batch_size] [num_runs]
# Example: sbatch run_experiment.sh distilgpt2 "8 16 32" "1 10 100 1000 3000" 5 1000 1024 5

# Default values
MODEL=${1:-distilgpt2}
LATENT_DIMS=${2:-"8 16 32"}
CACHE_SIZES=${3:-"1 10 100 1000"}
NUM_EPOCHS=${4:-5}
NUM_TRAIN_TEXTS=${5:-10000}
BATCH_SIZE=${6:-64}
NUM_RUNS=${7:-5}
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
=======
# Usage: sbatch run_experiments.sh [config_file]
# Example: sbatch run_experiments.sh ./configs/qwen_experiment.json
CONFIG_FILE=${1:-"./configs/default_experiment.json"}

# Print configuration
echo "==== KV Cache Compression Experiment ===="
echo "Config file: $CONFIG_FILE"
>>>>>>> main
echo "========================================"

# Load necessary modules (adjust for your cluster)
# module load cuda/11.7
# module load python/3.9

<<<<<<< HEAD
# Activate virtual environment (adjust path for your setup)
source venv/bin/activate

# Convert space-separated arguments to proper format for Python script
LATENT_DIMS_ARG=$(echo $LATENT_DIMS | tr ' ' ',')
CACHE_SIZES_ARG=$(echo $CACHE_SIZES | tr ' ' ',')

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
    --output_dir $OUTPUT_DIR

=======
# Run the experiment
echo "Starting experiment at $(date)"
python run_experiments.py --config $CONFIG_FILE
>>>>>>> main
echo "Experiment completed at $(date)"

# Run comparison analysis
echo "Generating comparison report..."
<<<<<<< HEAD
mkdir -p $OUTPUT_DIR/comparison

# Collect results directories for all latent dimensions
RESULT_DIRS=""
for latent_dim in $LATENT_DIMS; do
    RESULT_DIRS="$RESULT_DIRS $OUTPUT_DIR/benchmark_${MODEL}_latent${latent_dim}"
done

# Run comparison
python -m src.analysis.compare_results \
    --results $RESULT_DIRS \
    --output $OUTPUT_DIR/comparison
=======
result_dir=$(jq -r '.output_dir' $CONFIG_FILE)
mkdir -p $result_dir/comparison

# Use the experiment_summary.json file to get result directories
if [ -f "$result_dir/experiment_summary.json" ]; then
    echo "Using experiment summary to find result directories"
    # Extract result directories from the experiment summary using jq if available
    if command -v jq &> /dev/null; then
        RESULT_DIRS=$(jq -r '.results[].result_dir' "$result_dir/experiment_summary.json" | tr '\n' ' ')
    else
        # Fallback for when jq is not available
        echo "jq not found, comparison report may be incomplete"
        RESULT_DIRS=""
    fi
else
    echo "Experiment summary not found, comparison report may be incomplete"
    RESULT_DIRS=""
fi

echo "Found result directories: $RESULT_DIRS"

# Run comparison if we have result directories
if [ -n "$RESULT_DIRS" ]; then
    python -m src.analysis.compare_results \
        --results $RESULT_DIRS \
        --output $result_dir/comparison
    echo "Comparison analysis complete!"
else
    echo "No result directories found for comparison analysis!"
fi
>>>>>>> main

echo "Experiment and analysis complete!" 
