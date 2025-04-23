#!/bin/bash
# Change to the script's directory so relative paths work correctly
cd "$(dirname "$0")"
#SBATCH --job-name=kv_compression
#SBATCH --output=kv_compression_%j.log
#SBATCH --error=kv_compression_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

# Usage: sbatch run_experiments.sh <config_file>
# Require a config file argument
if [ $# -lt 1 ]; then
    echo "Error: No config file provided. Usage: sbatch $0 <config_file>"
    exit 1
fi
CONFIG_FILE=$1
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file $CONFIG_FILE not found!"
    exit 1
fi

# Print configuration
echo "==== KV Cache Compression Experiment ===="
echo "Config file: $CONFIG_FILE"
echo "========================================"

# Load necessary modules (adjust for your cluster)
# module load cuda/11.7
# module load python/3.9

# Run the experiment
echo "Starting experiment at $(date)"
python run_experiments.py --config $CONFIG_FILE
echo "Experiment completed at $(date)"

# Run comparison analysis
echo "Generating comparison report..."
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

echo "Experiment and analysis complete!" 
