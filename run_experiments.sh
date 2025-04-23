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

# Print configuration and set up directories
SCRIPT_DIR=$(pwd)
echo "==== KV Cache Compression Experiment ===="
echo "Config file: $CONFIG_FILE"
echo "Working directory: $SCRIPT_DIR"
echo "========================================"

# Determine the experiment result directory from the config, make it absolute
REL_OUTPUT_DIR=$(jq -r '.output_dir' "$CONFIG_FILE")
if [[ "$REL_OUTPUT_DIR" == /* ]]; then
    RESULT_DIR="$REL_OUTPUT_DIR"
else
    RESULT_DIR="$SCRIPT_DIR/$REL_OUTPUT_DIR"
fi
echo "Experiment results directory: $RESULT_DIR"
# Create comparison output directory
mkdir -p "$RESULT_DIR/comparison"

# Load necessary modules (adjust for your cluster)
# module load cuda/11.7
# module load python/3.9

# Run the experiment
echo "Starting experiment at $(date)"
python run_experiments.py --config $CONFIG_FILE
echo "Experiment completed at $(date)"

# Run comparison analysis
echo "Generating comparison report..."
# Use the resolved RESULT_DIR
echo "Looking for summary file at: $RESULT_DIR/experiment_summary.json"

if [ -f "$RESULT_DIR/experiment_summary.json" ]; then
    echo "Using experiment summary to find result directories"
    if command -v jq &> /dev/null; then
        # Read result directories into an array to handle spaces in paths
        mapfile -t RESULT_DIRS_ARRAY < <(jq -r '.results[].result_dir' "$RESULT_DIR/experiment_summary.json")
    else
        echo "jq not found, comparison report may be incomplete"
        RESULT_DIRS_ARRAY=()
    fi
else
    echo "Experiment summary not found at $RESULT_DIR/experiment_summary.json, comparison report may be incomplete"
    echo "Directory contents:" 
    ls -l "$RESULT_DIR" | cat
    RESULT_DIRS_ARRAY=()
fi

# Show extracted result directories (benchmarks)
echo "Found result directories: ${RESULT_DIRS_ARRAY[*]}"

if [ "${#RESULT_DIRS_ARRAY[@]}" -gt 0 ]; then
    python3 -m src.analysis.compare_results \
        --results "${RESULT_DIRS_ARRAY[@]}" \
        --output "$RESULT_DIR/comparison"
    echo "Comparison analysis complete!"
else
    echo "No result directories found for comparison analysis!"
fi

echo "Experiment and analysis complete!" 
