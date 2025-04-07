#!/bin/bash

# Set up environment
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Load config to get output directory
CONFIG_FILE="src/configs/default_config.json"
OUTPUT_DIR=$(python -c "import json; f=open('$CONFIG_FILE'); data=json.load(f); print(data['output_dir']); f.close()")

# Check if training has been completed
if [ ! -f "$OUTPUT_DIR/autoencoder_final.pth" ]; then
    echo "Error: Autoencoder model not found in $OUTPUT_DIR"
    echo "Please run training first:"
    echo "python src/dictionary_learning/train.py --config $CONFIG_FILE"
    exit 1
fi

# Run evaluation
echo "Starting evaluation..."
python src/evaluation/evaluate.py \
    --config $CONFIG_FILE

# Print completion message
echo "Evaluation complete! Results are saved in $OUTPUT_DIR" 