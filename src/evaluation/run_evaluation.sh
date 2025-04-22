#!/bin/bash

# Set up environment
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Load config to get output directory and model parameters
CONFIG_FILE="src/configs/default_config.json"
OUTPUT_DIR=$(python -c "import json; f=open('$CONFIG_FILE'); data=json.load(f); print(data['output_dir']); f.close()")
MODEL_NAME=$(python -c "import json; f=open('$CONFIG_FILE'); data=json.load(f); print(data['name']); f.close()")
LATENT_DIM=$(python -c "import json; f=open('$CONFIG_FILE'); data=json.load(f); print(data['latent_dim']); f.close()")

# Construct model directory path
MODEL_DIR="$OUTPUT_DIR/${MODEL_NAME}_${LATENT_DIM}"

# Check if training has been completed
if [ ! -f "$MODEL_DIR/autoencoder_finals.pth" ]; then
    echo "Error: Autoencoder model not found in $MODEL_DIR"
    echo "Please run training first:"
    echo "python src/dictionary_learning/train.py --config $CONFIG_FILE"
    exit 1
fi

# Run evaluation
echo "Starting evaluation..."
python src/evaluation/evaluate.py \
    --config $CONFIG_FILE

# Print completion message
echo "Evaluation complete! Results are saved in $MODEL_DIR" 