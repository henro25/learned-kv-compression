#!/bin/bash

# Set up environment
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run evaluation
python src/evaluation/evaluate.py \
    --config src/configs/default_config.json

# Print completion message
echo "Evaluation complete! Results are saved in the output directory." 