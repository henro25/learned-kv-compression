#!/bin/bash
#SBATCH --job-name=kv_compression
#SBATCH --output=kv_compression_%j.log
#SBATCH --error=kv_compression_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
sbatch run_experiment.sh distilgpt2 "1 2 3" "10 100 150" "1 2 3" "100 1000 1500" "64 65 66" "1 2 3"
