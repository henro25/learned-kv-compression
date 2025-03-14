# learned-kv-compression
Asymmetric Autoencoders for Learned KV Cache Compression

## Overview

- **KV Cache Extraction:**  
  A custom `Buffer` class extracts key and value vectors from a transformer model's caching mechanism. These vectors are then used to train an autoencoder.

- **Autoencoder Training:**  
  The autoencoder is optimized using a mean squared error (MSE) reconstruction loss to compress the KV vectors (for now). Training progress is logged via TensorBoard, and model checkpoints are saved at the end of each epoch.

- **Dataset:**  
  The project uses the WikiText-103 dataset to provide input texts for KV cache extraction.

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/henro25/learned-kv-compression.git
    cd learned-kv-compression
    ```

2. **Create a virtual environment and install dependencies:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use: venv\Scripts\activate
    pip install -r requirements.txt
    ```

## Usage

### Training the Autoencoder

Run the training script using the default configuration:

```bash
python -m src.dictionary_learning.train
```

### Command-Line Overrides

The training script accepts command-line arguments to override the default configuration parameters. For example:

```bash
python -m src.dictionary_learning.train --batch_size 8 --num_epochs 10 --lr 0.001
```

### Configuration File

The default configuration is loaded from `src/configs/default_config.json`. You can modify this file to change default hyperparameters. Command-line arguments (when provided) override these settings.

## Project Details

- **Buffer Module:** The Buffer class in `src/utils/buffer.py` extracts KV pairs from the transformer model. It processes batches of texts, tokenizes them, and extracts key-value vectors from each transformer layer using caching.
- **Autoencoder Model:** The autoencoder is defined in `src/models/autoencoder.py` and takes as input a KV vector (whose dimension equals the head dimension of the transformer). The model is trained to reconstruct these KV vectors with minimal loss.
- **Training Script:** The main training logic is in `src/dictionary_learning/train.py`. It loads the dataset, initializes the transformer model, tokenizer, buffer, and autoencoder, and then trains the autoencoder using a cosine annealing learning rate scheduler. Training progress is logged using TensorBoard.

## Logging and Checkpoints

- **TensorBoard:** Training progress and loss metrics are logged to the `runs/` directory. Launch TensorBoard with:

    ```bash
    tensorboard --logdir runs
    ```

- **Model Checkpoints:** Model checkpoints are saved at the end of each epoch as `autoencoder_epoch_{epoch}.pth`, and the final model is saved as `autoencoder.pth`.
