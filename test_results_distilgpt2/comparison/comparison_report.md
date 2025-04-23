# KV Cache Compression Benchmark Report

## Overview

- Model: distilgpt2
- Latent dimensions tested: [np.int64(8)]
- Cache sizes tested: [None] MB

## Key Findings

- Best speedup: **0.02x** (Latent dim=8, Cache size=None MB)
## Results for nan MB Cache

| Latent Dim | Baseline Time (s) | Baseline Std | Compressed Time (s) | Compressed Std | Speedup | Compression Ratio |
|------------|------------------|--------------|---------------------|----------------|---------|-------------------|

## Conclusions

- The optimal latent dimension depends on the size of the KV cache and the importance of speed vs. compression.

## Visualizations

See the generated PNG files for detailed comparisons:
- `time_comparison.png`: Comparison of time to first token
- `compression_ratio.png`: Achieved compression ratios
- `speedup.png`: Speedup factors
- `tradeoff.png`: Compression vs. speedup tradeoff
