# KV Cache Compression Benchmark Report

## Overview

- Model: distilgpt2
- Latent dimensions tested: [np.int64(8)]
- Cache sizes tested: [np.float64(1.0), np.float64(10.0), np.float64(100.0), np.float64(1000.0)] MB

## Key Findings

- Best speedup: **0.99x** (Latent dim=8, Cache size=1.0 MB)
- Best compression ratio: **32.00x** (Latent dim=8, Cache size=1.0 MB)

## Results for 1000.0 MB Cache

| Latent Dim | Baseline Time (s) | Baseline Std | Compressed Time (s) | Compressed Std | Speedup | Compression Ratio |
|------------|------------------|--------------|---------------------|----------------|---------|-------------------|
| 8 | 0.0074 | 0.0000 | 0.0076 | 0.0000 | 0.98 | 32.00 |
| 8 | 0.0077 | 0.0002 | 0.0078 | 0.0001 | 0.99 | 16.00 |

## Conclusions

- The optimal latent dimension depends on the size of the KV cache and the importance of speed vs. compression.

## Visualizations

See the generated PNG files for detailed comparisons:
- `time_comparison.png`: Comparison of time to first token
- `compression_ratio.png`: Achieved compression ratios
- `speedup.png`: Speedup factors
- `tradeoff.png`: Compression vs. speedup tradeoff
