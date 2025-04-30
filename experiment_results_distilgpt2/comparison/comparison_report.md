# KV Cache Compression Benchmark Report

## Overview

- Model: distilgpt2
- Latent dimensions tested: [np.int64(16), np.int64(32)]
- Cache sizes tested: [np.float64(1.0), np.float64(10.0), np.float64(100.0), np.float64(1000.0)] MB

## Key Findings

- Best speedup: **1.04x** (Latent dim=16, Cache size=1000.0 MB)
- Best compression ratio: **32.00x** (Latent dim=16, Cache size=1.0 MB)

## Results for 1000.0 MB Cache

| Latent Dim | Baseline Time (s) | Baseline Std | Compressed Time (s) | Compressed Std | Speedup | Compression Ratio |
|------------|------------------|--------------|---------------------|----------------|---------|-------------------|
| 16 | 0.0075 | 0.0001 | 0.0076 | 0.0000 | 0.99 | 32.00 |
| 16 | 0.0074 | 0.0001 | 0.0076 | 0.0000 | 0.98 | 16.00 |
| 16 | 0.0075 | 0.0001 | 0.0077 | 0.0001 | 0.98 | 8.00 |
| 16 | 0.0081 | 0.0006 | 0.0078 | 0.0000 | 1.04 | 4.00 |
| 32 | 0.0074 | 0.0000 | 0.0076 | 0.0000 | 0.97 | 16.00 |
| 32 | 0.0073 | 0.0000 | 0.0075 | 0.0001 | 0.97 | 8.00 |
| 32 | 0.0074 | 0.0002 | 0.0076 | 0.0002 | 0.98 | 4.00 |
| 32 | 0.0074 | 0.0001 | 0.0078 | 0.0001 | 0.96 | 2.00 |

## Conclusions

- The optimal latent dimension depends on the size of the KV cache and the importance of speed vs. compression.
- **Recommended configuration:** Latent dim=16 provides good balance with 4.00x compression and 1.04x speedup.

## Visualizations

See the generated PNG files for detailed comparisons:
- `time_comparison.png`: Comparison of time to first token
- `compression_ratio.png`: Achieved compression ratios
- `speedup.png`: Speedup factors
- `tradeoff.png`: Compression vs. speedup tradeoff
