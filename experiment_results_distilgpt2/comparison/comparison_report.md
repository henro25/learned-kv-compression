# KV Cache Compression Benchmark Report

## Overview

- Model: distilgpt2
- Latent dimensions tested: [np.int64(8), np.int64(16), np.int64(32)]
- Cache sizes tested: [np.float64(1.0), np.float64(10.0), np.float64(100.0), np.float64(1000.0), np.float64(3000.0)] MB

## Key Findings

- Best speedup: **1.33x** (Latent dim=8, Cache size=100.0 MB)
- Best compression ratio: **8.00x** (Latent dim=8, Cache size=1.0 MB)

## Results for 3000.0 MB Cache

| Latent Dim | Baseline Time (s) | Baseline Std | Compressed Time (s) | Compressed Std | Speedup | Compression Ratio |
|------------|------------------|--------------|---------------------|----------------|---------|-------------------|
| 8 | 0.0138 | 0.0003 | 0.0120 | 0.0002 | 1.16 | 8.00 |
| 16 | 0.0142 | 0.0006 | 0.0141 | 0.0003 | 1.01 | 4.00 |
| 32 | 0.0154 | 0.0017 | 0.0163 | 0.0016 | 0.95 | 2.00 |

## Conclusions

- The optimal latent dimension depends on the size of the KV cache and the importance of speed vs. compression.
- **Recommended configuration:** Latent dim=8 provides good balance with 8.00x compression and 1.33x speedup.

## Visualizations

See the generated PNG files for detailed comparisons:
- `time_comparison.png`: Comparison of time to first token
- `compression_ratio.png`: Achieved compression ratios
- `speedup.png`: Speedup factors
- `tradeoff.png`: Compression vs. speedup tradeoff
