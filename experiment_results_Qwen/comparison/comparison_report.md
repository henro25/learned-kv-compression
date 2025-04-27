# KV Cache Compression Benchmark Report

## Overview

- Model: Qwen/Qwen2.5-0.5B
- Latent dimensions tested: [np.int64(16)]
- Cache sizes tested: [np.float64(1.0), np.float64(10.0), np.float64(100.0), np.float64(1000.0)] MB

## Key Findings

- Best speedup: **1.02x** (Latent dim=16, Cache size=1000.0 MB)
- Best compression ratio: **32.00x** (Latent dim=16, Cache size=1.0 MB)

## Results for 1000.0 MB Cache

| Latent Dim | Baseline Time (s) | Baseline Std | Compressed Time (s) | Compressed Std | Speedup | Compression Ratio |
|------------|------------------|--------------|---------------------|----------------|---------|-------------------|
| 16 | 0.0415 | 0.0002 | 0.0439 | 0.0002 | 0.94 | 32.00 |
| 16 | 0.0413 | 0.0006 | 0.0444 | 0.0006 | 0.93 | 16.00 |
| 16 | 0.0449 | 0.0052 | 0.0438 | 0.0008 | 1.02 | 8.00 |
| 16 | 0.0416 | 0.0009 | 0.0448 | 0.0009 | 0.93 | 4.00 |

## Conclusions

- The optimal latent dimension depends on the size of the KV cache and the importance of speed vs. compression.
- **Recommended configuration:** Latent dim=16 provides good balance with 8.00x compression and 1.02x speedup.

## Visualizations

See the generated PNG files for detailed comparisons:
- `time_comparison.png`: Comparison of time to first token
- `compression_ratio.png`: Achieved compression ratios
- `speedup.png`: Speedup factors
- `tradeoff.png`: Compression vs. speedup tradeoff
