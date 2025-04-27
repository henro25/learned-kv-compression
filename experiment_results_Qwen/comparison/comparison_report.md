# KV Cache Compression Benchmark Report

## Overview

- Model: Qwen/Qwen2.5-0.5B
- Latent dimensions tested: [np.int64(32)]
- Cache sizes tested: [np.float64(1.0), np.float64(10.0), np.float64(100.0), np.float64(1000.0)] MB

## Key Findings

- Best speedup: **1.04x** (Latent dim=32, Cache size=1000.0 MB)
- Best compression ratio: **16.00x** (Latent dim=32, Cache size=1.0 MB)

## Results for 1000.0 MB Cache

| Latent Dim | Baseline Time (s) | Baseline Std | Compressed Time (s) | Compressed Std | Speedup | Compression Ratio |
|------------|------------------|--------------|---------------------|----------------|---------|-------------------|
| 32 | 0.0402 | 0.0001 | 0.0428 | 0.0003 | 0.94 | 16.00 |
| 32 | 0.0406 | 0.0004 | 0.0445 | 0.0040 | 0.91 | 8.00 |
| 32 | 0.0430 | 0.0041 | 0.0415 | 0.0004 | 1.04 | 4.00 |
| 32 | 0.0391 | 0.0002 | 0.0416 | 0.0010 | 0.94 | 2.00 |

## Conclusions

- The optimal latent dimension depends on the size of the KV cache and the importance of speed vs. compression.
- **Recommended configuration:** Latent dim=32 provides good balance with 4.00x compression and 1.04x speedup.

## Visualizations

See the generated PNG files for detailed comparisons:
- `time_comparison.png`: Comparison of time to first token
- `compression_ratio.png`: Achieved compression ratios
- `speedup.png`: Speedup factors
- `tradeoff.png`: Compression vs. speedup tradeoff
