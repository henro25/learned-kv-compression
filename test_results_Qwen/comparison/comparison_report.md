# KV Cache Compression Benchmark Report

## Overview

- Model: Qwen/Qwen2.5-0.5B
- Latent dimensions tested: [np.int64(8)]
- Cache sizes tested: [np.float64(1.0), np.float64(10.0), np.float64(100.0), np.float64(1000.0)] MB

## Key Findings

- Best speedup: **0.99x** (Latent dim=8, Cache size=1000.0 MB)
- Best compression ratio: **8.00x** (Latent dim=8, Cache size=1.0 MB)

## Results for 1000.0 MB Cache

| Latent Dim | Baseline Time (s) | Baseline Std | Compressed Time (s) | Compressed Std | Speedup | Compression Ratio |
|------------|------------------|--------------|---------------------|----------------|---------|-------------------|
| 8 | 0.0417 | 0.0038 | 0.0422 | 0.0003 | 0.99 | 8.00 |
| 8 | 0.0399 | 0.0005 | 0.0430 | 0.0004 | 0.93 | 8.00 |
| 8 | 0.0392 | 0.0004 | 0.0422 | 0.0004 | 0.93 | 8.00 |
| 8 | 0.0400 | 0.0004 | 0.0427 | 0.0015 | 0.94 | 8.00 |

## Conclusions

- The optimal latent dimension depends on the size of the KV cache and the importance of speed vs. compression.

## Visualizations

See the generated PNG files for detailed comparisons:
- `time_comparison.png`: Comparison of time to first token
- `compression_ratio.png`: Achieved compression ratios
- `speedup.png`: Speedup factors
- `tradeoff.png`: Compression vs. speedup tradeoff
