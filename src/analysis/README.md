# KV Cache Compression Analysis Tools

This directory contains scripts for analyzing and visualizing the results of KV cache compression experiments.

## Available Scripts

### compare_results.py

This script loads benchmark results from multiple experiment directories and generates comparative visualizations:

```bash
python -m src.analysis.compare_results \
    --results results/distilgpt2_8 results/distilgpt2_16 results/distilgpt2_32 \
    --output comparison_results
```

The script will:
1. Generate comparative plots of time to first token across different latent dimensions
2. Plot compression ratios for different cache sizes
3. Create speedup graphs to show where compression helps performance
4. Analyze the tradeoff between compression ratio and speedup
5. Generate a comprehensive Markdown report summarizing the findings

## Output

The script produces several visualization files:
- `time_comparison.png`: Line plot comparing time to first token
- `bar_comparison_largest.png`: Bar chart comparison for largest cache size
- `compression_ratio.png`: Line plot of compression ratios
- `bar_compression_ratio.png`: Bar chart of compression ratios
- `speedup.png`: Line plot of speedup factors
- `tradeoff.png`: Scatter plot of compression ratio vs. speedup
- `speedup_heatmap.png`: Heatmap of speedup by latent dim and cache size
- `comparison_report.md`: Markdown report with key findings and recommendations 