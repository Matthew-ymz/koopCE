## metric_heatmap
This matrix compares the strongest six summary metrics without using a chart title. KoopCE leads the primary metric Accuracy at 0.9. Cell colors show metric-wise standardized standing after orienting lower-is-better metrics so that warmer cells indicate stronger overall performance, while annotations preserve original values and per-metric ranks.

## performance_tradeoff
This scatter plot exposes the efficiency frontier between Latency Ms and Accuracy. The Pareto set is Baseline-Y, Baseline-X, Ablation-B, Ablation-A, KoopCE. Bubble area encodes Params M, adding model-size context without a title.

## metric_distribution
This figure focuses on Accuracy and preserves run-to-run spread across methods. KoopCE remains the strongest method on the central tendency of the primary metric. The overlay makes stability visible instead of showing only a single point estimate.

## dataset_heatmap
This heatmap compares Accuracy across datasets or operating regimes. KoopCE wins the largest number of dataset slices (4 out of 4). The panel is useful for spotting robustness gaps that an aggregate mean can hide.

## training_curves
This multi-panel curve summary shows convergence dynamics for up to four logged metrics. For the first panel, Baseline-Y ends with the strongest final Train Loss. Confidence bands summarize variation across runs.
