# Scenario 2: optimal constant by sliding-window timeframe

## Summary

This experiment independently sweeps the tested standard-deviation constants `c` for each requested rolling timeframe. The candidate grid is `[0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 40, 60, 80, 100, 150, 200, 300, 500, 1000, 2000, 5000]`; the selected value is the best candidate in that grid, not a continuous optimization. Constants are selected using pooled validation F1; the held-out test period is not used to select them. These results are for the checked-in **synthetic generator and row-count sliding-window implementation**, not production traffic.

## Protocol

- Signal: sum of `rows_returned` for the same database and database user over the trailing window `(timestamp - duration, timestamp]`, using `common.SlidingRowWindow`.
- Timeframes: 1m, 5m, 15m, 1h, 4h, 1d, 7d.
- Bucket semantics: each timeframe is divided into fixed, non-overlapping buckets aligned to the simulation start. A bucket's signal is the maximum rolling `rows_returned` total observed on reads in that bucket; its ground truth is anomalous if any read in that bucket is labeled anomalous. Buckets with no reads are included with signal zero and normal ground truth.
- Threshold: predict a bucket anomalous when its bucket signal is strictly greater than `training_mean + c * training_stddev`. The mean and population standard deviation are calculated from the training buckets' maximum rolling totals, including empty buckets.
- Constant selection: maximize pooled validation F1 (harmonic mean of precision and recall); ties prefer higher precision, then higher accuracy, with the lower `c` retained for any remaining tie. Accuracy, precision, and recall are reported alongside F1.
- Simulation: 28 UTC days of training, 7 validation days, and 7 held-out test days; legacy generator buckets are 60 minutes. Start time is 2026-01-05 00:00 UTC. Ten fixed seeds (42–51) use `v1/client/query_template_config_s2.textproto`.
- Rolling row totals still advance at each read using `common.SlidingRowWindow`; only the reported prediction and metric sample are aggregated to one per non-overlapping timeframe bucket. Rolling state carries across the train/validation/test boundaries.
- Test summary: arithmetic mean and population standard deviation of each metric across the ten per-seed test runs. Pooled test confusion counts are also included in the CSV.

## Bucket counts per seed

| Timeframe | Training buckets | Validation buckets | Test buckets |
|---:|---:|---:|---:|
| 1m | 40320 | 10080 | 10080 |
| 5m | 8064 | 2016 | 2016 |
| 15m | 2688 | 672 | 672 |
| 1h | 672 | 168 | 168 |
| 4h | 168 | 42 | 42 |
| 1d | 28 | 7 | 7 |
| 7d | 4 | 1 | 1 |

At 7d there is only one validation and one test bucket per seed (10 of each pooled across seeds); treat those weekly metrics as highly uncertain despite the displayed seed-to-seed spread.

## Results

| Timeframe | Selected `c` | Validation F1 | Validation accuracy | Validation precision | Validation recall | Test accuracy (mean ± SD) | Test precision (mean ± SD) | Test recall (mean ± SD) | Test F1 (mean ± SD) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1m | 5.00 | 0.825 | 0.975 | 0.702 | 1.000 | 0.975 ± 0.002 | 0.695 ± 0.031 | 1.000 ± 0.000 | 0.820 ± 0.022 |
| 5m | 2.00 | 0.708 | 0.839 | 0.572 | 0.929 | 0.839 ± 0.008 | 0.571 ± 0.016 | 0.922 ± 0.017 | 0.705 ± 0.015 |
| 15m | 0.50 | 0.745 | 0.695 | 0.633 | 0.906 | 0.697 ± 0.016 | 0.633 ± 0.021 | 0.901 ± 0.017 | 0.744 ± 0.019 |
| 1h | 0.10 | 0.921 | 0.856 | 0.941 | 0.901 | 0.864 ± 0.031 | 0.941 ± 0.022 | 0.908 ± 0.019 | 0.924 ± 0.019 |
| 4h | 0.10 | 0.987 | 0.974 | 1.000 | 0.974 | 0.974 ± 0.017 | 1.000 ± 0.000 | 0.974 ± 0.017 | 0.987 ± 0.009 |
| 1d | 0.10 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| 7d | 0.10 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |

Validation metrics by candidate constant (all seven timeframes):

![Validation metrics by timeframe and constant](validation_metrics.svg)

Held-out test metrics at the validation-selected constant:

![Test metrics by timeframe](test_metrics.svg)

Full validation sweep is in [`constant_sweep.csv`](constant_sweep.csv). Test metric columns are populated only for the validation-selected constant of each timeframe, to keep the held-out set out of constant selection.

## Interpretation and limitations

This branch's implemented `SlidingRowWindow` sums `rows_returned`, so this experiment tests **rolling returned-row volume**, not query count. Evaluation is per fixed timeframe bucket: the maximum rolling total predicts the bucket, and any generator-labeled anomalous read makes its bucket ground-truth anomalous. These bucket aggregation rules are explicit experiment choices because the progress doc did not define how event labels and read-level scores map to time buckets. The generator emits synthetic, deliberately injected anomalies; selected constants are specific to this configuration, aggregation rules, split, candidate grid, and seed set. Results should not be interpreted as production thresholds or compared numerically with the intern's results until their data, split, aggregation, and optimization objective are aligned. The next phase can combine timeframe signals after these single-frame constants are reviewed.
