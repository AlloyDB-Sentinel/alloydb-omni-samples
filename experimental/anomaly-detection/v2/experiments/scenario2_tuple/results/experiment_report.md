# Scenario 2: OR- and AND-combined timeframe subsets

## Summary

This experiment fixes clean training history at **20 weeks** and evaluates all **127 non-empty subsets** of the seven rolling timeframes under both **OR** and **AND** combination (254 candidates). Constants are selected independently per timeframe by pooled validation F1 across the 10 seeds; each subset/combiner pair is then scored at those constants, and the best candidate is selected by pooled validation F1. Test metrics are reported at the validation-selected settings and are not used for selection.

These are synthetic results from the checked-in generator and should not be read as production thresholds.

## Protocol

- Signal: sum of `rows_returned` for the same database and database user over each frame's trailing window, using `common.SlidingRowWindow`: 1m, 5m, 15m, 1h, 4h, 1d, 7d.
- Reporting grid: fixed, non-overlapping **4-hour** buckets aligned to the simulation start. Each frame contributes the **maximum** rolling total observed among reads in the bucket; buckets with no reads contribute zero.
- Ground truth: a 4-hour bucket is anomalous if any observed trailing 4-hour window contains an attack read, including spillover after an episode.
- Training: 1 warm-up week, then 20 clean training weeks. Baselines learn one `mean + c × population stddev` threshold per weekly slot (42 slots), using the 20 training observations for that slot.
- Split: 4 validation weeks, then 4 held-out test weeks. 10 deterministic seeds (42–51) use `v2/experiments/scenario2_timeframe/config.textproto`.
- Timezone: all timestamps and day/night peak windows are evaluated in UTC. The generator uses UTC explicitly because Windows ignores `TZ=UTC`.
- Candidate constants: `[0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 40, 60, 80, 100, 150, 200, 300, 500, 1000, 2000, 5000]`. This is the earlier timeframe sweep's grid, which includes sub-1 values needed by long windows; the training-duration experiment's integer 1–8 grid was specific to the fixed 4-hour frame.
- Selection: maximize pooled validation F1 across all subset/combiner candidates; ties prefer higher precision, then higher accuracy, then the earlier candidate (smaller subset, lexicographic member order, OR before AND).
- Combination: **OR** flags a bucket when at least one member exceeds its threshold; **AND** flags only when every member does. Singletons are identical under both combiners.

Each seed has 840 training buckets, 168 validation buckets, and 168 test buckets. The full 254-row candidate table is in `subset_results.csv`.

## Individual timeframe constants at 20 weeks

| Timeframe | Selected `c` | Validation accuracy | Validation precision | Validation recall | Validation F1 | Test accuracy | Test precision | Test recall | Test F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1m | 3 | 91.73% | 86.27% | 52.80% | 65.51% | 91.31% | 86.39% | 50.20% | 63.50% |
| 5m | 4 | 93.63% | 96.73% | 59.20% | 73.45% | 93.10% | 95.36% | 56.92% | 71.29% |
| 15m | 5 | 94.05% | 100.00% | 60.00% | 75.00% | 94.40% | 100.00% | 62.85% | 77.18% |
| 1h | 6 | 94.23% | 100.00% | 61.20% | 75.93% | 94.46% | 100.00% | 63.24% | 77.48% |
| 4h | 3 | 98.81% | 97.13% | 94.80% | 95.95% | 98.93% | 98.35% | 94.47% | 96.37% |
| 1d | 2 | 73.69% | 34.76% | 87.60% | 49.77% | 74.70% | 35.57% | 83.79% | 49.94% |
| 7d | 2.5 | 49.94% | 19.06% | 72.80% | 30.21% | 40.24% | 17.49% | 79.84% | 28.69% |

## Best subset by size

For each subset size, the best OR and AND candidate by pooled validation F1.

| Size | OR subset | OR validation F1 | OR test F1 | AND subset | AND validation F1 | AND test F1 |
|---:|---|---:|---:|---|---:|---:|
| 1 | 4h | 95.95% | 96.37% | 4h | 95.95% | 96.37% |
| 2 | 15m+4h | 96.99% | 97.41% | 4h+1d | 90.43% | 88.40% |
| 3 | 15m+1h+4h | 96.99% | 97.61% | 4h+1d+7d | 76.47% | 80.00% |
| 4 | 5m+15m+1h+4h | 96.03% | 96.27% | 5m+15m+1h+4h | 72.12% | 70.92% |
| 5 | 1m+5m+15m+1h+4h | 92.93% | 92.63% | 1m+5m+15m+1h+4h | 65.59% | 65.07% |
| 6 | 1m+5m+15m+1h+4h+1d | 53.48% | 55.07% | 1m+5m+15m+1h+4h+1d | 59.55% | 57.06% |
| 7 | 1m+5m+15m+1h+4h+1d+7d | 36.26% | 32.77% | 1m+5m+15m+1h+4h+1d+7d | 49.40% | 51.61% |

## Best OR and best AND

- **Best OR:** 15m+4h at `c = 5` for 15m, `c = 3` for 4h — validation F1 96.99%, test F1 97.41%.
- **Best AND:** 4h at `c = 3` for 4h — validation F1 95.95%, test F1 96.37%.
- **Best multi-member AND:** 4h+1d at `c = 3` for 4h, `c = 2` for 1d — validation F1 90.43%, test F1 88.40%.

## Comparison with the best single frame

The best single frame is **4h** at `c = 3` for 4h (validation F1 95.95%, test F1 96.37%). 7 OR candidates beat it on validation F1 and 3 on test F1; 0 AND candidates beat it on validation F1 and 0 on test F1.

## Top candidates

The top 20 candidates overall, sorted by pooled validation F1. The selected candidate is marked.

| Rank | Combination | Subset | Size | Validation accuracy | Validation precision | Validation recall | Validation F1 | Test accuracy | Test precision | Test recall | Test F1 |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | OR | **15m+4h** | 2 | 99.11% | 97.19% | 96.80% | 96.99% | 99.23% | 98.39% | 96.44% | 97.41% |
| 2 | OR | 1h+4h | 2 | 99.11% | 97.19% | 96.80% | 96.99% | 99.29% | 98.39% | 96.84% | 97.61% |
| 3 | OR | 15m+1h+4h | 3 | 99.11% | 97.19% | 96.80% | 96.99% | 99.29% | 98.39% | 96.84% | 97.61% |
| 4 | OR | 5m+4h | 2 | 98.81% | 95.28% | 96.80% | 96.03% | 98.81% | 95.69% | 96.44% | 96.06% |
| 5 | OR | 5m+15m+4h | 3 | 98.81% | 95.28% | 96.80% | 96.03% | 98.81% | 95.69% | 96.44% | 96.06% |
| 6 | OR | 5m+1h+4h | 3 | 98.81% | 95.28% | 96.80% | 96.03% | 98.87% | 95.70% | 96.84% | 96.27% |
| 7 | OR | 5m+15m+1h+4h | 4 | 98.81% | 95.28% | 96.80% | 96.03% | 98.87% | 95.70% | 96.84% | 96.27% |
| 8 | OR | 4h | 1 | 98.81% | 97.13% | 94.80% | 95.95% | 98.93% | 98.35% | 94.47% | 96.37% |
| 9 | AND | 4h | 1 | 98.81% | 97.13% | 94.80% | 95.95% | 98.93% | 98.35% | 94.47% | 96.37% |
| 10 | OR | 1m+4h | 2 | 97.98% | 90.00% | 97.20% | 93.46% | 97.98% | 91.01% | 96.05% | 93.46% |
| 11 | OR | 1m+15m+4h | 3 | 97.98% | 90.00% | 97.20% | 93.46% | 98.04% | 91.04% | 96.44% | 93.67% |
| 12 | OR | 1m+1h+4h | 3 | 97.98% | 90.00% | 97.20% | 93.46% | 98.10% | 91.08% | 96.84% | 93.87% |
| 13 | OR | 1m+15m+1h+4h | 4 | 97.98% | 90.00% | 97.20% | 93.46% | 98.10% | 91.08% | 96.84% | 93.87% |
| 14 | OR | 1m+5m+4h | 3 | 97.80% | 89.01% | 97.20% | 92.93% | 97.62% | 88.73% | 96.44% | 92.42% |
| 15 | OR | 1m+5m+15m+4h | 4 | 97.80% | 89.01% | 97.20% | 92.93% | 97.62% | 88.73% | 96.44% | 92.42% |
| 16 | OR | 1m+5m+1h+4h | 4 | 97.80% | 89.01% | 97.20% | 92.93% | 97.68% | 88.77% | 96.84% | 92.63% |
| 17 | OR | 1m+5m+15m+1h+4h | 5 | 97.80% | 89.01% | 97.20% | 92.93% | 97.68% | 88.77% | 96.84% | 92.63% |
| 18 | AND | 4h+1d | 2 | 97.38% | 99.05% | 83.20% | 90.43% | 96.85% | 99.02% | 79.84% | 88.40% |
| 19 | AND | 4h+7d | 2 | 94.94% | 96.09% | 68.80% | 80.19% | 96.25% | 98.47% | 76.28% | 85.97% |
| 20 | AND | 4h+1d+7d | 3 | 94.29% | 98.73% | 62.40% | 76.47% | 94.94% | 98.84% | 67.19% | 80.00% |

## Selected candidate

**15m+4h (OR)** with `c = 5` for 15m, `c = 3` for 4h.

- Validation: accuracy 99.11%, precision 97.19%, recall 96.80%, F1 96.99%
- Held-out test: accuracy 99.23%, precision 98.39%, recall 96.44%, F1 97.41%

If another candidate ties the selected candidate on pooled validation F1, the earlier candidate in size-then-lexicographic order is retained; the test column is not used for that decision.

## Interpretation and limitations

OR admits a bucket when any member fires, so it raises recall; AND requires every member, so it raises precision but can suppress recall. In this run, 7 OR candidates beat the best single frame on validation F1 and 3 on test, while 0 AND candidates beat it on validation and 0 on test. The best multi-member AND is 4h+1d: it reaches 99.05% validation precision and 83.20% recall, so it is only attractive if a false positive is much more costly than a miss. Beyond the winning pair, 1h+4h and 15m+1h+4h tie 15m+4h on validation F1, so 15m+4h is retained by the size-then-lexicographic tie-break even though both have the higher held-out test F1 (97.61% vs 97.41%). Adding 1m or 5m to 15m+4h lowers validation F1. The best 4- and 5-member OR subsets are below 15m+4h, and including 1d or 7d in OR collapses precision: all seven under OR reaches 100% validation recall at 22.14% precision.

Because every member uses the constant selected for that frame alone, neither combiner is jointly optimized. Long frames (1d and 7d) produce rolling totals that change slowly, so their 4-hour bucket signals are highly autocorrelated and their labels can remain positive for a long tail after an episode; their apparent performance should be read with that in mind. Constants and the candidate are both selected on the same four validation weeks, which makes the validation figures optimistic; the four test weeks are held out and are the better guide. The generator emits synthetic, deliberately injected anomalies with one database/user identity, and this runner does not exercise the RPC services. The next step would be to validate the selected candidate on a second independent period and to test a joint constant sweep for the winning subset.
