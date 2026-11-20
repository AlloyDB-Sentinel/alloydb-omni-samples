# Scenario 2 timeframe subsets (OR and AND)

[ALDB-16](https://linear.app/aubree-lianto/issue/ALDB-16/scenario-2-experiment-specified-timeframe-optimal-constants)

The final 20-week results and their reproduction command are in
[Controlled duration comparison](#controlled-duration-comparison).

From `experimental/anomaly-detection`, with Go 1.23 or later:

```sh
go test ./v2/experiments/scenario2_tuple ./v2/common ./v1/generator/...
```

The runner builds the existing generator, defaults to **20 weeks** of clean
training (`-training-weeks`), and evaluates every non-empty subset of the seven rolling timeframes under both
**OR** and **AND**: 127 subsets × 2 combiners = **254 candidates**. A 4-hour
reporting bucket is flagged by OR when any member is above its own
`mean + c × population stddev` threshold, and by AND only when every member is.
Each rolling window is measured continuously, but its threshold now comes from
the matching **weekday and hour** (168 baseline slots). An hourly exceedance
flags that frame in its enclosing 4-hour reporting bucket. AND members may
exceed their thresholds at different times within that reporting bucket.

The standalone CLI (`go run ./v2/experiments/scenario2_tuple`) writes, under `results/`:

- `frame_results.csv`: the selected constant and validation/test confusion counts for each timeframe.
- `frame_sweep.csv`: every timeframe/constant combination, for audit.
- `subset_results.csv`: all 254 candidates, sorted by pooled validation F1. The
  `constants` column is pipe-separated in subset order, e.g. `6|3` for `1h+4h`.
- `experiment_report.md`: protocol, best candidate per size, top 20, and the selected candidate.

Raw generated logs are temporary and deleted after the run. No service or
database is required.

## Experiment

- One warm-up week, 20 clean training weeks by default, 4 validation weeks, then 4 held-out test weeks.
- Ten deterministic seeds (42–51) using `v2/experiments/scenario2_timeframe/config.textproto`.
- One database/user identity; 2% one-hour attack episodes with 3× reads and normal rows per read.
- Baselines learn one threshold per weekday/hour (168 slots), using each training week's maximum rolling total within that hour. Reporting and trailing-4-hour ground-truth labels remain unchanged.
- Constants are selected per timeframe by pooled validation F1 across the ten seeds.
  Every subset uses those fixed per-frame constants; the subset/combiner candidate is
  selected by pooled validation F1. Ties prefer precision, then accuracy, then the
  smaller subset, lexicographic member order, and OR before AND.
- The candidate grid is the earlier timeframe sweep's `[0.1 … 5000]` grid because
  long windows need sub-1 constants.
- All timestamps and day/night peak windows are evaluated in UTC. The generator
  uses UTC explicitly because Windows ignores `TZ=UTC`.
- Traffic rate comes from the config. The checked-in `config.textproto` is the
  default; the historical higher-load run below differs in exactly two lines
  (`queries_per_bucket_mean: 1824.0`, `queries_variation_std_dev: 304.0`).

## Controlled duration comparison

Our final choice is **20 clean training weeks**, the weekly-pattern config at
**50,000 normal reads/day averaged over a week**, hourly weekday baselines, and
**15m+4h (OR)** with constants **6|4**. Metrics count 4-hour reporting buckets,
including the existing attack spillover labels.

| Training weeks | c (15m / 4h) | Validation F1 | Test precision | Test recall | Test F1 | Test FP | Test FN |
|---:|---|---:|---:|---:|---:|---:|---:|
| **20** | **6 / 4** | **96.42%** | **97.75%** | **94.22%** | **95.96%** | **6** | **16** |
| 24 | 5 / 4 | 96.83% | 97.73% | 93.14% | 95.38% | 6 | 19 |

24 weeks wins on validation; 20 detects three more attack-labeled test buckets
with the same false-positive count. We keep 20 as the project choice, not as a
claim that it is the best duration in general. At 20 weeks, 1h+4h and
15m+1h+4h tie the selected pair; the existing tie-break retains 15m+4h.

[duration_experiment_test.go](duration_experiment_test.go) reuses the existing
generator, aggregation, threshold fitting, selection, and CSV writers. It is
skipped during ordinary tests. To reproduce the full duration sweep from the
module root, run:

```sh
SCENARIO2_DURATION_OUTPUT="$(mktemp -d)" \
  go test ./v2/experiments/scenario2_tuple \
    -run '^TestControlledDurationSweep$' -count=1 -timeout=40m -v
```

`SCENARIO2_DURATION_OUTPUT` must be an absolute scratch-directory path. The
run prints that path and writes `weeks-NN/{frame_results,frame_sweep,subset_results}.csv`
for 1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 30, 40, and 50 weeks. It checks all
three CSVs at 20 and 24 weeks against the checked-in references in
[`results/controlled-duration/`](results/controlled-duration/). No previous
local run, external script, or temporary config is needed. Allow several
minutes and several GB of temporary disk space; raw logs are removed after
each seed.

The comparison holds evaluation data fixed:

- Config: `v1/client/query_template_config_s2.textproto`; seeds 42–51.
- Generate the original January 5–August 24, 2026 stream with clean training
  ending June 29. Prepend 26 clean weeks, July 7, 2025–January 5, 2026, using
  seeds 3042–3051. This preserves the original stream's random draws.
- Each duration uses the latest N clean weeks ending June 29. The first week
  of the combined history warms the longest rolling window.
- Validation: June 29–July 27, 2026; test: July 27–August 24, 2026. End dates
  are exclusive; all times are UTC. Each split has 1,680 reporting buckets
  across ten seeds.
- Constants are selected per frame on pooled validation F1; tuples use those
  constants and the existing validation ranking. Test scores do not select
  constants or tuples.

These are reused synthetic development data. A fresh holdout is still needed
for an independent estimate of performance.

## Hourly baseline comparison

With the weekly-pattern 50k/day config, 24 clean training weeks, and seeds
42–51, the same **1m+4h (OR)** pair improves as follows on the same 1,680 test
reporting buckets. The generator, windows, reporting grid, and labels are
unchanged. Constants are selected on validation for each baseline: `4|3`
before, `4|4` after.

| Baseline slots | Test precision | Test recall | Test F1 | False positives | False negatives |
|---|---:|---:|---:|---:|---:|
| 4 hours | 95.20% | 78.70% | 86.17% | 11 | 59 |
| 1 hour | 95.22% | 93.50% | 94.35% | 13 | 18 |

A separate per-record audit found **143/144 attack episodes detected while
active**, versus 115/144 before. Of the 18 missed reporting buckets, 17 contain
only spillover from an earlier attack and one contains active attack reads.

The full hourly-baseline sweep selects **15m+4h (OR)** on validation, with
constants `5|4`: test precision 97.73%, recall 93.14%, and F1 95.38%. The table
keeps the original pair fixed to make the baseline comparison clear.

```sh
go run ./v2/experiments/scenario2_tuple \
  -config v1/client/query_template_config_s2.textproto \
  -training-weeks 24 -output /tmp/scenario2-hourly-results
```

These are synthetic development comparisons. The same test cases informed
the hourly-baseline change; a fresh holdout is needed for a final performance
estimate. Ground truth still includes four-hour spillover after attacks.

## Historical results

The numerical results below and files directly in `results/` are historical runs
with **4-hour baseline slots**. They have not been regenerated for the hourly
baseline. Run the current experiment into a separate output directory; pass
`-config v1/client/query_template_config_s2.textproto` to use the weekly-pattern
50k/day configuration rather than the default daily-peak configuration.

**15m+4h (OR)** is selected overall, with `c = 5` for 15m and `c = 3` for 4h:

| Split | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Validation | 99.11% | 97.19% | 96.80% | 96.99% |
| Held-out test | 99.23% | 98.39% | 96.44% | 97.41% |

It wins on the validation tie-break: `1h+4h` and `15m+1h+4h` have the same
validation F1 and precision. `15m+4h` is retained because it is lexicographically
earlier than `1h+4h` and smaller than `15m+1h+4h`. Both tied candidates actually
have the better held-out test F1, 97.61%, versus 97.41% for `15m+4h`; the protocol
does not use test for selection.

The 4-hour window alone is the best single frame and the best AND candidate:
95.95% validation F1 and 96.37% test F1. Adding 15m raises validation recall from
94.80% to 96.80% and test recall from 94.47% to 96.44%, with precision essentially
unchanged (97.13% → 97.19% validation, 98.35% → 98.39% test). No multi-member AND
beats 4h alone on either validation or test F1.

The best **multi-member** AND is **4h+1d** with `c = 3` for 4h and `c = 2` for
1d. It reaches **99.05% validation precision** and **99.02% test precision**, but
only 83.20% validation recall and 79.84% test recall, for F1 90.43% and 88.40%.
That is a precision-first option, not a better F1 candidate.

### Best candidate by subset size

| Size | Best OR | Val F1 | Test F1 | Best AND | Val F1 | Test F1 |
|---:|---|---:|---:|---|---:|---:|
| 1 | 4h | 95.95% | 96.37% | 4h | 95.95% | 96.37% |
| 2 | 15m+4h | 96.99% | 97.41% | 4h+1d | 90.43% | 88.40% |
| 3 | 15m+1h+4h | 96.99% | 97.61% | 4h+1d+7d | 76.47% | 80.00% |
| 4 | 5m+15m+1h+4h | 96.03% | 96.27% | 5m+15m+1h+4h | 72.12% | 70.92% |
| 5 | 1m+5m+15m+1h+4h | 92.93% | 92.63% | 1m+5m+15m+1h+4h | 65.59% | 65.07% |
| 6 | 1m+5m+15m+1h+4h+1d | 53.48% | 55.07% | 1m+5m+15m+1h+4h+1d | 59.55% | 57.06% |
| 7 | all seven | 36.26% | 32.77% | all seven | 49.40% | 51.61% |

### What the sweep says

- **OR peaks at two members.** `15m+4h` adds 5 validation and 5 test true
  positives over 4h alone, with no extra false positives in either split.
- **The top OR candidates tie on validation.** `1h+4h` and `15m+1h+4h` match
  `15m+4h` on validation F1 (96.99%), precision, and accuracy. The protocol keeps
  `15m+4h` by the size-then-lexicographic tie-break, even though the tied
  candidates have higher held-out test F1 (97.61% vs 97.41%). Adding 1m or 5m to
  `15m+4h` lowers validation F1.
- **AND's best is a singleton.** Multi-member AND suppresses recall faster than
  it gains precision. `4h+1d` is the only multi-member AND worth considering,
  and only if a false positive is much more costly than a miss.
- **Long windows are not OR partners.** Any OR subset containing 1d or 7d
  collapses precision. All seven under OR is effectively an always-alert
  detector: 100% validation recall at 22.14% validation precision and 19.61%
  test precision. All seven under AND is the opposite: 100% precision but only
  32.80% validation recall and 34.78% test recall.
- **Validation and test agree on the main OR family.** The top candidates are
  all 4h plus one short window; validation prefers `15m+4h`, while test slightly
  prefers `1h+4h`.

## Result at ~50,000 queries/day

The same protocol at **~50,000 queries/day** (~50,232 records per day, measured
from the generated log) over a 29-week period. Results are pooled across the ten
seeds: 1,680 validation and 1,680 test buckets per tuple. Constants are shown in
tuple order; the first six columns are validation, the last four held-out test.

| Tuple | Best constants | Accuracy | Precision | Recall | F1 | Test accuracy | Test precision | Test recall | Test F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **5m+4h** | 6 \| 3 | 99.52% | 98.44% | 98.44% | 98.44% | 99.40% | 97.46% | 98.29% | 97.87% |
| 15m+4h | 5 \| 3 | 99.52% | 98.44% | 98.44% | 98.44% | 99.40% | 97.46% | 98.29% | 97.87% |
| 1h+4h | 6 \| 3 | 99.52% | 98.44% | 98.44% | 98.44% | 99.46% | 97.87% | 98.29% | 98.08% |
| 5m+1h+4h | 6 \| 6 \| 3 | 99.52% | 98.44% | 98.44% | 98.44% | 99.40% | 97.46% | 98.29% | 97.87% |
| 5m+15m+4h | 6 \| 5 \| 3 | 99.52% | 98.44% | 98.44% | 98.44% | 99.40% | 97.46% | 98.29% | 97.87% |
| 15m+1h+4h | 5 \| 6 \| 3 | 99.52% | 98.44% | 98.44% | 98.44% | 99.40% | 97.46% | 98.29% | 97.87% |
| 5m+15m+1h+4h | 6 \| 5 \| 6 \| 3 | 99.52% | 98.44% | 98.44% | 98.44% | 99.40% | 97.46% | 98.29% | 97.87% |
| 1m+4h | 4 \| 3 | 99.35% | 97.30% | 98.44% | 97.86% | 99.35% | 97.05% | 98.29% | 97.66% |
| 4h | 3 | 98.93% | 98.37% | 94.53% | 96.41% | 98.69% | 97.75% | 92.74% | 95.18% |
| 1h | 6 | 94.94% | 100.00% | 66.80% | 80.09% | 95.24% | 100.00% | 65.81% | 79.38% |
| 5m | 6 | 94.94% | 100.00% | 66.80% | 80.09% | 95.18% | 99.35% | 65.81% | 79.18% |
| 15m | 5 | 94.94% | 100.00% | 66.80% | 80.09% | 95.18% | 99.35% | 65.81% | 79.18% |
| 1m | 4 | 94.35% | 98.20% | 64.06% | 77.54% | 94.94% | 98.69% | 64.53% | 78.04% |
| 4h+1d | 3 \| 2 | 75.71% | 38.38% | 98.05% | 55.16% | 76.07% | 36.58% | 97.86% | 53.26% |
| 1d | 2 | 73.69% | 35.00% | 84.77% | 49.54% | 74.82% | 34.22% | 87.61% | 49.22% |
| 4h+7d | 3 \| 4 | 69.94% | 33.24% | 96.48% | 49.45% | 66.49% | 28.99% | 97.01% | 44.64% |
| 7d | 4 | 64.05% | 22.73% | 56.64% | 32.44% | 61.31% | 19.86% | 58.55% | 29.65% |

The table lists every tuple that ties for the top validation score, the
next-best pair, the seven single frames, and the long-window combinations that
fail. The remaining candidates in `subset_results.csv` are 1m+4h supersets and
1d/7d combinations, which fall between these rows.

**Result:** every tuple pairing a short window with **4h** ties at **98.44%**
validation F1 — 5m+4h, 15m+4h, 1h+4h, and the three- and four-member supersets
that contain them. They are not distinguished by validation at all; 5m+4h is
selected only by the documented tie-break (smaller subset, then frame order),
which is why 5m beats the later 1h. On held-out test the order reverses:
**1h+4h scores highest overall at 98.08%**, ahead of 5m+4h at 97.87%.

The 4-hour frame does the work. Alone at `c = 3` it reaches 96.41% validation
F1, and every sub-hour window on its own caps out near 80% with recall under
67% — they fire almost never (5m at `c = 6`: 171 true positives, **0 false
positives**, 85 misses). Adding a sub-hour window to 4h buys 10 extra detections
without adding any false alarms — the 4 false positives in the combined row are
4h's own. Widening the tuple past two members changes nothing measurable. Long
windows degrade badly: 1d and 7d carry `c = 2` and `c = 4` and produce mostly
false positives (precision 35% and 23%), and OR-ing them onto 4h drops F1 to
~50%.

Reproduce it with the higher-load config:

```sh
go run ./v2/experiments/scenario2_tuple \
  -config "$TEMP/scenario2-29wk-50k/config_50k.textproto" \
  -output "$TEMP/scenario2-29wk-50k/results"
```

Write to a scratch directory, not `results/`: that keeps the checked-in
artifacts as the default config's. Cost is a 1.2 GB protobuf log per seed
(10,197,049 records, ~46 s) and about 19 minutes for all ten seeds. The
standalone CLI retains all ten raw logs until the run ends. The controlled
duration runner above removes raw logs after each seed.

The historical training-duration table is a different experiment, in
`scenario2_timeframe`. For the current hourly tuple, use the controlled sweep
above. The standalone CLI’s `-training-weeks` flag shifts the validation/test
dates and generated samples; its scores cannot isolate the effect of training
duration.

## Caveats

These are synthetic validation/test results, not production thresholds. The
runner reuses the row window but does not exercise the RPC services. Constants
are selected independently per frame, not jointly per subset, and both the
constants and the candidate are selected on the same four validation weeks, so
the validation figures are optimistic; the four test weeks are the better
guide. The leading tuples are close or tied, so validate the chosen candidate
on a second independent period before relying on it.
