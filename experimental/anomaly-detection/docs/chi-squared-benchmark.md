# Benchmarking the existing chi-squared model

This runner calls the existing `ChiSquaredModel` through the prediction service's
file `Predict` RPC. It does not modify the model, train a baseline, fit a threshold,
or call `CalibrateWindows` / `ScoreWindows`.

## Build and run

Run from `experimental/anomaly-detection`:

```bash
make build-chi-squared-benchmark
python3 scripts/run_chi_squared_benchmark.py \
  --input output/window-study-one-percent-write-heavy/standard_user/window_length/point-00-repeat-00/validation/logs.binproto \
  --output-dir output/chi-squared-study \
  --historical-m 1000 --recent-n 100 --threshold 10 \
  --sweep both --points 20 \
  --window-min 20 --window-max 500 \
  --threshold-min 1 --threshold-max 50
```

The input is an existing, labeled, length-delimited `AuditLogEntry` file. The
example uses our saved 200,000-action stream: 1% anomalous actions, three standard
tokens, 90% write probability during anomaly generation, 30-action chunks and
unrestricted random gaps. Other labeled streams can be supplied without changing
the model. This script does not regenerate data or select a new seed.

`--sweep window` runs only the recent-window sweep; `--sweep threshold` runs only
the threshold sweep. `both` performs two isolated sweeps, not a Cartesian grid:

| Sweep | Changes | Held fixed |
|---|---|---|
| window | Recent length N, 20 integer settings from 20 to 500 | M=1000; threshold=10 |
| threshold | Raw statistic threshold, 20 settings from 1 to 50 | M=1000; N=100 |

Defaults are starting values, not statistically selected cutoffs. M must exceed
every N tested. Every trial creates a fresh configuration-only model and sends
the entire file in one RPC, so rolling state persists across actions but never
leaks between trials. The script starts/stops its own prediction service on an
available local port. No training service is started. Inputs and outputs use a
shared filesystem with that service.

The direct runner is `bin/chi-squared-benchmark`; its full options are shown with
`--help`. The script's `--warmup-actions` defaults to -1, meaning M. Its per-trial
timeout defaults to five minutes (direct runner `--timeout`). Use a fresh output
directory; this first version intentionally does not reuse cached trials.

## Prediction and evaluation semantics

The model is unchanged: the last N actions are contained inside the last M
historical actions. It counts individual normalized query templates, computes
the raw 2-by-K chi-squared homogeneity statistic and flags `stat >= threshold`.
It does not use n-grams or timestamps and its overlapping samples are not an
independent two-sample test. No p-values or significance levels are inferred by
this runner.

One prediction arrives per action. For each prediction, ground truth is positive
if **any of the last N actions** has `is_ground_truth_anomalous=true`. This matches
the recent window under comparison, rather than artificially grouping predictions
into non-overlapping windows or treating the current action as the entire window.

The first `warmup_actions` actions still pass through the model normally; only
their decisions are excluded from metrics. Default warm-up is M, ensuring a full
historical window before scoring metrics. You can choose a smaller warm-up, but
it must cover every N tested. Labels within warm-up remain in the recent-window
truth buffer when subsequent decisions are evaluated. The report counts anomalous
actions excluded during warm-up. Empty input templates, mismatched action order,
missing findings, extra findings and inputs shorter than warm-up are rejected.

TP/FP/TN/FN count evaluated **rolling decisions**. Precision is TP/(TP+FP), recall
is TP/(TP+FN), and false-positive rate is FP/(FP+TN). Undefined ratios are blank
in CSV and absent in JSON, not zero. Changing N also changes positive-window
labels, so recall across N values is not an identical classification problem.
Adjacent rolling decisions overlap and should not be treated as independent
observations when calculating statistical uncertainty.

This is a distribution-change detector, not a directional attack classifier:
after an anomaly, returning to normal can itself differ from the contaminated
historical window. Such alerts count as false positives under this label policy.
Long attacks can also enter the historical window and stop producing divergence.
These are model/evaluation behaviors to measure, not changes made by the runner.

## Call path and files

```text
scripts/run_chi_squared_benchmark.py
  ├─ bin/prediction-service
  │    └─ v2/prediction_service/server.go → ChiSquaredModel
  └─ bin/chi-squared-benchmark
       ├─ v2/chi_squared_benchmark/main.go: grid, model files, Predict RPC
       └─ evaluation.go: recent-window ground truth and precision/recall
```

```text
output/chi-squared-study/
  benchmark.json                # Exact sweep, warm-up, label policy, input SHA256
  prediction-service.log
  runner.log
  results.csv                   # Parameters, confusion counts, precision/recall
  window_length.png / .svg      # Precision and recall versus N
  threshold.png / .svg          # Precision and recall versus raw threshold
  completed.json                # Completion count and unchanged-input verification
  window_length/point-00/
    model.bin / model.textproto # Existing ChiSquaredScenario: M, N, threshold
    request.textproto           # The actual full-file Predict request
    findings.textproto          # Actual per-action output from the service
    evaluation.json             # Counts, metrics and warm-up exclusions
  threshold/point-20/            # Indices are global across the two sweeps
    ...
```

The input's hash is recorded before the run and verified again after all trials.
No copies of ground-truth labels enter model configuration. Existing prediction
findings do not expose the raw chi-squared statistic; the runner uses the actual
classification instead of reconstructing scores from heuristic confidence.

## Compatibility

The legacy `v2/client` on this branch still references the removed
`MarkovAnomalyProfile` schema and does not build. The dedicated build/test targets
avoid that package; repairing it is outside this benchmark path. Use:

```bash
make test-chi-squared-benchmark
```
