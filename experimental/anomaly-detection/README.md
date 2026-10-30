# Anomaly detection experiments

This module generates database actions, trains a normal n-gram baseline, calibrates
its window-score threshold on separate normal data, and scores validation windows.
Training, calibration and validation use independent streams. Scoring never updates
the learned counts, means or variances. Validation can interleave labeled segments
from profiles with different transition probabilities.

## Structure

```text
v1/generator/src/                 Markov generation, with optional --seed
v1/client/                       Generator configurations and legacy client
v2/client/main.go                Experiment orchestration and training RPC
v2/client/windows.go             Calibration and validation RPC orchestration
v2/client/injection.go           Validation-only graph/profile composition
v2/client/evaluation.go          Ground-truth comparison after scoring
v2/client/export.go              Record-count validation and JSONL export
v2/client/proto/experiment.proto Experiment configuration schema
v2/client/markov_experiment.textproto  Default experiment
v2/training_service/             Baseline training and persistence
v2/prediction_service/           Existing prediction APIs plus window RPCs
v2/prediction_service/service/ngram_window_scorer.go  Frozen-baseline scoring
v2/common/proto/model.proto      Models, window statistics and calibration metadata
```

The legacy `v1/client` depends on empty `v1/server/proto` definitions in this
checkout. The v2 client uses the working v2 services and the v1 generator; audit
records are wire-compatible. Existing single-record prediction models retain their
original API. Histogram models use the new window-specific RPCs.

## Build and run

Prerequisites: Go 1.23 or later, downloaded module dependencies, and a filesystem
shared by the client and both services. Run from `experimental/anomaly-detection`.

```bash
go build -o /tmp/markov-scoring-bin/ ./v1/generator/src ./v2/training_service ./v2/prediction_service ./v2/client
mkdir -p output/markov-window-scoring/models
```

Start training in one terminal:

```bash
/tmp/markov-scoring-bin/training_service \
  --port=50063 --model_dir="$PWD/output/markov-window-scoring/models"
```

Start prediction in another terminal:

```bash
/tmp/markov-scoring-bin/prediction_service \
  --port=50064 --models_dir="$PWD/output/markov-window-scoring/models"
```

Then run the full experiment:

```bash
/tmp/markov-scoring-bin/client \
  --generator_bin=/tmp/markov-scoring-bin/src \
  --experiment_config=v2/client/markov_experiment.textproto \
  --server_addr=localhost:50063 \
  --prediction_server_addr=localhost:50064 \
  --output_dir=output/markov-window-scoring
```

The experiment textproto configures the same normal Markov graph with:

| Stage | Actions | Seed | Complete 300-action windows |
|---|---:|---:|---:|
| Training | 30,000 | 42 | 100 |
| Calibration | 300,000 | 43 | 1,000 |
| Normal validation smoke test | 30,000 | 44 | 100 |

It also sets max_n=3, window_length=300 and calibration_percentile=0.99.
Training consumes every training record. There is no train/test split. Changing
these values means editing the textproto. Seeds must differ. Paths follow the
existing working-directory convention. The saved experiment and graph snapshot
make generation reproducible. Omitting `--seed` on standalone generator calls
still selects a random seed; explicit zero is a valid deterministic seed.

Use `--validation_log_path=...` with the full experiment to score an existing
validation dataset instead of generating the normal smoke-test stream.

## Workload profiles under v1/client

Each `markov_*_config.textproto` defines a complete normal workload graph. A node
specifies a normalized SQL template and placeholder values, a simulated completion
time, and outgoing weighted edges. The generator starts at the configured node,
emits its action, advances the simulated clock, and samples the next node. A
terminal node ends a walk; generation restarts until the requested action count.
These are synthetic logs: SQL is not executed and the process does not sleep.

| File | Behavior |
|---|---|
| `markov_standard_user_config.textproto` | Idle SELECT 1 actions with 20–40 second spacing, interrupted by read/write bursts with 10–14 second spacing |
| `markov_maintenance_user_config.textproto` | CREATE INDEX, ALTER TABLE and VACUUM sessions, ending at session_done before a fresh session starts |
| `markov_service_usage_config.textproto` | Fast point/range reads with rare writes, mostly 15–30 ms spacing for reads |

Edge weights are conditional probabilities, not overall action proportions. For
example, the standard user's idle node has an 85% chance to stay idle and a 15%
chance to enter active_read **when currently at idle**. The final histogram emerges
from repeated transitions through all nodes. Completion times affect timestamps,
not transition probabilities. The service graph does not automatically implement
peak/off-peak schedules. All current profiles use the generator's shared database
identity; their names describe workload behavior rather than distinct db_user values.

The idle node emits SELECT 1; it does not represent a period with no logged action.
The current scorer uses windows of actions and ignores timestamps, so timing-only
changes are not detected. Each workload is trained/evaluated separately in the
regression test. Its anomaly fixture changes only transition probabilities on the
same graph, preserving SQL templates and timings. Mixing these three complete
workloads together would also change SQL categories, which is a different test.

## Validation without training or calibration

Only the prediction service is needed after a calibrated model has been saved:

```bash
/tmp/markov-scoring-bin/client \
  --prediction_server_addr=localhost:50064 \
  --model_path=output/markov-window-scoring/calibrated_histogram.bin \
  --validation_log_path=output/markov-window-scoring/validation/logs.binproto \
  --output_dir=output/validation-only
```

This mode does not read the experiment, invoke generation, call the training
service, or recalibrate. The input must contain length-delimited audit records.
It also evaluates predictions against the supplied ground-truth labels afterward.

## Labeled mixed validation

Use the same services and binaries with the mixed example:

```bash
/tmp/markov-scoring-bin/client \
  --generator_bin=/tmp/markov-scoring-bin/src \
  --experiment_config=v2/client/markov_mixed_experiment.textproto \
  --server_addr=localhost:50063 \
  --prediction_server_addr=localhost:50064 \
  --output_dir=output/markov-mixed-validation
```

The example defines `browse-heavy` and `checkout-heavy` profiles by overriding
only edge weights in the original login/browse/checkout graph. Templates,
placeholder constraints, completion times, start selection and topology remain
unchanged. Each overridden node must preserve every original destination, have
finite nonnegative weights summing to one, and the profile must actually change
at least one probability. Other baseline Markov configs can be used with overrides
referring to their own node IDs.

Explicit injections specify a unique segment_id, profile_id, zero-based
start_action and positive num_actions. Ranges must be ordered, nonoverlapping,
and fit validation_actions. The example injects 4,350 of 30,000 validation actions
across six segments, with both aligned and boundary-crossing positions.

Only generated validation receives profiles/injections. Training and calibration
continue using the untouched normal graph. The composed graph and schedule are
saved to `validation/generator_config.textproto`. When supplying an external
validation_log_path, generation and the experiment's injection schedule are skipped;
evaluation uses the labels actually present in that file.

The generator pauses the normal walk at an injection, starts a fresh walk in the
selected profile, and resumes the paused normal state after the segment. Terminal
nodes restart within a segment. All walks share one seeded random source and one
continuous simulated clock. Injected random draws can change subsequent random
choices in the resumed normal walk; removing injected records does not promise
the exact sequence of a separately generated normal stream.

The generator reuses `AuditLogEntry.is_ground_truth_anomalous` (field 5) and adds
wire-compatible provenance in both v1 and v2 audit schemas:

| Field | Normal record in a mixed stream | Injected record |
|---|---|---|
| `is_ground_truth_anomalous` | false | true |
| `ground_truth_profile` (7) | normal | Selected profile ID |
| `ground_truth_segment_id` (8) | Empty | Configured segment ID |

Labels are assigned from the selected source profile, even though the action
templates are shared. In mixed mode this overrides template-level anomaly flags.
Ordinary generation without injections retains its original labeling behavior
and has empty provenance fields. The experiment client rejects injections and
anomalous template labels in the normal training/calibration specification.

After ScoreWindows returns, the client separately reads the labels and saves
`validation_evaluation.json`. A window is ground-truth positive if it contains
any anomalous action. Each result includes the anomalous-action count/fraction,
predicted flag, score, and intersecting segment IDs. The summary contains TP, FP,
TN, FN, precision, recall, F1, false-positive rate and segment detection rate.
Undefined metrics are omitted. Segments entirely in an unscored trailing suffix
are reported but excluded from scored segment recall. Older records with a boolean
label but no segment ID still contribute to window evaluation. Unset boolean
labels are treated as normal, matching protobuf's existing default.

This evaluation does not feed labels into scoring, recalibrate thresholds, or
update the baseline. The live mixed example detected 19/19 positive windows and
6/6 segments with 0 false positives among 81 normal windows. These are results
for deliberately strong, synthetic profile changes, not a general detection-rate
guarantee. `injection_verification.json` independently checks every label, range,
window comparison, segment comparison and unchanged normal dataset.

## Training statistics

Each normalized query template is a category. Placeholder values do not create
new categories. The trainer preserves overall counts and probabilities for every
observed 1–3-gram across the full training stream, including window boundaries:

```text
overall probability = count / (total_actions - n + 1)
```

It also partitions the stream into complete, nonoverlapping W-action windows.
Within each window, only n-grams wholly contained in the window are counted:

```text
window proportion = count_in_window / (W - n + 1)
mean = average of category proportions across K complete windows
sample variance = sum((proportion - mean)^2) / (K - 1)
```

Per-category `window_statistics` includes zero-occurrence windows. Welford
accumulation and merging missing zeros avoid storing the full sequence. These
are marginal proportion statistics, not a covariance matrix. Raw count means and
variances are recoverable by multiplying by W-n+1 and its square respectively.

Trailing actions contribute to overall training counts but not window statistics.
`complete_windows` and `trailing_actions` record the distinction. Statistics are
absent when there are no complete windows; optional variance is absent when
there is only one. The optional window_length field defaults to 300 for legacy
RPC configurations. Existing protobuf fields and oneof numbers are preserved.
Older models remain readable. Proto JSON represents int64 counts as strings.

## Variance-aware scoring and calibration

The scorer uses each n-gram category's learned window mean and sample variance.
It does not compare the validation histogram with the overall training histogram.
The overall counts/probabilities remain saved for inspection and compatibility.
A model needs at least two complete training windows with stored means/variances.

For a W-action validation window and n-gram order n, there are M=W-n+1 positions.
For every known category, including absent categories with observed count zero:

```text
observed proportion = category count / M
sigma_effective = max(sqrt(learned variance), floor_count / M)
z = (observed proportion - learned mean) / sigma_effective
```

`scoring_config.standard_deviation_floor_count` defaults to 1 occurrence and must
be finite and positive. This floor prevents division by zero and excessive
sensitivity to tiny learned variances. It is a modeling choice, not a probability
or statistical guarantee. The same value is persisted in calibration and used
for validation; changing it requires recalibration.

Previously unseen sequences share one unknown bucket per order, with training
mean/variance zero and the same floor. That bucket is always included in the
category denominator, even when its observed count is zero. The aggregate is:

```text
order_score(n)^2 = sum(z^2 across known categories and unknown bucket) / (known_categories + 1)
combined score = sqrt(average(order_score(n)^2 over n=1,2,3))
```

Both unusually frequent and unusually absent sequences contribute. Scores are
nonnegative and unbounded; they are not probabilities or confidence levels. Each
n-gram order has equal overall weight. Categories are correlated, so no Gaussian
or chi-squared distribution is assumed for the combined score.

`PredictionService.CalibrateWindows` scores a separate normal file and saves a
new model with calibration metadata, preserving the learned counts, moments and
training metadata. The default 1,000 calibration windows give more tail resolution
than the 100 training windows. The threshold is the nearest-rank empirical quantile:

```text
threshold = sorted_calibration_scores[ceil(percentile * K) - 1]
anomalous = validation_score > threshold
```

Equality is not anomalous. The configured percentile defaults to 0.99 in the
example experiments; it does not guarantee a 1% future false-positive rate.
Calibration stores the percentile, threshold, score mean/sample variance, action
and window counts, time, scoring configuration, and scoring version `variance_rms_v1`.

`PredictionService.ScoreWindows` uses a frozen calibrated model. Neither scoring
nor calibration calls the trainer. Labels, provenance, user identity and timestamps
are ignored by the scorer. Reports have per-order RMS scores in `order_scores`,
the combined score, classification, record ranges and trailing-action count.
`top_deviations` contains the ten largest contributions to score squared, with
sequence/order, observed proportion, learned mean/variance, effective sigma,
signed standardized deviation and weighted contribution. An unknown bucket is
explicitly marked rather than represented as a synthetic SQL template.

Complete, nonoverlapping windows are scored. Partial windows are reported but
not scored; input with no complete validation window is rejected. Old
Jensen–Shannon thresholds are rejected and must be recalibrated. The deprecated
`distances` protobuf field remains for wire compatibility but is no longer populated.
Older baselines with complete window statistics can be recalibrated; baselines
without those statistics need retraining. Running the full experiment handles
training and calibration automatically.

## Outputs

| File | Contents |
|---|---|
| `experiment.textproto`, `generator_config.textproto` | Actual settings and graph snapshot |
| `logs.binproto`, `logs.jsonl` | All training records, binary and readable |
| `training_request.textproto`, `training_response.json` | Training RPC artifacts |
| `models/model_v2_*.bin` | Model originally persisted by training service |
| `histogram.bin`, `histogram.json` | Unchanged training baseline |
| `calibration/logs.binproto`, `calibration/logs.jsonl` | Independent normal calibration stream |
| `calibration_request.textproto`, `calibration_response.json` | Calibration RPC artifacts |
| `calibration_scores.json` | All calibration-window RMS scores and explanations |
| `calibrated_histogram.bin`, `calibrated_histogram.json` | Baseline plus learned threshold |
| `validation/logs.binproto`, `validation/logs.jsonl` | Independently generated normal validation stream |
| `validation_request.textproto`, `validation_scores.json` | Validation RPC and per-window flags |
| `validation_evaluation.json` | Post-scoring window/segment ground-truth comparison |

Validation-only mode writes the validation request, scores and evaluation in its output
directory. Use a fresh output directory to distinguish experiments.

## Verification

```bash
go test ./v2/... ./v1/generator/...
```

This runs the full active-pipeline suite, including `TestMarkovPipeline`, which
builds and invokes the actual client CLI, generator and both service processes.
It tests a deterministic known-answer graph and all three upstream workload files
under `v1/client`: standard user, maintenance user and service usage. Each workload
runs with three training/calibration/validation seed groups (42/43/44, 142/143/144,
242/243/244), for ten end-to-end scenarios in total. The normal graph snapshot must
match its source file exactly; normal probabilities are not duplicated in test code.
Separate anomaly overrides and injection schedules are declared in textproto
fixtures under `v2/client/testdata`.
Independent references compute counts from full slices, mean/variance from
explicit batch vectors, standardized deviations computed independently in raw
count units, the
empirical calibration quantile, and ground-truth window/segment metrics.

The test includes aligned and boundary-crossing injections, unscored trailing
anomalies, a normal-only control, removal of all ground-truth labels, rejection
of uncalibrated models, and byte-unchanged validation-only model reuse. It does
not require perfect detection in stochastic cases: a false positive or missed
anomaly is accepted only when independently computed scores and thresholds agree.
These are functional correctness tests, not statistical performance guarantees.

Service processes are stopped on cleanup. `go test -short` skips this end-to-end
test. Local socket access is needed. To retain inputs, outputs, server logs and
independent reference reports, set MARKOV_E2E_ARTIFACT_DIR to an absolute directory.
To repeat the active suite with race detection in both tests and child processes
(requires the Go race detector/C compiler):

```bash
MARKOV_E2E_RACE=1 go test -race -count=3 ./v2/... ./v1/generator/...
```

Reports from the pre-commit audit are under `output/regression-audit`. Repeats use
the same seeds to check reproducibility; they are not independent performance
samples.

Individual component checks can also be run:

```bash
go test ./v2/client/... ./v2/prediction_service/service ./v2/training_service/service ./v1/generator/...
go test ./v2/prediction_service -run TestWindowCalibrationAndValidationRPC -v
go test ./v2/training_service -run TestTrainNGramHistogramPersistence -v
go test ./v1/generator/src -run TestMarkovInjection -v
go test ./v2/client -run 'TestGroundTruth|TestMixedExperiment' -v
```

Tests cover analytic standardized scores, learned variance, zero-variance floors,
unknown buckets, changed ordering with the same
unigram distribution, threshold equality, empirical quantiles, invalid models,
trailing windows, protobuf persistence, baseline immutability and label isolation.
Training tests independently verify 9,873 counting combinations and 953 explicit
batch-statistic comparisons.

Current variance-scoring results are under `output/variance-profile-audit`.
The full-size runs load the actual three v1/client workload files, train on all
30,000 normal records, calibrate on 300,000 separate normal records, and score
30,000 mixed validation records (100 windows). Each mixed stream contains 4,350
injected actions across six segments, using the same actions/timing and changed
transition probabilities only. Seeds are 42/43/44. All 19 positive windows and
six scored segments were detected in each run; false positives among 81 normal
windows were 1 (standard user), 3 (maintenance user), and 1 (service usage).
These deliberately strong synthetic examples do not establish performance on
subtler anomalies. Independent raw-count calculations verified every training
mean/variance, component score, combined score, threshold and evaluation result.
`full-scale/<profile>/independent_verification.json` records those checks.

The smaller automated profile fixtures additionally cover three seed groups,
normal-only controls, label isolation, trailing anomalies, and validation-only
model reuse. Those results include both false positives and missed windows. The
active suite passed three times with race detection. Reports and exact logs are
retained under `output/variance-profile-audit`. Older report directories preserve
historical Jensen–Shannon results and should not be confused with the current scorer.

The two previously failing error-handling tests now pass. A directory supplied
as a training file is classified as an operational error (Internal), while
malformed/truncated protobuf data remains InvalidArgument. Listing an explicitly
requested missing model location returns an empty result; prediction on a missing
location and a missing default model directory still report errors.

The legacy `v1/client` is outside this active-pipeline suite because its original
service protobuf source files are empty in this checkout.

Regenerate protobuf definitions with protoc and both Go plugins on PATH:

```bash
protoc -I . --go_out=. --go_opt=paths=source_relative \
  --go_opt=default_api_level=API_OPAQUE \
  --go-grpc_out=. --go-grpc_opt=paths=source_relative \
  v1/generator/proto/query_template_config.proto v1/generator/proto/audit_log_entry.proto \
  v2/common/proto/audit_log_entry.proto v2/common/proto/model.proto \
  v2/client/proto/experiment.proto v2/client/proto/evaluation.proto \
  v2/prediction_service/proto/prediction_service.proto
```

## Cleanup

Stop both foreground services with Ctrl-C. Remove chosen output directories and
`/tmp/markov-scoring-bin` when no longer needed. Earlier experiment outputs are
preserved. Generated results are not intended for source commits.
