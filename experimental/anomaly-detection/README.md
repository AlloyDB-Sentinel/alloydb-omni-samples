# Anomaly detection experiments

This Go module contains synthetic database log generators and model training and
prediction services. The Markov histogram MVP generates a sequence of database
actions, trains an empirical n-gram distribution on its first 40%, and saves the
model. It currently performs no anomaly detection.

## Structure

```text
v1/generator/src/                 Existing generators, including Markov generation
v1/client/                       Generator configuration examples and legacy client
v2/client/main.go                Orchestrates generation, splitting and training RPC
v2/client/split.go               Ordered split and readable generation export
v2/training_service/server.go    Selects trainer, ingests records, persists model
v2/training_service/service/     Trainers implementing Count and BuildModel
v2/common/proto/model.proto      Shared protobuf model and training configuration
v2/training_service/proto/       Training RPC schema
v2/prediction_service/           Existing prediction service; unused by this MVP
```

The legacy `v1/client` depends on `v1/server/proto` service definitions that are
empty in this checkout. The new `v2/client` uses the existing v2 training service
and the v1 generator, whose audit record format is wire-compatible with v2.
The earlier standalone `v1/histogram` implementation has been removed.

## Build and run

Prerequisites: Go 1.23 or later, downloaded Go module dependencies, and a local
filesystem shared by the client and training service. Run all commands from
`experimental/anomaly-detection`.

Build the three executables:

```bash
go build -o /tmp/markov-mvp-bin/ ./v1/generator/src ./v2/training_service ./v2/client
mkdir -p output/markov-training-demo/models
```

Start the training service in one terminal:

```bash
/tmp/markov-mvp-bin/training_service \
  --port=50061 \
  --model_dir="$PWD/output/markov-training-demo/models"
```

Run the orchestration client in another terminal:

```bash
/tmp/markov-mvp-bin/client \
  --generator_bin=/tmp/markov-mvp-bin/src \
  --config_path=v1/client/markov_generator_config.textproto \
  --server_addr=localhost:50061 \
  --output_dir=output/markov-training-demo \
  --num_logs=10000 --train_fraction=0.4 --max_n=3
```

The client invokes the generator, preserves the first 4,000 records as training
data, reserves the remaining 6,000, and calls `TrainingService.TrainModels`.
The trainer counts every record it receives; splitting is the client's job.
The service persists its model, and the client exports binary and JSON copies.
The service does not push histogram models to prediction because histogram
prediction is not implemented.

## Output files

All outputs are under the chosen `--output_dir`:

| File | Contents |
|---|---|
| `logs.binproto` | All generated audit records, length-delimited protobuf |
| `logs.jsonl` | All generated records as readable JSON, one action per line |
| `train.binproto` | First `floor(num_logs * train_fraction)` records |
| `test.binproto` | Remaining records, held out without prediction |
| `training_request.textproto` | Actual RPC request with training path and max_n |
| `training_response.json` | Service response with model location and metadata |
| `models/model_v2_*.bin` | Original protobuf model persisted by the service |
| `histogram.bin` | Client copy of the persisted protobuf model |
| `histogram.json` | Readable export of the full protobuf model |

`histogram.json` contains `ngram_histogram.model_data.histograms`, the configured
`max_n`, and training metadata. Each histogram entry has an `actions` array,
`count`, and `probability`. Protobuf JSON represents int64 counts as strings.

Each whole normalized query template is one action. Different placeholder values
remain the same action; different SELECT templates remain distinct. Overlapping
windows are counted independently for n=1, 2, and 3:

```text
total_windows(n) = max(training_actions - n + 1, 0)
probability(sequence) = count(sequence) / total_windows(n)
```

These are joint window frequencies, not conditional next-action probabilities.
Each nonempty histogram sums to one. Only observed sequences are stored; all
omitted sequences have probability zero. Short training prefixes yield empty
histograms for larger n. No smoothing, timing model, labels, or thresholds are
used. The input is one continuous sequence across users and Markov walk restarts;
the current generator does not emit session boundaries. Ordering is preserved,
with no shuffling or windows extending into the held-out suffix.

## Verification

```bash
go test ./v2/client ./v2/training_service/service ./v1/generator/...
go test ./v2/training_service -run TestTrainNGramHistogramPersistence -v
```

The first live run generated 10,000 actions and trained on 4,000. Its unigrams,
bigrams, and trigrams had 4,000, 3,999, and 3,998 windows respectively, with 3, 5,
and 8 observed sequences. Counts were independently checked against the first
4,000 rows of `logs.jsonl`; the result is saved in
`output/markov-training-demo/verification.json`. Generator randomness means future
runs can produce different counts.

A broader `go test ./v2/...` also exercises existing tests that currently fail:
`TestTrainModels/log_ingestion_internal_error` expects Internal but receives
InvalidArgument, and `TestListModels/single_model_location_nonexistent` expects
OK but receives Internal. These paths are outside the histogram flow.

If changing the protobuf schemas, regenerate their Go definitions with
`protoc`, `protoc-gen-go`, and the protobuf include files installed:

```bash
protoc -I . --go_out=. --go_opt=paths=source_relative \
  --go_opt=default_api_level=API_OPAQUE \
  v2/common/proto/model.proto \
  v2/training_service/proto/training_service.proto
```

Ensure `protoc-gen-go` is on PATH. The service methods are unchanged, so the
existing gRPC Go definitions do not require regeneration for this change.

## Cleanup

Stop the foreground training service with Ctrl-C. The client exits on completion.
The demo writes files only to `output/markov-training-demo` and executable files
to `/tmp/markov-mvp-bin`; remove those directories when no longer needed.
