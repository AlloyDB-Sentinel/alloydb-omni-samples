# Excessive Data Reads (Scenario 2)

`excessive_data_reads` counts rows read over the last 15 minutes and 4 hours
for each database/user. Each window has a baseline for each weekday and hour.
A read is anomalous when either total exceeds its baseline.

Training uses the largest rolling total observed in each hour. Hours with no
reads contribute zero. The threshold is mean + multiplier * population standard
deviation.
The selected multipliers are 6 for 15 minutes and 4 for 4 hours.

Use `TrainingService.TrainModels` with a request like this:

```textproto
train_log_path: "/path/to/clean-training.binproto"
training_configs {
  excessive_data_reads_config {
    short_window_std_dev_multiplier: 6
    long_window_std_dev_multiplier: 4
    training_start_time { seconds: 1767571200 } # 2026-01-05 00:00 UTC
    training_end_time { seconds: 1779667200 }   # 2026-05-25 00:00 UTC
  }
}
```

Training data must be clean and ordered by timestamp for each database/user.
The training interval must cover at least a week, with both ends on UTC hour
boundaries. The end time is exclusive. Reads at or after it are rejected.
Up to four hours of earlier reads can warm the windows without contributing
training peaks. The model stores all 168 hourly baselines for each known
database/user. Reads from unknown database/user pairs are anomalous.

Pass an ordered batch or binary log file and the saved model location to
`PredictionService.Predict`. Each request starts with empty windows. Separate
single-log requests do not share history, and initial totals cover only the
reads received so far.

For comparison with the experiment, prepend four hours of prediction warmup
reads and exclude their findings from scoring. The API returns one finding per
read. To match the experiment's reporting, group findings into four-hour UTC
buckets and flag a bucket if any finding is anomalous.

Run these commands from `experimental/anomaly-detection` to regenerate the Go
protobuf bindings and run the v2 tests:

```sh
protoc -I . --go_out=. --go_opt=paths=source_relative --go_opt=default_api_level=API_OPAQUE v2/common/proto/*.proto v2/training_service/proto/*.proto v2/prediction_service/proto/*.proto
protoc -I . --go-grpc_out=. --go-grpc_opt=paths=source_relative v2/training_service/proto/*.proto v2/prediction_service/proto/*.proto
go test ./v2/...
```
