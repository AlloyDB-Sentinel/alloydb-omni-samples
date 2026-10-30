# Scenario 2 training duration

[ALDB-16](https://linear.app/aubree-lianto/issue/ALDB-16/scenario-2-experiment-specified-timeframe-optimal-constants) · [Experiment notes](https://docs.google.com/document/d/102uW-D5MrNQTdyOZ9kMFOsCUFtwXFUjvNbSKoY3FRL8/edit)

From `experimental/anomaly-detection`, with Go 1.23 or later:

```sh
go run ./v2/experiments/scenario2_timeframe -output /tmp/scenario2-duration
go test ./v2/experiments/scenario2_timeframe ./v2/common ./v1/generator/...
```

The command builds the existing generator, runs five seeds, and writes:
- `training_duration.csv`: the table's 13 rows (percentages averaged across seeds).
- `training_duration.png`: one accuracy/precision/recall chart.
- `per_seed.csv`: all 520 candidate scores and confusion counts, for auditing.

Checked-in summary and chart are in `results/`. Raw generated logs are temporary
and deleted after the run. No service or database is required.

## Experiment

- Fixed 4-hour rolling returned-row volume, grouped by database/user; one identity.
- Each fixed 4-hour reporting bucket uses its maximum rolling total. Its label is
  anomalous if any observed rolling window contains an attack read, including spillover.
- Clean training: 1–50 whole weeks, ending at the same Monday boundary. Baselines
  use mean + c × population stddev for each of the 42 weekly slots separately.
- For each duration, select c from integers 1–8 by mean validation accuracy;
  ties use smaller c. Report all three metrics at that same c, averaged over five runs.
- Validation: the same four weeks for every candidate. One-hour attacks occur with
  2% probability, with 3× reads and normal rows/read; see `config.textproto`.
- Seeds 42–46 cover Jan 5–Jun 29, 2026 (21 clean weeks, four validation weeks).
  Seeds 3042–3046 prepend 30 clean weeks. Discard the first week as warmup;
  this preserves the original 1–20-week observations when adding 30/40/50 weeks.

20 and 40 weeks tie at 99.29% validation accuracy; prefer 20 for less training.
These are synthetic validation results, not an independent test or a production
optimum. This runner reuses the row window, but does not exercise the RPC services.
Violet's duration experiment instead fixed c = 5 and used 1-hour request counts.
