# Independent anomaly validation

The three default experiment configs now use `independent_anomalies`. Training
and calibration still use the unmodified normal workload. Existing explicit
`anomaly_profiles` / `injections` remain supported as a separate legacy mode;
they cannot be combined with the new mode.

```protobuf
independent_anomalies {
  alphabet_config_paths: "v1/client/markov_standard_user_config.textproto"
  preferred_templates: "UPDATE documents SET body = %v WHERE doc_id = %v"
  preferred_probability: 0.9
  chunk_actions: 30
  completion_min_ms: 1
  completion_max_ms: 10
}
```

## Action generator

The default experiment configs explicitly select only the standard workload's
three templates. With no alphabet paths, the normal workload supplies the alphabet.
With explicit paths, only those files supply it; the normal workload is not added
implicitly. Multiple paths combine Markov, weighted, and bucket templates. Templates are deduplicated by their
normalized query string; the first occurrence supplies placeholder constraints.
Connection-only configs have no query alphabet and are not accepted. Add new
query config paths when adding new actions to the repository. SQL is generated
as log text, never executed against a database.

A separate graph connects every action to every action, including itself. The
current default reserves 90% probability for the UPDATE template in every
transition row and in the initial draw. Thus the graph remains strongly
write-heavy across seeds: the standard normal graph assigns only 0–35% write
probability per row. The remaining 10% is randomly split among the other tokens.
The expected write share is 90%; finite samples will vary.

`preferred_templates` may list multiple actions, sharing `preferred_probability`
in total. Weights within the preferred and other groups are independently sampled
and normalized. Listed templates must exist in the alphabet and leave at least
one other token. Probability must be strictly between 0 and 1. When preferences
are omitted, rows revert to fully random weights and starts are uniform. The
resolved graph is fixed across chunks and saved for inspection.

Normal generation, anomaly generation, graph randomization, and schedule placement
use distinct, reproducible seeds derived from `validation_seed`. Their resolved
values are saved. Anomaly generation does not consume normal-generator RNG draws.
The anomaly walk continues across chunks; the normal walk pauses during them.

## Placement and exact budget

For N validation records, exactly N/100 (1%) are anomalous. N must be divisible by 100;
otherwise configuration validation reports an error instead of silently rounding.
For N=200,000, this means 2,000 anomalies: 66 chunks of 30 and one chunk of 20.
`chunk_actions` defaults to 30; the final chunk uses the remaining budget.

All remaining normal actions are divided randomly among the K+1 gaps surrounding
K chunks. The scheduler uniformly samples a weak composition using stars and
bars. Leading, internal, and trailing gaps may all be zero. Chunks may touch;
there is no minimum gap, normal-window guarantee, or scoring-window input.
Adjacent chunks retain distinct episode IDs even when they form one contiguous
anomalous run. Chunk size and action percentage are independent of timestamps.

Changing only scoring window length leaves the generated validation stream and
schedule unchanged. One percent anomalous records does not imply one percent
positive windows: the evaluator labels any window containing an anomalous action
positive and separately reports its anomalous fraction and unscored tail.

## Interleaving and time

Each source emits its required record count plus one lookahead record. The extra
record supplies the last consumed action's completion interval; it is never
included in the final dataset. The interleaver consumes each source in order,
assigns ground-truth labels from the schedule, and advances one shared simulated
clock by that source action's interval. Timestamps never move backwards. The
anomaly completion range defaults to 1–10 ms and is configurable. The n-gram
scorer uses action order/proportions, not timing.

## Saved artifacts

Inside each experiment's `validation/` directory:

- `schedule.json`: total/anomalous counts, all four seeds, chunk IDs and ranges.
- `anomaly_generator.textproto`: the fully resolved alphabet, randomized weights,
  start distribution, placeholder constraints and completion timings.
- `normal/` and `anomaly/`: independently generated source logs, including each
  source's single lookahead record.
- `logs.binproto` and `logs.jsonl`: final interleaved records, exactly N in total.

`validation_scores.json` and `validation_evaluation.json` remain in the experiment
root. They retain window positions, anomalous fractions, confusion counts and
per-episode detection. External `--validation_log_path` skips generation and uses
the supplied labels, so the generated-data 1% budget does not apply to it.

Run experiments with the existing client commands after `make build`. Use a new
output directory. Previous sweep graphs used the legacy anomaly generator and
are historical results; no parameter sweeps have been rerun for this change.

To expand beyond the three standard tokens, add another source inside
`independent_anomalies`, for example:

```protobuf
alphabet_config_paths: "v1/client/markov_maintenance_user_config.textproto"
```

Keep the standard path as well if its tokens should remain available. Independent
edge randomization within probability groups, timing defaults, and unrestricted
random gaps are retained.
Older graphs used either a 2% expanded alphabet or a 0.5% unrestricted random
three-token graph. Those historical results describe different conditions.
