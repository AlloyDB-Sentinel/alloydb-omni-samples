package main

import (
	"bytes"
	"math"
	"testing"
	"time"

	auditpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common/proto"
	"google.golang.org/protobuf/encoding/protodelim"
)

func TestWeeklyBaselineAndConfusionCounts(t *testing.T) {
	training := make([]bucket, 2*slotsPerWeek)
	for i := range training {
		training[i].peak = int64(100*(i%slotsPerWeek+1) + 20*(i/slotsPerWeek))
	}
	limits := thresholds(training, 5)
	for slot, limit := range limits {
		if want := float64(100*(slot+1) + 60); limit != want {
			t.Fatalf("slot %d: got %v, want %v", slot, limit, want)
		}
	}
	validation := []bucket{{161, true}, {260, true}, {361, false}, {459, false}}
	got := score(validation, limits)
	if want := (confusion{tp: 1, fp: 1, tn: 1, fn: 1}); got != want {
		t.Fatalf("got %+v, want %+v", got, want)
	}
	if got.metrics() != (metricSet{accuracy: .5, precision: .5, recall: .5}) {
		t.Fatal(got.metrics())
	}
	for c := 1; c <= maxConstant; c++ {
		for slot, limit := range thresholds(training[:slotsPerWeek], c) {
			if limit != float64(training[slot].peak) {
				t.Fatal("one week must have zero population variance")
			}
		}
	}
}

func TestTrainingBoundariesAndValidationIsolation(t *testing.T) {
	data := make([]bucket, totalWeeks*slotsPerWeek)
	for i := range data {
		data[i].peak = 100
	}
	data[0] = bucket{999999, true} // Excluded warmup.
	data[trainingEndBucket] = bucket{101, true}
	scores, err := evaluate(data)
	if err != nil {
		t.Fatal(err)
	}
	for _, duration := range scores {
		for _, counts := range duration {
			if counts != (confusion{tp: 1, tn: validationWeeks*slotsPerWeek - 1}) {
				t.Fatal(counts)
			}
		}
	}
	data[trainingEndBucket-1].anomaly = true
	if _, err := evaluate(data); err == nil {
		t.Fatal("accepted anomalous training")
	}
	if _, err := evaluate(data[:len(data)-1]); err == nil {
		t.Fatal("accepted truncated simulation")
	}
	data[trainingEndBucket-1].anomaly = false
	data[trainingEndBucket].anomaly = false
	if _, err := evaluate(data); err == nil {
		t.Fatal("accepted single-class validation")
	}
}

func TestSelectionUsesAccuracyAndAveragesSameConstant(t *testing.T) {
	runs := make([]sweep, 2)
	for seed := range runs {
		for i := range trainingWeeks {
			for c := range runs[seed][i] {
				runs[seed][i][c] = confusion{tn: 150, fn: 18}
			}
		}
	}
	for i := range trainingWeeks {
		runs[0][i][1] = confusion{tp: 1, fp: 1, tn: 165, fn: 1}
		runs[1][i][1] = confusion{tp: 9, fp: 1, tn: 157, fn: 1}
		runs[0][i][2], runs[1][i][2] = runs[0][i][1], runs[1][i][1] // Tie: smaller c wins.
	}
	for _, r := range summarize(runs) {
		if r.constant != 2 {
			t.Fatalf("selected c=%d", r.constant)
		}
		// Mean precision/recall = (1/2 + 9/10)/2, not pooled 10/12.
		want := [3]float64{166.0 / 168, .7, .7}
		for i := range want {
			if math.Abs(r.metrics.csvValues()[i]-want[i]) > 1e-12 {
				t.Fatalf("metrics=%v", r.metrics)
			}
		}
	}
}

func TestRollingLabelsAndInputValidation(t *testing.T) {
	start := time.Date(2026, 1, 5, 0, 0, 0, 0, time.UTC)
	entry := func(offset time.Duration, rows int64, attack bool, user string) *auditpb.AuditLogEntry {
		e := &auditpb.AuditLogEntry{}
		e.SetTimestampMs(start.Add(offset).UnixMilli())
		e.SetRowsReturned(rows)
		e.SetDbName("db")
		e.SetDbUser(user)
		e.SetIsGroundTruthAnomalous(attack)
		return e
	}
	aggregateEntries := func(entries ...*auditpb.AuditLogEntry) ([]bucket, error) {
		var input bytes.Buffer
		for _, e := range entries {
			if _, err := protodelim.MarshalTo(&input, e); err != nil {
				t.Fatal(err)
			}
		}
		return aggregate(&input, start)
	}
	data, err := aggregateEntries(entry(timeframe-time.Millisecond, 10, true, "u"), entry(timeframe, 2, false, "u"), entry(2*timeframe-time.Millisecond, 3, false, "u"), entry(2*timeframe, 4, false, "u"))
	if err != nil {
		t.Fatal(err)
	}
	if data[0] != (bucket{10, true}) || data[1] != (bucket{12, true}) || data[2] != (bucket{7, false}) {
		t.Fatalf("unexpected rolling values/labels: %v", data[:3])
	}
	for name, entries := range map[string][]*auditpb.AuditLogEntry{
		"empty":          nil,
		"unsorted":       {entry(time.Second, 1, false, "u"), entry(0, 1, false, "u")},
		"before start":   {entry(-time.Millisecond, 1, false, "u")},
		"after end":      {entry(totalWeeks*week, 1, false, "u")},
		"multiple users": {entry(0, 1, false, "u"), entry(time.Second, 1, false, "v")},
		"negative rows":  {entry(0, -1, false, "u")},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := aggregateEntries(entries...); err == nil {
				t.Fatal("accepted invalid input")
			}
		})
	}
	if _, err := aggregate(bytes.NewReader([]byte{0x80}), start); err == nil {
		t.Fatal("accepted truncated protobuf")
	}
}
