package main

import (
	"bytes"
	"math"
	"testing"
	"time"

	auditpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common/proto"
	"google.golang.org/protobuf/encoding/protodelim"
)

func TestLimitsForLearnsPerWeeklySlot(t *testing.T) {
	training := make([]int64, 2*slotsPerWeek)
	for i := range training {
		training[i] = int64(100*(i%slotsPerWeek+1) + 20*(i/slotsPerWeek))
	}
	for slot, limit := range limitsFor(training, 5) {
		if want := float64(100*(slot+1) + 60); math.Abs(limit-want) > 1e-9 {
			t.Fatalf("slot %d: got %v, want %v", slot, limit, want)
		}
	}
}

func TestScoreSubsetUsesORAndAND(t *testing.T) {
	labels := []bool{true, true, false, false}
	a := make([]int64, len(labels)*slotsPerBucket)
	b := make([]int64, len(labels)*slotsPerBucket)
	a[0] = 10
	b[slotsPerBucket] = 10
	data := dataset{peaks: [][]int64{a, b}, labels: labels}
	limits := make([][slotsPerWeek]float64, 2)
	for i := range limits {
		for j := range limits[i] {
			limits[i][j] = 5
		}
	}

	if got, want := scoreFrame(a, labels, limits[0], 0, len(labels)), (confusion{tp: 1, fn: 1, tn: 2}); got != want {
		t.Fatalf("single frame = %+v, want %+v", got, want)
	}
	if got, want := scoreSubset(data, []int{0, 1}, limits, orCombination, 0, len(labels)), (confusion{tp: 2, tn: 2}); got != want {
		t.Fatalf("OR subset = %+v, want %+v", got, want)
	}
	if got, want := scoreSubset(data, []int{0, 1}, limits, andCombination, 0, len(labels)), (confusion{tn: 2, fn: 2}); got != want {
		t.Fatalf("AND subset = %+v, want %+v", got, want)
	}

	// Both members must fire in the same reporting bucket for AND; their
	// exceedances need not occur in the same hour, as in the original protocol.
	b[slotsPerBucket] = 0
	b[1] = 10
	data.labels = []bool{true, false}
	if got, want := scoreSubset(data, []int{0, 1}, limits, andCombination, 0, 2), (confusion{tp: 1, tn: 1}); got != want {
		t.Fatalf("AND joint hit = %+v, want %+v", got, want)
	}
}

func TestHourlyBaselineSeparatesQuietHourFromBusyHour(t *testing.T) {
	training := make([]int64, 2*slotsPerWeek)
	for i := range training {
		training[i] = 100
		switch i % slotsPerWeek {
		case 5: // Monday 05:00
			training[i] = 500
		case 7: // Monday 07:00, in the same reporting bucket
			training[i] = 2000
		case 24 + 5: // Tuesday 05:00
			training[i] = 5000
		}
	}
	limits := limitsFor(training, 3)
	for _, test := range []struct {
		name string
		hour int
		want bool
	}{
		{"quiet Monday hour", 5, true},
		{"busy Monday hour", 7, false},
		{"same hour different weekday", 24 + 5, false},
		{"following Monday", slotsPerWeek + 5, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			peaks := make([]int64, 2*slotsPerWeek)
			peaks[test.hour] = 1750
			if got := frameExceedsLimit(peaks, limits, test.hour/slotsPerBucket); got != test.want {
				t.Fatalf("hour %d: got %v, want %v", test.hour, got, test.want)
			}
		})
	}
}

func TestAggregateKeepsWindowsAcrossHourlyAndReportingBoundaries(t *testing.T) {
	start := time.Date(2026, time.January, 5, 0, 0, 0, 0, time.UTC)
	var input bytes.Buffer
	for _, record := range []struct {
		after  time.Duration
		rows   int64
		attack bool
	}{
		{59 * time.Minute, 10, true},
		{time.Hour, 20, false},
		{4 * time.Hour, 30, false},
		{8 * time.Hour, 40, false},
	} {
		entry := &auditpb.AuditLogEntry{}
		entry.SetDbName("db")
		entry.SetDbUser("user")
		entry.SetTimestampMs(start.Add(record.after).UnixMilli())
		entry.SetRowsReturned(record.rows)
		entry.SetIsGroundTruthAnomalous(record.attack)
		if _, err := protodelim.MarshalTo(&input, entry); err != nil {
			t.Fatal(err)
		}
	}
	data, err := aggregate(&input, start)
	if err != nil {
		t.Fatal(err)
	}
	for fi, frame := range frames {
		switch frame.name {
		case "1m":
			if data.peaks[fi][0] != 10 || data.peaks[fi][1] != 20 {
				t.Fatalf("one-minute peaks cross the hourly boundary incorrectly: %v", data.peaks[fi][:2])
			}
		case "4h":
			if data.peaks[fi][1] != 30 || data.peaks[fi][4] != 60 || data.peaks[fi][8] != 40 {
				t.Fatalf("four-hour window reset or expired incorrectly: %v", data.peaks[fi][:9])
			}
		}
	}
	if !data.labels[0] || !data.labels[1] || data.labels[2] {
		t.Fatalf("four-hour reporting/spillover labels changed: %v", data.labels[:3])
	}
}

func TestTrainingPeaksExcludeWarmupAndValidation(t *testing.T) {
	peaks := make([]int64, totalBuckets*slotsPerBucket)
	first, end := trainingStart*slotsPerBucket, trainingEnd*slotsPerBucket
	peaks[first-1], peaks[first], peaks[end-1], peaks[end] = 999, 11, 22, 888
	got := trainingPeaks(dataset{peaks: [][]int64{peaks}}, 0)
	if len(got) != trainingWeeks*slotsPerWeek || got[0] != 11 || got[len(got)-1] != 22 {
		t.Fatalf("wrong training slice: length %d, first %d, last %d", len(got), got[0], got[len(got)-1])
	}
}

func TestSubsetSpecsCoversAllNonEmptySubsets(t *testing.T) {
	specs := subsetSpecs()
	if got, want := len(specs), 2*((1<<len(frames))-1); got != want {
		t.Fatalf("specs = %d, want %d", got, want)
	}

	type key struct {
		subset string
		comb   combiner
	}
	seen := make(map[key]int)
	bySize := make(map[int]int)
	for _, spec := range specs {
		if len(spec.members) == 0 {
			t.Fatal("empty subset")
		}
		seen[key{subsetName(spec.members), spec.combination}]++
		bySize[len(spec.members)]++
	}
	for mask := 1; mask < 1<<len(frames); mask++ {
		var members []int
		for fi := range frames {
			if mask&(1<<fi) != 0 {
				members = append(members, fi)
			}
		}
		for _, comb := range []combiner{orCombination, andCombination} {
			if got := seen[key{subsetName(members), comb}]; got != 1 {
				t.Fatalf("%s/%s seen %d times, want 1", subsetName(members), comb, got)
			}
		}
	}
	for size, want := range map[int]int{1: 14, 2: 42, 3: 70, 4: 70, 5: 42, 6: 14, 7: 2} {
		if got := bySize[size]; got != want {
			t.Fatalf("size %d: got %d candidates, want %d", size, got, want)
		}
	}

	if specs[0].combination != orCombination || len(specs[0].members) != 1 || specs[0].members[0] != 0 {
		t.Fatalf("first spec = %+v, want 1m/OR", specs[0])
	}
	if last := specs[len(specs)-1]; last.combination != andCombination || len(last.members) != len(frames) {
		t.Fatalf("last spec = %+v, want all frames/AND", last)
	}
}

func TestBetterPrefersF1ThenPrecisionThenAccuracy(t *testing.T) {
	base := metricSet{accuracy: .9, precision: .8, recall: .7, f1: .75}
	if !better(metricSet{accuracy: .5, precision: .1, recall: .9, f1: .8}, base) {
		t.Fatal("higher F1 must win")
	}
	if !better(metricSet{accuracy: .5, precision: .9, recall: .5, f1: .75}, base) {
		t.Fatal("equal F1 must prefer precision")
	}
	if !better(metricSet{accuracy: .95, precision: .8, recall: .5, f1: .75}, base) {
		t.Fatal("equal F1 and precision must prefer accuracy")
	}
	if better(base, base) {
		t.Fatal("equal metrics must keep the earlier candidate")
	}
}
