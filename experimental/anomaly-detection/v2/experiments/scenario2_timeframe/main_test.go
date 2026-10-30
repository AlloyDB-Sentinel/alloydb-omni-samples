package main

import "testing"

func TestEvaluateBucketsScoresOneObservationPerTimeframeBucket(t *testing.T) {
	originalFrames, originalConstants := frames, constants
	frames = []timeframe{{name: "test", duration: 1}}
	constants = []float64{1}
	t.Cleanup(func() {
		frames, constants = originalFrames, originalConstants
	})

	buckets := []bucketObservation{
		{maxRowsReturned: 0},                   // training: empty bucket
		{maxRowsReturned: 10},                  // training
		{maxRowsReturned: 20, anomalous: true}, // validation TP
		{maxRowsReturned: 5, anomalous: true},  // validation FN
		{maxRowsReturned: 20},                  // test FP
		{},                                     // test TN, empty bucket
	}
	baseline, validation, testing := evaluateBuckets(buckets, 2, 2)

	if baseline.n != 2 || baseline.mean != 5 || baseline.stddev() != 5 {
		t.Fatalf("training baseline = (n=%d, mean=%v, stddev=%v), want (2, 5, 5)", baseline.n, baseline.mean, baseline.stddev())
	}
	if got, want := validation[0], (confusion{tp: 1, fn: 1}); got != want {
		t.Errorf("validation confusion = %+v, want %+v", got, want)
	}
	if got, want := testing[0], (confusion{fp: 1, tn: 1}); got != want {
		t.Errorf("test confusion = %+v, want %+v", got, want)
	}
}
