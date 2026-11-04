package service

import (
	"fmt"
	"math/rand"
	"reflect"
	"testing"
	"testing/quick"

	auditpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common/proto"
)

// Feature: ngram-scoring-model, Property 3: Predict is total and non-panicking.
//
// For any finite sequence of inputs to Predict — including nil entries, entries
// with empty actions, and entries with arbitrary action strings, in any order —
// every call returns a non-nil AnomalyFinding and no call panics.
//
// Validates: Requirements 2.1, 2.4, 2.5, 3.5
//
// This file is intentionally self-contained apart from the shared package test
// helpers it reuses (histogramBaseline, scoringConfig, calibrationReport,
// FitNGramCalibration). Every top-level symbol is prefixed with PredictTotal to
// avoid collisions with other concurrently-authored test files in this package.

// predictTotalAlphabet is a small set of action strings. Keeping the alphabet
// small (relative to the window length) makes it likely that windows actually
// complete across a random stream, so the scoring/tumbling path in Predict is
// exercised, not just the "still filling" path.
var predictTotalAlphabet = []string{"A", "B", "C", "D"}

// predictTotalBuildCalibratedModel constructs a VALID, CALIBRATED
// NGramScoringModel via NewNGramScoringModel, reusing the existing package test
// helpers exactly as the scorer tests do: histogramBaseline builds a valid
// baseline over the alphabet and FitNGramCalibration (fed a calibrationReport)
// attaches a version-matching calibration. windowLength is 2 (the length baked
// into calibrationReport / histogramBaseline below).
func predictTotalBuildCalibratedModel(t *testing.T) *NGramScoringModel {
	t.Helper()
	baseline := histogramBaseline(t, []string{"A", "B"}, 1, 2)
	report := calibrationReport(8, 0, 4, 2)
	calibrated, err := FitNGramCalibration(baseline, report, 0.5)
	if err != nil {
		t.Fatalf("FitNGramCalibration: %v", err)
	}
	model, err := NewNGramScoringModel(calibrated)
	if err != nil {
		t.Fatalf("NewNGramScoringModel (calibrated): %v", err)
	}
	return model
}

// predictTotalBuildUncalibratedModel constructs a VALID but UNCALIBRATED
// NGramScoringModel (histogramBaseline without a calibration). Predict must
// remain total for this model too: when a window completes the scorer has no
// threshold, and Predict must still return a non-nil finding without panicking.
func predictTotalBuildUncalibratedModel(t *testing.T) *NGramScoringModel {
	t.Helper()
	baseline := histogramBaseline(t, []string{"A", "B"}, 1, 2)
	model, err := NewNGramScoringModel(baseline)
	if err != nil {
		t.Fatalf("NewNGramScoringModel (uncalibrated): %v", err)
	}
	return model
}

// predictTotalMakeEntry maps a small integer "kind" to one of the input shapes
// the property must cover:
//
//	kind%3 == 0 -> nil entry
//	kind%3 == 1 -> non-nil entry with an empty NormalizedQueryTemplate
//	kind%3 == 2 -> non-nil entry with an action drawn from predictTotalAlphabet
//
// action selects which alphabet symbol is used for the kind==2 case.
func predictTotalMakeEntry(kind, action int) *auditpb.AuditLogEntry {
	switch ((kind % 3) + 3) % 3 {
	case 0:
		return nil
	case 1:
		e := &auditpb.AuditLogEntry{}
		e.SetNormalizedQueryTemplate("")
		return e
	default:
		e := &auditpb.AuditLogEntry{}
		idx := ((action % len(predictTotalAlphabet)) + len(predictTotalAlphabet)) % len(predictTotalAlphabet)
		e.SetNormalizedQueryTemplate(predictTotalAlphabet[idx])
		return e
	}
}

// predictTotalStream is a randomly generated input sequence. Implementing
// quick.Generator lets testing/quick vary both the length of the stream and the
// per-entry kind/action, so the generated sequences mix nil entries,
// empty-action entries, and arbitrary short action strings, and span streams
// shorter and longer than the window length.
type predictTotalStream struct {
	kinds   []int
	actions []int
}

// Generate implements quick.Generator for predictTotalStream.
func (predictTotalStream) Generate(rnd *rand.Rand, size int) reflect.Value {
	// Build a stream whose length can be well above the window length so that
	// repeated calls span multiple completed windows. size is the quick-provided
	// scale; +1 guarantees at least one entry.
	n := rnd.Intn(size + 1)
	// Bias toward longer streams some of the time to make multi-window coverage
	// more likely regardless of the shrink size.
	if n < 6 {
		n += rnd.Intn(8)
	}
	s := predictTotalStream{
		kinds:   make([]int, n),
		actions: make([]int, n),
	}
	for i := 0; i < n; i++ {
		s.kinds[i] = rnd.Intn(3)
		s.actions[i] = rnd.Intn(len(predictTotalAlphabet))
	}
	return reflect.ValueOf(s)
}

// TestPredictTotalNeverPanics drives both a calibrated and an uncalibrated model
// with randomly generated input streams and asserts every Predict call returns a
// non-nil *AnomalyFinding and never panics.
func TestPredictTotalNeverPanics(t *testing.T) {
	calibrated := predictTotalBuildCalibratedModel(t)
	uncalibrated := predictTotalBuildUncalibratedModel(t)

	// predictTotalRun feeds a stream through one model in order and reports an
	// error string on the first violation (nil finding) or recovered panic. A
	// recover produces a clearer failure message than the raw panic unwinding
	// through quick.
	predictTotalRun := func(model *NGramScoringModel, s predictTotalStream) (ok bool, failure string) {
		defer func() {
			if r := recover(); r != nil {
				ok = false
				failure = fmt.Sprintf("Predict panicked: %v", r)
			}
		}()
		for i := range s.kinds {
			entry := predictTotalMakeEntry(s.kinds[i], s.actions[i])
			finding := model.Predict(entry)
			if finding == nil {
				return false, fmt.Sprintf("Predict returned nil finding at index %d (kind=%d action=%d)", i, s.kinds[i], s.actions[i])
			}
		}
		return true, ""
	}

	var lastFailure string
	property := func(s predictTotalStream) bool {
		if ok, failure := predictTotalRun(calibrated, s); !ok {
			lastFailure = "calibrated model: " + failure
			return false
		}
		if ok, failure := predictTotalRun(uncalibrated, s); !ok {
			lastFailure = "uncalibrated model: " + failure
			return false
		}
		return true
	}

	cfg := &quick.Config{MaxCount: 200}
	if err := quick.Check(property, cfg); err != nil {
		if lastFailure != "" {
			t.Fatalf("Predict totality property failed: %s (%v)", lastFailure, err)
		}
		t.Fatalf("Predict totality property failed: %v", err)
	}
}
