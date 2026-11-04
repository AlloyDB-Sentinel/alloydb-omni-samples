package service

import (
	"math"
	"testing"
	"testing/quick"

	auditpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common/proto"
)

// tumblingAlphabet is a small set of non-empty actions the generated streams
// draw from. A deliberately small alphabet (mixing calibrated-known categories
// "A"/"B" with novel ones "C"/"D") makes it likely that some tumbling windows
// score as anomalous and others as normal, so the equivalence check exercises
// both branches of the scorer's is_anomalous decision.
var tumblingAlphabet = []string{"A", "B", "C", "D"}

// tumblingEntry builds an AuditLogEntry whose action (normalized query template)
// is the given non-empty string. The action is the only field the scorer and
// the model consume for window scoring.
func tumblingEntry(action string) *auditpb.AuditLogEntry {
	e := &auditpb.AuditLogEntry{}
	e.SetNormalizedQueryTemplate(action)
	return e
}

// tumblingModelAndScorer builds a calibrated NGramScoringModel and an equivalent
// NGramWindowScorer from the SAME model proto, so both share an identical
// windowLength and calibrated threshold. The model is the system under test; the
// scorer is the oracle used to independently score each tumbling window.
//
// The proto reuses the shared scorer-test helpers (histogramBaseline,
// calibrationReport, FitNGramCalibration), so windowLength is fixed at 2 — small
// enough that short generated streams complete multiple windows.
func tumblingModelAndScorer(t *testing.T) (*NGramScoringModel, *NGramWindowScorer) {
	t.Helper()
	baseline := histogramBaseline(t, []string{"A", "B"}, 1, 2)
	calibrated, err := FitNGramCalibration(baseline, calibrationReport(0, 1), 0.5)
	if err != nil {
		t.Fatalf("failed to build calibrated fixture: %v", err)
	}
	model, err := NewNGramScoringModel(calibrated)
	if err != nil {
		t.Fatalf("failed to construct model: %v", err)
	}
	scorer, err := NewNGramWindowScorer(calibrated)
	if err != nil {
		t.Fatalf("failed to construct oracle scorer: %v", err)
	}
	if model.windowLength != scorer.WindowLength() {
		t.Fatalf("model/scorer windowLength mismatch: %d != %d", model.windowLength, scorer.WindowLength())
	}
	if !scorer.HasThreshold() {
		t.Fatalf("oracle scorer is not calibrated; cannot compare anomaly decisions")
	}
	return model, scorer
}

// tumblingStream maps a quick-generated seed and length byte to a stream of
// non-empty actions drawn from tumblingAlphabet. Deriving the stream from the
// generated integers (rather than a []string generator) keeps every action
// non-empty and within the small alphabet, which is the input domain Property 5
// reasons about.
func tumblingStream(seed uint64, lengthByte uint8) []string {
	// Length in [1, 12]: enough to complete several windows of length 2 while
	// also exercising a trailing, not-yet-complete action.
	n := 1 + int(lengthByte)%12
	stream := make([]string, n)
	s := seed
	for i := 0; i < n; i++ {
		// Simple LCG step so successive actions are varied yet deterministic.
		s = s*6364136223846793005 + 1442695040888963407
		stream[i] = tumblingAlphabet[int(s>>33)%len(tumblingAlphabet)]
	}
	return stream
}

// TestTumblingEquivalence is a property-based test asserting that the model's
// per-entry buffering reproduces exactly the scorer's tumbling-window scoring.
//
// Feature: ngram-scoring-model, Property 5: Tumbling-window equivalence to the
// scorer. For any stream of N non-empty actions and window length L =
// WindowLength(), window completions occur exactly on the entries at indices
// L-1, 2L-1, 3L-1, … (one per non-overlapping group of L actions), and the
// finding produced at the k-th completion has an is_anomalous_predict flag equal
// to scorer.Score(actions[k*L : (k+1)*L]).GetIsAnomalous() and a confidence_score
// derived solely from that same window's score; after each completion the buffer
// is empty.
//
// Validates: Requirements 3.1, 3.3, 3.5, 4.1.
func TestTumblingEquivalence(t *testing.T) {
	model, scorer := tumblingModelAndScorer(t)
	L := scorer.WindowLength()

	property := func(seed uint64, lengthByte uint8) bool {
		stream := tumblingStream(seed, lengthByte)

		// Reset the model so each generated stream starts with an empty buffer,
		// making its windows independent of previous iterations.
		model.ResetLiveCounts()

		for i, action := range stream {
			finding := model.Predict(tumblingEntry(action))
			if finding == nil {
				t.Errorf("seed=%d i=%d: Predict returned nil finding", seed, i)
				return false
			}

			// A window completes exactly when the 1-based index i+1 is a
			// multiple of L, i.e. at indices L-1, 2L-1, 3L-1, ...
			completed := (i+1)%L == 0
			if !completed {
				// Still-filling entries must be not anomalous (R3.2) and must
				// leave fewer than L actions buffered.
				if finding.GetIsAnomalousPredict() {
					t.Errorf("seed=%d i=%d: still-filling entry marked anomalous", seed, i)
					return false
				}
				if len(model.buffer) >= L {
					t.Errorf("seed=%d i=%d: buffer=%d not < L=%d while filling", seed, i, len(model.buffer), L)
					return false
				}
				continue
			}

			// Window completed at this entry. Compute k (0-based window index)
			// and independently score the exact tumbling slice with the oracle.
			k := (i + 1) / L
			windowSlice := stream[(k-1)*L : k*L]
			oracle, err := scorer.Score(windowSlice)
			if err != nil {
				t.Errorf("seed=%d i=%d: oracle scorer.Score failed: %v", seed, i, err)
				return false
			}

			// Anomaly decision must match the oracle's GetIsAnomalous() for that
			// same non-overlapping window (R3.3, R4.1).
			if finding.GetIsAnomalousPredict() != oracle.GetIsAnomalous() {
				t.Errorf("seed=%d i=%d window=%d %v: model anomalous=%v != oracle=%v",
					seed, i, k-1, windowSlice, finding.GetIsAnomalousPredict(), oracle.GetIsAnomalous())
				return false
			}

			// Confidence is derived solely from that window's score and the
			// calibrated threshold.
			wantConfidence := windowConfidence(oracle.GetScore(), scorer.Threshold())
			if math.Abs(finding.GetConfidenceScore()-wantConfidence) > 1e-12 {
				t.Errorf("seed=%d i=%d window=%d %v: confidence=%v != want=%v",
					seed, i, k-1, windowSlice, finding.GetConfidenceScore(), wantConfidence)
				return false
			}

			// Tumbling reset: after a completion the buffer is empty (R3.5), so
			// the next L actions form the next non-overlapping window.
			if len(model.buffer) != 0 {
				t.Errorf("seed=%d i=%d: buffer not cleared after completion, len=%d", seed, i, len(model.buffer))
				return false
			}
		}
		return true
	}

	if err := quick.Check(property, &quick.Config{MaxCount: 200}); err != nil {
		t.Fatalf("tumbling-window equivalence property failed: %v", err)
	}
}
