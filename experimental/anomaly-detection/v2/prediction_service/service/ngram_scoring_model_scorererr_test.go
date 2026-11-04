package service

import (
	"math"
	"testing"

	auditpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common/proto"
)

// This file is intentionally isolated from the other *_test.go files in this
// package (per the task's file-isolation rule). Every top-level symbol here is
// prefixed with ScorerErr. It reuses the shared package test helpers
// (histogramBaseline) without redefining them.

// scorerErrEntry builds an AuditLogEntry whose normalized query template is the
// given action, so common.Extract yields that action as the window element.
func scorerErrEntry(action string) *auditpb.AuditLogEntry {
	e := &auditpb.AuditLogEntry{}
	e.SetDbUser("u")
	e.SetDbName("db")
	e.SetQueryStatement("stmt")
	e.SetNormalizedQueryTemplate(action)
	return e
}

// TestScorerErrDegradesGracefully verifies the Predict degradation path when the
// NGramWindowScorer returns an error at window completion (Requirements 3.4 and
// 4.4): Predict must return a non-nil, NOT-anomalous finding with confidence
// notScoredConfidence (0.0), must clear the tumbling buffer, and must keep
// working on subsequent calls — all without panicking.
//
// HOW THE SCORER ERROR IS INDUCED (approach (b) from the task):
//
// The public construction path cannot build an error-inducing scorer, because
// resolvedScoringConfig rejects any non-finite or non-positive
// standard_deviation_floor_count before a scorer is ever returned. So this test
// first builds a fully valid scorer from the shared histogramBaseline helper
// (windowLength 2, maxN 1, real baseline), then — because the test is in-package
// (package service) — reaches into the scorer's unexported *config field and
// sets standard_deviation_floor_count to +Inf. That bypasses the constructor's
// finite guard but leaves the rest of the scorer valid.
//
// With an infinite floor, NGramWindowScorer.Score computes
// floor = standard_deviation_floor_count / positions = +Inf / 1 = +Inf, which
// trips its `!finite(floor)` guard and returns the deterministic
// "standard-deviation floor cannot be represented" error for ANY otherwise-valid
// length-2 window. Feeding exactly windowLength (2) non-empty actions through
// Predict completes a window and drives Score down that error path.
func TestScorerErrDegradesGracefully(t *testing.T) {
	baseline := histogramBaseline(t, []string{"A", "B"}, 1, 2)

	model, err := NewNGramScoringModel(baseline)
	if err != nil {
		t.Fatalf("constructing a valid ngram model must succeed, got: %v", err)
	}
	if model.windowLength != 2 {
		t.Fatalf("test assumes windowLength 2, got %d", model.windowLength)
	}

	// Induce the scorer error: set the resolved config's floor to +Inf so that
	// Score's floor computation becomes non-finite and errors on completion.
	model.scorer.config.SetStandardDeviationFloorCount(math.Inf(1))

	// Guard: the mutation must actually produce a Score error for a valid
	// length-2 window, otherwise the test would not exercise the error path.
	if _, scoreErr := model.scorer.Score([]string{"A", "B"}); scoreErr == nil {
		t.Fatal("precondition failed: crafted scorer did not error on a full window")
	}

	// First action only fills the buffer: a not-anomalous "still filling"
	// finding, buffer now holds 1 action, no scoring yet.
	first := model.Predict(scorerErrEntry("A"))
	if first == nil {
		t.Fatal("Predict returned nil for the first (buffering) action")
	}
	if first.GetIsAnomalousPredict() {
		t.Fatal("buffering finding must not be anomalous")
	}
	if len(model.buffer) != 1 {
		t.Fatalf("buffer should hold 1 action while filling, got %d", len(model.buffer))
	}

	// Second action completes the window (len == windowLength) and triggers the
	// scorer error path. Predict must degrade gracefully.
	completion := model.Predict(scorerErrEntry("B"))
	if completion == nil {
		t.Fatal("Predict returned nil on window completion with a scorer error")
	}
	if completion.GetIsAnomalousPredict() {
		t.Fatal("a scorer error must degrade to a NOT-anomalous finding")
	}
	if completion.GetConfidenceScore() != notScoredConfidence {
		t.Fatalf("scorer-error finding confidence = %v, want %v (0.0)",
			completion.GetConfidenceScore(), notScoredConfidence)
	}
	// The error path must clear the tumbling buffer.
	if len(model.buffer) != 0 {
		t.Fatalf("buffer must be empty after a window completion, got %d", len(model.buffer))
	}
	// The scorer error must be observable via the server seam (R4.4) but never
	// surface on the finding itself (R3.4).
	if model.lastScoreErr == nil {
		t.Fatal("expected the scorer error to be recorded for the server to log")
	}

	// State is not corrupted: subsequent Predict calls continue normally. The
	// next non-empty action begins a fresh window; the one after completes it,
	// again degrading gracefully (same induced error), proving the buffer reset
	// restored a usable model rather than leaving stale state behind.
	third := model.Predict(scorerErrEntry("A"))
	if third == nil {
		t.Fatal("Predict returned nil after recovering from a scorer error")
	}
	if len(model.buffer) != 1 {
		t.Fatalf("buffer should hold 1 action after starting a new window, got %d", len(model.buffer))
	}

	fourth := model.Predict(scorerErrEntry("B"))
	if fourth == nil {
		t.Fatal("Predict returned nil on the second window completion")
	}
	if fourth.GetIsAnomalousPredict() {
		t.Fatal("the second scorer-error completion must also be not anomalous")
	}
	if fourth.GetConfidenceScore() != notScoredConfidence {
		t.Fatalf("second scorer-error confidence = %v, want 0.0", fourth.GetConfidenceScore())
	}
	if len(model.buffer) != 0 {
		t.Fatalf("buffer must be empty after the second completion, got %d", len(model.buffer))
	}
}
