package service

import (
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"testing"
	"testing/quick"

	auditpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common/proto"
	modelpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common/proto"
)

// Feature: ngram-scoring-model, Property 6: Anomaly decision follows the
// calibrated threshold with strict-greater semantics.
//
// For any completed window, the finding is marked anomalous if and only if the
// window score is strictly greater than the calibrated threshold; a score
// exactly equal to the threshold is NOT anomalous. This mirrors the scorer's
// own rule `window.SetIsAnomalous(window.GetScore() > *s.threshold)` and
// FitNGramCalibration's documented "equality is not anomalous" boundary.
//
// Validates: Requirements 4.1, 4.2
//
// This file is intentionally self-contained apart from the shared package test
// helpers it reuses (histogramBaseline, calibrationReport, FitNGramCalibration).
// Every top-level symbol is prefixed with Boundary to avoid collisions with the
// other concurrently-authored test files in this package; it deliberately does
// NOT redeclare any of those shared helpers.

// boundaryAlphabet is a small action alphabet. A small alphabet relative to the
// window length means random streams complete windows frequently, so the
// scoring/decision path in Predict is exercised repeatedly rather than only the
// "still filling" path.
var boundaryAlphabet = []string{"A", "B", "C", "D"}

// boundaryBuildModelAndScorer constructs a VALID, CALIBRATED NGramScoringModel
// and an equivalent NGramWindowScorer from the SAME proto, so the model and the
// scorer agree on window length and threshold. The model drives Predict; the
// scorer is used independently to obtain the numeric window score for the same
// window the model just completed, which the property then compares against the
// finding's anomaly flag.
//
// The fixtures are built with the existing package helpers exactly as the other
// tests do: histogramBaseline over {"A","B"} (windowLength 2, maxN 1) and
// FitNGramCalibration fed a calibrationReport, so the model is construction
// valid and calibrated.
func boundaryBuildModelAndScorer(t *testing.T) (*NGramScoringModel, *NGramWindowScorer) {
	t.Helper()
	baseline := histogramBaseline(t, []string{"A", "B"}, 1, 2)
	// A spread of window scores so the picked threshold lands in the interior
	// of the observed scores; percentile 0.5 selects a mid-rank score.
	report := calibrationReport(8, 0, 4, 2)
	calibrated, err := FitNGramCalibration(baseline, report, 0.5)
	if err != nil {
		t.Fatalf("FitNGramCalibration: %v", err)
	}
	model, err := NewNGramScoringModel(calibrated)
	if err != nil {
		t.Fatalf("NewNGramScoringModel: %v", err)
	}
	scorer, err := NewNGramWindowScorer(calibrated)
	if err != nil {
		t.Fatalf("NewNGramWindowScorer: %v", err)
	}
	return model, scorer
}

// boundaryEntry builds a non-nil AuditLogEntry whose action is the alphabet
// symbol selected by action. Only non-empty actions are produced so that every
// entry advances the window buffer and windows actually complete.
func boundaryEntry(action int) *auditpb.AuditLogEntry {
	e := &auditpb.AuditLogEntry{}
	idx := ((action % len(boundaryAlphabet)) + len(boundaryAlphabet)) % len(boundaryAlphabet)
	e.SetNormalizedQueryTemplate(boundaryAlphabet[idx])
	return e
}

// boundaryStream is a randomly generated sequence of non-empty actions. Making
// it a quick.Generator lets testing/quick vary both the length (so multiple
// windows complete) and the per-position action symbol.
type boundaryStream struct {
	actions []int
}

// Generate implements quick.Generator for boundaryStream.
func (boundaryStream) Generate(rnd *rand.Rand, size int) reflect.Value {
	// Ensure the stream is long enough to complete several windows regardless of
	// the shrink size: windowLength is 2, so a handful of actions yields
	// multiple completions.
	n := rnd.Intn(size+1) + 4
	s := boundaryStream{actions: make([]int, n)}
	for i := range s.actions {
		s.actions[i] = rnd.Intn(len(boundaryAlphabet))
	}
	return reflect.ValueOf(s)
}

// TestBoundaryStrictGreaterAgainstScorer is the general property: over many
// random non-empty action streams, every completed-window finding's anomaly
// flag equals (scorer window score > scorer threshold). Because the scorer sets
// IsAnomalous with exactly that strict-greater comparison and the model reads
// that flag, the relationship must hold for strictly-greater, strictly-less,
// and equal cases alike — the equal case being covered by construction whenever
// a window's score coincides with the threshold.
//
// Validates: Requirements 4.1, 4.2
func TestBoundaryStrictGreaterAgainstScorer(t *testing.T) {
	model, scorer := boundaryBuildModelAndScorer(t)
	windowLength := scorer.WindowLength()
	threshold := scorer.Threshold()

	var lastFailure string
	property := func(s boundaryStream) bool {
		// Each property call starts from a clean buffer so window boundaries are
		// deterministic and independent of previous generated streams.
		model.ResetLiveCounts()

		var window []string
		for i, a := range s.actions {
			entry := boundaryEntry(a)
			action := boundaryAlphabet[((a%len(boundaryAlphabet))+len(boundaryAlphabet))%len(boundaryAlphabet)]
			window = append(window, action)

			finding := model.Predict(entry)
			if finding == nil {
				lastFailure = fmt.Sprintf("nil finding at index %d", i)
				return false
			}

			if len(window) < windowLength {
				// Still filling: the model must report not anomalous here, and no
				// window has completed to compare against the scorer yet.
				if finding.GetIsAnomalousPredict() {
					lastFailure = fmt.Sprintf("still-filling finding at index %d marked anomalous", i)
					return false
				}
				continue
			}

			// Window complete: independently score the same window via the public
			// Score path and compare the finding's decision to the strict-greater
			// relationship score > threshold.
			ws, err := scorer.Score(window)
			if err != nil {
				lastFailure = fmt.Sprintf("scorer.Score(%v) errored: %v", window, err)
				return false
			}
			score := ws.GetScore()
			wantAnomalous := score > threshold
			if finding.GetIsAnomalousPredict() != wantAnomalous {
				lastFailure = fmt.Sprintf(
					"window %v: finding.is_anomalous=%v but (score=%g > threshold=%g)=%v",
					window, finding.GetIsAnomalousPredict(), score, threshold, wantAnomalous)
				return false
			}
			// The scorer's own IsAnomalous flag must agree with the strict-greater
			// comparison too (single source of truth for the boundary).
			if ws.GetIsAnomalous() != wantAnomalous {
				lastFailure = fmt.Sprintf(
					"window %v: scorer.IsAnomalous=%v but (score=%g > threshold=%g)=%v",
					window, ws.GetIsAnomalous(), score, threshold, wantAnomalous)
				return false
			}

			// Tumbling reset: the next window starts fresh, matching the model's
			// own buffer reset on completion.
			window = window[:0]
		}
		return true
	}

	if err := quick.Check(property, &quick.Config{MaxCount: 200}); err != nil {
		if lastFailure != "" {
			t.Fatalf("strict-greater boundary property failed: %s (%v)", lastFailure, err)
		}
		t.Fatalf("strict-greater boundary property failed: %v", err)
	}
}

// TestBoundaryEqualityIsNotAnomalous pins the exact-equality case the random
// property can only cover opportunistically. It calibrates a model whose
// threshold is set exactly to the numeric score of a known window, then asserts
// that:
//
//   - a window whose score EQUALS the threshold is NOT anomalous (R4.2), and
//   - a window whose score is STRICTLY GREATER than the threshold IS anomalous
//     (R4.1).
//
// FitNGramCalibration picks the threshold directly from the supplied window
// scores (threshold = sortedScores[ceil(percentile*K)-1]). Feeding it the exact
// score that window ["A","A"] produces, with percentile 0.99 to select the
// maximum, makes the calibrated threshold equal that score. Everything flows
// through the public Score/Threshold/Predict surface — no unexported threshold
// field is touched.
//
// Validates: Requirements 4.1, 4.2
func TestBoundaryEqualityIsNotAnomalous(t *testing.T) {
	// For the {"A","B"} baseline with windowLength 2 and the default floor, the
	// scorer produces these deterministic scores (see TestNGramVarianceScores):
	//   ["A","B"] -> 0
	//   ["A","A"] -> sqrt(2/3)
	//   ["C","C"] -> sqrt(2)
	// Choose the equality anchor as the ["A","A"] score.
	equalScore := math.Sqrt(2.0 / 3.0) // == sqrt(2/3)

	baseline := histogramBaseline(t, []string{"A", "B"}, 1, 2)

	// Calibrate with a report whose maximum window score is exactly equalScore,
	// and a percentile (0.99) that selects the maximum, so the resulting
	// threshold equals equalScore.
	report := calibrationReport(0, equalScore)
	calibrated, err := FitNGramCalibration(baseline, report, 0.99)
	if err != nil {
		t.Fatalf("FitNGramCalibration: %v", err)
	}

	// Sanity check the calibrated threshold is exactly the anchor score, so the
	// equality assertion below genuinely exercises the score == threshold edge.
	gotThreshold := calibrated.GetNgramHistogram().GetCalibration().GetThreshold()
	if gotThreshold != equalScore {
		t.Fatalf("calibrated threshold = %g, want exactly the anchor score %g", gotThreshold, equalScore)
	}

	scorer, err := NewNGramWindowScorer(calibrated)
	if err != nil {
		t.Fatalf("NewNGramWindowScorer: %v", err)
	}
	// Confirm the scorer actually produces the anchor score for ["A","A"], so
	// the model-level equality assertion is meaningful.
	equalWindow, err := scorer.Score([]string{"A", "A"})
	if err != nil {
		t.Fatalf("scorer.Score([A A]): %v", err)
	}
	if equalWindow.GetScore() != scorer.Threshold() {
		t.Fatalf("expected ['A','A'] score %g to equal threshold %g",
			equalWindow.GetScore(), scorer.Threshold())
	}
	if equalWindow.GetIsAnomalous() {
		t.Fatal("scorer flagged a score exactly equal to the threshold as anomalous")
	}

	model, err := NewNGramScoringModel(calibrated)
	if err != nil {
		t.Fatalf("NewNGramScoringModel: %v", err)
	}

	// EQUALITY CASE (R4.2): drive a complete ["A","A"] window through the model.
	// The second entry completes the window, whose score equals the threshold,
	// so the finding must be NOT anomalous.
	model.ResetLiveCounts()
	_ = model.Predict(boundaryNamedEntry("A"))
	equalFinding := model.Predict(boundaryNamedEntry("A"))
	if equalFinding == nil {
		t.Fatal("equality-case finding was nil")
	}
	if equalFinding.GetIsAnomalousPredict() {
		t.Fatalf("score exactly equal to threshold (%g) must NOT be anomalous, but finding was anomalous",
			scorer.Threshold())
	}

	// STRICTLY-GREATER CASE (R4.1): drive a complete ["C","C"] window, whose
	// score sqrt(2) is strictly greater than the threshold sqrt(2/3), so the
	// finding must be anomalous.
	aboveWindow, err := scorer.Score([]string{"C", "C"})
	if err != nil {
		t.Fatalf("scorer.Score([C C]): %v", err)
	}
	if !(aboveWindow.GetScore() > scorer.Threshold()) {
		t.Fatalf("fixture invalid: ['C','C'] score %g is not > threshold %g",
			aboveWindow.GetScore(), scorer.Threshold())
	}
	model.ResetLiveCounts()
	_ = model.Predict(boundaryNamedEntry("C"))
	aboveFinding := model.Predict(boundaryNamedEntry("C"))
	if aboveFinding == nil {
		t.Fatal("strictly-greater-case finding was nil")
	}
	if !aboveFinding.GetIsAnomalousPredict() {
		t.Fatalf("score %g strictly greater than threshold %g must be anomalous, but finding was not",
			aboveWindow.GetScore(), scorer.Threshold())
	}
}

// boundaryNamedEntry builds a non-nil AuditLogEntry carrying the given action
// string directly (as opposed to an alphabet index), used by the focused
// equality/strictly-greater assertions.
func boundaryNamedEntry(action string) *auditpb.AuditLogEntry {
	e := &modelpb.AuditLogEntry{}
	e.SetNormalizedQueryTemplate(action)
	return e
}
