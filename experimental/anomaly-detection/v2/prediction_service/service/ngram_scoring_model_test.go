package service

import (
	"errors"
	"math"
	"strings"
	"testing"
	"testing/quick"

	"github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common"
	modelpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common/proto"
	"google.golang.org/protobuf/proto"
)

// Model is satisfied by *NGramScoringModel. The authoritative compile-time
// assertion lives in ngram_scoring_model.go (`var _ Model =
// (*NGramScoringModel)(nil)`); this differently named alias keeps the
// interface-satisfaction check visible from the test file without redeclaring
// the source-level assertion.
var _ Model = (*NGramScoringModel)(nil)

// TestNewNGramScoringModelWrongScenario verifies that passing a non-ngram
// scenario (Traffic, Volumetric, Temporal) — as well as a nil model and a model
// with no scenario set — returns a nil model and the scorer's wrong-scenario
// error, wrapping common.ErrInvalidArgument.
func TestNewNGramScoringModelWrongScenario(t *testing.T) {
	cases := []struct {
		name string
		pb   *modelpb.Model
	}{
		{
			name: "traffic",
			pb: func() *modelpb.Model {
				p := &modelpb.Model{}
				p.SetTrafficDeviation(&modelpb.TrafficDeviationScenario{})
				return p
			}(),
		},
		{
			name: "volumetric",
			pb: func() *modelpb.Model {
				p := &modelpb.Model{}
				p.SetVolumetricSpike(&modelpb.VolumetricSpikeScenario{})
				return p
			}(),
		},
		{
			name: "temporal",
			pb: func() *modelpb.Model {
				p := &modelpb.Model{}
				p.SetTemporalDeviation(&modelpb.TemporalDeviationScenario{})
				return p
			}(),
		},
		{
			name: "scenario_not_set",
			pb:   &modelpb.Model{},
		},
		{
			name: "nil",
			pb:   nil,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			model, err := NewNGramScoringModel(tc.pb)
			if model != nil {
				t.Fatalf("expected nil model for %s scenario, got %v", tc.name, model)
			}
			if err == nil {
				t.Fatalf("expected an error for %s scenario, got nil", tc.name)
			}
			if !errors.Is(err, common.ErrInvalidArgument) {
				t.Fatalf("error does not wrap ErrInvalidArgument: %v", err)
			}
			if !strings.Contains(err.Error(), "expected NgramHistogram scenario") {
				t.Fatalf("error does not name the expected scenario: %v", err)
			}
		})
	}
}

// TestNewNGramScoringModelWithoutCalibration documents the actual constructor
// behavior: an ngram model with no calibration constructs successfully. The
// scorer only treats calibration as a hard requirement once it is present, so
// the absence of calibration is not a constructor error.
func TestNewNGramScoringModelWithoutCalibration(t *testing.T) {
	baseline := histogramBaseline(t, []string{"A", "B"}, 1, 2)
	if baseline.GetNgramHistogram().GetCalibration() != nil {
		t.Fatal("fixture unexpectedly carried a calibration")
	}

	model, err := NewNGramScoringModel(baseline)
	if err != nil {
		t.Fatalf("construction without calibration must succeed, got error: %v", err)
	}
	if model == nil {
		t.Fatal("expected a non-nil model when construction succeeds")
	}
}

// TestNewNGramScoringModelObsoleteCalibration verifies that an ngram model whose
// present calibration is version-mismatched (obsolete) returns a nil model and
// the scorer's obsolete-calibration error, wrapping common.ErrInvalidArgument.
func TestNewNGramScoringModelObsoleteCalibration(t *testing.T) {
	baseline := histogramBaseline(t, []string{"A", "B"}, 1, 2)
	// Produce a valid, current calibration using the same helpers the scorer
	// tests use, then mutate only the scoring version so the calibration is
	// present but obsolete.
	calibrated, err := FitNGramCalibration(baseline, calibrationReport(0, 0), 0.99)
	if err != nil {
		t.Fatalf("failed to build a calibrated fixture: %v", err)
	}
	calibrated.GetNgramHistogram().GetCalibration().SetScoringVersion("js_distance_base2_mean_v1")

	model, err := NewNGramScoringModel(calibrated)
	if model != nil {
		t.Fatalf("expected nil model for obsolete calibration, got %v", model)
	}
	if err == nil {
		t.Fatal("expected an error for obsolete calibration, got nil")
	}
	if !errors.Is(err, common.ErrInvalidArgument) {
		t.Fatalf("error does not wrap ErrInvalidArgument: %v", err)
	}
	if !strings.Contains(err.Error(), "invalid or obsolete calibration") {
		t.Fatalf("error is not the obsolete-calibration error: %v", err)
	}
}

// generateDelegationProto maps an arbitrary seed to one of a diverse family of
// model protos spanning the construction outcomes the delegation property must
// cover: valid-calibrated, valid-uncalibrated, wrong-scenario, invalid-baseline,
// obsolete-calibration, and nil. Reusing the scorer-test helpers
// (histogramBaseline, calibrationReport, FitNGramCalibration) keeps the valid
// fixtures construction-valid, exactly as the existing scorer tests do.
//
// The seed also varies the baseline shape (action set, maxN, window length) so
// the property runs against structurally different histograms rather than a
// single fixed model. A description is returned purely for diagnostics.
func generateDelegationProto(t *testing.T, seed uint64) (pb *modelpb.Model, desc string) {
	t.Helper()

	// Vary the baseline shape deterministically from the seed.
	actionSets := [][]string{
		{"A", "B"},
		{"A", "B", "C"},
		{"X", "Y"},
		{"A", "A", "B", "B"},
	}
	actions := actionSets[(seed/7)%uint64(len(actionSets))]
	length := 2 + int((seed/3)%2) // 2 or 3
	maxN := 1 + int((seed/5)%2)   // 1 or 2
	if maxN > length {
		maxN = length
	}

	switch seed % 6 {
	case 0:
		// Wrong scenario (Traffic): scorer rejects with ErrInvalidArgument.
		p := &modelpb.Model{}
		p.SetTrafficDeviation(&modelpb.TrafficDeviationScenario{})
		return p, "wrong-scenario-traffic"
	case 1:
		// Scenario not set: scorer rejects with ErrInvalidArgument.
		return &modelpb.Model{}, "scenario-not-set"
	case 2:
		// Nil proto: scorer rejects with ErrInvalidArgument.
		return nil, "nil"
	case 3:
		// Valid, uncalibrated ngram baseline: scorer construction succeeds.
		return histogramBaseline(t, actions, maxN, length), "valid-uncalibrated"
	case 4:
		// Valid, calibrated ngram baseline: scorer construction succeeds. The
		// calibrationReport helper fixes windowLength=2, so calibrated fixtures
		// use the matching length-2 / maxN-1 baseline shape (as the existing
		// calibration tests do); the action set still varies.
		baseline := histogramBaseline(t, calibratedActions(seed), 1, 2)
		calibrated, err := FitNGramCalibration(baseline, calibrationReport(0, 0), 0.99)
		if err != nil {
			t.Fatalf("failed to build calibrated fixture: %v", err)
		}
		return calibrated, "valid-calibrated"
	default:
		// Obsolete calibration: present but version-mismatched, scorer rejects.
		baseline := histogramBaseline(t, calibratedActions(seed), 1, 2)
		calibrated, err := FitNGramCalibration(baseline, calibrationReport(0, 0), 0.99)
		if err != nil {
			t.Fatalf("failed to build calibrated fixture: %v", err)
		}
		calibrated.GetNgramHistogram().GetCalibration().SetScoringVersion("js_distance_base2_mean_v1")
		return calibrated, "obsolete-calibration"
	}
}

// calibratedActions picks a windowLength-2-compatible action set for the
// calibrated/obsolete fixtures, varying it deterministically from the seed.
func calibratedActions(seed uint64) []string {
	sets := [][]string{{"A", "B"}, {"X", "Y"}, {"A", "A", "B", "B"}}
	return sets[(seed/11)%uint64(len(sets))]
}

// TestNGramScoringModelConstructionDelegation is a property-based test for the
// delegation identity between NewNGramScoringModel and NewNGramWindowScorer.
//
// Feature: ngram-scoring-model, Property 1: Construction delegates to the scorer
// (success and error identity). For any *modelpb.Model value (valid or invalid,
// any scenario, with or without calibration), NewNGramScoringModel(pb) returns a
// non-nil model with a nil error if and only if NewNGramWindowScorer(pb) returns
// a non-nil scorer with a nil error; and when NewNGramWindowScorer(pb) returns an
// error, NewNGramScoringModel(pb) returns a nil model and that same error
// unchanged.
//
// Validates: Requirements 1.1, 1.2, 1.4, 1.5, 5.1.
func TestNGramScoringModelConstructionDelegation(t *testing.T) {
	property := func(seed uint64) bool {
		pb, desc := generateDelegationProto(t, seed)

		scorer, scorerErr := NewNGramWindowScorer(pb)
		model, modelErr := NewNGramScoringModel(pb)

		if scorerErr != nil {
			// Scorer failed: the model must return a nil model and the SAME
			// error, unchanged (same pointer identity via errors.Is against the
			// concrete error, and the sentinel must still be ErrInvalidArgument).
			if model != nil {
				t.Errorf("%s (seed=%d): scorer errored but model is non-nil", desc, seed)
				return false
			}
			if modelErr == nil {
				t.Errorf("%s (seed=%d): scorer errored but model error is nil", desc, seed)
				return false
			}
			// Same error unchanged: the model does no re-wrapping. Each
			// constructor call builds its own error value (fmt.Errorf), so the
			// two instances are not pointer-identical; "unchanged" is verified by
			// matching the wrapped sentinel and the rendered message exactly.
			if modelErr.Error() != scorerErr.Error() {
				t.Errorf("%s (seed=%d): model error message differs from scorer error: model=%q scorer=%q",
					desc, seed, modelErr.Error(), scorerErr.Error())
				return false
			}
			if errors.Is(scorerErr, common.ErrInvalidArgument) != errors.Is(modelErr, common.ErrInvalidArgument) {
				t.Errorf("%s (seed=%d): model/scorer disagree on ErrInvalidArgument sentinel: model=%v scorer=%v",
					desc, seed, modelErr, scorerErr)
				return false
			}
			if !errors.Is(modelErr, common.ErrInvalidArgument) {
				t.Errorf("%s (seed=%d): model error does not wrap ErrInvalidArgument: %v",
					desc, seed, modelErr)
				return false
			}
			return true
		}

		// Scorer succeeded: the model must also succeed with a non-nil model and
		// a nil error, and the cached windowLength must equal the scorer's.
		if scorer == nil {
			t.Errorf("%s (seed=%d): scorer returned nil scorer with nil error", desc, seed)
			return false
		}
		if modelErr != nil {
			t.Errorf("%s (seed=%d): scorer succeeded but model errored: %v", desc, seed, modelErr)
			return false
		}
		if model == nil {
			t.Errorf("%s (seed=%d): scorer succeeded but model is nil", desc, seed)
			return false
		}
		if model.windowLength != scorer.WindowLength() {
			t.Errorf("%s (seed=%d): cached windowLength=%d != scorer.WindowLength()=%d",
				desc, seed, model.windowLength, scorer.WindowLength())
			return false
		}
		return true
	}

	if err := quick.Check(property, &quick.Config{MaxCount: 200}); err != nil {
		t.Fatalf("construction delegation property failed: %v", err)
	}
}

// TestNGramScoringModelConstructorImmutability is a property-based test
// asserting that constructing an NGramScoringModel never mutates the caller's
// input proto, regardless of whether construction succeeds or fails.
//
// Feature: ngram-scoring-model, Property 2: Constructor leaves the supplied
// proto unchanged. For any *modelpb.Model value, calling NewNGramScoringModel(pb)
// (whether it succeeds or fails) leaves pb deep-equal to a snapshot taken
// immediately before the call.
//
// The property deep-clones the generated proto with proto.Clone before
// construction, invokes NewNGramScoringModel, then asserts the original proto is
// proto.Equal to its pre-construction clone. It reuses generateDelegationProto
// for a diverse family of valid and invalid inputs, but skips the nil case
// (seed%6==2) because proto.Clone/proto.Equal over nil is trivial and a nil
// input cannot be mutated.
//
// Validates: Requirements 1.6, 5.4.
func TestNGramScoringModelConstructorImmutability(t *testing.T) {
	property := func(seed uint64) bool {
		// Skip the nil case: proto.Clone(nil)/proto.Equal(nil, nil) is trivial
		// and a nil proto has no contents to mutate.
		if seed%6 == 2 {
			return true
		}

		pb, desc := generateDelegationProto(t, seed)
		if pb == nil {
			// Defensive guard in case the generator ever returns nil outside the
			// skipped case; nothing to assert about an unmutatable nil input.
			return true
		}

		// Snapshot the input before construction, then construct and compare.
		snapshot := proto.Clone(pb)
		_, _ = NewNGramScoringModel(pb)

		if !proto.Equal(pb, snapshot) {
			t.Errorf("%s (seed=%d): constructor mutated the input proto\n before: %v\n  after: %v",
				desc, seed, snapshot, pb)
			return false
		}
		return true
	}

	if err := quick.Check(property, &quick.Config{MaxCount: 200}); err != nil {
		t.Fatalf("constructor immutability property failed: %v", err)
	}
}

// TestConfidenceTransformProperties is a property-based test for the
// windowConfidence transform.
//
// Feature: ngram-scoring-model, Property 8: Confidence is bounded,
// deterministic, and monotonic in distance from threshold. For any window score
// and calibrated threshold, the derived confidence lies in the inclusive range
// [0.0, 1.0]; identical (score, threshold) inputs always yield an identical
// confidence; and for a fixed threshold, if |scoreA - threshold| >=
// |scoreB - threshold| then confidence(scoreA) >= confidence(scoreB).
//
// Validates: Requirements 4.3.
func TestConfidenceTransformProperties(t *testing.T) {
	// BOUNDED: for any finite score/threshold the result is within [0.0, 1.0].
	bounded := func(score, threshold float64) bool {
		if !confidenceInputsFinite(score, threshold) {
			return true // non-finite inputs are covered by the defensive examples below.
		}
		got := windowConfidence(score, threshold)
		return got >= 0.0 && got <= 1.0
	}
	if err := quick.Check(bounded, &quick.Config{MaxCount: 200}); err != nil {
		t.Fatalf("confidence BOUNDED property failed: %v", err)
	}

	// DETERMINISTIC: two calls with identical inputs yield the identical value.
	deterministic := func(score, threshold float64) bool {
		if !confidenceInputsFinite(score, threshold) {
			return true
		}
		return windowConfidence(score, threshold) == windowConfidence(score, threshold)
	}
	if err := quick.Check(deterministic, &quick.Config{MaxCount: 200}); err != nil {
		t.Fatalf("confidence DETERMINISTIC property failed: %v", err)
	}

	// MONOTONIC: for a fixed threshold, a larger distance from the threshold
	// never produces a smaller confidence. Construct scoreA = threshold + |dA|
	// and scoreB = threshold + |dB| so the two scores sit at known distances
	// |dA| and |dB| from the threshold, then assert the ordering of the
	// confidences matches the ordering of those distances.
	monotonic := func(threshold, dA, dB float64) bool {
		if !confidenceInputsFinite(threshold, dA, dB) {
			return true
		}
		distA := math.Abs(dA)
		distB := math.Abs(dB)
		scoreA := threshold + distA
		scoreB := threshold + distB
		// Skip cases where constructing the score overflowed to a non-finite
		// value: such inputs are outside the "any finite score/threshold"
		// domain this property reasons about.
		if !confidenceInputsFinite(scoreA, scoreB) {
			return true
		}
		confA := windowConfidence(scoreA, threshold)
		confB := windowConfidence(scoreB, threshold)
		if distA <= distB {
			return confA <= confB
		}
		return confA >= confB
	}
	if err := quick.Check(monotonic, &quick.Config{MaxCount: 200}); err != nil {
		t.Fatalf("confidence MONOTONIC property failed: %v", err)
	}

	// Defensive example assertions for the non-finite inputs the properties
	// above skip: a NaN distance yields 0.0 and a +Inf distance yields 1.0.
	if got := windowConfidence(math.NaN(), 0.0); got != 0.0 {
		t.Errorf("windowConfidence(NaN, 0.0) = %v, want 0.0", got)
	}
	if got := windowConfidence(0.0, math.NaN()); got != 0.0 {
		t.Errorf("windowConfidence(0.0, NaN) = %v, want 0.0", got)
	}
	if got := windowConfidence(math.Inf(1), 0.0); got != 1.0 {
		t.Errorf("windowConfidence(+Inf, 0.0) = %v, want 1.0", got)
	}
	if got := windowConfidence(math.Inf(-1), 0.0); got != 1.0 {
		t.Errorf("windowConfidence(-Inf, 0.0) = %v, want 1.0", got)
	}
}

// confidenceInputsFinite reports whether every supplied value is a finite
// float64 (neither NaN nor an infinity). The confidence properties reason about
// the "any finite score/threshold" domain; non-finite inputs are covered
// separately by the defensive example assertions.
func confidenceInputsFinite(vs ...float64) bool {
	for _, v := range vs {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return false
		}
	}
	return true
}
