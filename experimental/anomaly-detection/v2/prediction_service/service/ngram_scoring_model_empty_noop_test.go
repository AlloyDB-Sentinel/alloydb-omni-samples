package service

import (
	"testing"
	"testing/quick"

	auditpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common/proto"
	findingpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common/proto"
)

// EmptyNoopAlphabet is a small action alphabet so windows (windowLength 2)
// complete frequently over short generated streams. The actions overlap the
// calibrated baseline's known categories ("A", "B") and also include a novel
// one ("C") so that both anomalous and non-anomalous windows arise.
var EmptyNoopAlphabet = []string{"A", "B", "C"}

// emptyNoopNewCalibratedModel builds a fresh, construction-valid calibrated
// NGramScoringModel the same way the existing scorer/model tests do:
// histogramBaseline -> FitNGramCalibration (via calibrationReport) -> construct.
// Each call returns an independent model with its own empty buffer, which the
// property below relies on to compare two models fed different interleavings of
// the same real actions. windowLength is 2 (fixed by calibrationReport).
func emptyNoopNewCalibratedModel(t *testing.T) *NGramScoringModel {
	t.Helper()
	baseline := histogramBaseline(t, []string{"A", "B"}, 1, 2)
	calibrated, err := FitNGramCalibration(baseline, calibrationReport(8, 0, 4, 2), 0.5)
	if err != nil {
		t.Fatalf("failed to build calibrated fixture: %v", err)
	}
	model, err := NewNGramScoringModel(calibrated)
	if err != nil {
		t.Fatalf("failed to construct calibrated model: %v", err)
	}
	return model
}

// emptyNoopRealEntry builds a non-nil AuditLogEntry whose extracted action is a
// non-empty string, by setting NormalizedQueryTemplate via its setter.
func emptyNoopRealEntry(action string) *auditpb.AuditLogEntry {
	e := &auditpb.AuditLogEntry{}
	e.SetNormalizedQueryTemplate(action)
	return e
}

// emptyNoopEmptyActionEntry builds a non-nil AuditLogEntry whose extracted
// action is the empty string "" (NormalizedQueryTemplate left unset).
func emptyNoopEmptyActionEntry() *auditpb.AuditLogEntry {
	return &auditpb.AuditLogEntry{}
}

// emptyNoopSanitizeActions maps an arbitrary generated []string onto the small
// EmptyNoopAlphabet and drops any empty strings, so the "real actions" sequence
// contains only genuine, non-empty actions. The interleaving of empty/nil
// entries is added separately and deterministically from a second seed.
func emptyNoopSanitizeActions(raw []string) []string {
	out := make([]string, 0, len(raw))
	for i, s := range raw {
		// Deterministically fold each generated string onto the alphabet; the
		// exact mapping does not matter, only that it is non-empty and drawn
		// from a small set so windows complete.
		idx := (len(s) + i) % len(EmptyNoopAlphabet)
		out = append(out, EmptyNoopAlphabet[idx])
	}
	return out
}

// emptyNoopFindingsMatch reports whether two findings carry the same anomaly
// decision and confidence. The empty/nil no-op property only needs the scored
// signal to be identical at the real-action positions; identifying fields are
// derived from the (identical) real entries in both models.
func emptyNoopFindingsMatch(a, b *findingpb.AnomalyFinding) bool {
	return a.GetIsAnomalousPredict() == b.GetIsAnomalousPredict() &&
		a.GetConfidenceScore() == b.GetConfidenceScore()
}

// TestEmptyNoopEmptyAndNilAreNoOps is a property-based test asserting that
// empty-action and nil entries are no-ops: interleaving them among real actions
// yields exactly the same sequence of anomaly decisions and window completions
// as feeding only the real actions, and each interleaved empty/nil entry returns
// exactly one non-nil, not-anomalous finding.
//
// Feature: ngram-scoring-model, Property 4: Empty actions are no-ops that never
// complete a window. For any stream of entries, removing all entries whose
// action is empty yields the same sequence of window completions (same scored
// windows, in the same order) as the original stream; and each empty-action
// entry produces exactly one not-anomalous finding without changing the buffered
// action count.
//
// Strategy: build two fresh, identically-constructed calibrated models. Model A
// is fed only the real (non-empty) actions. Model B is fed the same real actions
// with nil and empty-action entries deterministically interleaved from a second
// seed. Assert that the subsequence of B's findings at the real-action positions
// equals A's findings exactly (same is_anomalous, same confidence), and that
// every interleaved empty/nil entry in B returns a non-nil, not-anomalous
// finding. Equal decisions at real positions imply the empty/nil entries neither
// advanced window progress nor altered the buffer.
//
// Validates: Requirements 2.4, 2.5, 3.6.
func TestEmptyNoopEmptyAndNilAreNoOps(t *testing.T) {
	property := func(rawActions []string, interleaveSeed uint64) bool {
		realActions := emptyNoopSanitizeActions(rawActions)

		// Model A: real actions only. Record one finding per real action.
		modelA := emptyNoopNewCalibratedModel(t)
		wantFindings := make([]*findingpb.AnomalyFinding, 0, len(realActions))
		for _, action := range realActions {
			f := modelA.Predict(emptyNoopRealEntry(action))
			if f == nil {
				t.Errorf("model A returned nil finding for action %q", action)
				return false
			}
			wantFindings = append(wantFindings, f)
		}

		// Model B: the same real actions with nil/empty entries interleaved.
		// At each step we decide, deterministically from interleaveSeed, how
		// many no-op entries to inject before the next real action (and a
		// trailing batch after the last real action). Each injected entry is
		// either a nil *AuditLogEntry or an empty-action entry.
		modelB := emptyNoopNewCalibratedModel(t)
		seed := interleaveSeed
		realIdx := 0

		// injectNoOps feeds `count` no-op entries, alternating nil and empty,
		// asserting each returns a non-nil, not-anomalous finding.
		injectNoOps := func(count int, nilFirst bool) bool {
			for j := 0; j < count; j++ {
				var entry *auditpb.AuditLogEntry
				if (j%2 == 0) == nilFirst {
					entry = nil // nil entry: extraction fails -> zero-value finding
				} else {
					entry = emptyNoopEmptyActionEntry() // empty action no-op
				}
				f := modelB.Predict(entry)
				if f == nil {
					t.Errorf("model B returned nil finding for a no-op entry (nil=%v)", entry == nil)
					return false
				}
				if f.GetIsAnomalousPredict() {
					t.Errorf("no-op entry (nil=%v) produced an anomalous finding", entry == nil)
					return false
				}
			}
			return true
		}

		for realIdx < len(realActions) {
			noOpCount := int(seed % 3) // 0, 1, or 2 no-ops before this real action
			nilFirst := (seed/3)%2 == 0
			seed = seed*6364136223846793005 + 1442695040888963407 // LCG step
			if !injectNoOps(noOpCount, nilFirst) {
				return false
			}

			got := modelB.Predict(emptyNoopRealEntry(realActions[realIdx]))
			if got == nil {
				t.Errorf("model B returned nil finding for real action %q", realActions[realIdx])
				return false
			}
			if !emptyNoopFindingsMatch(wantFindings[realIdx], got) {
				t.Errorf("finding mismatch at real index %d (action %q): "+
					"A{anom=%v conf=%v} B{anom=%v conf=%v}",
					realIdx, realActions[realIdx],
					wantFindings[realIdx].GetIsAnomalousPredict(), wantFindings[realIdx].GetConfidenceScore(),
					got.GetIsAnomalousPredict(), got.GetConfidenceScore())
				return false
			}
			realIdx++
		}

		// Trailing no-ops after the last real action must also be harmless.
		trailing := int(seed % 3)
		return injectNoOps(trailing, (seed/3)%2 == 0)
	}

	if err := quick.Check(property, &quick.Config{MaxCount: 200}); err != nil {
		t.Fatalf("empty/nil no-op property failed: %v", err)
	}
}
