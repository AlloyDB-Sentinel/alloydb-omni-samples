package service

import (
	"fmt"
	"testing"
	"testing/quick"

	auditpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common/proto"
)

// MirrorActionAlphabet is the small set of actions the mirror streams draw
// from. "A"/"B" are the calibrated-known categories and "C"/"D" are novel, so
// completed windows exercise both the anomalous and not-anomalous branches of
// the scorer's decision. The empty string "" is included so the empty-action
// no-op path (R3.6) is covered alongside non-empty actions.
var MirrorActionAlphabet = []string{"A", "B", "C", "D", ""}

// MirrorModel builds a calibrated NGramScoringModel from the shared scorer-test
// fixture helpers (histogramBaseline + FitNGramCalibration + calibrationReport).
// windowLength is fixed at 2 — small enough that short generated streams
// complete some windows while leaving others still filling.
func MirrorModel(t *testing.T) *NGramScoringModel {
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
	return model
}

// MirrorEntry builds an AuditLogEntry whose four identifying fields are derived
// deterministically from the generated seed and whose action (normalized query
// template) is drawn from MirrorActionAlphabet. The normalized query template
// is BOTH an identifying field mirrored into the finding AND the action the
// model buffers, so this single generator exercises the empty-action,
// still-filling, and completed-window paths while randomizing every identifying
// field the finding must mirror.
func MirrorEntry(seed uint64) *auditpb.AuditLogEntry {
	action := MirrorActionAlphabet[int(seed>>3)%len(MirrorActionAlphabet)]
	e := &auditpb.AuditLogEntry{}
	e.SetDbUser(fmt.Sprintf("user-%d", seed%97))
	e.SetDbName(fmt.Sprintf("db-%d", (seed/97)%89))
	e.SetQueryStatement(fmt.Sprintf("SELECT %d FROM t WHERE x = %d", seed%13, (seed/13)%17))
	e.SetNormalizedQueryTemplate(action)
	return e
}

// MirrorIdentityMatches asserts the finding's four identifying fields equal the
// entry's corresponding fields, using the SAME mapping generateAnomalyFinding
// applies (db user, db name, query statement, normalized query template). It
// returns false and logs on the first mismatch.
func MirrorIdentityMatches(t *testing.T, entry *auditpb.AuditLogEntry, finding interface {
	GetDbUser() string
	GetDbName() string
	GetQueryStatement() string
	GetNormalizedQueryTemplate() string
}) bool {
	t.Helper()
	if got, want := finding.GetDbUser(), entry.GetDbUser(); got != want {
		t.Errorf("db_user mirrored incorrectly: got %q want %q", got, want)
		return false
	}
	if got, want := finding.GetDbName(), entry.GetDbName(); got != want {
		t.Errorf("db_name mirrored incorrectly: got %q want %q", got, want)
		return false
	}
	if got, want := finding.GetQueryStatement(), entry.GetQueryStatement(); got != want {
		t.Errorf("query_statement mirrored incorrectly: got %q want %q", got, want)
		return false
	}
	if got, want := finding.GetNormalizedQueryTemplate(), entry.GetNormalizedQueryTemplate(); got != want {
		t.Errorf("normalized_query_template mirrored incorrectly: got %q want %q", got, want)
		return false
	}
	return true
}

// TestMirrorIdentifyingFields is a property-based test asserting that every
// finding produced for a non-nil log entry mirrors that entry's identifying
// fields exactly as the shared generateAnomalyFinding contract does, regardless
// of whether the window completed, is still filling, or the action was empty.
//
// Feature: ngram-scoring-model, Property 7: Findings mirror the log entry's
// identifying fields. For any non-nil log entry, the returned finding's db user,
// db name, query statement, and normalized query template equal the
// corresponding fields of the entry, and the finding's anomaly flag and
// confidence score fields are set.
//
// The stream mixes empty actions (no-op, R3.6), non-empty actions that leave the
// buffer still filling (R3.2), and actions that complete a tumbling window
// (R3.3): the mirroring of identifying fields must hold on all three paths. The
// nil-entry path is excluded here (it returns a zero-value finding with no
// mirroring, R2.4) and is asserted separately in TestMirrorNilEntryNoMirroring.
//
// Validates: Requirements 2.3.
func TestMirrorIdentifyingFields(t *testing.T) {
	model := MirrorModel(t)

	property := func(seeds []uint64) bool {
		// Each generated stream starts from an empty buffer so completions are
		// independent of previous iterations.
		model.ResetLiveCounts()

		for i, seed := range seeds {
			entry := MirrorEntry(seed)
			finding := model.Predict(entry)
			if finding == nil {
				t.Errorf("i=%d seed=%d: Predict returned nil finding", i, seed)
				return false
			}
			// Identifying fields must mirror the entry on every path.
			if !MirrorIdentityMatches(t, entry, finding) {
				t.Errorf("i=%d seed=%d action=%q: identifying-field mismatch",
					i, seed, entry.GetNormalizedQueryTemplate())
				return false
			}
			// The anomaly flag and confidence score fields are present on the
			// finding (set by generateAnomalyFinding). Reading them must not
			// panic; the confidence must be a defined value in [0,1].
			_ = finding.GetIsAnomalousPredict()
			if c := finding.GetConfidenceScore(); c < 0.0 || c > 1.0 {
				t.Errorf("i=%d seed=%d: confidence_score %v out of [0,1]", i, seed, c)
				return false
			}
		}
		return true
	}

	if err := quick.Check(property, &quick.Config{MaxCount: 200}); err != nil {
		t.Fatalf("findings-mirror-identifying-fields property failed: %v", err)
	}
}

// TestMirrorNilEntryNoMirroring pins the excluded nil-entry case: Predict(nil)
// returns a non-nil, zero-value finding with no identifying fields mirrored and
// the anomaly/confidence fields at their zero defaults (R2.4). This complements
// the property above, which reasons only about non-nil entries.
func TestMirrorNilEntryNoMirroring(t *testing.T) {
	model := MirrorModel(t)

	finding := model.Predict(nil)
	if finding == nil {
		t.Fatal("Predict(nil) returned a nil finding")
	}
	if got := finding.GetDbUser(); got != "" {
		t.Errorf("nil entry: db_user = %q, want empty", got)
	}
	if got := finding.GetDbName(); got != "" {
		t.Errorf("nil entry: db_name = %q, want empty", got)
	}
	if got := finding.GetQueryStatement(); got != "" {
		t.Errorf("nil entry: query_statement = %q, want empty", got)
	}
	if got := finding.GetNormalizedQueryTemplate(); got != "" {
		t.Errorf("nil entry: normalized_query_template = %q, want empty", got)
	}
	if finding.GetIsAnomalousPredict() {
		t.Errorf("nil entry: is_anomalous_predict = true, want false")
	}
	if got := finding.GetConfidenceScore(); got != 0.0 {
		t.Errorf("nil entry: confidence_score = %v, want 0.0", got)
	}
}
