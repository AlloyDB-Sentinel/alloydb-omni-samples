package main

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"testing/quick"

	"github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common"
	training "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/training_service/service"
	"google.golang.org/protobuf/proto"

	auditpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common/proto"
	findingpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common/proto"
	predictionpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/prediction_service/proto"
)

// bufResetWindowLength is the window length of the calibrated ngram model built
// for this test. It is kept small so that short generated batch requests can
// complete one or more tumbling windows (and thus produce scored findings whose
// equality across requests is meaningful).
const bufResetWindowLength = 3

// bufResetAlphabet is the small set of non-empty actions the generated batch
// requests draw from. It mixes calibrated-known categories ("A"/"B") with novel
// ones ("C"/"D") so that different windows score as anomalous or normal, giving
// the cross-request equality check something non-trivial to compare.
var bufResetAlphabet = []string{"A", "B", "C", "D"}

// findingAlias is a local alias for the AnomalyFinding proto type, kept so the
// helper signatures below stay readable without redeclaring any shared symbol.
type findingAlias = findingpb.AnomalyFinding

// predictionServerPredictor is the minimal surface of the prediction server this
// test drives. NewPredictionServer returns a value satisfying it.
type predictionServerPredictor interface {
	Predict(ctx context.Context, req *predictionpb.PredictRequest) (*predictionpb.PredictResponse, error)
}

// bufResetCalibratedModelPath builds a calibrated NgramHistogram model on disk
// following window_server_test.go's calibration pipeline (trainer -> baseline ->
// CalibrateWindows RPC) and returns the on-disk path of the calibrated model.
//
// windowLength is fixed at bufResetWindowLength (small) so generated batch
// requests can complete windows. The returned path is suitable as a
// PredictRequest model_location.
func bufResetCalibratedModelPath(ctx context.Context, t *testing.T) string {
	t.Helper()
	dir := t.TempDir()

	trainer, err := training.NewNGramHistogramTrainer(3, bufResetWindowLength)
	if err != nil {
		t.Fatalf("failed to build trainer: %v", err)
	}
	// Feed a repeating A/B pattern so the baseline carries at least two complete
	// training windows (required by the scorer) over known categories A and B.
	for _, a := range []string{"A", "A", "B", "B", "A", "A", "B", "B", "A", "A", "B", "B"} {
		trainer.Count(common.ExtractedLogFeatures{QueryTemplate: a})
	}
	baseline := trainer.BuildModel()
	baselineBytes, err := proto.Marshal(baseline)
	if err != nil {
		t.Fatalf("failed to marshal baseline: %v", err)
	}
	basePath := filepath.Join(dir, "baseline.bin")
	calPath := filepath.Join(dir, "calibrated.bin")
	if err := os.WriteFile(basePath, baselineBytes, 0644); err != nil {
		t.Fatalf("failed to write baseline model: %v", err)
	}

	// Write a calibration log long enough to produce at least two complete
	// windows (12 actions / windowLength 3 = 4 complete windows).
	logs := filepath.Join(dir, "calibration.binproto")
	writeWindowLogs(t, logs, []string{"A", "A", "B", "B", "B", "B", "A", "A", "A", "B", "A", "B"}, false)

	srv := NewPredictionServer()
	req := &predictionpb.CalibrateWindowsRequest{}
	req.SetModelLocation(basePath)
	req.SetCalibrationLogPath(logs)
	req.SetOutputModelLocation(calPath)
	req.SetPercentile(0.99)
	if _, err := srv.CalibrateWindows(ctx, req); err != nil {
		t.Fatalf("failed to calibrate model: %v", err)
	}
	return calPath
}

// bufResetBatchRequest maps a quick-generated seed and length byte to a batch
// PredictRequest of non-empty actions drawn from bufResetAlphabet, targeting the
// given calibrated model location.
//
// Deriving the actions from the generated integers (rather than a []string
// generator) keeps every action non-empty and within the small alphabet, which
// is the input domain this property reasons about. Each entry also carries
// identifying fields so the findings populate db user/name and query statement.
func bufResetBatchRequest(seed uint64, lengthByte uint8, modelLocation string) *predictionpb.PredictRequest {
	// Length in [1, 10]: enough to complete one or more windows of length 3
	// while also exercising a trailing, not-yet-complete action.
	n := 1 + int(lengthByte)%10
	entries := make([]*auditpb.AuditLogEntry, n)
	s := seed
	for i := 0; i < n; i++ {
		// Simple LCG step so successive actions are varied yet deterministic.
		s = s*6364136223846793005 + 1442695040888963407
		action := bufResetAlphabet[int(s>>33)%len(bufResetAlphabet)]
		e := &auditpb.AuditLogEntry{}
		e.SetNormalizedQueryTemplate(action)
		e.SetQueryStatement(action + " statement")
		e.SetDbUser("buf_reset_user")
		e.SetDbName("buf_reset_db")
		entries[i] = e
	}
	return makeBatchLogsPredictRequest(entries, modelLocation)
}

// bufResetFindings serves a batch PredictRequest against the server and returns
// the resulting findings slice. A batch request always yields a batch-findings
// response (even for a single model), so the findings come from
// GetBatchFindings().GetFindings().
func bufResetFindings(ctx context.Context, t *testing.T, srv predictionServerPredictor, req *predictionpb.PredictRequest) []*findingAlias {
	t.Helper()
	resp, err := srv.Predict(ctx, req)
	if err != nil {
		t.Fatalf("Predict returned error: %v", err)
	}
	batch := resp.GetBatchFindings()
	if batch == nil {
		t.Fatalf("expected batch findings, got nil")
	}
	out := make([]*findingAlias, len(batch.GetFindings()))
	copy(out, batch.GetFindings())
	return out
}

// bufResetFindingsEqual reports whether two findings slices agree in length and,
// per entry, in the anomaly flag and confidence score. Those two fields are the
// only window-scoring-derived outputs of a finding, so they are what a buffer
// leak across requests would perturb.
func bufResetFindingsEqual(a, b []*findingAlias) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i].GetIsAnomalousPredict() != b[i].GetIsAnomalousPredict() {
			return false
		}
		if a[i].GetConfidenceScore() != b[i].GetConfidenceScore() {
			return false
		}
	}
	return true
}

// TestBufResetAcrossRequests is a property-based test asserting that the ngram
// model's buffered window state never leaks across prediction requests.
//
// Feature: ngram-scoring-model, Property 9: Buffer never spans two prediction
// requests. For any split of an action stream into a first request followed by a
// second request, where each batch/file request resets the model via
// ResetLiveCounts() before its first entry, the window completions of the second
// request are identical to scoring the second request's actions on a freshly
// constructed model; equivalently, ResetLiveCounts() returns the buffer to empty
// so no window straddles the request boundary.
//
// The property generates two random batch requests, reqA (a warm-up / prefix
// request) and reqB (the request under test), against the SAME calibrated ngram
// model on disk. It serves Predict(reqA) then Predict(reqB) on one server and
// collects reqB's findings; independently it serves ONLY reqB on a fresh server
// and collects its findings. Because predictBatch calls resetModels (which
// invokes ResetLiveCounts) at the start of every batch request, reqB's findings
// must be identical whether or not reqA preceded it.
//
// Validates: Requirements 6.5.
func TestBufResetAcrossRequests(t *testing.T) {
	ctx := context.Background()
	modelPath := bufResetCalibratedModelPath(ctx, t)

	property := func(seedA uint64, lenA uint8, seedB uint64, lenB uint8) bool {
		reqA := bufResetBatchRequest(seedA, lenA, modelPath)
		reqB := bufResetBatchRequest(seedB, lenB, modelPath)

		// Server 1: serve the warm-up request first, then the request under
		// test. resetModels runs at the start of predictBatch for BOTH requests,
		// so the models loaded for reqB start with an empty buffer regardless of
		// what reqA fed.
		srvWithPrefix := NewPredictionServer()
		_ = bufResetFindings(ctx, t, srvWithPrefix, reqA)
		withPrefix := bufResetFindings(ctx, t, srvWithPrefix, reqB)

		// Server 2 (fresh): serve ONLY the request under test.
		srvFresh := NewPredictionServer()
		alone := bufResetFindings(ctx, t, srvFresh, reqB)

		if !bufResetFindingsEqual(withPrefix, alone) {
			t.Errorf("reqB findings differ after a preceding request: seedA=%d lenA=%d seedB=%d lenB=%d (withPrefix=%d entries, alone=%d entries)",
				seedA, lenA, seedB, lenB, len(withPrefix), len(alone))
			return false
		}
		return true
	}

	if err := quick.Check(property, &quick.Config{MaxCount: 200}); err != nil {
		t.Fatalf("buffer-reset-across-requests property failed: %v", err)
	}
}
