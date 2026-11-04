package main

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common"
	training "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/training_service/service"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"

	modelpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common/proto"
	predictionpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/prediction_service/proto"
)

// Regression coverage for Requirement 7 (Preserve existing window-scoring
// behavior) of the ngram-scoring-model feature. The ngram Predict wiring
// (modelFromProto NgramHistogram case + removal of the ScoreWindows
// precondition in Predict) is purely additive: it must NOT alter the behavior
// of the window-oriented RPCs. These tests drive the same CalibrateWindows /
// ScoreWindows paths exercised by TestWindowCalibrationAndValidationRPC (in
// window_server_test.go) with deterministic action streams and assert concrete,
// stable outcomes. They reuse writeWindowLogs rather than redeclaring it.
//
// Validates: Requirements 7.1, 7.2, 7.3, 7.4

// windowRegressBaselineModel builds a histogram baseline model from the same
// deterministic "A/A/B/B" action pattern the existing window-server test uses,
// so the calibrated model and window scores are reproducible.
func windowRegressBaselineModel(t *testing.T) *modelpb.Model {
	t.Helper()
	trainer, err := training.NewNGramHistogramTrainer(3, 4)
	if err != nil {
		t.Fatal(err)
	}
	for _, a := range []string{"A", "A", "B", "B", "A", "A", "B", "B"} {
		trainer.Count(common.ExtractedLogFeatures{QueryTemplate: a})
	}
	return trainer.BuildModel()
}

// windowRegressCalibratedModelPath writes the baseline to disk, calibrates it
// through the CalibrateWindows RPC, and returns the calibrated model's path.
// Asserts the calibration report is identical to the window-server baseline
// (3 complete windows, no self-classification threshold).
func windowRegressCalibratedModelPath(t *testing.T, ctx context.Context, srv predictionpb.PredictionServiceServer, dir string) string {
	t.Helper()
	baseline := windowRegressBaselineModel(t)
	baselineBytes, err := proto.Marshal(baseline)
	if err != nil {
		t.Fatal(err)
	}
	basePath := filepath.Join(dir, "windowregress_baseline.bin")
	calPath := filepath.Join(dir, "windowregress_calibrated.bin")
	if err := os.WriteFile(basePath, baselineBytes, 0644); err != nil {
		t.Fatal(err)
	}
	logs := filepath.Join(dir, "windowregress_calibration.binproto")
	writeWindowLogs(t, logs, []string{"A", "A", "B", "B", "B", "B", "A", "A", "A", "B", "A", "B"}, false)

	req := &predictionpb.CalibrateWindowsRequest{}
	req.SetModelLocation(basePath)
	req.SetCalibrationLogPath(logs)
	req.SetOutputModelLocation(calPath)
	req.SetPercentile(0.99)
	resp, err := srv.CalibrateWindows(ctx, req)
	if err != nil {
		t.Fatalf("CalibrateWindows failed unexpectedly: %v", err)
	}
	// Calibration must behave exactly as before the ngram Predict wiring:
	// a histogram baseline calibrates to 3 complete windows and the
	// calibration report itself carries no (self-classifying) threshold.
	if got := resp.GetCalibration().GetCompleteWindows(); got != 3 {
		t.Fatalf("calibration complete windows = %d, want 3", got)
	}
	if resp.GetScores().HasThreshold() {
		t.Fatal("calibration report unexpectedly classified its own windows")
	}
	return calPath
}

// TestWindowRegressScoreWindowsUnchanged confirms a calibrated histogram model
// still scores correctly through the ScoreWindows RPC after the ngram Predict
// wiring. The deterministic validation stream has one in-distribution window
// (A,A,B,B) and one novel window (C,C,C,C); the former must score as normal and
// the latter as anomalous, exactly as the pre-change window path did.
//
// Validates: Requirements 7.1, 7.3
func TestWindowRegressScoreWindowsUnchanged(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	srv := NewPredictionServer()

	calPath := windowRegressCalibratedModelPath(t, ctx, srv, dir)

	validation := filepath.Join(dir, "windowregress_validation.binproto")
	actions := []string{"A", "A", "B", "B", "C", "C", "C", "C", "A"}
	writeWindowLogs(t, validation, actions, false)

	scoreReq := &predictionpb.ScoreWindowsRequest{}
	scoreReq.SetModelLocation(calPath)
	scoreReq.SetValidationLogPath(validation)
	report, err := srv.ScoreWindows(ctx, scoreReq)
	if err != nil {
		t.Fatalf("ScoreWindows failed unexpectedly: %v", err)
	}

	// The window path must still report the same structural counts: two
	// complete tumbling windows of length 4, one trailing action, nine total
	// actions, and a present (calibrated) threshold.
	if report.GetCompleteWindows() != 2 {
		t.Fatalf("complete windows = %d, want 2", report.GetCompleteWindows())
	}
	if report.GetTrailingActions() != 1 {
		t.Fatalf("trailing actions = %d, want 1", report.GetTrailingActions())
	}
	if report.GetTotalActions() != 9 {
		t.Fatalf("total actions = %d, want 9", report.GetTotalActions())
	}
	if !report.HasThreshold() {
		t.Fatal("scored report is missing the calibrated threshold")
	}

	// Non-overlapping (tumbling) window positions are unchanged.
	for i, w := range report.GetWindows() {
		if w.GetWindowIndex() != int64(i) || w.GetStartAction() != int64(i*4) || w.GetEndAction() != int64((i+1)*4) {
			t.Fatalf("window %d positions changed: %v", i, w)
		}
	}

	// This is the assertion that would FAIL if the ngram Predict wiring had
	// accidentally broken the window scoring path: the in-distribution window
	// must be normal and the novel window must be anomalous.
	if report.GetWindows()[0].GetIsAnomalous() {
		t.Fatalf("in-distribution window was classified anomalous: %v", report.GetWindows()[0])
	}
	if !report.GetWindows()[1].GetIsAnomalous() {
		t.Fatalf("novel window was not classified anomalous: %v", report.GetWindows()[1])
	}
}

// TestWindowRegressGroundTruthNeverLeaks confirms that ground-truth labels,
// users, and timestamps on the validation log still never affect window
// scoring after the ngram Predict wiring. Scoring the same actions with and
// without altered labels must produce byte-identical reports.
//
// Validates: Requirements 7.1, 7.3
func TestWindowRegressGroundTruthNeverLeaks(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	srv := NewPredictionServer()

	calPath := windowRegressCalibratedModelPath(t, ctx, srv, dir)

	validation := filepath.Join(dir, "windowregress_gt.binproto")
	actions := []string{"A", "A", "B", "B", "C", "C", "C", "C", "A"}

	scoreReq := &predictionpb.ScoreWindowsRequest{}
	scoreReq.SetModelLocation(calPath)
	scoreReq.SetValidationLogPath(validation)

	writeWindowLogs(t, validation, actions, false)
	plain, err := srv.ScoreWindows(ctx, scoreReq)
	if err != nil {
		t.Fatalf("ScoreWindows (plain) failed unexpectedly: %v", err)
	}

	writeWindowLogs(t, validation, actions, true)
	labeled, err := srv.ScoreWindows(ctx, scoreReq)
	if err != nil {
		t.Fatalf("ScoreWindows (labeled) failed unexpectedly: %v", err)
	}

	if !proto.Equal(plain, labeled) {
		t.Fatal("ground truth leaked into window scoring or positions were not reset")
	}
}

// TestWindowRegressErrorPropagationUnchanged confirms the window RPCs still
// reject the same invalid inputs with the same gRPC codes after the ngram
// Predict wiring. In particular an uncalibrated histogram model must still be
// rejected by ScoreWindows with FailedPrecondition (it must NOT silently flow
// through the per-entry Predict path), and a validation log with no complete
// window must still fail with InvalidArgument.
//
// Validates: Requirements 7.2, 7.3, 7.4
func TestWindowRegressErrorPropagationUnchanged(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	srv := NewPredictionServer()

	// An uncalibrated histogram baseline on disk.
	baseline := windowRegressBaselineModel(t)
	baselineBytes, err := proto.Marshal(baseline)
	if err != nil {
		t.Fatal(err)
	}
	basePath := filepath.Join(dir, "windowregress_uncalibrated.bin")
	if err := os.WriteFile(basePath, baselineBytes, 0644); err != nil {
		t.Fatal(err)
	}

	validation := filepath.Join(dir, "windowregress_err_validation.binproto")
	writeWindowLogs(t, validation, []string{"A", "A", "B", "B"}, false)

	scoreReq := &predictionpb.ScoreWindowsRequest{}
	scoreReq.SetModelLocation(basePath)
	scoreReq.SetValidationLogPath(validation)
	if _, err := srv.ScoreWindows(ctx, scoreReq); status.Code(err) != codes.FailedPrecondition {
		t.Fatalf("uncalibrated ScoreWindows code = %v, want FailedPrecondition", status.Code(err))
	}

	// A calibrated model with a validation log that contains no complete
	// window must still be an InvalidArgument error.
	calPath := windowRegressCalibratedModelPath(t, ctx, srv, dir)
	partial := filepath.Join(dir, "windowregress_partial.binproto")
	writeWindowLogs(t, partial, []string{"A"}, false)
	scoreReq.SetModelLocation(calPath)
	scoreReq.SetValidationLogPath(partial)
	if _, err := srv.ScoreWindows(ctx, scoreReq); status.Code(err) != codes.InvalidArgument {
		t.Fatalf("partial-window ScoreWindows code = %v, want InvalidArgument", status.Code(err))
	}

	// CalibrateWindows must still reject an out-of-range percentile.
	calLogs := filepath.Join(dir, "windowregress_cal_logs.binproto")
	writeWindowLogs(t, calLogs, []string{"A", "A", "B", "B", "B", "B", "A", "A", "A", "B", "A", "B"}, false)
	calReq := &predictionpb.CalibrateWindowsRequest{}
	calReq.SetModelLocation(basePath)
	calReq.SetCalibrationLogPath(calLogs)
	calReq.SetOutputModelLocation(filepath.Join(dir, "windowregress_bad_percentile.bin"))
	calReq.SetPercentile(1)
	if _, err := srv.CalibrateWindows(ctx, calReq); status.Code(err) != codes.InvalidArgument {
		t.Fatalf("invalid-percentile CalibrateWindows code = %v, want InvalidArgument", status.Code(err))
	}
}
