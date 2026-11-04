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

	auditpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common/proto"
	predictionpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/prediction_service/proto"
)

// predictRPCWindowLength is the window length used for the integration models.
// It is kept small so a short batch or file completes at least one window.
const predictRPCWindowLength = 4

// predictRPCCalibratedModelPath builds a baseline NgramHistogram model from a
// repeating action pattern, writes it to disk, and calibrates it through the
// CalibrateWindows RPC. It returns the absolute path to the calibrated model on
// disk, ready to serve as a Predict RPC model_location. The on-disk model
// carries a version-matching NGramCalibration, so modelFromProto constructs an
// NGramScoringModel with a real anomaly threshold and Predict yields genuine
// anomaly decisions (no FailedPrecondition for ngram).
func predictRPCCalibratedModelPath(ctx context.Context, t *testing.T, dir string) string {
	t.Helper()

	trainer, err := training.NewNGramHistogramTrainer(3, predictRPCWindowLength)
	if err != nil {
		t.Fatalf("failed to create ngram trainer: %v", err)
	}
	// A repeating A,A,B,B pattern yields a well-formed baseline whose complete
	// A,A,B,B windows score as normal and whose novel-action windows score as
	// anomalous once calibrated (mirrors window_server_test.go).
	for _, a := range []string{"A", "A", "B", "B", "A", "A", "B", "B"} {
		trainer.Count(common.ExtractedLogFeatures{QueryTemplate: a})
	}
	baseline := trainer.BuildModel()
	baselineBytes, err := proto.Marshal(baseline)
	if err != nil {
		t.Fatalf("failed to marshal baseline model: %v", err)
	}

	basePath := filepath.Join(dir, "predictrpc_baseline.bin")
	calPath := filepath.Join(dir, "predictrpc_calibrated.bin")
	if err := os.WriteFile(basePath, baselineBytes, 0644); err != nil {
		t.Fatalf("failed to write baseline model: %v", err)
	}

	// Calibration log: enough complete windows (windowLength=4) for the
	// percentile fit to produce a threshold.
	calLogs := filepath.Join(dir, "predictrpc_calibration.binproto")
	writeWindowLogs(t, calLogs, []string{"A", "A", "B", "B", "B", "B", "A", "A", "A", "B", "A", "B"}, false)

	srv := NewPredictionServer()
	calReq := &predictionpb.CalibrateWindowsRequest{}
	calReq.SetModelLocation(basePath)
	calReq.SetCalibrationLogPath(calLogs)
	calReq.SetOutputModelLocation(calPath)
	calReq.SetPercentile(0.99)
	if _, err := srv.CalibrateWindows(ctx, calReq); err != nil {
		t.Fatalf("failed to calibrate ngram model: %v", err)
	}

	abs, err := filepath.Abs(calPath)
	if err != nil {
		t.Fatalf("failed to resolve calibrated model path: %v", err)
	}
	return abs
}

// predictRPCNormalEntry builds a non-nil AuditLogEntry whose normalized query
// template (the scored action) is the given action, with stable identifying
// fields so findings can be inspected.
func predictRPCNormalEntry(action string) *auditpb.AuditLogEntry {
	e := &auditpb.AuditLogEntry{}
	e.SetNormalizedQueryTemplate(action)
	e.SetQueryStatement("stmt-" + action)
	e.SetDbUser("predictrpc_user")
	e.SetDbName("predictrpc_db")
	return e
}

// TestPredictRPCNgramSingleLog drives the Predict RPC single-log path against a
// calibrated ngram model. One log entry cannot complete a windowLength window,
// so the response must be a SingleFinding (one model) that is not anomalous,
// and the RPC must not reject ngram with FailedPrecondition. (R6.3, R6.4)
func TestPredictRPCNgramSingleLog(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	modelPath := predictRPCCalibratedModelPath(ctx, t, dir)

	srv := &predictionServer{}
	req := makeSingleLogPredictRequest(predictRPCNormalEntry("A"), modelPath)

	resp, err := srv.Predict(ctx, req)
	if err != nil {
		t.Fatalf("Predict() returned error: %v", err)
	}
	if got := status.Code(err); got == codes.FailedPrecondition {
		t.Fatalf("Predict() rejected ngram with FailedPrecondition")
	}
	if resp.WhichResult() != predictionpb.PredictResponse_SingleFinding_case {
		t.Fatalf("expected SingleFinding payload, got %v", resp.WhichResult())
	}
	finding := resp.GetSingleFinding()
	if finding == nil {
		t.Fatalf("expected a non-nil single finding")
	}
	// A single entry only partially fills the window, so it is never anomalous.
	if finding.GetIsAnomalousPredict() {
		t.Errorf("single still-filling entry should not be anomalous")
	}
	// Identifying fields must mirror the input entry.
	if finding.GetNormalizedQueryTemplate() != "A" || finding.GetDbUser() != "predictrpc_user" || finding.GetDbName() != "predictrpc_db" {
		t.Errorf("finding identifying fields not mirrored: %+v", finding)
	}
}

// TestPredictRPCNgramBatchLogs drives the Predict RPC batch path against a
// calibrated ngram model. A stream of windowLength actions completes a window
// and yields a scored finding on the window-completing entry; still-filling
// entries are not anomalous. The batch path resets the buffer before the first
// entry, so results are deterministic across repeated requests. (R6.3, R6.5)
func TestPredictRPCNgramBatchLogs(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	modelPath := predictRPCCalibratedModelPath(ctx, t, dir)

	srv := &predictionServer{}

	// A normal window (A,A,B,B) followed by a clearly novel window (C,C,C,C).
	// Completions occur on the windowLength-th entry of each group.
	actions := []string{"A", "A", "B", "B", "C", "C", "C", "C"}
	var entries []*auditpb.AuditLogEntry
	for _, a := range actions {
		entries = append(entries, predictRPCNormalEntry(a))
	}
	req := makeBatchLogsPredictRequest(entries, modelPath)

	resp, err := srv.Predict(ctx, req)
	if err != nil {
		t.Fatalf("Predict() returned error: %v", err)
	}
	if got := status.Code(err); got == codes.FailedPrecondition {
		t.Fatalf("Predict() rejected ngram batch with FailedPrecondition")
	}
	if resp.WhichResult() != predictionpb.PredictResponse_BatchFindings_case {
		t.Fatalf("expected BatchFindings payload, got %v", resp.WhichResult())
	}
	findings := resp.GetBatchFindings().GetFindings()
	if len(findings) != len(actions) {
		t.Fatalf("expected %d findings (one per entry), got %d", len(actions), len(findings))
	}

	// Still-filling entries (indices 0,1,2 and 4,5,6) are never anomalous.
	for _, i := range []int{0, 1, 2, 4, 5, 6} {
		if findings[i].GetIsAnomalousPredict() {
			t.Errorf("still-filling finding[%d] should not be anomalous", i)
		}
	}
	// The window-completing entries are the scored ones: index 3 ends the normal
	// window (not anomalous) and index 7 ends the novel window (anomalous).
	if findings[3].GetIsAnomalousPredict() {
		t.Errorf("normal completed window (finding[3]) should not be anomalous")
	}
	if !findings[7].GetIsAnomalousPredict() {
		t.Errorf("novel completed window (finding[7]) should be anomalous")
	}

	// Repeated request: because the batch path resets the buffer before the
	// first entry, the response is identical across calls (no window spans two
	// requests). (R6.5)
	req2 := makeBatchLogsPredictRequest(entries, modelPath)
	resp2, err := srv.Predict(ctx, req2)
	if err != nil {
		t.Fatalf("second Predict() returned error: %v", err)
	}
	if !proto.Equal(resp, resp2) {
		t.Errorf("batch results not deterministic across repeated requests; buffer leaked between requests")
	}
}

// TestPredictRPCNgramFilePath drives the Predict RPC file path against a
// calibrated ngram model. A file whose first windowLength actions complete a
// novel window yields an anomalous finding for that window; the path returns a
// findings file. The file path resets the buffer before the first entry, so
// repeated requests produce the same findings. (R6.3, R6.5)
func TestPredictRPCNgramFilePath(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	modelPath := predictRPCCalibratedModelPath(ctx, t, dir)

	// windowLength novel actions complete one anomalous window.
	logPath := filepath.Join(dir, "predictrpc_file_input.binproto")
	writeWindowLogs(t, logPath, []string{"C", "C", "C", "C"}, false)

	srv := &predictionServer{}
	req := makeFilePredictRequest(logPath, modelPath)

	resp, err := srv.Predict(ctx, req)
	if err != nil {
		t.Fatalf("Predict() returned error: %v", err)
	}
	if got := status.Code(err); got == codes.FailedPrecondition {
		t.Fatalf("Predict() rejected ngram file with FailedPrecondition")
	}
	if resp.WhichResult() != predictionpb.PredictResponse_FindingsPath_case {
		t.Fatalf("expected FindingsPath payload, got %v", resp.WhichResult())
	}
	outPath := resp.GetFindingsPath()
	if outPath == "" {
		t.Fatalf("expected a non-empty findings path")
	}
	t.Cleanup(func() { os.Remove(outPath) })
	if _, err := os.Stat(outPath); err != nil {
		t.Fatalf("findings file %q does not exist: %v", outPath, err)
	}
	firstBytes, err := os.ReadFile(outPath)
	if err != nil {
		t.Fatalf("failed to read findings file: %v", err)
	}

	// Repeated request: the file path resets the buffer before the first entry,
	// so the findings are identical across calls (deterministic, R6.5).
	resp2, err := srv.Predict(ctx, makeFilePredictRequest(logPath, modelPath))
	if err != nil {
		t.Fatalf("second Predict() returned error: %v", err)
	}
	out2 := resp2.GetFindingsPath()
	if out2 == "" {
		t.Fatalf("expected a non-empty findings path on repeated request")
	}
	t.Cleanup(func() { os.Remove(out2) })
	secondBytes, err := os.ReadFile(out2)
	if err != nil {
		t.Fatalf("failed to read repeated findings file: %v", err)
	}
	if string(firstBytes) != string(secondBytes) {
		t.Errorf("file results not deterministic across repeated requests; buffer leaked between requests")
	}
}
