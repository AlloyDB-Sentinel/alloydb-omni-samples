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
	"google.golang.org/protobuf/encoding/protodelim"
	"google.golang.org/protobuf/proto"

	modelpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common/proto"
	predictionpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/prediction_service/proto"
)

func writeWindowLogs(t *testing.T, path string, actions []string, alterLabels bool) {
	t.Helper()
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	for i, a := range actions {
		e := &modelpb.AuditLogEntry{}
		e.SetNormalizedQueryTemplate(a)
		if alterLabels {
			e.SetIsGroundTruthAnomalous(true)
			e.SetGroundTruthProfile("different-profile")
			e.SetGroundTruthSegmentId("different-segment")
			e.SetTimestampMs(int64(i * 123))
			e.SetDbUser("different-user")
		}
		if _, err := protodelim.MarshalTo(f, e); err != nil {
			t.Fatal(err)
		}
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestWindowCalibrationAndValidationRPC(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()
	trainer, err := training.NewNGramHistogramTrainer(3, 4)
	if err != nil {
		t.Fatal(err)
	}
	for _, a := range []string{"A", "A", "B", "B", "A", "A", "B", "B"} {
		trainer.Count(common.ExtractedLogFeatures{QueryTemplate: a})
	}
	baseline := trainer.BuildModel()
	baselineBytes, err := proto.Marshal(baseline)
	if err != nil {
		t.Fatal(err)
	}
	basePath := filepath.Join(dir, "baseline.bin")
	calPath := filepath.Join(dir, "calibrated.bin")
	if err := os.WriteFile(basePath, baselineBytes, 0644); err != nil {
		t.Fatal(err)
	}
	logs := filepath.Join(dir, "calibration.binproto")
	writeWindowLogs(t, logs, []string{"A", "A", "B", "B", "B", "B", "A", "A", "A", "B", "A", "B"}, false)
	srv := NewPredictionServer()
	req := &predictionpb.CalibrateWindowsRequest{}
	req.SetModelLocation(basePath)
	req.SetCalibrationLogPath(logs)
	req.SetOutputModelLocation(calPath)
	req.SetPercentile(0.99)
	resp, err := srv.CalibrateWindows(ctx, req)
	if err != nil {
		t.Fatal(err)
	}
	if resp.GetCalibration().GetCompleteWindows() != 3 || resp.GetScores().HasThreshold() {
		t.Fatal("incorrect calibration report")
	}
	for _, w := range resp.GetScores().GetWindows() {
		if w.HasIsAnomalous() {
			t.Fatal("calibration classified its own windows")
		}
	}
	bytes, err := os.ReadFile(calPath)
	if err != nil {
		t.Fatal(err)
	}
	calibrated := &modelpb.Model{}
	if err := proto.Unmarshal(bytes, calibrated); err != nil {
		t.Fatal(err)
	}
	if !proto.Equal(calibrated.GetNgramHistogram().GetModelData(), baseline.GetNgramHistogram().GetModelData()) || !proto.Equal(calibrated.GetMetadata(), baseline.GetMetadata()) {
		t.Fatal("calibration changed baseline")
	}
	after, err := os.ReadFile(basePath)
	if err != nil || string(after) != string(baselineBytes) {
		t.Fatal("calibration overwrote original model")
	}
	validation := filepath.Join(dir, "validation.binproto")
	actions := []string{"A", "A", "B", "B", "C", "C", "C", "C", "A"}
	writeWindowLogs(t, validation, actions, false)
	scoreReq := &predictionpb.ScoreWindowsRequest{}
	scoreReq.SetModelLocation(calPath)
	scoreReq.SetValidationLogPath(validation)
	report, err := srv.ScoreWindows(ctx, scoreReq)
	if err != nil {
		t.Fatal(err)
	}
	if report.GetCompleteWindows() != 2 || report.GetTrailingActions() != 1 || report.GetTotalActions() != 9 || !report.HasThreshold() {
		t.Fatalf("invalid report: %v", report)
	}
	for i, w := range report.GetWindows() {
		if w.GetWindowIndex() != int64(i) || w.GetStartAction() != int64(i*4) || w.GetEndAction() != int64((i+1)*4) {
			t.Fatal("incorrect window positions")
		}
	}
	if report.GetWindows()[0].GetIsAnomalous() || !report.GetWindows()[1].GetIsAnomalous() {
		t.Fatalf("normal/novel windows classified incorrectly: %v", report)
	}
	// Ground-truth labels, users and timing must never affect scoring.
	writeWindowLogs(t, validation, actions, true)
	repeated, err := srv.ScoreWindows(ctx, scoreReq)
	if err != nil {
		t.Fatal(err)
	}
	if !proto.Equal(report, repeated) {
		t.Fatal("ground truth leaked into scoring or positions were not reset")
	}
	after, err = os.ReadFile(calPath)
	if err != nil || string(after) != string(bytes) {
		t.Fatal("validation modified saved model")
	}
	filter := &predictionpb.ModelFilter{}
	filter.SetNgramHistogram(calibrated.GetNgramHistogram().GetConfig())
	listReq := &predictionpb.ListModelsRequest{}
	listReq.SetModelLocation(calPath)
	listReq.SetFilter(filter)
	listed, err := srv.ListModels(ctx, listReq)
	if err != nil || len(listed.GetModels()) != 1 || !listed.GetModels()[0].HasNgramHistogramConfig() {
		t.Fatalf("histogram listing/filtering failed: %v %v", listed, err)
	}
	// Failure modes must not overwrite either model.
	scoreReq.SetModelLocation(basePath)
	if _, err := srv.ScoreWindows(ctx, scoreReq); status.Code(err) != codes.FailedPrecondition {
		t.Fatalf("uncalibrated validation: %v", err)
	}
	scoreReq.SetModelLocation(calPath)
	writeWindowLogs(t, validation, []string{"A"}, false)
	if _, err := srv.ScoreWindows(ctx, scoreReq); status.Code(err) != codes.InvalidArgument {
		t.Fatalf("partial validation: %v", err)
	}
	req.SetOutputModelLocation(basePath)
	if _, err := srv.CalibrateWindows(ctx, req); status.Code(err) != codes.InvalidArgument {
		t.Fatalf("model overwrite: %v", err)
	}
	alias := filepath.Join(dir, "alias.bin")
	if err := os.Link(basePath, alias); err != nil {
		t.Fatal(err)
	}
	req.SetOutputModelLocation(alias)
	if _, err := srv.CalibrateWindows(ctx, req); status.Code(err) != codes.InvalidArgument {
		t.Fatalf("model alias overwrite: %v", err)
	}
	req.SetOutputModelLocation(calPath)
	req.SetPercentile(1)
	if _, err := srv.CalibrateWindows(ctx, req); status.Code(err) != codes.InvalidArgument {
		t.Fatalf("invalid percentile: %v", err)
	}
	req.SetPercentile(0.99)
	req.SetCalibrationLogPath(validation)
	if _, err := srv.CalibrateWindows(ctx, req); status.Code(err) != codes.InvalidArgument {
		t.Fatalf("insufficient calibration: %v", err)
	}
}
