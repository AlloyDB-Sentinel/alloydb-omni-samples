package main

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common"
	training "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/training_service/service"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/encoding/protodelim"
	"google.golang.org/protobuf/types/known/timestamppb"

	modelpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common/proto"
	predictionpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/prediction_service/proto"
)

func TestExcessiveDataReadsService(t *testing.T) {
	ctx := context.Background()
	start := time.Date(2026, 1, 5, 0, 0, 0, 0, time.UTC)
	config := &modelpb.ExcessiveDataReadsTrainingConfig{}
	config.SetShortWindowStdDevMultiplier(6)
	config.SetLongWindowStdDevMultiplier(4)
	config.SetTrainingStartTime(timestamppb.New(start))
	config.SetTrainingEndTime(timestamppb.New(start.Add(7 * 24 * time.Hour)))
	trainer, err := training.NewExcessiveDataReadsTrainer(config)
	if err != nil {
		t.Fatal(err)
	}
	trainer.Count(common.ExtractedLogFeatures{TimestampMs: start.UnixMilli(), RowsReturned: 100})
	path := filepath.Join(t.TempDir(), "model.bin")
	createTestModelFile(ctx, t, path, trainer.BuildModel())
	server := NewPredictionServer()
	filter := &predictionpb.ModelFilter{}
	filter.SetExcessiveDataReads(config)
	listReq := &predictionpb.ListModelsRequest{}
	listReq.SetModelLocation(path)
	listReq.SetFilter(filter)
	listed, err := server.ListModels(ctx, listReq)
	if err != nil || len(listed.GetModels()) != 1 || listed.GetModels()[0].GetExcessiveDataReadsConfig() == nil {
		t.Fatalf("ListModels() = %v, %v", listed, err)
	}
	entries := []*modelpb.AuditLogEntry{{}, {}}
	for i, entry := range entries {
		entry.SetTimestampMs(start.Add(time.Duration(i) * time.Minute).UnixMilli())
		entry.SetRowsReturned(60)
	}
	request := makeBatchLogsPredictRequest(entries, path)
	// Each request starts a fresh stream; reads within the batch share state.
	for i := 0; i < 2; i++ {
		resp, err := server.Predict(ctx, request)
		if err != nil {
			t.Fatal(err)
		}
		findings := resp.GetBatchFindings().GetFindings()
		if len(findings) != 2 || findings[0].GetIsAnomalousPredict() || !findings[1].GetIsAnomalousPredict() {
			t.Fatalf("unexpected batch findings: %v", findings)
		}
	}
	entries[1].SetTimestampMs(start.Add(-time.Minute).UnixMilli())
	if _, err := server.Predict(ctx, request); status.Code(err) != codes.InvalidArgument {
		t.Fatalf("out-of-order batch error = %v, want InvalidArgument", err)
	}
	logPath := filepath.Join(t.TempDir(), "invalid.binproto")
	file, err := os.Create(logPath)
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range entries {
		if _, err := protodelim.MarshalTo(file, entry); err != nil {
			t.Fatal(err)
		}
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	if _, err := server.Predict(ctx, makeFilePredictRequest(logPath, path)); status.Code(err) != codes.InvalidArgument {
		t.Fatalf("out-of-order file error = %v, want InvalidArgument", err)
	}
}
