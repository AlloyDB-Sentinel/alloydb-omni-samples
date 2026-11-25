package main

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/encoding/protodelim"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/timestamppb"

	modelpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common/proto"
	trainingpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/training_service/proto"
)

func TestTrainExcessiveDataReadsModel(t *testing.T) {
	start := time.Date(2026, 1, 5, 0, 0, 0, 0, time.UTC)
	config := &modelpb.ExcessiveDataReadsTrainingConfig{}
	config.SetShortWindowStdDevMultiplier(6)
	config.SetLongWindowStdDevMultiplier(4)
	config.SetTrainingStartTime(timestamppb.New(start))
	config.SetTrainingEndTime(timestamppb.New(start.Add(14 * 24 * time.Hour)))
	trainingConfig := &trainingpb.ModelTrainingConfig{}
	trainingConfig.SetExcessiveDataReadsConfig(config)
	oldDir := *modelDir
	t.Cleanup(func() { *modelDir = oldDir })
	for _, test := range []struct {
		name    string
		offsets []time.Duration
		code    codes.Code
	}{
		{"valid", []time.Duration{0, 7 * 24 * time.Hour}, codes.OK},
		{"out_of_order", []time.Duration{time.Minute, 0}, codes.InvalidArgument},
		{"warmup_only", []time.Duration{-time.Minute}, codes.InvalidArgument},
	} {
		t.Run(test.name, func(t *testing.T) {
			*modelDir = t.TempDir()
			path := filepath.Join(t.TempDir(), "training.binproto")
			file, err := os.Create(path)
			if err != nil {
				t.Fatal(err)
			}
			for _, offset := range test.offsets {
				entry := &modelpb.AuditLogEntry{}
				entry.SetDbName("db")
				entry.SetDbUser("alice")
				entry.SetTimestampMs(start.Add(offset).UnixMilli())
				entry.SetRowsReturned(100)
				if _, err := protodelim.MarshalTo(file, entry); err != nil {
					t.Fatal(err)
				}
			}
			if err := file.Close(); err != nil {
				t.Fatal(err)
			}
			resp, err := NewTrainingServer(nil).TrainModels(context.Background(), newReq(path, trainingConfig))
			if status.Code(err) != test.code {
				t.Fatalf("TrainModels() error = %v, want %v", err, test.code)
			}
			if test.code != codes.OK {
				files, err := os.ReadDir(*modelDir)
				if err != nil || len(files) != 0 {
					t.Fatalf("invalid input saved a partial model: %v, %v", files, err)
				}
				return
			}
			if len(resp.GetResults()) != 1 {
				t.Fatalf("got %d models, want 1", len(resp.GetResults()))
			}
			data, err := os.ReadFile(resp.GetResults()[0].GetModelLocation())
			if err != nil {
				t.Fatal(err)
			}
			model := &modelpb.Model{}
			if err := proto.Unmarshal(data, model); err != nil {
				t.Fatal(err)
			}
			if len(model.GetExcessiveDataReads().GetModelData().GetEntries()) != 168 || !proto.Equal(model.GetExcessiveDataReads().GetConfig(), config) {
				t.Fatal("saved model is missing baselines or config")
			}
		})
	}
}
