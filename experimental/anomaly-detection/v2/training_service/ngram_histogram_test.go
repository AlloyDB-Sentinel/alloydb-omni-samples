package main

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/encoding/protodelim"
	"google.golang.org/protobuf/proto"

	auditpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common/proto"
	modelpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common/proto"
	trainingpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/training_service/proto"
)

func TestTrainNGramHistogramPersistence(t *testing.T) {
	dir := t.TempDir()
	previous := *modelDir
	*modelDir = dir
	t.Cleanup(func() { *modelDir = previous })
	path := filepath.Join(dir, "train.binproto")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, action := range []string{"A", "B", "A", "B"} {
		e := &auditpb.AuditLogEntry{}
		e.SetNormalizedQueryTemplate(action)
		if _, err := protodelim.MarshalTo(f, e); err != nil {
			t.Fatal(err)
		}
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	cfg := &modelpb.NGramHistogramTrainingConfig{}
	cfg.SetMaxN(3)
	c := &trainingpb.ModelTrainingConfig{}
	c.SetNgramHistogramConfig(cfg)
	pusher := &mockPusher{pushFunc: func(context.Context, string, *modelpb.Model) error {
		t.Fatal("histogram must not be pushed to prediction service")
		return nil
	}}
	srv := NewTrainingServer(pusher)
	resp, err := srv.TrainModels(context.Background(), newReq(path, c))
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.GetResults()) != 1 {
		t.Fatal("expected one model")
	}
	data, err := os.ReadFile(resp.GetResults()[0].GetModelLocation())
	if err != nil {
		t.Fatal(err)
	}
	m := &modelpb.Model{}
	if err := proto.Unmarshal(data, m); err != nil {
		t.Fatal(err)
	}
	if !m.HasNgramHistogram() || m.GetMetadata().GetTotalLogEntriesProcessed() != 4 {
		t.Fatalf("incorrect persisted model: %v", m)
	}
	for i, d := range m.GetNgramHistogram().GetModelData().GetHistograms() {
		if d.GetTotalWindows() != int64(4-i) || len(d.GetEntries()) != 2 {
			t.Fatalf("incorrect distribution: %v", d)
		}
	}
	cfg.SetMaxN(4)
	if _, err := srv.TrainModels(context.Background(), newReq(path, c)); status.Code(err) != codes.InvalidArgument {
		t.Fatalf("invalid max_n: %v", err)
	}
	cfg.SetMaxN(3)
	if err := os.WriteFile(path, []byte{0}, 0644); err != nil {
		t.Fatal(err)
	}
	if _, err := srv.TrainModels(context.Background(), newReq(path, c)); status.Code(err) != codes.InvalidArgument {
		t.Fatalf("empty template: %v", err)
	}
}
