package main

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"testing"

	"google.golang.org/protobuf/encoding/protodelim"
	"google.golang.org/protobuf/proto"

	genpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v1/generator/proto"
	clientpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/client/proto"
	predictionpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/prediction_service/proto"
)

func TestGroundTruthWindowEvaluation(t *testing.T) {
	path := filepath.Join(t.TempDir(), "validation.binproto")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	// 4 windows of 3 actions; a segment crosses windows 1/2, and a
	// separate trailing segment has no scored coverage.
	for i := 0; i < 14; i++ {
		e := &genpb.AuditLogEntry{}
		e.SetNormalizedQueryTemplate("A")
		if i >= 5 && i < 8 {
			e.SetIsGroundTruthAnomalous(true)
			e.SetGroundTruthProfile("changed")
			e.SetGroundTruthSegmentId("crossing")
		}
		if i >= 12 {
			e.SetIsGroundTruthAnomalous(true)
			e.SetGroundTruthProfile("changed")
			e.SetGroundTruthSegmentId("trailing")
		}
		if _, err := protodelim.MarshalTo(f, e); err != nil {
			t.Fatal(err)
		}
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	scores := &predictionpb.WindowScoreReport{}
	scores.SetWindowLength(3)
	scores.SetTotalActions(14)
	scores.SetCompleteWindows(4)
	scores.SetTrailingActions(2)
	var windows []*predictionpb.WindowScore
	for i, flag := range []bool{true, false, true, false} {
		w := &predictionpb.WindowScore{}
		w.SetWindowIndex(int64(i))
		w.SetStartAction(int64(i * 3))
		w.SetEndAction(int64((i + 1) * 3))
		w.SetScore(0.5)
		w.SetIsAnomalous(flag)
		windows = append(windows, w)
	}
	scores.SetWindows(windows)
	before := proto.Clone(scores)
	r, err := evaluateWindows(context.Background(), path, scores)
	if err != nil {
		t.Fatal(err)
	}
	if r.GetTruePositives() != 1 || r.GetFalsePositives() != 1 || r.GetTrueNegatives() != 1 || r.GetFalseNegatives() != 1 {
		t.Fatalf("wrong confusion matrix: %v", r)
	}
	if r.GetAnomalousActions() != 5 || r.GetScoredAnomalousActions() != 3 || r.GetTrailingAnomalousActions() != 2 {
		t.Fatal("wrong ground-truth counts")
	}
	if r.GetPrecision() != 0.5 || r.GetRecall() != 0.5 || r.GetFalsePositiveRate() != 0.5 || r.GetF1() != 0.5 {
		t.Fatal("wrong metrics")
	}
	if math.Abs(r.GetWindows()[1].GetAnomalousFraction()-1.0/3) > 1e-12 || math.Abs(r.GetWindows()[2].GetAnomalousFraction()-2.0/3) > 1e-12 {
		t.Fatal("wrong boundary fractions")
	}
	if r.GetScoredSegments() != 1 || r.GetDetectedSegments() != 1 || r.GetSegmentRecall() != 1 || r.GetSegments()[1].GetScoredActions() != 0 {
		t.Fatal("trailing segment affected scored recall")
	}
	if !proto.Equal(scores, before) {
		t.Fatal("ground-truth evaluation mutated detector scores")
	}
	// Undefined metrics are absent rather than silently reported as perfect/zero.
	empty := filepath.Join(t.TempDir(), "empty.binproto")
	if err := os.WriteFile(empty, nil, 0644); err != nil {
		t.Fatal(err)
	}
	emptyScores := &predictionpb.WindowScoreReport{}
	emptyScores.SetWindowLength(3)
	r, err = evaluateWindows(context.Background(), empty, emptyScores)
	if err != nil {
		t.Fatal(err)
	}
	if r.HasPrecision() || r.HasRecall() || r.HasFalsePositiveRate() || r.HasF1() || r.HasSegmentRecall() {
		t.Fatal("undefined metrics should be absent")
	}
}

func TestMixedExperimentIsolation(t *testing.T) {
	experiment, err := readExperiment("configs/standard_user.textproto")
	if err != nil {
		t.Fatal(err)
	}
	if len(experiment.GetAnomalyProfiles()) != 1 || len(experiment.GetInjections()) != 6 {
		t.Fatal("incorrect example schedule")
	}
	base := &genpb.QueryTemplateConfigs{}
	graph := &genpb.MarkovSamplingConfig{}
	base.SetMarkovSampling(graph)
	before := proto.Clone(base)
	mixed := validationGeneratorConfig(base, experiment)
	if !proto.Equal(base, before) || len(mixed.GetMarkovSampling().GetInjections()) != 6 {
		t.Fatal("validation polluted the normal config")
	}
	for _, mutate := range []func(*clientpb.MarkovExperimentConfig){
		func(c *clientpb.MarkovExperimentConfig) { c.GetInjections()[0].SetStartAction(-1) },
		func(c *clientpb.MarkovExperimentConfig) { c.GetInjections()[0].SetNumActions(0) },
		func(c *clientpb.MarkovExperimentConfig) {
			c.GetInjections()[0].SetNumActions(c.GetValidationActions() + 1)
		},
		func(c *clientpb.MarkovExperimentConfig) { c.GetInjections()[0].SetProfileId("missing") },
		func(c *clientpb.MarkovExperimentConfig) {
			c.GetInjections()[1].SetStartAction(c.GetInjections()[0].GetStartAction())
		},
	} {
		c := proto.Clone(experiment).(*clientpb.MarkovExperimentConfig)
		mutate(c)
		if err := validateExperimentInjections(c); err == nil {
			t.Fatal("invalid experiment injection accepted")
		}
	}
}
