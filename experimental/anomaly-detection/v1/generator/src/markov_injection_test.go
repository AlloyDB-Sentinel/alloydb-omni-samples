package main

import (
	"bytes"
	"io"
	"math"
	"math/rand"
	"strings"
	"testing"

	"google.golang.org/protobuf/proto"

	genpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v1/generator/proto"
)

const injectionConfig = `markov_sampling {
 nodes {id:"a" template:{normalized_query_template:"A" is_ground_truth_anomalous:true}
 completion_time:{fixed:{nanos:1000000}}
 edges:{destination_id:"a" weight:0} edges:{destination_id:"b" weight:1}}
 nodes {id:"b" template:{normalized_query_template:"B"}
 completion_time:{fixed:{nanos:2000000}} edges:{destination_id:"a" weight:1}}
 start:{node_id:"a"}
 anomaly_profiles {profile_id:"repeat-a" transitions {node_id:"a"
 edges:{destination_id:"a" weight:1} edges:{destination_id:"b" weight:0}}}
 injections {segment_id:"injected" profile_id:"repeat-a" start_action:1 num_actions:2}
}`

func TestMarkovInjectionLabelsClockAndResume(t *testing.T) {
	cfg := parseProto(t, injectionConfig, &genpb.QueryTemplateConfigs{})
	if err := validateConfigs(cfg); err != nil {
		t.Fatal(err)
	}
	before := proto.Clone(cfg)
	entries := runMarkov(t, injectionConfig, 5, 100, 42)
	want := []string{"A", "A", "A", "B", "A"}
	times := []int64{100, 101, 102, 103, 105}
	for i, e := range entries {
		if e.GetNormalizedQueryTemplate() != want[i] || e.GetTimestampMs() != times[i] {
			t.Fatalf("record %d: %v", i, e)
		}
		anomalous := i == 1 || i == 2
		if e.GetIsGroundTruthAnomalous() != anomalous {
			t.Fatalf("record %d label wrong", i)
		}
		if anomalous {
			if e.GetGroundTruthProfile() != "repeat-a" || e.GetGroundTruthSegmentId() != "injected" {
				t.Fatal("missing injection provenance")
			}
		} else if e.GetGroundTruthProfile() != "normal" || e.GetGroundTruthSegmentId() != "" {
			t.Fatal("incorrect normal provenance")
		}
	}
	var buf bytes.Buffer
	if err := GenerateLogsMarkov(cfg, 5, 100, rand.New(rand.NewSource(42)), &buf); err != nil {
		t.Fatal(err)
	}
	if !proto.Equal(cfg, before) {
		t.Fatal("generation mutated normal graph or profile")
	}
	if !bytes.Equal(buf.Bytes(), runMarkovRaw(t, injectionConfig, 5, 100, 42)) {
		t.Fatal("injection generation is not deterministic")
	}
	if err := GenerateLogsMarkov(cfg, 2, 0, rand.New(rand.NewSource(42)), io.Discard); err == nil {
		t.Fatal("accepted out-of-range injection")
	}
}

func TestMarkovInjectionValidation(t *testing.T) {
	base := parseProto(t, injectionConfig, &genpb.QueryTemplateConfigs{})
	for _, mutate := range []func(*genpb.MarkovSamplingConfig){
		func(c *genpb.MarkovSamplingConfig) { c.GetAnomalyProfiles()[0].SetProfileId("normal") },
		func(c *genpb.MarkovSamplingConfig) {
			c.SetAnomalyProfiles(append(c.GetAnomalyProfiles(), proto.Clone(c.GetAnomalyProfiles()[0]).(*genpb.MarkovAnomalyProfile)))
		},
		func(c *genpb.MarkovSamplingConfig) {
			c.GetAnomalyProfiles()[0].GetTransitions()[0].SetNodeId("missing")
		},
		func(c *genpb.MarkovSamplingConfig) {
			c.GetAnomalyProfiles()[0].GetTransitions()[0].GetEdges()[0].SetWeight(math.NaN())
		},
		func(c *genpb.MarkovSamplingConfig) {
			c.GetAnomalyProfiles()[0].GetTransitions()[0].GetEdges()[0].SetWeight(0.5)
		},
		func(c *genpb.MarkovSamplingConfig) {
			c.GetAnomalyProfiles()[0].GetTransitions()[0].GetEdges()[1].SetDestinationId("a")
		},
		func(c *genpb.MarkovSamplingConfig) {
			c.GetAnomalyProfiles()[0].GetTransitions()[0].GetEdges()[1].SetDestinationId("missing")
		},
		func(c *genpb.MarkovSamplingConfig) { c.GetAnomalyProfiles()[0].SetTransitions(nil) },
		func(c *genpb.MarkovSamplingConfig) { c.GetInjections()[0].SetSegmentId("") },
		func(c *genpb.MarkovSamplingConfig) { c.GetInjections()[0].SetProfileId("missing") },
		func(c *genpb.MarkovSamplingConfig) { c.GetInjections()[0].SetStartAction(-1) },
		func(c *genpb.MarkovSamplingConfig) { c.GetInjections()[0].SetNumActions(0) },
		func(c *genpb.MarkovSamplingConfig) { c.GetInjections()[0].SetNumActions(math.MaxInt64) },
		func(c *genpb.MarkovSamplingConfig) {
			second := proto.Clone(c.GetInjections()[0]).(*genpb.MarkovInjection)
			second.SetSegmentId("overlap")
			c.SetInjections(append(c.GetInjections(), second))
		},
	} {
		cfg := proto.Clone(base).(*genpb.QueryTemplateConfigs)
		mutate(cfg.GetMarkovSampling())
		if err := validateConfigs(cfg); err == nil {
			t.Fatalf("accepted invalid injection config: %v", cfg)
		}
	}
	// Adjacent segments and a final segment ending exactly at the cap are valid.
	adjacent := strings.Replace(injectionConfig, "start_action:1 num_actions:2", "start_action:0 num_actions:2", 1)
	adjacent = strings.TrimSuffix(adjacent, "}") + `injections {segment_id:"second" profile_id:"repeat-a" start_action:2 num_actions:3}}`
	cfg := parseProto(t, adjacent, &genpb.QueryTemplateConfigs{})
	if err := validateConfigs(cfg); err != nil {
		t.Fatal(err)
	}
	entries := runMarkov(t, adjacent, 5, 0, 1)
	for i, e := range entries {
		want := "injected"
		if i >= 2 {
			want = "second"
		}
		if !e.GetIsGroundTruthAnomalous() || e.GetGroundTruthSegmentId() != want {
			t.Fatal("adjacent segment labels incorrect")
		}
	}
}
