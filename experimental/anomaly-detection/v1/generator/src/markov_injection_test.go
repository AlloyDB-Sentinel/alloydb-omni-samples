package main

import (
	"bufio"
	"bytes"
	"context"
	"io"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"google.golang.org/protobuf/encoding/protodelim"
	"google.golang.org/protobuf/proto"

	genpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v1/generator/proto"
)

// Base graph: a -> b -> a. Node "a" emits template "A", node "b" emits "B".
// Completion times are 1ms at a and 2ms at b (fixed) so the clock is predictable.
const injectionBaseGraph = `markov_sampling {
 nodes {id:"a" template:{normalized_query_template:"A"} completion_time:{fixed:{nanos:1000000}}
 edges:{destination_id:"b" weight:1}}
 nodes {id:"b" template:{normalized_query_template:"B"} completion_time:{fixed:{nanos:2000000}}
 edges:{destination_id:"a" weight:1}}
 start:{node_id:"a"}
}`

// Anomaly graph: a single terminal node "x" emitting template "X" with a 1ms completion time.
// It is a complete, independent markov_sampling graph with its OWN start spec and node set, not a
// reweighting of the base graph.
const injectionAnomalyGraph = `markov_sampling {
 nodes {id:"x" template:{normalized_query_template:"X"} completion_time:{fixed:{nanos:1000000}}}
 start:{node_id:"x"}
}`

// writeInjectionFixture writes the base graph, the anomaly graph, and a variant config that runs
// the base graph with one injection segment referencing the anomaly graph by relative path over
// actions [startAction, startAction+numActions). Returns the variant file path.
func writeInjectionFixture(t *testing.T, dir string, startAction, numActions int64) string {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, "base.textpb"), []byte(injectionBaseGraph), 0o644); err != nil {
		t.Fatalf("write base: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "anomaly.textpb"), []byte(injectionAnomalyGraph), 0o644); err != nil {
		t.Fatalf("write anomaly: %v", err)
	}
	variant := injectionBaseGraph[:len(injectionBaseGraph)-len("\n}")] +
		"\n injections {segment_id:\"seg1\" anomaly_graph_path:\"anomaly.textpb\"" +
		" start_action:" + strconv.FormatInt(startAction, 10) + " num_actions:" + strconv.FormatInt(numActions, 10) + "}\n}"
	variantPath := filepath.Join(dir, "variant.textpb")
	if err := os.WriteFile(variantPath, []byte(variant), 0o644); err != nil {
		t.Fatalf("write variant: %v", err)
	}
	return variantPath
}

// decodeEntries reads all length-delimited AuditLogEntry records from b.
func decodeEntries(t *testing.T, b []byte) []*genpb.AuditLogEntry {
	t.Helper()
	var entries []*genpb.AuditLogEntry
	r := bufio.NewReader(bytes.NewReader(b))
	for {
		e := &genpb.AuditLogEntry{}
		if err := protodelim.UnmarshalFrom(r, e); err != nil {
			if err == io.EOF {
				break
			}
			t.Fatalf("UnmarshalFrom: %v", err)
		}
		entries = append(entries, e)
	}
	return entries
}

// TestMarkovInjectionFullGraph verifies that an injection segment runs a referenced full anomaly
// graph on its own start spec, labels injected entries with the anomaly_graph_path and segment_id,
// advances one shared non-decreasing clock across the pause/resume boundary, resumes the normal
// cursor where it paused, and is byte-identical deterministic.
func TestMarkovInjectionFullGraph(t *testing.T) {
	dir := t.TempDir()
	// Inject over actions [1, 3): positions 1 and 2 run the anomaly graph ("X"), the rest are normal.
	variantPath := writeInjectionFixture(t, dir, 1, 2)

	cfg, err := loadConfigs(context.Background(), variantPath)
	if err != nil {
		t.Fatalf("loadConfigs: %v", err)
	}
	before := proto.Clone(cfg)

	var buf bytes.Buffer
	if err := GenerateLogsMarkov(cfg, 5, 100, rand.New(rand.NewSource(42)), &buf); err != nil {
		t.Fatalf("GenerateLogsMarkov: %v", err)
	}
	entries := decodeEntries(t, buf.Bytes())

	// Normal walk a->b->a->...; positions 1,2 replaced by the anomaly graph's "X".
	// Normal cursor pauses after emitting position 0 ("A") and resumes at position 3 from where it
	// left off (the next node "b"). Clock advances by each emitted node's completion time:
	// A(100,+1) X(101,+1) X(102,+1) B(103,+2) A(105,+1).
	want := []string{"A", "X", "X", "B", "A"}
	times := []int64{100, 101, 102, 103, 105}
	if len(entries) != len(want) {
		t.Fatalf("got %d entries, want %d", len(entries), len(want))
	}
	for i, e := range entries {
		if e.GetNormalizedQueryTemplate() != want[i] || e.GetTimestampMs() != times[i] {
			t.Fatalf("record %d = (%q,%d), want (%q,%d)", i, e.GetNormalizedQueryTemplate(), e.GetTimestampMs(), want[i], times[i])
		}
		injected := i == 1 || i == 2
		if e.GetIsGroundTruthAnomalous() != injected {
			t.Fatalf("record %d anomalous=%v, want %v", i, e.GetIsGroundTruthAnomalous(), injected)
		}
		if injected {
			if e.GetGroundTruthProfile() != "anomaly.textpb" || e.GetGroundTruthSegmentId() != "seg1" {
				t.Fatalf("record %d injected provenance = (%q,%q)", i, e.GetGroundTruthProfile(), e.GetGroundTruthSegmentId())
			}
		} else if e.GetGroundTruthProfile() != "normal" || e.GetGroundTruthSegmentId() != "" {
			t.Fatalf("record %d normal provenance = (%q,%q)", i, e.GetGroundTruthProfile(), e.GetGroundTruthSegmentId())
		}
	}

	if !proto.Equal(cfg, before) {
		t.Fatal("generation mutated the config")
	}

	// Determinism: a second run with the same seed/flags is byte-identical.
	var buf2 bytes.Buffer
	if err := GenerateLogsMarkov(cfg, 5, 100, rand.New(rand.NewSource(42)), &buf2); err != nil {
		t.Fatalf("GenerateLogsMarkov (2nd): %v", err)
	}
	if !bytes.Equal(buf.Bytes(), buf2.Bytes()) {
		t.Fatal("injection generation is not deterministic")
	}
}

// TestMarkovInjectionOutOfRange rejects an injection whose range exceeds --num_logs.
func TestMarkovInjectionOutOfRange(t *testing.T) {
	dir := t.TempDir()
	variantPath := writeInjectionFixture(t, dir, 1, 2) // needs num_logs >= 3
	cfg, err := loadConfigs(context.Background(), variantPath)
	if err != nil {
		t.Fatalf("loadConfigs: %v", err)
	}
	if err := GenerateLogsMarkov(cfg, 2, 0, rand.New(rand.NewSource(42)), io.Discard); err == nil {
		t.Fatal("accepted out-of-range injection")
	}
}

// TestMarkovInjectionResolution covers anomaly-graph file resolution and the no-chaining rule.
func TestMarkovInjectionResolution(t *testing.T) {
	t.Run("missing file rejected", func(t *testing.T) {
		dir := t.TempDir()
		if err := os.WriteFile(filepath.Join(dir, "base.textpb"), []byte(injectionBaseGraph), 0o644); err != nil {
			t.Fatal(err)
		}
		variant := injectionBaseGraph[:len(injectionBaseGraph)-len("\n}")] +
			"\n injections {segment_id:\"seg1\" anomaly_graph_path:\"nope.textpb\" start_action:1 num_actions:2}\n}"
		variantPath := filepath.Join(dir, "variant.textpb")
		if err := os.WriteFile(variantPath, []byte(variant), 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := loadConfigs(context.Background(), variantPath); err == nil {
			t.Fatal("accepted injection referencing a missing anomaly graph file")
		}
	})

	t.Run("chained injection rejected", func(t *testing.T) {
		dir := t.TempDir()
		if err := os.WriteFile(filepath.Join(dir, "base.textpb"), []byte(injectionBaseGraph), 0o644); err != nil {
			t.Fatal(err)
		}
		// The referenced anomaly graph itself carries an injection -> must be rejected (no chaining).
		chained := injectionAnomalyGraph[:len(injectionAnomalyGraph)-len("\n}")] +
			"\n injections {segment_id:\"nested\" anomaly_graph_path:\"base.textpb\" start_action:0 num_actions:1}\n}"
		if err := os.WriteFile(filepath.Join(dir, "anomaly.textpb"), []byte(chained), 0o644); err != nil {
			t.Fatal(err)
		}
		variant := injectionBaseGraph[:len(injectionBaseGraph)-len("\n}")] +
			"\n injections {segment_id:\"seg1\" anomaly_graph_path:\"anomaly.textpb\" start_action:1 num_actions:2}\n}"
		variantPath := filepath.Join(dir, "variant.textpb")
		if err := os.WriteFile(variantPath, []byte(variant), 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := loadConfigs(context.Background(), variantPath); err == nil {
			t.Fatal("accepted a chained anomaly-graph reference")
		}
	})
}
