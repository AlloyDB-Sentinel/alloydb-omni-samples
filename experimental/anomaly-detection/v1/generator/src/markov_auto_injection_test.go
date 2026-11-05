package main

import (
	"bytes"
	"context"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	genpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v1/generator/proto"
)

// Reuses injectionBaseGraph and injectionAnomalyGraph from markov_injection_test.go.

// writeAutoFixture writes the base graph, two anomaly graphs, and a variant config whose
// markov_sampling carries an auto_injection targeting anomalousFraction with the given segment
// length range and two weighted anomaly references. Returns the variant file path.
func writeAutoFixture(t *testing.T, dir string, fraction float64, minA, maxA int64) string {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, "base.textpb"), []byte(injectionBaseGraph), 0o644); err != nil {
		t.Fatalf("write base: %v", err)
	}
	// Two distinct single-node anomaly graphs so weighted selection is observable.
	anomX := injectionAnomalyGraph // node "x" -> "X"
	anomY := `markov_sampling {
 nodes {id:"y" template:{normalized_query_template:"Y"} completion_time:{fixed:{nanos:1000000}}}
 start:{node_id:"y"}
}`
	if err := os.WriteFile(filepath.Join(dir, "anomX.textpb"), []byte(anomX), 0o644); err != nil {
		t.Fatalf("write anomX: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "anomY.textpb"), []byte(anomY), 0o644); err != nil {
		t.Fatalf("write anomY: %v", err)
	}
	variant := injectionBaseGraph[:len(injectionBaseGraph)-len("\n}")] +
		"\n auto_injection {" +
		" anomalous_fraction:" + ftoa(fraction) +
		" min_actions:" + itoa(minA) + " max_actions:" + itoa(maxA) +
		" anomaly_refs:{anomaly_graph_path:\"anomX.textpb\" selection_weight:3}" +
		" anomaly_refs:{anomaly_graph_path:\"anomY.textpb\" selection_weight:1}" +
		" }\n}"
	variantPath := filepath.Join(dir, "variant.textpb")
	if err := os.WriteFile(variantPath, []byte(variant), 0o644); err != nil {
		t.Fatalf("write variant: %v", err)
	}
	return variantPath
}

func itoa(v int64) string   { return strconv.FormatInt(v, 10) }
func ftoa(v float64) string { return strconv.FormatFloat(v, 'f', -1, 64) }

// TestMarkovAutoInjectionPlacement verifies budget-bounded, in-bounds, deterministic auto placement
// with correct provenance labels.
func TestMarkovAutoInjectionPlacement(t *testing.T) {
	dir := t.TempDir()
	// fraction 0.3 of 100 actions => budget 30 anomalous actions, segments of length 2..5.
	variantPath := writeAutoFixture(t, dir, 0.3, 2, 5)
	cfg, err := loadConfigs(context.Background(), variantPath)
	if err != nil {
		t.Fatalf("loadConfigs: %v", err)
	}

	const total = 100
	var buf bytes.Buffer
	if err := GenerateLogsMarkov(cfg, total, 0, rand.New(rand.NewSource(7)), &buf); err != nil {
		t.Fatalf("GenerateLogsMarkov: %v", err)
	}
	entries := decodeEntries(t, buf.Bytes())
	if len(entries) != total {
		t.Fatalf("got %d entries, want %d", len(entries), total)
	}

	anomalous := 0
	// Track ordering: an anomalous run has a single segment id; segment ids must be distinct and in
	// "auto-N" order as positions increase.
	var lastSeg string
	var segOrder []string
	for i, e := range entries {
		if !e.GetIsGroundTruthAnomalous() {
			if e.GetGroundTruthProfile() != "normal" || e.GetGroundTruthSegmentId() != "" {
				t.Fatalf("record %d: normal entry has provenance (%q,%q)", i, e.GetGroundTruthProfile(), e.GetGroundTruthSegmentId())
			}
			lastSeg = ""
			continue
		}
		anomalous++
		prof := e.GetGroundTruthProfile()
		seg := e.GetGroundTruthSegmentId()
		if prof != "anomX.textpb" && prof != "anomY.textpb" {
			t.Fatalf("record %d: unexpected anomaly profile %q", i, prof)
		}
		if !strings.HasPrefix(seg, "auto-") {
			t.Fatalf("record %d: segment id %q not auto-prefixed", i, seg)
		}
		// The emitted template must come from the referenced graph (X or Y), never the base (A/B).
		if tmpl := e.GetNormalizedQueryTemplate(); tmpl != "X" && tmpl != "Y" {
			t.Fatalf("record %d: anomalous entry emitted base template %q", i, tmpl)
		}
		if seg != lastSeg {
			segOrder = append(segOrder, seg)
			lastSeg = seg
		}
	}

	// Budget: realized anomalous count must not exceed round(0.3*100)=30, and should be > 0.
	if anomalous == 0 || anomalous > 30 {
		t.Fatalf("anomalous count %d outside (0, 30]", anomalous)
	}
	// Segment ids appear in auto-0, auto-1, ... order.
	for i, seg := range segOrder {
		if seg != "auto-"+itoa(int64(i)) {
			t.Fatalf("segment %d has id %q, want auto-%d", i, seg, i)
		}
	}

	// Determinism: identical seed/flags -> byte-identical output.
	var buf2 bytes.Buffer
	if err := GenerateLogsMarkov(cfg, total, 0, rand.New(rand.NewSource(7)), &buf2); err != nil {
		t.Fatalf("GenerateLogsMarkov (2nd): %v", err)
	}
	if !bytes.Equal(buf.Bytes(), buf2.Bytes()) {
		t.Fatal("auto-injection generation is not deterministic")
	}
}

// TestMarkovAutoInjectionMutualExclusion rejects a config that sets both injections and
// auto_injection.
func TestMarkovAutoInjectionMutualExclusion(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "base.textpb"), []byte(injectionBaseGraph), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "anomaly.textpb"), []byte(injectionAnomalyGraph), 0o644); err != nil {
		t.Fatal(err)
	}
	variant := injectionBaseGraph[:len(injectionBaseGraph)-len("\n}")] +
		"\n injections {segment_id:\"m\" anomaly_graph_path:\"anomaly.textpb\" start_action:1 num_actions:2}" +
		"\n auto_injection { anomalous_fraction:0.2 min_actions:1 max_actions:2 anomaly_refs:{anomaly_graph_path:\"anomaly.textpb\"} }\n}"
	variantPath := filepath.Join(dir, "variant.textpb")
	if err := os.WriteFile(variantPath, []byte(variant), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := loadConfigs(context.Background(), variantPath); err == nil {
		t.Fatal("accepted config with both injections and auto_injection")
	}
}

// TestMarkovAutoInjectionValidation rejects malformed auto_injection specs.
func TestMarkovAutoInjectionValidation(t *testing.T) {
	cases := map[string]func(*genpb.MarkovAutoInjection){
		"fraction>1": func(a *genpb.MarkovAutoInjection) { a.SetAnomalousFraction(1.5) },
		"fraction<0": func(a *genpb.MarkovAutoInjection) { a.SetAnomalousFraction(-0.1) },
		"min<1":      func(a *genpb.MarkovAutoInjection) { a.SetMinActions(0) },
		"max<min":    func(a *genpb.MarkovAutoInjection) { a.SetMinActions(5); a.SetMaxActions(2) },
		"no refs":    func(a *genpb.MarkovAutoInjection) { a.SetAnomalyRefs(nil) },
		"neg weight": func(a *genpb.MarkovAutoInjection) { a.GetAnomalyRefs()[0].SetSelectionWeight(-1) },
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			auto := genpb.MarkovAutoInjection_builder{
				AnomalousFraction: 0.2,
				MinActions:        1,
				MaxActions:        3,
				AnomalyRefs: []*genpb.MarkovAutoAnomalyRef{
					genpb.MarkovAutoAnomalyRef_builder{AnomalyGraphPath: "anomaly.textpb"}.Build(),
				},
			}.Build()
			mutate(auto)
			cfg := parseProto(t, injectionBaseGraph, &genpb.QueryTemplateConfigs{})
			cfg.GetMarkovSampling().SetAutoInjection(auto)
			if err := validateMarkovAutoInjection(cfg.GetMarkovSampling()); err == nil {
				t.Fatalf("case %q: accepted invalid auto_injection", name)
			}
		})
	}
}

// TestShippedAutoInjectionExample loads the shipped auto_injection example config end-to-end
// against the real on-disk anomaly graph files, so relative-path resolution, validation, planning,
// and generation are all exercised exactly as in production.
func TestShippedAutoInjectionExample(t *testing.T) {
	path := filepath.Join("..", "..", "client", "markov_profiles", "standard", "auto_injection.textproto")
	if _, err := os.Stat(path); err != nil {
		t.Skipf("shipped example not found: %v", err)
	}
	cfg, err := loadConfigs(context.Background(), path)
	if err != nil {
		t.Fatalf("loadConfigs(%s): %v", path, err)
	}
	if cfg.GetMarkovSampling().GetAutoInjection() == nil {
		t.Fatal("example is not an auto_injection config")
	}
	var buf bytes.Buffer
	if err := GenerateLogsMarkov(cfg, 200, 0, rand.New(rand.NewSource(1)), &buf); err != nil {
		t.Fatalf("GenerateLogsMarkov: %v", err)
	}
	entries := decodeEntries(t, buf.Bytes())
	if len(entries) != 200 {
		t.Fatalf("got %d entries, want 200", len(entries))
	}
	anomalous := 0
	for _, e := range entries {
		if e.GetIsGroundTruthAnomalous() {
			anomalous++
		}
	}
	// budget = round(0.08*200) = 16; realized count must be in (0, 16].
	if anomalous == 0 || anomalous > 16 {
		t.Fatalf("anomalous count %d outside (0, 16]", anomalous)
	}
}
