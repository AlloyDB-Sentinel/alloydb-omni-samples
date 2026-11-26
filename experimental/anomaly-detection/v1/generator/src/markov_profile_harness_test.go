package main

// Verification harness for the markov-profile-library feature.
//
// This file adds reusable Go test helpers that let later property/example tests
// in this package load, validate, and sample the new workload profiles through
// the existing, unchanged generator. It introduces no production-code changes and
// no proto edits; it only wraps loadConfigs / validateConfigs /
// validateMarkovSampling and GenerateLogs for inspection in tests.
//
// Two helpers are provided:
//
//   - loadProfile(t, path): parses + resolves references + runs validateConfigs
//     (exactly as production loadConfigs does) and, for plain markov_sampling base
//     graphs, re-checks with validateMarkovSampling. It returns a structured
//     profileResult (scenario kind, node set, edges, resolved base) that later
//     tests can assert against without re-walking the proto each time.
//
//   - sampleProfile(t, path, numLogs, seed): runs the full production
//     load -> GenerateLogs path into a temp dir, reads back logs.binproto, and
//     returns the decoded actions plus per-action read/write classification and
//     the simulated-clock timestamps, so samples can be inspected for read/write
//     mix, action count, and action rate.
//
// Manual inspection outside the Go tests uses the generator binary directly. Build
// and run it against any profile, pointing --config_file at an absolute path and
// choosing an output dir, count, and seed, e.g.:
//
//	go build -o /tmp/generator ./v1/generator/src
//	/tmp/generator \
//	    --config_file=<abs path to markov_<name>_config.textproto> \
//	    --output_dir=<dir> \
//	    --num_logs=<n> \
//	    --seed=<s>
//
// The run writes <dir>/logs.binproto, a stream of length-delimited AuditLogEntry
// records that the helpers here (and the other tests in this package) decode with
// protodelim.UnmarshalFrom.

import (
	"bufio"
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"google.golang.org/protobuf/encoding/protodelim"

	genpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v1/generator/proto"
)

// scenarioKind names which QueryTemplateConfigs.scenario oneof a loaded profile
// sets, so tests can branch on profile shape without re-inspecting the proto.
type scenarioKind string

const (
	scenarioMarkovSampling   scenarioKind = "markov_sampling"
	scenarioMarkovSubtype    scenarioKind = "markov_subtype"
	scenarioMarkovIndividual scenarioKind = "markov_individual"
	scenarioOther            scenarioKind = "other"
)

// profileEdge is one resolved outgoing transition: a (source, destination, weight)
// triple pulled from a node's edges.
type profileEdge struct {
	SourceID      string
	DestinationID string
	Weight        float64
}

// profileResult is the structured view of a loaded-and-validated profile, reused
// by later property/example tests so each test does not re-walk the proto.
//
// For a plain markov_sampling base profile, Graph is the config's own graph and
// ResolvedBase is nil. For a markov_subtype / markov_individual variant, Graph is
// nil and ResolvedBase is the base graph loadConfigs resolved and inlined from the
// variant's base_graph_path (the package-global resolvedBaseGraph captured at load
// time). NodeIDs, Edges, and HasInjection describe whichever graph is present.
type profileResult struct {
	// Path is the on-disk path the profile was loaded from.
	Path string
	// Kind is the scenario oneof the loaded config sets.
	Kind scenarioKind
	// Configs is the raw loaded QueryTemplateConfigs, for tests needing more detail.
	Configs *genpb.QueryTemplateConfigs
	// Graph is the plain markov_sampling graph for base profiles; nil for variants.
	Graph *genpb.MarkovSamplingConfig
	// ResolvedBase is the base graph resolved from a variant's base_graph_path; nil
	// for plain base profiles.
	ResolvedBase *genpb.MarkovSamplingConfig
	// NodeIDs lists the node ids of the describing graph, in declaration order.
	NodeIDs []string
	// Edges lists every outgoing transition of the describing graph.
	Edges []profileEdge
	// HasInjection is true when the config carries injections or an auto_injection
	// block (i.e. it is an auto-injection variant, not a plain base profile).
	HasInjection bool
}

// describingGraph returns the graph a profileResult's NodeIDs/Edges describe: the
// profile's own markov_sampling graph for base profiles, or the resolved base for
// variants.
func (r *profileResult) describingGraph() *genpb.MarkovSamplingConfig {
	if r.Graph != nil {
		return r.Graph
	}
	return r.ResolvedBase
}

// loadProfile loads the profile at path through the production loadConfigs path
// (parse + resolve base/anomaly references + validateConfigs) and, for plain
// markov_sampling base graphs, re-checks the graph with validateMarkovSampling. It
// fails the test on any load/validation error and otherwise returns the structured
// profileResult.
//
// loadConfigs sets the package-global resolvedBaseGraph as a side effect, so tests
// that call loadProfile must not run in parallel with one another.
func loadProfile(t *testing.T, path string) *profileResult {
	t.Helper()

	configs, err := loadConfigs(context.Background(), path)
	if err != nil {
		t.Fatalf("loadProfile: loadConfigs(%q) failed: %v", path, err)
	}

	res := &profileResult{
		Path:    path,
		Configs: configs,
		Kind:    scenarioOther,
	}

	switch {
	case configs.GetMarkovSampling() != nil:
		res.Kind = scenarioMarkovSampling
		res.Graph = configs.GetMarkovSampling()
		res.HasInjection = len(res.Graph.GetInjections()) > 0 || res.Graph.GetAutoInjection() != nil
		// Re-check the base graph independently of loadConfigs' validateConfigs
		// dispatch, so a regression in either path surfaces here.
		if err := validateMarkovSampling(res.Graph); err != nil {
			t.Fatalf("loadProfile: validateMarkovSampling(%q) failed: %v", path, err)
		}
	case configs.GetMarkovSubtype() != nil:
		res.Kind = scenarioMarkovSubtype
		// loadConfigs stashed the resolved base in the package global.
		res.ResolvedBase = resolvedBaseGraph
	case configs.GetMarkovIndividual() != nil:
		res.Kind = scenarioMarkovIndividual
		res.ResolvedBase = resolvedBaseGraph
	}

	graph := res.describingGraph()
	if graph != nil {
		for _, node := range graph.GetNodes() {
			res.NodeIDs = append(res.NodeIDs, node.GetId())
			for _, edge := range node.GetEdges() {
				res.Edges = append(res.Edges, profileEdge{
					SourceID:      node.GetId(),
					DestinationID: edge.GetDestinationId(),
					Weight:        edge.GetWeight(),
				})
			}
		}
	}

	return res
}

// sampledAction is one decoded action from a bounded sample: the emitted
// AuditLogEntry plus derived fields useful for inspecting a profile's behavior.
type sampledAction struct {
	// Entry is the raw decoded record.
	Entry *genpb.AuditLogEntry
	// IsWrite is true when the action's template is a data-modifying statement
	// (INSERT / UPDATE / DELETE, or DDL such as CREATE / ALTER / DROP / TRUNCATE /
	// VACUUM), following the feature's read/write classification; everything else
	// (SELECT, SELECT 1) is a read.
	IsWrite bool
	// TimestampMs is the simulated-clock timestamp of the action.
	TimestampMs int64
}

// sampleResult holds a bounded sample and a few aggregate views over it.
type sampleResult struct {
	Actions    []sampledAction
	WriteCount int
	ReadCount  int
}

// WriteFraction reports the observed proportion of write actions in the sample,
// or 0 for an empty sample.
func (s *sampleResult) WriteFraction() float64 {
	if len(s.Actions) == 0 {
		return 0
	}
	return float64(s.WriteCount) / float64(len(s.Actions))
}

// ElapsedMs reports the span of the simulated clock across the sample
// (last timestamp minus first), or 0 for fewer than two actions. Because the
// generator advances a single non-decreasing simulated clock by each node's
// completion time, this is the simulated wall-time the sample would have taken and
// is the basis for an observed action-rate check.
func (s *sampleResult) ElapsedMs() int64 {
	if len(s.Actions) < 2 {
		return 0
	}
	return s.Actions[len(s.Actions)-1].TimestampMs - s.Actions[0].TimestampMs
}

// sampleProfile runs the profile at path end-to-end through the production
// load -> GenerateLogs path for numLogs actions under seed, writing into a fresh
// temp dir, then reads logs.binproto back and returns the decoded actions with
// read/write classification and simulated-clock timestamps. It fails the test on
// any load, generation, or decode error.
//
// Like loadProfile, this calls loadConfigs and so must not run in parallel with
// other harness callers.
func sampleProfile(t *testing.T, path string, numLogs int, seed int64) *sampleResult {
	t.Helper()

	ctx := context.Background()
	configs, err := loadConfigs(ctx, path)
	if err != nil {
		t.Fatalf("sampleProfile: loadConfigs(%q) failed: %v", path, err)
	}

	outputDir := t.TempDir()
	if err := GenerateLogs(ctx, configs, numLogs, seed, outputDir); err != nil {
		t.Fatalf("sampleProfile: GenerateLogs(%q, numLogs=%d, seed=%d) failed: %v", path, numLogs, seed, err)
	}

	entries := readSampleEntries(t, outputDir)

	res := &sampleResult{}
	for _, entry := range entries {
		write := isWriteTemplate(entry.GetNormalizedQueryTemplate())
		if write {
			res.WriteCount++
		} else {
			res.ReadCount++
		}
		res.Actions = append(res.Actions, sampledAction{
			Entry:       entry,
			IsWrite:     write,
			TimestampMs: entry.GetTimestampMs(),
		})
	}
	return res
}

// readSampleEntries opens <outputDir>/logs.binproto and decodes every
// length-delimited AuditLogEntry until io.EOF, matching how the other scenario
// tests in this package read their output back.
func readSampleEntries(t *testing.T, outputDir string) []*genpb.AuditLogEntry {
	t.Helper()

	f, err := os.Open(filepath.Join(outputDir, "logs.binproto"))
	if err != nil {
		t.Fatalf("readSampleEntries: failed to open logs.binproto: %v", err)
	}
	defer f.Close()

	var entries []*genpb.AuditLogEntry
	r := bufio.NewReader(f)
	for {
		entry := &genpb.AuditLogEntry{}
		if err := protodelim.UnmarshalFrom(r, entry); err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			t.Fatalf("readSampleEntries: failed to decode log entry: %v", err)
		}
		entries = append(entries, entry)
	}
	return entries
}

// isWriteTemplate classifies a normalized query template as a write (data-modifying
// or DDL) per the feature glossary, versus a read. Classification keys on the first
// SQL keyword so "SELECT 1" and "SELECT ... FROM ..." are reads while
// "INSERT ...", "UPDATE ...", "DELETE ..." and DDL are writes.
func isWriteTemplate(template string) bool {
	fields := strings.Fields(strings.TrimSpace(template))
	if len(fields) == 0 {
		return false
	}
	switch strings.ToUpper(fields[0]) {
	case "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP", "TRUNCATE", "VACUUM", "MERGE", "UPSERT":
		return true
	default:
		return false
	}
}
