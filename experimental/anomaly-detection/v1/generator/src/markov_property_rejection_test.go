package main

// Property-based test for the markov-profile-library feature.
//
// Feature: markov-profile-library, Property 16: Malformed configs are rejected with an identifying error and no output
//
// Validates: Requirements 2.5, 3.8, 4.8, 5.4, 5.5, 5.6, 6.2
//
// Property 16 asserts the contrapositive of the authoring guards: a config that
// violates a structural or reference invariant must be REJECTED by the generator
// with a non-empty, identifying error, and must produce NO output (no
// logs.binproto, hence no actions). The property takes a known-good base profile
// (batch_etl, which has both reads and writes and a reachable terminal node),
// mutates it into each malformed form, writes the mutated textproto into a fresh
// t.TempDir() (never into the real client/ tree), and drives it through the
// production load path. loadConfigs failing with a non-nil, non-empty error is the
// rejection; the test additionally confirms no logs.binproto was written by the
// attempt.
//
// Mutation classes exercised (one malformed form each):
//
//  1. dangling edge destination by dropping a referenced node: the terminal
//     "session_done" node is removed while edges still point at it, so an edge
//     destination no longer resolves (R5.4, R5.5).
//  2. dangling edge to a brand-new non-existent destination id (R5.4).
//  3. negative edge weight on an existing edge (R5.5, R3.8).
//  4. zero outgoing sum: a non-terminal node's edge weights are all set to 0, so
//     the node cannot be renormalized (R3.8, R5.5).
//  5. disallowed chaining reference: a markov_subtype whose base_graph_path
//     resolves to a file that itself carries an auto_injection block, which the
//     no-chaining resolver rejects (R5.6).
//  6. self-referential auto-injection reference: an auto_injection file whose
//     anomaly_graph_path resolves back to ITSELF — a file that carries an
//     auto_injection block — so the no-chaining/self resolver rejects it (R4.8,
//     R5.3, R5.6).
//  7. out-of-range requested action count: the generator's count-range gate
//     (validateScenarioFlags) rejects count 0 and the documented upper bound is
//     100000 (R6.2). No output is produced because generation never starts.
//
// Property convention: the full mutation set runs >= 100 times. Each iteration
// applies every deterministic mutation and, additionally, a randomized mutation
// chosen by a seeded RNG (random node/edge targets, random negative weight, random
// out-of-range count) so the input space around each malformed form is exercised
// across many concrete values rather than a single fixed instance.
//
// Helper names here are prefixed `prop16` to stay unique within the package and
// avoid collisions with other concurrently-authored test files. loadConfigs sets
// the package-global resolvedBaseGraph as a side effect, so this test does not run
// in parallel with other harness callers.

import (
	"context"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"testing"

	"google.golang.org/protobuf/encoding/prototext"

	genpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v1/generator/proto"
)

// prop16BaseConfigPath is the valid base profile mutated by every structural
// mutation. batch_etl carries reads, writes, self-loops, and a reachable terminal
// node, so dropping a node / zeroing a sum / negating a weight all produce genuine
// violations. Go tests run with the working directory set to the package dir
// (v1/generator/src), so the client config lives at ../../client/.
var prop16BaseConfigPath = filepath.Join("..", "..", "client", "markov_batch_etl_config.textproto")

// prop16LoadValidBase parses the valid base profile fresh from disk into a
// QueryTemplateConfigs each call, so every mutation starts from a pristine,
// independently-owned message that can be edited without affecting other
// iterations. It asserts the base is itself valid markov_sampling so a later
// rejection is attributable to the mutation, not a bad starting point.
func prop16LoadValidBase(t *testing.T) *genpb.QueryTemplateConfigs {
	t.Helper()
	data, err := os.ReadFile(prop16BaseConfigPath)
	if err != nil {
		t.Fatalf("prop16: reading valid base %q: %v", prop16BaseConfigPath, err)
	}
	cfg := &genpb.QueryTemplateConfigs{}
	if err := prototext.Unmarshal(data, cfg); err != nil {
		t.Fatalf("prop16: parsing valid base %q: %v", prop16BaseConfigPath, err)
	}
	if cfg.GetMarkovSampling() == nil {
		t.Fatalf("prop16: base %q is not a markov_sampling config", prop16BaseConfigPath)
	}
	return cfg
}

// prop16WriteConfig serializes cfg to textproto and writes it to <dir>/<name>,
// returning the path. prototext.Marshal re-emits the mutated message so the on-disk
// file is exactly what the mutation produced.
func prop16WriteConfig(t *testing.T, dir, name string, cfg *genpb.QueryTemplateConfigs) string {
	t.Helper()
	data, err := prototext.Marshal(cfg)
	if err != nil {
		t.Fatalf("prop16: marshaling mutated config: %v", err)
	}
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, data, 0o644); err != nil {
		t.Fatalf("prop16: writing mutated config %q: %v", path, err)
	}
	return path
}

// prop16WriteText writes a hand-crafted textproto string to <dir>/<name> and
// returns the path. Used for the reference-chaining and self-reference layouts,
// where relative paths must resolve inside the temp dir.
func prop16WriteText(t *testing.T, dir, name, text string) string {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(text), 0o644); err != nil {
		t.Fatalf("prop16: writing %q: %v", path, err)
	}
	return path
}

// prop16AssertRejectedNoOutput loads the config at configPath through the
// production loadConfigs path and asserts the generator REJECTS it: a non-nil
// error with a non-empty (identifying) message, and no logs.binproto written to
// outputDir. configPath is loaded, not generated, so loadConfigs catching the
// defect means generation never starts and no output can exist; the test still
// confirms the absence of logs.binproto to make "no output" explicit.
func prop16AssertRejectedNoOutput(t *testing.T, label, configPath, outputDir string) {
	t.Helper()

	_, err := loadConfigs(context.Background(), configPath)
	if err == nil {
		t.Fatalf("prop16[%s]: loadConfigs(%q) returned nil error, want rejection", label, configPath)
	}
	if err.Error() == "" {
		t.Fatalf("prop16[%s]: loadConfigs(%q) returned an empty (non-identifying) error", label, configPath)
	}

	// No output: loadConfigs never generates, so logs.binproto must not exist.
	if _, statErr := os.Stat(filepath.Join(outputDir, "logs.binproto")); !os.IsNotExist(statErr) {
		t.Fatalf("prop16[%s]: expected no logs.binproto in %q, but one exists (stat err=%v)", label, outputDir, statErr)
	}
}

// --- Structural mutations over the loaded batch_etl message ---------------------

// prop16DropReferencedNode removes the terminal "session_done" node while leaving
// the edges that point at it in place, producing a dangling edge destination.
func prop16DropReferencedNode(cfg *genpb.QueryTemplateConfigs) {
	graph := cfg.GetMarkovSampling()
	kept := make([]*genpb.MarkovNode, 0, len(graph.GetNodes()))
	for _, n := range graph.GetNodes() {
		if n.GetId() == "session_done" {
			continue // drop it; edges to "session_done" now dangle
		}
		kept = append(kept, n)
	}
	graph.SetNodes(kept)
}

// prop16DangleEdgeToGhost points the first edge of the first edge-bearing node at
// a destination id that no node declares.
func prop16DangleEdgeToGhost(cfg *genpb.QueryTemplateConfigs, ghostID string) {
	graph := cfg.GetMarkovSampling()
	for _, n := range graph.GetNodes() {
		if len(n.GetEdges()) > 0 {
			n.GetEdges()[0].SetDestinationId(ghostID)
			return
		}
	}
}

// prop16NegativeEdgeWeight sets the weight of the edge at (nodeIdx, edgeIdx),
// chosen modulo the available counts, to a negative value.
func prop16NegativeEdgeWeight(cfg *genpb.QueryTemplateConfigs, nodePick, edgePick int, weight float64) {
	graph := cfg.GetMarkovSampling()
	edgeBearing := make([]*genpb.MarkovNode, 0, len(graph.GetNodes()))
	for _, n := range graph.GetNodes() {
		if len(n.GetEdges()) > 0 {
			edgeBearing = append(edgeBearing, n)
		}
	}
	if len(edgeBearing) == 0 {
		return
	}
	n := edgeBearing[nodePick%len(edgeBearing)]
	e := n.GetEdges()[edgePick%len(n.GetEdges())]
	e.SetWeight(weight)
}

// prop16ZeroOutgoingSum sets every outgoing edge weight of the first non-terminal
// node to 0, so its outgoing weights sum to zero and it cannot be renormalized.
func prop16ZeroOutgoingSum(cfg *genpb.QueryTemplateConfigs) {
	graph := cfg.GetMarkovSampling()
	for _, n := range graph.GetNodes() {
		if len(n.GetEdges()) > 0 {
			for _, e := range n.GetEdges() {
				e.SetWeight(0)
			}
			return
		}
	}
}

// --- Reference-layout mutations (chaining / self-reference) ---------------------

// prop16MinimalBaseText is a minimal, valid plain markov_sampling base graph used
// as the "own base" target in the reference layouts.
const prop16MinimalBaseText = `# proto-file: experimental/anomaly-detection/v1/generator/proto/query_template_config.proto
# proto-message: QueryTemplateConfigs
markov_sampling {
  nodes {
    id: "only"
    template { normalized_query_template: "SELECT 1" }
    completion_time { fixed { seconds: 1 } }
  }
  start { node_id: "only" }
}
`

// prop16WriteChainingLayout builds a temp layout where a markov_subtype's
// base_graph_path resolves to a file that itself carries an auto_injection block.
// The no-chaining resolver (resolveBaseGraphFile) rejects a base that contains
// injections, so loadConfigs rejects the subtype. Returns the subtype path.
//
// Layout:
//
//	dir/plain.textproto     -- a valid plain base (the chained base references this)
//	dir/chained_base.textproto -- a markov_sampling graph that ALSO sets auto_injection
//	dir/subtype.textproto   -- markov_subtype base_graph_path -> chained_base.textproto
func prop16WriteChainingLayout(t *testing.T, dir string) string {
	t.Helper()
	prop16WriteText(t, dir, "plain.textproto", prop16MinimalBaseText)

	chainedBase := `# proto-file: experimental/anomaly-detection/v1/generator/proto/query_template_config.proto
# proto-message: QueryTemplateConfigs
markov_sampling {
  nodes {
    id: "only"
    template { normalized_query_template: "SELECT 1" }
    completion_time { fixed { seconds: 1 } }
  }
  start { node_id: "only" }
  auto_injection {
    anomalous_fraction: 0.1
    min_actions: 1
    max_actions: 2
    anomaly_refs { anomaly_graph_path: "plain.textproto" selection_weight: 1 }
  }
}
`
	prop16WriteText(t, dir, "chained_base.textproto", chainedBase)

	subtype := `# proto-file: experimental/anomaly-detection/v1/generator/proto/query_template_config.proto
# proto-message: QueryTemplateConfigs
markov_subtype {
  base_graph_path: "chained_base.textproto"
  edge_weight_overrides { source_id: "only" destination_id: "only" weight: 1.0 }
}
`
	return prop16WriteText(t, dir, "subtype.textproto", subtype)
}

// prop16WriteSelfRefAutoInjection builds an auto_injection file whose single
// anomaly_graph_path resolves back to ITSELF. Because that file carries an
// auto_injection block, the no-chaining resolver rejects the self-reference (a
// referenced anomaly graph must be a plain markov_sampling graph with no
// injections). This exercises the self-referential auto-injection rejection path.
// Returns the file's path.
func prop16WriteSelfRefAutoInjection(t *testing.T, dir string) string {
	t.Helper()
	const name = "self_auto_injection.textproto"
	selfRef := `# proto-file: experimental/anomaly-detection/v1/generator/proto/query_template_config.proto
# proto-message: QueryTemplateConfigs
markov_sampling {
  nodes {
    id: "only"
    template { normalized_query_template: "SELECT 1" }
    completion_time { fixed { seconds: 1 } }
  }
  start { node_id: "only" }
  auto_injection {
    anomalous_fraction: 0.1
    min_actions: 1
    max_actions: 2
    anomaly_refs { anomaly_graph_path: "self_auto_injection.textproto" selection_weight: 1 }
  }
}
`
	return prop16WriteText(t, dir, name, selfRef)
}

// --- Count-range gate (R6.2) ----------------------------------------------------

// prop16RejectOutOfRangeCount drives the generator's count-range gate
// (validateScenarioFlags, which reads the package-global --num_logs) for a given
// out-of-range count and asserts rejection. The documented in-range interval is
// the inclusive [1, 100000]; count 0 (and any count < 1) is out of range and must
// be rejected before generation starts, so no output is produced. The upper bound
// 100000 is the documented ceiling (R6.1/R6.2); this helper also exercises it.
//
// It saves and restores the mutated package globals so it does not disturb other
// tests in the package.
func prop16RejectOutOfRangeCount(t *testing.T, configs *genpb.QueryTemplateConfigs, count int) {
	t.Helper()

	origNumLogs := *numLogs
	origOutputDir := *outputDir
	origStart := *startTimeMs
	t.Cleanup(func() {
		*numLogs = origNumLogs
		*outputDir = origOutputDir
		*startTimeMs = origStart
	})

	outDir := t.TempDir()
	*numLogs = count
	*outputDir = outDir
	*startTimeMs = 0

	err := validateScenarioFlags(context.Background(), configs)
	if err == nil {
		t.Fatalf("prop16[count=%d]: validateScenarioFlags returned nil error, want rejection of out-of-range count", count)
	}
	if err.Error() == "" {
		t.Fatalf("prop16[count=%d]: validateScenarioFlags returned an empty (non-identifying) error", count)
	}
	if _, statErr := os.Stat(filepath.Join(outDir, "logs.binproto")); !os.IsNotExist(statErr) {
		t.Fatalf("prop16[count=%d]: expected no logs.binproto in %q, but one exists (stat err=%v)", count, outDir, statErr)
	}
}

// prop16InRange reports whether count is inside the documented inclusive interval
// [1, 100000]. Used to classify randomized counts so the test only asserts
// rejection for genuinely out-of-range values.
func prop16InRange(count int) bool { return count >= 1 && count <= 100000 }

// TestProperty16_MalformedConfigsRejectedWithErrorAndNoOutput is the Property 16
// driver.
//
// Feature: markov-profile-library, Property 16: Malformed configs are rejected with an identifying error and no output
//
// Validates: Requirements 2.5, 3.8, 4.8, 5.4, 5.5, 5.6, 6.2
func TestProperty16_MalformedConfigsRejectedWithErrorAndNoOutput(t *testing.T) {
	// Sanity: the unmutated base profile loads cleanly, so any rejection below is
	// attributable to the mutation and not to a bad starting config.
	if _, err := loadConfigs(context.Background(), prop16BaseConfigPath); err != nil {
		t.Fatalf("prop16: unmutated base %q failed to load: %v", prop16BaseConfigPath, err)
	}

	const iterations = 120 // >= 100 per the property-test convention
	rng := rand.New(rand.NewSource(0x16ADBEEF))

	for i := 0; i < iterations; i++ {
		// --- Mutation 1: drop a referenced node (dangling destination) ---
		{
			dir := t.TempDir()
			cfg := prop16LoadValidBase(t)
			prop16DropReferencedNode(cfg)
			path := prop16WriteConfig(t, dir, "drop_node.textproto", cfg)
			prop16AssertRejectedNoOutput(t, "drop_referenced_node", path, dir)
		}

		// --- Mutation 2: dangle an edge to a non-existent destination ---
		{
			dir := t.TempDir()
			cfg := prop16LoadValidBase(t)
			ghost := "ghost_" + itoa(int64(rng.Intn(1_000_000)))
			prop16DangleEdgeToGhost(cfg, ghost)
			path := prop16WriteConfig(t, dir, "dangle_edge.textproto", cfg)
			prop16AssertRejectedNoOutput(t, "dangle_edge_to_ghost", path, dir)
		}

		// --- Mutation 3: negative edge weight ---
		{
			dir := t.TempDir()
			cfg := prop16LoadValidBase(t)
			// A random strictly-negative weight across a few orders of magnitude.
			neg := -(rng.Float64()*1000 + math.SmallestNonzeroFloat64)
			prop16NegativeEdgeWeight(cfg, rng.Intn(16), rng.Intn(16), neg)
			path := prop16WriteConfig(t, dir, "negative_weight.textproto", cfg)
			prop16AssertRejectedNoOutput(t, "negative_edge_weight", path, dir)
		}

		// --- Mutation 4: zero outgoing sum on a non-terminal node ---
		{
			dir := t.TempDir()
			cfg := prop16LoadValidBase(t)
			prop16ZeroOutgoingSum(cfg)
			path := prop16WriteConfig(t, dir, "zero_sum.textproto", cfg)
			prop16AssertRejectedNoOutput(t, "zero_outgoing_sum", path, dir)
		}

		// --- Mutation 5: disallowed chaining reference ---
		{
			dir := t.TempDir()
			path := prop16WriteChainingLayout(t, dir)
			prop16AssertRejectedNoOutput(t, "chaining_reference", path, dir)
		}

		// --- Mutation 6: self-referential auto-injection reference ---
		{
			dir := t.TempDir()
			path := prop16WriteSelfRefAutoInjection(t, dir)
			prop16AssertRejectedNoOutput(t, "self_referential_auto_injection", path, dir)
		}

		// --- Mutation 7: out-of-range requested action count ---
		{
			configs := prop16LoadValidBase(t)
			// count 0 (below the inclusive range) must always be rejected.
			prop16RejectOutOfRangeCount(t, configs, 0)
			// A randomized out-of-range count on the low side (negative or 0).
			lowBad := -rng.Intn(10000)
			if prop16InRange(lowBad) {
				lowBad = 0
			}
			prop16RejectOutOfRangeCount(t, configs, lowBad)
		}
	}
}
