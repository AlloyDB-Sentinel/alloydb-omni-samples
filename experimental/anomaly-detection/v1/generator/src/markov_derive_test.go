package main

import (
	"bytes"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/testing/protocmp"

	genpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v1/generator/proto"
)

// writeTempProto writes text into a file named name inside a fresh temp dir and
// returns the full path. Writing real fixture files lets resolveBaseGraphFile
// exercise its relative-path resolution and os.ReadFile error handling exactly
// as it would at load time.
func writeTempProto(t *testing.T, dir, name, text string) string {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(text), 0o644); err != nil {
		t.Fatalf("failed to write temp proto %q: %v", path, err)
	}
	return path
}

// A valid, independently loadable markov_sampling base graph: a two-node chain
// start -> end where end is terminal.
const validBaseGraph = `
markov_sampling {
	nodes: {
		id: "start"
		template: { normalized_query_template: "SELECT 1" }
		completion_time: { fixed: { nanos: 1000000 } }
		edges: { destination_id: "end" weight: 1.0 }
	}
	nodes: {
		id: "end"
		template: { normalized_query_template: "SELECT 2" }
		completion_time: { fixed: { nanos: 1000000 } }
	}
	start: { node_id: "start" }
}`

// TestResolveBaseGraphFile verifies base-graph file resolution relative to the
// variant file's directory: a valid base resolves and validates, a missing file
// surfaces the "could not be read" error, a referenced non-markov_sampling
// scenario (references never chain) surfaces the "is not a markov_sampling
// config" error, and an invalid base propagates validateMarkovSampling's error.
//
// Validates R12.2, R13.2, R16.1
func TestResolveBaseGraphFile(t *testing.T) {
	// A base whose only non-terminal node's outgoing weights do not sum to 1.0,
	// so validateMarkovSampling rejects it and resolveBaseGraphFile wraps that.
	const invalidBaseGraph = `
markov_sampling {
	nodes: {
		id: "start"
		template: { normalized_query_template: "SELECT 1" }
		completion_time: { fixed: { nanos: 1000000 } }
		edges: { destination_id: "end" weight: 0.5 }
	}
	nodes: {
		id: "end"
		template: { normalized_query_template: "SELECT 2" }
		completion_time: { fixed: { nanos: 1000000 } }
	}
	start: { node_id: "start" }
}`

	// A subtype config (not a markov_sampling scenario) referenced as a base must
	// be rejected so base references can never chain.
	const subtypeAsBase = `
markov_subtype {
	base_graph_path: "whatever.textpb"
	edge_weight_overrides: { source_id: "start" destination_id: "end" weight: 1.0 }
}`

	tests := []struct {
		name string
		// baseContent is written to base.textpb when non-empty; when empty the
		// base file is intentionally absent to exercise the read error.
		baseContent string
		// wantErrSub, when non-empty, is a substring the returned error must
		// contain; when empty the call must succeed.
		wantErrSub string
	}{
		{
			name:        "valid_base_resolves",
			baseContent: validBaseGraph,
		},
		{
			name:       "missing_file",
			wantErrSub: "could not be read",
		},
		{
			name:        "non_markov_sampling_scenario",
			baseContent: subtypeAsBase,
			wantErrSub:  "is not a markov_sampling config",
		},
		{
			name:        "invalid_base_propagates_validation_error",
			baseContent: invalidBaseGraph,
			wantErrSub:  "is invalid",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			dir := t.TempDir()

			// Always create the variant file beside the (maybe-absent) base file,
			// so the relative base_graph_path resolves against the same directory.
			variantPath := writeTempProto(t, dir, "variant.textpb", `
markov_subtype {
	base_graph_path: "base.textpb"
}`)
			if test.baseContent != "" {
				writeTempProto(t, dir, "base.textpb", test.baseContent)
			}

			got, err := resolveBaseGraphFile(variantPath, "base.textpb")

			if test.wantErrSub != "" {
				if err == nil {
					t.Fatalf("resolveBaseGraphFile() error = nil, want error containing %q", test.wantErrSub)
				}
				if !strings.Contains(err.Error(), test.wantErrSub) {
					t.Errorf("resolveBaseGraphFile() error = %q, want substring %q", err.Error(), test.wantErrSub)
				}
				return
			}

			if err != nil {
				t.Fatalf("resolveBaseGraphFile() error = %v, want nil", err)
			}
			if got == nil {
				t.Fatal("resolveBaseGraphFile() returned nil config, want resolved base graph")
			}
			if n := len(got.GetNodes()); n != 2 {
				t.Errorf("resolved base has %d nodes, want 2", n)
			}
			if id := got.GetStart().GetNodeId(); id != "start" {
				t.Errorf("resolved base start node = %q, want %q", id, "start")
			}
		})
	}
}

// TestRenormalizeOutgoing verifies that after renormalization every non-terminal
// node's outgoing weights sum to 1.0 within FloatTolerance and each weight equals
// its pre-renorm value scaled by the reciprocal of the node's pre-renorm sum;
// terminal nodes are left untouched; and a non-positive-sum node is rejected.
//
// Property 20, Validates R12.7, R15.3, R15.5
func TestRenormalizeOutgoing(t *testing.T) {
	t.Run("scales_to_unit_sum_and_preserves_ratios", func(t *testing.T) {
		// A branching node whose weights sum to 4.0, a self-looping node whose
		// weights sum to 0.5, and a terminal node with no edges.
		nodes := []*genpb.MarkovNode{
			newNode("a", []*genpb.MarkovEdge{newEdge("b", 1.0), newEdge("c", 3.0)}), // sum 4.0
			newNode("b", []*genpb.MarkovEdge{newEdge("b", 0.2), newEdge("c", 0.3)}), // sum 0.5, self-loop
			newNode("c", nil), // terminal
		}

		// Capture each node's pre-renorm sum so we can assert exact scaling.
		preSum := map[string]float64{}
		for _, n := range nodes {
			var s float64
			for _, e := range n.GetEdges() {
				s += e.GetWeight()
			}
			preSum[n.GetId()] = s
		}
		// Snapshot pre-renorm weights keyed by (node, dst).
		type key struct{ node, dst string }
		pre := map[key]float64{}
		for _, n := range nodes {
			for _, e := range n.GetEdges() {
				pre[key{n.GetId(), e.GetDestinationId()}] = e.GetWeight()
			}
		}

		if err := renormalizeOutgoing(nodes); err != nil {
			t.Fatalf("renormalizeOutgoing() error = %v, want nil", err)
		}

		for _, n := range nodes {
			edges := n.GetEdges()
			if len(edges) == 0 {
				// Terminal node: nothing to check (left untouched).
				continue
			}

			var sum float64
			for _, e := range edges {
				sum += e.GetWeight()
			}
			if math.Abs(sum-1.0) > FloatTolerance {
				t.Errorf("node %q outgoing sum = %v, want 1.0 within %v", n.GetId(), sum, FloatTolerance)
			}

			scale := 1.0 / preSum[n.GetId()]
			for _, e := range edges {
				want := pre[key{n.GetId(), e.GetDestinationId()}] * scale
				if math.Abs(e.GetWeight()-want) > FloatTolerance {
					t.Errorf("node %q edge -> %q weight = %v, want %v (pre * 1/sum)",
						n.GetId(), e.GetDestinationId(), e.GetWeight(), want)
				}
			}
		}
	})

	t.Run("rejects_non_positive_sum", func(t *testing.T) {
		// A non-terminal node whose outgoing weights sum to zero cannot be
		// renormalized to 1.0 and must be rejected by id.
		nodes := []*genpb.MarkovNode{
			newNode("zero", []*genpb.MarkovEdge{newEdge("x", 0.0)}),
		}
		err := renormalizeOutgoing(nodes)
		if err == nil {
			t.Fatal("renormalizeOutgoing() error = nil, want non-positive-sum error")
		}
		if !strings.Contains(err.Error(), "zero") || !strings.Contains(err.Error(), "must be positive to renormalize") {
			t.Errorf("renormalizeOutgoing() error = %q, want it to name node %q and %q",
				err.Error(), "zero", "must be positive to renormalize")
		}
	})
}

// newEdge constructs a MarkovEdge via the opaque-API setters, matching how the
// derivation code mutates edges.
func newEdge(dst string, weight float64) *genpb.MarkovEdge {
	e := &genpb.MarkovEdge{}
	e.SetDestinationId(dst)
	e.SetWeight(weight)
	return e
}

// newNode constructs a MarkovNode with the given id and outgoing edges via the
// opaque-API setters.
func newNode(id string, edges []*genpb.MarkovEdge) *genpb.MarkovNode {
	n := &genpb.MarkovNode{}
	n.SetId(id)
	if edges != nil {
		n.SetEdges(edges)
	}
	return n
}

// A three-node base graph used by the subtype derivation tests. Node "a" is a
// non-terminal branching node whose two outgoing edges sum to exactly 1.0, "b"
// is a self-looping non-terminal node whose edges also sum to 1.0, and "c" is a
// terminal node. Overrides chosen in the tests below keep each node's outgoing
// sum at 1.0 so renormalization is a no-op and the overridden/non-overridden
// weights are directly observable (pre-renormalization fidelity).
const subtypeBaseGraph = `
markov_sampling {
	nodes: {
		id: "a"
		template: { normalized_query_template: "SELECT 1" }
		completion_time: { fixed: { nanos: 2000000 } }
		edges: { destination_id: "b" weight: 0.6 }
		edges: { destination_id: "c" weight: 0.4 }
	}
	nodes: {
		id: "b"
		template: {
			normalized_query_template: "SELECT * FROM t WHERE id = %v"
			placeholder_values: { int_placeholder: { min: 1 max: 10 } }
		}
		completion_time: { range: { min: { nanos: 1000000 } max: { nanos: 5000000 } } }
		edges: { destination_id: "b" weight: 0.3 }
		edges: { destination_id: "c" weight: 0.7 }
	}
	nodes: {
		id: "c"
		template: { normalized_query_template: "SELECT 3" }
		completion_time: { fixed: { nanos: 3000000 } }
	}
	start: { node_id: "a" }
}`

// parseSubtypeBase parses subtypeBaseGraph into its MarkovSamplingConfig base
// graph via GetMarkovSampling(), the resolved base that deriveSubtypeGraph and
// validateMarkovSampling consume.
func parseSubtypeBase(t *testing.T) *genpb.MarkovSamplingConfig {
	t.Helper()
	cfgs := parseProto(t, subtypeBaseGraph, &genpb.QueryTemplateConfigs{})
	base := cfgs.GetMarkovSampling()
	if base == nil {
		t.Fatal("subtypeBaseGraph did not parse into a markov_sampling config")
	}
	return base
}

// parseSubtype parses a markov_subtype QueryTemplateConfigs textproto into its
// MarkovSubtypeConfig via GetMarkovSubtype().
func parseSubtype(t *testing.T, text string) *genpb.MarkovSubtypeConfig {
	t.Helper()
	cfgs := parseProto(t, text, &genpb.QueryTemplateConfigs{})
	sub := cfgs.GetMarkovSubtype()
	if sub == nil {
		t.Fatal("subtype textproto did not parse into a markov_subtype config")
	}
	return sub
}

// TestDeriveSubtypeGraph_Fidelity verifies that subtype derivation preserves the
// base graph's structure (node set, ids, templates, completion times) and that
// overridden edges carry the new weight while non-overridden edges are unchanged.
//
// Overrides are chosen so each affected node's outgoing weights still sum to 1.0,
// making renormalization a no-op; the exact pre-renormalization weights are then
// directly observable on the derived graph. The test also asserts the base graph
// passed in is NOT mutated, confirming deriveSubtypeGraph clones.
//
// Property 19, Validates R12.3, R12.5, R12.6
func TestDeriveSubtypeGraph_Fidelity(t *testing.T) {
	base := parseSubtypeBase(t)

	// Snapshot the base before derivation so we can assert it is left untouched.
	baseBefore := proto.Clone(base).(*genpb.MarkovSamplingConfig)

	// Override one edge on each non-terminal node, keeping each node's sum at 1.0:
	//   a: b 0.6 -> 0.9, c left at 0.4 would break the sum, so override c too (0.1).
	//   b: b 0.3 -> 0.5, c 0.7 -> 0.5.
	sub := parseSubtype(t, `
markov_subtype {
	base_graph_path: "base.textpb"
	edge_weight_overrides: { source_id: "a" destination_id: "b" weight: 0.9 }
	edge_weight_overrides: { source_id: "a" destination_id: "c" weight: 0.1 }
	edge_weight_overrides: { source_id: "b" destination_id: "b" weight: 0.5 }
	edge_weight_overrides: { source_id: "b" destination_id: "c" weight: 0.5 }
}`)

	derived, err := deriveSubtypeGraph(sub, base)
	if err != nil {
		t.Fatalf("deriveSubtypeGraph() error = %v, want nil", err)
	}

	// Structure fidelity: same node ids in the same order, and each node's
	// template and completion_time are preserved exactly. Compare those fields
	// via proto equality (ignoring edges, which the overrides change).
	if got, want := len(derived.GetNodes()), len(base.GetNodes()); got != want {
		t.Fatalf("derived graph has %d nodes, want %d", got, want)
	}
	for i, dn := range derived.GetNodes() {
		bn := baseBefore.GetNodes()[i]
		if dn.GetId() != bn.GetId() {
			t.Errorf("node %d id = %q, want %q", i, dn.GetId(), bn.GetId())
		}
		if diff := cmp.Diff(bn.GetTemplate(), dn.GetTemplate(), protocmp.Transform()); diff != "" {
			t.Errorf("node %q template changed (-base +derived):\n%s", bn.GetId(), diff)
		}
		if diff := cmp.Diff(bn.GetCompletionTime(), dn.GetCompletionTime(), protocmp.Transform()); diff != "" {
			t.Errorf("node %q completion_time changed (-base +derived):\n%s", bn.GetId(), diff)
		}
	}
	// Start spec is preserved.
	if got, want := derived.GetStart().GetNodeId(), baseBefore.GetStart().GetNodeId(); got != want {
		t.Errorf("derived start node = %q, want %q", got, want)
	}

	// Weight fidelity: overridden edges carry the new weight; non-overridden
	// edges are unchanged. Because each node's sum stays 1.0, renorm is a no-op,
	// so the exact weights are directly assertable.
	weightOf := func(cfg *genpb.MarkovSamplingConfig, src, dst string) (float64, bool) {
		for _, n := range cfg.GetNodes() {
			if n.GetId() != src {
				continue
			}
			for _, e := range n.GetEdges() {
				if e.GetDestinationId() == dst {
					return e.GetWeight(), true
				}
			}
		}
		return 0, false
	}

	wantWeights := []struct {
		src, dst string
		weight   float64
	}{
		{"a", "b", 0.9}, // overridden
		{"a", "c", 0.1}, // overridden
		{"b", "b", 0.5}, // overridden
		{"b", "c", 0.5}, // overridden
	}
	for _, w := range wantWeights {
		got, ok := weightOf(derived, w.src, w.dst)
		if !ok {
			t.Errorf("derived graph missing edge %s -> %s", w.src, w.dst)
			continue
		}
		if math.Abs(got-w.weight) > FloatTolerance {
			t.Errorf("edge %s -> %s weight = %v, want %v", w.src, w.dst, got, w.weight)
		}
	}

	// A subtype that overrides only one edge, leaving the node's other edge
	// untouched (and the sum != 1.0 so renorm scales both) is covered elsewhere;
	// here we additionally assert a non-overridden structure is byte-preserved by
	// confirming the base graph itself was not mutated by derivation.
	if diff := cmp.Diff(baseBefore, base, protocmp.Transform()); diff != "" {
		t.Errorf("deriveSubtypeGraph mutated the base graph (-before +after):\n%s", diff)
	}
}

// TestDerivedGraphIsValid asserts that graphs derived from valid subtype configs
// satisfy every Requirement-4 graph invariant, i.e. validateMarkovSampling
// returns nil for the derived MarkovSamplingConfig.
//
// Property 21, Validates R12.8, R16.7
func TestDerivedGraphIsValid(t *testing.T) {
	tests := []struct {
		name    string
		subtype string
	}{
		{
			name: "sum_preserving_overrides",
			subtype: `
markov_subtype {
	base_graph_path: "base.textpb"
	edge_weight_overrides: { source_id: "a" destination_id: "b" weight: 0.9 }
	edge_weight_overrides: { source_id: "a" destination_id: "c" weight: 0.1 }
}`,
		},
		{
			name: "renormalized_overrides",
			// Override a single edge so the node's sum leaves 1.0; renormalization
			// must restore the unit sum, keeping the derived graph valid.
			subtype: `
markov_subtype {
	base_graph_path: "base.textpb"
	edge_weight_overrides: { source_id: "a" destination_id: "b" weight: 3.0 }
}`,
		},
		{
			name: "no_overrides",
			subtype: `
markov_subtype {
	base_graph_path: "base.textpb"
}`,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			base := parseSubtypeBase(t)
			sub := parseSubtype(t, test.subtype)

			derived, err := deriveSubtypeGraph(sub, base)
			if err != nil {
				t.Fatalf("deriveSubtypeGraph() error = %v, want nil", err)
			}
			if err := validateMarkovSampling(derived); err != nil {
				t.Errorf("validateMarkovSampling(derived) = %v, want nil", err)
			}
		})
	}
}

// A base graph used by the per-individual derivation tests. Node "a" is a
// non-terminal branching node with three outgoing edges in a fixed declaration
// order (b, c, d) whose class weights sum to 1.0; "b" is a self-looping
// non-terminal node; "c" and "d" are terminal. The fixed edge order lets the
// tests reason about the exact rng draw sequence deriveIndividualGraph consumes.
const individualBaseGraph = `
markov_sampling {
	nodes: {
		id: "a"
		template: { normalized_query_template: "SELECT 1" }
		completion_time: { fixed: { nanos: 2000000 } }
		edges: { destination_id: "b" weight: 0.2 }
		edges: { destination_id: "c" weight: 0.3 }
		edges: { destination_id: "d" weight: 0.5 }
	}
	nodes: {
		id: "b"
		template: { normalized_query_template: "SELECT 2" }
		completion_time: { fixed: { nanos: 1000000 } }
		edges: { destination_id: "b" weight: 0.4 }
		edges: { destination_id: "c" weight: 0.6 }
	}
	nodes: {
		id: "c"
		template: { normalized_query_template: "SELECT 3" }
		completion_time: { fixed: { nanos: 1000000 } }
	}
	nodes: {
		id: "d"
		template: { normalized_query_template: "SELECT 4" }
		completion_time: { fixed: { nanos: 1000000 } }
	}
	start: { node_id: "a" }
}`

// parseIndividualBase parses individualBaseGraph into its MarkovSamplingConfig
// base graph, the resolved base that deriveIndividualGraph consumes.
func parseIndividualBase(t *testing.T) *genpb.MarkovSamplingConfig {
	t.Helper()
	cfgs := parseProto(t, individualBaseGraph, &genpb.QueryTemplateConfigs{})
	base := cfgs.GetMarkovSampling()
	if base == nil {
		t.Fatal("individualBaseGraph did not parse into a markov_sampling config")
	}
	return base
}

// parseIndividual parses a markov_individual QueryTemplateConfigs textproto into
// its MarkovIndividualConfig via GetMarkovIndividual().
func parseIndividual(t *testing.T, text string) *genpb.MarkovIndividualConfig {
	t.Helper()
	cfgs := parseProto(t, text, &genpb.QueryTemplateConfigs{})
	ind := cfgs.GetMarkovIndividual()
	if ind == nil {
		t.Fatal("individual textproto did not parse into a markov_individual config")
	}
	return ind
}

// weightOf returns the weight of the (src -> dst) edge in cfg, or (0, false)
// when no such edge exists.
func weightOf(cfg *genpb.MarkovSamplingConfig, src, dst string) (float64, bool) {
	for _, n := range cfg.GetNodes() {
		if n.GetId() != src {
			continue
		}
		for _, e := range n.GetEdges() {
			if e.GetDestinationId() == dst {
				return e.GetWeight(), true
			}
		}
	}
	return 0, false
}

// TestDeriveIndividual_NoMechanisms verifies that when an individual configures
// none of individual_edge_overrides, edge_variances, or weight_jitter (even with
// subtype_overrides applied to form the class graph), the derived edge weights
// equal the class weights: no random value is drawn, and because the class sums
// are already 1.0 the single renormalization is a no-op, so the weights are
// exactly the post-subtype-override class weights. It also asserts the base graph
// is not mutated.
//
// Property 22, Validates R14.1, R14.6
func TestDeriveIndividual_NoMechanisms(t *testing.T) {
	base := parseIndividualBase(t)
	baseBefore := proto.Clone(base).(*genpb.MarkovSamplingConfig)

	// An individual that applies subtype_overrides to form the class graph but
	// configures no per-edge mechanisms. The overrides keep node "a"'s sum at 1.0
	// so the class weights are directly observable (renorm is a no-op).
	ind := parseIndividual(t, `
markov_individual {
	base_graph_path: "base.textpb"
	subtype_overrides: { source_id: "a" destination_id: "b" weight: 0.1 }
	subtype_overrides: { source_id: "a" destination_id: "c" weight: 0.4 }
	subtype_overrides: { source_id: "a" destination_id: "d" weight: 0.5 }
}`)

	// A seeded rng that the derivation must NOT draw from (no mechanisms). We
	// capture its state by snapshotting the first draw an untouched rng would
	// produce, then confirm an identically-seeded rng is still at that state
	// after derivation (i.e. derivation consumed zero draws).
	const seed = 12345
	rng := rand.New(rand.NewSource(seed))

	derived, err := deriveIndividualGraph(ind, base, rng)
	if err != nil {
		t.Fatalf("deriveIndividualGraph() error = %v, want nil", err)
	}

	// Derived weights equal the (post-subtype-override) class weights exactly.
	wantWeights := map[string]float64{"b": 0.1, "c": 0.4, "d": 0.5}
	for dst, want := range wantWeights {
		got, ok := weightOf(derived, "a", dst)
		if !ok {
			t.Errorf("derived graph missing edge a -> %s", dst)
			continue
		}
		if math.Abs(got-want) > FloatTolerance {
			t.Errorf("edge a -> %s weight = %v, want %v (class weight)", dst, got, want)
		}
	}
	// Node "b" had no overrides and its class weights (0.4, 0.6) already sum to
	// 1.0, so they pass through unchanged too.
	for dst, want := range map[string]float64{"b": 0.4, "c": 0.6} {
		got, _ := weightOf(derived, "b", dst)
		if math.Abs(got-want) > FloatTolerance {
			t.Errorf("edge b -> %s weight = %v, want %v (class weight)", dst, got, want)
		}
	}

	// No draws were consumed: an identically-seeded rng advanced by zero draws
	// still yields the same next value as rng does now.
	fresh := rand.New(rand.NewSource(seed))
	if got, want := rng.Float64(), fresh.Float64(); got != want {
		t.Errorf("derivation consumed rng draws: next draw = %v, want %v (zero draws expected)", got, want)
	}

	// Base graph is untouched.
	if diff := cmp.Diff(baseBefore, base, protocmp.Transform()); diff != "" {
		t.Errorf("deriveIndividualGraph mutated the base graph (-before +after):\n%s", diff)
	}
}

// TestDeriveIndividual_Precedence verifies the strict per-edge precedence ladder
// on a single branching node that mixes all three mechanisms: a pinned edge (via
// individual_edge_override), a variance edge (positive down/up), and a jitter-only
// edge (governed by the global jitter bound because it has neither an override nor
// a variance entry).
//
// It asserts: (a) the pinned edge's pre-renormalization weight equals the override
// exactly and draws nothing; (b) the variance edge is variance-governed only and
// NOT additionally jittered; (c) the remaining edge is jitter-governed; (d) the
// total draw count equals the number of non-pinned variance/jitter-governed edges
// (one Float64 per edge, mutual exclusion). Draw counting is deterministic: an
// identically-seeded rng is advanced by the expected number of Float64 draws in
// the implementation's node/edge declaration order, and the by-hand weights from
// those exact draws are compared against the derived (pre-renorm) weights.
//
// Property 23, Validates R14.2, R15.1, R15.2, R15.3, R15.4
func TestDeriveIndividual_Precedence(t *testing.T) {
	base := parseIndividualBase(t)

	// Node "a" edges in declaration order: b (pinned), c (variance), d (jitter).
	//   - b: individual_edge_override -> 0.25, draws nothing.
	//   - c: edge_variance down=0.1 up=0.2, variance-governed (one draw).
	//   - d: no override, no variance -> jitter-governed under bound=0.05 (one draw).
	// So node "a" consumes exactly 2 draws, in edge order c then d.
	// Node "b" has no mechanisms but jitter bound > 0 governs both its edges:
	//   b->b and b->c each consume one jitter draw (2 more draws, after node a).
	ind := parseIndividual(t, `
markov_individual {
	base_graph_path: "base.textpb"
	individual_edge_overrides: { source_id: "a" destination_id: "b" weight: 0.25 }
	edge_variances: { source_id: "a" destination_id: "c" down: 0.1 up: 0.2 }
	weight_jitter: { bound: 0.05 }
}`)

	const seed = 987654321

	// Class weights (post-subtype-override; none here, so == base weights).
	const (
		cwAB = 0.2 // a -> b class weight (pinned, unused)
		cwAC = 0.3 // a -> c class weight (variance)
		cwAD = 0.5 // a -> d class weight (jitter)
		cwBB = 0.4 // b -> b class weight (jitter)
		cwBC = 0.6 // b -> c class weight (jitter)
	)
	const (
		varDown = 0.1
		varUp   = 0.2
		jitterB = 0.05
	)

	// Replay the exact draw sequence the implementation uses: nodes in
	// declaration order (a, b, c, d); within a node, edges in declaration order.
	//   Node a: b pinned (no draw), c variance (draw #1), d jitter (draw #2).
	//   Node b: b jitter (draw #3), c jitter (draw #4).
	//   Nodes c, d: terminal (no edges, no draws).
	replay := rand.New(rand.NewSource(seed))
	dAC := replay.Float64() // variance draw for a -> c
	dAD := replay.Float64() // jitter draw for a -> d
	dBB := replay.Float64() // jitter draw for b -> b
	dBC := replay.Float64() // jitter draw for b -> c

	// By-hand pre-renorm weights from those exact draws.
	preAC := cwAC - varDown + dAC*(varDown+varUp)
	preAD := cwAD + (dAD*2-1)*jitterB
	preBB := cwBB + (dBB*2-1)*jitterB
	preBC := cwBC + (dBC*2-1)*jitterB

	// Pre-renorm node sums (used to recover expected post-renorm weights).
	sumA := 0.25 + preAC + preAD // a: pinned + variance + jitter
	sumB := preBB + preBC

	// Run the derivation with a fresh, identically-seeded rng.
	rng := rand.New(rand.NewSource(seed))
	derived, err := deriveIndividualGraph(ind, base, rng)
	if err != nil {
		t.Fatalf("deriveIndividualGraph() error = %v, want nil", err)
	}

	// (d) Draw count: the implementation must have consumed exactly 4 Float64
	// draws (c, d, b->b, b->c). After derivation, rng and replay are in lockstep,
	// so their next draws must match.
	if got, want := rng.Float64(), replay.Float64(); got != want {
		t.Errorf("draw count mismatch: rng next = %v, replay next = %v; derivation consumed != 4 draws", got, want)
	}

	// Assert each derived (post-renorm) weight equals the by-hand pre-renorm value
	// scaled by 1/sum for its node. This simultaneously confirms:
	//   (a) the pinned edge used the override and no draw;
	//   (b) the variance edge is variance-governed only (matches preAC, not a
	//       jittered value);
	//   (c) the remaining edge is jitter-governed (matches preAD).
	checks := []struct {
		src, dst string
		pre, sum float64
	}{
		{"a", "b", 0.25, sumA},
		{"a", "c", preAC, sumA},
		{"a", "d", preAD, sumA},
		{"b", "b", preBB, sumB},
		{"b", "c", preBC, sumB},
	}
	for _, c := range checks {
		got, ok := weightOf(derived, c.src, c.dst)
		if !ok {
			t.Errorf("derived graph missing edge %s -> %s", c.src, c.dst)
			continue
		}
		want := c.pre / c.sum
		if math.Abs(got-want) > FloatTolerance {
			t.Errorf("edge %s -> %s weight = %v, want %v (pre-renorm %v / sum %v)",
				c.src, c.dst, got, want, c.pre, c.sum)
		}
	}
}

// TestDeriveIndividual_Bounds verifies that over many fixed seeds each
// variance-governed edge's pre-clamp effective weight lies in [w-down, w+up] and
// each jitter-governed edge's lies in [w-b, w+b], and that no edge reflects both
// perturbations. Bounds are checked on reconstructed pre-renorm values (replayed
// from an identically-seeded rng) so renormalization scaling does not obscure the
// interval membership; the closed interval allows the >= 0 clamp.
//
// Property 24, Validates R14.3, R14.4
func TestDeriveIndividual_Bounds(t *testing.T) {
	base := parseIndividualBase(t)

	// a -> c is variance-governed; a -> d, b -> b, b -> c are jitter-governed;
	// a -> b is pinned (exempt from both). Bounds chosen large enough to be
	// interesting but kept non-negative here (clamp is exercised separately).
	ind := parseIndividual(t, `
markov_individual {
	base_graph_path: "base.textpb"
	individual_edge_overrides: { source_id: "a" destination_id: "b" weight: 0.25 }
	edge_variances: { source_id: "a" destination_id: "c" down: 0.1 up: 0.2 }
	weight_jitter: { bound: 0.05 }
}`)

	const (
		cwAC = 0.3
		cwAD = 0.5
		cwBB = 0.4
		cwBC = 0.6

		varDown = 0.1
		varUp   = 0.2
		jitterB = 0.05
	)

	for seed := int64(1); seed <= 200; seed++ {
		// Replay the implementation's draw order: a->c variance, a->d jitter,
		// b->b jitter, b->c jitter.
		replay := rand.New(rand.NewSource(seed))
		dAC := replay.Float64()
		dAD := replay.Float64()
		dBB := replay.Float64()
		dBC := replay.Float64()

		preAC := cwAC - varDown + dAC*(varDown+varUp)
		preAD := cwAD + (dAD*2-1)*jitterB
		preBB := cwBB + (dBB*2-1)*jitterB
		preBC := cwBC + (dBC*2-1)*jitterB

		// Variance-governed edge within [w-down, w+up].
		if preAC < cwAC-varDown-FloatTolerance || preAC > cwAC+varUp+FloatTolerance {
			t.Fatalf("seed %d: variance edge a->c pre-clamp weight %v outside [%v, %v]",
				seed, preAC, cwAC-varDown, cwAC+varUp)
		}
		// Jitter-governed edges within [w-b, w+b].
		for _, jc := range []struct {
			name   string
			pre, w float64
		}{
			{"a->d", preAD, cwAD},
			{"b->b", preBB, cwBB},
			{"b->c", preBC, cwBC},
		} {
			if jc.pre < jc.w-jitterB-FloatTolerance || jc.pre > jc.w+jitterB+FloatTolerance {
				t.Fatalf("seed %d: jitter edge %s pre-clamp weight %v outside [%v, %v]",
					seed, jc.name, jc.pre, jc.w-jitterB, jc.w+jitterB)
			}
		}

		// The derived graph (post-renorm) must be consistent with those exact
		// pre-renorm values: each edge equals pre/sum for its node. This confirms
		// no edge reflects both a variance and a jitter perturbation (the variance
		// edge matches preAC built from varDown/varUp only; the jitter edges match
		// preAD/preBB/preBC built from jitterB only).
		rng := rand.New(rand.NewSource(seed))
		derived, err := deriveIndividualGraph(ind, base, rng)
		if err != nil {
			t.Fatalf("seed %d: deriveIndividualGraph() error = %v, want nil", seed, err)
		}
		sumA := 0.25 + preAC + preAD
		sumB := preBB + preBC
		exp := []struct {
			src, dst string
			pre, sum float64
		}{
			{"a", "c", preAC, sumA},
			{"a", "d", preAD, sumA},
			{"b", "b", preBB, sumB},
			{"b", "c", preBC, sumB},
		}
		for _, e := range exp {
			got, _ := weightOf(derived, e.src, e.dst)
			want := e.pre / e.sum
			if math.Abs(got-want) > FloatTolerance {
				t.Fatalf("seed %d: edge %s->%s = %v, want %v (pre %v / sum %v)",
					seed, e.src, e.dst, got, want, e.pre, e.sum)
			}
		}
	}
}

// TestDeriveIndividual_Clamp verifies that deltas and jitter bounds large enough
// to drive a sampled weight below zero are clamped to zero before renormalization,
// so every pre- and post-renormalization weight is >= 0. Run over many seeds.
//
// Property 25, Validates R16.9
func TestDeriveIndividual_Clamp(t *testing.T) {
	base := parseIndividualBase(t)

	// A variance with a down delta far larger than the class weight on a->c, and
	// a large global jitter bound so a->d / b->* can also go negative pre-clamp.
	// a->b is pinned to a positive value so node "a"'s clamped sum stays positive
	// and renormalization succeeds.
	ind := parseIndividual(t, `
markov_individual {
	base_graph_path: "base.textpb"
	individual_edge_overrides: { source_id: "a" destination_id: "b" weight: 0.5 }
	edge_variances: { source_id: "a" destination_id: "c" down: 5.0 up: 0.1 }
	weight_jitter: { bound: 10.0 }
}`)

	for seed := int64(1); seed <= 200; seed++ {
		rng := rand.New(rand.NewSource(seed))
		derived, err := deriveIndividualGraph(ind, base, rng)
		// With such large bounds a node's clamped sum can collapse to the pinned
		// weight only (e.g. b->b and b->c both clamped to 0 makes node "b"
		// non-renormalizable). Node "b" has no pinned edge, so if both its jitter
		// draws clamp to 0 the derivation legitimately returns the R16.8 error.
		if err != nil {
			if !strings.Contains(err.Error(), "must be positive to renormalize") {
				t.Fatalf("seed %d: unexpected error = %v", seed, err)
			}
			continue
		}
		// When derivation succeeds, every post-renorm weight must be >= 0.
		for _, n := range derived.GetNodes() {
			for _, e := range n.GetEdges() {
				if e.GetWeight() < 0 {
					t.Fatalf("seed %d: edge %s->%s post-renorm weight = %v, want >= 0",
						seed, n.GetId(), e.GetDestinationId(), e.GetWeight())
				}
			}
		}
	}
}

// TestDerivedGraphIsValid_Individual asserts that graphs derived from valid
// individual configs satisfy every Requirement-4 graph invariant, i.e.
// validateMarkovSampling returns nil for the derived MarkovSamplingConfig. It is
// a sibling of TestDerivedGraphIsValid (which covers the subtype cases) to avoid
// editing that function's table.
//
// Property 21, Validates R12.8, R16.7
func TestDerivedGraphIsValid_Individual(t *testing.T) {
	tests := []struct {
		name       string
		individual string
	}{
		{
			name: "no_mechanisms",
			individual: `
markov_individual {
	base_graph_path: "base.textpb"
}`,
		},
		{
			name: "subtype_overrides_only",
			individual: `
markov_individual {
	base_graph_path: "base.textpb"
	subtype_overrides: { source_id: "a" destination_id: "b" weight: 0.1 }
	subtype_overrides: { source_id: "a" destination_id: "c" weight: 0.4 }
	subtype_overrides: { source_id: "a" destination_id: "d" weight: 0.5 }
}`,
		},
		{
			name: "pinned_variance_and_jitter",
			individual: `
markov_individual {
	base_graph_path: "base.textpb"
	individual_edge_overrides: { source_id: "a" destination_id: "b" weight: 0.25 }
	edge_variances: { source_id: "a" destination_id: "c" down: 0.1 up: 0.2 }
	weight_jitter: { bound: 0.05 }
}`,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			base := parseIndividualBase(t)
			ind := parseIndividual(t, test.individual)
			rng := rand.New(rand.NewSource(42))

			derived, err := deriveIndividualGraph(ind, base, rng)
			if err != nil {
				t.Fatalf("deriveIndividualGraph() error = %v, want nil", err)
			}
			if err := validateMarkovSampling(derived); err != nil {
				t.Errorf("validateMarkovSampling(derived) = %v, want nil", err)
			}
		})
	}
}

// TestDeriveIndividual_NonPositiveSumError verifies R16.8: an individual whose
// individual_edge_override zeroes a node's only outgoing edge produces a
// non-positive clamped sum, so deriveIndividualGraph (via renormalizeOutgoing)
// returns the node-naming "must be positive to renormalize" error rather than a
// silently invalid graph. This exercises the derivation-time enforcement that
// validateIndividual intentionally does not duplicate.
//
// Validates R16.8
func TestDeriveIndividual_NonPositiveSumError(t *testing.T) {
	// A base where node "a" has a single outgoing edge to terminal "b".
	const singleEdgeBase = `
markov_sampling {
	nodes: {
		id: "a"
		template: { normalized_query_template: "SELECT 1" }
		completion_time: { fixed: { nanos: 1000000 } }
		edges: { destination_id: "b" weight: 1.0 }
	}
	nodes: {
		id: "b"
		template: { normalized_query_template: "SELECT 2" }
		completion_time: { fixed: { nanos: 1000000 } }
	}
	start: { node_id: "a" }
}`
	base := parseProto(t, singleEdgeBase, &genpb.QueryTemplateConfigs{}).GetMarkovSampling()
	if base == nil {
		t.Fatal("singleEdgeBase did not parse into a markov_sampling config")
	}

	// Pin node "a"'s only edge to 0, collapsing its outgoing sum to 0.
	ind := parseIndividual(t, `
markov_individual {
	base_graph_path: "base.textpb"
	individual_edge_overrides: { source_id: "a" destination_id: "b" weight: 0.0 }
}`)

	rng := rand.New(rand.NewSource(1))
	_, err := deriveIndividualGraph(ind, base, rng)
	if err == nil {
		t.Fatal("deriveIndividualGraph() error = nil, want non-positive-sum error")
	}
	if !strings.Contains(err.Error(), "a") || !strings.Contains(err.Error(), "must be positive to renormalize") {
		t.Errorf("deriveIndividualGraph() error = %q, want it to name node %q and %q",
			err.Error(), "a", "must be positive to renormalize")
	}
}

// TestEdgeVarianceMean is a statistical test that the pre-clamp sampled weight of
// a variance-governed edge is, on average, centered at w + (up-down)/2 (which
// equals w when down == up).
//
// deriveIndividualGraph computes a variance edge's pre-clamp weight as
// w - down + Float64()*(down+up), so a single Float64 draw is a uniform sample
// over [w-down, w+up], whose population mean is w + (up-down)/2. The test iterates
// a fixed set of distinct seeds (1..N) so it stays fully deterministic, and
// averages the replayed draws exactly as the implementation would compute them.
// A separate derivation per a couple of seeds ties the statistical replay to the
// real implementation by confirming the derived (post-renorm) weight is consistent
// with the replayed pre-renorm value divided by the node's pre-renorm sum.
//
// The band is sized for N samples of a uniform of width (down+up): population std
// is (down+up)/sqrt(12), the std of the mean of N is that/sqrt(N), and the band is
// a generous 6-sigma so a fixed seed set never flakes.
//
// Property 26, Validates R14.3
func TestEdgeVarianceMean(t *testing.T) {
	base := parseIndividualBase(t)

	// A single variance-governed edge a->c whose pre-clamp sampled value is
	// recoverable: a->b is pinned (draws nothing, consumes no rng), so a->c's
	// variance draw is the FIRST Float64 drawn in the whole derivation. a->d and
	// node "b"'s edges keep the class weight (no mechanism) so they draw nothing.
	// With no jitter configured the ONLY draw per derivation is the a->c variance
	// draw, which makes the replay trivially exact.
	makeInd := func(down, up float64) *genpb.MarkovIndividualConfig {
		return parseIndividual(t, `
markov_individual {
	base_graph_path: "base.textpb"
	individual_edge_overrides: { source_id: "a" destination_id: "b" weight: 0.2 }
	edge_variances: { source_id: "a" destination_id: "c" down: `+formatFloat(down)+` up: `+formatFloat(up)+` }
}`)
	}

	const (
		cwAC = 0.3   // a -> c class weight
		n    = 20000 // distinct fixed seeds 1..n
	)

	cases := []struct {
		name     string
		down, up float64
	}{
		{name: "asymmetric_down_lt_up", down: 0.1, up: 0.3}, // expected mean = w + 0.1
		{name: "symmetric", down: 0.2, up: 0.2},             // expected mean = w
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			wantMean := cwAC + (c.up-c.down)/2.0

			// Empirical mean of the replayed pre-clamp sampled weight. For each
			// seed the derivation draws exactly one Float64 for the a->c variance
			// edge (first and only draw), so replaying one Float64 per seed matches
			// the implementation's pre-clamp value exactly.
			var sum float64
			for seed := int64(1); seed <= n; seed++ {
				draw := rand.New(rand.NewSource(seed)).Float64()
				sum += cwAC - c.down + draw*(c.down+c.up)
			}
			mean := sum / float64(n)

			// 6-sigma band for the mean of n uniform samples of width (down+up).
			width := c.down + c.up
			sigmaMean := width / math.Sqrt(12.0) / math.Sqrt(float64(n))
			band := 6 * sigmaMean
			if math.Abs(mean-wantMean) > band {
				t.Errorf("%s: empirical mean sampled weight = %v, want within %v of %v (diff %v)",
					c.name, mean, band, wantMean, math.Abs(mean-wantMean))
			}

			// Tie the replay to the real implementation: for a couple of seeds,
			// derive and confirm the derived (post-renorm) a->c weight equals the
			// replayed pre-renorm value divided by node "a"'s pre-renorm sum.
			ind := makeInd(c.down, c.up)
			for _, seed := range []int64{1, 2, n} {
				draw := rand.New(rand.NewSource(seed)).Float64()
				preAC := cwAC - c.down + draw*(c.down+c.up)
				if preAC < 0 {
					preAC = 0
				}
				// Node "a" pre-renorm sum: pinned a->b (0.2) + a->c pre + a->d
				// class weight (0.5, no mechanism).
				sumA := 0.2 + preAC + 0.5

				rng := rand.New(rand.NewSource(seed))
				derived, err := deriveIndividualGraph(ind, base, rng)
				if err != nil {
					t.Fatalf("%s seed %d: deriveIndividualGraph() error = %v", c.name, seed, err)
				}
				got, ok := weightOf(derived, "a", "c")
				if !ok {
					t.Fatalf("%s seed %d: derived graph missing edge a->c", c.name, seed)
				}
				want := preAC / sumA
				if math.Abs(got-want) > FloatTolerance {
					t.Errorf("%s seed %d: derived a->c weight = %v, want %v (pre %v / sum %v)",
						c.name, seed, got, want, preAC, sumA)
				}
			}
		})
	}
}

// formatFloat renders a float64 for inline textproto literals without exponent
// notation so small deltas like 0.1 embed cleanly into the markov_individual text.
func formatFloat(f float64) string {
	return strconv.FormatFloat(f, 'f', -1, 64)
}

// TestDerive_Determinism verifies that per-individual derivation plus the walk are
// jointly deterministic under a fixed seed: deriving the same individual config
// twice with identically-seeded rngs yields proto-equal derived graphs, and
// running the full derivation + walk through the single shared rng twice produces
// byte-identical output.
//
// The byte comparison drives the whole single-rng flow: one rng calls
// deriveIndividualGraph (consuming the variance/jitter draws) and then feeds the
// SAME rng into generateLogsMarkovFromConfig (consuming the walk draws), mirroring
// how GenerateLogs wires the markov_individual scenario. The individual exercises
// both variance and jitter so draws actually occur during derivation.
//
// Property 27, Validates R14.5, R15.6
func TestDerive_Determinism(t *testing.T) {
	// Exercises variance (a->c) and jitter (a->d, b->b, b->c) so derivation draws
	// from the rng, plus a pinned edge (a->b) that draws nothing.
	const individualText = `
markov_individual {
	base_graph_path: "base.textpb"
	individual_edge_overrides: { source_id: "a" destination_id: "b" weight: 0.25 }
	edge_variances: { source_id: "a" destination_id: "c" down: 0.1 up: 0.2 }
	weight_jitter: { bound: 0.05 }
}`

	const (
		seed        = int64(20240531)
		numLogs     = 200
		startTimeMs = int64(1_700_000_000_000)
	)

	// Proto-equality of the derived graphs from two identically-seeded rngs.
	derived1, err := deriveIndividualGraph(
		parseIndividual(t, individualText), parseIndividualBase(t), rand.New(rand.NewSource(seed)))
	if err != nil {
		t.Fatalf("first deriveIndividualGraph() error = %v", err)
	}
	derived2, err := deriveIndividualGraph(
		parseIndividual(t, individualText), parseIndividualBase(t), rand.New(rand.NewSource(seed)))
	if err != nil {
		t.Fatalf("second deriveIndividualGraph() error = %v", err)
	}
	if !proto.Equal(derived1, derived2) {
		t.Errorf("deriveIndividualGraph() not deterministic: two same-seed derivations differ")
	}

	// End-to-end single-rng flow: derive then walk with ONE rng, twice, and assert
	// byte-identical output. Each run uses a fresh rng seeded identically so the
	// derivation draws and the subsequent walk draws come from the same stream.
	runOnce := func() []byte {
		rng := rand.New(rand.NewSource(seed))
		derived, err := deriveIndividualGraph(
			parseIndividual(t, individualText), parseIndividualBase(t), rng)
		if err != nil {
			t.Fatalf("deriveIndividualGraph() error = %v", err)
		}
		var buf bytes.Buffer
		if err := generateLogsMarkovFromConfig(derived, numLogs, startTimeMs, rng, &buf); err != nil {
			t.Fatalf("generateLogsMarkovFromConfig() error = %v", err)
		}
		return buf.Bytes()
	}

	out1 := runOnce()
	out2 := runOnce()
	if !bytes.Equal(out1, out2) {
		t.Errorf("derivation + walk not deterministic: two same-seed runs produced different bytes (%d vs %d bytes)",
			len(out1), len(out2))
	}
	if len(out1) == 0 {
		t.Error("derivation + walk produced no output; expected length-delimited entries")
	}
}
