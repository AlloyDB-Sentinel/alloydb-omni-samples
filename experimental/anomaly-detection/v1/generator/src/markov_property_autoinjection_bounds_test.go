package main

// Property-based test for the markov-profile-library feature.
//
// Feature: markov-profile-library, Property 13: Auto-injection variants embed a valid base graph and well-formed bounds
//
// Validates: Requirements 4.6
//
// Property 13 quantifies over the six authored auto-injection variants, one per
// base profile, each at ../../client/markov_profiles/<name>/auto_injection.textproto.
// Each such variant is a plain markov_sampling config whose graph embeds a verbatim
// copy of its base profile's graph plus a MarkovAutoInjection block. For every
// variant the property asserts two things:
//
//  1. the embedded markov_sampling graph is structurally well-formed, exactly as
//     Property 1 (markov_property_structure_test.go) requires of a base graph:
//     node ids non-empty and unique; every edge destination_id and the start spec
//     resolve to declared nodes; edge weights finite and non-negative; every
//     non-terminal node's outgoing weights sum to a strictly positive total; and
//     every declared node reachable from the start.
//  2. the auto_injection block carries well-formed bounds: anomalous_fraction in
//     the closed interval [0.0, 1.0]; min_actions >= 1; max_actions >= min_actions.
//
// The property ranges over a finite, enumerated set of six variants, so the
// invariants are checked exhaustively on every iteration. To honor the package's
// >= 100-iteration property-test convention, the full six-variant enumeration runs
// at least 100 times, and each iteration additionally drives a randomized
// reachability probe over a random (variant, target node) pair confirmed against a
// randomized-frontier BFS, exercising reachability across many traversal orderings.
//
// Helper names here are prefixed `prop13` to stay unique within the package and
// avoid collisions with the prop1* helpers in markov_property_structure_test.go.
// The shared `reachable` BFS helper (defined in markov_batch_etl_profile_test.go)
// is reused and NOT redefined.
//
// No t.Parallel(): loadProfile calls loadConfigs, which sets the package-global
// resolvedBaseGraph, so harness callers must run sequentially.

import (
	"math"
	"math/rand"
	"path/filepath"
	"testing"

	genpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v1/generator/proto"
)

// prop13AutoInjectionPaths enumerates the six auto-injection variants, one per
// base profile, under ../../client/markov_profiles/<name>/auto_injection.textproto.
// Go tests run with the working directory set to the package dir
// (v1/generator/src), so client configs live at ../../client/, matching the path
// convention in markov_analytics_profile_test.go.
var prop13AutoInjectionPaths = []string{
	filepath.Join("..", "..", "client", "markov_profiles", "analytics", "auto_injection.textproto"),
	filepath.Join("..", "..", "client", "markov_profiles", "batch_etl", "auto_injection.textproto"),
	filepath.Join("..", "..", "client", "markov_profiles", "read_replica", "auto_injection.textproto"),
	filepath.Join("..", "..", "client", "markov_profiles", "bursty_api", "auto_injection.textproto"),
	filepath.Join("..", "..", "client", "markov_profiles", "noisy_client", "auto_injection.textproto"),
	filepath.Join("..", "..", "client", "markov_profiles", "oltp_app", "auto_injection.textproto"),
}

// prop13StartNodeIDs returns every node id a graph's start spec resolves to: the
// single node_id for a point start, or each distribution entry's node_id for a
// weighted start. A well-formed graph sets exactly one of these.
func prop13StartNodeIDs(graph *genpb.MarkovSamplingConfig) []string {
	start := graph.GetStart()
	if start == nil {
		return nil
	}
	if dist := start.GetDistribution(); dist != nil {
		var ids []string
		for _, s := range dist.GetStarts() {
			ids = append(ids, s.GetNodeId())
		}
		return ids
	}
	// Point start. GetNodeId returns "" when node_id is not the set oneof arm,
	// which the caller treats as a missing/invalid start.
	return []string{start.GetNodeId()}
}

// prop13CheckEmbeddedGraphWellFormed asserts the same structural invariants as
// Property 1 against the embedded markov_sampling graph of an auto-injection
// variant. It is deterministic and independent of iteration count.
func prop13CheckEmbeddedGraphWellFormed(t *testing.T, profile *profileResult) {
	t.Helper()

	graph := profile.describingGraph()
	if graph == nil {
		t.Fatalf("%s: auto-injection variant has no embedded markov_sampling graph", profile.Path)
	}

	// --- node ids non-empty and unique ---
	seen := make(map[string]bool, len(profile.NodeIDs))
	for _, id := range profile.NodeIDs {
		if id == "" {
			t.Fatalf("%s: embedded graph declares a node with an empty id", profile.Path)
		}
		if seen[id] {
			t.Fatalf("%s: duplicate node id %q", profile.Path, id)
		}
		seen[id] = true
	}
	if len(profile.NodeIDs) == 0 {
		t.Fatalf("%s: embedded graph declares no nodes", profile.Path)
	}

	// --- every edge destination resolves to a declared node; weights finite,
	// non-negative --- and accumulate per-source outgoing weight totals.
	outgoingSum := make(map[string]float64, len(profile.NodeIDs))
	outgoingCount := make(map[string]int, len(profile.NodeIDs))
	for _, e := range profile.Edges {
		if !seen[e.DestinationID] {
			t.Fatalf("%s: edge %s -> %s points at an undeclared node", profile.Path, e.SourceID, e.DestinationID)
		}
		if math.IsNaN(e.Weight) || math.IsInf(e.Weight, 0) {
			t.Fatalf("%s: edge %s -> %s has non-finite weight %v", profile.Path, e.SourceID, e.DestinationID, e.Weight)
		}
		if e.Weight < 0 {
			t.Fatalf("%s: edge %s -> %s has negative weight %v", profile.Path, e.SourceID, e.DestinationID, e.Weight)
		}
		outgoingSum[e.SourceID] += e.Weight
		outgoingCount[e.SourceID]++
	}

	// --- every NON-terminal node's outgoing weights sum to a positive total ---
	// A node is terminal iff it has zero outgoing edges (e.g. session_done); such
	// nodes are excluded from the positive-sum requirement.
	for _, id := range profile.NodeIDs {
		if outgoingCount[id] == 0 {
			continue // terminal node: zero outgoing edges, excluded by design
		}
		if !(outgoingSum[id] > 0) {
			t.Fatalf("%s: non-terminal node %q has outgoing weight sum %v, want > 0",
				profile.Path, id, outgoingSum[id])
		}
	}

	// --- start spec resolves to declared node(s) ---
	startIDs := prop13StartNodeIDs(graph)
	if len(startIDs) == 0 {
		t.Fatalf("%s: embedded graph sets no start spec", profile.Path)
	}
	for _, sid := range startIDs {
		if sid == "" {
			t.Fatalf("%s: start spec contains an empty node id", profile.Path)
		}
		if !seen[sid] {
			t.Fatalf("%s: start spec references undeclared node %q", profile.Path, sid)
		}
	}

	// --- every declared node is reachable from the start ---
	// With a weighted-distribution start, a node counts as reachable if it is
	// reachable from any start entry. reachable() is the shared BFS helper from
	// markov_batch_etl_profile_test.go.
	for _, target := range profile.NodeIDs {
		if !prop13ReachableFromAnyStart(profile.Edges, startIDs, target) {
			t.Fatalf("%s: node %q is not reachable from any start node %v", profile.Path, target, startIDs)
		}
	}
}

// prop13CheckBounds asserts the auto_injection block's bounds are well-formed:
// anomalous_fraction in [0.0, 1.0], min_actions >= 1, max_actions >= min_actions.
func prop13CheckBounds(t *testing.T, profile *profileResult) {
	t.Helper()

	graph := profile.describingGraph()
	if graph == nil {
		t.Fatalf("%s: auto-injection variant has no embedded markov_sampling graph", profile.Path)
	}

	auto := graph.GetAutoInjection()
	if auto == nil {
		t.Fatalf("%s: variant carries no auto_injection block, want one", profile.Path)
	}

	frac := auto.GetAnomalousFraction()
	if math.IsNaN(frac) || math.IsInf(frac, 0) {
		t.Fatalf("%s: auto_injection anomalous_fraction %v is non-finite", profile.Path, frac)
	}
	if frac < 0.0 || frac > 1.0 {
		t.Fatalf("%s: auto_injection anomalous_fraction = %v, want in [0.0, 1.0]", profile.Path, frac)
	}

	minA := auto.GetMinActions()
	maxA := auto.GetMaxActions()
	if minA < 1 {
		t.Fatalf("%s: auto_injection min_actions = %d, want >= 1", profile.Path, minA)
	}
	if maxA < minA {
		t.Fatalf("%s: auto_injection max_actions = %d, want >= min_actions (%d)", profile.Path, maxA, minA)
	}
}

// prop13ReachableFromAnyStart reports whether target is reachable from at least
// one of the start node ids via the directed edge set, reusing the shared
// reachable() BFS helper.
func prop13ReachableFromAnyStart(edges []profileEdge, startIDs []string, target string) bool {
	for _, start := range startIDs {
		if reachable(edges, start, target) {
			return true
		}
	}
	return false
}

// TestProperty13_AutoInjectionVariantsEmbedValidGraphAndBounds is the Property 13
// driver.
//
// Feature: markov-profile-library, Property 13: Auto-injection variants embed a valid base graph and well-formed bounds
//
// Validates: Requirements 4.6
func TestProperty13_AutoInjectionVariantsEmbedValidGraphAndBounds(t *testing.T) {
	// Load each auto-injection variant once through the harness. An
	// auto_injection variant is a plain markov_sampling config that additionally
	// sets an auto_injection block, so loadProfile reports Kind ==
	// scenarioMarkovSampling with HasInjection == true and populates NodeIDs/Edges
	// from the embedded graph.
	profiles := make([]*profileResult, 0, len(prop13AutoInjectionPaths))
	for _, path := range prop13AutoInjectionPaths {
		profile := loadProfile(t, path)
		if profile.Kind != scenarioMarkovSampling {
			t.Fatalf("%s: variant kind = %q, want %q", path, profile.Kind, scenarioMarkovSampling)
		}
		if !profile.HasInjection {
			t.Fatalf("%s: variant reports HasInjection = false, want true (auto_injection block expected)", path)
		}
		profiles = append(profiles, profile)
	}

	// Property driver: run the full enumeration >= 100 times. The structural and
	// bounds invariants are deterministic over the finite set, so each iteration
	// checks them exhaustively; on top of that, each iteration drives one
	// randomized reachability probe (random variant, random target node)
	// confirmed against a randomized-frontier BFS, so reachability is exercised
	// across many orderings.
	const iterations = 200 // >= 100 per the property-test convention
	rng := rand.New(rand.NewSource(1))

	for i := 0; i < iterations; i++ {
		for _, profile := range profiles {
			prop13CheckEmbeddedGraphWellFormed(t, profile)
			prop13CheckBounds(t, profile)
		}

		// Randomized reachability probe over a random (variant, target) pair.
		profile := profiles[rng.Intn(len(profiles))]
		target := profile.NodeIDs[rng.Intn(len(profile.NodeIDs))]
		startIDs := prop13StartNodeIDs(profile.describingGraph())
		if !prop13ReachableRandomized(rng, profile.Edges, startIDs, target) {
			t.Fatalf("iteration %d: randomized probe found node %q in %s unreachable from start %v",
				i, target, profile.Path, startIDs)
		}
	}
}

// prop13ReachableRandomized reports whether target is reachable from any start id,
// using a BFS whose neighbor expansion order is shuffled by rng each call. It is an
// independent cross-check of the shared reachable() helper that varies traversal
// order across iterations; the reachability answer must not depend on order.
func prop13ReachableRandomized(rng *rand.Rand, edges []profileEdge, startIDs []string, target string) bool {
	for _, s := range startIDs {
		if s == target {
			return true
		}
	}

	adjacency := make(map[string][]string)
	for _, e := range edges {
		adjacency[e.SourceID] = append(adjacency[e.SourceID], e.DestinationID)
	}

	visited := make(map[string]bool)
	var queue []string
	for _, s := range startIDs {
		if !visited[s] {
			visited[s] = true
			queue = append(queue, s)
		}
	}

	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]

		neighbors := append([]string(nil), adjacency[current]...)
		rng.Shuffle(len(neighbors), func(a, b int) { neighbors[a], neighbors[b] = neighbors[b], neighbors[a] })

		for _, next := range neighbors {
			if next == target {
				return true
			}
			if !visited[next] {
				visited[next] = true
				queue = append(queue, next)
			}
		}
	}
	return false
}
