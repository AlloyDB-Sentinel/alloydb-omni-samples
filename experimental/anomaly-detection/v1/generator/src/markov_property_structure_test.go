package main

// Property-based test for the markov-profile-library feature.
//
// Feature: markov-profile-library, Property 1: Base graphs are structurally well-formed
//
// Validates: Requirements 1.9, 5.1, 3.8
//
// Property 1 quantifies over the six authored base profiles: each loaded graph
// must be structurally well-formed. Concretely, for every base graph:
//
//   - node ids are non-empty and unique,
//   - every edge destination_id resolves to a declared node, and the start spec
//     (single node_id or weighted distribution) resolves only to declared nodes,
//   - all edge weights are finite and non-negative,
//   - every NON-terminal node (one with >= 1 outgoing edge) has outgoing weights
//     summing to a strictly positive total (terminal nodes such as session_done
//     carry zero outgoing edges and are excluded),
//   - every declared node is reachable from the start spec.
//
// The property ranges over a finite, enumerated set of six base profiles, so the
// structural invariants are checked exhaustively on every iteration. To honor the
// package's >= 100-iteration property-test convention, a testing/quick-style
// driver runs the full six-profile enumeration at least 100 times, and each
// iteration additionally drives a randomized reachability probe: it picks a random
// declared node in a random base graph and reconfirms it is reachable from the
// start via an independently randomized BFS frontier order, so the reachability
// guarantee is exercised across many randomized orderings rather than one fixed
// traversal.
//
// Helper names here are prefixed `prop1` to stay unique within the package and
// avoid collisions with other concurrently-authored test files. The shared
// `reachable` / `containsNode` helpers (defined in markov_batch_etl_profile_test.go)
// are reused and NOT redefined.
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

// prop1BaseProfilePaths enumerates the six new base profiles authored under
// ../../client/. Go tests run with the working directory set to the package dir
// (v1/generator/src), so client configs live at ../../client/, matching the path
// convention in markov_analytics_profile_test.go.
var prop1BaseProfilePaths = []string{
	filepath.Join("..", "..", "client", "markov_analytics_user_config.textproto"),
	filepath.Join("..", "..", "client", "markov_batch_etl_config.textproto"),
	filepath.Join("..", "..", "client", "markov_read_replica_config.textproto"),
	filepath.Join("..", "..", "client", "markov_bursty_api_config.textproto"),
	filepath.Join("..", "..", "client", "markov_noisy_client_config.textproto"),
	filepath.Join("..", "..", "client", "markov_oltp_app_config.textproto"),
}

// prop1StartNodeIDs returns every node id a graph's start spec resolves to: the
// single node_id for a point start, or each distribution entry's node_id for a
// weighted start. A base graph must set exactly one of these.
func prop1StartNodeIDs(graph *genpb.MarkovSamplingConfig) []string {
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

// prop1CheckWellFormed asserts the full set of structural invariants for one
// loaded base profile. It is deterministic and independent of iteration count.
func prop1CheckWellFormed(t *testing.T, profile *profileResult) {
	t.Helper()

	graph := profile.describingGraph()
	if graph == nil {
		t.Fatalf("%s: base profile has no describing graph", profile.Path)
	}

	// --- node ids non-empty and unique ---
	seen := make(map[string]bool, len(profile.NodeIDs))
	for _, id := range profile.NodeIDs {
		if id == "" {
			t.Fatalf("%s: graph declares a node with an empty id", profile.Path)
		}
		if seen[id] {
			t.Fatalf("%s: duplicate node id %q", profile.Path, id)
		}
		seen[id] = true
	}
	if len(profile.NodeIDs) == 0 {
		t.Fatalf("%s: graph declares no nodes", profile.Path)
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
	startIDs := prop1StartNodeIDs(graph)
	if len(startIDs) == 0 {
		t.Fatalf("%s: graph sets no start spec", profile.Path)
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
		if !prop1ReachableFromAnyStart(profile.Edges, startIDs, target) {
			t.Fatalf("%s: node %q is not reachable from any start node %v", profile.Path, target, startIDs)
		}
	}
}

// prop1ReachableFromAnyStart reports whether target is reachable from at least one
// of the start node ids via the directed edge set, reusing the shared reachable()
// BFS helper.
func prop1ReachableFromAnyStart(edges []profileEdge, startIDs []string, target string) bool {
	for _, start := range startIDs {
		if reachable(edges, start, target) {
			return true
		}
	}
	return false
}

// TestProperty1_BaseGraphsStructurallyWellFormed is the Property 1 driver.
//
// Feature: markov-profile-library, Property 1: Base graphs are structurally well-formed
//
// Validates: Requirements 1.9, 5.1, 3.8
func TestProperty1_BaseGraphsStructurallyWellFormed(t *testing.T) {
	// Load each base profile once; loadProfile fails the test on any load or
	// validation error and re-checks plain base graphs with validateMarkovSampling.
	profiles := make([]*profileResult, 0, len(prop1BaseProfilePaths))
	for _, path := range prop1BaseProfilePaths {
		profile := loadProfile(t, path)
		if profile.Kind != scenarioMarkovSampling {
			t.Fatalf("%s: base profile kind = %q, want %q", path, profile.Kind, scenarioMarkovSampling)
		}
		profiles = append(profiles, profile)
	}

	// Property driver: run the full enumeration >= 100 times. The structural
	// invariants are deterministic over the finite set, so each iteration checks
	// them exhaustively; on top of that, each iteration drives one randomized
	// reachability probe (random profile, random target node) confirmed against a
	// randomized-frontier BFS, so reachability is exercised across many orderings.
	const iterations = 200 // >= 100 per the property-test convention
	rng := rand.New(rand.NewSource(1))

	for i := 0; i < iterations; i++ {
		for _, profile := range profiles {
			prop1CheckWellFormed(t, profile)
		}

		// Randomized reachability probe over a random (profile, target) pair.
		profile := profiles[rng.Intn(len(profiles))]
		target := profile.NodeIDs[rng.Intn(len(profile.NodeIDs))]
		startIDs := prop1StartNodeIDs(profile.describingGraph())
		if !prop1ReachableRandomized(rng, profile.Edges, startIDs, target) {
			t.Fatalf("iteration %d: randomized probe found node %q in %s unreachable from start %v",
				i, target, profile.Path, startIDs)
		}
	}
}

// prop1ReachableRandomized reports whether target is reachable from any start id,
// using a BFS whose neighbor expansion order is shuffled by rng each call. It is an
// independent cross-check of the shared reachable() helper that varies traversal
// order across iterations; the reachability answer must not depend on order.
func prop1ReachableRandomized(rng *rand.Rand, edges []profileEdge, startIDs []string, target string) bool {
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
