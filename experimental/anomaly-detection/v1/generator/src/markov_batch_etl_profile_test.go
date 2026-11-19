package main

import (
	"path/filepath"
	"testing"
)

// batchETLConfigPath is the batch ETL / bulk loader base profile authored by
// task 3.1. Go tests run with the working directory set to the package dir
// (v1/generator/src), so the client config lives at ../../client/ relative to
// the test, the same path convention markov_analytics_profile_test.go uses.
var batchETLConfigPath = filepath.Join("..", "..", "client", "markov_batch_etl_config.textproto")

// batchETLVariantPaths enumerates the six variant files authored by task 3.2,
// under ../../client/markov_profiles/batch_etl/. Each must load and validate
// through the harness (loadProfile fails the test on any load/validation error).
var batchETLVariantPaths = []string{
	filepath.Join("..", "..", "client", "markov_profiles", "batch_etl", "subtype.textproto"),
	filepath.Join("..", "..", "client", "markov_profiles", "batch_etl", "individual_overrides.textproto"),
	filepath.Join("..", "..", "client", "markov_profiles", "batch_etl", "individual_overrides_b.textproto"),
	filepath.Join("..", "..", "client", "markov_profiles", "batch_etl", "individual_variance.textproto"),
	filepath.Join("..", "..", "client", "markov_profiles", "batch_etl", "individual_jitter.textproto"),
	filepath.Join("..", "..", "client", "markov_profiles", "batch_etl", "auto_injection.textproto"),
}

// TestBatchETLProfile_LoadsTerminatesAndIsWriteDominant verifies the batch ETL
// unit through the markov-profile-library harness:
//
//  1. The base profile and all six variants load and validate (R5.1, R5.2).
//  2. The terminal "session_done" node is reachable from the start node and has
//     zero outgoing edges, so each walk is a bounded batch session (R2.6, R3.2).
//  3. A >= 1,000-action sample is write-dominant at ~70% +/- 0.05 write (R6.1,
//     R6.4), matching the profile's documented Behavioral_Intent.
//
// Validates: Requirements 2.6, 3.2, 5.1, 5.2, 6.1, 6.4
//
// No t.Parallel(): loadProfile/sampleProfile call loadConfigs, which sets the
// package-global resolvedBaseGraph, so harness callers must run sequentially.
func TestBatchETLProfile_LoadsTerminatesAndIsWriteDominant(t *testing.T) {
	// 1a. The base profile loads and validates as a plain markov_sampling graph
	// with no injection. loadProfile re-checks base graphs with
	// validateMarkovSampling and fails on any error.
	base := loadProfile(t, batchETLConfigPath)
	if base.Kind != scenarioMarkovSampling {
		t.Fatalf("batch_etl base profile kind = %q, want %q", base.Kind, scenarioMarkovSampling)
	}
	if base.HasInjection {
		t.Fatalf("batch_etl base profile carries an injection/auto_injection, want none")
	}

	// 1b. All six variants load and validate. loadProfile resolves each
	// variant's base_graph_path and runs validateConfigs; any failure fails here.
	for _, variantPath := range batchETLVariantPaths {
		variant := loadProfile(t, variantPath)
		if variant.Kind == scenarioOther {
			t.Fatalf("batch_etl variant %q loaded as scenario kind %q, want a markov scenario", variantPath, variant.Kind)
		}
	}

	// 2. The terminal session_done node is reachable from the start node and has
	// zero outgoing edges. Inspect the loaded base graph's nodes and edges.
	const startNode = "extract_read"
	const terminalNode = "session_done"

	if !containsNode(base.NodeIDs, terminalNode) {
		t.Fatalf("base graph has no %q node; node ids = %v", terminalNode, base.NodeIDs)
	}
	if !containsNode(base.NodeIDs, startNode) {
		t.Fatalf("base graph has no %q start node; node ids = %v", startNode, base.NodeIDs)
	}

	// Zero outgoing edges from the terminal node.
	for _, e := range base.Edges {
		if e.SourceID == terminalNode {
			t.Fatalf("terminal node %q has an outgoing edge to %q (weight %v), want zero outgoing edges",
				terminalNode, e.DestinationID, e.Weight)
		}
	}

	// Reachable from the start node via the edge set (forward BFS).
	if !reachable(base.Edges, startNode, terminalNode) {
		t.Fatalf("terminal node %q is not reachable from start node %q via the edge set", terminalNode, startNode)
	}

	// 3. A >= 1,000-action sample is write-dominant at ~70% +/- 0.05. Check a
	// couple of seeds so the result does not hinge on a single RNG stream.
	const numLogs = 2000
	const (
		minWriteFraction = 0.65
		maxWriteFraction = 0.75
	)
	for _, seed := range []int64{1, 7, 42} {
		sample := sampleProfile(t, batchETLConfigPath, numLogs, seed)
		if got := len(sample.Actions); got != numLogs {
			t.Fatalf("seed %d: sample action count = %d, want %d", seed, got, numLogs)
		}
		wf := sample.WriteFraction()
		if wf < minWriteFraction || wf > maxWriteFraction {
			t.Fatalf("seed %d: observed write fraction = %.4f (%d writes / %d actions), want within [%.2f, %.2f] (~70%% +/- 0.05)",
				seed, wf, sample.WriteCount, len(sample.Actions), minWriteFraction, maxWriteFraction)
		}
	}
}

// containsNode reports whether id appears in the node id list.
func containsNode(ids []string, id string) bool {
	for _, n := range ids {
		if n == id {
			return true
		}
	}
	return false
}

// reachable reports whether target is reachable from start by following the
// directed edge set (breadth-first, ignoring edge weights).
func reachable(edges []profileEdge, start, target string) bool {
	if start == target {
		return true
	}
	adjacency := make(map[string][]string)
	for _, e := range edges {
		adjacency[e.SourceID] = append(adjacency[e.SourceID], e.DestinationID)
	}
	visited := map[string]bool{start: true}
	queue := []string{start}
	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]
		for _, next := range adjacency[current] {
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
