package main

import (
	"path/filepath"
	"testing"

	genpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v1/generator/proto"
)

// burstyAPIConfigPath is the bursty API base profile authored by task 5.1. Go
// tests run with the working directory set to the package dir (v1/generator/src),
// so the client config lives at ../../client/ relative to the test, the same way
// markov_analytics_profile_test.go reaches the shipped client configs.
var burstyAPIConfigPath = filepath.Join("..", "..", "client", "markov_bursty_api_config.textproto")

// burstyBaselineNodes and burstyBurstNodes partition the bursty profile's nodes
// into its two regimes by id, matching the design: baseline idles at a low rate,
// burst fires fast.
var (
	burstyBaselineNodes = map[string]bool{"baseline_poll": true, "baseline_status": true}
	burstyBurstNodes    = map[string]bool{"burst_read": true, "burst_write": true}
)

// TestBurstyAPIProfile_TwoRegimeStructureAndBurstRatio verifies the bursty API
// base profile's two-regime shape structurally over the loaded graph: it loads
// and validates as a plain markov_sampling graph, its two regimes are connected
// by edges in both directions (baseline->burst and burst->baseline), and the
// burst regime's action rate is at least 5x the baseline regime's rate.
//
// The regime rate is computed from node completion times, per the design's
// completion-time-to-action-rate mapping: regime rate = 60 / (mean completion
// seconds across the regime's nodes), where each node's completion seconds is the
// midpoint of its range, (min+max)/2.
//
// Validates: Requirements 3.7, 5.1
//
// No t.Parallel(): loadProfile calls loadConfigs, which sets the package-global
// resolvedBaseGraph, so harness callers must run sequentially.
func TestBurstyAPIProfile_TwoRegimeStructureAndBurstRatio(t *testing.T) {
	// Assertion 1: loads and validates via the harness as a plain markov_sampling
	// base profile with no injection.
	profile := loadProfile(t, burstyAPIConfigPath)
	if profile.Kind != scenarioMarkovSampling {
		t.Fatalf("bursty_api profile kind = %q, want %q", profile.Kind, scenarioMarkovSampling)
	}
	if profile.HasInjection {
		t.Fatalf("bursty_api base profile carries an injection/auto_injection, want none")
	}

	// Sanity-check the regime partition covers exactly the declared nodes, so a
	// renamed/added node can't silently fall outside both regimes and skew the
	// rate computation.
	for _, id := range profile.NodeIDs {
		if !burstyBaselineNodes[id] && !burstyBurstNodes[id] {
			t.Fatalf("node %q belongs to neither regime; update the regime partition", id)
		}
	}

	// Assertions 2 & 3: at least one edge in each cross-regime direction.
	var baselineToBurst, burstToBaseline int
	for _, e := range profile.Edges {
		switch {
		case burstyBaselineNodes[e.SourceID] && burstyBurstNodes[e.DestinationID]:
			baselineToBurst++
		case burstyBurstNodes[e.SourceID] && burstyBaselineNodes[e.DestinationID]:
			burstToBaseline++
		}
	}
	if baselineToBurst < 1 {
		t.Fatalf("found %d baseline->burst edges, want >= 1", baselineToBurst)
	}
	if burstToBaseline < 1 {
		t.Fatalf("found %d burst->baseline edges, want >= 1", burstToBaseline)
	}

	// Assertion 4: burst-regime action rate >= 5x baseline-regime rate, computed
	// from node completion-time midpoints.
	baselineRate := regimeRate(t, profile.describingGraph(), burstyBaselineNodes)
	burstRate := regimeRate(t, profile.describingGraph(), burstyBurstNodes)
	if baselineRate <= 0 {
		t.Fatalf("baseline regime rate = %.6f ops/min, want > 0", baselineRate)
	}

	const minRatio = 5.0
	ratio := burstRate / baselineRate
	if ratio < minRatio {
		t.Fatalf("burst/baseline rate ratio = %.3f (burst %.4f ops/min, baseline %.4f ops/min), want >= %.1f",
			ratio, burstRate, baselineRate, minRatio)
	}
}

// regimeRate returns the action rate in operations per minute for the regime
// whose node ids are in members: 60 / (mean completion seconds across the
// regime's nodes), where each node's completion seconds is the midpoint of its
// range. It fails the test if no regime node is found or any member lacks a usable
// completion-time range.
func regimeRate(t *testing.T, graph *genpb.MarkovSamplingConfig, members map[string]bool) float64 {
	t.Helper()

	var sum float64
	var count int
	for _, node := range graph.GetNodes() {
		if !members[node.GetId()] {
			continue
		}
		sum += completionMidpointSeconds(t, node)
		count++
	}
	if count == 0 {
		t.Fatalf("regimeRate: no nodes matched the regime; members=%v", members)
	}
	meanSeconds := sum / float64(count)
	if meanSeconds <= 0 {
		t.Fatalf("regimeRate: mean completion = %.6f s, want > 0", meanSeconds)
	}
	return 60.0 / meanSeconds
}

// completionMidpointSeconds returns the midpoint of a node's completion-time range
// in seconds, (min+max)/2, reading google.protobuf.Duration seconds+nanos. It
// fails the test if the node has no range completion time.
func completionMidpointSeconds(t *testing.T, node *genpb.MarkovNode) float64 {
	t.Helper()

	r := node.GetCompletionTime().GetRange()
	if r == nil {
		t.Fatalf("completionMidpointSeconds: node %q has no completion_time range", node.GetId())
	}
	minS := durationSeconds(r.GetMin())
	maxS := durationSeconds(r.GetMax())
	return (minS + maxS) / 2.0
}

// durationSeconds converts a google.protobuf.Duration to seconds as a float,
// folding in sub-second nanos.
func durationSeconds(d interface {
	GetSeconds() int64
	GetNanos() int32
}) float64 {
	if d == nil {
		return 0
	}
	return float64(d.GetSeconds()) + float64(d.GetNanos())/1e9
}
