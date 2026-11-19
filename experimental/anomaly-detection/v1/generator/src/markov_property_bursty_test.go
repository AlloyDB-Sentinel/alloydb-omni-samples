package main

// Property test for the markov-profile-library feature.
//
// Feature: markov-profile-library, Property 8: Bursty profile connects two
// regimes with a >=5x burst ratio
//
// Validates: Requirements 3.7
//
// This file adds a single property-based test that, across many randomized
// iterations, re-checks the bursty_api base profile's two-regime invariant over
// its loaded graph: the baseline regime {baseline_poll, baseline_status} and the
// burst regime {burst_read, burst_write} are connected by at least one edge in
// each direction, and the burst regime's action rate (from completion-time
// midpoints) is at least 5x the baseline regime's rate.
//
// Task 5.3 (TestBurstyAPIProfile_TwoRegimeStructureAndBurstRatio in
// markov_bursty_api_profile_test.go) already asserts this invariant once; this
// test wraps the same approach in a testing/quick driver that re-checks the
// invariant over >= 100 iterations to satisfy the feature's property-test
// convention and carry the Property 8 tag.
//
// It reuses the package-level helpers and vars defined by task 5.3 directly
// (burstyAPIConfigPath, burstyBaselineNodes, burstyBurstNodes, regimeRate) and the
// harness (loadProfile / profileResult) without redeclaring any of them. Any
// helper added here is prefixed prop8 to stay unique within the package.
//
// No production-code or proto changes; it only inspects authored artifacts.

import (
	"math/rand"
	"testing"
	"testing/quick"
)

// prop8MinBurstRatio is the minimum burst/baseline action-rate ratio required by
// the design's two-regime bursty profile (burst rate >= 5x baseline rate).
const prop8MinBurstRatio = 5.0

// TestProp8BurstyProfileConnectsTwoRegimes is Property 8: the bursty_api profile
// connects a baseline regime and a burst regime with at least one edge in each
// direction, and the burst regime's action rate is at least 5x the baseline
// regime's rate.
//
// Feature: markov-profile-library, Property 8: Bursty profile connects two
// regimes with a >=5x burst ratio
//
// Validates: Requirements 3.7
//
// The invariant is a structural property of a single loaded graph, so the test
// loads the bursty_api base profile once through the harness (loadProfile calls
// loadConfigs, which sets the package-global resolvedBaseGraph, so this test must
// not run in parallel with other harness callers) and then drives testing/quick
// to re-check the invariant over >= 100 iterations, matching the package's
// property-test convention.
func TestProp8BurstyProfileConnectsTwoRegimes(t *testing.T) {
	// Load once: the invariant is deterministic over the loaded graph, and
	// loadProfile mutates package-global state, so re-loading per iteration would
	// add cost without exercising anything new.
	profile := loadProfile(t, burstyAPIConfigPath)

	// Guard the regime partition up front: a renamed or added node that falls
	// outside both regimes would silently skew the rate computation.
	for _, id := range profile.NodeIDs {
		if !burstyBaselineNodes[id] && !burstyBurstNodes[id] {
			t.Fatalf("node %q belongs to neither regime; update the regime partition", id)
		}
	}

	property := func(_ uint16) bool {
		return prop8AssertTwoRegimeBurstRatio(t, profile)
	}

	cfg := &quick.Config{
		MaxCount: 150, // >= 100 iterations re-checking the loaded graph's invariant.
		Rand:     rand.New(rand.NewSource(8)),
	}
	if err := quick.Check(property, cfg); err != nil {
		t.Fatalf("Property 8 failed: %v", err)
	}
}

// prop8AssertTwoRegimeBurstRatio re-checks Property 8 over an already-loaded
// bursty_api profile: it must be a plain markov_sampling graph with no injection,
// have at least one baseline->burst edge and one burst->baseline edge, and have a
// burst-regime action rate at least prop8MinBurstRatio times the baseline-regime
// rate (rates from completion-time midpoints via the reused regimeRate helper). It
// fails the test (and returns false) on any violation so testing/quick reports it.
func prop8AssertTwoRegimeBurstRatio(t *testing.T, profile *profileResult) bool {
	t.Helper()

	// Plain markov_sampling base profile with no injection.
	if profile.Kind != scenarioMarkovSampling {
		t.Errorf("bursty_api profile kind = %q, want %q", profile.Kind, scenarioMarkovSampling)
		return false
	}
	if profile.HasInjection {
		t.Errorf("bursty_api base profile carries an injection/auto_injection, want none")
		return false
	}

	// At least one edge in each cross-regime direction.
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
		t.Errorf("found %d baseline->burst edges, want >= 1", baselineToBurst)
		return false
	}
	if burstToBaseline < 1 {
		t.Errorf("found %d burst->baseline edges, want >= 1", burstToBaseline)
		return false
	}

	// Burst-regime action rate >= 5x baseline-regime rate, from completion-time
	// midpoints via the reused regimeRate helper (defined in task 5.3).
	graph := profile.describingGraph()
	baselineRate := regimeRate(t, graph, burstyBaselineNodes)
	burstRate := regimeRate(t, graph, burstyBurstNodes)
	if baselineRate <= 0 {
		t.Errorf("baseline regime rate = %.6f ops/min, want > 0", baselineRate)
		return false
	}
	ratio := burstRate / baselineRate
	if ratio < prop8MinBurstRatio {
		t.Errorf("burst/baseline rate ratio = %.3f (burst %.4f ops/min, baseline %.4f ops/min), want >= %.1f",
			ratio, burstRate, baselineRate, prop8MinBurstRatio)
		return false
	}

	return true
}
