package main

// Property-based test for the markov-profile-library feature.
//
// Feature: markov-profile-library, Property 7: Read-only profiles emit zero write actions
//
// Validates: Requirements 3.5, 3.6, 6.3
//
// Property 7 quantifies over the two authored read-only base profiles,
// analytics (markov_analytics_user_config.textproto) and read-replica
// (markov_read_replica_config.textproto). For each, the property asserts both a
// static and a behavioral guarantee:
//
//   - Static: no node's query template is data-modifying. Every template is
//     classified with the harness's read/write classifier (isWriteTemplate) and
//     must classify as a read. (Requirements 3.5 read-only analytics, 3.6
//     read-only read-replica.)
//   - Behavioral: a >= 1,000-action sample generated through the production
//     load -> GenerateLogs path contains zero write actions
//     (sampleResult.WriteCount == 0). (Requirement 6.3.)
//
// Additionally, for the analytics profile specifically (Requirement 3.5's rate
// and completion-time guarantees):
//
//   - Every node's completion-time range has a minimum of >= 30 s, read from the
//     loaded proto.
//   - The observed action rate of a >= 1,000-action sample is <= 2 ops/min,
//     computed as rate = (N-1) / (ElapsedMs / 60000) over the simulated clock.
//
// The property ranges over a finite, enumerated set of profiles under several
// seeds. To honor the package's >= 100-iteration property-test convention, the
// driver runs the full per-profile check sweep across a slate of seeds until it
// has executed at least 100 sample/enumeration checks, re-sweeping as needed.
//
// Helper names here are prefixed `prop7` to stay unique within the package and
// avoid collisions with other concurrently-authored test files. The shared
// harness helpers (loadProfile, sampleProfile, isWriteTemplate, sampleResult) and
// the shared completion-time helper durationSeconds (defined in
// markov_bursty_api_profile_test.go) are reused and NOT redefined.
//
// No t.Parallel(): loadProfile / sampleProfile call loadConfigs, which sets the
// package-global resolvedBaseGraph, so harness callers must run sequentially.

import (
	"path/filepath"
	"testing"
)

// prop7ReadOnlyProfiles enumerates the two new read-only base profiles, reached at
// ../../client/ the same way the other profile tests reach the shipped client
// configs (Go tests run with the package dir as cwd).
var prop7AnalyticsProfile = filepath.Join("..", "..", "client", "markov_analytics_user_config.textproto")
var prop7ReadReplicaProfile = filepath.Join("..", "..", "client", "markov_read_replica_config.textproto")

var prop7ReadOnlyProfiles = []string{
	prop7AnalyticsProfile,
	prop7ReadReplicaProfile,
}

// prop7SampleSize is the minimum action count each behavioral sample draws; the
// property requires a >= 1,000-action sample per profile.
const prop7SampleSize = 1000

// prop7Seeds is the slate of seeds the sample checks sweep over. Several seeds
// exercise different random walks through each read-only graph; the zero-write
// guarantee must hold under every one.
var prop7Seeds = []int64{1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43}

// prop7AnalyticsMinCompletionSeconds is the lower bound every analytics node's
// completion-time range minimum must meet (Requirement 3.5).
const prop7AnalyticsMinCompletionSeconds = 30.0

// prop7AnalyticsMaxRateOpsPerMin is the ceiling on the analytics profile's
// observed action rate, in operations per minute (Requirement 3.5).
const prop7AnalyticsMaxRateOpsPerMin = 2.0

// prop7AssertTemplatesAllRead loads the profile at path and asserts that no node's
// query template classifies as a write under the harness classifier, so the
// profile is statically read-only.
func prop7AssertTemplatesAllRead(t *testing.T, path string) *profileResult {
	t.Helper()

	profile := loadProfile(t, path)
	graph := profile.describingGraph()
	if graph == nil {
		t.Fatalf("prop7: profile %q produced no describing graph", path)
	}

	nodeCount := 0
	for _, node := range graph.GetNodes() {
		nodeCount++
		template := node.GetTemplate().GetNormalizedQueryTemplate()
		if isWriteTemplate(template) {
			t.Fatalf("prop7: profile %q node %q is data-modifying, want read-only: %q",
				path, node.GetId(), template)
		}
	}
	if nodeCount == 0 {
		t.Fatalf("prop7: profile %q declares no nodes", path)
	}
	return profile
}

// prop7AssertAnalyticsCompletionFloor asserts every node in the analytics graph
// has a completion-time range whose minimum is >= 30 s, read from the loaded
// proto (Requirement 3.5).
func prop7AssertAnalyticsCompletionFloor(t *testing.T, profile *profileResult) {
	t.Helper()

	graph := profile.describingGraph()
	if graph == nil {
		t.Fatalf("prop7: analytics profile produced no describing graph")
	}
	for _, node := range graph.GetNodes() {
		r := node.GetCompletionTime().GetRange()
		if r == nil {
			t.Fatalf("prop7: analytics node %q has no completion_time range", node.GetId())
		}
		minSeconds := durationSeconds(r.GetMin())
		if minSeconds < prop7AnalyticsMinCompletionSeconds {
			t.Fatalf("prop7: analytics node %q completion min = %.3f s, want >= %.0f s",
				node.GetId(), minSeconds, prop7AnalyticsMinCompletionSeconds)
		}
	}
}

// prop7ObservedRateOpsPerMin computes the observed action rate of a sample in
// operations per minute: rate = (N-1) / (ElapsedMs / 60000). It returns 0 for a
// sample with fewer than two actions or a non-positive elapsed span (no
// meaningful rate to report).
func prop7ObservedRateOpsPerMin(sample *sampleResult) float64 {
	n := len(sample.Actions)
	if n < 2 {
		return 0
	}
	elapsedMs := sample.ElapsedMs()
	if elapsedMs <= 0 {
		return 0
	}
	elapsedMinutes := float64(elapsedMs) / 60000.0
	return float64(n-1) / elapsedMinutes
}

// TestProperty7_ReadOnlyProfilesEmitZeroWrites is the Property 7 driver.
//
// Feature: markov-profile-library, Property 7: Read-only profiles emit zero write actions
//
// Validates: Requirements 3.5, 3.6, 6.3
func TestProperty7_ReadOnlyProfilesEmitZeroWrites(t *testing.T) {
	// Static read-only check: every template in each read-only profile must
	// classify as a read. Load each profile once up front.
	profiles := make(map[string]*profileResult, len(prop7ReadOnlyProfiles))
	for _, path := range prop7ReadOnlyProfiles {
		profiles[path] = prop7AssertTemplatesAllRead(t, path)
	}

	// Analytics-specific static guarantee: every completion min >= 30 s.
	prop7AssertAnalyticsCompletionFloor(t, profiles[prop7AnalyticsProfile])

	// Behavioral property driver: sweep every read-only profile across the seed
	// slate, drawing a >= 1,000-action sample each time and asserting zero writes.
	// The analytics profile additionally must show an observed rate <= 2 ops/min.
	// Re-sweep the (profile, seed) grid until at least 100 sample checks have run,
	// honoring the >= 100-iteration property-test convention.
	const minIterations = 100
	checks := 0
	for checks < minIterations {
		for _, path := range prop7ReadOnlyProfiles {
			for _, seed := range prop7Seeds {
				sample := sampleProfile(t, path, prop7SampleSize, seed)

				if got := len(sample.Actions); got < prop7SampleSize {
					t.Fatalf("prop7: profile %q seed %d produced %d actions, want >= %d",
						path, seed, got, prop7SampleSize)
				}
				if sample.WriteCount != 0 {
					t.Fatalf("prop7: profile %q seed %d emitted %d write actions, want 0",
						path, seed, sample.WriteCount)
				}

				if path == prop7AnalyticsProfile {
					rate := prop7ObservedRateOpsPerMin(sample)
					if rate > prop7AnalyticsMaxRateOpsPerMin {
						t.Fatalf("prop7: analytics seed %d observed rate = %.4f ops/min, want <= %.1f",
							seed, rate, prop7AnalyticsMaxRateOpsPerMin)
					}
				}

				checks++
			}
		}
	}

	if checks < minIterations {
		t.Fatalf("prop7: ran %d sample checks, want >= %d", checks, minIterations)
	}
}
