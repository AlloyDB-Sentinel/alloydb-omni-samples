package main

// Property-based test for the markov-profile-library feature.
//
// Feature: markov-profile-library, Property 15: In-range runs emit exactly the requested number of actions
//
// Validates: Requirements 6.1
//
// Property 15 quantifies over the six authored base profiles and requested action
// counts in the inclusive range [1, 100000]: for any base profile and any count in
// range, a generated sample must contain EXACTLY that many actions. The check is
// sampleProfile(path, count, seed) followed by len(sample.Actions) == count.
//
// The requested-count space [1, 100000] is large and large counts are expensive to
// generate, so the driver samples it rather than enumerating it exhaustively:
//
//   - Boundary coverage: {1, 2, 100000} is tested once per profile so both the
//     minimum in-range counts and the maximum are exercised. The count=100000 runs
//     use noisy_client (sub-second completion nodes) so the upper bound is checked
//     on a fast profile without running many 100k samples across the board.
//   - Randomized coverage: ~100+ randomized moderate counts in [1, 5000] are spread
//     across the six profiles, satisfying the package's >= 100-iteration convention
//     while keeping total runtime modest.
//
// Helper names here are prefixed `prop15` to stay unique within the package and
// avoid collisions with other concurrently-authored test files. The shared harness
// helpers (sampleProfile, sampleResult) from markov_profile_harness_test.go are
// reused and NOT redefined.
//
// No t.Parallel(): sampleProfile calls loadConfigs, which sets the package-global
// resolvedBaseGraph, so harness callers must run sequentially.

import (
	"math/rand"
	"path/filepath"
	"testing"
)

// prop15BaseProfile names one authored base profile and its config path (relative
// to the package dir v1/generator/src, matching the path convention in
// markov_analytics_profile_test.go and markov_property_mix_test.go).
type prop15BaseProfile struct {
	// name is a short label for failure messages.
	name string
	// path is the on-disk config path, relative to the package dir.
	path string
	// fast marks a profile whose nodes complete sub-second, so running a very
	// large (near-100000) sample on it stays within reasonable test runtime.
	fast bool
}

// prop15BaseProfiles enumerates the six base profiles under test.
var prop15BaseProfiles = []prop15BaseProfile{
	{name: "analytics_user", path: filepath.Join("..", "..", "client", "markov_analytics_user_config.textproto")},
	{name: "read_replica", path: filepath.Join("..", "..", "client", "markov_read_replica_config.textproto")},
	{name: "batch_etl", path: filepath.Join("..", "..", "client", "markov_batch_etl_config.textproto")},
	{name: "bursty_api", path: filepath.Join("..", "..", "client", "markov_bursty_api_config.textproto")},
	// noisy_client has sub-second completions, so it is the profile used to
	// exercise the count=100000 upper bound.
	{name: "noisy_client", path: filepath.Join("..", "..", "client", "markov_noisy_client_config.textproto"), fast: true},
	{name: "oltp_app", path: filepath.Join("..", "..", "client", "markov_oltp_app_config.textproto")},
}

// TestProperty15_InRangeRunsEmitExactCount is the Property 15 driver.
//
// Feature: markov-profile-library, Property 15: In-range runs emit exactly the requested number of actions
//
// Validates: Requirements 6.1
//
// It asserts that every in-range requested count produces a sample of exactly that
// many actions, across all six base profiles. Coverage combines per-profile
// boundary values {1, 2, 100000} with ~100 randomized moderate counts spread across
// the profiles, keeping the total iteration count >= 100 while bounding runtime.
func TestProperty15_InRangeRunsEmitExactCount(t *testing.T) {
	const (
		rangeLow  = 1
		rangeHigh = 100000
		// moderateHigh bounds the randomized counts so the bulk of the iterations
		// run quickly; the full upper bound is still exercised via the boundary
		// values below.
		moderateHigh = 5000
		// randomIterations is the number of randomized moderate-count checks,
		// spread across the six profiles. Combined with the boundary checks this
		// clears the package's >= 100-iteration property-test convention.
		randomIterations = 108
	)

	totalChecks := 0

	// assertExactCount samples the profile for the requested count and asserts the
	// emitted action count matches exactly. It uses t.Errorf (not Fatalf) so a
	// single mismatch does not hide the remaining coverage.
	assertExactCount := func(name, path string, count int, seed int64) {
		sample := sampleProfile(t, path, count, seed)
		if got := len(sample.Actions); got != count {
			t.Errorf("%s: requested count %d (seed %d): emitted %d actions, want exactly %d",
				name, count, seed, got, count)
		}
		totalChecks++
	}

	// Boundary coverage: {1, 2} once per profile exercises the low end of the
	// in-range interval for every profile. The count=100000 upper bound is run
	// once, on the fast noisy_client profile, to keep runtime reasonable.
	for _, p := range prop15BaseProfiles {
		assertExactCount(p.name, p.path, rangeLow, 1)   // count = 1
		assertExactCount(p.name, p.path, rangeLow+1, 2) // count = 2
		if p.fast {
			assertExactCount(p.name, p.path, rangeHigh, 3) // count = 100000
		}
	}

	// Randomized coverage: deterministic RNG (fixed seed) so failures reproduce.
	// Each iteration picks a random profile and a random moderate count in
	// [1, 5000], biased toward smaller values by squaring a uniform draw so the
	// bulk of the runs are cheap.
	rng := rand.New(rand.NewSource(0xC0FFEE))
	for i := 0; i < randomIterations; i++ {
		p := prop15BaseProfiles[rng.Intn(len(prop15BaseProfiles))]

		// Bias toward small/medium counts: a squared uniform draw in [0,1) keeps
		// most counts well below moderateHigh while still reaching toward it.
		u := rng.Float64()
		count := rangeLow + int(u*u*float64(moderateHigh-rangeLow))
		if count < rangeLow {
			count = rangeLow
		}
		if count > moderateHigh {
			count = moderateHigh
		}

		assertExactCount(p.name, p.path, count, int64(1000+i))
	}

	if totalChecks < 100 {
		t.Fatalf("property driver performed only %d count checks, want >= 100", totalChecks)
	}
	t.Logf("Property 15: performed %d exact-count checks across %d base profiles (incl. count=1, 2, and 100000)",
		totalChecks, len(prop15BaseProfiles))
}
