package main

// Property-based test for the markov-profile-library feature.
//
// Feature: markov-profile-library, Property 11: Variance variants carry an asymmetric, positive variance entry
//
// Validates: Requirements 4.4
//
// Property 11 quantifies over the six authored variant sets. Each profile dir
// under ../../client/markov_profiles/<name>/ ships an individual_variance.textproto
// (a markov_individual variant). For every such file:
//
//   - the loaded config is a markov_individual scenario,
//   - its edge_variances list contains >= 1 entry that is asymmetric
//     (down != up) AND positive (at least one of down/up > 0),
//   - that entry's down and up are both finite and non-negative.
//
// The property ranges over a finite, enumerated set of six variance files, so the
// invariant is checked exhaustively on every iteration. To honor the package's
// >= 100-iteration property-test convention, a testing/quick-style driver runs the
// full six-file enumeration at least 100 times, and each iteration additionally
// drives a randomized probe: it picks a random variance file and reconfirms, over
// a randomized scan order of that file's edge_variances, that a qualifying
// asymmetric-positive entry exists, so the guarantee is exercised across many
// orderings rather than one fixed scan.
//
// Helper names here are prefixed `prop11` to stay unique within the package and
// avoid collisions with other concurrently-authored test files. Shared harness
// helpers (loadProfile / profileResult) are reused and NOT redefined.
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

// prop11VarianceFilePaths enumerates the individual_variance.textproto file for
// each of the six variant sets authored under ../../client/markov_profiles/. Go
// tests run with the working directory set to the package dir (v1/generator/src),
// so client configs live at ../../client/, matching the path convention in
// markov_analytics_profile_test.go.
var prop11VarianceFilePaths = []string{
	filepath.Join("..", "..", "client", "markov_profiles", "analytics", "individual_variance.textproto"),
	filepath.Join("..", "..", "client", "markov_profiles", "batch_etl", "individual_variance.textproto"),
	filepath.Join("..", "..", "client", "markov_profiles", "read_replica", "individual_variance.textproto"),
	filepath.Join("..", "..", "client", "markov_profiles", "bursty_api", "individual_variance.textproto"),
	filepath.Join("..", "..", "client", "markov_profiles", "noisy_client", "individual_variance.textproto"),
	filepath.Join("..", "..", "client", "markov_profiles", "oltp_app", "individual_variance.textproto"),
}

// prop11IsAsymmetricPositive reports whether a single variance entry is asymmetric
// (down != up) AND positive (at least one of down/up > 0), with both down and up
// finite and non-negative. An entry failing the finite/non-negative guard does not
// qualify even if it looks asymmetric or positive.
func prop11IsAsymmetricPositive(ev *genpb.EdgeVariance) bool {
	down := ev.GetDown()
	up := ev.GetUp()

	// Both deltas must be finite and non-negative.
	if math.IsNaN(down) || math.IsInf(down, 0) || down < 0 {
		return false
	}
	if math.IsNaN(up) || math.IsInf(up, 0) || up < 0 {
		return false
	}

	asymmetric := down != up
	positive := down > 0 || up > 0
	return asymmetric && positive
}

// prop11CheckVarianceFile asserts Property 11's invariant for one loaded variance
// profile: it is a markov_individual scenario whose edge_variances carry >= 1
// asymmetric, positive, finite, non-negative entry. It is deterministic and
// independent of iteration count.
func prop11CheckVarianceFile(t *testing.T, profile *profileResult) {
	t.Helper()

	if profile.Kind != scenarioMarkovIndividual {
		t.Fatalf("%s: variance profile kind = %q, want %q",
			profile.Path, profile.Kind, scenarioMarkovIndividual)
	}

	variances := profile.Configs.GetMarkovIndividual().GetEdgeVariances()
	if len(variances) == 0 {
		t.Fatalf("%s: markov_individual carries no edge_variances, want >= 1 asymmetric positive entry",
			profile.Path)
	}

	for _, ev := range variances {
		if prop11IsAsymmetricPositive(ev) {
			return // found a qualifying entry; invariant satisfied
		}
	}
	t.Fatalf("%s: no edge_variances entry is asymmetric (down != up) and positive "+
		"(down > 0 || up > 0) with both finite and non-negative", profile.Path)
}

// TestProperty11_VarianceVariantsAsymmetricPositive is the Property 11 driver.
//
// Feature: markov-profile-library, Property 11: Variance variants carry an asymmetric, positive variance entry
//
// Validates: Requirements 4.4
func TestProperty11_VarianceVariantsAsymmetricPositive(t *testing.T) {
	// Load each variance file once; loadProfile fails the test on any load or
	// validation error.
	profiles := make([]*profileResult, 0, len(prop11VarianceFilePaths))
	for _, path := range prop11VarianceFilePaths {
		profiles = append(profiles, loadProfile(t, path))
	}

	// Property driver: run the full enumeration >= 100 times. The invariant is
	// deterministic over the finite set, so each iteration checks it exhaustively;
	// on top of that, each iteration drives one randomized probe (random file,
	// randomized scan order) confirming a qualifying entry still surfaces
	// regardless of traversal order.
	const iterations = 200 // >= 100 per the property-test convention
	rng := rand.New(rand.NewSource(1))

	for i := 0; i < iterations; i++ {
		for _, profile := range profiles {
			prop11CheckVarianceFile(t, profile)
		}

		// Randomized probe over a random variance file with a shuffled scan order.
		profile := profiles[rng.Intn(len(profiles))]
		if !prop11HasQualifyingEntryRandomized(rng, profile) {
			t.Fatalf("iteration %d: randomized probe found no asymmetric positive variance entry in %s",
				i, profile.Path)
		}
	}
}

// prop11HasQualifyingEntryRandomized reports whether profile's edge_variances
// contain an asymmetric-positive entry, scanning the entries in an order shuffled
// by rng each call. It is an independent cross-check that the answer does not
// depend on scan order.
func prop11HasQualifyingEntryRandomized(rng *rand.Rand, profile *profileResult) bool {
	variances := profile.Configs.GetMarkovIndividual().GetEdgeVariances()
	order := rng.Perm(len(variances))
	for _, idx := range order {
		if prop11IsAsymmetricPositive(variances[idx]) {
			return true
		}
	}
	return false
}
