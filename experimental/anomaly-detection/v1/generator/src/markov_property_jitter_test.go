package main

// Property test for the markov-profile-library feature.
//
// Feature: markov-profile-library, Property 12: Jitter variants carry only a
// bound
//
// Validates: Requirements 4.5
//
// Each profile dir under v1/client/markov_profiles/<name>/ ships an
// individual_jitter.textproto that demonstrates the weight-jitter mechanism in
// isolation: a markov_individual config whose only perturbation knob is a single
// global weight_jitter.bound. This property asserts, across many randomized
// iterations over the six jitter variants, that each one:
//
//   - loads as a markov_individual scenario,
//   - sets weight_jitter with a single finite, non-negative bound,
//   - carries NO individual_edge_overrides, and
//   - carries NO edge_variances.
//
// It reuses the markov-profile-library harness (loadProfile / profileResult) and
// the ../../client/ path convention established by
// markov_analytics_profile_test.go. No production-code or proto changes; it only
// inspects authored artifacts.
//
// Helper and variable names are prefixed prop12 to stay unique within the package
// and avoid collisions with other concurrently-added test files.

import (
	"math"
	"math/rand"
	"path/filepath"
	"testing"
	"testing/quick"
)

// prop12JitterVariantPaths enumerates the six jitter variants, one per profile
// dir. Go tests run with the working directory set to the package dir
// (v1/generator/src), so the client configs live at ../../client/ relative to the
// test, matching markov_analytics_profile_test.go.
//
// Locally scoped (prop12-prefixed) to avoid redeclaration collisions with other
// test files that define their own profile path slices.
var prop12JitterVariantPaths = []string{
	filepath.Join("..", "..", "client", "markov_profiles", "analytics", "individual_jitter.textproto"),
	filepath.Join("..", "..", "client", "markov_profiles", "batch_etl", "individual_jitter.textproto"),
	filepath.Join("..", "..", "client", "markov_profiles", "read_replica", "individual_jitter.textproto"),
	filepath.Join("..", "..", "client", "markov_profiles", "bursty_api", "individual_jitter.textproto"),
	filepath.Join("..", "..", "client", "markov_profiles", "noisy_client", "individual_jitter.textproto"),
	filepath.Join("..", "..", "client", "markov_profiles", "oltp_app", "individual_jitter.textproto"),
}

// TestProp12JitterVariantsCarryOnlyABound is Property 12: every jitter variant is
// a markov_individual whose only perturbation knob is a single finite,
// non-negative weight_jitter.bound, with no edge overrides and no variances.
//
// Feature: markov-profile-library, Property 12: Jitter variants carry only a
// bound
//
// Validates: Requirements 4.5
//
// The property space is the finite set of six jitter variants, so the test drives
// testing/quick to pick a random variant index each iteration and asserts the
// invariant on the chosen variant, running >= 100 iterations. loadProfile calls
// loadConfigs, which sets the package-global resolvedBaseGraph, so this test does
// not run in parallel with other harness callers.
func TestProp12JitterVariantsCarryOnlyABound(t *testing.T) {
	property := func(pick uint) bool {
		// Reduce in unsigned space first: casting a large uint straight to int
		// can go negative and index out of range.
		idx := int(pick % uint(len(prop12JitterVariantPaths)))
		return prop12AssertJitterOnly(t, prop12JitterVariantPaths[idx])
	}

	cfg := &quick.Config{
		MaxCount: 200, // >= 100 iterations across the six jitter variants.
		Rand:     rand.New(rand.NewSource(12)),
	}
	if err := quick.Check(property, cfg); err != nil {
		t.Fatalf("Property 12 failed: %v", err)
	}
}

// prop12AssertJitterOnly loads the jitter variant at path through the harness and
// asserts it carries only a single finite, non-negative weight_jitter bound with
// no overrides and no variances. It fails the test (and returns false) on any
// violation so testing/quick reports the offending variant.
func prop12AssertJitterOnly(t *testing.T, path string) bool {
	t.Helper()

	profile := loadProfile(t, path)

	// It must be a markov_individual variant.
	if profile.Kind != scenarioMarkovIndividual {
		t.Errorf("variant %q kind = %q, want %q", path, profile.Kind, scenarioMarkovIndividual)
		return false
	}

	individual := profile.Configs.GetMarkovIndividual()
	if individual == nil {
		t.Errorf("variant %q has nil markov_individual config, want set", path)
		return false
	}

	// weight_jitter must be set with a single finite, non-negative bound.
	if !individual.HasWeightJitter() {
		t.Errorf("variant %q has no weight_jitter, want a single bound", path)
		return false
	}
	bound := individual.GetWeightJitter().GetBound()
	if math.IsNaN(bound) || math.IsInf(bound, 0) {
		t.Errorf("variant %q weight_jitter.bound = %v, want finite", path, bound)
		return false
	}
	if bound < 0 {
		t.Errorf("variant %q weight_jitter.bound = %v, want non-negative", path, bound)
		return false
	}

	// individual_edge_overrides must be empty.
	if overrides := individual.GetIndividualEdgeOverrides(); len(overrides) != 0 {
		t.Errorf("variant %q carries %d individual_edge_overrides, want 0 (jitter-only)", path, len(overrides))
		return false
	}

	// edge_variances must be empty.
	if variances := individual.GetEdgeVariances(); len(variances) != 0 {
		t.Errorf("variant %q carries %d edge_variances, want 0 (jitter-only)", path, len(variances))
		return false
	}

	return true
}
