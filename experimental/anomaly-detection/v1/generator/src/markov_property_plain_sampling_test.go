package main

// Property test for the markov-profile-library feature.
//
// Feature: markov-profile-library, Property 2: Base profiles are plain
// markov_sampling with no injections
//
// Validates: Requirements 1.7, 2.4
//
// This file adds a single property-based test that, across many randomized
// iterations, picks one of the six new base profiles and asserts it is a plain
// markov_sampling scenario carrying no injections and no auto_injection. It
// reuses the markov-profile-library harness (loadProfile / profileResult) and the
// ../../client/ path convention established by markov_analytics_profile_test.go.
//
// No production-code or proto changes; it only inspects authored artifacts.
//
// Helper and variable names are prefixed prop2 to stay unique within the package
// and avoid collisions with other concurrently-added test files.

import (
	"math/rand"
	"path/filepath"
	"testing"
	"testing/quick"
)

// prop2BaseProfilePaths enumerates the six new base profiles authored under
// v1/client/. Go tests run with the working directory set to the package dir
// (v1/generator/src), so the client configs live at ../../client/ relative to the
// test, matching markov_analytics_profile_test.go.
//
// Locally scoped (prop2-prefixed) to avoid redeclaration collisions with other
// test files that may define their own base-profile path slices.
var prop2BaseProfilePaths = []string{
	filepath.Join("..", "..", "client", "markov_analytics_user_config.textproto"),
	filepath.Join("..", "..", "client", "markov_batch_etl_config.textproto"),
	filepath.Join("..", "..", "client", "markov_read_replica_config.textproto"),
	filepath.Join("..", "..", "client", "markov_bursty_api_config.textproto"),
	filepath.Join("..", "..", "client", "markov_noisy_client_config.textproto"),
	filepath.Join("..", "..", "client", "markov_oltp_app_config.textproto"),
}

// TestProp2BaseProfilesArePlainMarkovSampling is Property 2: every base profile is
// a plain markov_sampling scenario with no injections and no auto_injection.
//
// Feature: markov-profile-library, Property 2: Base profiles are plain
// markov_sampling with no injections
//
// Validates: Requirements 1.7, 2.4
//
// The property space is the finite set of six base profiles, so the test drives
// testing/quick to pick a random profile index each iteration and asserts the
// invariant on the chosen profile, running >= 100 iterations. loadProfile calls
// loadConfigs, which sets the package-global resolvedBaseGraph, so this test does
// not run in parallel with other harness callers.
func TestProp2BaseProfilesArePlainMarkovSampling(t *testing.T) {
	property := func(pick uint) bool {
		// Reduce in unsigned space first: casting a large uint straight to int
		// can go negative and index out of range.
		idx := int(pick % uint(len(prop2BaseProfilePaths)))
		return prop2AssertPlainSampling(t, prop2BaseProfilePaths[idx])
	}

	cfg := &quick.Config{
		MaxCount: 200, // >= 100 iterations across the six base profiles.
		Rand:     rand.New(rand.NewSource(2)),
	}
	if err := quick.Check(property, cfg); err != nil {
		t.Fatalf("Property 2 failed: %v", err)
	}
}

// prop2AssertPlainSampling loads the profile at path through the harness and
// asserts it is a plain markov_sampling base profile with no injections. It
// fails the test (and returns false) on any violation so testing/quick reports
// the offending profile.
func prop2AssertPlainSampling(t *testing.T, path string) bool {
	t.Helper()

	profile := loadProfile(t, path)

	// The parsed QueryTemplateConfigs must set the markov_sampling scenario.
	if profile.Kind != scenarioMarkovSampling {
		t.Errorf("profile %q kind = %q, want %q", path, profile.Kind, scenarioMarkovSampling)
		return false
	}

	// It must set NEITHER injections NOR auto_injection.
	if profile.HasInjection {
		t.Errorf("profile %q reports HasInjection = true, want false (plain base profile)", path)
		return false
	}

	markov := profile.Configs.GetMarkovSampling()
	if markov == nil {
		t.Errorf("profile %q has nil markov_sampling config, want set", path)
		return false
	}
	if injections := markov.GetInjections(); len(injections) != 0 {
		t.Errorf("profile %q carries %d injections, want 0", path, len(injections))
		return false
	}
	if auto := markov.GetAutoInjection(); auto != nil {
		t.Errorf("profile %q carries an auto_injection block, want nil", path)
		return false
	}

	return true
}
