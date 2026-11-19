package main

import (
	"path/filepath"
	"testing"
)

// analyticsConfigPath is the analytics/reporting base profile authored by task
// 1.2. Go tests run with the working directory set to the package dir
// (v1/generator/src), so the client config lives at ../../client/ relative to the
// test, the same way markov_examples_test.go reaches the shipped client configs.
var analyticsConfigPath = filepath.Join("..", "..", "client", "markov_analytics_user_config.textproto")

// TestAnalyticsBaseProfile_LoadsAndSamplesReadOnly verifies the analytics base
// profile through the markov-profile-library harness: it loads and passes
// validation as a plain markov_sampling graph with no injection, and a large
// sample contains zero writes while honoring the documented behavioral intent
// (every completion >= 30 s, so the observed action rate stays <= 2 ops/min).
//
// Validates: Requirements 5.1, 6.1, 6.3, 3.5
//
// No t.Parallel(): loadProfile/sampleProfile call loadConfigs, which sets the
// package-global resolvedBaseGraph, so harness callers must run sequentially.
func TestAnalyticsBaseProfile_LoadsAndSamplesReadOnly(t *testing.T) {
	// Load + validate. loadProfile fails the test on any load/validation error
	// and, for a plain markov_sampling base graph, re-checks with
	// validateMarkovSampling.
	profile := loadProfile(t, analyticsConfigPath)

	// It must be a plain markov_sampling base profile with no injection.
	if profile.Kind != scenarioMarkovSampling {
		t.Fatalf("analytics profile kind = %q, want %q", profile.Kind, scenarioMarkovSampling)
	}
	if profile.HasInjection {
		t.Fatalf("analytics base profile carries an injection/auto_injection, want none")
	}

	// A >= 1,000-action sample. GenerateLogs emits exactly numLogs actions.
	const numLogs = 1000
	const seed = int64(1)
	sample := sampleProfile(t, analyticsConfigPath, numLogs, seed)

	if got := len(sample.Actions); got != numLogs {
		t.Fatalf("sample action count = %d, want %d", got, numLogs)
	}

	// Zero writes: a read-only profile must emit no write actions (R6.3, R3.5).
	if sample.WriteCount != 0 {
		t.Fatalf("sample contains %d write actions, want 0 (profile is 100%% read)", sample.WriteCount)
	}
	if wf := sample.WriteFraction(); wf != 0 {
		t.Fatalf("sample write fraction = %v, want 0", wf)
	}

	// Observed action rate <= 2 ops/min. The generator advances a single
	// simulated clock by each node's completion time, so ElapsedMs spans the
	// N-1 completions between the first and last action. The observed rate is
	// those intervals over the elapsed simulated minutes.
	elapsedMs := sample.ElapsedMs()
	if elapsedMs <= 0 {
		t.Fatalf("sample elapsed simulated time = %d ms, want > 0 to measure rate", elapsedMs)
	}
	elapsedMinutes := float64(elapsedMs) / 60000.0
	intervals := float64(len(sample.Actions) - 1)
	rate := intervals / elapsedMinutes

	// Every completion range starts at >= 30 s, so the ceiling is 2 ops/min.
	// Allow a tiny tolerance for floating-point division only.
	const maxRate = 2.0
	const tolerance = 1e-6
	if rate > maxRate+tolerance {
		t.Fatalf("observed action rate = %.6f ops/min, want <= %.1f (every completion must be >= 30 s)", rate, maxRate)
	}
}
