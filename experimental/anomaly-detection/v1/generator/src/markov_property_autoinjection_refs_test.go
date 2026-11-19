package main

// Property-based test for the markov-profile-library feature.
//
// Feature: markov-profile-library, Property 14: Auto-injection references resolve to non-self sibling plain graphs
//
// Validates: Requirements 4.7, 4.8, 5.3, 5.6
//
// Property 14 quantifies over the anomaly_refs of the six authored auto-injection
// variants, one per base profile, each at
// ../../client/markov_profiles/<name>/auto_injection.textproto. For every variant
// the property asserts:
//
//  1. the auto_injection block declares at least one anomaly_ref (R4.6/4.7).
//  2. each ref's anomaly_graph_path, resolved relative to the variant file's own
//     directory, lands on a file that exists, is one of the six new base profiles,
//     and is NOT the variant's own base — so no reference is self-referential
//     (R4.7, R4.8). The refs are authored as sibling relative paths of the form
//     "../../markov_<sibling>_config.textproto".
//  3. each resolved target is a plain markov_sampling graph that declares neither
//     injections nor auto_injection, i.e. a non-chaining base graph (R5.3, R5.6).
//     This is checked by loading the resolved target through the harness and
//     asserting Kind == scenarioMarkovSampling and HasInjection == false.
//
// The property ranges over a finite, enumerated set (six variants, each with one
// or more refs), so the invariants are checked exhaustively on every iteration. To
// honor the package's >= 100-iteration property-test convention the full
// enumeration runs at least 100 times.
//
// Helper names here are prefixed `prop14` to stay unique within the package and
// avoid collisions with other prop* helpers.
//
// No t.Parallel(): loadProfile calls loadConfigs, which sets the package-global
// resolvedBaseGraph, so harness callers must run sequentially.

import (
	"os"
	"path/filepath"
	"testing"
)

// prop14ClientDir is the directory holding the six new base profiles, relative to
// the package dir (v1/generator/src) that Go tests run from.
var prop14ClientDir = filepath.Join("..", "..", "client")

// prop14Variant pairs an auto-injection variant file with the basename of the base
// profile it is a variant of. The variant's anomaly_refs must point at sibling base
// profiles only, never this own base (R4.7, R4.8).
type prop14Variant struct {
	// path is the variant's auto_injection.textproto, under
	// ../../client/markov_profiles/<name>/.
	path string
	// ownBaseName is the basename of the variant's own base profile; a resolved
	// anomaly_ref equal to this would be self-referential and is forbidden.
	ownBaseName string
}

// prop14Variants enumerates the six auto-injection variants and each one's own
// base-profile basename, from the dir->own-base map in the design cross-reference
// section. Client configs live at ../../client/, matching the path convention in
// markov_analytics_profile_test.go and markov_property_autoinjection_bounds_test.go.
var prop14Variants = []prop14Variant{
	{
		path:        filepath.Join(prop14ClientDir, "markov_profiles", "analytics", "auto_injection.textproto"),
		ownBaseName: "markov_analytics_user_config.textproto",
	},
	{
		path:        filepath.Join(prop14ClientDir, "markov_profiles", "batch_etl", "auto_injection.textproto"),
		ownBaseName: "markov_batch_etl_config.textproto",
	},
	{
		path:        filepath.Join(prop14ClientDir, "markov_profiles", "read_replica", "auto_injection.textproto"),
		ownBaseName: "markov_read_replica_config.textproto",
	},
	{
		path:        filepath.Join(prop14ClientDir, "markov_profiles", "bursty_api", "auto_injection.textproto"),
		ownBaseName: "markov_bursty_api_config.textproto",
	},
	{
		path:        filepath.Join(prop14ClientDir, "markov_profiles", "noisy_client", "auto_injection.textproto"),
		ownBaseName: "markov_noisy_client_config.textproto",
	},
	{
		path:        filepath.Join(prop14ClientDir, "markov_profiles", "oltp_app", "auto_injection.textproto"),
		ownBaseName: "markov_oltp_app_config.textproto",
	},
}

// prop14NewBaseProfileNames is the set of the six new base-profile basenames. Every
// resolved anomaly_ref must land on one of these (a sibling new base profile).
var prop14NewBaseProfileNames = map[string]bool{
	"markov_analytics_user_config.textproto": true,
	"markov_batch_etl_config.textproto":      true,
	"markov_read_replica_config.textproto":   true,
	"markov_bursty_api_config.textproto":     true,
	"markov_noisy_client_config.textproto":   true,
	"markov_oltp_app_config.textproto":       true,
}

// prop14CheckVariantRefs asserts every Property 14 invariant for a single
// auto-injection variant: at least one anomaly_ref; each ref resolves to a sibling
// new base profile (exists, is one of the six, is not the variant's own base); and
// each resolved target is a plain markov_sampling graph with no injections/
// auto_injection. It is deterministic and independent of iteration count.
func prop14CheckVariantRefs(t *testing.T, v prop14Variant) {
	t.Helper()

	profile := loadProfile(t, v.path)
	if profile.Kind != scenarioMarkovSampling {
		t.Fatalf("%s: variant kind = %q, want %q", v.path, profile.Kind, scenarioMarkovSampling)
	}

	auto := profile.Configs.GetMarkovSampling().GetAutoInjection()
	if auto == nil {
		t.Fatalf("%s: variant carries no auto_injection block, want one", v.path)
	}

	refs := auto.GetAnomalyRefs()
	if len(refs) == 0 {
		t.Fatalf("%s: auto_injection declares no anomaly_refs, want >= 1", v.path)
	}

	// The variant file's directory is the base for resolving relative ref paths.
	variantDir := filepath.Dir(v.path)

	for _, ref := range refs {
		refPath := ref.GetAnomalyGraphPath()
		if refPath == "" {
			t.Fatalf("%s: anomaly_ref has an empty anomaly_graph_path", v.path)
		}

		// Resolve the ref relative to the variant file's directory.
		resolved := filepath.Join(variantDir, refPath)
		resolvedName := filepath.Base(resolved)

		// The resolved target must exist on disk.
		if _, err := os.Stat(resolved); err != nil {
			t.Fatalf("%s: anomaly_ref %q resolves to %q, which does not exist: %v",
				v.path, refPath, resolved, err)
		}

		// It must be one of the six new base profiles.
		if !prop14NewBaseProfileNames[resolvedName] {
			t.Fatalf("%s: anomaly_ref %q resolves to %q, which is not one of the six new base profiles",
				v.path, refPath, resolvedName)
		}

		// It must NOT be the variant's own base — no self-reference (R4.7, R4.8).
		if resolvedName == v.ownBaseName {
			t.Fatalf("%s: anomaly_ref %q resolves to the variant's own base %q, want a sibling (self-reference forbidden)",
				v.path, refPath, v.ownBaseName)
		}

		// The resolved target must be a plain markov_sampling graph with no
		// injections and no auto_injection — proving no chaining (R5.3, R5.6).
		target := loadProfile(t, resolved)
		if target.Kind != scenarioMarkovSampling {
			t.Fatalf("%s: anomaly_ref %q target %q kind = %q, want %q (must be a plain base graph)",
				v.path, refPath, resolvedName, target.Kind, scenarioMarkovSampling)
		}
		if target.HasInjection {
			t.Fatalf("%s: anomaly_ref %q target %q carries injections/auto_injection, want none (no chaining)",
				v.path, refPath, resolvedName)
		}
	}
}

// TestProperty14_AutoInjectionRefsResolveToNonSelfSiblingPlainGraphs is the
// Property 14 driver.
//
// Feature: markov-profile-library, Property 14: Auto-injection references resolve to non-self sibling plain graphs
//
// Validates: Requirements 4.7, 4.8, 5.3, 5.6
func TestProperty14_AutoInjectionRefsResolveToNonSelfSiblingPlainGraphs(t *testing.T) {
	// Run the full six-variant enumeration >= 100 times. The invariants are
	// deterministic over the finite set of variants and their refs, so each
	// iteration re-checks them exhaustively; running the enumeration repeatedly
	// honors the package's >= 100-iteration property-test convention.
	const iterations = 200 // >= 100 per the property-test convention

	for i := 0; i < iterations; i++ {
		for _, v := range prop14Variants {
			prop14CheckVariantRefs(t, v)
		}
	}
}
