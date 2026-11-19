package main

// Property-based test for the markov-profile-library feature.
//
// Feature: markov-profile-library, Property 9: Subtype variants reference their own base and override real edges
//
// Validates: Requirements 4.2, 5.2
//
// Property 9 quantifies over the six authored subtype variants
// (markov_profiles/<name>/subtype.textproto). For every subtype variant:
//
//   - loadProfile reports Kind == scenarioMarkovSubtype,
//   - MarkovSubtypeConfig.base_graph_path resolves to the variant's OWN base
//     profile: the path points at "../../markov_<name>_config.textproto", so its
//     filename component must equal the expected base file for that directory,
//   - there is at least one edge_weight_override,
//   - every override's (source_id, destination_id) names a REAL edge present in
//     the resolved base graph, and its replacement weight is finite and
//     non-negative.
//
// The property ranges over a finite, enumerated set of six subtype variants, so
// the invariants are checked exhaustively on every iteration. To honor the
// package's >= 100-iteration property-test convention, a testing/quick-style
// driver runs the full six-variant enumeration at least 100 times; on top of the
// deterministic checks, each iteration picks a random override on a random variant
// and reconfirms, via an independently built base edge set, that it names a real
// base edge with a finite, non-negative weight.
//
// Helper names here are prefixed `prop9` to stay unique within the package and
// avoid collisions with other concurrently-authored test files. Shared harness
// helpers (loadProfile, profileResult, profileEdge) are reused and NOT redefined.
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

// prop9SubtypeVariant pairs a subtype variant file with the base config filename
// it must reference. The base_graph_path in each subtype.textproto is
// "../../markov_<name>_config.textproto", so only the filename component is
// checked here (the "../../" prefix and resolution are exercised by loadProfile
// succeeding, which fails the test on any unresolved reference).
type prop9SubtypeVariant struct {
	// Path is the subtype variant file, under ../../client/markov_profiles/<dir>/.
	Path string
	// ExpectedBaseFile is the basename the variant's base_graph_path must name.
	ExpectedBaseFile string
}

// prop9SubtypeVariants enumerates the six subtype variants and the OWN base file
// each must reference. Go tests run with the working directory set to the package
// dir (v1/generator/src), so client configs live at ../../client/, matching the
// path convention in markov_analytics_profile_test.go.
var prop9SubtypeVariants = func() []prop9SubtypeVariant {
	dirToBase := []struct {
		dir      string
		baseFile string
	}{
		{"analytics", "markov_analytics_user_config.textproto"},
		{"batch_etl", "markov_batch_etl_config.textproto"},
		{"read_replica", "markov_read_replica_config.textproto"},
		{"bursty_api", "markov_bursty_api_config.textproto"},
		{"noisy_client", "markov_noisy_client_config.textproto"},
		{"oltp_app", "markov_oltp_app_config.textproto"},
	}
	variants := make([]prop9SubtypeVariant, 0, len(dirToBase))
	for _, d := range dirToBase {
		variants = append(variants, prop9SubtypeVariant{
			Path:             filepath.Join("..", "..", "client", "markov_profiles", d.dir, "subtype.textproto"),
			ExpectedBaseFile: d.baseFile,
		})
	}
	return variants
}()

// prop9EdgeKey identifies a directed edge by (source, destination).
type prop9EdgeKey struct {
	Source      string
	Destination string
}

// prop9BaseEdgeSet builds the set of real (source, destination) edges from a
// resolved base graph's edge list.
func prop9BaseEdgeSet(edges []profileEdge) map[prop9EdgeKey]bool {
	set := make(map[prop9EdgeKey]bool, len(edges))
	for _, e := range edges {
		set[prop9EdgeKey{Source: e.SourceID, Destination: e.DestinationID}] = true
	}
	return set
}

// prop9CheckSubtype asserts the full set of Property 9 invariants for one loaded
// subtype variant. It is deterministic and independent of iteration count.
func prop9CheckSubtype(t *testing.T, variant prop9SubtypeVariant, profile *profileResult) {
	t.Helper()

	// --- Kind == scenarioMarkovSubtype ---
	if profile.Kind != scenarioMarkovSubtype {
		t.Fatalf("%s: subtype variant kind = %q, want %q", variant.Path, profile.Kind, scenarioMarkovSubtype)
	}

	subtype := profile.Configs.GetMarkovSubtype()
	if subtype == nil {
		t.Fatalf("%s: GetMarkovSubtype() is nil for a markov_subtype variant", variant.Path)
	}

	// --- base_graph_path resolves to the variant's OWN base ---
	// The authored path is "../../markov_<name>_config.textproto"; verify the
	// filename component matches the expected base for this directory. loadProfile
	// already resolved and inlined the base (ResolvedBase), so the reference is
	// known to resolve; this pins it to the correct sibling base.
	gotBaseFile := filepath.Base(subtype.GetBaseGraphPath())
	if gotBaseFile != variant.ExpectedBaseFile {
		t.Fatalf("%s: base_graph_path = %q (basename %q), want basename %q (its own base)",
			variant.Path, subtype.GetBaseGraphPath(), gotBaseFile, variant.ExpectedBaseFile)
	}

	// --- at least one edge_weight_override ---
	overrides := subtype.GetEdgeWeightOverrides()
	if len(overrides) < 1 {
		t.Fatalf("%s: edge_weight_overrides is empty, want >= 1", variant.Path)
	}

	// --- resolved base graph is available to validate overrides against ---
	if profile.ResolvedBase == nil {
		t.Fatalf("%s: resolved base graph is nil; cannot validate overrides", variant.Path)
	}
	baseEdges := prop9BaseEdgeSet(profile.Edges)
	if len(baseEdges) == 0 {
		t.Fatalf("%s: resolved base graph has no edges", variant.Path)
	}

	// --- every override names a REAL base edge with a finite, non-negative weight ---
	for i, ov := range overrides {
		key := prop9EdgeKey{Source: ov.GetSourceId(), Destination: ov.GetDestinationId()}
		if !baseEdges[key] {
			t.Fatalf("%s: edge_weight_overrides[%d] (%s -> %s) is not a real edge in the base graph",
				variant.Path, i, ov.GetSourceId(), ov.GetDestinationId())
		}
		w := ov.GetWeight()
		if math.IsNaN(w) || math.IsInf(w, 0) {
			t.Fatalf("%s: edge_weight_overrides[%d] (%s -> %s) has non-finite weight %v",
				variant.Path, i, ov.GetSourceId(), ov.GetDestinationId(), w)
		}
		if w < 0 {
			t.Fatalf("%s: edge_weight_overrides[%d] (%s -> %s) has negative weight %v",
				variant.Path, i, ov.GetSourceId(), ov.GetDestinationId(), w)
		}
	}
}

// TestProperty9_SubtypeVariantsReferenceOwnBaseAndOverrideRealEdges is the
// Property 9 driver.
//
// Feature: markov-profile-library, Property 9: Subtype variants reference their own base and override real edges
//
// Validates: Requirements 4.2, 5.2
func TestProperty9_SubtypeVariantsReferenceOwnBaseAndOverrideRealEdges(t *testing.T) {
	// Load each subtype variant once. loadProfile fails the test on any load or
	// reference-resolution error and stashes the resolved base in profile.ResolvedBase.
	type loaded struct {
		variant   prop9SubtypeVariant
		profile   *profileResult
		baseEdges map[prop9EdgeKey]bool
		overrides []*genpb.EdgeWeightOverride
	}
	loadedVariants := make([]loaded, 0, len(prop9SubtypeVariants))
	for _, variant := range prop9SubtypeVariants {
		profile := loadProfile(t, variant.Path)
		prop9CheckSubtype(t, variant, profile)
		loadedVariants = append(loadedVariants, loaded{
			variant:   variant,
			profile:   profile,
			baseEdges: prop9BaseEdgeSet(profile.Edges),
			overrides: profile.Configs.GetMarkovSubtype().GetEdgeWeightOverrides(),
		})
	}

	// Property driver: run the full enumeration >= 100 times. The invariants are
	// deterministic over the finite set, so each iteration re-checks them
	// exhaustively; on top of that, each iteration picks a random override on a
	// random variant and reconfirms it names a real base edge with a finite,
	// non-negative weight, exercising the override-vs-base relationship across many
	// randomized samples.
	const iterations = 200 // >= 100 per the property-test convention
	rng := rand.New(rand.NewSource(1))

	for i := 0; i < iterations; i++ {
		for _, lv := range loadedVariants {
			prop9CheckSubtype(t, lv.variant, lv.profile)
		}

		// Randomized probe: random variant, random override.
		lv := loadedVariants[rng.Intn(len(loadedVariants))]
		ov := lv.overrides[rng.Intn(len(lv.overrides))]
		key := prop9EdgeKey{Source: ov.GetSourceId(), Destination: ov.GetDestinationId()}
		if !lv.baseEdges[key] {
			t.Fatalf("iteration %d: randomized probe found override (%s -> %s) in %s not present in its base graph",
				i, ov.GetSourceId(), ov.GetDestinationId(), lv.variant.Path)
		}
		w := ov.GetWeight()
		if math.IsNaN(w) || math.IsInf(w, 0) || w < 0 {
			t.Fatalf("iteration %d: randomized probe found override (%s -> %s) in %s with bad weight %v",
				i, ov.GetSourceId(), ov.GetDestinationId(), lv.variant.Path, w)
		}
	}
}
