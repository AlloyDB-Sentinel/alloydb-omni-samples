package main

import (
	"context"
	"math/rand"
	"path/filepath"
	"testing"

	genpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v1/generator/proto"
)

// TestShippedMarkovExamples loads each shipped example subtype/individual
// textproto under v1/client/markov_profiles/ through loadConfigs with its real
// on-disk path, so the relative base_graph_path in each variant resolves against
// the actual client/markov_*_config.textproto base graphs exactly as it would in
// production. loadConfigs resolves-and-inlines the base graph (setting the
// package-global resolvedBaseGraph) and runs validateConfigs, so a nil error
// confirms R13/R16 acceptance and relative-path resolution. Each example is then
// derived (deriveSubtypeGraph for markov_subtype, deriveIndividualGraph for
// markov_individual, consuming resolvedBaseGraph just like the GenerateLogs
// dispatch does) and the derived graph is re-checked with validateMarkovSampling,
// confirming it satisfies every R4 graph-validation invariant.
//
// Validates R17.6 (acceptance under R13/R16 + derived-graph R4 validity +
// relative base_graph_path resolution).
func TestShippedMarkovExamples(t *testing.T) {
	// Go tests run with the working directory set to the package dir
	// (v1/generator/src), so the shipped examples live at
	// ../../client/markov_profiles/... relative to the test. Each variant's own
	// base_graph_path ("../../markov_*_config.textproto") is resolved by
	// loadConfigs relative to the variant file's real directory.
	profiles := filepath.Join("..", "..", "client", "markov_profiles")

	// The shipped examples authored by tasks 16.1/16.2. Hardcoded (not globbed)
	// so a missing example surfaces as a load failure rather than silently
	// shrinking the table; all seven known files are exercised.
	examples := []string{
		filepath.Join("service", "peak_subtype.textproto"),
		filepath.Join("standard", "subtype.textproto"),
		filepath.Join("maintenance", "subtype.textproto"),
		filepath.Join("standard", "individual_overrides.textproto"),
		filepath.Join("standard", "individual_overrides_b.textproto"),
		filepath.Join("standard", "individual_variance.textproto"),
		filepath.Join("standard", "individual_jitter.textproto"),
	}

	for _, rel := range examples {
		rel := rel
		// No t.Parallel(): loadConfigs sets the package-global resolvedBaseGraph,
		// which the derivation step reads, so cases must run sequentially.
		t.Run(rel, func(t *testing.T) {
			path := filepath.Join(profiles, rel)
			ctx := context.Background()

			// loadConfigs resolves base_graph_path, sets resolvedBaseGraph, and
			// runs validateConfigs; a nil error is R17.6 acceptance.
			cfg, err := loadConfigs(ctx, path)
			if err != nil {
				t.Fatalf("loadConfigs(%q) failed: %v", path, err)
			}

			// Derive the plain MarkovSamplingConfig the walker would consume,
			// feeding the resolved base exactly as the GenerateLogs dispatch does.
			var derived *genpb.MarkovSamplingConfig
			switch {
			case cfg.GetMarkovSubtype() != nil:
				derived, err = deriveSubtypeGraph(cfg.GetMarkovSubtype(), resolvedBaseGraph)
			case cfg.GetMarkovIndividual() != nil:
				derived, err = deriveIndividualGraph(cfg.GetMarkovIndividual(), resolvedBaseGraph, rand.New(rand.NewSource(1)))
			default:
				t.Fatalf("example %q is neither a markov_subtype nor a markov_individual variant", path)
			}
			if err != nil {
				t.Fatalf("deriving %q failed: %v", path, err)
			}

			// The derived graph must satisfy every R4 graph-validation invariant.
			if err := validateMarkovSampling(derived); err != nil {
				t.Fatalf("derived graph from %q failed validateMarkovSampling: %v", path, err)
			}
		})
	}
}
