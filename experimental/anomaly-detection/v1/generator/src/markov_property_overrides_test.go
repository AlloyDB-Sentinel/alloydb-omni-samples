package main

// Property-based test for the markov-profile-library feature.
//
// Feature: markov-profile-library, Property 10: The two individual-override files pin non-identical edge sets
//
// Validates: Requirements 4.3
//
// Property 10 quantifies over the six authored profile directories
// {analytics, batch_etl, read_replica, bursty_api, noisy_client, oltp_app}. Each
// ships two sibling individual-override files -- individual_overrides.textproto
// and individual_overrides_b.textproto -- because one QueryTemplateConfigs holds
// exactly one markov_individual. Requirement 4.3 asks that the two individuals be
// genuinely distinct, so for every profile dir this property asserts:
//
//   - both files load as Kind == scenarioMarkovIndividual,
//   - each pins >= 1 edge via individual_edge_overrides, and
//   - the SET of (source_id, destination_id, weight) triples pinned by the two
//     files is NOT identical.
//
// The property ranges over a finite, enumerated set of six profile dirs, so the
// invariants are checked exhaustively on every iteration. To honor the package's
// >= 100-iteration property-test convention, the driver runs the full
// six-profile enumeration at least 100 times; each iteration also drives a
// randomized cross-check that rebuilds the two triple sets in a randomized order
// and reconfirms they differ, so the set-inequality result is exercised
// independently of insertion order.
//
// Helper names here are prefixed `prop10` to stay unique within the package and
// avoid collisions with other concurrently-authored test files. Shared helpers
// (loadProfile, profileResult, scenarioMarkovIndividual) are reused and NOT
// redefined.
//
// No t.Parallel(): loadProfile calls loadConfigs, which sets the package-global
// resolvedBaseGraph, so harness callers must run sequentially.

import (
	"fmt"
	"math/rand"
	"path/filepath"
	"sort"
	"testing"
)

// prop10ProfileDirs enumerates the six profile directories under
// ../../client/markov_profiles/. Go tests run with the working directory set to
// the package dir (v1/generator/src), so the profiles live at
// ../../client/markov_profiles/<name>/, matching the path convention in
// markov_analytics_profile_test.go.
var prop10ProfileDirs = []string{
	"analytics",
	"batch_etl",
	"read_replica",
	"bursty_api",
	"noisy_client",
	"oltp_app",
}

// prop10OverrideTriple is one pinned edge: a (source_id, destination_id, weight)
// entry from a file's individual_edge_overrides.
type prop10OverrideTriple struct {
	SourceID      string
	DestinationID string
	Weight        float64
}

// prop10OverridePath builds the path to one override file for a profile dir.
func prop10OverridePath(dir, file string) string {
	return filepath.Join("..", "..", "client", "markov_profiles", dir, file)
}

// prop10PinnedTriples returns the individual_edge_overrides of a loaded
// markov_individual profile as (source, dest, weight) triples, in the order
// authored. The profile must be a markov_individual (asserted by the caller).
func prop10PinnedTriples(profile *profileResult) []prop10OverrideTriple {
	cfg := profile.Configs.GetMarkovIndividual()
	if cfg == nil {
		return nil
	}
	overrides := cfg.GetIndividualEdgeOverrides()
	triples := make([]prop10OverrideTriple, 0, len(overrides))
	for _, ov := range overrides {
		triples = append(triples, prop10OverrideTriple{
			SourceID:      ov.GetSourceId(),
			DestinationID: ov.GetDestinationId(),
			Weight:        ov.GetWeight(),
		})
	}
	return triples
}

// prop10SortedKeys renders a set of triples into a sorted, canonical slice of
// string keys, so two sets compare equal iff they contain the same triples
// regardless of authoring order. Using the sorted-key slice (rather than a map)
// makes the comparison deterministic and easy to report.
func prop10SortedKeys(triples []prop10OverrideTriple) []string {
	keys := make([]string, 0, len(triples))
	for _, tr := range triples {
		// %v on the float gives a stable, lossless textual form for comparison.
		keys = append(keys, fmt.Sprintf("%s|%s|%v", tr.SourceID, tr.DestinationID, tr.Weight))
	}
	sort.Strings(keys)
	return keys
}

// prop10SetsEqual reports whether two triple sets are identical: same number of
// distinct triples and the same sorted canonical keys.
func prop10SetsEqual(a, b []prop10OverrideTriple) bool {
	ka := prop10SortedKeys(a)
	kb := prop10SortedKeys(b)
	if len(ka) != len(kb) {
		return false
	}
	for i := range ka {
		if ka[i] != kb[i] {
			return false
		}
	}
	return true
}

// prop10CheckDir asserts the full set of invariants for one profile dir: both
// override files load as markov_individual, each pins >= 1 edge, and the two
// pinned (source, dest, weight) sets are not identical. Deterministic and
// independent of iteration count.
func prop10CheckDir(t *testing.T, dir string) {
	t.Helper()

	pathA := prop10OverridePath(dir, "individual_overrides.textproto")
	pathB := prop10OverridePath(dir, "individual_overrides_b.textproto")

	profileA := loadProfile(t, pathA)
	profileB := loadProfile(t, pathB)

	if profileA.Kind != scenarioMarkovIndividual {
		t.Fatalf("%s: kind = %q, want %q", pathA, profileA.Kind, scenarioMarkovIndividual)
	}
	if profileB.Kind != scenarioMarkovIndividual {
		t.Fatalf("%s: kind = %q, want %q", pathB, profileB.Kind, scenarioMarkovIndividual)
	}

	triplesA := prop10PinnedTriples(profileA)
	triplesB := prop10PinnedTriples(profileB)

	if len(triplesA) < 1 {
		t.Fatalf("%s: pins %d edges via individual_edge_overrides, want >= 1", pathA, len(triplesA))
	}
	if len(triplesB) < 1 {
		t.Fatalf("%s: pins %d edges via individual_edge_overrides, want >= 1", pathB, len(triplesB))
	}

	if prop10SetsEqual(triplesA, triplesB) {
		t.Fatalf("%s: the two individual-override files pin an identical edge set %v, want non-identical",
			dir, prop10SortedKeys(triplesA))
	}
}

// TestProperty10_IndividualOverrideFilesPinNonIdenticalEdgeSets is the Property 10
// driver.
//
// Feature: markov-profile-library, Property 10: The two individual-override files pin non-identical edge sets
//
// Validates: Requirements 4.3
func TestProperty10_IndividualOverrideFilesPinNonIdenticalEdgeSets(t *testing.T) {
	// Load each dir's two override files once, up front. loadProfile fails the
	// test on any load/validation error. Cache the pinned triples so the >= 100
	// iterations below do not re-walk the proto each pass.
	type dirTriples struct {
		dir string
		a   []prop10OverrideTriple
		b   []prop10OverrideTriple
	}
	cached := make([]dirTriples, 0, len(prop10ProfileDirs))

	for _, dir := range prop10ProfileDirs {
		prop10CheckDir(t, dir)

		profileA := loadProfile(t, prop10OverridePath(dir, "individual_overrides.textproto"))
		profileB := loadProfile(t, prop10OverridePath(dir, "individual_overrides_b.textproto"))
		cached = append(cached, dirTriples{
			dir: dir,
			a:   prop10PinnedTriples(profileA),
			b:   prop10PinnedTriples(profileB),
		})
	}

	// Property driver: run the full enumeration >= 100 times. The invariants are
	// deterministic over the finite set, so each iteration checks them
	// exhaustively; on top of that, each iteration drives one randomized
	// cross-check -- it shuffles both triple slices for a random dir and
	// reconfirms the two sets are still not identical -- so set inequality is
	// exercised independently of authoring/insertion order.
	const iterations = 200 // >= 100 per the property-test convention
	rng := rand.New(rand.NewSource(1))

	for i := 0; i < iterations; i++ {
		for _, dt := range cached {
			if prop10SetsEqual(dt.a, dt.b) {
				t.Fatalf("iteration %d: %s pins an identical edge set across its two override files", i, dt.dir)
			}
		}

		// Randomized cross-check over a random dir: shuffle both slices and
		// reconfirm inequality. Set comparison must not depend on order.
		dt := cached[rng.Intn(len(cached))]
		shuffledA := append([]prop10OverrideTriple(nil), dt.a...)
		shuffledB := append([]prop10OverrideTriple(nil), dt.b...)
		rng.Shuffle(len(shuffledA), func(x, y int) { shuffledA[x], shuffledA[y] = shuffledA[y], shuffledA[x] })
		rng.Shuffle(len(shuffledB), func(x, y int) { shuffledB[x], shuffledB[y] = shuffledB[y], shuffledB[x] })
		if prop10SetsEqual(shuffledA, shuffledB) {
			t.Fatalf("iteration %d: randomized cross-check found %s override sets identical under reordering",
				i, dt.dir)
		}
	}
}
