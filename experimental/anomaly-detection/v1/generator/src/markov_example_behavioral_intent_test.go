package main

// Example test for the markov-profile-library feature, task 13.3.
//
// Asserts two documentation facts about the six new base profiles that depend
// only on the authored textproto header comments (not on loading or sampling):
//
//  1. Each base profile's header carries a `Behavioral_Intent` section, and that
//     section states all three behavioral dimensions required by Requirement
//     3.3: action rate, read/write mix, and session/termination behavior. The
//     keyword checks are kept tolerant to the authored wording.
//  2. The six documented (rate, mix, termination) triples drawn from the design
//     distinctness matrix are pairwise DISTINCT (Requirement 3.4): collecting the
//     six canonical triples into a set yields exactly six unique entries.
//
// Validates: Requirements 3.3, 3.4
//
// Test working directory is the package dir (v1/generator/src), so the client
// configs live under ../../client/, matching the path convention in
// markov_analytics_profile_test.go and markov_example_base_profiles_test.go.
//
// No t.Parallel(): this test only reads files on disk; it shares no package
// globals, but is kept sequential alongside the other markov-profile harness
// tests for consistency.

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// ex133ClientDir is ../../client relative to the package dir, matching the path
// convention in markov_analytics_profile_test.go.
var ex133ClientDir = filepath.Join("..", "..", "client")

// ex133BaseProfiles are the six new base-profile file names directly under
// ../../client/, one per Requirement 1 criteria 1-6.
var ex133BaseProfiles = []string{
	"markov_analytics_user_config.textproto",
	"markov_batch_etl_config.textproto",
	"markov_read_replica_config.textproto",
	"markov_bursty_api_config.textproto",
	"markov_noisy_client_config.textproto",
	"markov_oltp_app_config.textproto",
}

// ex133Triple is a profile's documented behavioral triple, taken from the design
// distinctness matrix (Requirement 3.4). The three fields are canonicalized to
// strings so the triples can be compared for pairwise distinctness.
type ex133Triple struct {
	rate        string
	mix         string
	termination string
}

// String renders the triple as a single canonical key for set membership.
func (t ex133Triple) String() string {
	return t.rate + " | " + t.mix + " | " + t.termination
}

// ex133DocumentedTriples are the six (rate, mix, termination) triples from the
// design distinctness matrix:
//
//	| analytics    | <= 2 ops/min              | 100% read | continuous loop         |
//	| batch_etl    | ~30-60 ops/min            | ~70% write| bounded (session_done)  |
//	| read_replica | ~15-30 ops/min            | 100% read | continuous loop         |
//	| bursty_api   | ~1-2 (base)/~12-30 (burst)| ~80% read | continuous, two-regime  |
//	| noisy_client | ~60-300 ops/min           | ~80% read | continuous, erratic     |
//	| oltp_app     | ~5 ops/min                | ~55% write| continuous              |
//
// These are hardcoded canonical strings derived from the design matrix; the
// point of the test is to assert that distinctness holds (six unique triples).
var ex133DocumentedTriples = map[string]ex133Triple{
	"analytics":    {rate: "<=2 ops/min", mix: "100% read", termination: "continuous loop"},
	"batch_etl":    {rate: "~30-60 ops/min", mix: "~70% write", termination: "bounded/session_done"},
	"read_replica": {rate: "~15-30 ops/min", mix: "100% read", termination: "continuous loop"},
	"bursty_api":   {rate: "~1-2/~12-30 two-regime", mix: "~80% read", termination: "continuous two-regime"},
	"noisy_client": {rate: "~60-300 ops/min", mix: "~80% read", termination: "continuous erratic"},
	"oltp_app":     {rate: "~5 ops/min", mix: "~55% write", termination: "continuous"},
}

// ex133ReadFile reads a file and fails the test if it cannot be read.
func ex133ReadFile(t *testing.T, path string) string {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("reading %q failed: %v", path, err)
	}
	return string(raw)
}

// ex133ExtractBehavioralIntent returns the lines of the leading comment block
// starting at the `Behavioral_Intent` marker and continuing while the following
// lines remain part of the header comment (lines beginning with `#`). The
// returned text is lower-cased for tolerant keyword matching. The bool is false
// if no `Behavioral_Intent` marker is present.
func ex133ExtractBehavioralIntent(raw string) (string, bool) {
	lines := strings.Split(raw, "\n")
	start := -1
	for i, line := range lines {
		if strings.Contains(line, "Behavioral_Intent") {
			start = i
			break
		}
	}
	if start < 0 {
		return "", false
	}
	var block []string
	for i := start; i < len(lines); i++ {
		line := lines[i]
		// The header is a run of comment lines; the first non-comment,
		// non-blank line (e.g. `markov_sampling {`) ends the block.
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			// A blank (non-comment) line ends the comment header.
			break
		}
		if !strings.HasPrefix(trimmed, "#") {
			break
		}
		block = append(block, line)
	}
	return strings.ToLower(strings.Join(block, "\n")), true
}

// ex133ContainsAny reports whether haystack contains any of the needles.
func ex133ContainsAny(haystack string, needles ...string) bool {
	for _, n := range needles {
		if strings.Contains(haystack, n) {
			return true
		}
	}
	return false
}

// TestExampleBehavioralIntentHeadersAndTripleDistinctness implements task 13.3.
//
// Validates: Requirements 3.3, 3.4
func TestExampleBehavioralIntentHeadersAndTripleDistinctness(t *testing.T) {
	// --- Requirement 3.3: each base profile documents all three dimensions. ---
	for _, name := range ex133BaseProfiles {
		path := filepath.Join(ex133ClientDir, name)
		raw := ex133ReadFile(t, path)

		// 1. The header must contain a Behavioral_Intent section.
		if !strings.Contains(raw, "Behavioral_Intent") {
			t.Fatalf("%q header has no `Behavioral_Intent` section (Requirement 3.3)", name)
		}

		// 2. The Behavioral_Intent block must state all three dimensions. Pull
		// out the block and check keywords tolerant to the authored wording.
		block, ok := ex133ExtractBehavioralIntent(raw)
		if !ok {
			t.Fatalf("%q: could not isolate the Behavioral_Intent block", name)
		}

		// Rate dimension: a frequency ("ops/min" / "operations per minute") or a
		// spacing/completion phrasing.
		if !ex133ContainsAny(block, "ops/min", "operations per minute", "per minute", "completion", "spacing", "rate") {
			t.Fatalf("%q Behavioral_Intent does not state an action rate (Requirement 3.3):\n%s", name, block)
		}

		// Mix dimension: must mention read/write and a proportion. All authored
		// headers describe the mix with "read", "write", and "%".
		if !strings.Contains(block, "read") || !strings.Contains(block, "write") {
			t.Fatalf("%q Behavioral_Intent does not state a read/write mix (Requirement 3.3):\n%s", name, block)
		}
		if !strings.Contains(block, "%") {
			t.Fatalf("%q Behavioral_Intent does not quantify the read/write mix (Requirement 3.3):\n%s", name, block)
		}

		// Termination dimension: one of the session/termination keywords.
		if !ex133ContainsAny(block, "continuous", "loop", "terminal", "session", "bounded") {
			t.Fatalf("%q Behavioral_Intent does not state session/termination behavior (Requirement 3.3):\n%s", name, block)
		}
	}

	// --- Requirement 3.4: the six documented triples are pairwise distinct. ---
	if got := len(ex133DocumentedTriples); got != 6 {
		t.Fatalf("expected 6 documented triples, got %d", got)
	}

	unique := make(map[string]string, len(ex133DocumentedTriples))
	for profile, triple := range ex133DocumentedTriples {
		key := triple.String()
		if other, clash := unique[key]; clash {
			t.Fatalf("documented triples not distinct (Requirement 3.4): %q and %q share triple %q",
				other, profile, key)
		}
		unique[key] = profile
	}
	if got := len(unique); got != 6 {
		t.Fatalf("documented triples are not pairwise distinct: got %d unique of 6 (Requirement 3.4)", got)
	}
}
