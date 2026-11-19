package main

// Property test for the markov-profile-library feature.
//
// Feature: markov-profile-library, Property 4: Every new textproto follows the
// header convention
//
// Property 4 asserts that every one of the 42 new textproto files authored by this
// feature opens with the required two-line proto header: a first non-empty line
// that declares the proto-file (naming query_template_config.proto) and a second
// non-empty line that declares `# proto-message: QueryTemplateConfigs`. This is a
// pure, on-disk convention check, so the property is checked by reading each file's
// raw text and inspecting its first two non-empty lines.
//
// Validates: Requirements 2.2
//
// Test working directory is the package dir (v1/generator/src), so the base
// profiles live under ../../client/ and the variant files under
// ../../client/markov_profiles/<dir>/, exactly as markov_analytics_profile_test.go
// reaches the shipped client configs.

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"testing/quick"
)

// prop4ClientDir is ../../client relative to the package dir, matching the path
// convention in markov_analytics_profile_test.go.
var prop4ClientDir = filepath.Join("..", "..", "client")

// prop4BaseProfiles are the six new base-profile file names directly under
// ../../client/.
var prop4BaseProfiles = []string{
	"markov_analytics_user_config.textproto",
	"markov_batch_etl_config.textproto",
	"markov_read_replica_config.textproto",
	"markov_bursty_api_config.textproto",
	"markov_noisy_client_config.textproto",
	"markov_oltp_app_config.textproto",
}

// prop4VariantDirs are the six variant directories under
// ../../client/markov_profiles/, one per base profile.
var prop4VariantDirs = []string{
	"analytics",
	"batch_etl",
	"read_replica",
	"bursty_api",
	"noisy_client",
	"oltp_app",
}

// prop4VariantFiles are the six variant file names present in every variant dir.
var prop4VariantFiles = []string{
	"subtype.textproto",
	"individual_overrides.textproto",
	"individual_overrides_b.textproto",
	"individual_variance.textproto",
	"individual_jitter.textproto",
	"auto_injection.textproto",
}

// prop4AllTextprotoPaths builds the full list of 42 new textproto paths
// programmatically: the six base profiles under ../../client/ plus, for each of
// the six variant dirs, the six variant files (6 + 6*6 = 42).
func prop4AllTextprotoPaths() []string {
	paths := make([]string, 0, len(prop4BaseProfiles)+len(prop4VariantDirs)*len(prop4VariantFiles))
	for _, base := range prop4BaseProfiles {
		paths = append(paths, filepath.Join(prop4ClientDir, base))
	}
	for _, dir := range prop4VariantDirs {
		for _, file := range prop4VariantFiles {
			paths = append(paths, filepath.Join(prop4ClientDir, "markov_profiles", dir, file))
		}
	}
	return paths
}

// prop4FirstTwoNonEmptyLines returns the first two non-empty (after trimming
// surrounding whitespace) lines of text, or fewer if text has fewer non-empty
// lines.
func prop4FirstTwoNonEmptyLines(text string) []string {
	var out []string
	for _, raw := range strings.Split(text, "\n") {
		line := strings.TrimSpace(raw)
		if line == "" {
			continue
		}
		out = append(out, line)
		if len(out) == 2 {
			break
		}
	}
	return out
}

// prop4HeaderIsValid reports whether the first two non-empty lines of a file's
// raw text follow the header convention: line 1 starts with "# proto-file:" and
// names query_template_config.proto, and line 2 is exactly
// "# proto-message: QueryTemplateConfigs".
func prop4HeaderIsValid(text string) (bool, string) {
	lines := prop4FirstTwoNonEmptyLines(text)
	if len(lines) < 2 {
		return false, "fewer than two non-empty lines"
	}
	if !strings.HasPrefix(lines[0], "# proto-file:") {
		return false, "line 1 does not start with \"# proto-file:\": " + lines[0]
	}
	if !strings.Contains(lines[0], "query_template_config.proto") {
		return false, "line 1 does not name query_template_config.proto: " + lines[0]
	}
	if lines[1] != "# proto-message: QueryTemplateConfigs" {
		return false, "line 2 is not \"# proto-message: QueryTemplateConfigs\": " + lines[1]
	}
	return true, ""
}

// TestProperty4NewTextprotosFollowHeaderConvention checks Property 4: every one of
// the 42 new textproto files opens with the two-line proto header.
//
// Feature: markov-profile-library, Property 4: Every new textproto follows the
// header convention
//
// Validates: Requirements 2.2
//
// The list of 42 paths is built programmatically (6 base + 6 dirs * 6 files) and
// asserted to have exactly 42 entries. Then, using testing/quick over >= 100
// iterations, a randomly chosen path from that fixed set is read from disk and its
// first two non-empty lines are asserted to match the convention. Every iteration
// inspects a real on-disk file, so no mocks or synthetic data are involved.
//
// No t.Parallel(): this test only reads files and shares no package globals, but
// it is kept sequential alongside the other markov-profile harness tests for
// consistency.
func TestProperty4NewTextprotosFollowHeaderConvention(t *testing.T) {
	paths := prop4AllTextprotoPaths()

	// The programmatically built list must contain exactly the 42 new files.
	const wantCount = 42
	if got := len(paths); got != wantCount {
		t.Fatalf("built %d textproto paths, want exactly %d", got, wantCount)
	}

	// Up front, confirm every file exists and satisfies the convention, so a
	// violation is reported against the specific offending path rather than only
	// surfacing probabilistically through the randomized driver below.
	for _, path := range paths {
		raw, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("Property 4: reading %q failed: %v", path, err)
		}
		if ok, reason := prop4HeaderIsValid(string(raw)); !ok {
			t.Fatalf("Property 4 violated for %q: %s", path, reason)
		}
	}

	// Property driver: over >= 100 iterations, pick an arbitrary index into the
	// fixed 42-path set and re-check that file's header convention. Reusing
	// testing/quick keeps this consistent with the feature's property-test
	// convention (>= 100 iterations) rather than hand-rolling an iteration loop.
	property := func(n uint16) bool {
		path := paths[int(n)%len(paths)]
		raw, err := os.ReadFile(path)
		if err != nil {
			t.Errorf("Property 4: reading %q failed: %v", path, err)
			return false
		}
		ok, reason := prop4HeaderIsValid(string(raw))
		if !ok {
			t.Errorf("Property 4 violated for %q: %s", path, reason)
		}
		return ok
	}

	if err := quick.Check(property, &quick.Config{MaxCount: 200}); err != nil {
		t.Fatalf("Property 4: header-convention property failed: %v", err)
	}
}
