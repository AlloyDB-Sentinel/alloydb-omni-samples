package main

import (
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

// ex132ProfilesDir is the markov_profiles/ directory that holds the per-profile
// variant sets authored by the markov-profile-library tasks. Go tests run with
// the working directory set to the package dir (v1/generator/src), so the client
// configs live at ../../client/ relative to the test, the same convention used by
// markov_analytics_profile_test.go.
var ex132ProfilesDir = filepath.Join("..", "..", "client", "markov_profiles")

// ex132VariantDirs are the six NEW variant directories introduced by this
// feature. The pre-existing standard/, service/, and maintenance/ dirs are
// intentionally excluded.
var ex132VariantDirs = []string{
	"analytics",
	"batch_etl",
	"read_replica",
	"bursty_api",
	"noisy_client",
	"oltp_app",
}

// ex132RequiredFiles is the exact set of six variant files every new profile dir
// must contain, and nothing else (among .textproto files).
var ex132RequiredFiles = []string{
	"subtype.textproto",
	"individual_overrides.textproto",
	"individual_overrides_b.textproto",
	"individual_variance.textproto",
	"individual_jitter.textproto",
	"auto_injection.textproto",
}

// ex132TextprotoNames reads dir and returns the sorted set of its .textproto
// entry names (non-recursive, files only).
func ex132TextprotoNames(t *testing.T, dir string) []string {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("ReadDir(%q) failed: %v", dir, err)
	}
	var names []string
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		if strings.HasSuffix(e.Name(), ".textproto") {
			names = append(names, e.Name())
		}
	}
	sort.Strings(names)
	return names
}

// TestVariantDirsHaveExactlySixRequiredFiles asserts that each of the six new
// variant directories under markov_profiles/ contains exactly the six required
// variant files and no other .textproto files (no missing, no extra).
//
// Validates: Requirements 4.1
func TestVariantDirsHaveExactlySixRequiredFiles(t *testing.T) {
	want := append([]string(nil), ex132RequiredFiles...)
	sort.Strings(want)

	for _, name := range ex132VariantDirs {
		dir := filepath.Join(ex132ProfilesDir, name)
		got := ex132TextprotoNames(t, dir)

		if len(got) != len(want) {
			t.Errorf("%s: found %d .textproto files %v, want exactly %d %v",
				name, len(got), got, len(want), want)
			continue
		}

		gotSet := make(map[string]bool, len(got))
		for _, f := range got {
			gotSet[f] = true
		}
		for _, req := range want {
			if !gotSet[req] {
				t.Errorf("%s: missing required file %q (have %v)", name, req, got)
			}
		}
		wantSet := make(map[string]bool, len(want))
		for _, f := range want {
			wantSet[f] = true
		}
		for _, f := range got {
			if !wantSet[f] {
				t.Errorf("%s: unexpected extra file %q (want only %v)", name, f, want)
			}
		}
	}
}
