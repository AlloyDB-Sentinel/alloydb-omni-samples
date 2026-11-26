package main

// Example test for the markov-profile-library feature, task 13.1.
//
// Asserts the two authoring facts that do not depend on loading or sampling:
//
//  1. The six new base profiles exist on disk at their distinct paths under
//     ../../client/, and those six paths are all unique (a six-element set).
//  2. The proto schema file query_template_config.proto is UNCHANGED by this
//     feature (Requirement 2.1: the feature leaves the schema untouched). The
//     feature is authoring-only, so "unchanged" is asserted structurally: the
//     file still exists at its location and still declares the
//     QueryTemplateConfigs message. When a git baseline is available, we also
//     assert the file has no diff against HEAD; when git is unavailable the
//     existence + declaration check stands alone (no hard dependency on git).
//
// Validates: Requirements 1.8, 2.1
//
// Test working directory is the package dir (v1/generator/src), so the client
// configs live under ../../client/ and the proto under ../proto/, matching the
// path convention in markov_analytics_profile_test.go (which reaches the shipped
// client configs via ../../client/...).
//
// No t.Parallel(): this test only reads files on disk and runs a git command; it
// shares no package globals, but is kept sequential alongside the other
// markov-profile harness tests for consistency.

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// ex131ClientDir is ../../client relative to the package dir, matching the path
// convention in markov_analytics_profile_test.go.
var ex131ClientDir = filepath.Join("..", "..", "client")

// ex131ProtoPath is ../proto/query_template_config.proto relative to the package
// dir (v1/generator/src). The proto lives in v1/generator/proto/.
var ex131ProtoPath = filepath.Join("..", "proto", "query_template_config.proto")

// ex131BaseProfiles are the six new base-profile file names directly under
// ../../client/, one per Requirement 1 criteria 1-6.
var ex131BaseProfiles = []string{
	"markov_analytics_user_config.textproto",
	"markov_batch_etl_config.textproto",
	"markov_read_replica_config.textproto",
	"markov_bursty_api_config.textproto",
	"markov_noisy_client_config.textproto",
	"markov_oltp_app_config.textproto",
}

// ex131FileExists reports whether path names a regular (non-directory) file.
func ex131FileExists(t *testing.T, path string) bool {
	t.Helper()
	info, err := os.Stat(path)
	if err != nil {
		return false
	}
	return !info.IsDir()
}

// TestExampleBaseProfilesExistAtDistinctPaths_ProtoUnchanged implements task 13.1.
//
// Validates: Requirements 1.8, 2.1
func TestExampleBaseProfilesExistAtDistinctPaths_ProtoUnchanged(t *testing.T) {
	// --- Requirement 1.8: exactly six new base profiles at distinct paths. ---

	// Build the six full paths and verify each file exists.
	paths := make([]string, 0, len(ex131BaseProfiles))
	for _, name := range ex131BaseProfiles {
		p := filepath.Join(ex131ClientDir, name)
		paths = append(paths, p)
		if !ex131FileExists(t, p) {
			t.Fatalf("base profile missing: %q does not exist as a file", p)
		}
	}

	// The six paths must be pairwise distinct: collecting them into a set must
	// yield exactly six unique entries.
	unique := make(map[string]struct{}, len(paths))
	for _, p := range paths {
		unique[p] = struct{}{}
	}
	if got := len(unique); got != 6 {
		t.Fatalf("base profile paths are not distinct: got %d unique of %d paths (%v)",
			got, len(paths), paths)
	}

	// --- Requirement 2.1: query_template_config.proto left unchanged. ---

	// The schema file must still exist at ../proto/query_template_config.proto.
	if !ex131FileExists(t, ex131ProtoPath) {
		t.Fatalf("proto schema missing: %q does not exist as a file", ex131ProtoPath)
	}

	// It must still parse as a .proto that declares the QueryTemplateConfigs
	// message (the message the new profiles are authored against).
	raw, err := os.ReadFile(ex131ProtoPath)
	if err != nil {
		t.Fatalf("reading proto %q failed: %v", ex131ProtoPath, err)
	}
	if !strings.Contains(string(raw), "message QueryTemplateConfigs") {
		t.Fatalf("proto %q no longer declares `message QueryTemplateConfigs`", ex131ProtoPath)
	}

	// When a git baseline is available, additionally assert the schema file has no
	// uncommitted diff against HEAD, proving this feature did not modify it. This
	// is strictly supplementary: if git is unavailable or the path is not in a
	// repository, the check is skipped rather than failing the test, so there is
	// no hard dependency on git.
	if ex131GitAvailable(t) {
		if diff, ok := ex131GitDiff(t, ex131ProtoPath); ok && strings.TrimSpace(diff) != "" {
			t.Fatalf("proto %q has uncommitted changes against git HEAD; "+
				"Requirement 2.1 requires it to remain unchanged:\n%s", ex131ProtoPath, diff)
		}
	}
}

// ex131GitAvailable reports whether a `git` binary is on PATH and the package dir
// is inside a git work tree. Any failure (no git, not a repo) means "unavailable".
func ex131GitAvailable(t *testing.T) bool {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		return false
	}
	cmd := exec.Command("git", "rev-parse", "--is-inside-work-tree")
	out, err := cmd.Output()
	if err != nil {
		return false
	}
	return strings.TrimSpace(string(out)) == "true"
}

// ex131GitDiff returns the combined staged+unstaged diff for path against HEAD.
// The bool is false if the git command itself failed (treated as "no baseline").
func ex131GitDiff(t *testing.T, path string) (string, bool) {
	t.Helper()
	cmd := exec.Command("git", "diff", "HEAD", "--", path)
	out, err := cmd.Output()
	if err != nil {
		return "", false
	}
	return string(out), true
}
