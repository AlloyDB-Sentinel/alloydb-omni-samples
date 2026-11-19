package main

// Example test for the markov-profile-library feature, task 14.2.
//
// Task 14.1 added six rows to the "Workload profiles under v1/client" table in
// experimental/anomaly-detection/README.md, one per new base profile. This test
// asserts those rows are present with the correct structure and coverage:
//
//  1. The README contains a markdown table row for EACH of the six new base
//     profiles, with the File column naming the config file in backticks
//     (Requirement 7.1).
//  2. Each row preserves the two-column `| File | Behavior |` single-line
//     structure: a line starting with "| " that contains the backticked
//     filename and has a second "|" separator and a non-empty Behavior cell
//     (Requirement 7.1).
//  3. Each row's Behavior cell covers the three Behavioral_Intent dimensions
//     (Requirement 7.2): action-rate / completion spacing, read/write mix, and
//     session / termination behavior. Checks are tolerant keyword checks keyed
//     to the terminology the existing rows use.
//
// Validates: Requirements 7.1, 7.2
//
// Test working directory is the package dir (v1/generator/src). The README lives
// at experimental/anomaly-detection/README.md, which is three levels up from the
// package dir (src -> generator -> v1 -> anomaly-detection), i.e.
// ../../../README.md. The path is resolved and existence-checked before any row
// assertions so a misplaced README fails loudly rather than silently.
//
// No t.Parallel(): this test only reads a file on disk and shares no package
// globals, but is kept sequential alongside the other markov-profile example
// tests for consistency.

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// ex142ReadmePath is experimental/anomaly-detection/README.md relative to the
// package dir (v1/generator/src): three levels up.
var ex142ReadmePath = filepath.Join("..", "..", "..", "README.md")

// ex142BaseProfiles are the six new base-profile config file names that task 14.1
// added rows for, one per Requirement 1 criteria 1-6.
var ex142BaseProfiles = []string{
	"markov_analytics_user_config.textproto",
	"markov_batch_etl_config.textproto",
	"markov_read_replica_config.textproto",
	"markov_bursty_api_config.textproto",
	"markov_noisy_client_config.textproto",
	"markov_oltp_app_config.textproto",
}

// ex142containsAny reports whether s (already lowercased) contains at least one
// of the given lowercase tokens.
func ex142containsAny(s string, tokens ...string) bool {
	for _, tok := range tokens {
		if strings.Contains(s, tok) {
			return true
		}
	}
	return false
}

// ex142findRow returns the first README line that is a markdown table row (starts
// with "| ") and names the given filename inside backticks, along with whether
// such a line was found.
func ex142findRow(lines []string, filename string) (string, bool) {
	backticked := "`" + filename + "`"
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if !strings.HasPrefix(trimmed, "| ") {
			continue
		}
		if strings.Contains(trimmed, backticked) {
			return trimmed, true
		}
	}
	return "", false
}

// TestExampleReadmeRowsPresentWithStructureAndCoverage implements task 14.2.
//
// Validates: Requirements 7.1, 7.2
func TestExampleReadmeRowsPresentWithStructureAndCoverage(t *testing.T) {
	// Resolve and read the README up front so a misplaced file fails loudly.
	raw, err := os.ReadFile(ex142ReadmePath)
	if err != nil {
		t.Fatalf("reading README %q failed: %v", ex142ReadmePath, err)
	}
	lines := strings.Split(string(raw), "\n")

	for _, name := range ex142BaseProfiles {
		name := name
		t.Run(name, func(t *testing.T) {
			// --- Requirement 7.1: a table row naming the config file exists. ---
			row, ok := ex142findRow(lines, name)
			if !ok {
				t.Fatalf("no README table row found naming %q in backticks", name)
			}

			// --- Requirement 7.1: two-column `| File | Behavior |` structure. ---
			// A single-line two-column row has a leading "| ", the backticked
			// filename in the File cell, then a "|" separator, then a Behavior
			// cell. Split on "|" and require at least File + Behavior cells, both
			// non-empty, with the filename in the first content cell.
			parts := strings.Split(row, "|")
			// A well-formed row "| File | Behavior |" splits into:
			//   ["", " File ", " Behavior ", ""]  -> 4 parts, >= 4 for 2 columns.
			if len(parts) < 4 {
				t.Fatalf("row for %q is not a two-column `| File | Behavior |` row: %q", name, row)
			}
			fileCell := strings.TrimSpace(parts[1])
			behaviorCell := strings.TrimSpace(parts[2])
			if !strings.Contains(fileCell, "`"+name+"`") {
				t.Fatalf("File cell for %q does not name the backticked config file; got File cell %q in row %q",
					name, fileCell, row)
			}
			if behaviorCell == "" {
				t.Fatalf("Behavior cell for %q is empty in row %q", name, row)
			}

			// --- Requirement 7.2: Behavior cell covers the three dimensions. ---
			// Tolerant keyword checks keyed to the existing rows' terminology.
			lc := strings.ToLower(behaviorCell)

			// (a) action-rate / completion spacing: a rate token like "ops/min"
			//     or a spacing/completion token like "second", "spacing", "ms",
			//     or "completion".
			if !ex142containsAny(lc, "ops/min", "op/min", "second", "spacing", "completion", "ms") {
				t.Errorf("Behavior cell for %q lacks an action-rate/spacing token (ops/min, second, spacing, completion, ms); got %q",
					name, behaviorCell)
			}

			// (b) read/write mix: mentions read or write (or a percentage mix).
			if !ex142containsAny(lc, "read", "write", "%") {
				t.Errorf("Behavior cell for %q lacks a read/write-mix token (read, write, %%); got %q",
					name, behaviorCell)
			}

			// (c) session / termination behavior: a termination token like
			//     "continuous"/"loop"/"looping" (no terminal) or
			//     "session"/"bounded"/"terminal"/"session_done" (bounded).
			if !ex142containsAny(lc, "continuous", "loop", "session", "bounded", "terminal", "ending") {
				t.Errorf("Behavior cell for %q lacks a session/termination token (continuous, loop, session, bounded, terminal, ending); got %q",
					name, behaviorCell)
			}
		})
	}
}
