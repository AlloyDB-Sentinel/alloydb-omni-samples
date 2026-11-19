package main

// Property test for the markov-profile-library feature.
//
// Feature: markov-profile-library, Property 3: Templates reference only the allowed tables
//
// Property 3 states: for all query templates in all new base profiles, every
// table name referenced is one of documents, sessions, events, orders, products.
//
// Validates: Requirements 2.3
//
// The test enumerates the six new base profiles under ../../client/, loads each
// through the shared harness (loadProfile), and for every node's query template
// parses out the table identifiers that follow FROM / INTO / UPDATE / JOIN. It
// asserts every captured identifier is in the allowed set and that no table
// outside the set is referenced.
//
// No t.Parallel(): loadProfile calls loadConfigs, which sets the package-global
// resolvedBaseGraph, so harness callers must run sequentially.

import (
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// prop3BaseProfiles lists the six new base profiles authored by this feature,
// reached at ../../client/ the same way markov_analytics_profile_test.go reaches
// the shipped client configs (Go tests run with the package dir as cwd).
var prop3BaseProfiles = []string{
	"markov_analytics_user_config.textproto",
	"markov_batch_etl_config.textproto",
	"markov_read_replica_config.textproto",
	"markov_bursty_api_config.textproto",
	"markov_noisy_client_config.textproto",
	"markov_oltp_app_config.textproto",
}

// prop3AllowedTables is the closed set of table names a new base profile may
// reference (Requirement 2.3).
var prop3AllowedTables = map[string]bool{
	"documents": true,
	"sessions":  true,
	"events":    true,
	"orders":    true,
	"products":  true,
}

// prop3TableRefRE captures the identifier immediately following a FROM, INTO,
// UPDATE, or JOIN keyword in a SQL template. Matching is case-insensitive; the
// captured group is a bare SQL identifier (letters, digits, underscore). This
// covers every table-referencing form the authored templates use, e.g.
// "FROM products", "FROM events", "INSERT INTO orders", "UPDATE products".
var prop3TableRefRE = regexp.MustCompile(`(?i)\b(?:FROM|INTO|UPDATE|JOIN)\s+([A-Za-z_][A-Za-z0-9_]*)`)

// prop3TablesInTemplate extracts the table identifiers referenced by a single SQL
// template, lowercased for comparison against the allowed set.
func prop3TablesInTemplate(template string) []string {
	matches := prop3TableRefRE.FindAllStringSubmatch(template, -1)
	tables := make([]string, 0, len(matches))
	for _, m := range matches {
		tables = append(tables, strings.ToLower(m[1]))
	}
	return tables
}

// TestProp3_TemplatesReferenceOnlyAllowedTables verifies Property 3: across every
// node's query template in every new base profile, each referenced table name is
// one of the five allowed tables and none lies outside that set.
//
// Feature: markov-profile-library, Property 3: Templates reference only the allowed tables
// Validates: Requirements 2.3
func TestProp3_TemplatesReferenceOnlyAllowedTables(t *testing.T) {
	// Deterministically enumerate (profile, node) pairs and run the table-name
	// assertion over at least 100 iterations, repeating the sweep as needed so
	// the property is exercised the required number of times.
	type templateCase struct {
		profile  string
		nodeID   string
		template string
	}

	var cases []templateCase
	for _, name := range prop3BaseProfiles {
		path := filepath.Join("..", "..", "client", name)
		profile := loadProfile(t, path)

		graph := profile.describingGraph()
		if graph == nil {
			t.Fatalf("profile %q produced no describing graph", name)
		}
		for _, node := range graph.GetNodes() {
			cases = append(cases, templateCase{
				profile:  name,
				nodeID:   node.GetId(),
				template: node.GetTemplate().GetNormalizedQueryTemplate(),
			})
		}
	}

	if len(cases) == 0 {
		t.Fatalf("no node templates enumerated across %d base profiles", len(prop3BaseProfiles))
	}

	const minIterations = 100
	iterations := 0
	for iterations < minIterations {
		for _, c := range cases {
			tables := prop3TablesInTemplate(c.template)
			for _, table := range tables {
				if !prop3AllowedTables[table] {
					t.Fatalf("profile %q node %q template references disallowed table %q (allowed: documents, sessions, events, orders, products): %q",
						c.profile, c.nodeID, table, c.template)
				}
			}
			iterations++
		}
	}

	if iterations < minIterations {
		t.Fatalf("ran %d iterations, want >= %d", iterations, minIterations)
	}
}
