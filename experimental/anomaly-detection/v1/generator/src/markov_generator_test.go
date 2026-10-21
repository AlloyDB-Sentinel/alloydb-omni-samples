package main

import (
	"bufio"
	"bytes"
	"io"
	"math/rand"
	"sort"
	"testing"

	"google.golang.org/protobuf/encoding/protodelim"

	genpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v1/generator/proto"
)

// markovCfg parses a textproto QueryTemplateConfigs and returns its MarkovSamplingConfig.
func markovCfg(t *testing.T, text string) *genpb.MarkovSamplingConfig {
	t.Helper()
	return parseProto(t, text, &genpb.QueryTemplateConfigs{}).GetMarkovSampling()
}

// TestNewMarkovGraph verifies the in-memory graph has exactly one node per configured id and
// that each node's outgoing destination set matches its configured edges, including self-loops
// and empty sets for terminal nodes.
//
// Property 10, Validates R3.2, R3.4, R4.7, R4.10
func TestNewMarkovGraph(t *testing.T) {
	tests := []struct {
		name string
		text string
		// want maps each node id to its expected set of outgoing destination ids.
		want map[string][]string
	}{
		{
			name: "linear_and_terminal",
			text: `
markov_sampling {
	nodes: {
		id: "a"
		template: { normalized_query_template: "SELECT 1" }
		completion_time: { fixed: { nanos: 2000000 } }
		edges: { destination_id: "b" weight: 1.0 }
	}
	nodes: {
		id: "b"
		template: { normalized_query_template: "SELECT 2" }
		completion_time: { fixed: { nanos: 2000000 } }
	}
	start: { node_id: "a" }
}`,
			want: map[string][]string{
				"a": {"b"},
				"b": {},
			},
		},
		{
			name: "self_loop_and_branching",
			text: `
markov_sampling {
	nodes: {
		id: "login"
		template: { normalized_query_template: "SELECT 1" }
		completion_time: { fixed: { nanos: 2000000 } }
		edges: { destination_id: "browse" weight: 0.7 }
		edges: { destination_id: "checkout" weight: 0.3 }
	}
	nodes: {
		id: "browse"
		template: { normalized_query_template: "SELECT 2" }
		completion_time: { range: { min: { nanos: 1000000 } max: { nanos: 5000000 } } }
		edges: { destination_id: "browse" weight: 0.5 }
		edges: { destination_id: "checkout" weight: 0.5 }
	}
	nodes: {
		id: "checkout"
		template: { normalized_query_template: "SELECT 3" }
		completion_time: { fixed: { nanos: 3000000 } }
	}
	start: { node_id: "login" }
}`,
			want: map[string][]string{
				"login":    {"browse", "checkout"},
				"browse":   {"browse", "checkout"},
				"checkout": {},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			graph := newMarkovGraph(markovCfg(t, test.text))

			if got, want := len(graph.nodes), len(test.want); got != want {
				t.Errorf("newMarkovGraph() produced %d nodes, want %d", got, want)
			}

			for id, wantDests := range test.want {
				node, ok := graph.nodes[id]
				if !ok {
					t.Errorf("newMarkovGraph() missing node %q", id)
					continue
				}

				gotDests := append([]string(nil), node.destIDs...)
				sort.Strings(gotDests)
				wantSorted := append([]string(nil), wantDests...)
				sort.Strings(wantSorted)

				if !equalStrings(gotDests, wantSorted) {
					t.Errorf("node %q destIDs = %v, want %v", id, gotDests, wantSorted)
				}

				// The cumulative-weight table must stay aligned with the destination list.
				if len(node.cumulativeWeights) != len(node.destIDs) {
					t.Errorf("node %q has %d cumulative weights, want %d (one per edge)",
						id, len(node.cumulativeWeights), len(node.destIDs))
				}
			}
		})
	}
}

// TestSampleCompletionMs verifies fixed completion times always return the exact value and range
// completion times (including min == max) always fall within the closed interval [minMs, maxMs].
//
// Property 7, Validates R2.5, R2.6, R2.7
func TestSampleCompletionMs(t *testing.T) {
	const (
		seed    = int64(42)
		samples = 1000
	)

	tests := []struct {
		name  string
		text  string
		minMs int64
		maxMs int64
		// exact is true when every draw must equal minMs (fixed, or a range with min == max).
		exact bool
	}{
		{
			name:  "fixed",
			text:  `completion_time: { fixed: { nanos: 2000000 } }`,
			minMs: 2,
			maxMs: 2,
			exact: true,
		},
		{
			name:  "range",
			text:  `completion_time: { range: { min: { nanos: 1000000 } max: { nanos: 5000000 } } }`,
			minMs: 1,
			maxMs: 5,
			exact: false,
		},
		{
			name:  "range_min_equals_max",
			text:  `completion_time: { range: { min: { nanos: 4000000 } max: { nanos: 4000000 } } }`,
			minMs: 4,
			maxMs: 4,
			exact: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			ct := completionTime(t, test.text)
			rng := rand.New(rand.NewSource(seed))

			sawMin, sawMax := false, false
			for i := 0; i < samples; i++ {
				got := sampleCompletionMs(ct, rng)
				if got < test.minMs || got > test.maxMs {
					t.Fatalf("sampleCompletionMs() = %d, want in [%d, %d]", got, test.minMs, test.maxMs)
				}
				if test.exact && got != test.minMs {
					t.Fatalf("sampleCompletionMs() = %d, want exactly %d", got, test.minMs)
				}
				if got == test.minMs {
					sawMin = true
				}
				if got == test.maxMs {
					sawMax = true
				}
			}

			// A non-trivial range should exercise both endpoints across many draws.
			if !test.exact && (!sawMin || !sawMax) {
				t.Errorf("range never reached an endpoint over %d samples: sawMin=%v sawMax=%v",
					samples, sawMin, sawMax)
			}
		})
	}
}

// completionTime parses a single-node markov config and returns that node's CompletionTime.
func completionTime(t *testing.T, ctText string) *genpb.CompletionTime {
	t.Helper()
	cfg := markovCfg(t, `
markov_sampling {
	nodes: {
		id: "n"
		template: { normalized_query_template: "SELECT 1" }
		`+ctText+`
	}
	start: { node_id: "n" }
}`)
	return cfg.GetNodes()[0].GetCompletionTime()
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// --- Task 5.4 / 5.5 walker, generation, and statistical tests ---

// runMarkov generates numLogs entries from the given textproto config using a
// deterministic seed and decodes all length-delimited AuditLogEntry records.
func runMarkov(t *testing.T, text string, numLogs int, startTimeMs int64, seed int64) []*genpb.AuditLogEntry {
	t.Helper()
	cfg := parseProto(t, text, &genpb.QueryTemplateConfigs{})
	var buf bytes.Buffer
	if err := GenerateLogsMarkov(cfg, numLogs, startTimeMs, rand.New(rand.NewSource(seed)), &buf); err != nil {
		t.Fatalf("GenerateLogsMarkov() error = %v", err)
	}
	var entries []*genpb.AuditLogEntry
	r := bufio.NewReader(&buf)
	for {
		e := &genpb.AuditLogEntry{}
		if err := protodelim.UnmarshalFrom(r, e); err != nil {
			if err == io.EOF {
				break
			}
			t.Fatalf("protodelim.UnmarshalFrom() error = %v", err)
		}
		entries = append(entries, e)
	}
	return entries
}

// runMarkovRaw generates numLogs entries and returns the raw serialized bytes,
// used by determinism assertions that compare output byte-for-byte.
func runMarkovRaw(t *testing.T, text string, numLogs int, startTimeMs int64, seed int64) []byte {
	t.Helper()
	cfg := parseProto(t, text, &genpb.QueryTemplateConfigs{})
	var buf bytes.Buffer
	if err := GenerateLogsMarkov(cfg, numLogs, startTimeMs, rand.New(rand.NewSource(seed)), &buf); err != nil {
		t.Fatalf("GenerateLogsMarkov() error = %v", err)
	}
	return buf.Bytes()
}

// A 3-node chain login -> browse(self-loop) -> checkout(terminal), each with a
// distinct template, used by several generation tests. Walks always start at
// login, so the login template marks the start of every walk segment.
const chainConfig = `
markov_sampling {
	nodes: {
		id: "login"
		template: { normalized_query_template: "LOGIN QUERY" }
		completion_time: { fixed: { nanos: 1000000 } }
		edges: { destination_id: "browse" weight: 1.0 }
	}
	nodes: {
		id: "browse"
		template: { normalized_query_template: "BROWSE QUERY" }
		completion_time: { fixed: { nanos: 1000000 } }
		edges: { destination_id: "browse" weight: 0.5 }
		edges: { destination_id: "checkout" weight: 0.5 }
	}
	nodes: {
		id: "checkout"
		template: { normalized_query_template: "CHECKOUT QUERY" }
		completion_time: { fixed: { nanos: 1000000 } }
	}
	start: { node_id: "login" }
}`

// TestGenerateLogsMarkov_Count verifies the emitted record count equals num_logs
// exactly and never exceeds it, across several num_logs values and configs,
// including a chain that reaches a terminal node and forces restarts.
//
// Property 1, Validates R7.1, R9.1, R9.2, R9.3, R9.5, R9.7
func TestGenerateLogsMarkov_Count(t *testing.T) {
	// A single-node terminal start: the walk emits one entry then restarts.
	const terminalStart = `
markov_sampling {
	nodes: {
		id: "only"
		template: { normalized_query_template: "ONLY QUERY" }
		completion_time: { fixed: { nanos: 1000000 } }
	}
	start: { node_id: "only" }
}`

	tests := []struct {
		name string
		text string
	}{
		{name: "chain_with_restarts", text: chainConfig},
		{name: "terminal_start", text: terminalStart},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			for _, numLogs := range []int{0, 1, 2, 5, 13, 50} {
				entries := runMarkov(t, test.text, numLogs, 0, 7)
				if len(entries) != numLogs {
					t.Errorf("num_logs=%d: got %d entries, want %d", numLogs, len(entries), numLogs)
				}
			}
		})
	}
}

// TestGenerateLogsMarkov_Timestamps verifies timestamp_ms is non-decreasing, the
// first entry equals start_time_ms, fixed-duration walks produce the exact
// running-sum sequence, and the clock carries across restarts.
//
// Property 2, Validates R7.4, R7.5, R7.6, R7.7, R7.8, R7.9
func TestGenerateLogsMarkov_Timestamps(t *testing.T) {
	const startTimeMs = int64(1000)

	// Fixed 1ms completion everywhere, terminal start -> every walk is one node.
	// With a terminal start node the clock still carries across restarts, so the
	// sequence is a simple running sum: 1000, 1001, 1002, ...
	const terminalFixed = `
markov_sampling {
	nodes: {
		id: "only"
		template: { normalized_query_template: "ONLY QUERY" }
		completion_time: { fixed: { nanos: 1000000 } }
	}
	start: { node_id: "only" }
}`

	t.Run("exact_running_sum_across_restarts", func(t *testing.T) {
		const numLogs = 6
		entries := runMarkov(t, terminalFixed, numLogs, startTimeMs, 11)
		for i, e := range entries {
			want := startTimeMs + int64(i) // 1ms per emission
			if got := e.GetTimestampMs(); got != want {
				t.Errorf("entry[%d].timestamp_ms = %d, want %d", i, got, want)
			}
		}
	})

	// The chain uses fixed 1ms completion at every node too, so even with
	// multi-node walks and restarts the sequence is a perfect running sum.
	t.Run("chain_non_decreasing_and_exact", func(t *testing.T) {
		const numLogs = 20
		entries := runMarkov(t, chainConfig, numLogs, startTimeMs, 3)
		if len(entries) == 0 {
			t.Fatal("expected entries")
		}
		if got := entries[0].GetTimestampMs(); got != startTimeMs {
			t.Errorf("first timestamp_ms = %d, want %d", got, startTimeMs)
		}
		prev := int64(-1)
		for i, e := range entries {
			ts := e.GetTimestampMs()
			if ts < prev {
				t.Errorf("entry[%d].timestamp_ms = %d < previous %d (not non-decreasing)", i, ts, prev)
			}
			if want := startTimeMs + int64(i); ts != want {
				t.Errorf("entry[%d].timestamp_ms = %d, want %d", i, ts, want)
			}
			prev = ts
		}
	})
}

// TestGenerateLogsMarkov_Transitions verifies the observed template sequence
// decomposes into repeated [login, browse*, checkout] walk segments where every
// adjacency corresponds to a configured edge and the terminal node's template
// appears only as the last entry of a walk.
//
// Property 3, Validates R6.1, R6.3, R9.4
func TestGenerateLogsMarkov_Transitions(t *testing.T) {
	// Map each node template back to its id, and the set of allowed edges.
	templateToID := map[string]string{
		"LOGIN QUERY":    "login",
		"BROWSE QUERY":   "browse",
		"CHECKOUT QUERY": "checkout",
	}
	edges := map[string]map[string]bool{
		"login":  {"browse": true},
		"browse": {"browse": true, "checkout": true},
	}
	terminal := map[string]bool{"checkout": true}

	// Generate enough entries to exercise many restarts.
	entries := runMarkov(t, chainConfig, 200, 0, 99)

	var ids []string
	for i, e := range entries {
		id, ok := templateToID[e.GetNormalizedQueryTemplate()]
		if !ok {
			t.Fatalf("entry[%d] has unknown template %q", i, e.GetNormalizedQueryTemplate())
		}
		ids = append(ids, id)
	}

	// The very first entry must be the fixed start node.
	if ids[0] != "login" {
		t.Errorf("first visited id = %q, want %q", ids[0], "login")
	}

	for i := 0; i < len(ids); i++ {
		cur := ids[i]
		if i+1 >= len(ids) {
			break
		}
		next := ids[i+1]

		if terminal[cur] {
			// A terminal node ends the walk; the next entry must be a fresh start.
			if next != "login" {
				t.Errorf("entry[%d]=%q is terminal but next=%q, want a walk restart at %q",
					i, cur, next, "login")
			}
			continue
		}

		// Non-terminal: the adjacency must be a configured edge.
		if !edges[cur][next] {
			t.Errorf("entry[%d]->[%d]: transition %q->%q is not a configured edge", i, i+1, cur, next)
		}
	}
}

// TestGenerateLogsMarkov_EntryFields verifies each entry carries the node's exact
// normalized_query_template, a query_statement rendered from that template, and
// the shared db_user / db_name constants.
//
// Property 8, Validates R7.2, R7.3, R7.10, R10.3
func TestGenerateLogsMarkov_EntryFields(t *testing.T) {
	// A single node with one fixed string placeholder makes the query statement
	// fully deterministic regardless of rng.
	const fieldsConfig = `
markov_sampling {
	nodes: {
		id: "only"
		template: {
			normalized_query_template: "SELECT * FROM t WHERE region = '%v'"
			placeholder_values: { string_placeholder: { allowed_values: "us-east" } }
		}
		completion_time: { fixed: { nanos: 1000000 } }
	}
	start: { node_id: "only" }
}`

	entries := runMarkov(t, fieldsConfig, 5, 0, 1)
	if len(entries) != 5 {
		t.Fatalf("got %d entries, want 5", len(entries))
	}

	const (
		wantTemplate  = "SELECT * FROM t WHERE region = '%v'"
		wantStatement = "SELECT * FROM t WHERE region = 'us-east'"
	)
	for i, e := range entries {
		if got := e.GetNormalizedQueryTemplate(); got != wantTemplate {
			t.Errorf("entry[%d].normalized_query_template = %q, want %q", i, got, wantTemplate)
		}
		if got := e.GetQueryStatement(); got != wantStatement {
			t.Errorf("entry[%d].query_statement = %q, want %q", i, got, wantStatement)
		}
		if got := e.GetDbUser(); got != constUser {
			t.Errorf("entry[%d].db_user = %q, want %q", i, got, constUser)
		}
		if got := e.GetDbName(); got != constDbName {
			t.Errorf("entry[%d].db_name = %q, want %q", i, got, constDbName)
		}
	}
}

// TestGenerateLogsMarkov_AnomalyFlag verifies is_ground_truth_anomalous mirrors
// each node template's value: true when set, false when left unset.
//
// Property 9, Validates R8.1, R8.2
func TestGenerateLogsMarkov_AnomalyFlag(t *testing.T) {
	// normal -> anomalous(terminal). normal leaves the flag unset (false);
	// anomalous sets it true.
	const flagConfig = `
markov_sampling {
	nodes: {
		id: "normal"
		template: { normalized_query_template: "NORMAL QUERY" }
		completion_time: { fixed: { nanos: 1000000 } }
		edges: { destination_id: "anomalous" weight: 1.0 }
	}
	nodes: {
		id: "anomalous"
		template: {
			normalized_query_template: "ANOMALOUS QUERY"
			is_ground_truth_anomalous: true
		}
		completion_time: { fixed: { nanos: 1000000 } }
	}
	start: { node_id: "normal" }
}`

	entries := runMarkov(t, flagConfig, 40, 0, 5)
	for i, e := range entries {
		switch e.GetNormalizedQueryTemplate() {
		case "NORMAL QUERY":
			if e.GetIsGroundTruthAnomalous() {
				t.Errorf("entry[%d] (normal) is_ground_truth_anomalous = true, want false", i)
			}
		case "ANOMALOUS QUERY":
			if !e.GetIsGroundTruthAnomalous() {
				t.Errorf("entry[%d] (anomalous) is_ground_truth_anomalous = false, want true", i)
			}
		default:
			t.Fatalf("entry[%d] unexpected template %q", i, e.GetNormalizedQueryTemplate())
		}
	}
}

// TestMarkovFixedStart verifies a fixed start { node_id: "X" } always produces an
// entry for node X as the first record, across multiple seeds.
//
// Property 5, Validates R5.2
func TestMarkovFixedStart(t *testing.T) {
	// browse is reachable from login but login is the configured fixed start, so
	// the first entry must always be login's template.
	for _, seed := range []int64{1, 2, 3, 42, 100, 2024} {
		entries := runMarkov(t, chainConfig, 10, 0, seed)
		if len(entries) == 0 {
			t.Fatalf("seed=%d: expected entries", seed)
		}
		if got := entries[0].GetNormalizedQueryTemplate(); got != "LOGIN QUERY" {
			t.Errorf("seed=%d: first entry template = %q, want %q", seed, got, "LOGIN QUERY")
		}
	}
}

// TestGenerateLogsMarkov_Determinism verifies two runs with identical config,
// seed, num_logs, and start produce byte-identical output.
//
// Property 11, Validates R11.1, R11.2, R11.3
func TestGenerateLogsMarkov_Determinism(t *testing.T) {
	// Include a string placeholder so buildQueryStatement also draws from rng,
	// exercising full randomness determinism.
	const detConfig = `
markov_sampling {
	nodes: {
		id: "login"
		template: {
			normalized_query_template: "LOGIN %v"
			placeholder_values: { string_placeholder: { allowed_values: "a" allowed_values: "b" } }
		}
		completion_time: { range: { min: { nanos: 1000000 } max: { nanos: 9000000 } } }
		edges: { destination_id: "browse" weight: 0.6 }
		edges: { destination_id: "checkout" weight: 0.4 }
	}
	nodes: {
		id: "browse"
		template: { normalized_query_template: "BROWSE QUERY" }
		completion_time: { fixed: { nanos: 2000000 } }
		edges: { destination_id: "browse" weight: 0.3 }
		edges: { destination_id: "checkout" weight: 0.7 }
	}
	nodes: {
		id: "checkout"
		template: { normalized_query_template: "CHECKOUT QUERY" }
		completion_time: { fixed: { nanos: 3000000 } }
	}
	start: { node_id: "login" }
}`

	const (
		numLogs     = 100
		startTimeMs = 500
		seed        = 777
	)
	a := runMarkovRaw(t, detConfig, numLogs, startTimeMs, seed)
	b := runMarkovRaw(t, detConfig, numLogs, startTimeMs, seed)
	if !bytes.Equal(a, b) {
		t.Errorf("two runs with identical inputs produced different output (%d vs %d bytes)", len(a), len(b))
	}
}

// TestMarkovTransitionFrequencies verifies selectNextID picks destinations in
// proportion to their configured edge weights over many draws under a fixed seed.
//
// Property 4, Validates R6.1, R6.2
func TestMarkovTransitionFrequencies(t *testing.T) {
	// Node "a" transitions to b (0.7) and c (0.3). Sampling selectNextID directly
	// avoids walk-restart complications.
	const text = `
markov_sampling {
	nodes: {
		id: "a"
		template: { normalized_query_template: "A QUERY" }
		completion_time: { fixed: { nanos: 1000000 } }
		edges: { destination_id: "b" weight: 0.7 }
		edges: { destination_id: "c" weight: 0.3 }
	}
	nodes: {
		id: "b"
		template: { normalized_query_template: "B QUERY" }
		completion_time: { fixed: { nanos: 1000000 } }
	}
	nodes: {
		id: "c"
		template: { normalized_query_template: "C QUERY" }
		completion_time: { fixed: { nanos: 1000000 } }
	}
	start: { node_id: "a" }
}`

	const (
		iterations = 100000
		tolerance  = 0.02
	)
	graph := newMarkovGraph(markovCfg(t, text))
	walker := &markovWalker{
		graph:     graph,
		generator: NewLogGenerator(rand.New(rand.NewSource(12345))),
		rng:       rand.New(rand.NewSource(12345)),
	}
	nodeA := graph.nodes["a"]

	counts := map[string]int{}
	for i := 0; i < iterations; i++ {
		counts[walker.selectNextID(nodeA)]++
	}

	want := map[string]float64{"b": 0.7, "c": 0.3}
	for id, wantFreq := range want {
		gotFreq := float64(counts[id]) / float64(iterations)
		if gotFreq < wantFreq-tolerance || gotFreq > wantFreq+tolerance {
			t.Errorf("destination %q frequency = %.4f, want %.2f (+/- %.2f)", id, gotFreq, wantFreq, tolerance)
		}
	}
	// No unexpected destinations should ever be returned.
	for id := range counts {
		if _, ok := want[id]; !ok {
			t.Errorf("selectNextID returned unexpected destination %q", id)
		}
	}
}

// TestMarkovWeightedStart verifies selectStartID draws start nodes in proportion
// to their configured weights, returning only listed start ids, under a fixed seed.
//
// Property 6, Validates R5.3
func TestMarkovWeightedStart(t *testing.T) {
	// Weighted start distribution over x (0.2), y (0.5), z (0.3).
	const text = `
markov_sampling {
	nodes: {
		id: "x"
		template: { normalized_query_template: "X QUERY" }
		completion_time: { fixed: { nanos: 1000000 } }
	}
	nodes: {
		id: "y"
		template: { normalized_query_template: "Y QUERY" }
		completion_time: { fixed: { nanos: 1000000 } }
	}
	nodes: {
		id: "z"
		template: { normalized_query_template: "Z QUERY" }
		completion_time: { fixed: { nanos: 1000000 } }
	}
	start: {
		distribution: {
			starts: { node_id: "x" weight: 0.2 }
			starts: { node_id: "y" weight: 0.5 }
			starts: { node_id: "z" weight: 0.3 }
		}
	}
}`

	const (
		iterations = 100000
		tolerance  = 0.02
	)
	graph := newMarkovGraph(markovCfg(t, text))
	walker := &markovWalker{
		graph:     graph,
		generator: NewLogGenerator(rand.New(rand.NewSource(54321))),
		rng:       rand.New(rand.NewSource(54321)),
	}

	want := map[string]float64{"x": 0.2, "y": 0.5, "z": 0.3}
	counts := map[string]int{}
	for i := 0; i < iterations; i++ {
		id := walker.selectStartID()
		if _, ok := want[id]; !ok {
			t.Fatalf("selectStartID returned unlisted start id %q", id)
		}
		counts[id]++
	}

	for id, wantFreq := range want {
		gotFreq := float64(counts[id]) / float64(iterations)
		if gotFreq < wantFreq-tolerance || gotFreq > wantFreq+tolerance {
			t.Errorf("start %q frequency = %.4f, want %.2f (+/- %.2f)", id, gotFreq, wantFreq, tolerance)
		}
	}
}
