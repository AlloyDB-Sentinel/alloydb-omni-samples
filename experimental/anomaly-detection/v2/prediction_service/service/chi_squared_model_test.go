package service

import (
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"testing"
	"testing/quick"

	"google.golang.org/protobuf/proto"

	modelpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common/proto"
)

// windowContents returns the window's logical contents in arrival (FIFO) order,
// walking the ring buffer from head over size elements modulo capacity. The
// first element is the oldest, the last is the most recently appended.
func windowContents(w *window) []string {
	capacity := len(w.buf)
	out := make([]string, 0, w.size)
	for i := 0; i < w.size; i++ {
		out = append(out, w.buf[(w.head+i)%capacity])
	}
	return out
}

// countsOf builds the expected per-Action count map for a slice of Actions.
func countsOf(actions []string) map[string]int {
	m := make(map[string]int)
	for _, a := range actions {
		m[a]++
	}
	return m
}

// Feature: chi-squared-anomaly-detection, Property 4: Windows retain the most recent, FIFO-ordered Actions bounded by capacity
func TestWindowFIFOCapacitySemantics(t *testing.T) {
	// property checks a single (capacity, stream) draw: after appending the
	// whole stream, the window must equal the last min(len, cap) Actions in
	// arrival order and all window invariants must hold.
	property := func(rawCap uint8, stream []string) bool {
		// Constrain capacity to a small, non-negative range. A zero-capacity
		// window holds nothing (used by the nil-proto model); include it.
		capacity := int(rawCap % 33) // 0..32

		w := newWindow(capacity)
		for _, action := range stream {
			w.append(action)
		}

		// Expected logical contents: the last min(len(stream), capacity)
		// Actions in arrival order. A zero-capacity window retains nothing.
		var expected []string
		if capacity == 0 {
			expected = []string{}
		} else if len(stream) <= capacity {
			expected = append(expected, stream...)
		} else {
			expected = append(expected, stream[len(stream)-capacity:]...)
		}

		got := windowContents(w)

		// size == min(number of appends, capacity).
		if w.size != len(expected) {
			t.Logf("size mismatch: got %d, want %d (cap=%d, stream=%d)", w.size, len(expected), capacity, len(stream))
			return false
		}

		// Contents equal the last min(len, cap) Actions in arrival order. This
		// also pins down FIFO eviction: the oldest surviving Action is the
		// element that was appended earliest among those retained, so the one
		// evicted on overflow is always the earliest-appended.
		if len(got) != len(expected) {
			return false
		}
		for i := range expected {
			if got[i] != expected[i] {
				t.Logf("contents mismatch at %d: got %q, want %q", i, got[i], expected[i])
				return false
			}
		}

		// Invariant: sum(counts) == size.
		sum := 0
		for _, c := range w.counts {
			sum += c
		}
		if sum != w.size {
			t.Logf("sum(counts)=%d != size=%d", sum, w.size)
			return false
		}

		// Invariant: every count >= 1; no zero-count keys present.
		for k, c := range w.counts {
			if c < 1 {
				t.Logf("non-positive count for key %q: %d", k, c)
				return false
			}
		}

		// Invariant: the count map matches the multiset of logical contents
		// (which also confirms no stale/zero-count keys linger after eviction).
		expectedCounts := countsOf(got)
		if len(expectedCounts) != len(w.counts) {
			t.Logf("distinct-key count mismatch: counts=%d, contents=%d", len(w.counts), len(expectedCounts))
			return false
		}
		for k, c := range expectedCounts {
			if w.counts[k] != c {
				t.Logf("count mismatch for key %q: counts=%d, contents=%d", k, w.counts[k], c)
				return false
			}
		}

		return true
	}

	// Use a generator that biases toward a small Action alphabet so overflow
	// and eviction are exercised often (capacities are small, streams can be
	// long). quick's default string generator would produce mostly-distinct
	// values; a small alphabet makes the count-map eviction path meaningful.
	cfg := &quick.Config{
		MaxCount: 1000,
		Values: func(args []reflect.Value, rng *rand.Rand) {
			args[0] = reflect.ValueOf(uint8(rng.Intn(256)))

			n := rng.Intn(80) // stream length 0..79
			stream := make([]string, n)
			for i := range stream {
				// alphabet of ~5 Query_Types
				stream[i] = fmt.Sprintf("q%d", rng.Intn(5))
			}
			args[1] = reflect.ValueOf(stream)
		},
	}

	if err := quick.Check(property, cfg); err != nil {
		t.Error(err)
	}
}

// Feature: chi-squared-anomaly-detection, Property 1: Construction validates parameters and leaves the proto unchanged
func TestConstructionValidatesParametersAndLeavesProtoUnchanged(t *testing.T) {
	// A small set of "interesting" threshold values the generator draws from so
	// the NaN/±Inf/0/negative rejection branches are hit frequently rather than
	// relying on a raw float64 draw (which would almost never produce them).
	interestingThresholds := []float64{
		math.NaN(),
		math.Inf(1),
		math.Inf(-1),
		0.0,
		-0.0,
		-1.0,
		-1e-9,
		1e-9,
		0.5,
		1.0,
		5.0,
		1000.0,
	}

	// chiSquaredProto builds a chi-squared Model proto from (m, n, threshold)
	// using the generated builder API.
	chiSquaredProto := func(m, n int64, threshold float64) *modelpb.Model {
		cfg := (&modelpb.ChiSquaredTrainingConfig_builder{
			M:                m,
			N:                n,
			AnomalyThreshold: threshold,
		}).Build()
		scenario := (&modelpb.ChiSquaredScenario_builder{Config: cfg}).Build()
		return (&modelpb.Model_builder{ChiSquared: scenario}).Build()
	}

	// property checks a single draw: construction succeeds (non-nil model, nil
	// error) iff the config is valid, fails (nil model, non-nil error)
	// otherwise, and in all cases leaves the proto byte-for-byte unchanged.
	property := func(pb *modelpb.Model, wantValid bool) bool {
		before := proto.Clone(pb)

		model, err := NewChiSquaredModel(pb)

		// Validity predicate: non-nil model + nil error iff valid.
		if wantValid {
			if model == nil || err != nil {
				t.Logf("valid config rejected: model=%v, err=%v", model, err)
				return false
			}
		} else {
			if model != nil || err == nil {
				t.Logf("invalid/wrong-scenario config accepted: model=%v, err=%v", model, err)
				return false
			}
		}

		// The proto must be unchanged after the call, whether it succeeded or
		// failed (R1.5).
		if !proto.Equal(before, pb) {
			t.Logf("proto mutated by NewChiSquaredModel: before=%v, after=%v", before, pb)
			return false
		}
		return true
	}

	cfg := &quick.Config{
		MaxCount: 1000,
		Values: func(args []reflect.Value, rng *rand.Rand) {
			// Draw one of a few proto shapes so both valid and invalid
			// chi-squared configs and wrong-scenario protos are exercised.
			switch rng.Intn(4) {
			case 0:
				// Chi-squared proto with boundary-ish M/N and an interesting
				// threshold. M/N drawn from a small range including 0 so the
				// M<=0, N<=0, and N>=M branches are all reachable.
				m := int64(rng.Intn(9))        // 0..8
				n := int64(rng.Intn(9))        // 0..8
				threshold := interestingThresholds[rng.Intn(len(interestingThresholds))]
				pb := chiSquaredProto(m, n, threshold)
				valid := m > 0 && n > 0 && n < m &&
					!math.IsNaN(threshold) && !math.IsInf(threshold, 0) && threshold > 0
				args[0] = reflect.ValueOf(pb)
				args[1] = reflect.ValueOf(valid)
			case 1:
				// Chi-squared proto with wider random M/N and a random positive
				// threshold, biased toward the valid region to exercise the
				// success path more.
				m := int64(rng.Intn(1000) + 1) // 1..1000
				n := int64(rng.Intn(1000) + 1) // 1..1000
				threshold := rng.Float64()*100 + 1e-6
				pb := chiSquaredProto(m, n, threshold)
				valid := m > 0 && n > 0 && n < m &&
					!math.IsNaN(threshold) && !math.IsInf(threshold, 0) && threshold > 0
				args[0] = reflect.ValueOf(pb)
				args[1] = reflect.ValueOf(valid)
			case 2:
				// Wrong-scenario proto: a volumetric-spike Model. Never valid
				// for the chi-squared constructor.
				scenario := (&modelpb.VolumetricSpikeScenario_builder{}).Build()
				pb := (&modelpb.Model_builder{VolumetricSpike: scenario}).Build()
				args[0] = reflect.ValueOf(pb)
				args[1] = reflect.ValueOf(false)
			default:
				// Empty Model (no scenario set): also wrong-scenario, never valid.
				pb := (&modelpb.Model_builder{}).Build()
				args[0] = reflect.ValueOf(pb)
				args[1] = reflect.ValueOf(false)
			}
		},
	}

	if err := quick.Check(property, cfg); err != nil {
		t.Error(err)
	}
}

// chiSquaredModelProto builds a valid-shaped chi-squared Model proto from
// (m, n, threshold) using the generated builder API. The parameter values are
// not validated here; callers pass invalid values on purpose to exercise the
// constructor's validation branches.
func chiSquaredModelProto(m, n int64, threshold float64) *modelpb.Model {
	cfg := (&modelpb.ChiSquaredTrainingConfig_builder{
		M:                m,
		N:                n,
		AnomalyThreshold: threshold,
	}).Build()
	scenario := (&modelpb.ChiSquaredScenario_builder{Config: cfg}).Build()
	return (&modelpb.Model_builder{ChiSquared: scenario}).Build()
}

// TestNewChiSquaredModelWrongScenarioErrors asserts that a non-nil proto whose
// active scenario is not chi-squared yields a nil model and the exact
// "invalid model scenario: expected ChiSquared, got %v" error naming the found
// scenario, for each of the Traffic/Volumetric/Temporal scenarios.
//
// Requirements: 1.3.
func TestNewChiSquaredModelWrongScenarioErrors(t *testing.T) {
	tests := []struct {
		name string
		pb   *modelpb.Model
	}{
		{
			name: "traffic deviation",
			pb: (&modelpb.Model_builder{
				TrafficDeviation: (&modelpb.TrafficDeviationScenario_builder{}).Build(),
			}).Build(),
		},
		{
			name: "volumetric spike",
			pb: (&modelpb.Model_builder{
				VolumetricSpike: (&modelpb.VolumetricSpikeScenario_builder{}).Build(),
			}).Build(),
		},
		{
			name: "temporal deviation",
			pb: (&modelpb.Model_builder{
				TemporalDeviation: (&modelpb.TemporalDeviationScenario_builder{}).Build(),
			}).Build(),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// The error names the scenario that was found via WhichScenario,
			// formatted with %v exactly as the constructor does.
			wantErr := fmt.Sprintf("invalid model scenario: expected ChiSquared, got %v", tt.pb.WhichScenario())

			model, err := NewChiSquaredModel(tt.pb)
			if model != nil {
				t.Errorf("expected nil model for wrong scenario, got %v", model)
			}
			if err == nil {
				t.Fatalf("expected non-nil error for wrong scenario, got nil")
			}
			if err.Error() != wantErr {
				t.Errorf("error text mismatch:\n got %q\nwant %q", err.Error(), wantErr)
			}
		})
	}
}

// TestNewChiSquaredModelInvalidParameterErrors asserts that each invalid
// parameter combination yields a nil model and the specific validation error
// text from validateConfig: M<=0, N<=0, N>=M, and a threshold that is not a
// finite value greater than zero (NaN, +Inf, -Inf, 0, negative).
//
// Requirements: 1.3, 2.1, 2.2, 2.3, 2.4.
func TestNewChiSquaredModelInvalidParameterErrors(t *testing.T) {
	tests := []struct {
		name      string
		m         int64
		n         int64
		threshold float64
		wantErr   string
	}{
		// M <= 0 (checked before N and threshold).
		{name: "M zero", m: 0, n: 1, threshold: 1.0, wantErr: "M must be greater than zero"},
		{name: "M negative", m: -5, n: 1, threshold: 1.0, wantErr: "M must be greater than zero"},

		// N <= 0 (M valid, checked before N>=M and threshold).
		{name: "N zero", m: 10, n: 0, threshold: 1.0, wantErr: "N must be greater than zero"},
		{name: "N negative", m: 10, n: -3, threshold: 1.0, wantErr: "N must be greater than zero"},

		// N >= M (M and N positive, checked before threshold).
		{name: "N equals M", m: 5, n: 5, threshold: 1.0, wantErr: "M must be greater than N"},
		{name: "N greater than M", m: 5, n: 8, threshold: 1.0, wantErr: "M must be greater than N"},

		// Threshold not a finite value > 0 (M/N valid).
		{name: "threshold NaN", m: 10, n: 2, threshold: math.NaN(), wantErr: "anomaly threshold must be a finite value greater than zero"},
		{name: "threshold +Inf", m: 10, n: 2, threshold: math.Inf(1), wantErr: "anomaly threshold must be a finite value greater than zero"},
		{name: "threshold -Inf", m: 10, n: 2, threshold: math.Inf(-1), wantErr: "anomaly threshold must be a finite value greater than zero"},
		{name: "threshold zero", m: 10, n: 2, threshold: 0.0, wantErr: "anomaly threshold must be a finite value greater than zero"},
		{name: "threshold negative", m: 10, n: 2, threshold: -1.5, wantErr: "anomaly threshold must be a finite value greater than zero"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pb := chiSquaredModelProto(tt.m, tt.n, tt.threshold)

			model, err := NewChiSquaredModel(pb)
			if model != nil {
				t.Errorf("expected nil model for invalid config, got %v", model)
			}
			if err == nil {
				t.Fatalf("expected non-nil error for invalid config, got nil")
			}
			if err.Error() != tt.wantErr {
				t.Errorf("error text mismatch:\n got %q\nwant %q", err.Error(), tt.wantErr)
			}
		})
	}
}

// TestNewChiSquaredModelNilProto asserts that a nil proto yields a non-nil model
// with a nil error and both windows initialized empty (non-nil, size 0), matching
// the nil-proto contract of the other prediction models.
//
// Requirements: 1.4, 1.6.
func TestNewChiSquaredModelNilProto(t *testing.T) {
	model, err := NewChiSquaredModel(nil)
	if err != nil {
		t.Fatalf("expected nil error for nil proto, got %v", err)
	}
	if model == nil {
		t.Fatalf("expected non-nil model for nil proto, got nil")
	}

	if model.historical == nil {
		t.Fatalf("expected non-nil historical window for nil proto")
	}
	if model.recent == nil {
		t.Fatalf("expected non-nil recent window for nil proto")
	}
	if model.historical.size != 0 {
		t.Errorf("expected empty historical window, got size %d", model.historical.size)
	}
	if model.recent.size != 0 {
		t.Errorf("expected empty recent window, got size %d", model.recent.size)
	}
}

// TestChiSquaredStatisticKnownValue cross-checks chiSquaredStatistic against an
// independently hand-computed 2xK example. The historical window holds
// {A:90, B:10} (row total 100) and the recent window holds {A:5, B:5} (row
// total 10), for a grand total of 110. The reference statistic is recomputed
// inline from the straightforward (obs-exp)^2/exp formula (not a rounded
// literal) so the test is a genuine cross-check of the model's implementation.
//
// Expected counts:
//   column A (colTotal 95): expH = 100*95/110, expR = 10*95/110
//   column B (colTotal 15): expH = 100*15/110, expR = 10*15/110
// statistic = sum over both cells of both columns of (obs-exp)^2/exp.
//
// Requirements: 5.1.
func TestChiSquaredStatisticKnownValue(t *testing.T) {
	// Build the two windows with capacities large enough to hold the contents
	// without eviction: historical holds 100 Actions, recent holds 10.
	hist := newWindow(100)
	for i := 0; i < 90; i++ {
		hist.append("A")
	}
	for i := 0; i < 10; i++ {
		hist.append("B")
	}

	recent := newWindow(10)
	for i := 0; i < 5; i++ {
		recent.append("A")
	}
	for i := 0; i < 5; i++ {
		recent.append("B")
	}

	// Independent reference computation using plain float64 arithmetic, written
	// straightforwardly cell-by-cell rather than reusing the model's loop.
	const (
		obsHA, obsHB = 90.0, 10.0 // historical observed counts
		obsRA, obsRB = 5.0, 5.0   // recent observed counts
	)
	histTotal := obsHA + obsHB // 100
	recTotal := obsRA + obsRB  // 10
	grand := histTotal + recTotal

	colA := obsHA + obsRA // 95
	colB := obsHB + obsRB // 15

	term := func(obs, rowTotal, colTotal float64) float64 {
		exp := rowTotal * colTotal / grand
		d := obs - exp
		return d * d / exp
	}

	want := term(obsHA, histTotal, colA) +
		term(obsRA, recTotal, colA) +
		term(obsHB, histTotal, colB) +
		term(obsRB, recTotal, colB)

	got := chiSquaredStatistic(hist, recent)

	const tol = 1e-9
	if math.Abs(got-want) > tol {
		t.Errorf("chiSquaredStatistic mismatch: got %.15g, want %.15g (|diff|=%.3g > tol=%.1g)",
			got, want, math.Abs(got-want), tol)
	}
}

// Feature: chi-squared-anomaly-detection, Property 5: Contingency table totals are exact and columns are the union of present types
func TestContingencyTableTotalsAndUnionColumns(t *testing.T) {
	// property checks a single draw: two non-empty windows built from random
	// Action streams on a small alphabet. The contingency table is materialized
	// implicitly as the two windows' count maps plus their sizes, so the
	// properties are asserted directly against that window state:
	//   - historical row total == historical.size, recent row total == recent.size
	//   - grand total == historical.size + recent.size
	//   - the column set == union of keys(historical.counts) and keys(recent.counts),
	//     which is exactly the distinct Query_Types present in at least one window
	//   - for each column the cell value is counts[key] (0 if absent), every cell
	//     is non-negative, and at least one window is nonzero for that column
	//   - an absent Query_Type has a zero cell in the window that lacks it
	property := func(histCap, recentCap uint8, histStream, recentStream []string) bool {
		// Build the two windows the same way the model does. The generator
		// guarantees both streams are non-empty, so both windows end up
		// non-empty (capacities are >= 1).
		hist := newWindow(int(histCap))
		for _, action := range histStream {
			hist.append(action)
		}
		recent := newWindow(int(recentCap))
		for _, action := range recentStream {
			recent.append(action)
		}

		// Guard: the properties below assume both windows are non-empty, which
		// the generator ensures. Bail defensively if that ever breaks.
		if hist.size == 0 || recent.size == 0 {
			t.Logf("unexpected empty window: hist.size=%d recent.size=%d", hist.size, recent.size)
			return false
		}

		// Row totals equal the window sizes, and the grand total equals their
		// sum. The row total of each window is just its size, which the FIFO
		// append maintains as min(appends, capacity).
		histRowTotal := hist.size
		recRowTotal := recent.size
		grand := histRowTotal + recRowTotal

		wantHistRow := len(histStream)
		if wantHistRow > int(histCap) {
			wantHistRow = int(histCap)
		}
		wantRecRow := len(recentStream)
		if wantRecRow > int(recentCap) {
			wantRecRow = int(recentCap)
		}
		if histRowTotal != wantHistRow {
			t.Logf("historical row total %d != size %d", histRowTotal, wantHistRow)
			return false
		}
		if recRowTotal != wantRecRow {
			t.Logf("recent row total %d != size %d", recRowTotal, wantRecRow)
			return false
		}
		if grand != histRowTotal+recRowTotal {
			t.Logf("grand total %d != sum of row totals %d", grand, histRowTotal+recRowTotal)
			return false
		}

		// The column set is exactly the union of the two windows' count-map
		// keys. Because zero-count keys are deleted on eviction, those keys are
		// precisely the distinct Query_Types present in at least one window.
		columns := make(map[string]struct{}, len(hist.counts)+len(recent.counts))
		for key := range hist.counts {
			columns[key] = struct{}{}
		}
		for key := range recent.counts {
			columns[key] = struct{}{}
		}

		// Independently derive the distinct Query_Types present in at least one
		// window from the windows' logical contents and assert set equality.
		present := make(map[string]struct{})
		for _, a := range windowContents(hist) {
			present[a] = struct{}{}
		}
		for _, a := range windowContents(recent) {
			present[a] = struct{}{}
		}
		if len(columns) != len(present) {
			t.Logf("column count %d != distinct present types %d", len(columns), len(present))
			return false
		}
		for key := range present {
			if _, ok := columns[key]; !ok {
				t.Logf("present type %q missing from columns", key)
				return false
			}
		}

		// For each column assert the cell values and column-level invariants.
		for key := range columns {
			histCell := hist.counts[key]   // 0 if absent in the historical window
			recentCell := recent.counts[key] // 0 if absent in the recent window

			// Cells are non-negative.
			if histCell < 0 || recentCell < 0 {
				t.Logf("negative cell for column %q: hist=%d recent=%d", key, histCell, recentCell)
				return false
			}

			// A column exists only where some count is nonzero (positive column
			// total), so at least one window is nonzero for every column.
			if histCell == 0 && recentCell == 0 {
				t.Logf("column %q has zero cells in both windows", key)
				return false
			}

			// An absent Query_Type has a zero cell in the window that lacks it:
			// a nonzero count iff the key is present in that window's contents.
			histPresent := false
			for _, a := range windowContents(hist) {
				if a == key {
					histPresent = true
					break
				}
			}
			recentPresent := false
			for _, a := range windowContents(recent) {
				if a == key {
					recentPresent = true
					break
				}
			}
			if histPresent != (histCell > 0) {
				t.Logf("historical cell/presence mismatch for %q: present=%v cell=%d", key, histPresent, histCell)
				return false
			}
			if recentPresent != (recentCell > 0) {
				t.Logf("recent cell/presence mismatch for %q: present=%v cell=%d", key, recentPresent, recentCell)
				return false
			}
		}

		return true
	}

	// Generator producing two non-empty Action streams over a small (~5 symbol)
	// alphabet and small capacities (>= 1), so overflow/eviction and the
	// union-of-types column set are exercised regularly.
	cfg := &quick.Config{
		MaxCount: 1000,
		Values: func(args []reflect.Value, rng *rand.Rand) {
			// Capacities in 1..16 (never zero, so windows can hold Actions).
			args[0] = reflect.ValueOf(uint8(rng.Intn(16) + 1))
			args[1] = reflect.ValueOf(uint8(rng.Intn(16) + 1))

			makeStream := func() []string {
				// Non-empty stream, length 1..40, over a ~5 symbol alphabet.
				n := rng.Intn(40) + 1
				s := make([]string, n)
				for i := range s {
					s[i] = fmt.Sprintf("q%d", rng.Intn(5))
				}
				return s
			}
			args[2] = reflect.ValueOf(makeStream())
			args[3] = reflect.ValueOf(makeStream())
		},
	}

	if err := quick.Check(property, cfg); err != nil {
		t.Error(err)
	}
}

// Feature: chi-squared-anomaly-detection, Property 6: The statistic is deterministic, order-independent, and division-safe
func TestStatisticDeterministicOrderIndependentDivisionSafe(t *testing.T) {
	// shuffled returns a copy of in with its elements permuted using rng. The
	// multiset of Actions is preserved; only the arrival order changes.
	shuffled := func(in []string, rng *rand.Rand) []string {
		out := make([]string, len(in))
		copy(out, in)
		rng.Shuffle(len(out), func(i, j int) { out[i], out[j] = out[j], out[i] })
		return out
	}

	// buildWindow appends every Action of stream into a fresh window whose
	// capacity equals the stream length, so no eviction occurs and the window's
	// final contents are exactly the whole (possibly reordered) stream. Because
	// the capacity matches the length, the window's count map depends only on
	// the multiset of Actions, not on their arrival order.
	buildWindow := func(stream []string) *window {
		w := newWindow(len(stream))
		for _, action := range stream {
			w.append(action)
		}
		return w
	}

	// property checks a single draw: two non-empty Action streams (the
	// historical and recent samples) plus a random permutation seed used to
	// rebuild the same multisets in a different arrival order.
	property := func(histStream, recentStream []string, seed int64) bool {
		// The generator guarantees both streams are non-empty; bail defensively
		// if that ever breaks, since the properties assume non-empty windows.
		if len(histStream) == 0 || len(recentStream) == 0 {
			t.Logf("unexpected empty stream: hist=%d recent=%d", len(histStream), len(recentStream))
			return false
		}

		hist := buildWindow(histStream)
		recent := buildWindow(recentStream)

		// Determinism: computing the statistic twice over the same window
		// contents yields exactly the same value (bit-for-bit) (R5.5).
		stat1 := chiSquaredStatistic(hist, recent)
		stat2 := chiSquaredStatistic(hist, recent)
		if stat1 != stat2 {
			t.Logf("non-deterministic statistic: %v != %v", stat1, stat2)
			return false
		}

		// Finiteness / division safety at the statistic level: because both
		// windows are non-empty and every column total is positive, every
		// expected count is strictly positive, so the statistic is a finite
		// real number (no NaN from 0/0, no Inf from x/0) (R5.1, R5.6).
		if math.IsNaN(stat1) || math.IsInf(stat1, 0) {
			t.Logf("statistic not finite: %v", stat1)
			return false
		}

		// Order-independence: rebuild both windows from the SAME multiset of
		// Actions in a DIFFERENT (shuffled) arrival order. Since the statistic
		// depends only on the per-Query_Type counts (sorted-union iteration),
		// the statistic over the reordered windows must be identical (R4.5, R5.5).
		rng := rand.New(rand.NewSource(seed))
		histShuffled := buildWindow(shuffled(histStream, rng))
		recentShuffled := buildWindow(shuffled(recentStream, rng))
		statShuffled := chiSquaredStatistic(histShuffled, recentShuffled)
		if statShuffled != stat1 {
			t.Logf("statistic depends on arrival order: %v (shuffled) != %v (original)", statShuffled, stat1)
			return false
		}

		// Strict positivity of every expected count used by the statistic.
		// Independently recompute expected counts over the union of keys:
		//   expH = histTotal * colTotal / grand
		//   expR = recTotal  * colTotal / grand
		// and assert each is strictly > 0 and finite. This mirrors the cells the
		// statistic divides by, confirming no division by zero (R5.2, R5.6).
		histTotal := float64(hist.size)
		recTotal := float64(recent.size)
		grand := histTotal + recTotal

		union := make(map[string]struct{}, len(hist.counts)+len(recent.counts))
		for key := range hist.counts {
			union[key] = struct{}{}
		}
		for key := range recent.counts {
			union[key] = struct{}{}
		}
		// Every present Query_Type must contribute a column (R5.2).
		if len(union) == 0 {
			t.Logf("empty column union over non-empty windows")
			return false
		}
		for key := range union {
			colTotal := float64(hist.counts[key] + recent.counts[key])
			if colTotal <= 0 {
				t.Logf("non-positive column total for %q: %v", key, colTotal)
				return false
			}
			expH := histTotal * colTotal / grand
			expR := recTotal * colTotal / grand
			if !(expH > 0) || math.IsInf(expH, 0) || math.IsNaN(expH) {
				t.Logf("non-positive/non-finite expected (historical) for %q: %v", key, expH)
				return false
			}
			if !(expR > 0) || math.IsInf(expR, 0) || math.IsNaN(expR) {
				t.Logf("non-positive/non-finite expected (recent) for %q: %v", key, expR)
				return false
			}
		}

		return true
	}

	// Generator producing two non-empty Action streams over a small (~5 symbol)
	// alphabet so the two windows' column sets overlap (shared Query_Types) and
	// the union-column path is exercised, plus a seed for the shuffle.
	cfg := &quick.Config{
		MaxCount: 1000,
		Values: func(args []reflect.Value, rng *rand.Rand) {
			makeStream := func() []string {
				// Non-empty stream, length 1..40, over a ~5 symbol alphabet.
				n := rng.Intn(40) + 1
				s := make([]string, n)
				for i := range s {
					s[i] = fmt.Sprintf("q%d", rng.Intn(5))
				}
				return s
			}
			args[0] = reflect.ValueOf(makeStream())
			args[1] = reflect.ValueOf(makeStream())
			args[2] = reflect.ValueOf(rng.Int63())
		},
	}

	if err := quick.Check(property, cfg); err != nil {
		t.Error(err)
	}
}

// Feature: chi-squared-anomaly-detection, Property 7: Identical distributions yield a zero statistic (homogeneity sanity)
func TestHomogeneityZeroStatistic(t *testing.T) {
	// property checks a single draw: a random base distribution over a small
	// alphabet (each per-Query_Type count >= 1) and two positive integer scale
	// factors sH and sR. The historical window is filled with sH copies of the
	// base distribution and the recent window with sR copies, so the two
	// windows share identical per-Query_Type proportions (their counts are
	// scalar multiples). A chi-squared test of homogeneity on two rows with
	// identical column proportions yields a statistic of exactly zero up to
	// floating-point rounding, so the statistic must be within a tight tolerance
	// of zero (R5.1, R5.2).
	property := func(base []int, sH, sR int) bool {
		// The generator guarantees a non-empty base distribution with every
		// count >= 1 and positive scale factors; bail defensively otherwise so
		// the homogeneity premise (non-empty, proportional windows) holds.
		if len(base) == 0 || sH <= 0 || sR <= 0 {
			t.Logf("unexpected degenerate draw: base=%v sH=%d sR=%d", base, sH, sR)
			return false
		}

		// fill appends scale*ci copies of key qi so the resulting window's count
		// map is exactly {qi: scale*base[i]}. Capacity equals the total count so
		// no eviction occurs and the window holds the whole scaled distribution.
		fill := func(scale int) *window {
			total := 0
			for _, c := range base {
				total += scale * c
			}
			w := newWindow(total)
			for i, c := range base {
				key := fmt.Sprintf("q%d", i)
				for j := 0; j < scale*c; j++ {
					w.append(key)
				}
			}
			return w
		}

		hist := fill(sH)
		recent := fill(sR)

		stat := chiSquaredStatistic(hist, recent)

		// Identical proportions across the two rows make every cell's observed
		// count equal its expected count, so the statistic is zero up to
		// float64 rounding.
		const tol = 1e-9
		if math.Abs(stat) > tol {
			t.Logf("non-zero statistic for proportional windows: %.15g (|stat|=%.3g > tol=%.1g, base=%v sH=%d sR=%d)",
				stat, math.Abs(stat), tol, base, sH, sR)
			return false
		}
		return true
	}

	// Generator producing a small base distribution (2..5 Query_Types, each
	// count 1..6) and two small positive scale factors (1..8), keeping the
	// scaled window sizes modest so the test stays fast.
	cfg := &quick.Config{
		MaxCount: 1000,
		Values: func(args []reflect.Value, rng *rand.Rand) {
			k := rng.Intn(4) + 2 // 2..5 distinct Query_Types
			base := make([]int, k)
			for i := range base {
				base[i] = rng.Intn(6) + 1 // 1..6
			}
			args[0] = reflect.ValueOf(base)
			args[1] = reflect.ValueOf(rng.Intn(8) + 1) // sH 1..8
			args[2] = reflect.ValueOf(rng.Intn(8) + 1) // sR 1..8
		},
	}

	if err := quick.Check(property, cfg); err != nil {
		t.Error(err)
	}
}
// Feature: chi-squared-anomaly-detection, Property 9: Confidence is bounded, deterministic, and monotonic in the statistic
func TestConfidenceBoundedDeterministicMonotonic(t *testing.T) {
	// property checks a single draw: two statistics stat1 <= stat2 (both >= 0)
	// and a threshold > 0. The confidence transform stat/(stat+threshold) must:
	//   - lie in the inclusive range [0.0, 1.0] for each statistic (R7.1),
	//   - be deterministic: the same (stat, threshold) yields the same value on
	//     repeated calls (R7.2), and
	//   - be monotonic nondecreasing in the statistic under a fixed threshold:
	//     stat1 <= stat2 implies confidence(stat1) <= confidence(stat2) (R7.3).
	property := func(stat1, stat2, threshold float64) bool {
		// The generator guarantees stat1, stat2 >= 0, stat1 <= stat2, and
		// threshold > 0; bail defensively if that premise ever breaks.
		if stat1 < 0 || stat2 < 0 || stat1 > stat2 || !(threshold > 0) {
			t.Logf("unexpected degenerate draw: stat1=%v stat2=%v threshold=%v", stat1, stat2, threshold)
			return false
		}

		c1 := confidence(stat1, threshold)
		c2 := confidence(stat2, threshold)

		// Bounded in [0, 1] (R7.1).
		for _, c := range []float64{c1, c2} {
			if math.IsNaN(c) || c < 0.0 || c > 1.0 {
				t.Logf("confidence out of [0,1]: %v (stat pair=%v,%v threshold=%v)", c, stat1, stat2, threshold)
				return false
			}
		}

		// Deterministic: a repeated call with identical inputs yields a
		// bit-for-bit identical result (R7.2).
		if confidence(stat1, threshold) != c1 || confidence(stat2, threshold) != c2 {
			t.Logf("non-deterministic confidence for stat1=%v stat2=%v threshold=%v", stat1, stat2, threshold)
			return false
		}

		// Monotonic nondecreasing in the statistic under a fixed threshold:
		// a greater statistic yields a greater-or-equal confidence (R7.3).
		if c2 < c1 {
			t.Logf("confidence decreased with larger statistic: confidence(%v)=%v > confidence(%v)=%v (threshold=%v)",
				stat1, c1, stat2, c2, threshold)
			return false
		}

		return true
	}

	// Generator producing an ordered pair of statistics (stat1 <= stat2, both
	// >= 0) and a positive threshold, spanning a wide range of magnitudes
	// (including values near zero and very large) so the transform is exercised
	// across small, moderate, and large inputs.
	cfg := &quick.Config{
		MaxCount: 1000,
		Values: func(args []reflect.Value, rng *rand.Rand) {
			// Draw statistics across several orders of magnitude: pick a random
			// scale, then a nonnegative value within that scale. Occasionally
			// emit exactly zero to pin the lower bound.
			drawStat := func() float64 {
				if rng.Intn(10) == 0 {
					return 0.0
				}
				scale := math.Pow(10, float64(rng.Intn(9)-3)) // 1e-3 .. 1e5
				return rng.Float64() * scale
			}
			a := drawStat()
			b := drawStat()
			if a > b {
				a, b = b, a // ensure stat1 <= stat2
			}

			// Threshold strictly > 0, also across a range of magnitudes.
			scale := math.Pow(10, float64(rng.Intn(9)-3)) // 1e-3 .. 1e5
			threshold := rng.Float64()*scale + 1e-9

			args[0] = reflect.ValueOf(a)
			args[1] = reflect.ValueOf(b)
			args[2] = reflect.ValueOf(threshold)
		},
	}

	if err := quick.Check(property, cfg); err != nil {
		t.Error(err)
	}
}

// Feature: chi-squared-anomaly-detection, Property 2: Predict is total and never panics
func TestPredictIsTotalAndNeverPanics(t *testing.T) {
	// property checks a single draw: a valid model config plus a random sequence
	// of prediction "operations". Each operation is one of
	//   - a nil log entry,
	//   - a well-formed entry whose normalized query template is empty, or
	//   - a well-formed entry with random identifying fields over a small
	//     alphabet (so overflow/eviction and real statistic computation are
	//     exercised).
	// Over the whole sequence every Predict call must return a non-nil
	// AnomalyFinding and must never panic (R6.1, R6.2). A panic surfaces via the
	// recover below as a failed property; a nil finding fails the property
	// directly.
	property := func(m, n int64, threshold float64, ops []predictOp) (ok bool) {
		// Wrap the whole sequence in a recover so a panic becomes a clear,
		// attributable property failure rather than crashing the test binary.
		// (quick.Check would also surface a panic on its own, but this yields a
		// more actionable message pinpointing the offending inputs.)
		defer func() {
			if r := recover(); r != nil {
				t.Logf("Predict panicked (m=%d n=%d threshold=%v): %v", m, n, threshold, r)
				ok = false
			}
		}()

		model, err := NewChiSquaredModel(chiSquaredModelProto(m, n, threshold))
		if err != nil || model == nil {
			// The generator only emits valid configs, so construction must
			// succeed; bail defensively otherwise.
			t.Logf("unexpected construction failure (m=%d n=%d threshold=%v): model=%v err=%v", m, n, threshold, model, err)
			return false
		}

		for i, op := range ops {
			var entry *modelpb.AuditLogEntry
			if !op.nilEntry {
				// modelpb aliases the same package as auditpb, so an
				// *modelpb.AuditLogEntry is the same underlying type Predict
				// accepts as *auditpb.AuditLogEntry.
				entry = &modelpb.AuditLogEntry{}
				entry.SetNormalizedQueryTemplate(op.template)
				entry.SetDbUser(op.dbUser)
				entry.SetQueryStatement(op.queryStmt)
				entry.SetDbName(op.dbName)
			}

			finding := model.Predict(entry)
			if finding == nil {
				t.Logf("Predict returned nil finding at op %d (m=%d n=%d threshold=%v, op=%+v)", i, m, n, threshold, op)
				return false
			}
		}
		return true
	}

	// Generator producing a valid config (M in 2..50, N in 1..M-1, threshold > 0)
	// and a random sequence of operations mixing nil entries, empty-template
	// entries, and well-formed entries over a small alphabet so both the
	// no-op/warm-up paths and the real statistic path are exercised.
	cfg := &quick.Config{
		MaxCount: 300,
		Values: func(args []reflect.Value, rng *rand.Rand) {
			m := int64(rng.Intn(49) + 2) // 2..50
			n := int64(rng.Intn(int(m)-1) + 1) // 1..m-1 (guarantees N < M, N >= 1)
			threshold := rng.Float64()*100 + 1e-6 // > 0, finite

			// A sequence long enough to fill the recent window and beyond, so
			// the warm-up guard clears and the statistic path runs.
			numOps := rng.Intn(120) + 1 // 1..120
			ops := make([]predictOp, numOps)
			for i := range ops {
				switch rng.Intn(5) {
				case 0:
					// nil log entry
					ops[i] = predictOp{nilEntry: true}
				case 1:
					// well-formed entry with an empty normalized query template
					// (empty Action -> window no-op), other fields random.
					ops[i] = predictOp{
						template:  "",
						dbUser:    fmt.Sprintf("u%d", rng.Intn(3)),
						queryStmt: fmt.Sprintf("s%d", rng.Intn(3)),
						dbName:    fmt.Sprintf("d%d", rng.Intn(3)),
					}
				default:
					// well-formed entry with a non-empty Action over a small
					// alphabet, plus random identifying fields.
					ops[i] = predictOp{
						template:  fmt.Sprintf("q%d", rng.Intn(5)),
						dbUser:    fmt.Sprintf("u%d", rng.Intn(3)),
						queryStmt: fmt.Sprintf("s%d", rng.Intn(3)),
						dbName:    fmt.Sprintf("d%d", rng.Intn(3)),
					}
				}
			}

			args[0] = reflect.ValueOf(m)
			args[1] = reflect.ValueOf(n)
			args[2] = reflect.ValueOf(threshold)
			args[3] = reflect.ValueOf(ops)
		},
	}

	if err := quick.Check(property, cfg); err != nil {
		t.Error(err)
	}
}

// predictOp describes one Predict invocation for the totality property test: a
// nil entry, or a well-formed entry with the given fields (an empty template
// models an entry with an empty Action / empty field).
type predictOp struct {
	nilEntry  bool
	template  string
	dbUser    string
	queryStmt string
	dbName    string
}

// Feature: chi-squared-anomaly-detection, Property 3: Empty and failed extractions are window no-ops
func TestEmptyAndFailedExtractionsAreWindowNoOps(t *testing.T) {
	// interleavedOp describes one entry in the interleaved stream: either a
	// no-op entry (a nil log entry whose extraction fails, or a well-formed
	// entry with an empty normalized query template -> empty Action), or a
	// real entry carrying a non-empty Action.
	type interleavedOp struct {
		noop   bool   // true -> this op is a window no-op (nil entry or empty Action)
		nilOp  bool   // when noop: true -> nil entry (extraction fails), false -> empty Action
		action string // when !noop: the non-empty Action (normalized query template)
	}

	// buildEntry turns an op into the *modelpb.AuditLogEntry fed to Predict.
	// modelpb aliases the same package as auditpb, so an *modelpb.AuditLogEntry
	// is the same underlying type Predict accepts as *auditpb.AuditLogEntry.
	buildEntry := func(op interleavedOp) *modelpb.AuditLogEntry {
		if op.noop && op.nilOp {
			return nil // extraction fails -> window no-op (R6.5)
		}
		entry := &modelpb.AuditLogEntry{}
		if op.noop {
			entry.SetNormalizedQueryTemplate("") // empty Action -> explicit no-op (R3.5)
		} else {
			entry.SetNormalizedQueryTemplate(op.action)
		}
		return entry
	}

	// property checks a single draw: a valid model config, a base stream of
	// non-empty Actions, and an interleaved stream that is the base stream with
	// nil/empty-Action no-op entries inserted at arbitrary positions. Model A
	// processes the interleaved stream; model B (same config) processes only the
	// non-empty (stripped) stream. Since no-op entries never touch the windows,
	// after processing both models must have identical window contents, and the
	// sequence of (anomaly, confidence) results captured on the non-empty ops of
	// the interleaved stream must match model B's result sequence elementwise
	// (R3.5, R6.5).
	property := func(m, n int64, interleaved []interleavedOp) bool {
		modelA, errA := NewChiSquaredModel(chiSquaredModelProto(m, n, 1.0))
		modelB, errB := NewChiSquaredModel(chiSquaredModelProto(m, n, 1.0))
		if errA != nil || modelA == nil || errB != nil || modelB == nil {
			// The generator only emits valid configs, so construction must
			// succeed; bail defensively otherwise.
			t.Logf("unexpected construction failure (m=%d n=%d): errA=%v errB=%v", m, n, errA, errB)
			return false
		}

		// result captures the decision an entry's finding carries, so the two
		// models' computed-statistic sequences can be compared elementwise.
		type result struct {
			anomaly    bool
			confidence float64
		}

		// Feed model A the interleaved stream, recording the result for every
		// non-empty (non-no-op) op only, since no-op ops never change state and
		// never contribute a computed statistic.
		var gotA []result
		for _, op := range interleaved {
			finding := modelA.Predict(buildEntry(op))
			if finding == nil {
				t.Logf("Predict returned nil finding on interleaved stream")
				return false
			}
			if !op.noop {
				gotA = append(gotA, result{
					anomaly:    finding.GetIsAnomalousPredict(),
					confidence: finding.GetConfidenceScore(),
				})
			}
		}

		// Feed model B only the stripped (non-empty) stream, recording every
		// result in order.
		var gotB []result
		for _, op := range interleaved {
			if op.noop {
				continue
			}
			entry := &modelpb.AuditLogEntry{}
			entry.SetNormalizedQueryTemplate(op.action)
			finding := modelB.Predict(entry)
			if finding == nil {
				t.Logf("Predict returned nil finding on stripped stream")
				return false
			}
			gotB = append(gotB, result{
				anomaly:    finding.GetIsAnomalousPredict(),
				confidence: finding.GetConfidenceScore(),
			})
		}

		// Same final window contents: no-op entries left both windows of model A
		// identical to model B's (R3.5, R6.5).
		if !reflect.DeepEqual(windowContents(modelA.historical), windowContents(modelB.historical)) {
			t.Logf("historical window mismatch:\n  A=%v\n  B=%v",
				windowContents(modelA.historical), windowContents(modelB.historical))
			return false
		}
		if !reflect.DeepEqual(windowContents(modelA.recent), windowContents(modelB.recent)) {
			t.Logf("recent window mismatch:\n  A=%v\n  B=%v",
				windowContents(modelA.recent), windowContents(modelB.recent))
			return false
		}

		// Same sequence of computed statistics: the (anomaly, confidence)
		// results captured on the non-empty ops of the interleaved stream match
		// model B's result sequence elementwise, bit-for-bit on confidence.
		if len(gotA) != len(gotB) {
			t.Logf("result-count mismatch: interleaved non-empty=%d stripped=%d", len(gotA), len(gotB))
			return false
		}
		for i := range gotA {
			if gotA[i].anomaly != gotB[i].anomaly || gotA[i].confidence != gotB[i].confidence {
				t.Logf("result mismatch at %d: A=%+v B=%+v", i, gotA[i], gotB[i])
				return false
			}
		}

		return true
	}

	// Generator producing a valid config (M in 2..40, N in 1..M-1) and an
	// interleaved stream over a small (~5 symbol) alphabet. Roughly 40% of the
	// ops are no-ops split between nil entries (extraction fails) and empty
	// Actions (explicit no-op); the rest carry a non-empty Action. The stream is
	// long enough to overflow both windows and clear the warm-up guard, so the
	// statistic path runs on the non-empty entries.
	cfg := &quick.Config{
		MaxCount: 300,
		Values: func(args []reflect.Value, rng *rand.Rand) {
			m := int64(rng.Intn(39) + 2)        // 2..40
			n := int64(rng.Intn(int(m)-1) + 1) // 1..m-1 (guarantees N < M, N >= 1)

			numOps := rng.Intn(120) + 1 // 1..120
			ops := make([]interleavedOp, numOps)
			for i := range ops {
				switch rng.Intn(5) {
				case 0:
					// nil log entry -> extraction fails -> window no-op.
					ops[i] = interleavedOp{noop: true, nilOp: true}
				case 1:
					// well-formed entry with an empty Action -> explicit no-op.
					ops[i] = interleavedOp{noop: true, nilOp: false}
				default:
					// well-formed entry with a non-empty Action over a small
					// alphabet.
					ops[i] = interleavedOp{action: fmt.Sprintf("q%d", rng.Intn(5))}
				}
			}

			args[0] = reflect.ValueOf(m)
			args[1] = reflect.ValueOf(n)
			args[2] = reflect.ValueOf(ops)
		},
	}

	if err := quick.Check(property, cfg); err != nil {
		t.Error(err)
	}
}

// Feature: chi-squared-anomaly-detection, Property 8: The anomaly decision follows the threshold with >= semantics
func TestAnomalyDecisionFollowsThreshold(t *testing.T) {
	// feedStream pushes every Action of stream through Predict (via a
	// well-formed entry carrying it as the normalized query template) and
	// returns the finding from the LAST call. modelpb aliases the same package
	// as auditpb, so an *modelpb.AuditLogEntry is the type Predict accepts.
	feedStream := func(model *ChiSquaredModel, stream []string) *modelpb.AnomalyFinding {
		var last *modelpb.AnomalyFinding
		for _, action := range stream {
			entry := &modelpb.AuditLogEntry{}
			entry.SetNormalizedQueryTemplate(action)
			last = model.Predict(entry)
		}
		return last
	}

	// property checks a single draw: a valid config whose stream is long enough
	// that, after the final Predict, the recent window is full (so the warm-up
	// guard has cleared and the finding is derived from a computed statistic).
	// The statistic the model saw on its last call is recomputed independently
	// from the model's own windows, and the finding must be anomalous iff that
	// statistic is >= the configured threshold (R5.3, R5.4).
	property := func(m, n int64, threshold float64, stream []string) bool {
		model, err := NewChiSquaredModel(chiSquaredModelProto(m, n, threshold))
		if err != nil || model == nil {
			// The generator only emits valid configs; bail defensively.
			t.Logf("unexpected construction failure (m=%d n=%d threshold=%v): model=%v err=%v", m, n, threshold, model, err)
			return false
		}

		last := feedStream(model, stream)
		if last == nil {
			t.Logf("no finding produced for stream of length %d", len(stream))
			return false
		}

		// The generator guarantees a full recent sample after the final append,
		// so the last finding is derived from a computed statistic. Recompute
		// the statistic the model would have seen from its current windows: this
		// is exactly the statistic that drove the last decision, since Predict
		// computes it from these same windows (R5.1, R5.2, R5.6).
		if model.recent.size < model.n || model.historical.size == 0 {
			t.Logf("recent window not full after stream (recent=%d n=%d hist=%d): stream too short",
				model.recent.size, model.n, model.historical.size)
			return false
		}

		stat := chiSquaredStatistic(model.historical, model.recent)
		want := stat >= model.threshold

		if last.GetIsAnomalousPredict() != want {
			t.Logf("anomaly decision mismatch: stat=%.15g threshold=%.15g want anomalous=%v got %v",
				stat, model.threshold, want, last.GetIsAnomalousPredict())
			return false
		}
		return true
	}

	// Generator producing a valid config (M in 2..40, N in 1..M-1) and a stream
	// over a small (~4 symbol) alphabet long enough to fill the recent window
	// several times over (so the warm-up guard clears). Thresholds span a wide
	// range of magnitudes so that both the anomalous (stat >= threshold) and
	// non-anomalous (stat < threshold) outcomes are produced across draws.
	cfg := &quick.Config{
		MaxCount: 1000,
		Values: func(args []reflect.Value, rng *rand.Rand) {
			m := int64(rng.Intn(39) + 2)       // 2..40
			n := int64(rng.Intn(int(m)-1) + 1) // 1..m-1 (guarantees N < M, N >= 1)

			// Threshold across magnitudes so outcomes straddle it. The
			// chi-squared statistic over a few symbols is typically O(0.1..100),
			// so a 1e-3..1e2 spread yields a healthy mix of both decisions.
			scale := math.Pow(10, float64(rng.Intn(6)-3)) // 1e-3 .. 1e2
			threshold := rng.Float64()*scale + 1e-9

			// Stream length comfortably exceeds M (and hence N), guaranteeing a
			// full recent window after the final Predict.
			streamLen := int(m) + rng.Intn(60) + int(n) + 1
			stream := make([]string, streamLen)
			for i := range stream {
				stream[i] = fmt.Sprintf("q%d", rng.Intn(4))
			}

			args[0] = reflect.ValueOf(m)
			args[1] = reflect.ValueOf(n)
			args[2] = reflect.ValueOf(threshold)
			args[3] = reflect.ValueOf(stream)
		},
	}

	if err := quick.Check(property, cfg); err != nil {
		t.Error(err)
	}

	// Exact-equality boundary sub-case (>= is inclusive): build the two windows
	// from a fixed stream, compute the statistic those windows produce, then
	// construct a model whose threshold is set EXACTLY equal to that statistic
	// and replay the same stream. Because the decision is stat >= threshold with
	// stat == threshold, the finding must be marked anomalous (R5.3). This pins
	// down the inclusive boundary that random thresholds almost never hit.
	t.Run("exact equality is anomalous", func(t *testing.T) {
		const (
			mCap = 6
			nCap = 3
		)
		// A fixed stream whose last nCap Actions differ from the historical
		// baseline enough to yield a strictly positive statistic, so the
		// threshold set equal to it is a valid finite value > 0.
		stream := []string{"a", "a", "a", "a", "b", "b", "c", "c", "c"}

		// Compute the statistic the model will see on its final Predict by
		// replaying the same stream into standalone windows of the same
		// capacities, mirroring what Predict does internally.
		hist := newWindow(mCap)
		recent := newWindow(nCap)
		for _, action := range stream {
			hist.append(action)
			recent.append(action)
		}
		if recent.size < nCap || hist.size == 0 {
			t.Fatalf("test setup error: windows not full (recent=%d hist=%d)", recent.size, hist.size)
		}
		stat := chiSquaredStatistic(hist, recent)
		if !(stat > 0) || math.IsNaN(stat) || math.IsInf(stat, 0) {
			t.Fatalf("test setup error: statistic not a finite positive value: %v", stat)
		}

		// Construct a model whose threshold is EXACTLY the computed statistic.
		model, err := NewChiSquaredModel(chiSquaredModelProto(mCap, nCap, stat))
		if err != nil || model == nil {
			t.Fatalf("unexpected construction failure for threshold=%v: model=%v err=%v", stat, model, err)
		}

		var last *modelpb.AnomalyFinding
		for _, action := range stream {
			entry := &modelpb.AuditLogEntry{}
			entry.SetNormalizedQueryTemplate(action)
			last = model.Predict(entry)
		}
		if last == nil {
			t.Fatalf("no finding produced")
		}

		// stat == threshold, so >= holds and the finding must be anomalous.
		if !last.GetIsAnomalousPredict() {
			t.Errorf("exact-equality boundary not anomalous: stat == threshold == %.15g, want anomalous, got not anomalous", stat)
		}
	})
}

// Feature: chi-squared-anomaly-detection, Property 10: Non-computed findings carry zero confidence and no anomaly
func TestNonComputedFindingsCarryZeroConfidenceAndNoAnomaly(t *testing.T) {
	// nonComputedOp describes one Predict invocation that must NOT yield a
	// computed statistic, so its finding must be not-anomalous with confidence
	// exactly 0.0:
	//   - nilEntry: a nil log entry -> extraction fails (R6.5) -> empty finding.
	//   - emptyAction: a well-formed entry with an empty normalized query
	//     template -> empty Action -> window no-op, not-anomalous, 0.0 (R7.4).
	//   - warm-up Action: a well-formed entry carrying a non-empty Action. The
	//     generator caps the TOTAL number of these across the whole sequence at
	//     N-1, so the recent window can never reach N Actions; every such
	//     Predict sees recent.size < N and returns not-anomalous, 0.0 (R4.6, R6.6).
	type nonComputedOp struct {
		nilEntry    bool
		emptyAction bool
		action      string // only used when neither nilEntry nor emptyAction
	}

	// buildEntry turns an op into the *modelpb.AuditLogEntry fed to Predict.
	// modelpb aliases the same package as auditpb, so an *modelpb.AuditLogEntry
	// is the same underlying type Predict accepts as *auditpb.AuditLogEntry.
	buildEntry := func(op nonComputedOp) *modelpb.AuditLogEntry {
		if op.nilEntry {
			return nil // extraction fails -> window no-op, empty finding (R6.5)
		}
		entry := &modelpb.AuditLogEntry{}
		if op.emptyAction {
			entry.SetNormalizedQueryTemplate("") // empty Action -> no-op (R3.5, R7.4)
		} else {
			entry.SetNormalizedQueryTemplate(op.action)
		}
		// Populate the other identifying fields with arbitrary values; they do
		// not affect the (anomaly, confidence) decision for a non-computed path.
		entry.SetDbUser("u")
		entry.SetDbName("d")
		entry.SetQueryStatement("s")
		return entry
	}

	// property checks a single draw: a valid config (M >= 2, 1 <= N < M,
	// threshold > 0) and a sequence of non-computed ops. Because the number of
	// non-empty Actions in the sequence is capped at N-1, the recent window
	// never fills, so NO call yields a computed statistic. Every returned
	// finding must therefore be not-anomalous with confidence exactly 0.0.
	property := func(m, n int64, threshold float64, ops []nonComputedOp) bool {
		model, err := NewChiSquaredModel(chiSquaredModelProto(m, n, threshold))
		if err != nil || model == nil {
			// The generator only emits valid configs; bail defensively.
			t.Logf("unexpected construction failure (m=%d n=%d threshold=%v): model=%v err=%v", m, n, threshold, model, err)
			return false
		}

		// Count non-empty Actions to assert the generator's invariant that the
		// recent window can never fill (strictly fewer than N Actions ever
		// appended). If this is ever violated, the premise of the property
		// (no computed statistic) would not hold, so fail loudly.
		nonEmpty := 0
		for _, op := range ops {
			if !op.nilEntry && !op.emptyAction {
				nonEmpty++
			}
		}
		if int64(nonEmpty) >= n {
			t.Logf("generator invariant violated: nonEmpty=%d >= N=%d (recent window could fill)", nonEmpty, n)
			return false
		}

		for i, op := range ops {
			finding := model.Predict(buildEntry(op))
			if finding == nil {
				t.Logf("Predict returned nil finding at op %d (m=%d n=%d)", i, m, n)
				return false
			}

			// Non-computed path: must be not-anomalous with confidence exactly
			// 0.0. The 0.0 is set as a literal on these paths (not computed from
			// a statistic), so an exact == comparison is correct here.
			if finding.GetIsAnomalousPredict() {
				t.Logf("non-computed finding marked anomalous at op %d (m=%d n=%d, op=%+v)", i, m, n, op)
				return false
			}
			if finding.GetConfidenceScore() != 0.0 {
				t.Logf("non-computed finding has non-zero confidence %v at op %d (m=%d n=%d, op=%+v)",
					finding.GetConfidenceScore(), i, m, n, op)
				return false
			}
		}

		// Sanity: after the whole sequence the recent window is still below
		// capacity, confirming no Predict could have taken the computed path.
		if model.recent.size >= model.n {
			t.Logf("recent window unexpectedly full after sequence: recent=%d n=%d", model.recent.size, model.n)
			return false
		}
		return true
	}

	// Generator producing a valid config (M in 2..50, N in 1..M-1, threshold > 0)
	// and a sequence that mixes nil entries, empty-Action entries, and a capped
	// number (0..N-1) of non-empty warm-up Actions. The non-empty count is
	// chosen up front and then those ops are shuffled among the no-op ops, so
	// the recent window grows to at most N-1 and never reaches a full sample.
	cfg := &quick.Config{
		MaxCount: 300,
		Values: func(args []reflect.Value, rng *rand.Rand) {
			m := int64(rng.Intn(49) + 2)       // 2..50
			n := int64(rng.Intn(int(m)-1) + 1) // 1..m-1 (guarantees N < M, N >= 1)

			// Cap non-empty Actions strictly below N so the recent window never
			// fills: pick 0..N-1 of them. (When N == 1 this is always 0.)
			maxNonEmpty := int(n) - 1
			nonEmpty := 0
			if maxNonEmpty > 0 {
				nonEmpty = rng.Intn(maxNonEmpty + 1) // 0..N-1
			}

			// A handful of no-op entries (nil / empty Action) interleaved with
			// the warm-up Actions. Keep the total modest; the property holds
			// regardless of length.
			numNoop := rng.Intn(40)
			ops := make([]nonComputedOp, 0, nonEmpty+numNoop)
			for i := 0; i < nonEmpty; i++ {
				ops = append(ops, nonComputedOp{action: fmt.Sprintf("q%d", rng.Intn(5))})
			}
			for i := 0; i < numNoop; i++ {
				if rng.Intn(2) == 0 {
					ops = append(ops, nonComputedOp{nilEntry: true})
				} else {
					ops = append(ops, nonComputedOp{emptyAction: true})
				}
			}
			// Shuffle so warm-up Actions and no-ops are interleaved arbitrarily.
			rng.Shuffle(len(ops), func(i, j int) { ops[i], ops[j] = ops[j], ops[i] })

			args[0] = reflect.ValueOf(m)
			args[1] = reflect.ValueOf(n)
			args[2] = reflect.ValueOf(rng.Float64()*100 + 1e-6) // threshold > 0, finite
			args[3] = reflect.ValueOf(ops)
		},
	}

	if err := quick.Check(property, cfg); err != nil {
		t.Error(err)
	}
}

// Feature: chi-squared-anomaly-detection, Property 11: Findings mirror the log entry's identifying fields
func TestFindingsMirrorIdentifyingFields(t *testing.T) {
	// A fixed valid config is enough: Property 11 is about the finding's
	// identifying fields, which generateAnomalyFinding copies from the log entry
	// on every non-nil-entry path (empty Action, warm-up, and computed alike),
	// independently of M/N/threshold or the anomaly decision.
	const (
		mCap = 8
		nCap = 3
	)

	// property checks a single draw: four identifying field values (db user, db
	// name, query statement, normalized query template) drawn over a small
	// alphabet that includes the empty string. A non-nil entry carrying those
	// values is fed to Predict, and the returned finding's four identifying
	// fields must equal the entry's (R6.3). The normalized query template
	// doubles as the Action, so empty templates exercise the empty-Action path
	// and non-empty templates drive Actions into the windows; the identifying
	// fields must mirror the entry on every one of those paths.
	property := func(dbUser, dbName, queryStmt, template string) bool {
		model, err := NewChiSquaredModel(chiSquaredModelProto(mCap, nCap, 1.0))
		if err != nil || model == nil {
			t.Logf("unexpected construction failure: model=%v err=%v", model, err)
			return false
		}

		// modelpb aliases the same package as auditpb, so an
		// *modelpb.AuditLogEntry is the same underlying type Predict accepts as
		// *auditpb.AuditLogEntry.
		entry := &modelpb.AuditLogEntry{}
		entry.SetDbUser(dbUser)
		entry.SetDbName(dbName)
		entry.SetQueryStatement(queryStmt)
		entry.SetNormalizedQueryTemplate(template)

		finding := model.Predict(entry)
		if finding == nil {
			t.Logf("Predict returned nil finding for entry %+v", entry)
			return false
		}

		// Every identifying field of the finding must equal the entry's (R6.3).
		if finding.GetDbUser() != entry.GetDbUser() {
			t.Logf("db user mismatch: got %q, want %q", finding.GetDbUser(), entry.GetDbUser())
			return false
		}
		if finding.GetDbName() != entry.GetDbName() {
			t.Logf("db name mismatch: got %q, want %q", finding.GetDbName(), entry.GetDbName())
			return false
		}
		if finding.GetQueryStatement() != entry.GetQueryStatement() {
			t.Logf("query statement mismatch: got %q, want %q", finding.GetQueryStatement(), entry.GetQueryStatement())
			return false
		}
		if finding.GetNormalizedQueryTemplate() != entry.GetNormalizedQueryTemplate() {
			t.Logf("normalized query template mismatch: got %q, want %q",
				finding.GetNormalizedQueryTemplate(), entry.GetNormalizedQueryTemplate())
			return false
		}
		return true
	}

	// Generator producing each identifying field over a small alphabet that
	// includes the empty string, so both empty and non-empty values (and, via
	// the template field, both the empty-Action and real-Action Predict paths)
	// are exercised.
	cfg := &quick.Config{
		MaxCount: 1000,
		Values: func(args []reflect.Value, rng *rand.Rand) {
			// drawField returns the empty string ~1/5 of the time and otherwise
			// a short value from a small alphabet.
			drawField := func(prefix string) string {
				if rng.Intn(5) == 0 {
					return ""
				}
				return fmt.Sprintf("%s%d", prefix, rng.Intn(5))
			}
			args[0] = reflect.ValueOf(drawField("u"))
			args[1] = reflect.ValueOf(drawField("d"))
			args[2] = reflect.ValueOf(drawField("s"))
			args[3] = reflect.ValueOf(drawField("q"))
		},
	}

	if err := quick.Check(property, cfg); err != nil {
		t.Error(err)
	}

	// Nil-entry sub-case: a nil log entry yields a finding whose four
	// identifying fields are all left at the empty-string default (R6.4).
	t.Run("nil entry yields empty identifying fields", func(t *testing.T) {
		model, err := NewChiSquaredModel(chiSquaredModelProto(mCap, nCap, 1.0))
		if err != nil || model == nil {
			t.Fatalf("unexpected construction failure: model=%v err=%v", model, err)
		}

		finding := model.Predict(nil)
		if finding == nil {
			t.Fatalf("Predict returned nil finding for nil entry")
		}
		if got := finding.GetDbUser(); got != "" {
			t.Errorf("nil entry db user: got %q, want empty string", got)
		}
		if got := finding.GetDbName(); got != "" {
			t.Errorf("nil entry db name: got %q, want empty string", got)
		}
		if got := finding.GetQueryStatement(); got != "" {
			t.Errorf("nil entry query statement: got %q, want empty string", got)
		}
		if got := finding.GetNormalizedQueryTemplate(); got != "" {
			t.Errorf("nil entry normalized query template: got %q, want empty string", got)
		}
	})
}

// Feature: chi-squared-anomaly-detection, Property 12: Reset returns the model to a fresh state
func TestResetReturnsModelToFreshState(t *testing.T) {
	// finding captures the decision an entry's finding carries, so a reset
	// model's result sequence can be compared elementwise against a freshly
	// constructed model's.
	type finding struct {
		anomaly    bool
		confidence float64
	}

	// feed pushes every Action of stream through Predict (via a well-formed
	// entry carrying it as the normalized query template) and returns the
	// (anomaly, confidence) decision from each call, in order. modelpb aliases
	// the same package as auditpb, so an *modelpb.AuditLogEntry is the same
	// underlying type Predict accepts as *auditpb.AuditLogEntry.
	feed := func(model *ChiSquaredModel, stream []string) []finding {
		out := make([]finding, 0, len(stream))
		for _, action := range stream {
			entry := &modelpb.AuditLogEntry{}
			entry.SetNormalizedQueryTemplate(action)
			f := model.Predict(entry)
			if f == nil {
				return nil
			}
			out = append(out, finding{
				anomaly:    f.GetIsAnomalousPredict(),
				confidence: f.GetConfidenceScore(),
			})
		}
		return out
	}

	// property checks a single draw: a valid config and an Action stream split
	// at a random index into part1 and part2.
	//   - Model A processes part1, calls ResetLiveCounts(), then processes
	//     part2, collecting a finding per part2 entry.
	//   - Model B (freshly constructed, same config) processes part2 only,
	//     collecting a finding per entry.
	// After reset, model A must behave exactly like model B on part2: the two
	// finding sequences must match elementwise (R8.1, R8.2). Both decisions run
	// the identical deterministic computation over identical window states, so
	// confidence is compared with exact == rather than a tolerance. Reset must
	// also preserve M/N/threshold (R8.2).
	property := func(m, n int64, threshold float64, stream []string, splitRaw uint32) bool {
		modelA, errA := NewChiSquaredModel(chiSquaredModelProto(m, n, threshold))
		modelB, errB := NewChiSquaredModel(chiSquaredModelProto(m, n, threshold))
		if errA != nil || modelA == nil || errB != nil || modelB == nil {
			// The generator only emits valid configs; bail defensively.
			t.Logf("unexpected construction failure (m=%d n=%d threshold=%v): errA=%v errB=%v", m, n, threshold, errA, errB)
			return false
		}

		// Split the stream at a random index in [0, len]. An empty part1 (split
		// 0) exercises reset-on-a-brand-new-model; an empty part2 exercises
		// reset with no subsequent entries.
		split := 0
		if len(stream) > 0 {
			split = int(splitRaw % uint32(len(stream)+1))
		}
		part1 := stream[:split]
		part2 := stream[split:]

		// Model A: process part1, reset, then process part2.
		if feed(modelA, part1) == nil {
			t.Logf("Predict returned nil finding on part1")
			return false
		}
		modelA.ResetLiveCounts()

		// Reset preserves the configured parameters (R8.2).
		if modelA.m != int(m) || modelA.n != int(n) || modelA.threshold != threshold {
			t.Logf("reset did not preserve config: got m=%d n=%d threshold=%v, want m=%d n=%d threshold=%v",
				modelA.m, modelA.n, modelA.threshold, m, n, threshold)
			return false
		}

		// Reset returns both windows to the empty state (R8.2).
		if modelA.historical.size != 0 || modelA.recent.size != 0 {
			t.Logf("reset left non-empty windows: historical=%d recent=%d", modelA.historical.size, modelA.recent.size)
			return false
		}

		gotA := feed(modelA, part2)
		if part2 != nil && gotA == nil {
			t.Logf("Predict returned nil finding on part2 (reset model)")
			return false
		}

		// Model B: freshly constructed, process part2 only.
		gotB := feed(modelB, part2)
		if part2 != nil && gotB == nil {
			t.Logf("Predict returned nil finding on part2 (fresh model)")
			return false
		}

		// The two finding sequences must match elementwise. Both models run the
		// identical deterministic computation on identical window states, so an
		// exact == on confidence is correct.
		if len(gotA) != len(gotB) {
			t.Logf("finding-count mismatch: reset=%d fresh=%d", len(gotA), len(gotB))
			return false
		}
		for i := range gotA {
			if gotA[i].anomaly != gotB[i].anomaly || gotA[i].confidence != gotB[i].confidence {
				t.Logf("finding mismatch at %d: reset=%+v fresh=%+v", i, gotA[i], gotB[i])
				return false
			}
		}

		// Reset on an already-empty model must not panic and must leave both
		// windows empty (R8.3). Model A's windows are empty here only if part2
		// contributed no Actions; reset again regardless and assert emptiness.
		modelA.ResetLiveCounts()
		modelA.ResetLiveCounts()
		if modelA.historical.size != 0 || modelA.recent.size != 0 {
			t.Logf("double reset left non-empty windows: historical=%d recent=%d", modelA.historical.size, modelA.recent.size)
			return false
		}

		return true
	}

	// Generator producing a valid config (M in 2..40, N in 1..M-1, threshold > 0)
	// and an Action stream over a small (~5 symbol) alphabet long enough to
	// overflow both windows and clear the warm-up guard on each part, so the
	// computed-statistic path is exercised after reset. The split index is drawn
	// as a raw uint32 and reduced modulo len+1 inside the property, so every
	// split position (including the empty-part1 and empty-part2 extremes) is
	// reachable.
	cfg := &quick.Config{
		MaxCount: 300,
		Values: func(args []reflect.Value, rng *rand.Rand) {
			m := int64(rng.Intn(39) + 2)       // 2..40
			n := int64(rng.Intn(int(m)-1) + 1) // 1..m-1 (guarantees N < M, N >= 1)
			threshold := rng.Float64()*100 + 1e-6

			// Stream length comfortably exceeds M so each part can fill the
			// windows, sometimes short so reset-on-empty extremes are hit.
			streamLen := int(m) + rng.Intn(80)
			stream := make([]string, streamLen)
			for i := range stream {
				stream[i] = fmt.Sprintf("q%d", rng.Intn(5))
			}

			args[0] = reflect.ValueOf(m)
			args[1] = reflect.ValueOf(n)
			args[2] = reflect.ValueOf(threshold)
			args[3] = reflect.ValueOf(stream)
			args[4] = reflect.ValueOf(rng.Uint32())
		},
	}

	if err := quick.Check(property, cfg); err != nil {
		t.Error(err)
	}

	// Reset-on-a-brand-new-empty-model sub-case: calling ResetLiveCounts on a
	// freshly constructed model (both windows already empty) must not panic and
	// must leave both windows empty (R8.3). This pins the already-empty case
	// directly rather than relying on a random split landing at index 0.
	t.Run("reset on brand-new empty model does not panic", func(t *testing.T) {
		model, err := NewChiSquaredModel(chiSquaredModelProto(10, 3, 1.0))
		if err != nil || model == nil {
			t.Fatalf("unexpected construction failure: model=%v err=%v", model, err)
		}

		// Two resets in a row on an untouched model.
		model.ResetLiveCounts()
		model.ResetLiveCounts()

		if model.historical.size != 0 || model.recent.size != 0 {
			t.Errorf("reset on empty model left non-empty windows: historical=%d recent=%d",
				model.historical.size, model.recent.size)
		}
		// Config preserved.
		if model.m != 10 || model.n != 3 || model.threshold != 1.0 {
			t.Errorf("reset on empty model changed config: got m=%d n=%d threshold=%v, want m=10 n=3 threshold=1",
				model.m, model.n, model.threshold)
		}
	})
}
