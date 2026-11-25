package service

import (
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"sort"
	"strings"
	"testing"
	"testing/quick"

	"google.golang.org/protobuf/proto"

	modelpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common/proto"
)

// sumCounts returns the sum of all values in a count map.
func sumCounts(counts map[string]int) int {
	sum := 0
	for _, c := range counts {
		sum += c
	}
	return sum
}

// checkTestWindowInvariants asserts the structural invariants a testWindow must
// always satisfy: 0 <= size <= capacity, sum(counts) == size, every count is
// >= 1, and no zero-count keys linger. It returns false and logs on the first
// violation so callers can fail the enclosing property.
func checkTestWindowInvariants(t *testing.T, w *testWindow) bool {
	if w.size < 0 || w.size > w.capacity {
		t.Logf("size out of range: size=%d capacity=%d", w.size, w.capacity)
		return false
	}
	if got := sumCounts(w.counts); got != w.size {
		t.Logf("sum(counts)=%d != size=%d", got, w.size)
		return false
	}
	for key, c := range w.counts {
		if c < 1 {
			t.Logf("non-positive count for key %q: %d", key, c)
			return false
		}
	}
	return true
}

// Feature: consecutive-window-anomaly-detection, Property 4: Consecutive windows partition the stream into disjoint W-blocks
func TestTestWindowFillAndClearSemantics(t *testing.T) {
	// property checks a single (capacity, stream) draw. It appends the stream
	// one Action at a time up to capacity, asserting the buffer grows to exactly
	// W and becomes full() only at the W-th Action (and only for W > 0), that
	// clear() resets the buffer to an empty, fresh count map, and that the
	// fill-and-count invariants hold throughout. A zero-capacity window must
	// never report full(), which is what lets the nil-proto model skip every
	// Window_Boundary.
	property := func(rawCap uint8, stream []string) bool {
		// Constrain capacity to a small, non-negative range including zero so
		// the zero-capacity (nil-proto) path is exercised alongside positive W.
		capacity := int(rawCap % 17) // 0..16

		w := newTestWindow(capacity)

		// A freshly constructed window is empty with an allocated count map and
		// is never full (even at capacity 0, and before any append at W > 0).
		if w.size != 0 {
			t.Logf("fresh window not empty: size=%d", w.size)
			return false
		}
		if w.counts == nil {
			t.Logf("fresh window has nil count map")
			return false
		}
		if w.full() {
			t.Logf("fresh window reports full (capacity=%d)", capacity)
			return false
		}
		if !checkTestWindowInvariants(t, w) {
			return false
		}

		// Append up to capacity Actions one at a time, checking size growth and
		// the full() boundary at each step. We only drive the buffer to exactly
		// capacity because a testWindow fills to W and is then cleared; it never
		// grows past W in the model's fill-then-clear discipline.
		n := len(stream)
		if n > capacity {
			n = capacity
		}
		for i := 0; i < n; i++ {
			w.append(stream[i])

			// Size grows by exactly one per append.
			if w.size != i+1 {
				t.Logf("size after %d appends = %d, want %d", i+1, w.size, i+1)
				return false
			}

			// full() is true exactly when the buffer has reached a positive
			// capacity. For capacity == 0 the loop body never runs; for
			// capacity > 0 it is full only at the final (W-th) append.
			wantFull := capacity > 0 && w.size == capacity
			if w.full() != wantFull {
				t.Logf("full()=%v, want %v (size=%d capacity=%d)", w.full(), wantFull, w.size, capacity)
				return false
			}

			if !checkTestWindowInvariants(t, w) {
				return false
			}
		}

		// After filling a positive-capacity window to exactly W, it must be
		// full and hold exactly W Actions; the count map must account for every
		// one of them.
		if capacity > 0 && n == capacity {
			if !w.full() {
				t.Logf("window not full after reaching capacity %d", capacity)
				return false
			}
			if w.size != capacity {
				t.Logf("size=%d after filling to capacity %d", w.size, capacity)
				return false
			}
			expected := countsOf(stream[:n])
			if len(expected) != len(w.counts) {
				t.Logf("distinct-key mismatch: counts=%d, contents=%d", len(w.counts), len(expected))
				return false
			}
			for key, c := range expected {
				if w.counts[key] != c {
					t.Logf("count mismatch for %q: counts=%d, contents=%d", key, w.counts[key], c)
					return false
				}
			}
		}

		// A zero-capacity window is never full regardless of how the state
		// looks after the (no-op) fill loop.
		if capacity == 0 && w.full() {
			t.Logf("zero-capacity window reports full")
			return false
		}

		// clear() resets the buffer to zero with a fresh, empty, non-nil count
		// map, and a cleared window is never full (R3.3).
		w.clear()
		if w.size != 0 {
			t.Logf("size after clear = %d, want 0", w.size)
			return false
		}
		if w.counts == nil {
			t.Logf("count map is nil after clear")
			return false
		}
		if len(w.counts) != 0 {
			t.Logf("count map not empty after clear: %d keys", len(w.counts))
			return false
		}
		if w.full() {
			t.Logf("window reports full after clear (capacity=%d)", capacity)
			return false
		}
		if !checkTestWindowInvariants(t, w) {
			return false
		}

		return true
	}

	// Generator biases toward a small Action alphabet so the count map holds
	// repeated Query_Types (exercising the sum-of-counts and no-zero-key
	// invariants) and draws capacities including zero so the nil-proto path is
	// covered.
	cfg := &quick.Config{
		MaxCount: 1000,
		Values: func(args []reflect.Value, rng *rand.Rand) {
			args[0] = reflect.ValueOf(uint8(rng.Intn(256)))

			nStream := rng.Intn(40) // 0..39 candidate Actions
			stream := make([]string, nStream)
			for i := range stream {
				stream[i] = fmt.Sprintf("q%d", rng.Intn(5))
			}
			args[1] = reflect.ValueOf(stream)
		},
	}

	if err := quick.Check(property, cfg); err != nil {
		t.Error(err)
	}
}

// Feature: consecutive-window-anomaly-detection, Property 5: The baseline is a FIFO of whole completed windows capped at B
func TestBaselineFIFOWholeBlockSemantics(t *testing.T) {
	// property folds a long stream of completed W-blocks into a baseline capped
	// at B = kW (k >= 2), enough to overflow and then churn the FIFO many times
	// over. After each fold it asserts the whole-block cap, growth-vs-eviction
	// behavior, FIFO retention of the most-recent B/W blocks, and the aggregate
	// invariants (sum(agg) == size, every count >= 1, no zero-count keys).
	property := func(rawW uint8, rawK uint8, rawExtra uint8, seed int64) bool {
		// W in 1..8; k in 2..6 so B = kW is a multiple of W strictly greater
		// than W; B/W is the steady-state block count.
		w := int(rawW%8) + 1
		k := int(rawK%5) + 2
		b := k * w
		maxBlocks := b / w // == k

		// Fold more blocks than fit so the FIFO overflows and churns: a few
		// multiples of the capacity plus a random extra tail.
		numBlocks := maxBlocks*3 + int(rawExtra%7)

		rng := rand.New(rand.NewSource(seed))

		bw := newBaselineWindow(w, b)

		// A fresh baseline is empty with an allocated, empty aggregate map.
		if bw.size != 0 || len(bw.blocks) != 0 {
			t.Logf("fresh baseline not empty: size=%d blocks=%d", bw.size, len(bw.blocks))
			return false
		}
		if bw.agg == nil {
			t.Logf("fresh baseline has nil agg map")
			return false
		}

		// Keep an independent model of the blocks we have folded, each as the
		// exact count map of a completed W-block, so we can check FIFO retention
		// and the aggregate counts against a ground-truth recomputation.
		var folded []map[string]int

		for blockIdx := 0; blockIdx < numBlocks; blockIdx++ {
			sizeBefore := bw.size

			// Build a completed W-block: exactly W Actions drawn from a small
			// alphabet so blocks share Query_Types (exercising aggregate
			// addition and eviction-to-zero key deletion).
			actions := make([]string, w)
			for i := range actions {
				actions[i] = fmt.Sprintf("q%d", rng.Intn(4))
			}
			blockCounts := countsOf(actions)

			// Record a defensive copy in our ground-truth FIFO, then fold.
			folded = append(folded, copyCounts(blockCounts))
			bw.fold(blockCounts)

			// size is always an exact multiple of W and never exceeds B.
			if bw.size%w != 0 {
				t.Logf("size=%d not a multiple of W=%d", bw.size, w)
				return false
			}
			if bw.size < 0 || bw.size > b {
				t.Logf("size=%d out of range [0, %d]", bw.size, b)
				return false
			}
			// The block FIFO and size agree: size == len(blocks) * W.
			if bw.size != len(bw.blocks)*w {
				t.Logf("size=%d != len(blocks)=%d * W=%d", bw.size, len(bw.blocks), w)
				return false
			}

			if sizeBefore < b {
				// Below capacity: folding grows the baseline by exactly W with
				// no eviction.
				if bw.size != sizeBefore+w {
					t.Logf("below-cap fold grew by %d, want %d", bw.size-sizeBefore, w)
					return false
				}
			} else {
				// At capacity: folding evicts exactly one oldest block, so the
				// baseline stays at exactly B Actions and B/W blocks.
				if bw.size != b {
					t.Logf("full baseline size=%d, want %d", bw.size, b)
					return false
				}
				if len(bw.blocks) != maxBlocks {
					t.Logf("full baseline holds %d blocks, want %d", len(bw.blocks), maxBlocks)
					return false
				}
			}

			// The retained blocks are the most recent B/W folded blocks, in
			// arrival (oldest-to-newest) order.
			retained := folded
			if len(retained) > maxBlocks {
				retained = retained[len(retained)-maxBlocks:]
			}
			if len(bw.blocks) != len(retained) {
				t.Logf("retained-block count=%d, want %d", len(bw.blocks), len(retained))
				return false
			}
			for i := range retained {
				if !reflect.DeepEqual(bw.blocks[i], retained[i]) {
					t.Logf("retained block %d mismatch: got %v want %v", i, bw.blocks[i], retained[i])
					return false
				}
			}

			// The aggregate equals the element-wise sum of the retained blocks,
			// sum(agg) == size, every agg count is >= 1, and no zero-count keys
			// linger (the basis for positive column totals / division safety).
			wantAgg := make(map[string]int)
			for _, blk := range retained {
				for key, c := range blk {
					wantAgg[key] += c
				}
			}
			if !reflect.DeepEqual(bw.agg, wantAgg) {
				t.Logf("agg mismatch: got %v want %v", bw.agg, wantAgg)
				return false
			}
			if got := sumCounts(bw.agg); got != bw.size {
				t.Logf("sum(agg)=%d != size=%d", got, bw.size)
				return false
			}
			for key, c := range bw.agg {
				if c < 1 {
					t.Logf("non-positive agg count for key %q: %d", key, c)
					return false
				}
			}
		}

		return true
	}

	cfg := &quick.Config{
		MaxCount: 500,
		Values: func(args []reflect.Value, rng *rand.Rand) {
			args[0] = reflect.ValueOf(uint8(rng.Intn(256)))
			args[1] = reflect.ValueOf(uint8(rng.Intn(256)))
			args[2] = reflect.ValueOf(uint8(rng.Intn(256)))
			args[3] = reflect.ValueOf(rng.Int63())
		},
	}

	if err := quick.Check(property, cfg); err != nil {
		t.Error(err)
	}
}

// copyCounts returns an independent copy of a count map so the ground-truth FIFO
// in the baseline property is unaffected by any later mutation of the source.
func copyCounts(counts map[string]int) map[string]int {
	out := make(map[string]int, len(counts))
	for key, c := range counts {
		out[key] = c
	}
	return out
}

// Feature: consecutive-window-anomaly-detection, Property 1: Construction validates parameters and leaves the proto unchanged
func TestConsecutiveConstructionValidatesParametersAndLeavesProtoUnchanged(t *testing.T) {
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

	// consecutiveProto builds a consecutive-window Model proto from
	// (w, b, threshold) using the generated opaque builder API. The parameter
	// values are not validated here; callers pass invalid values on purpose to
	// exercise the constructor's validation branches.
	consecutiveProto := func(w, b int64, threshold float64) *modelpb.Model {
		cfg := (&modelpb.ConsecutiveWindowTrainingConfig_builder{
			W:                w,
			B:                b,
			AnomalyThreshold: threshold,
		}).Build()
		scenario := (&modelpb.ConsecutiveWindowScenario_builder{Config: cfg}).Build()
		return (&modelpb.Model_builder{ConsecutiveWindow: scenario}).Build()
	}

	// wantErrSubstring independently computes the error substring the
	// constructor must report for a given (w, b, threshold) config, following
	// the exact first-violated-constraint order W>0, B>0, B>W, B%W==0, threshold
	// (R2.7). It returns "" when the config is valid. b%w is only evaluated
	// after w > 0, mirroring the production guard against division by zero.
	wantErrSubstring := func(w, b int64, threshold float64) string {
		switch {
		case w <= 0:
			return "W must be greater than zero"
		case b <= 0:
			return "B must be greater than zero"
		case b <= w:
			return "B must be greater than W"
		case b%w != 0:
			return "B must be an integer multiple of W"
		case math.IsNaN(threshold) || math.IsInf(threshold, 0) || threshold <= 0:
			return "anomaly threshold must be a finite value greater than zero"
		default:
			return ""
		}
	}

	// property checks a single draw: construction succeeds (non-nil model with
	// both windows holding zero Actions, nil error) iff the config is valid,
	// fails (nil model, non-nil error naming the first violated constraint)
	// otherwise, and in all cases leaves the proto byte-for-byte unchanged
	// (whether the call succeeded or failed).
	property := func(pb *modelpb.Model, wantSubstr string) bool {
		before := proto.Clone(pb)

		model, err := NewConsecutiveWindowModel(pb)

		if wantSubstr == "" {
			// Valid config: non-nil model, nil error, and both windows empty.
			if model == nil || err != nil {
				t.Logf("valid config rejected: model=%v, err=%v", model, err)
				return false
			}
			if model.test == nil || model.baseline == nil {
				t.Logf("valid config left a nil window: test=%v baseline=%v", model.test, model.baseline)
				return false
			}
			if model.test.size != 0 || model.baseline.size != 0 {
				t.Logf("fresh model windows not empty: test.size=%d baseline.size=%d",
					model.test.size, model.baseline.size)
				return false
			}
		} else {
			// Invalid / wrong-scenario config: nil model, non-nil error whose
			// text names the first violated constraint.
			if model != nil || err == nil {
				t.Logf("invalid config accepted: model=%v, err=%v", model, err)
				return false
			}
			if !strings.Contains(err.Error(), wantSubstr) {
				t.Logf("error text mismatch: got %q, want substring %q", err.Error(), wantSubstr)
				return false
			}
		}

		// The proto must be unchanged after the call, success or failure (R1.5).
		if !proto.Equal(before, pb) {
			t.Logf("proto mutated by NewConsecutiveWindowModel: before=%v, after=%v", before, pb)
			return false
		}
		return true
	}

	cfg := &quick.Config{
		MaxCount: 1000,
		Values: func(args []reflect.Value, rng *rand.Rand) {
			switch rng.Intn(5) {
			case 0:
				// Small boundary-ish W/B (including 0 and negatives) with an
				// interesting threshold, so every validation branch — W<=0,
				// B<=0, B<=W, B%W!=0, and the threshold checks — is reachable.
				w := int64(rng.Intn(11) - 2) // -2..8
				b := int64(rng.Intn(21) - 2) // -2..18
				threshold := interestingThresholds[rng.Intn(len(interestingThresholds))]
				pb := consecutiveProto(w, b, threshold)
				args[0] = reflect.ValueOf(pb)
				args[1] = reflect.ValueOf(wantErrSubstring(w, b, threshold))
			case 1:
				// Guaranteed-valid region: W >= 1, B = kW with k >= 2, finite
				// positive threshold. Exercises the success path heavily.
				w := int64(rng.Intn(8) + 1) // 1..8
				k := int64(rng.Intn(5) + 2) // 2..6
				b := k * w                  // multiple of W, strictly > W
				threshold := rng.Float64()*100 + 1e-6
				pb := consecutiveProto(w, b, threshold)
				args[0] = reflect.ValueOf(pb)
				args[1] = reflect.ValueOf(wantErrSubstring(w, b, threshold))
			case 2:
				// B a positive multiple-ish of W but deliberately NOT a multiple
				// (B > W, B%W != 0), pinning the B%W==0 branch, combined with an
				// interesting threshold to occasionally let an earlier branch
				// win and confirm ordering.
				w := int64(rng.Intn(6) + 2)     // 2..7
				b := w*int64(rng.Intn(4)+2) + 1 // k*W + 1, so > W and not a multiple
				threshold := interestingThresholds[rng.Intn(len(interestingThresholds))]
				pb := consecutiveProto(w, b, threshold)
				args[0] = reflect.ValueOf(pb)
				args[1] = reflect.ValueOf(wantErrSubstring(w, b, threshold))
			case 3:
				// Multi-violation config: several constraints fail at once so
				// the test pins that the FIRST violated constraint is reported.
				// e.g. w<=0 AND b<=0 AND bad threshold -> must report W.
				w := int64(-rng.Intn(5)) // -4..0
				b := int64(-rng.Intn(5)) // -4..0
				threshold := math.NaN()
				pb := consecutiveProto(w, b, threshold)
				args[0] = reflect.ValueOf(pb)
				args[1] = reflect.ValueOf(wantErrSubstring(w, b, threshold))
			default:
				// Wrong-scenario protos: a different scenario (or none) is set,
				// which the constructor rejects before reading any config.
				var pb *modelpb.Model
				switch rng.Intn(3) {
				case 0:
					scenario := (&modelpb.ChiSquaredScenario_builder{}).Build()
					pb = (&modelpb.Model_builder{ChiSquared: scenario}).Build()
				case 1:
					scenario := (&modelpb.VolumetricSpikeScenario_builder{}).Build()
					pb = (&modelpb.Model_builder{VolumetricSpike: scenario}).Build()
				default:
					pb = (&modelpb.Model_builder{}).Build()
				}
				args[0] = reflect.ValueOf(pb)
				args[1] = reflect.ValueOf("invalid model scenario: expected ConsecutiveWindow")
			}
		},
	}

	if err := quick.Check(property, cfg); err != nil {
		t.Error(err)
	}
}

// TestConsecutiveConstructorErrorCases pins the exact constructor behavior for
// wrong-scenario protos and for each individual parameter violation, using
// hand-written examples rather than random draws. It complements the Property 1
// test (TestConsecutiveConstructionValidatesParametersAndLeavesProtoUnchanged)
// by anchoring the specific error text and the nil-proto success contract.
//
// Validates: Requirements 1.3, 1.4, 1.6, 2.1, 2.2, 2.3, 2.4, 2.5, 2.7
func TestConsecutiveConstructorErrorCases(t *testing.T) {
	// consecutiveProto builds a consecutive-window Model proto from
	// (w, b, threshold) via the generated opaque builder API. Callers pass
	// deliberately invalid values to drive the validation branches.
	consecutiveProto := func(w, b int64, threshold float64) *modelpb.Model {
		cfg := (&modelpb.ConsecutiveWindowTrainingConfig_builder{
			W:                w,
			B:                b,
			AnomalyThreshold: threshold,
		}).Build()
		scenario := (&modelpb.ConsecutiveWindowScenario_builder{Config: cfg}).Build()
		return (&modelpb.Model_builder{ConsecutiveWindow: scenario}).Build()
	}

	// Each case names the proto under test and the error substring the
	// constructor must report. A nil model is always required for the error
	// cases below.
	cases := []struct {
		name       string
		pb         *modelpb.Model
		wantSubstr string
	}{
		// Wrong-scenario protos: rejected before any config is read (R1.3).
		{
			name:       "wrong scenario traffic",
			pb:         (&modelpb.Model_builder{TrafficDeviation: (&modelpb.TrafficDeviationScenario_builder{}).Build()}).Build(),
			wantSubstr: "invalid model scenario: expected ConsecutiveWindow",
		},
		{
			name:       "wrong scenario volumetric",
			pb:         (&modelpb.Model_builder{VolumetricSpike: (&modelpb.VolumetricSpikeScenario_builder{}).Build()}).Build(),
			wantSubstr: "invalid model scenario: expected ConsecutiveWindow",
		},
		{
			name:       "wrong scenario temporal",
			pb:         (&modelpb.Model_builder{TemporalDeviation: (&modelpb.TemporalDeviationScenario_builder{}).Build()}).Build(),
			wantSubstr: "invalid model scenario: expected ConsecutiveWindow",
		},
		{
			name:       "wrong scenario chi-squared",
			pb:         (&modelpb.Model_builder{ChiSquared: (&modelpb.ChiSquaredScenario_builder{}).Build()}).Build(),
			wantSubstr: "invalid model scenario: expected ConsecutiveWindow",
		},
		// Individual parameter violations (R2.1..R2.5).
		{
			name:       "W zero",
			pb:         consecutiveProto(0, 10, 5.0),
			wantSubstr: "W must be greater than zero",
		},
		{
			name:       "W negative",
			pb:         consecutiveProto(-3, 10, 5.0),
			wantSubstr: "W must be greater than zero",
		},
		{
			name:       "B zero",
			pb:         consecutiveProto(5, 0, 5.0),
			wantSubstr: "B must be greater than zero",
		},
		{
			name:       "B negative",
			pb:         consecutiveProto(5, -4, 5.0),
			wantSubstr: "B must be greater than zero",
		},
		{
			name:       "B equal to W",
			pb:         consecutiveProto(5, 5, 5.0),
			wantSubstr: "B must be greater than W",
		},
		{
			name:       "B less than W",
			pb:         consecutiveProto(5, 3, 5.0),
			wantSubstr: "B must be greater than W",
		},
		{
			name:       "B not a multiple of W",
			pb:         consecutiveProto(5, 12, 5.0),
			wantSubstr: "B must be an integer multiple of W",
		},
		{
			name:       "threshold NaN",
			pb:         consecutiveProto(5, 10, math.NaN()),
			wantSubstr: "anomaly threshold must be a finite value greater than zero",
		},
		{
			name:       "threshold +Inf",
			pb:         consecutiveProto(5, 10, math.Inf(1)),
			wantSubstr: "anomaly threshold must be a finite value greater than zero",
		},
		{
			name:       "threshold -Inf",
			pb:         consecutiveProto(5, 10, math.Inf(-1)),
			wantSubstr: "anomaly threshold must be a finite value greater than zero",
		},
		{
			name:       "threshold zero",
			pb:         consecutiveProto(5, 10, 0.0),
			wantSubstr: "anomaly threshold must be a finite value greater than zero",
		},
		{
			name:       "threshold negative",
			pb:         consecutiveProto(5, 10, -2.5),
			wantSubstr: "anomaly threshold must be a finite value greater than zero",
		},
		// Multi-violation config: W<=0 AND B<=0 AND B<=W AND NaN threshold all
		// fail, but the constructor must report only the FIRST in order (R2.7):
		// W > 0.
		{
			name:       "multi-violation reports W first",
			pb:         consecutiveProto(-1, -1, math.NaN()),
			wantSubstr: "W must be greater than zero",
		},
		// Multi-violation config where W is valid but B<=0, B<=W, and bad
		// threshold all fail: must report B > 0 first (R2.7).
		{
			name:       "multi-violation reports B positivity first",
			pb:         consecutiveProto(5, -2, math.NaN()),
			wantSubstr: "B must be greater than zero",
		},
		// Multi-violation config where W and B>0 are valid but B<=W and the
		// threshold is bad: must report B > W first (R2.7).
		{
			name:       "multi-violation reports B greater than W first",
			pb:         consecutiveProto(8, 4, math.NaN()),
			wantSubstr: "B must be greater than W",
		},
		// Multi-violation config where W, B>0, B>W are valid but B is not a
		// multiple of W and the threshold is bad: must report the multiple
		// constraint first (R2.7).
		{
			name:       "multi-violation reports B multiple first",
			pb:         consecutiveProto(5, 13, math.NaN()),
			wantSubstr: "B must be an integer multiple of W",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			model, err := NewConsecutiveWindowModel(tc.pb)
			if model != nil {
				t.Errorf("expected nil model, got %v", model)
			}
			if err == nil {
				t.Fatalf("expected non-nil error, got nil")
			}
			if !strings.Contains(err.Error(), tc.wantSubstr) {
				t.Errorf("error text = %q, want substring %q", err.Error(), tc.wantSubstr)
			}
		})
	}

	// A nil proto yields a non-nil model with both windows empty and a nil
	// error, matching the nil-proto behavior of the other models (R1.4, R1.6).
	t.Run("nil proto yields empty model", func(t *testing.T) {
		model, err := NewConsecutiveWindowModel(nil)
		if err != nil {
			t.Fatalf("nil proto returned error: %v", err)
		}
		if model == nil {
			t.Fatalf("nil proto returned nil model")
		}
		if model.test == nil || model.baseline == nil {
			t.Fatalf("nil proto left a nil window: test=%v baseline=%v", model.test, model.baseline)
		}
		if model.test.size != 0 || model.baseline.size != 0 {
			t.Errorf("nil-proto model windows not empty: test.size=%d baseline.size=%d",
				model.test.size, model.baseline.size)
		}
	})
}

// TestConsecutiveChiSquaredStatisticKnownValue pins consecutiveChiSquaredStatistic
// against a hand-computed 2xK contingency table so an arithmetic regression in the
// statistic surfaces immediately.
//
// Baseline aggregate {A:90, B:10} (baseSize = 100) against a completed Test_Window
// {A:5, B:5} with W = 10. Grand total = 110.
//
//	Column A: colTotal = 95 -> expBaseA = 100*95/110, expTestA = 10*95/110
//	Column B: colTotal = 15 -> expBaseB = 100*15/110, expTestB = 10*15/110
//
//	stat = (90-expBaseA)^2/expBaseA + (5-expTestA)^2/expTestA
//	     + (10-expBaseB)^2/expBaseB + (5-expTestB)^2/expTestB
//
// The expected value is recomputed here from the same closed form (not the
// implementation) and compared within a tight 1e-9 absolute tolerance.
//
// Validates: Requirements 6.1
func TestConsecutiveChiSquaredStatisticKnownValue(t *testing.T) {
	baseAgg := map[string]int{"A": 90, "B": 10}
	baseSize := 100
	testCounts := map[string]int{"A": 5, "B": 5}
	w := 10

	const grand = 110.0
	const baseTotal = 100.0
	const testTotal = 10.0

	// Column A (colTotal = 95).
	expBaseA := baseTotal * 95.0 / grand
	expTestA := testTotal * 95.0 / grand
	// Column B (colTotal = 15).
	expBaseB := baseTotal * 15.0 / grand
	expTestB := testTotal * 15.0 / grand

	cell := func(obs, exp float64) float64 {
		d := obs - exp
		return d * d / exp
	}
	want := cell(90, expBaseA) + cell(5, expTestA) + cell(10, expBaseB) + cell(5, expTestB)

	got := consecutiveChiSquaredStatistic(baseAgg, baseSize, testCounts, w)

	if math.Abs(got-want) >= 1e-9 {
		t.Errorf("consecutiveChiSquaredStatistic = %.17g, want %.17g (|diff| = %g)",
			got, want, math.Abs(got-want))
	}
}

// Feature: consecutive-window-anomaly-detection, Property 6: Contingency table totals are exact and columns are the union of present types
func TestConsecutiveContingencyTableTotalsAndColumns(t *testing.T) {
	// The 2xK contingency table is never materialized as a standalone value: it
	// lives transiently inside consecutiveChiSquaredStatistic, which iterates the
	// sorted union of Query_Types in baseAgg and testCounts and reads each cell
	// from the two maps (an absent key defaulting to a zero count). This property
	// validates the table-CONSTRUCTION CONTRACT at the data-model level by
	// reconstructing the table exactly the way the statistic does and asserting
	// the structural invariants Requirement 5 places on it:
	//
	//   - the baseline row total equals the baseline size (R5.3);
	//   - the completed-window row total equals W (R5.3);
	//   - the grand total equals their sum, baseSize + W (R5.4);
	//   - the columns are exactly the distinct Query_Types present in at least
	//     one row, in a fixed total ordering (the sorted deduped union of keys)
	//     (R5.1, R5.2);
	//   - an absent Query_Type has a zero cell in the row that lacks it, via map
	//     default (R5.2).
	//
	// Inputs mirror the real window invariants: both maps are non-empty, all
	// counts are >= 1, and no zero-count keys exist; baseSize is the sum of the
	// baseline counts and W is the sum of the completed-window counts.
	property := func(baseAgg, testCounts map[string]int, baseSize, w int) bool {
		// Guard the invariants the generator is responsible for, so a bad draw
		// fails loudly rather than silently weakening the assertions.
		if len(baseAgg) == 0 || len(testCounts) == 0 {
			t.Logf("empty input map: baseAgg=%v testCounts=%v", baseAgg, testCounts)
			return false
		}
		for key, c := range baseAgg {
			if c < 1 {
				t.Logf("non-positive baseline count for %q: %d", key, c)
				return false
			}
		}
		for key, c := range testCounts {
			if c < 1 {
				t.Logf("non-positive completed-window count for %q: %d", key, c)
				return false
			}
		}
		if got := sumCounts(baseAgg); got != baseSize {
			t.Logf("sum(baseAgg)=%d != baseSize=%d", got, baseSize)
			return false
		}
		if got := sumCounts(testCounts); got != w {
			t.Logf("sum(testCounts)=%d != W=%d", got, w)
			return false
		}

		// Reconstruct the columns the same way the statistic does: the sorted
		// union of the keys of both rows.
		keySet := make(map[string]struct{}, len(baseAgg)+len(testCounts))
		for key := range baseAgg {
			keySet[key] = struct{}{}
		}
		for key := range testCounts {
			keySet[key] = struct{}{}
		}
		columns := make([]string, 0, len(keySet))
		for key := range keySet {
			columns = append(columns, key)
		}
		sort.Strings(columns)

		// Independently compute the expected column ordering: a sorted, deduped
		// union of the two key sets built without reusing the production logic.
		wantColumns := append(sortedKeys(baseAgg), sortedKeys(testCounts)...)
		sort.Strings(wantColumns)
		wantColumns = dedupeSorted(wantColumns)

		// Columns are exactly the distinct present Query_Types in fixed order.
		if !reflect.DeepEqual(columns, wantColumns) {
			t.Logf("column set/order mismatch: got %v want %v", columns, wantColumns)
			return false
		}

		// Every column corresponds to a Query_Type present in at least one row,
		// and an absent Query_Type reads as a zero cell in the row that lacks it.
		baseRowTotal := 0
		testRowTotal := 0
		for _, key := range columns {
			_, inBase := baseAgg[key]
			_, inTest := testCounts[key]
			if !inBase && !inTest {
				t.Logf("column %q present in neither row", key)
				return false
			}
			// Map default: a key absent from a row yields a zero cell there.
			if !inBase && baseAgg[key] != 0 {
				t.Logf("absent baseline cell for %q not zero: %d", key, baseAgg[key])
				return false
			}
			if !inTest && testCounts[key] != 0 {
				t.Logf("absent completed-window cell for %q not zero: %d", key, testCounts[key])
				return false
			}
			baseRowTotal += baseAgg[key]
			testRowTotal += testCounts[key]
		}

		// Row totals: the baseline row sums to baseSize and the completed-window
		// row sums to W, summed over the reconstructed columns (R5.3).
		if baseRowTotal != baseSize {
			t.Logf("baseline row total over columns=%d != baseSize=%d", baseRowTotal, baseSize)
			return false
		}
		if testRowTotal != w {
			t.Logf("completed-window row total over columns=%d != W=%d", testRowTotal, w)
			return false
		}

		// Grand total equals the sum of the two row totals (R5.4).
		grand := baseRowTotal + testRowTotal
		if grand != baseSize+w {
			t.Logf("grand total=%d != baseSize+W=%d", grand, baseSize+w)
			return false
		}

		return true
	}

	// Generator draws two non-empty count maps over a small shared alphabet so
	// the two rows overlap on some Query_Types and differ on others (exercising
	// the union, the fixed ordering, and the absent-cell default), with all
	// counts >= 1 and no zero-count keys. baseSize and W are derived as the sums
	// of the respective maps so they mirror the real window row totals.
	cfg := &quick.Config{
		MaxCount: 1000,
		Values: func(args []reflect.Value, rng *rand.Rand) {
			randomCounts := func() map[string]int {
				// 1..6 distinct Query_Types drawn from a small alphabet; each
				// present key gets a count of 1..9 (always >= 1, never zero).
				m := make(map[string]int)
				n := rng.Intn(6) + 1
				for i := 0; i < n; i++ {
					key := fmt.Sprintf("q%d", rng.Intn(8))
					m[key] = rng.Intn(9) + 1
				}
				return m
			}

			baseAgg := randomCounts()
			testCounts := randomCounts()
			args[0] = reflect.ValueOf(baseAgg)
			args[1] = reflect.ValueOf(testCounts)
			args[2] = reflect.ValueOf(sumCounts(baseAgg))
			args[3] = reflect.ValueOf(sumCounts(testCounts))
		},
	}

	if err := quick.Check(property, cfg); err != nil {
		t.Error(err)
	}
}

// sortedKeys returns the keys of a count map in ascending order. It is a small
// independent helper for the Property 6 column-ordering check so the expected
// column set is computed without reusing the statistic's own union logic.
func sortedKeys(counts map[string]int) []string {
	keys := make([]string, 0, len(counts))
	for key := range counts {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

// dedupeSorted collapses runs of equal values in an already-sorted slice,
// returning the distinct values in order. Used to build the expected deduped
// union of Query_Types independently of the production code.
func dedupeSorted(sorted []string) []string {
	if len(sorted) == 0 {
		return []string{}
	}
	out := sorted[:1]
	for _, v := range sorted[1:] {
		if v != out[len(out)-1] {
			out = append(out, v)
		}
	}
	return out
}

// Feature: consecutive-window-anomaly-detection, Property 7: The statistic is deterministic, order-independent, and division-safe
func TestConsecutiveStatisticDeterministicAndDivisionSafe(t *testing.T) {
	// property checks a single random (baseline, completed) pair that mirrors
	// the real boundary invariants: both count maps are non-empty with every
	// count >= 1 (no zero-count keys), baseSize is the sum of the baseline
	// counts, and W is the sum of the completed-window counts (both > 0). On
	// that pair it asserts four facts about consecutiveChiSquaredStatistic:
	//
	//   - the statistic is a finite float64 (neither NaN nor +/-Inf), which can
	//     only hold if no expected count was zero — division safety (R6.5, R6.6);
	//   - it is DETERMINISTIC: recomputing on the identical contents yields the
	//     bit-for-bit same float (R6.1, R6.2);
	//   - it is ORDER-INDEPENDENT: rebuilding both maps with their keys inserted
	//     in a different, shuffled order yields the bit-for-bit same float, since
	//     the statistic iterates a SORTED union of keys rather than Go's
	//     randomized map order (R5.5, R6.5);
	//   - every expected count the statistic divides by is STRICTLY POSITIVE,
	//     recomputed here the same way the statistic does (expected =
	//     rowTotal*colTotal/grand over the sorted union, for both rows) so the
	//     no-division-by-zero guarantee is checked directly (R6.6).
	property := func(baseAgg, testCounts map[string]int, baseSize, w int) bool {
		// Guard the invariants the generator is responsible for, so a bad draw
		// fails loudly rather than silently weakening the assertions.
		if len(baseAgg) == 0 || len(testCounts) == 0 {
			t.Logf("empty input map: baseAgg=%v testCounts=%v", baseAgg, testCounts)
			return false
		}
		for key, c := range baseAgg {
			if c < 1 {
				t.Logf("non-positive baseline count for %q: %d", key, c)
				return false
			}
		}
		for key, c := range testCounts {
			if c < 1 {
				t.Logf("non-positive completed-window count for %q: %d", key, c)
				return false
			}
		}
		if got := sumCounts(baseAgg); got != baseSize {
			t.Logf("sum(baseAgg)=%d != baseSize=%d", got, baseSize)
			return false
		}
		if got := sumCounts(testCounts); got != w {
			t.Logf("sum(testCounts)=%d != W=%d", got, w)
			return false
		}
		if baseSize <= 0 || w <= 0 {
			t.Logf("non-positive row total: baseSize=%d w=%d", baseSize, w)
			return false
		}

		stat := consecutiveChiSquaredStatistic(baseAgg, baseSize, testCounts, w)

		// Division safety: a zero expected count would have produced +Inf or NaN
		// via the (obs-exp)^2/exp term, so a finite result witnesses that every
		// expected count was strictly positive.
		if math.IsInf(stat, 0) {
			t.Logf("statistic is infinite: %v (baseAgg=%v testCounts=%v)", stat, baseAgg, testCounts)
			return false
		}
		if math.IsNaN(stat) {
			t.Logf("statistic is NaN (baseAgg=%v testCounts=%v)", baseAgg, testCounts)
			return false
		}

		// Determinism: identical contents yield the exact same float on a second
		// call. Bitwise == is appropriate because the computation is fully
		// deterministic given the sorted-key iteration.
		statAgain := consecutiveChiSquaredStatistic(baseAgg, baseSize, testCounts, w)
		if stat != statAgain {
			t.Logf("non-deterministic statistic: %v != %v", stat, statAgain)
			return false
		}

		// Order-independence: rebuild both maps by inserting their keys in a
		// different (shuffled) order. Go map iteration order is already
		// randomized, but reinserting in a deliberately shuffled order makes the
		// test robust regardless of runtime behavior. The statistic iterates a
		// sorted union of keys, so the result must be bit-for-bit identical.
		baseShuffled := reinsertShuffled(baseAgg)
		testShuffled := reinsertShuffled(testCounts)
		statShuffled := consecutiveChiSquaredStatistic(baseShuffled, baseSize, testShuffled, w)
		if stat != statShuffled {
			t.Logf("order-dependent statistic: %v != %v (shuffled key insertion order)", stat, statShuffled)
			return false
		}

		// Expected-count positivity: recompute the expected counts the statistic
		// divides by, over the sorted union of keys, and assert each is > 0.
		baseTotal := float64(baseSize)
		testTotal := float64(w)
		grand := baseTotal + testTotal
		for _, key := range dedupeSorted(func() []string {
			u := append(sortedKeys(baseAgg), sortedKeys(testCounts)...)
			sort.Strings(u)
			return u
		}()) {
			colTotal := float64(baseAgg[key] + testCounts[key])
			expBase := baseTotal * colTotal / grand
			expTest := testTotal * colTotal / grand
			if !(expBase > 0) {
				t.Logf("non-positive expected baseline count for %q: %v", key, expBase)
				return false
			}
			if !(expTest > 0) {
				t.Logf("non-positive expected completed-window count for %q: %v", key, expTest)
				return false
			}
		}

		return true
	}

	// Generator draws two non-empty count maps over a small shared alphabet so
	// the rows overlap on some Query_Types and differ on others, with every
	// count >= 1 and no zero-count keys. baseSize and W are the sums of the two
	// maps, mirroring the real baseline and completed-window row totals.
	cfg := &quick.Config{
		MaxCount: 1000,
		Values: func(args []reflect.Value, rng *rand.Rand) {
			randomCounts := func() map[string]int {
				m := make(map[string]int)
				n := rng.Intn(6) + 1 // 1..6 distinct Query_Types
				for i := 0; i < n; i++ {
					key := fmt.Sprintf("q%d", rng.Intn(8))
					m[key] = rng.Intn(9) + 1 // 1..9, always >= 1
				}
				return m
			}

			baseAgg := randomCounts()
			testCounts := randomCounts()
			args[0] = reflect.ValueOf(baseAgg)
			args[1] = reflect.ValueOf(testCounts)
			args[2] = reflect.ValueOf(sumCounts(baseAgg))
			args[3] = reflect.ValueOf(sumCounts(testCounts))
		},
	}

	if err := quick.Check(property, cfg); err != nil {
		t.Error(err)
	}
}

// reinsertShuffled returns a copy of a count map built by inserting its keys in
// a shuffled order, so the Property 7 order-independence check can confirm the
// statistic ignores key insertion order. The returned map has identical
// contents to the input; only the construction order differs.
func reinsertShuffled(counts map[string]int) map[string]int {
	keys := make([]string, 0, len(counts))
	for key := range counts {
		keys = append(keys, key)
	}
	rand.Shuffle(len(keys), func(i, j int) { keys[i], keys[j] = keys[j], keys[i] })
	out := make(map[string]int, len(counts))
	for _, key := range keys {
		out[key] = counts[key]
	}
	return out
}

// Feature: consecutive-window-anomaly-detection, Property 8: Identical distributions yield a zero statistic (homogeneity sanity)
func TestConsecutiveStatisticHomogeneityZero(t *testing.T) {
	// property draws a random base distribution — a "shape" map of Query_Type ->
	// unit count (every count >= 1) — and two positive integer scale factors
	// sBase and sTest. It builds the baseline aggregate and the completed
	// Test_Window as proportional (scalar-multiple) copies of that shape:
	//
	//	baseAgg[key]    = shape[key] * sBase
	//	testCounts[key] = shape[key] * sTest
	//
	// so both rows of the 2xK contingency table share the SAME proportional
	// distribution across Query_Types. Under a chi-squared test of homogeneity,
	// when the two rows have identical proportions the observed count equals the
	// expected count in every cell, so every (obs-exp)^2/exp term is exactly zero
	// and the statistic is zero. Floating-point rounding may leave a tiny nonzero
	// residue, so we assert |got| < 1e-9 rather than exact equality (R6.1, R6.2).
	//
	// Concretely, for a column with unit count u and column total
	// colTotal = u*(sBase+sTest), the baseline expected count is
	//	baseTotal * colTotal / grand
	//	  = (size*sBase) * u*(sBase+sTest) / (size*(sBase+sTest))
	//	  = u*sBase = baseAgg[key],
	// where size = sum(shape). The observed count equals the expected count
	// exactly in exact arithmetic, so the statistic is zero.
	property := func(shape map[string]int, sBase, sTest int) bool {
		// Guard the invariants the generator is responsible for.
		if len(shape) == 0 {
			t.Logf("empty shape map")
			return false
		}
		for key, c := range shape {
			if c < 1 {
				t.Logf("non-positive shape count for %q: %d", key, c)
				return false
			}
		}
		if sBase < 1 || sTest < 1 {
			t.Logf("non-positive scale factor: sBase=%d sTest=%d", sBase, sTest)
			return false
		}

		// Build the two rows as scalar multiples of the shared shape. Every
		// count stays >= 1 (shape count >= 1, scale >= 1) so no zero-count keys
		// appear, matching the real window invariants.
		baseAgg := make(map[string]int, len(shape))
		testCounts := make(map[string]int, len(shape))
		for key, c := range shape {
			baseAgg[key] = c * sBase
			testCounts[key] = c * sTest
		}
		baseSize := sumCounts(baseAgg) // == sum(shape) * sBase
		w := sumCounts(testCounts)     // == sum(shape) * sTest

		got := consecutiveChiSquaredStatistic(baseAgg, baseSize, testCounts, w)

		// A finite near-zero result: identical proportions make observed ==
		// expected in every cell, so the statistic is zero up to rounding.
		if math.IsNaN(got) || math.IsInf(got, 0) {
			t.Logf("non-finite statistic for homogeneous rows: %v (baseAgg=%v testCounts=%v)",
				got, baseAgg, testCounts)
			return false
		}
		if math.Abs(got) >= 1e-9 {
			t.Logf("homogeneous rows gave nonzero statistic: %.17g (shape=%v sBase=%d sTest=%d)",
				got, shape, sBase, sTest)
			return false
		}

		return true
	}

	// Generator draws a small shape map (1..6 distinct Query_Types, each unit
	// count 1..9) and two positive scale factors. Scale factors are kept modest
	// so the multiplied counts stay well within int range and float64 can
	// represent the totals exactly, isolating the homogeneity property from
	// precision loss. sBase and sTest are drawn independently (and may be equal)
	// so the proportional-but-differently-sized case is covered.
	cfg := &quick.Config{
		MaxCount: 1000,
		Values: func(args []reflect.Value, rng *rand.Rand) {
			shape := make(map[string]int)
			n := rng.Intn(6) + 1 // 1..6 distinct Query_Types
			for i := 0; i < n; i++ {
				key := fmt.Sprintf("q%d", rng.Intn(8))
				shape[key] = rng.Intn(9) + 1 // unit count 1..9
			}
			args[0] = reflect.ValueOf(shape)
			args[1] = reflect.ValueOf(rng.Intn(50) + 1) // sBase 1..50
			args[2] = reflect.ValueOf(rng.Intn(50) + 1) // sTest 1..50
		},
	}

	if err := quick.Check(property, cfg); err != nil {
		t.Error(err)
	}
}

// Feature: consecutive-window-anomaly-detection, Property 10: Confidence is bounded, deterministic, and monotonic in the statistic
func TestConsecutiveConfidenceBoundedDeterministicMonotonic(t *testing.T) {
	// property draws two statistics stat1 <= stat2 and a shared strictly
	// positive threshold, all finite, matching the contract confidence is
	// defined for (stat >= 0, threshold > 0). It asserts the four facets of
	// Property 10 against the shared confidence transform:
	//   - [0,1] bounds (R8.1): the score never leaves the inclusive unit range;
	//   - determinism (R8.2): a second call on the same inputs is bitwise-equal;
	//   - equal-statistic equality (R8.4): two independently-computed calls with
	//     the same (stat, threshold) are bitwise-equal;
	//   - monotonicity (R8.3): under the SAME threshold a greater statistic
	//     yields a greater-or-equal confidence.
	property := func(stat1, stat2, threshold float64) bool {
		// The generator already orders stat1 <= stat2, but guard here so the
		// property is self-contained regardless of how values arrive.
		if stat1 > stat2 {
			stat1, stat2 = stat2, stat1
		}

		c1 := confidence(stat1, threshold)
		c2 := confidence(stat2, threshold)

		// [0,1] bounds for both statistics (R8.1). The transform actually lands
		// in [0,1) for these inputs, but the contract permits the inclusive
		// bound, so assert 0.0 <= c <= 1.0.
		for _, c := range []float64{c1, c2} {
			if math.IsNaN(c) || c < 0.0 || c > 1.0 {
				t.Logf("confidence out of [0,1]: %v (stat1=%v stat2=%v threshold=%v)", c, stat1, stat2, threshold)
				return false
			}
		}

		// Bitwise determinism for repeated inputs (R8.2): a second call on the
		// same (stat, threshold) must reproduce the identical bits.
		if confidence(stat1, threshold) != c1 {
			t.Logf("non-deterministic for stat1=%v threshold=%v", stat1, threshold)
			return false
		}
		if confidence(stat2, threshold) != c2 {
			t.Logf("non-deterministic for stat2=%v threshold=%v", stat2, threshold)
			return false
		}

		// Bitwise equality for equal statistics under the same threshold (R8.4):
		// two separately-computed calls with identical inputs are exactly equal.
		if confidence(stat1, threshold) != confidence(stat1, threshold) {
			t.Logf("equal-statistic calls differ for stat1=%v threshold=%v", stat1, threshold)
			return false
		}

		// Monotonicity under the shared threshold (R8.3): stat1 <= stat2 implies
		// confidence(stat1) <= confidence(stat2).
		if c1 > c2 {
			t.Logf("not monotonic: c(%v)=%v > c(%v)=%v (threshold=%v)", stat1, c1, stat2, c2, threshold)
			return false
		}

		return true
	}

	// Generator draws non-negative statistics (scaled up, with an occasional
	// exact 0.0 to exercise the lower bound) and a strictly positive threshold
	// (scaled, plus a small epsilon so it never reaches zero). It orders the two
	// statistics so stat1 <= stat2, giving the monotonicity check a well-defined
	// direction under the shared threshold. All inputs stay finite because the
	// contract is defined only for stat >= 0 and threshold > 0.
	cfg := &quick.Config{
		MaxCount: 1000,
		Values: func(args []reflect.Value, rng *rand.Rand) {
			const scale = 1000.0
			nonNegStat := func() float64 {
				if rng.Intn(10) == 0 {
					return 0.0
				}
				return rng.Float64() * scale
			}
			a := nonNegStat()
			b := nonNegStat()
			if a > b {
				a, b = b, a
			}
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

// Feature: consecutive-window-anomaly-detection, Property 2: Predict is total and never panics
func TestConsecutivePredictTotalNeverPanics(t *testing.T) {
	// property feeds a random stream of log entries into a single valid model
	// and asserts every Predict call returns a non-nil finding. The stream mixes
	// nil entries, empty entries (all fields unset), and well-formed entries with
	// random identifying fields drawn from a small alphabet, so the filling, the
	// empty-Action no-op, the first-window empty-baseline boundary, and repeated
	// evaluate-and-fold boundaries are all exercised. A panic anywhere inside
	// Predict propagates out of quick.Check and fails the test directly, so no
	// recover is needed to satisfy the "never panics" obligation (R7.1, R7.2).
	const (
		w         = 3
		b         = 6
		threshold = 5.0
	)

	// consecutiveProto builds the valid consecutive-window Model proto used to
	// construct the model under test, via the generated opaque builder API.
	consecutiveProto := func(w, b int64, threshold float64) *modelpb.Model {
		cfg := (&modelpb.ConsecutiveWindowTrainingConfig_builder{
			W:                w,
			B:                b,
			AnomalyThreshold: threshold,
		}).Build()
		scenario := (&modelpb.ConsecutiveWindowScenario_builder{Config: cfg}).Build()
		return (&modelpb.Model_builder{ConsecutiveWindow: scenario}).Build()
	}

	model, err := NewConsecutiveWindowModel(consecutiveProto(w, b, threshold))
	if err != nil || model == nil {
		t.Fatalf("failed to build valid model: model=%v err=%v", model, err)
	}

	// entryKind selects the shape of each generated entry.
	const (
		kindNil        = iota // nil *AuditLogEntry
		kindEmpty             // non-nil entry with no fields set
		kindWellFormed        // non-nil entry with a random Action and identifying fields
		kindCount
	)

	// property replays one random stream against the shared model. Each stream
	// step is a (kind, field-index) pair. Returning the non-nil check as a bool
	// lets quick.Check shrink a failing stream; a panic would abort before this
	// returns and still fail the test.
	property := func(steps []uint16) bool {
		for _, step := range steps {
			kind := int(step % kindCount)
			sel := int(step / kindCount)

			var entry *modelpb.AuditLogEntry
			switch kind {
			case kindNil:
				entry = nil // exercises the nil-entry / extraction-failure path
			case kindEmpty:
				// All fields unset: empty Action is an explicit window no-op.
				entry = &modelpb.AuditLogEntry{}
			default:
				// Well-formed entry. The Action is drawn from a small alphabet
				// (q0..q3) so blocks repeat and baseline folds overlap, while the
				// other identifying fields vary over a small set too.
				entry = &modelpb.AuditLogEntry{}
				entry.SetNormalizedQueryTemplate(fmt.Sprintf("q%d", sel%4))
				entry.SetDbUser(fmt.Sprintf("u%d", (sel/4)%3))
				entry.SetDbName(fmt.Sprintf("d%d", (sel/12)%3))
				entry.SetQueryStatement(fmt.Sprintf("s%d", (sel/36)%3))
			}

			finding := model.Predict(entry)
			if finding == nil {
				t.Logf("Predict returned nil finding for step kind=%d sel=%d", kind, sel)
				return false
			}
		}
		return true
	}

	cfg := &quick.Config{MaxCount: 200}
	if err := quick.Check(property, cfg); err != nil {
		t.Error(err)
	}
}

// Feature: consecutive-window-anomaly-detection, Property 3: Empty and failed extractions are window no-ops
func TestConsecutiveEmptyAndFailedExtractionsAreNoOps(t *testing.T) {
	// property replays one random stream in two ways against two independently
	// constructed models that share an identical valid config. modelA is fed the
	// FULL interleaved stream, which mixes three no-op shapes in among the
	// well-formed entries:
	//
	//   - a nil *AuditLogEntry, for which common.Extract returns an error so
	//     Predict takes the extraction-failure path (R7.5);
	//   - a non-nil entry whose NormalizedQueryTemplate is "" (empty Action), for
	//     which Predict takes the empty-Action path (R3.5);
	//   - well-formed entries with a non-empty Action drawn from a small alphabet.
	//
	// modelB is fed only the "empty-stripped" stream: the same well-formed
	// entries in the same order, with every nil / empty-Action entry removed.
	//
	// The property asserts that the no-op entries are invisible to window state
	// and to the finding sequence:
	//
	//   - every no-op entry fed to modelA yields a non-anomalous, zero-confidence
	//     finding (R7.5, R3.5), and leaves both of modelA's windows byte-for-byte
	//     unchanged across that call;
	//   - after the whole stream, modelA's Test_Window and Baseline_Window
	//     (size, counts, agg, blocks) are deeply equal to modelB's, so the no-op
	//     entries never advanced toward a boundary nor perturbed the baseline;
	//   - the sequence of findings modelA produces for its well-formed entries
	//     (in order) equals, finding-by-finding, the sequence modelB produces for
	//     the same entries, so the interleaved no-ops did not shift any computed
	//     statistic or anomaly decision.
	const (
		w         = 3
		b         = 6
		threshold = 5.0
	)

	// consecutiveProto builds the shared valid consecutive-window Model proto via
	// the generated opaque builder API, matching the other tests in this file.
	consecutiveProto := func(w, b int64, threshold float64) *modelpb.Model {
		cfg := (&modelpb.ConsecutiveWindowTrainingConfig_builder{
			W:                w,
			B:                b,
			AnomalyThreshold: threshold,
		}).Build()
		scenario := (&modelpb.ConsecutiveWindowScenario_builder{Config: cfg}).Build()
		return (&modelpb.Model_builder{ConsecutiveWindow: scenario}).Build()
	}

	// entryKind selects the shape of each generated stream step.
	const (
		kindNil        = iota // nil *AuditLogEntry (extraction failure, R7.5)
		kindEmpty             // non-nil entry with empty Action (R3.5)
		kindWellFormed        // non-nil entry with a non-empty Action
		kindCount
	)

	// windowsEqual reports whether two models hold deeply-equal Test_Window and
	// Baseline_Window state: matching sizes, the live test counts, the baseline
	// aggregate, and the baseline block FIFO (order and contents).
	windowsEqual := func(x, y *ConsecutiveWindowModel) bool {
		if x.test.size != y.test.size {
			return false
		}
		if !reflect.DeepEqual(x.test.counts, y.test.counts) {
			return false
		}
		if x.baseline.size != y.baseline.size {
			return false
		}
		if !reflect.DeepEqual(x.baseline.agg, y.baseline.agg) {
			return false
		}
		if !reflect.DeepEqual(x.baseline.blocks, y.baseline.blocks) {
			return false
		}
		return true
	}

	// findingsEqual compares two findings on the fields Property 3 cares about:
	// the anomaly flag and the confidence score. proto.Equal would also work, but
	// comparing the decision fields directly keeps the failure messages legible.
	findingsEqual := func(a, c *modelpb.AnomalyFinding) bool {
		return a.GetIsAnomalousPredict() == c.GetIsAnomalousPredict() &&
			a.GetConfidenceScore() == c.GetConfidenceScore()
	}

	// property replays one random stream. Each step is a (kind, field-index)
	// pair packed into a uint16, matching the encoding used by the Property 2
	// test so quick.Check can shrink a failing stream.
	property := func(steps []uint16) bool {
		modelA, errA := NewConsecutiveWindowModel(consecutiveProto(w, b, threshold))
		modelB, errB := NewConsecutiveWindowModel(consecutiveProto(w, b, threshold))
		if errA != nil || modelA == nil || errB != nil || modelB == nil {
			t.Logf("failed to build models: A=(%v,%v) B=(%v,%v)", modelA, errA, modelB, errB)
			return false
		}

		var findingsA []*modelpb.AnomalyFinding // modelA findings for well-formed entries
		var findingsB []*modelpb.AnomalyFinding // modelB findings for the stripped stream

		for _, step := range steps {
			kind := int(step % kindCount)
			sel := int(step / kindCount)

			if kind == kindWellFormed {
				// A well-formed entry advances both models identically: feed the
				// same entry value to each and record both findings in order.
				action := fmt.Sprintf("q%d", sel%4)

				entryA := &modelpb.AuditLogEntry{}
				entryA.SetNormalizedQueryTemplate(action)
				entryA.SetDbUser(fmt.Sprintf("u%d", (sel/4)%3))
				entryA.SetDbName(fmt.Sprintf("d%d", (sel/12)%3))
				entryA.SetQueryStatement(fmt.Sprintf("s%d", (sel/36)%3))

				entryB := &modelpb.AuditLogEntry{}
				entryB.SetNormalizedQueryTemplate(action)
				entryB.SetDbUser(fmt.Sprintf("u%d", (sel/4)%3))
				entryB.SetDbName(fmt.Sprintf("d%d", (sel/12)%3))
				entryB.SetQueryStatement(fmt.Sprintf("s%d", (sel/36)%3))

				findingsA = append(findingsA, modelA.Predict(entryA))
				findingsB = append(findingsB, modelB.Predict(entryB))
				continue
			}

			// No-op step: feed the no-op entry only to modelA, snapshotting its
			// window state so we can assert the call left both windows unchanged.
			var entry *modelpb.AuditLogEntry
			if kind == kindEmpty {
				entry = &modelpb.AuditLogEntry{} // empty Action (R3.5)
			} // else kindNil: leave entry nil (extraction failure, R7.5)

			testSizeBefore := modelA.test.size
			testCountsBefore := copyCounts(modelA.test.counts)
			baseSizeBefore := modelA.baseline.size
			baseAggBefore := copyCounts(modelA.baseline.agg)
			// Snapshot the block FIFO, preserving nil-vs-empty: a fresh baseline
			// has a nil blocks slice, so only allocate when blocks are present,
			// otherwise reflect.DeepEqual(nil, []map{}{}) would spuriously differ.
			var blocksBefore []map[string]int
			if modelA.baseline.blocks != nil {
				blocksBefore = make([]map[string]int, len(modelA.baseline.blocks))
				for i, blk := range modelA.baseline.blocks {
					blocksBefore[i] = copyCounts(blk)
				}
			}

			finding := modelA.Predict(entry)

			// A no-op finding is non-anomalous with exactly zero confidence.
			if finding == nil {
				t.Logf("no-op Predict returned nil finding (kind=%d)", kind)
				return false
			}
			if finding.GetIsAnomalousPredict() {
				t.Logf("no-op finding marked anomalous (kind=%d)", kind)
				return false
			}
			if finding.GetConfidenceScore() != 0.0 {
				t.Logf("no-op finding confidence=%v, want 0.0 (kind=%d)", finding.GetConfidenceScore(), kind)
				return false
			}

			// The no-op must not touch either window: size, counts, agg, and the
			// block FIFO are unchanged across the call.
			if modelA.test.size != testSizeBefore {
				t.Logf("no-op changed test.size: %d -> %d (kind=%d)", testSizeBefore, modelA.test.size, kind)
				return false
			}
			if !reflect.DeepEqual(modelA.test.counts, testCountsBefore) {
				t.Logf("no-op changed test.counts: %v -> %v (kind=%d)", testCountsBefore, modelA.test.counts, kind)
				return false
			}
			if modelA.baseline.size != baseSizeBefore {
				t.Logf("no-op changed baseline.size: %d -> %d (kind=%d)", baseSizeBefore, modelA.baseline.size, kind)
				return false
			}
			if !reflect.DeepEqual(modelA.baseline.agg, baseAggBefore) {
				t.Logf("no-op changed baseline.agg: %v -> %v (kind=%d)", baseAggBefore, modelA.baseline.agg, kind)
				return false
			}
			if !reflect.DeepEqual(modelA.baseline.blocks, blocksBefore) {
				t.Logf("no-op changed baseline.blocks: %v -> %v (kind=%d)", blocksBefore, modelA.baseline.blocks, kind)
				return false
			}
		}

		// After the whole stream, the interleaved-with-no-ops model and the
		// empty-stripped model hold byte-for-byte identical window state.
		if !windowsEqual(modelA, modelB) {
			t.Logf("final window state differs between interleaved and stripped streams:\n"+
				"  A: test(size=%d counts=%v) baseline(size=%d agg=%v blocks=%v)\n"+
				"  B: test(size=%d counts=%v) baseline(size=%d agg=%v blocks=%v)",
				modelA.test.size, modelA.test.counts, modelA.baseline.size, modelA.baseline.agg, modelA.baseline.blocks,
				modelB.test.size, modelB.test.counts, modelB.baseline.size, modelB.baseline.agg, modelB.baseline.blocks)
			return false
		}

		// The two finding sequences for the well-formed entries must match
		// finding-by-finding (same length, same anomaly flag and confidence).
		if len(findingsA) != len(findingsB) {
			t.Logf("finding-count mismatch: A=%d B=%d", len(findingsA), len(findingsB))
			return false
		}
		for i := range findingsA {
			if !findingsEqual(findingsA[i], findingsB[i]) {
				t.Logf("finding %d differs: A=(anom=%v conf=%v) B=(anom=%v conf=%v)",
					i,
					findingsA[i].GetIsAnomalousPredict(), findingsA[i].GetConfidenceScore(),
					findingsB[i].GetIsAnomalousPredict(), findingsB[i].GetConfidenceScore())
				return false
			}
		}

		return true
	}

	cfg := &quick.Config{MaxCount: 200}
	if err := quick.Check(property, cfg); err != nil {
		t.Error(err)
	}
}

// Feature: consecutive-window-anomaly-detection, Property 4: Consecutive windows partition the stream into disjoint W-blocks
func TestConsecutivePredictPartitionsStreamIntoDisjointBlocks(t *testing.T) {
	// This is the MODEL-LEVEL Property 4 over Predict (the testWindow-unit
	// variant lives in TestTestWindowFillAndClearSemantics). It observes the
	// internal model as a random non-empty Action stream is fed through Predict
	// one entry at a time and asserts that the model partitions the arrival-order
	// stream into non-overlapping blocks of exactly W Actions:
	//
	//   - each Action belongs to exactly one block (the arrival-order partition
	//     into W-sized chunks, dropping any trailing partial chunk) (R3.1, R3.4);
	//   - consecutive blocks share no Action: the partition is disjoint by
	//     construction, which the model realizes by clearing the Test_Window at
	//     every boundary so the next W Actions reuse none of the completed ones
	//     (R3.3);
	//   - exactly one evaluation (boundary) is produced on the Predict call that
	//     fills the Test_Window to W: right after that call the Test_Window is
	//     cleared, so test.size returns to 0 precisely on the calls whose
	//     cumulative appended count is a multiple of W (R3.2, R3.3). Per step we
	//     assert test.size == cumulativeAppended % W, which holds iff the window
	//     fills and clears on exactly the boundary calls and nowhere else.
	//
	// After the whole stream, the completed blocks are independently
	// reconstructed by partitioning the appended-Action stream into consecutive
	// W-sized chunks (floor(n/W) complete blocks). The baseline must retain the
	// most recent min(B/W, numCompleteBlocks) of those blocks as count maps in
	// FIFO order, its aggregate must equal their element-wise sum, and the live
	// Test_Window must hold exactly the trailing partial chunk (the still-filling
	// remainder) with test.size == n % W.
	property := func(rawW uint8, rawK uint8, actions []string) bool {
		// W in 1..5; k in 2..4 so B = kW is a multiple of W strictly greater
		// than W; B/W == k is the steady-state completed-block count.
		w := int(rawW%5) + 1
		k := int(rawK%3) + 2
		b := k * w
		maxBlocks := b / w // == k
		threshold := 5.0

		// The generator guarantees a non-empty stream; guard it so a bad draw
		// fails loudly rather than silently weakening the property.
		if len(actions) == 0 {
			t.Logf("empty action stream")
			return false
		}

		cfg := (&modelpb.ConsecutiveWindowTrainingConfig_builder{
			W:                int64(w),
			B:                int64(b),
			AnomalyThreshold: threshold,
		}).Build()
		scenario := (&modelpb.ConsecutiveWindowScenario_builder{Config: cfg}).Build()
		pb := (&modelpb.Model_builder{ConsecutiveWindow: scenario}).Build()

		model, err := NewConsecutiveWindowModel(pb)
		if err != nil || model == nil {
			t.Logf("failed to build valid model: model=%v err=%v", model, err)
			return false
		}

		// Feed the stream one well-formed entry at a time, tracking the
		// cumulative count of appended (non-empty) Actions. Every Action here is
		// non-empty, so each Predict appends exactly one Action and the
		// cumulative count equals the number of entries fed so far.
		for i, action := range actions {
			entry := &modelpb.AuditLogEntry{}
			entry.SetNormalizedQueryTemplate(action)

			finding := model.Predict(entry)
			if finding == nil {
				t.Logf("Predict returned nil finding at step %d", i)
				return false
			}

			cumulative := i + 1

			// One-evaluation-per-boundary: the Test_Window fills to W and is
			// cleared exactly on the calls where the cumulative appended count is
			// a multiple of W. Between boundaries it holds the in-progress chunk.
			// test.size == cumulative % W captures both facts in a single check.
			if model.test.size != cumulative%w {
				t.Logf("after step %d: test.size=%d, want %d (cumulative=%d W=%d)",
					i, model.test.size, cumulative%w, cumulative, w)
				return false
			}
		}

		n := len(actions)
		numCompleteBlocks := n / w // floor: trailing partial chunk is dropped
		remainder := n % w         // Actions in the still-filling Test_Window

		// Independently partition the arrival-order stream into consecutive,
		// non-overlapping W-sized chunks. Each chunk is the count map of W
		// consecutive Actions; consecutive chunks cover disjoint index ranges, so
		// no Action belongs to two blocks.
		var completedBlocks []map[string]int
		for start := 0; start+w <= n; start += w {
			completedBlocks = append(completedBlocks, countsOf(actions[start:start+w]))
		}
		if len(completedBlocks) != numCompleteBlocks {
			t.Logf("reconstructed %d complete blocks, want %d", len(completedBlocks), numCompleteBlocks)
			return false
		}

		// The baseline retains the most recent min(B/W, numCompleteBlocks) blocks
		// in FIFO (oldest-to-newest) order.
		retained := completedBlocks
		if len(retained) > maxBlocks {
			retained = retained[len(retained)-maxBlocks:]
		}

		if len(model.baseline.blocks) != len(retained) {
			t.Logf("baseline holds %d blocks, want %d", len(model.baseline.blocks), len(retained))
			return false
		}
		for i := range retained {
			if !reflect.DeepEqual(model.baseline.blocks[i], retained[i]) {
				t.Logf("retained block %d mismatch: got %v want %v", i, model.baseline.blocks[i], retained[i])
				return false
			}
		}

		// baseline.size is an exact multiple of W equal to len(retained)*W.
		wantBaseSize := len(retained) * w
		if model.baseline.size != wantBaseSize {
			t.Logf("baseline.size=%d, want %d", model.baseline.size, wantBaseSize)
			return false
		}

		// The aggregate equals the element-wise sum of the retained blocks.
		wantAgg := make(map[string]int)
		for _, blk := range retained {
			for key, c := range blk {
				wantAgg[key] += c
			}
		}
		if len(wantAgg) == 0 {
			// Normalize: a baseline with no retained blocks has an empty (not
			// nil) agg map, matching newBaselineWindow.
			wantAgg = map[string]int{}
		}
		if !reflect.DeepEqual(model.baseline.agg, wantAgg) {
			t.Logf("baseline.agg mismatch: got %v want %v", model.baseline.agg, wantAgg)
			return false
		}

		// The live Test_Window holds exactly the trailing partial chunk: its
		// size is n % W and its counts are the count map of the remainder
		// Actions. This is the still-filling window that shares no Action with
		// any completed block.
		if model.test.size != remainder {
			t.Logf("test.size=%d, want remainder %d", model.test.size, remainder)
			return false
		}
		wantTestCounts := countsOf(actions[n-remainder:])
		if !reflect.DeepEqual(model.test.counts, wantTestCounts) {
			t.Logf("test.counts mismatch: got %v want %v", model.test.counts, wantTestCounts)
			return false
		}

		return true
	}

	// Generator draws small W/k and a NON-EMPTY Action stream over a small
	// alphabet so blocks repeat Query_Types and the baseline FIFO churns once
	// the stream is long enough to overflow B. Stream length ranges well past
	// B = kW (up to ~40 Actions) so many boundaries and several evictions occur.
	cfg := &quick.Config{
		MaxCount: 300,
		Values: func(args []reflect.Value, rng *rand.Rand) {
			args[0] = reflect.ValueOf(uint8(rng.Intn(256)))
			args[1] = reflect.ValueOf(uint8(rng.Intn(256)))

			nStream := rng.Intn(40) + 1 // 1..40 Actions, always non-empty
			stream := make([]string, nStream)
			for i := range stream {
				stream[i] = fmt.Sprintf("q%d", rng.Intn(4))
			}
			args[2] = reflect.ValueOf(stream)
		},
	}

	if err := quick.Check(property, cfg); err != nil {
		t.Error(err)
	}
}

// Feature: consecutive-window-anomaly-detection, Property 9: The anomaly decision follows the threshold with >= semantics
func TestConsecutiveAnomalyDecisionFollowsThreshold(t *testing.T) {
	// property crafts a boundary evaluation whose chi-squared statistic is known
	// in advance, then drives the Predict anomaly decision across the three
	// regions around that statistic to pin the >= semantics of Requirements 6.3
	// and 6.4:
	//
	//   - threshold strictly below the statistic  -> anomalous (stat >= thr);
	//   - threshold exactly equal to the statistic -> anomalous (the boundary is
	//     inclusive: stat >= thr holds on equality);
	//   - threshold strictly above the statistic   -> not anomalous (stat < thr).
	//
	// Construction of a known boundary. The model evaluates a completed
	// Test_Window against the Baseline_Window as it stands BEFORE the completed
	// window is folded in. So for each draw we:
	//   1. build a valid model (W, B = kW, any positive threshold);
	//   2. feed one full W-block (block 1) so the first boundary folds it into
	//      the baseline, leaving baseline.size == W and baseline.agg == counts of
	//      block 1 (the first boundary itself has an empty baseline, so it is a
	//      non-anomalous warm-up and does not perturb this argument);
	//   3. feed the first W-1 Actions of a second W-block (block 2) WITHOUT
	//      completing it, so the baseline is unchanged and the Test_Window holds
	//      the partial block 2;
	//   4. independently compute the statistic the model WILL compute on the
	//      boundary call that completes block 2:
	//         wantStat = consecutiveChiSquaredStatistic(baseline.agg,
	//                        baseline.size, countsOf(block2), W)
	//      using the full block-2 counts (the Test_Window will hold exactly those
	//      once the final Action is appended);
	//   5. set model.threshold directly (an in-package unexported field) to a
	//      value chosen relative to wantStat for the case under test, then feed
	//      the final (W-th) Action of block 2 to trigger the boundary and read
	//      finding.GetIsAnomalousPredict().
	//
	// To place a positive threshold strictly below wantStat we need wantStat > 0,
	// which requires block 1 and block 2 to differ in distribution (identical
	// distributions give a zero statistic). The generator draws two deliberately
	// different single-Query_Type blocks so wantStat is comfortably positive; the
	// property still guards wantStat > 0 and skips (returns true) on the rare
	// degenerate draw so a zero statistic can never make a chosen threshold
	// non-positive and invalid.
	property := func(rawW uint8, rawK uint8, seed int64) bool {
		// W in 2..6; k in 2..4 so B = kW is a multiple of W strictly greater
		// than W. W >= 2 keeps each block large enough to express two distinct
		// distributions.
		w := int(rawW%5) + 2
		k := int(rawK%3) + 2
		b := k * w

		rng := rand.New(rand.NewSource(seed))

		// block1 and block2 are each W Actions. To make the two distributions
		// differ (so wantStat > 0), block1 is all "a" and block2 is biased the
		// other way: a single "a" followed by W-1 "b" Actions. For W >= 2 these
		// are clearly different distributions, giving a positive statistic.
		block1 := make([]string, w)
		for i := range block1 {
			block1[i] = "a"
		}
		block2 := make([]string, w)
		block2[0] = "a"
		for i := 1; i < w; i++ {
			block2[i] = "b"
		}
		// Shuffle block2's interior so the boundary-completing Action is not
		// always the same symbol; the completed-window COUNTS are unchanged by a
		// permutation, so wantStat is unaffected, but this exercises different
		// final Actions arriving on the boundary call.
		rng.Shuffle(len(block2), func(i, j int) { block2[i], block2[j] = block2[j], block2[i] })

		// buildSeededModel constructs a fresh valid model and feeds block 1 to a
		// full boundary so the baseline holds exactly block 1 (baseline.size ==
		// W). It then feeds the first W-1 Actions of block 2, leaving the
		// Test_Window one Action short of the boundary that will evaluate block 2.
		// It returns the model poised at that pre-boundary state.
		buildSeededModel := func() *ConsecutiveWindowModel {
			cfg := (&modelpb.ConsecutiveWindowTrainingConfig_builder{
				W:                int64(w),
				B:                int64(b),
				AnomalyThreshold: 1.0, // placeholder; overwritten before the boundary
			}).Build()
			scenario := (&modelpb.ConsecutiveWindowScenario_builder{Config: cfg}).Build()
			pb := (&modelpb.Model_builder{ConsecutiveWindow: scenario}).Build()

			model, err := NewConsecutiveWindowModel(pb)
			if err != nil || model == nil {
				t.Fatalf("failed to build valid model: model=%v err=%v", model, err)
			}

			// Feed block 1 to completion: the first boundary has an empty
			// baseline (warm-up, not anomalous) and folds block 1 in.
			for _, action := range block1 {
				entry := &modelpb.AuditLogEntry{}
				entry.SetNormalizedQueryTemplate(action)
				model.Predict(entry)
			}
			// Feed the first W-1 Actions of block 2 without completing it.
			for i := 0; i < w-1; i++ {
				entry := &modelpb.AuditLogEntry{}
				entry.SetNormalizedQueryTemplate(block2[i])
				model.Predict(entry)
			}
			return model
		}

		// Compute, independently of the Predict path, the statistic the model
		// will produce on the block-2 boundary: the pre-fold baseline is exactly
		// block 1 (size W), and the completed Test_Window is exactly block 2.
		wantStat := consecutiveChiSquaredStatistic(countsOf(block1), w, countsOf(block2), w)

		// Guard: a positive statistic is required so the strictly-below threshold
		// stays positive and valid. The chosen blocks differ, so this should
		// always hold; skip the degenerate draw defensively rather than place an
		// invalid threshold.
		if !(wantStat > 0) {
			return true
		}

		// runBoundary seeds a fresh model, overwrites its threshold to the given
		// value (an in-package unexported field), feeds the final block-2 Action
		// to trigger the boundary evaluation, and returns the finding's anomaly
		// flag together with the threshold actually used.
		runBoundary := func(threshold float64) bool {
			model := buildSeededModel()
			model.threshold = threshold

			entry := &modelpb.AuditLogEntry{}
			entry.SetNormalizedQueryTemplate(block2[w-1])
			finding := model.Predict(entry)
			if finding == nil {
				t.Fatalf("boundary Predict returned nil finding")
			}
			return finding.GetIsAnomalousPredict()
		}

		// Case 1: threshold strictly below wantStat -> anomalous (stat >= thr).
		// Keep the threshold strictly positive and strictly below wantStat.
		below := wantStat / 2
		if !(below > 0) {
			// wantStat is a tiny positive; fall back to a value just under it.
			below = wantStat * 0.9
		}
		if got := runBoundary(below); got != (wantStat >= below) || !got {
			t.Logf("below case: anomalous=%v, want true (wantStat=%v threshold=%v)", got, wantStat, below)
			return false
		}

		// Case 2: threshold exactly equal to wantStat -> anomalous (>= is
		// inclusive at the boundary).
		if got := runBoundary(wantStat); got != (wantStat >= wantStat) || !got {
			t.Logf("equal case: anomalous=%v, want true (wantStat=%v threshold=%v)", got, wantStat, wantStat)
			return false
		}

		// Case 3: threshold strictly above wantStat -> not anomalous (stat < thr).
		above := wantStat * 2
		if !(above > wantStat) {
			above = wantStat + 1
		}
		if got := runBoundary(above); got != (wantStat >= above) || got {
			t.Logf("above case: anomalous=%v, want false (wantStat=%v threshold=%v)", got, wantStat, above)
			return false
		}

		return true
	}

	cfg := &quick.Config{
		MaxCount: 300,
		Values: func(args []reflect.Value, rng *rand.Rand) {
			args[0] = reflect.ValueOf(uint8(rng.Intn(256)))
			args[1] = reflect.ValueOf(uint8(rng.Intn(256)))
			args[2] = reflect.ValueOf(rng.Int63())
		},
	}

	if err := quick.Check(property, cfg); err != nil {
		t.Error(err)
	}
}

// Feature: consecutive-window-anomaly-detection, Property 11: Non-computed findings carry zero confidence and no anomaly
func TestConsecutiveNonComputedFindingsZeroConfidenceNoAnomaly(t *testing.T) {
	// A "non-computed finding" is any finding Predict returns WITHOUT running the
	// chi-squared statistic / confidence transform. The design produces one in
	// four distinct situations, all of which must yield a finding that is NOT
	// anomalous and whose confidence is EXACTLY 0.0 (bit-equal, not merely close):
	//
	//   1. a nil *AuditLogEntry: common.Extract fails, so Predict returns a bare
	//      &AnomalyFinding{} whose zero-value fields give IsAnomalousPredict()
	//      == false and ConfidenceScore() == 0.0 (R7.5);
	//   2. an entry with an empty Action (NormalizedQueryTemplate == ""): Predict
	//      takes the empty-Action no-op path and returns a (false, 0.0) finding
	//      (R3.5, R8.5);
	//   3. a still-filling Test_Window: before the window reaches W, every
	//      Predict call returns a (false, 0.0) finding (R7.6);
	//   4. the empty-baseline FIRST-window boundary: the Predict call that fills
	//      the very first window to W completes a boundary whose baseline is
	//      still empty, so no statistic is computed and the finding is (false,
	//      0.0) (R5.6). On that same call the model folds the first block into
	//      the baseline, but the finding RETURNED for the call is the
	//      non-computed one.
	//
	// The property builds a valid model with random W (2..6) and B = kW, feeds
	// the entries described for each case, and asserts each relevant finding is
	// non-anomalous with confidence bit-equal to 0.0. A finite positive threshold
	// is used throughout; it is irrelevant to these paths (no statistic is ever
	// compared against it) but keeps the model valid.
	property := func(rawW uint8, rawK uint8, seed int64) bool {
		// W in 2..6; k in 2..4 so B = kW is a multiple of W strictly greater
		// than W. W >= 2 ensures the "still-filling" case has at least one
		// pre-boundary step to observe.
		w := int(rawW%5) + 2
		k := int(rawK%3) + 2
		b := k * w
		threshold := 5.0

		rng := rand.New(rand.NewSource(seed))

		// newModel builds a fresh valid consecutive-window model via the shared
		// opaque builder API used throughout this file.
		newModel := func() *ConsecutiveWindowModel {
			cfg := (&modelpb.ConsecutiveWindowTrainingConfig_builder{
				W:                int64(w),
				B:                int64(b),
				AnomalyThreshold: threshold,
			}).Build()
			scenario := (&modelpb.ConsecutiveWindowScenario_builder{Config: cfg}).Build()
			pb := (&modelpb.Model_builder{ConsecutiveWindow: scenario}).Build()
			model, err := NewConsecutiveWindowModel(pb)
			if err != nil || model == nil {
				t.Fatalf("failed to build valid model: model=%v err=%v", model, err)
			}
			return model
		}

		// assertNonComputed asserts a finding is non-nil, not anomalous, and has
		// confidence bit-equal to 0.0.
		assertNonComputed := func(f *modelpb.AnomalyFinding, context string) bool {
			if f == nil {
				t.Logf("%s: finding is nil", context)
				return false
			}
			if f.GetIsAnomalousPredict() {
				t.Logf("%s: finding marked anomalous, want false", context)
				return false
			}
			if f.GetConfidenceScore() != 0.0 {
				t.Logf("%s: confidence=%v, want exactly 0.0", context, f.GetConfidenceScore())
				return false
			}
			return true
		}

		// Case 1: nil *AuditLogEntry (extraction failure, R7.5).
		{
			model := newModel()
			if !assertNonComputed(model.Predict(nil), "case 1 nil entry") {
				return false
			}
		}

		// Case 2: entry with an empty Action (R3.5, R8.5). A non-nil entry whose
		// NormalizedQueryTemplate is left unset extracts to an empty Action.
		{
			model := newModel()
			entry := &modelpb.AuditLogEntry{}
			if !assertNonComputed(model.Predict(entry), "case 2 empty action") {
				return false
			}
		}

		// Case 3: still-filling Test_Window (R7.6). Feed W-1 well-formed Actions
		// into a fresh model; none of these reaches the boundary, so every
		// finding must be non-computed. The Actions come from a small alphabet so
		// the window holds a real (non-empty) count map while remaining below W.
		{
			model := newModel()
			for i := 0; i < w-1; i++ {
				entry := &modelpb.AuditLogEntry{}
				entry.SetNormalizedQueryTemplate(fmt.Sprintf("q%d", rng.Intn(3)))
				f := model.Predict(entry)
				if !assertNonComputed(f, fmt.Sprintf("case 3 still-filling step %d/%d", i+1, w)) {
					return false
				}
			}
			// Sanity: the window is one Action short of the boundary, so it has
			// not evaluated yet (no fold occurred, baseline still empty).
			if model.test.size != w-1 {
				t.Logf("case 3: test.size=%d after W-1 fills, want %d", model.test.size, w-1)
				return false
			}
			if model.baseline.size != 0 {
				t.Logf("case 3: baseline.size=%d after W-1 fills, want 0", model.baseline.size)
				return false
			}
		}

		// Case 4: empty-baseline first-window boundary (R5.6). Feed exactly W
		// well-formed Actions into a fresh model. The W-th call completes the
		// FIRST window; its baseline is still empty, so no statistic is computed
		// and the finding returned for that boundary call is the non-computed
		// one. (On that same call the model folds the first block into the
		// baseline, which we confirm after the fact.)
		{
			model := newModel()
			actions := make([]string, w)
			for i := range actions {
				actions[i] = fmt.Sprintf("q%d", rng.Intn(3))
			}

			var boundaryFinding *modelpb.AnomalyFinding
			for i, action := range actions {
				entry := &modelpb.AuditLogEntry{}
				entry.SetNormalizedQueryTemplate(action)
				f := model.Predict(entry)
				if i < w-1 {
					// The pre-boundary fills are also non-computed (case 3
					// territory), but the focus here is the final boundary call.
					if !assertNonComputed(f, fmt.Sprintf("case 4 pre-boundary step %d/%d", i+1, w)) {
						return false
					}
					continue
				}
				boundaryFinding = f
			}

			// The first-window boundary finding is non-computed: not anomalous,
			// confidence exactly 0.0, despite the window having reached W.
			if !assertNonComputed(boundaryFinding, "case 4 empty-baseline first boundary") {
				return false
			}

			// Confirm the boundary folded the first block into the baseline (so
			// the empty-baseline condition applied to THIS call, not a later one)
			// and cleared the Test_Window.
			if model.baseline.size != w {
				t.Logf("case 4: baseline.size=%d after first boundary, want %d (fold expected)",
					model.baseline.size, w)
				return false
			}
			if model.test.size != 0 {
				t.Logf("case 4: test.size=%d after first boundary, want 0 (clear expected)",
					model.test.size)
				return false
			}
		}

		return true
	}

	cfg := &quick.Config{
		MaxCount: 300,
		Values: func(args []reflect.Value, rng *rand.Rand) {
			args[0] = reflect.ValueOf(uint8(rng.Intn(256)))
			args[1] = reflect.ValueOf(uint8(rng.Intn(256)))
			args[2] = reflect.ValueOf(rng.Int63())
		},
	}

	if err := quick.Check(property, cfg); err != nil {
		t.Error(err)
	}
}

// Feature: consecutive-window-anomaly-detection, Property 12: Findings mirror the log entry's identifying fields
func TestConsecutiveFindingsMirrorIdentifyingFields(t *testing.T) {
	// property checks that for any NON-NIL entry, the finding produced by
	// Predict reflects the entry's four identifying fields verbatim (db user,
	// db name, query statement, normalized query template), regardless of
	// whether the Action was empty or whether the call landed on a
	// Window_Boundary (R7.3). generateAnomalyFinding copies those fields on
	// every path taken for a non-nil entry: the empty-Action no-op, the
	// still-filling path, the empty-baseline first boundary, and the computed
	// boundary all route through generateAnomalyFinding(logEntry, ...) which
	// reads the fields off the entry. Extraction only fails for a nil entry
	// (common.Extract returns ErrInvalidArgument iff logEntry == nil), so a
	// non-nil entry never hits the empty-finding error path. For a NIL entry,
	// Predict returns an empty finding whose identifying fields are all the
	// empty string (R7.4).
	//
	// Validates: Requirements 7.3, 7.4

	// consecutiveProto builds a valid consecutive-window Model proto via the
	// generated opaque builder API, matching the other tests in this file.
	consecutiveProto := func(w, b int64, threshold float64) *modelpb.Model {
		cfg := (&modelpb.ConsecutiveWindowTrainingConfig_builder{
			W:                w,
			B:                b,
			AnomalyThreshold: threshold,
		}).Build()
		scenario := (&modelpb.ConsecutiveWindowScenario_builder{Config: cfg}).Build()
		return (&modelpb.Model_builder{ConsecutiveWindow: scenario}).Build()
	}

	property := func(rawW uint8, rawK uint8, rawThreshold uint16, stream []*modelpb.AuditLogEntry) bool {
		// Build a valid model: W in 1..8, B = kW with k >= 2, finite positive
		// threshold. Fold behavior is irrelevant to this property, but a valid
		// model lets the stream exercise every finding-producing path.
		w := int64(rawW%8) + 1
		k := int64(rawK%5) + 2
		b := k * w
		threshold := float64(rawThreshold%1000) + 1e-6

		model, err := NewConsecutiveWindowModel(consecutiveProto(w, b, threshold))
		if err != nil {
			t.Logf("unexpected constructor error for valid config (W=%d B=%d thr=%g): %v",
				w, b, threshold, err)
			return false
		}

		// Every non-nil entry's finding must mirror its four identifying
		// fields, on whatever path through Predict the entry takes.
		for i, entry := range stream {
			finding := model.Predict(entry)
			if finding == nil {
				t.Logf("entry %d: Predict returned nil finding", i)
				return false
			}
			if got, want := finding.GetDbUser(), entry.GetDbUser(); got != want {
				t.Logf("entry %d: db user = %q, want %q", i, got, want)
				return false
			}
			if got, want := finding.GetDbName(), entry.GetDbName(); got != want {
				t.Logf("entry %d: db name = %q, want %q", i, got, want)
				return false
			}
			if got, want := finding.GetQueryStatement(), entry.GetQueryStatement(); got != want {
				t.Logf("entry %d: query statement = %q, want %q", i, got, want)
				return false
			}
			if got, want := finding.GetNormalizedQueryTemplate(), entry.GetNormalizedQueryTemplate(); got != want {
				t.Logf("entry %d: normalized query template = %q, want %q", i, got, want)
				return false
			}
		}

		// A NIL entry fails extraction, so Predict returns an empty finding
		// whose identifying fields are all the empty string (R7.4). GetDbUser
		// etc. on a non-nil entry built from the empty entry would likewise be
		// "", so we assert the empty-string contract directly.
		nilFinding := model.Predict(nil)
		if nilFinding == nil {
			t.Logf("Predict(nil) returned nil finding")
			return false
		}
		if nilFinding.GetDbUser() != "" {
			t.Logf("nil entry: db user = %q, want \"\"", nilFinding.GetDbUser())
			return false
		}
		if nilFinding.GetDbName() != "" {
			t.Logf("nil entry: db name = %q, want \"\"", nilFinding.GetDbName())
			return false
		}
		if nilFinding.GetQueryStatement() != "" {
			t.Logf("nil entry: query statement = %q, want \"\"", nilFinding.GetQueryStatement())
			return false
		}
		if nilFinding.GetNormalizedQueryTemplate() != "" {
			t.Logf("nil entry: normalized query template = %q, want \"\"", nilFinding.GetNormalizedQueryTemplate())
			return false
		}

		return true
	}

	// Generator builds streams of non-nil entries with identifying fields drawn
	// from small alphabets, occasionally drawing the empty string for a field
	// (including the normalized query template, which makes the Action empty and
	// routes through the empty-Action window no-op). Draws are biased so both
	// empty-Action no-ops and well-formed filling/boundary paths occur.
	field := func(rng *rand.Rand, prefix string, mod int) string {
		// ~1 in 4 draws is the empty string so empty fields (and empty Actions)
		// are exercised alongside populated ones.
		if rng.Intn(4) == 0 {
			return ""
		}
		return fmt.Sprintf("%s%d", prefix, rng.Intn(mod))
	}

	cfg := &quick.Config{
		MaxCount: 500,
		Values: func(args []reflect.Value, rng *rand.Rand) {
			args[0] = reflect.ValueOf(uint8(rng.Intn(256)))
			args[1] = reflect.ValueOf(uint8(rng.Intn(256)))
			args[2] = reflect.ValueOf(uint16(rng.Intn(65536)))

			n := rng.Intn(40) // 0..39 entries per stream
			stream := make([]*modelpb.AuditLogEntry, n)
			for i := range stream {
				entry := &modelpb.AuditLogEntry{}
				entry.SetNormalizedQueryTemplate(field(rng, "q", 4))
				entry.SetDbUser(field(rng, "u", 3))
				entry.SetDbName(field(rng, "d", 3))
				entry.SetQueryStatement(field(rng, "s", 3))
				stream[i] = entry
			}
			args[3] = reflect.ValueOf(stream)
		},
	}

	if err := quick.Check(property, cfg); err != nil {
		t.Error(err)
	}
}

// Feature: consecutive-window-anomaly-detection, Property 13: Reset returns the model to a fresh state
func TestConsecutiveResetReturnsFreshState(t *testing.T) {
	// A ResetLiveCounts() call must put the model back into exactly the state a
	// freshly constructed model with the same config starts in: both windows
	// empty, the configured W/B/threshold untouched, and — critically — every
	// subsequent prediction behaving as if the pre-reset stream had never been
	// seen. The property pins all three facets of Requirement 9:
	//
	//   - R9.2 (empties both windows, preserves config): after processing an
	//     arbitrary "part one" prefix and then calling ResetLiveCounts(), the
	//     Test_Window and Baseline_Window each hold zero entries (test.size == 0,
	//     baseline.size == 0, and no retained baseline blocks), while w, b, and
	//     threshold are unchanged from their values recorded before the reset.
	//   - R9.1 / R9.2 (reset truly restores a fresh state): feeding an arbitrary
	//     "part two" suffix through the reset model yields, finding-by-finding,
	//     the SAME sequence (same IsAnomalousPredict, bit-equal ConfidenceScore)
	//     a second model built fresh from the identical config produces when fed
	//     ONLY part two. If the reset left any residue of part one in either
	//     window, a boundary in part two would see a different baseline and the
	//     finding sequences would diverge.
	//   - R9.3 (idempotent / panic-free when already empty): calling
	//     ResetLiveCounts() on an already-empty model (and a second time in a
	//     row) does not panic and leaves both windows empty.
	//
	// The split index is drawn anywhere in [0, len(stream)] so the part-one
	// prefix ranges from empty (reset right after construction) through the whole
	// stream (reset after every entry), exercising resets mid-fill, exactly on a
	// boundary, and after several baseline folds and evictions.

	// consecutiveProto builds a valid consecutive-window Model proto via the
	// generated opaque builder API, matching the other tests in this file.
	consecutiveProto := func(w, b int64, threshold float64) *modelpb.Model {
		cfg := (&modelpb.ConsecutiveWindowTrainingConfig_builder{
			W:                w,
			B:                b,
			AnomalyThreshold: threshold,
		}).Build()
		scenario := (&modelpb.ConsecutiveWindowScenario_builder{Config: cfg}).Build()
		return (&modelpb.Model_builder{ConsecutiveWindow: scenario}).Build()
	}

	// feed runs every Action in the slice through Predict as a well-formed entry
	// and returns the findings in order, so two models can be driven over the
	// same suffix and compared finding-by-finding.
	feed := func(model *ConsecutiveWindowModel, actions []string) []*modelpb.AnomalyFinding {
		findings := make([]*modelpb.AnomalyFinding, 0, len(actions))
		for _, action := range actions {
			entry := &modelpb.AuditLogEntry{}
			entry.SetNormalizedQueryTemplate(action)
			findings = append(findings, model.Predict(entry))
		}
		return findings
	}

	property := func(rawW uint8, rawK uint8, rawSplit uint16, actions []string) bool {
		// W in 2..6; k in 2..4 so B = kW is a multiple of W strictly greater
		// than W. A finite positive threshold keeps the model valid.
		w := int(rawW%5) + 2
		k := int(rawK%3) + 2
		b := k * w
		threshold := 5.0

		// Split the stream at a random index in [0, len(actions)]: part one is
		// the prefix fed-then-reset, part two the suffix replayed on both models.
		var split int
		if len(actions) > 0 {
			split = int(rawSplit) % (len(actions) + 1)
		}
		part1 := actions[:split]
		part2 := actions[split:]

		// modelReset: feed part one, record the config, reset, then feed part two.
		modelReset, err := NewConsecutiveWindowModel(consecutiveProto(int64(w), int64(b), threshold))
		if err != nil || modelReset == nil {
			t.Logf("failed to build reset model: model=%v err=%v", modelReset, err)
			return false
		}
		feed(modelReset, part1)

		// Record W/B/threshold before the reset so we can assert they survive it.
		wBefore := modelReset.w
		bBefore := modelReset.b
		thresholdBefore := modelReset.threshold

		modelReset.ResetLiveCounts()

		// R9.2: the config is unchanged across the reset.
		if modelReset.w != wBefore || modelReset.b != bBefore || modelReset.threshold != thresholdBefore {
			t.Logf("reset changed config: w %d->%d, b %d->%d, threshold %v->%v",
				wBefore, modelReset.w, bBefore, modelReset.b, thresholdBefore, modelReset.threshold)
			return false
		}

		// R9.2: both windows are empty right after the reset — the Test_Window
		// holds zero Actions and the Baseline_Window holds zero Actions with no
		// retained blocks.
		if modelReset.test.size != 0 {
			t.Logf("reset left test.size=%d, want 0", modelReset.test.size)
			return false
		}
		if modelReset.baseline.size != 0 {
			t.Logf("reset left baseline.size=%d, want 0", modelReset.baseline.size)
			return false
		}
		if len(modelReset.baseline.blocks) != 0 {
			t.Logf("reset left %d baseline blocks, want 0", len(modelReset.baseline.blocks))
			return false
		}

		resetFindings := feed(modelReset, part2)

		// modelFresh: a second model built from the identical config, fed ONLY
		// part two. Its findings are the ground truth for "as if part one never
		// happened".
		modelFresh, err := NewConsecutiveWindowModel(consecutiveProto(int64(w), int64(b), threshold))
		if err != nil || modelFresh == nil {
			t.Logf("failed to build fresh model: model=%v err=%v", modelFresh, err)
			return false
		}
		freshFindings := feed(modelFresh, part2)

		// R9.1/R9.2: the two suffix finding sequences match element-by-element —
		// same anomaly flag and bit-equal confidence — so the reset left no trace
		// of part one in either window.
		if len(resetFindings) != len(freshFindings) {
			t.Logf("finding-count mismatch: reset=%d fresh=%d", len(resetFindings), len(freshFindings))
			return false
		}
		for i := range resetFindings {
			r := resetFindings[i]
			f := freshFindings[i]
			if r == nil || f == nil {
				t.Logf("nil finding at index %d: reset=%v fresh=%v", i, r, f)
				return false
			}
			if r.GetIsAnomalousPredict() != f.GetIsAnomalousPredict() {
				t.Logf("finding %d anomaly flag differs: reset=%v fresh=%v",
					i, r.GetIsAnomalousPredict(), f.GetIsAnomalousPredict())
				return false
			}
			if r.GetConfidenceScore() != f.GetConfidenceScore() {
				t.Logf("finding %d confidence differs: reset=%v fresh=%v",
					i, r.GetConfidenceScore(), f.GetConfidenceScore())
				return false
			}
		}

		// R9.3: resetting an already-empty model is a panic-free no-op that
		// leaves both windows empty — call it twice in a row on a fresh model
		// (whose windows start empty) and confirm the windows stay empty. A panic
		// here would propagate out of quick.Check and fail the test directly.
		idempotent, err := NewConsecutiveWindowModel(consecutiveProto(int64(w), int64(b), threshold))
		if err != nil || idempotent == nil {
			t.Logf("failed to build idempotence model: model=%v err=%v", idempotent, err)
			return false
		}
		idempotent.ResetLiveCounts()
		idempotent.ResetLiveCounts()
		if idempotent.test.size != 0 || idempotent.baseline.size != 0 || len(idempotent.baseline.blocks) != 0 {
			t.Logf("double reset on empty model left state: test.size=%d baseline.size=%d blocks=%d",
				idempotent.test.size, idempotent.baseline.size, len(idempotent.baseline.blocks))
			return false
		}

		return true
	}

	// Generator draws small W/k, a split selector, and an Action stream over a
	// small alphabet so blocks repeat Query_Types and the baseline FIFO churns
	// (overflowing B) for longer streams. Stream length includes 0 so the
	// reset-right-after-construction case is covered; lengths up to ~40 drive
	// many boundaries and several evictions before and after the split.
	cfg := &quick.Config{
		MaxCount: 500,
		Values: func(args []reflect.Value, rng *rand.Rand) {
			args[0] = reflect.ValueOf(uint8(rng.Intn(256)))
			args[1] = reflect.ValueOf(uint8(rng.Intn(256)))
			args[2] = reflect.ValueOf(uint16(rng.Intn(65536)))

			nStream := rng.Intn(41) // 0..40 Actions
			stream := make([]string, nStream)
			for i := range stream {
				stream[i] = fmt.Sprintf("q%d", rng.Intn(4))
			}
			args[3] = reflect.ValueOf(stream)
		},
	}

	if err := quick.Check(property, cfg); err != nil {
		t.Error(err)
	}
}
