package service

import (
	"fmt"
	"math"
	"sort"

	"github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common"

	auditpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common/proto"
	findingpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common/proto"
	modelpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common/proto"
)

// testWindow is an append-only buffer of up to W Actions plus a live
// per-Query_Type count. It fills from empty to capacity W one Action at a time,
// is evaluated once at the Window_Boundary, and is then cleared back to empty.
//
// Unlike the rolling window used by ChiSquaredModel, a testWindow never evicts
// Action by Action: it only grows to W and resets to empty at a boundary. The
// count map is maintained incrementally so the contingency table can be built
// in O(K) over distinct Query_Types rather than O(W).
//
// Invariants maintained by append and clear:
//   - 0 <= size <= capacity
//   - sum(counts values) == size
//   - every value in counts is >= 1; no zero-count keys exist
//
// A zero-capacity testWindow (used by the nil-proto model) never becomes full,
// so it never triggers a Window_Boundary.
type testWindow struct {
	capacity int            // W (0 for the nil-proto model)
	size     int            // number of Actions currently buffered, in [0, W]
	counts   map[string]int // Query_Type -> count over the buffered Actions
}

// newTestWindow constructs an empty testWindow with the given capacity and an
// empty count map.
func newTestWindow(capacity int) *testWindow {
	return &testWindow{
		capacity: capacity,
		size:     0,
		counts:   make(map[string]int),
	}
}

// append admits action as the most-recently-arrived element of the current
// non-overlapping window, incrementing the buffered size and the per-Query_Type
// count (R3.1).
func (t *testWindow) append(action string) {
	t.size++
	t.counts[action]++
}

// full reports whether the testWindow has reached its capacity W and so
// completes the current non-overlapping window (R3.2). A zero-capacity window is
// never full, so the nil-proto model never triggers a boundary.
func (t *testWindow) full() bool {
	return t.size == t.capacity && t.capacity > 0
}

// clear resets the testWindow to empty after a boundary evaluation, allocating a
// fresh empty count map so the next W Actions form a new window that shares no
// Action with the completed one (R3.3).
func (t *testWindow) clear() {
	t.size = 0
	t.counts = make(map[string]int)
}

// baselineWindow accumulates completed W-blocks up to a total of B Actions. It
// is the tumbling Baseline_Window of the consecutive-window model.
//
// Each folded block is stored as its own per-Query_Type count map in FIFO order
// (oldest first), and a single aggregate count map is maintained incrementally
// on fold-in and eviction so the contingency table reads in O(K) over distinct
// Query_Types. Storing one count map per block makes whole-block eviction cheap
// and makes the whole-block invariant structural: it is impossible to represent
// a partially filled baseline block.
//
// Invariants maintained by fold:
//   - size == len(blocks) * blockSize, so size is always an exact integer
//     multiple of W with 0 <= size <= capacity (R4.4)
//   - sum(agg values) == size
//   - every value in agg is >= 1; no zero-count keys exist (the basis for
//     positive column totals and division safety, R6.6)
//   - blocks is ordered oldest-to-newest, so the block evicted on overflow is
//     always blocks[0], the earliest folded among those retained (R4.3)
type baselineWindow struct {
	blockSize int              // W: the size of each completed block
	capacity  int              // B: max total Actions (an exact multiple of W)
	blocks    []map[string]int // FIFO of completed-block count maps (len*W == size)
	size      int              // total Actions currently held, a multiple of W
	agg       map[string]int   // aggregate Query_Type -> count across all blocks
}

// newBaselineWindow constructs an empty baselineWindow with the given block size
// (W) and capacity (B), an empty FIFO of blocks, and an empty aggregate count
// map. A zero block size / capacity yields the empty baseline used by the
// nil-proto model.
func newBaselineWindow(blockSize, capacity int) *baselineWindow {
	return &baselineWindow{
		blockSize: blockSize,
		capacity:  capacity,
		blocks:    nil,
		size:      0,
		agg:       make(map[string]int),
	}
}

// fold absorbs a completed block (given as its per-Query_Type count map) at the
// FIFO tail, evicting the oldest block first when the baseline is already full
// (R4.1, R4.2, R4.3, R4.6).
//
// If the baseline is already at capacity, the oldest block (blocks[0]) is
// evicted: its counts are subtracted from agg (deleting keys that reach zero),
// it is dropped from the front of blocks, and size is decremented by blockSize.
// Because B is an exact multiple of W, a full baseline holds exactly B/W blocks
// and eviction restores room for exactly one new block. The new block's counts
// are then added into agg and a COPY of the block's count map is appended to the
// tail of blocks, growing size by blockSize.
//
// The incoming map is copied so that later clearing of the Test_Window (which
// shares that map) cannot disturb the folded block (R4.5 note).
func (bw *baselineWindow) fold(blockCounts map[string]int) {
	// At capacity: evict the oldest whole block before admitting the new one.
	if bw.size == bw.capacity {
		oldest := bw.blocks[0]
		for key, count := range oldest {
			bw.agg[key] -= count
			if bw.agg[key] == 0 {
				delete(bw.agg, key)
			}
		}
		bw.blocks = bw.blocks[1:]
		bw.size -= bw.blockSize
	}

	// Copy the incoming block so the baseline owns its own count map,
	// independent of the Test_Window that produced it.
	block := make(map[string]int, len(blockCounts))
	for key, count := range blockCounts {
		block[key] = count
		bw.agg[key] += count
	}
	bw.blocks = append(bw.blocks, block)
	bw.size += bw.blockSize
}

// ConsecutiveWindowModel detects divergence between a tumbling Baseline_Window
// of completed W-blocks and the most recently completed non-overlapping
// Test_Window of W Actions via a chi-squared test of homogeneity. It carries no
// trained model data; the baseline is built live from the stream.
type ConsecutiveWindowModel struct {
	w         int     // Test_Window capacity (W)
	b         int     // Baseline_Window capacity in Actions (B, a multiple of W)
	threshold float64 // anomaly threshold on the chi-squared statistic

	test     *testWindow     // append-only buffer of up to W Actions
	baseline *baselineWindow // FIFO of completed W-blocks up to B Actions
}

// Compile-time assertion that *ConsecutiveWindowModel satisfies the Model
// interface (R7.1).
var _ Model = (*ConsecutiveWindowModel)(nil)

// validateConsecutiveConfig rejects invalid W, B, and threshold parameters,
// returning a descriptive error for the first constraint violated (R2.7).
//
// Constraints are checked in a fixed order: W > 0, B > 0, B > W, B an integer
// multiple of W, and a finite threshold greater than zero. The b % w test runs
// only after w > 0 passes, so the modulo never divides by zero.
func validateConsecutiveConfig(w, b int, threshold float64) error {
	if w <= 0 {
		return fmt.Errorf("W must be greater than zero")
	}
	if b <= 0 {
		return fmt.Errorf("B must be greater than zero")
	}
	if b <= w {
		return fmt.Errorf("B must be greater than W")
	}
	if b%w != 0 {
		return fmt.Errorf("B must be an integer multiple of W")
	}
	if math.IsNaN(threshold) || math.IsInf(threshold, 0) || threshold <= 0 {
		return fmt.Errorf("anomaly threshold must be a finite value greater than zero")
	}
	return nil
}

// NewConsecutiveWindowModel constructs the model from proto, mirroring the
// constructor shape and conventions of NewChiSquaredModel.
//
// A nil proto yields a non-nil model with both windows empty (never flags) and a
// nil error. A proto whose active scenario is not consecutive-window yields a nil
// model and an error naming the expected and found scenarios. Otherwise the
// consecutive-window config is validated and, on success, W/B/threshold are
// copied into the struct, the Test_Window is allocated at capacity W, and the
// Baseline_Window is allocated with block size W and action cap B. The proto is
// read only through generated getters, so it is left unchanged.
func NewConsecutiveWindowModel(pb *modelpb.Model) (*ConsecutiveWindowModel, error) {
	// If the pb is nil, a new empty model is created.
	if pb == nil {
		return &ConsecutiveWindowModel{
			test:     newTestWindow(0),
			baseline: newBaselineWindow(0, 0),
		}, nil
	}

	// If the pb has no consecutive-window scenario, an error is returned.
	if !pb.HasConsecutiveWindow() {
		return nil, fmt.Errorf("invalid model scenario: expected ConsecutiveWindow, got %v", pb.WhichScenario())
	}

	cfg := pb.GetConsecutiveWindow().GetConfig()
	w := int(cfg.GetW())
	b := int(cfg.GetB())
	threshold := cfg.GetAnomalyThreshold()
	if err := validateConsecutiveConfig(w, b, threshold); err != nil {
		return nil, err
	}

	return &ConsecutiveWindowModel{
		w:         w,
		b:         b,
		threshold: threshold,
		test:      newTestWindow(w),
		baseline:  newBaselineWindow(w, b),
	}, nil
}

// consecutiveChiSquaredStatistic computes the chi-squared test-of-homogeneity
// statistic over the 2xK contingency table formed by the Baseline_Window
// aggregate and the completed Test_Window.
//
// NOTE: the design pseudocode names this chiSquaredStatistic, but that name is
// already taken by the ChiSquaredModel's window-based variant in the same
// `service` package (chi_squared_model.go). Go has no overloading, so this
// map-based variant carries a package-unique name. Task 8.1's Predict must call
// consecutiveChiSquaredStatistic(m.baseline.agg, m.baseline.size, completed, m.w).
//
// The two rows are the baseline (row total baseSize) and the completed
// Test_Window (row total W); the K columns are the sorted union of the
// Query_Types present in either map. Iterating a sorted union of keys makes the
// summation order deterministic, so identical window contents always yield an
// identical statistic regardless of Go map iteration order (R5.5, R6.5).
//
// For each column the expected count of a cell is (rowTotal * colTotal) / grand.
// The boundary path calls this only when baseSize > 0 and testTotal == W > 0, so
// both row totals are positive; every column total is positive because
// zero-count keys are never present in either map. Every expected count is
// therefore strictly positive and no division by zero can occur (R6.6).
//
// All arithmetic is in float64; the integer row/column/grand totals are promoted
// to float64.
func consecutiveChiSquaredStatistic(baseAgg map[string]int, baseSize int, testCounts map[string]int, w int) float64 {
	baseTotal := float64(baseSize) // baseline row total (R5.3)
	testTotal := float64(w)        // completed Test_Window row total == W (R5.3)
	grand := baseTotal + testTotal // grand total (R5.4)

	// Build the sorted union of Query_Types across both rows (R5.1, R5.2).
	keySet := make(map[string]struct{}, len(baseAgg)+len(testCounts))
	for key := range baseAgg {
		keySet[key] = struct{}{}
	}
	for key := range testCounts {
		keySet[key] = struct{}{}
	}
	keys := make([]string, 0, len(keySet))
	for key := range keySet {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	stat := 0.0
	for _, key := range keys {
		// Column total is positive by construction: a key is present only when
		// at least one row has a nonzero count for it.
		colTotal := float64(baseAgg[key] + testCounts[key])

		// Baseline cell: absent Query_Types default to a zero observed count.
		obsB := float64(baseAgg[key])
		expB := baseTotal * colTotal / grand // strictly > 0 (R6.6)
		diffB := obsB - expB
		stat += diffB * diffB / expB

		// Completed Test_Window cell.
		obsT := float64(testCounts[key])
		expT := testTotal * colTotal / grand // strictly > 0 (R6.6)
		diffT := obsT - expT
		stat += diffT * diffT / expT
	}
	return stat
}

// Predict admits one Action into the Test_Window and, at each Window_Boundary,
// evaluates the completed window against the Baseline_Window before folding it
// in.
//
// A nil entry or extraction failure returns an empty finding and leaves both
// windows unchanged (R7.5). An empty Action is an explicit window no-op: a
// non-anomalous, zero-confidence finding with no admission (R3.5, R8.5). A valid
// Action is appended to the Test_Window (R3.1); while the window is still
// filling the call returns a non-anomalous, zero-confidence finding (R3.2,
// R7.6).
//
// When the Test_Window reaches W a boundary is crossed. The completed block
// counts are snapshotted before any mutation. With an empty baseline no
// statistic is defined, so the finding is non-anomalous with zero confidence
// (R5.6); otherwise the chi-squared statistic over the baseline aggregate and
// the completed window drives the >= threshold anomaly decision (R6.3, R6.4) and
// the confidence transform. Evaluation happens strictly before the fold: the
// completed block is folded into the baseline (R4.1, R4.5), the Test_Window is
// cleared to start the next non-overlapping window (R3.3), and the finding
// computed before the fold is returned. Identifying fields are filled by the
// shared generateAnomalyFinding helper (R7.3, R7.4).
func (m *ConsecutiveWindowModel) Predict(logEntry *auditpb.AuditLogEntry) *findingpb.AnomalyFinding {
	features, err := common.Extract(logEntry)
	if err != nil {
		// Nil entry or extraction failure: windows unchanged (R7.5).
		return &findingpb.AnomalyFinding{}
	}

	action := features.QueryTemplate
	if action == "" {
		// Empty Action: explicit window no-op (R3.5, R8.5).
		return generateAnomalyFinding(logEntry, false, 0.0)
	}

	m.test.append(action) // R3.1
	if !m.test.full() {
		// Test_Window still filling (R3.2, R7.6).
		return generateAnomalyFinding(logEntry, false, 0.0)
	}

	// Window_Boundary: the Test_Window just reached W. Snapshot the completed
	// block counts before any mutation (evaluate-before-fold).
	completed := m.test.counts

	var finding *findingpb.AnomalyFinding
	if m.baseline.size == 0 {
		// Empty baseline: no statistic is defined (R5.6).
		finding = generateAnomalyFinding(logEntry, false, 0.0)
	} else {
		stat := consecutiveChiSquaredStatistic(m.baseline.agg, m.baseline.size, completed, m.w) // R6.1, R6.2
		isAnomaly := stat >= m.threshold                                                        // R6.3, R6.4
		conf := confidence(stat, m.threshold)                                                   // R8.1-R8.4
		finding = generateAnomalyFinding(logEntry, isAnomaly, conf)
	}

	m.baseline.fold(completed) // evaluate-before-fold; fold copies the map internally (R4.1, R4.5)
	m.test.clear()             // start the next non-overlapping window (R3.3)
	return finding
}

// ResetLiveCounts returns both windows to the empty state, leaving the
// configured W, B, and threshold unchanged (R9.2). It reallocates the
// Test_Window at capacity W and the Baseline_Window with block size W and action
// cap B, so a subsequent Predict stream starts from a clean baseline with no
// buffered Actions (R9.1). Calling it on an already-empty model is a harmless,
// panic-free no-op (R9.3). This is the duck-typed reset method the
// Prediction_Server discovers on stateful models.
func (m *ConsecutiveWindowModel) ResetLiveCounts() {
	m.test = newTestWindow(m.w)
	m.baseline = newBaselineWindow(m.w, m.b)
}
