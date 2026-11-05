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

// window is a fixed-capacity FIFO of Actions plus a live per-Query_Type count.
//
// It is a ring buffer paired with an incrementally maintained count map. The
// ring buffer gives O(1) amortized append and eviction with bounded memory
// (capacity fixed at construction); the count map lets the contingency-table
// build run in O(K) over distinct Query_Types rather than O(window size).
//
// Invariants maintained by append:
//   - size == min(number of appends, capacity)
//   - sum(counts values) == size
//   - every value in counts is >= 1; no zero-count keys exist
//   - the multiset of buf[head .. head+size) equals the multiset in counts
//
// Deleting zero-count keys keeps counts holding only present Query_Types, which
// is the basis for the contingency table having positive column totals (and
// hence division-by-zero safety in the chi-squared statistic).
type window struct {
	buf    []string       // ring storage, len == capacity (0 means zero-capacity)
	head   int            // index of the oldest element
	size   int            // number of elements currently stored
	counts map[string]int // Query_Type -> count over the elements currently stored
}

// newWindow constructs an empty window with the given capacity. A capacity of
// zero yields a window that holds no elements (used by the nil-proto model).
func newWindow(capacity int) *window {
	return &window{
		buf:    make([]string, capacity),
		head:   0,
		size:   0,
		counts: make(map[string]int),
	}
}

// append admits action as the most-recently-arrived element, evicting the
// oldest element first when the window is already at capacity (FIFO).
//
// A zero-capacity window can hold no elements, so append is a no-op for it.
func (w *window) append(action string) {
	capacity := len(w.buf)
	if capacity == 0 {
		return
	}

	// At capacity: evict the oldest element before admitting the new one.
	if w.size == capacity {
		evicted := w.buf[w.head]
		w.counts[evicted]--
		if w.counts[evicted] == 0 {
			delete(w.counts, evicted)
		}
		w.head = (w.head + 1) % capacity
		w.size--
	}

	// Write the new Action at the logical tail.
	tail := (w.head + w.size) % capacity
	w.buf[tail] = action
	w.size++
	w.counts[action]++
}

// ChiSquaredModel detects divergence between a historical and a recent rolling
// window of query-type Actions via a chi-squared test of homogeneity. It carries
// no trained model data; the baseline is built live from the stream.
type ChiSquaredModel struct {
	m         int     // historical window capacity (M)
	n         int     // recent window capacity (N)
	threshold float64 // anomaly threshold on the chi-squared statistic

	historical *window // rolling window of up to M Actions
	recent     *window // rolling window of up to N Actions
}

// Compile-time assertion that *ChiSquaredModel satisfies the Model interface.
var _ Model = (*ChiSquaredModel)(nil)

// validateConfig rejects invalid M, N, and threshold parameters, returning a
// descriptive error for the first constraint violated.
func validateConfig(m, n int, threshold float64) error {
	if m <= 0 {
		return fmt.Errorf("M must be greater than zero")
	}
	if n <= 0 {
		return fmt.Errorf("N must be greater than zero")
	}
	if n >= m {
		return fmt.Errorf("M must be greater than N")
	}
	if math.IsNaN(threshold) || math.IsInf(threshold, 0) || threshold <= 0 {
		return fmt.Errorf("anomaly threshold must be a finite value greater than zero")
	}
	return nil
}

// NewChiSquaredModel constructs the model from proto, mirroring the constructor
// shape and conventions of NewVolumetricSpikeModel.
//
// A nil proto yields a non-nil model with both windows empty (never flags) and a
// nil error. A proto whose active scenario is not chi-squared yields a nil model
// and an error naming the expected and found scenarios. Otherwise the chi-squared
// config is validated and, on success, M/N/threshold are copied into the struct
// and both windows are allocated at capacities M and N. The proto is read only
// through generated getters, so it is left unchanged.
func NewChiSquaredModel(pb *modelpb.Model) (*ChiSquaredModel, error) {
	// If the pb is nil, a new empty model is created.
	if pb == nil {
		return &ChiSquaredModel{
			historical: newWindow(0),
			recent:     newWindow(0),
		}, nil
	}

	// If the pb has no chi-squared scenario, an error is returned.
	if !pb.HasChiSquared() {
		return nil, fmt.Errorf("invalid model scenario: expected ChiSquared, got %v", pb.WhichScenario())
	}

	cfg := pb.GetChiSquared().GetConfig()
	m := int(cfg.GetM())
	n := int(cfg.GetN())
	threshold := cfg.GetAnomalyThreshold()
	if err := validateConfig(m, n, threshold); err != nil {
		return nil, err
	}

	return &ChiSquaredModel{
		m:          m,
		n:          n,
		threshold:  threshold,
		historical: newWindow(m),
		recent:     newWindow(n),
	}, nil
}

// chiSquaredStatistic computes the chi-squared test-of-homogeneity statistic
// over the 2xK contingency table formed by the historical and recent windows.
//
// The two rows are the windows; the K columns are the sorted union of the
// Query_Types present in either window. Iterating a sorted union of keys makes
// the summation order deterministic, so identical window contents always yield
// an identical statistic regardless of Go map iteration order (R5.5).
//
// For each column the expected count of a cell is (rowTotal * colTotal) / grand.
// Both callers guarantee the windows are non-empty, so both row totals are
// positive; every column total is positive because zero-count keys are never
// present in either count map. Every expected count is therefore strictly
// positive and no division by zero can occur (R5.6).
//
// All arithmetic is in float64; the integer row/column/grand totals are promoted
// to float64.
func chiSquaredStatistic(hist, recent *window) float64 {
	histTotal := float64(hist.size)   // historical row total (R4.3)
	recTotal := float64(recent.size)  // recent row total (R4.3)
	grand := histTotal + recTotal     // grand total (R4.4)

	// Build the sorted union of Query_Types across both windows (R4.1, R4.2).
	keySet := make(map[string]struct{}, len(hist.counts)+len(recent.counts))
	for key := range hist.counts {
		keySet[key] = struct{}{}
	}
	for key := range recent.counts {
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
		// at least one window has a nonzero count for it.
		colTotal := float64(hist.counts[key] + recent.counts[key])

		// Historical cell: absent Query_Types default to a zero observed count.
		obsH := float64(hist.counts[key])
		expH := histTotal * colTotal / grand // strictly > 0 (R5.6)
		diffH := obsH - expH
		stat += diffH * diffH / expH

		// Recent cell.
		obsR := float64(recent.counts[key])
		expR := recTotal * colTotal / grand // strictly > 0 (R5.6)
		diffR := obsR - expR
		stat += diffR * diffR / expR
	}
	return stat
}

// Predict admits the log entry's Action to both rolling windows and runs a
// chi-squared test of homogeneity on the current window contents.
//
// A nil log entry or extraction failure returns an empty finding (R6.5). An
// empty Action is a window no-op and returns a not-anomalous finding with zero
// confidence (R3.5, R7.4). Otherwise the Action is appended to both windows; if
// the recent window is not yet full or the historical window is empty, the
// warm-up guard returns a not-anomalous finding with zero confidence (R4.6,
// R6.6). Once both windows are populated, the statistic drives the anomaly
// decision (>= threshold) and the confidence score (R5.3, R5.4, R7.1-R7.3).
func (m *ChiSquaredModel) Predict(logEntry *auditpb.AuditLogEntry) *findingpb.AnomalyFinding {
	features, err := common.Extract(logEntry)
	if err != nil {
		return &findingpb.AnomalyFinding{}
	}

	action := features.QueryTemplate
	if action == "" {
		return generateAnomalyFinding(logEntry, false, 0.0)
	}

	m.historical.append(action)
	m.recent.append(action)

	// Warm-up guard: need a full recent sample and a non-empty historical window.
	if m.recent.size < m.n || m.historical.size == 0 {
		return generateAnomalyFinding(logEntry, false, 0.0)
	}

	stat := chiSquaredStatistic(m.historical, m.recent)
	isAnomaly := stat >= m.threshold
	conf := confidence(stat, m.threshold)
	return generateAnomalyFinding(logEntry, isAnomaly, conf)
}

// confidence maps a chi-squared statistic to a bounded confidence score using
// the squashing transform stat / (stat + threshold).
//
// For stat >= 0 and threshold > 0 (guaranteed by the constructor's config
// validation), the result lies in [0, 1): it is 0 at stat == 0 and approaches
// but never reaches 1 as stat grows (R7.1). It is a pure function of its inputs,
// so identical inputs yield identical output (R7.2). Its derivative in stat,
// threshold / (stat + threshold)^2, is strictly positive, so the score is
// strictly increasing in stat (R7.3). It equals exactly 0.5 when stat == threshold,
// i.e. at the anomaly decision boundary.
func confidence(stat, threshold float64) float64 {
	return stat / (stat + threshold)
}

// ResetLiveCounts returns both rolling windows to the empty state, leaving the
// configured M, N, and threshold unchanged (R8.2). It reallocates each window at
// its configured capacity, so a subsequent Predict stream starts from a clean
// baseline. It is idempotent and panic-free when the windows are already empty
// (R8.3). This is the duck-typed reset method the Prediction_Server discovers on
// stateful models (R8.1).
func (m *ChiSquaredModel) ResetLiveCounts() {
	m.historical = newWindow(m.m)
	m.recent = newWindow(m.n)
}
