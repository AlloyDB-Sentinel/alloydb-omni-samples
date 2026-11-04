package service

import (
	"math"

	"github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common"

	findingpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common/proto"
	auditpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common/proto"
	modelpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common/proto"
)

// notScoredConfidence is the confidence assigned to not-anomalous findings that
// do not come from a scored window: entries whose action is empty and entries
// that only advance the buffer while a window is still filling. It is a fixed,
// documented value so these "no decision yet" findings are reported
// consistently.
const notScoredConfidence = 0.0

// NGramScoringModel adapts the window-oriented NGramWindowScorer to the
// per-log-entry Model interface. It buffers actions across successive Predict
// calls until it holds windowLength actions, then scores the completed window
// as a non-overlapping (tumbling) window.
type NGramScoringModel struct {
	scorer       *NGramWindowScorer
	windowLength int
	buffer       []string
	// lastScoreErr holds the error from the most recent Score call that
	// failed, or nil if the most recent window completion scored cleanly (or no
	// window has completed yet). Predict swallows the scorer error in the
	// returned finding (R3.4, R4.4), so this field is the only seam through
	// which the server can observe and log that a scoring error occurred. It is
	// set on every window completion (to the error on failure, to nil on
	// success) and consumed via TakeLastScoreError.
	lastScoreErr error
}

// Compile-time assertion that NGramScoringModel satisfies the Model interface.
var _ Model = (*NGramScoringModel)(nil)

// NewNGramScoringModel constructs the model from a model proto by delegating to
// the NGramWindowScorer construction. On error it returns that error unchanged
// along with a nil model; on success it caches the scorer's window length and
// allocates an empty buffer.
func NewNGramScoringModel(pb *modelpb.Model) (*NGramScoringModel, error) {
	scorer, err := NewNGramWindowScorer(pb)
	if err != nil {
		return nil, err
	}
	return &NGramScoringModel{
		scorer:       scorer,
		windowLength: scorer.WindowLength(),
		buffer:       make([]string, 0, scorer.WindowLength()),
	}, nil
}

// threshold reports the scorer's calibrated anomaly threshold, delegating to
// NGramWindowScorer.Threshold.
func (m *NGramScoringModel) threshold() float64 { return m.scorer.Threshold() }

// windowConfidence maps a window score and the calibrated threshold to a
// confidence value in the inclusive range [0.0, 1.0].
//
// The transform depends only on the distance d = |score - threshold| and uses
// d / (1 + d), which is 0 at d == 0, strictly increasing in d, and saturates
// toward 1 as d grows. This satisfies Requirement 4.3: the result is
//   - bounded to [0.0, 1.0],
//   - deterministic (identical inputs yield an identical value), and
//   - monotonic in |score - threshold| (a larger distance never lowers the
//     confidence).
//
// NaN and infinite inputs are handled defensively so the result stays
// well-defined: a NaN distance yields 0.0 and an infinite distance yields 1.0.
func windowConfidence(score, threshold float64) float64 {
	d := math.Abs(score - threshold)
	if math.IsNaN(d) {
		return 0.0
	}
	if math.IsInf(d, 0) {
		return 1.0
	}
	return d / (1.0 + d)
}

// Predict buffers the entry's action and, when a window completes, scores it,
// translating the window result into an AnomalyFinding.
//
// This method is built up across tasks. The portion implemented here is the
// front guard of Predict:
//   - Feature extraction via common.Extract. A nil log entry (or any other
//     extraction failure) yields an empty &findingpb.AnomalyFinding{} with the
//     buffer left unchanged (R2.4, R2.5). The guard never touches the scorer's
//     threshold, so a model constructed without calibration never dereferences
//     a nil threshold on this path.
//   - Empty-action no-op: an entry whose extracted action is "" produces exactly
//     one not-anomalous finding and leaves the buffer unchanged (R3.6). This
//     path likewise does not reach threshold()/windowConfidence.
//
// Buffer accumulation and the "still filling" finding are implemented here
// (4.2). Window completion, tumbling reset, scoring, and window-score
// translation are implemented in task 4.3, continuing from the marked point
// below.
func (m *NGramScoringModel) Predict(logEntry *auditpb.AuditLogEntry) *findingpb.AnomalyFinding {
	features, err := common.Extract(logEntry)
	if err != nil {
		// Nil or otherwise unextractable entry: zero-value finding, buffer
		// untouched. Mirrors the other models' extraction-failure behavior.
		return &findingpb.AnomalyFinding{}
	}

	action := features.QueryTemplate
	if action == "" {
		// Empty actions are a no-op: one not-anomalous finding, buffer unchanged.
		return generateAnomalyFinding(logEntry, false, notScoredConfidence)
	}

	// Non-empty action: append it to the buffer as the last element, preserving
	// the arrival order of previously buffered actions (R3.1). The buffer
	// accumulates across successive Predict calls and is only reset on window
	// completion or ResetLiveCounts.
	m.buffer = append(m.buffer, action)

	if len(m.buffer) < m.windowLength {
		// Window still filling: the buffer holds strictly fewer than
		// windowLength actions, so there is no window to score yet. Return
		// exactly one not-anomalous "no decision yet" finding (R3.2). This path
		// never touches threshold()/windowConfidence, so an uncalibrated model
		// would not dereference a nil threshold here.
		return generateAnomalyFinding(logEntry, false, notScoredConfidence)
	}

	// Window complete: len(m.buffer) == m.windowLength here. The buffer grows
	// one action at a time and is cleared on each completion, so it can never
	// exceed windowLength — equality is the only way to reach this point.
	//
	// WINDOW COMPLETION + SCORING: score the full window, then apply the
	// TUMBLING RESET. The buffer is cleared on BOTH the success and error paths
	// so successive windows are non-overlapping (tumbling, R3.5) and the model
	// recovers cleanly after a scorer error (R3.4). The reset happens regardless
	// of the scorer outcome.
	window, err := m.scorer.Score(m.buffer)
	m.buffer = m.buffer[:0]
	// Record the outcome of this window completion so the server can observe it
	// (R4.4). A failed Score sets lastScoreErr to the error; a successful Score
	// clears it. The finding itself does not reveal the error (R3.4): the error
	// surfaces only through TakeLastScoreError, which the server drains and logs
	// at the Predict call site.
	m.lastScoreErr = err
	if err != nil {
		// SCORER ERROR HANDLING: return a not-anomalous finding with confidence
		// 0.0 and a cleared buffer; do not panic (R3.4, R4.4). The error is
		// recorded above for the server to log at the call site (task 6.2);
		// here we only handle it gracefully on the finding.
		return generateAnomalyFinding(logEntry, false, notScoredConfidence)
	}

	// Uncalibrated guard: windowConfidence(window.GetScore(), m.threshold())
	// dereferences the scorer's *threshold and would panic if the model were
	// constructed without calibration. A calibrated scorer also sets
	// window.IsAnomalous during Score; an uncalibrated one leaves it false. So
	// for an uncalibrated scorer we treat the window as not anomalous with
	// confidence 0.0 and never call m.threshold(). In practice a constructed
	// NGramScoringModel is always calibrated (R5), but this keeps Predict total
	// and non-panicking for any scorer state.
	if !m.scorer.HasThreshold() {
		return generateAnomalyFinding(logEntry, false, notScoredConfidence)
	}

	// TRANSLATION on success: the anomaly decision comes from the scorer, which
	// set IsAnomalous using the calibrated threshold with strict-greater
	// semantics (R4.1, R4.2). The window's numeric score reaches the finding
	// only through the confidence transform, which is monotonic in the score's
	// distance from the threshold (R4.3). The AnomalyFinding shape carries only
	// the identifying fields plus is_anomalous_predict and confidence_score, so
	// there is nothing further to mirror from the WindowScore.
	isAnomalous := window.GetIsAnomalous()
	confidence := windowConfidence(window.GetScore(), m.threshold())
	return generateAnomalyFinding(logEntry, isAnomalous, confidence)
}

// ResetLiveCounts clears the Window_Buffer so a fresh prediction request starts
// with an empty window and no window ever spans two separate requests (R6.5).
// The prediction server invokes this (via the optional
// interface{ ResetLiveCounts() } assertion in resetModels) before the first log
// entry of a batch or file request. The underlying capacity is preserved so the
// next request reuses the same backing array.
func (m *NGramScoringModel) ResetLiveCounts() {
	m.buffer = m.buffer[:0]
	m.lastScoreErr = nil
}

// TakeLastScoreError returns the error from the most recent window completion
// that failed in Predict and clears it, so a subsequent call reports nil until
// the next failing completion. It returns nil when the most recent completion
// scored cleanly or no window has completed yet.
//
// Predict degrades a scorer error to a not-anomalous, zero-confidence finding
// (R3.4, R4.4); the AnomalyFinding carries no error field. This accessor is the
// seam the Prediction_Server uses to observe and log that scoring failed,
// without changing the finding semantics. Draining on read keeps each error
// logged at most once at the Predict call site.
func (m *NGramScoringModel) TakeLastScoreError() error {
	err := m.lastScoreErr
	m.lastScoreErr = nil
	return err
}
