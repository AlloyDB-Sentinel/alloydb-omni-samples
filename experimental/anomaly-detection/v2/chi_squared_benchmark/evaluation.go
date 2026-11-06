package main

import (
	"bufio"
	"context"
	"fmt"
	"os"

	"github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common"
	auditpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common/proto"
	"google.golang.org/protobuf/encoding/prototext"
)

type evaluation struct {
	Policy                 string   `json:"ground_truth_policy"`
	WarmupActions          int      `json:"warmup_actions"`
	TotalActions           int64    `json:"total_actions"`
	EvaluatedDecisions     int64    `json:"evaluated_decisions"`
	AnomalousActions       int64    `json:"anomalous_actions"`
	WarmupAnomalousActions int64    `json:"warmup_anomalous_actions"`
	TP                     int64    `json:"tp"`
	FP                     int64    `json:"fp"`
	TN                     int64    `json:"tn"`
	FN                     int64    `json:"fn"`
	Precision              *float64 `json:"precision,omitempty"`
	Recall                 *float64 `json:"recall,omitempty"`
	FalsePositiveRate      *float64 `json:"false_positive_rate,omitempty"`
}

func ratio(n, d int64) *float64 {
	if d == 0 {
		return nil
	}
	v := float64(n) / float64(d)
	return &v
}

// Evaluate every native rolling decision, not artificial tumbling windows.
// The first warmup actions still pass through the model; only evaluation skips
// them. Recent-window truth includes earlier labels, including warm-up labels.
func evaluate(ctx context.Context, input, findings string, n, warmup int) (evaluation, error) {
	e := evaluation{Policy: "any_anomalous_action_in_recent_window", WarmupActions: warmup}
	if n <= 0 || warmup < n {
		return e, fmt.Errorf("require positive N and warmup>=N")
	}
	f, err := os.Open(findings)
	if err != nil {
		return e, err
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 4096), 4<<20)
	recent := make([]bool, n)
	var active int
	err = common.IngestBinProtoStream(ctx, input, func(entry *auditpb.AuditLogEntry) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		index := e.TotalActions
		if entry.GetNormalizedQueryTemplate() == "" {
			return fmt.Errorf("empty input template at action %d", index)
		}
		if !scanner.Scan() {
			if err := scanner.Err(); err != nil {
				return err
			}
			return fmt.Errorf("missing prediction at action %d", index)
		}
		finding := &auditpb.AnomalyFinding{}
		if err := prototext.Unmarshal(scanner.Bytes(), finding); err != nil {
			return fmt.Errorf("finding %d: %w", index, err)
		}
		if finding.GetNormalizedQueryTemplate() != entry.GetNormalizedQueryTemplate() {
			return fmt.Errorf("finding/input mismatch at action %d", index)
		}
		slot := int(index % int64(n))
		if recent[slot] {
			active--
		}
		recent[slot] = entry.GetIsGroundTruthAnomalous()
		if recent[slot] {
			active++
			e.AnomalousActions++
		}
		e.TotalActions++
		if index < int64(warmup) {
			if recent[slot] {
				e.WarmupAnomalousActions++
			}
			return nil
		}
		e.EvaluatedDecisions++
		positive, predicted := active > 0, finding.GetIsAnomalousPredict()
		switch {
		case positive && predicted:
			e.TP++
		case positive:
			e.FN++
		case predicted:
			e.FP++
		default:
			e.TN++
		}
		return nil
	})
	if err != nil {
		return e, err
	}
	if scanner.Scan() {
		return e, fmt.Errorf("extra predictions after end of input")
	}
	if err = scanner.Err(); err != nil {
		return e, err
	}
	if e.EvaluatedDecisions == 0 {
		return e, fmt.Errorf("input has no decisions after warmup")
	}
	e.Precision = ratio(e.TP, e.TP+e.FP)
	e.Recall = ratio(e.TP, e.TP+e.FN)
	e.FalsePositiveRate = ratio(e.FP, e.FP+e.TN)
	return e, nil
}
