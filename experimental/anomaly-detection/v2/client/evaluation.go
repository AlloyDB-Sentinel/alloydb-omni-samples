package main

import (
	"context"
	"fmt"
	"sort"

	"github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common"

	clientpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/client/proto"
	auditpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common/proto"
	predictionpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/prediction_service/proto"
)

// evaluateWindows runs only after scoring. The prediction service never reads
// provenance or ground-truth labels. Any anomalous action makes a positive window;
// fractions are retained to expose diluted and boundary-crossing injections.
func evaluateWindows(ctx context.Context, path string, scores *predictionpb.WindowScoreReport) (*clientpb.WindowEvaluationReport, error) {
	if scores == nil || scores.GetWindowLength() <= 0 || scores.GetCompleteWindows() != int64(len(scores.GetWindows())) {
		return nil, fmt.Errorf("invalid window score report")
	}
	width := int64(scores.GetWindowLength())
	limit := scores.GetCompleteWindows() * width
	for i, w := range scores.GetWindows() {
		if !w.HasIsAnomalous() || w.GetWindowIndex() != int64(i) || w.GetStartAction() != int64(i)*width || w.GetEndAction() != int64(i+1)*width {
			return nil, fmt.Errorf("invalid scored window positions or missing classification")
		}
	}
	report := &clientpb.WindowEvaluationReport{}
	report.SetPositiveWindowPolicy("any_anomalous_action")
	segments := map[string]*clientpb.SegmentEvaluation{}
	var orderedSegments []*clientpb.SegmentEvaluation
	var windows []*clientpb.WindowEvaluation
	var total, anomalies, scoredAnomalies, trailingAnomalies, tp, fp, tn, fn int64
	var windowAnomalies int64
	windowSegments := map[string]bool{}
	err := common.IngestBinProtoStream(ctx, path, func(entry *auditpb.AuditLogEntry) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		position := total
		total++
		if entry.GetIsGroundTruthAnomalous() {
			anomalies++
			if position < limit {
				scoredAnomalies++
				windowAnomalies++
			} else {
				trailingAnomalies++
			}
			if id := entry.GetGroundTruthSegmentId(); id != "" {
				segment := segments[id]
				if segment == nil {
					segment = &clientpb.SegmentEvaluation{}
					segment.SetSegmentId(id)
					segment.SetProfileId(entry.GetGroundTruthProfile())
					segment.SetStartAction(position)
					segment.SetEndAction(position)
					segments[id] = segment
					orderedSegments = append(orderedSegments, segment)
				}
				if segment.GetEndAction() != position || segment.GetProfileId() != entry.GetGroundTruthProfile() {
					return fmt.Errorf("segment %q is noncontiguous or has inconsistent profiles", id)
				}
				segment.SetEndAction(position + 1)
				if position < limit {
					segment.SetScoredActions(segment.GetScoredActions() + 1)
					windowSegments[id] = true
				}
			}
		}
		if total <= limit && total%width == 0 {
			index := int(total/width - 1)
			prediction := scores.GetWindows()[index]
			positive, flagged := windowAnomalies > 0, prediction.GetIsAnomalous()
			switch {
			case positive && flagged:
				tp++
			case positive:
				fn++
			case flagged:
				fp++
			default:
				tn++
			}
			window := &clientpb.WindowEvaluation{}
			window.SetWindowIndex(int64(index))
			window.SetStartAction(total - width)
			window.SetEndAction(total)
			window.SetAnomalousActions(windowAnomalies)
			window.SetAnomalousFraction(float64(windowAnomalies) / float64(width))
			window.SetExpectedAnomalous(positive)
			window.SetPredictedAnomalous(flagged)
			window.SetScore(prediction.GetScore())
			ids := make([]string, 0, len(windowSegments))
			for id := range windowSegments {
				ids = append(ids, id)
				if flagged {
					segments[id].SetDetected(true)
				}
			}
			sort.Strings(ids)
			window.SetSegmentIds(ids)
			windows = append(windows, window)
			windowAnomalies = 0
			clear(windowSegments)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	if total != scores.GetTotalActions() || int64(len(windows)) != scores.GetCompleteWindows() || total-limit != scores.GetTrailingActions() {
		return nil, fmt.Errorf("ground-truth records do not match scored stream")
	}
	report.SetTotalActions(total)
	report.SetAnomalousActions(anomalies)
	report.SetScoredAnomalousActions(scoredAnomalies)
	report.SetTrailingAnomalousActions(trailingAnomalies)
	report.SetTruePositives(tp)
	report.SetFalsePositives(fp)
	report.SetTrueNegatives(tn)
	report.SetFalseNegatives(fn)
	if tp+fp > 0 {
		report.SetPrecision(float64(tp) / float64(tp+fp))
	}
	if tp+fn > 0 {
		report.SetRecall(float64(tp) / float64(tp+fn))
	}
	if fp+tn > 0 {
		report.SetFalsePositiveRate(float64(fp) / float64(fp+tn))
	}
	if 2*tp+fp+fn > 0 {
		report.SetF1(float64(2*tp) / float64(2*tp+fp+fn))
	}
	var scored, detected int64
	for _, segment := range orderedSegments {
		if segment.GetScoredActions() > 0 {
			scored++
			if segment.GetDetected() {
				detected++
			}
		}
	}
	report.SetScoredSegments(scored)
	report.SetDetectedSegments(detected)
	if scored > 0 {
		report.SetSegmentRecall(float64(detected) / float64(scored))
	}
	report.SetWindows(windows)
	report.SetSegments(orderedSegments)
	return report, nil
}
