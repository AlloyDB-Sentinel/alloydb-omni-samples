package main

import (
	"context"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"slices"
	"sort"
	"testing"

	"github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common"
	"google.golang.org/protobuf/encoding/protodelim"
	"google.golang.org/protobuf/proto"

	clientpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/client/proto"
	modelpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common/proto"
	predictionpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/prediction_service/proto"
)

func pipelineRecords(t *testing.T, path string) ([]*modelpb.AuditLogEntry, []string) {
	t.Helper()
	var entries []*modelpb.AuditLogEntry
	var actions []string
	err := common.IngestBinProtoStream(context.Background(), path, func(e *modelpb.AuditLogEntry) error {
		entries = append(entries, e)
		actions = append(actions, e.GetNormalizedQueryTemplate())
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	return entries, actions
}

// These references deliberately use full slices/JSON tuple keys rather than
// the trainer/scorer's streaming counters and fixed-array representation.
func pipelineCounts(actions []string, n int) map[string]int64 {
	counts := map[string]int64{}
	for start := 0; start+n <= len(actions); start++ {
		key, _ := json.Marshal(actions[start : start+n])
		counts[string(key)]++
	}
	return counts
}

type pipelineReferenceWindow struct {
	components    []float64
	categories    []int64
	unknownCounts []int64
	score         float64
}

func pipelineFloor(config *clientpb.MarkovExperimentConfig) float64 {
	if config.GetScoringConfig() == nil {
		return 1
	}
	return config.GetScoringConfig().GetStandardDeviationFloorCount()
}

// Independent reference works in raw count units, using explicit batch vectors
// and two-pass sample variances. Production works in proportion units and reads
// the streaming trainer's saved moments. Those formulations must agree.
func pipelineReferenceScores(training, actions []string, width, maxN int, floorCount float64) []pipelineReferenceWindow {
	type moments struct{ mean, variance float64 }
	baseline := make([]map[string]moments, maxN)
	batches := len(training) / width
	for n := 1; n <= maxN; n++ {
		counts := pipelineCounts(training, n)
		batchCounts := make([]map[string]int64, batches)
		for batch := range batchCounts {
			batchCounts[batch] = pipelineCounts(training[batch*width:(batch+1)*width], n)
		}
		baseline[n-1] = map[string]moments{}
		for key := range counts {
			var mean, variance float64
			for _, c := range batchCounts {
				mean += float64(c[key]) / float64(batches)
			}
			for _, c := range batchCounts {
				delta := float64(c[key]) - mean
				variance += delta * delta / float64(batches-1)
			}
			baseline[n-1][key] = moments{mean, variance}
		}
	}
	var result []pipelineReferenceWindow
	for start := 0; start+width <= len(actions); start += width {
		window := pipelineReferenceWindow{}
		var combinedSquared float64
		for order, categories := range baseline {
			counts := pipelineCounts(actions[start:start+width], order+1)
			keys := make([]string, 0, len(categories))
			for key := range categories {
				keys = append(keys, key)
			}
			sort.Strings(keys)
			var squaredSum float64
			for _, key := range keys {
				m := categories[key]
				sigma := math.Max(math.Sqrt(m.variance), floorCount)
				deviation := (float64(counts[key]) - m.mean) / sigma
				squaredSum += deviation * deviation
			}
			var unknown int64
			for key, count := range counts {
				if _, known := categories[key]; !known {
					unknown += count
				}
			}
			unknownZ := float64(unknown) / floorCount
			squaredSum += unknownZ * unknownZ
			meanSquared := squaredSum / float64(len(categories)+1)
			window.components = append(window.components, math.Sqrt(meanSquared))
			window.categories = append(window.categories, int64(len(categories)+1))
			window.unknownCounts = append(window.unknownCounts, unknown)
			combinedSquared += meanSquared / float64(maxN)
		}
		window.score = math.Sqrt(combinedSquared)
		result = append(result, window)
	}
	return result
}

func pipelineQuantile(scores []pipelineReferenceWindow, percentile float64) float64 {
	values := make([]float64, len(scores))
	for i, w := range scores {
		values[i] = w.score
	}
	sort.Float64s(values)
	return values[int(math.Ceil(percentile*float64(len(values))))-1]
}

func comparePipelineScores(t *testing.T, report *predictionpb.WindowScoreReport, reference []pipelineReferenceWindow, actions, width int, threshold *float64) float64 {
	t.Helper()
	if report.GetTotalActions() != int64(actions) || report.GetCompleteWindows() != int64(len(reference)) || len(report.GetWindows()) != len(reference) || report.GetWindowLength() != int32(width) || report.GetTrailingActions() != int64(actions%width) {
		t.Fatal("score report window counts/positions incorrect")
	}
	if report.HasThreshold() != (threshold != nil) {
		t.Fatal("incorrect threshold presence")
	}
	var maxError float64
	for i, w := range report.GetWindows() {
		if w.GetWindowIndex() != int64(i) || w.GetStartAction() != int64(i*width) || w.GetEndAction() != int64((i+1)*width) || len(w.GetOrderScores()) != len(reference[i].components) {
			t.Fatal("incorrect score window boundaries/orders")
		}
		for order, d := range w.GetOrderScores() {
			if d.GetN() != int32(order+1) || d.GetCategoryCount() != reference[i].categories[order] || d.GetUnknownOccurrences() != reference[i].unknownCounts[order] {
				t.Fatal("missing or reordered n-gram order score")
			}
			delta := math.Abs(d.GetScore() - reference[i].components[order])
			maxError = math.Max(maxError, delta)
			if delta > 1e-10 {
				t.Fatalf("window %d order %d RMS score=%g expected=%g", i, order+1, d.GetScore(), reference[i].components[order])
			}
		}
		delta := math.Abs(w.GetScore() - reference[i].score)
		maxError = math.Max(maxError, delta)
		if delta > 1e-10 {
			t.Fatalf("window %d score=%g expected=%g", i, w.GetScore(), reference[i].score)
		}
		if threshold == nil {
			if w.HasIsAnomalous() {
				t.Fatal("calibration classified itself")
			}
		} else if !w.HasIsAnomalous() || w.GetIsAnomalous() != (reference[i].score > *threshold) {
			t.Fatalf("window %d classification disagrees with independently calculated score/threshold", i)
		}
	}
	return maxError
}

func verifyPipelineReference(t *testing.T, out string, config *clientpb.MarkovExperimentConfig, baseline, calibrated *modelpb.Model, scores *predictionpb.WindowScoreReport, evaluation *clientpb.WindowEvaluationReport) {
	t.Helper()
	width, maxN := int(config.GetTrainingConfig().GetWindowLength()), int(config.GetTrainingConfig().GetMaxN())
	trainingEntries, training := pipelineRecords(t, filepath.Join(out, "logs.binproto"))
	// Assert that the original workload is loaded, not a duplicated normal matrix.
	source, err := os.ReadFile(config.GetGeneratorConfigPath())
	if err != nil {
		t.Fatal(err)
	}
	snapshot, err := os.ReadFile(filepath.Join(out, "generator_config.textproto"))
	if err != nil {
		t.Fatal(err)
	}
	if string(source) != string(snapshot) {
		t.Fatal("normal workload snapshot differs from source profile")
	}

	calibrationEntries, calibration := pipelineRecords(t, filepath.Join(out, "calibration", "logs.binproto"))
	validationEntries, validation := pipelineRecords(t, filepath.Join(out, "validation", "logs.binproto"))
	if int64(len(training)) != config.GetTrainingActions() || int64(len(calibration)) != config.GetCalibrationActions() || int64(len(validation)) != config.GetValidationActions() {
		t.Fatal("generated dataset lengths differ from experiment")
	}
	for _, entries := range [][]*modelpb.AuditLogEntry{trainingEntries, calibrationEntries} {
		for _, e := range entries {
			if e.GetIsGroundTruthAnomalous() || e.GetGroundTruthProfile() != "" || e.GetGroundTruthSegmentId() != "" {
				t.Fatal("training/calibration contaminated with anomaly labels")
			}
		}
	}
	vocabulary := map[string]bool{}
	for _, a := range training {
		vocabulary[a] = true
	}
	var expectedAnomalies int64
	for i, e := range validationEntries {
		var profile, segment string
		for _, s := range config.GetInjections() {
			if int64(i) >= s.GetStartAction() && int64(i) < s.GetStartAction()+s.GetNumActions() {
				profile, segment = s.GetProfileId(), s.GetSegmentId()
				break
			}
		}
		if e.GetIsGroundTruthAnomalous() != (segment != "") || e.GetGroundTruthSegmentId() != segment {
			t.Fatalf("incorrect label at %d", i)
		}
		if segment != "" {
			expectedAnomalies++
		} else {
			profile = "normal"
		}
		if e.GetGroundTruthProfile() != profile || !vocabulary[e.GetNormalizedQueryTemplate()] {
			t.Fatalf("new action or incorrect source profile at %d", i)
		}
		if i > 0 && e.GetTimestampMs() < validationEntries[i-1].GetTimestampMs() {
			t.Fatal("validation clock moved backwards")
		}
	}
	data := baseline.GetNgramHistogram().GetModelData()
	batches := len(training) / width
	if baseline.GetMetadata().GetTotalLogEntriesProcessed() != int64(len(training)) || data.GetCompleteWindows() != int64(batches) || data.GetTrailingActions() != int64(len(training)%width) {
		t.Fatal("incorrect baseline metadata")
	}
	var maxMeanError, maxVarianceError float64
	for _, d := range data.GetHistograms() {
		n := int(d.GetN())
		counts := pipelineCounts(training, n)
		if len(counts) != len(d.GetEntries()) || d.GetTotalWindows() != int64(len(training)-n+1) {
			t.Fatal("incorrect overall histogram")
		}
		batchCounts := make([]map[string]int64, batches)
		for i := range batchCounts {
			batchCounts[i] = pipelineCounts(training[i*width:(i+1)*width], n)
		}
		for _, entry := range d.GetEntries() {
			key, _ := json.Marshal(entry.GetActions())
			expectedCount := counts[string(key)]
			if entry.GetCount() != expectedCount || math.Abs(entry.GetProbability()-float64(expectedCount)/float64(len(training)-n+1)) > 1e-12 {
				t.Fatal("learned counts/probabilities disagree with raw training data")
			}
			values := make([]float64, batches)
			var mean, variance float64
			for i, c := range batchCounts {
				values[i] = float64(c[string(key)]) / float64(width-n+1)
				mean += values[i] / float64(batches)
			}
			for _, v := range values {
				variance += (v - mean) * (v - mean) / float64(batches-1)
			}
			stats := entry.GetWindowStatistics()
			if stats == nil || !stats.HasVariance() {
				t.Fatal("missing batch statistics")
			}
			maxMeanError = math.Max(maxMeanError, math.Abs(stats.GetMean()-mean))
			maxVarianceError = math.Max(maxVarianceError, math.Abs(stats.GetVariance()-variance))
			if maxMeanError > 1e-12 || maxVarianceError > 1e-12 {
				t.Fatal("learned window statistics disagree with explicit batch vectors")
			}
		}
	}
	calReference := pipelineReferenceScores(training, calibration, width, maxN, pipelineFloor(config))
	validationReference := pipelineReferenceScores(training, validation, width, maxN, pipelineFloor(config))
	threshold := pipelineQuantile(calReference, config.GetCalibrationPercentile())
	calReport := &predictionpb.WindowScoreReport{}
	readPipelineProto(t, filepath.Join(out, "calibration_scores.json"), calReport, true)
	maxDistanceError := comparePipelineScores(t, calReport, calReference, len(calibration), width, nil)
	maxDistanceError = math.Max(maxDistanceError, comparePipelineScores(t, scores, validationReference, len(validation), width, &threshold))
	cal := calibrated.GetNgramHistogram().GetCalibration()
	thresholdError := math.Abs(cal.GetThreshold() - threshold)
	if thresholdError > 1e-10 || math.Abs(scores.GetThreshold()-threshold) > 1e-10 || cal.GetCompleteWindows() != int64(len(calReference)) {
		t.Fatal("threshold is not the independent normal-calibration quantile")
	}
	var calMean, calVariance float64
	for _, w := range calReference {
		calMean += w.score / float64(len(calReference))
	}
	for _, w := range calReference {
		calVariance += (w.score - calMean) * (w.score - calMean) / float64(len(calReference)-1)
	}
	if math.Abs(cal.GetScoreMean()-calMean) > 1e-10 || math.Abs(cal.GetScoreVariance()-calVariance) > 1e-10 {
		t.Fatal("incorrect calibration summary statistics")
	}
	var tp, fp, tn, fn, scoredAnomalies int64
	for i, w := range evaluation.GetWindows() {
		var count int64
		ids := map[string]bool{}
		for _, e := range validationEntries[i*width : (i+1)*width] {
			if e.GetIsGroundTruthAnomalous() {
				count++
				ids[e.GetGroundTruthSegmentId()] = true
			}
		}
		positive, flag := count > 0, validationReference[i].score > threshold
		switch {
		case positive && flag:
			tp++
		case positive:
			fn++
		case flag:
			fp++
		default:
			tn++
		}
		scoredAnomalies += count
		segmentIDs := make([]string, 0, len(ids))
		for id := range ids {
			segmentIDs = append(segmentIDs, id)
		}
		sort.Strings(segmentIDs)
		if w.GetAnomalousActions() != count || math.Abs(w.GetAnomalousFraction()-float64(count)/float64(width)) > 1e-12 || w.GetExpectedAnomalous() != positive || w.GetPredictedAnomalous() != flag || !slices.Equal(w.GetSegmentIds(), segmentIDs) {
			t.Fatalf("window %d evaluation mismatch: got=%v expected count=%d positive=%v flag=%v segments=%v", i, w, count, positive, flag, segmentIDs)
		}
	}
	if len(evaluation.GetWindows()) != len(validationReference) || evaluation.GetAnomalousActions() != expectedAnomalies || evaluation.GetScoredAnomalousActions() != scoredAnomalies || evaluation.GetTrailingAnomalousActions() != expectedAnomalies-scoredAnomalies || evaluation.GetTruePositives() != tp || evaluation.GetFalsePositives() != fp || evaluation.GetTrueNegatives() != tn || evaluation.GetFalseNegatives() != fn {
		t.Fatal("confusion matrix/ground-truth totals incorrect")
	}
	if len(evaluation.GetSegments()) != len(config.GetInjections()) {
		t.Fatal("missing segment evaluation")
	}
	var scoredSegments, detectedSegments int64
	for i, s := range config.GetInjections() {
		end := s.GetStartAction() + s.GetNumActions()
		limit := int64(len(validationReference) * width)
		coverage := int64(math.Max(0, float64(min(end, limit)-s.GetStartAction())))
		detected := false
		for j, w := range validationReference {
			if w.score > threshold && int64(j*width) < end && int64((j+1)*width) > s.GetStartAction() {
				detected = true
			}
		}
		actual := evaluation.GetSegments()[i]
		if actual.GetSegmentId() != s.GetSegmentId() || actual.GetProfileId() != s.GetProfileId() || actual.GetStartAction() != s.GetStartAction() || actual.GetEndAction() != end || actual.GetScoredActions() != coverage || actual.GetDetected() != detected {
			t.Fatal("segment boundaries/coverage/detection incorrect")
		}
		if coverage > 0 {
			scoredSegments++
			if detected {
				detectedSegments++
			}
		}
	}
	if evaluation.GetScoredSegments() != scoredSegments || evaluation.GetDetectedSegments() != detectedSegments {
		t.Fatal("incorrect segment recall counts")
	}
	metrics := []struct {
		has                    bool
		actual                 float64
		numerator, denominator int64
	}{
		{evaluation.HasPrecision(), evaluation.GetPrecision(), tp, tp + fp}, {evaluation.HasRecall(), evaluation.GetRecall(), tp, tp + fn}, {evaluation.HasFalsePositiveRate(), evaluation.GetFalsePositiveRate(), fp, fp + tn}, {evaluation.HasF1(), evaluation.GetF1(), 2 * tp, 2*tp + fp + fn}, {evaluation.HasSegmentRecall(), evaluation.GetSegmentRecall(), detectedSegments, scoredSegments},
	}
	for _, m := range metrics {
		if m.has != (m.denominator > 0) || (m.has && math.Abs(m.actual-float64(m.numerator)/float64(m.denominator)) > 1e-12) {
			t.Fatal("incorrect evaluation metric or undefined-metric handling")
		}
	}
	summary := map[string]any{"independent_reference_passed": true, "normal_profile_config": config.GetGeneratorConfigPath(), "normal_profile_source_matches_snapshot": true, "scoring_version": scores.GetScoringVersion(), "standard_deviation_floor_count": pipelineFloor(config), "training_actions": len(training), "calibration_actions": len(calibration), "validation_actions": len(validation), "window_length": width, "max_mean_error": maxMeanError, "max_variance_error": maxVarianceError, "max_standardized_score_error": maxDistanceError, "threshold_error": thresholdError, "confusion": map[string]int64{"TP": tp, "FP": fp, "TN": tn, "FN": fn}, "scored_segments": scoredSegments, "detected_segments": detectedSegments, "trailing_anomalous_actions": expectedAnomalies - scoredAnomalies}
	encoded, _ := json.MarshalIndent(summary, "", "  ")
	if err := os.WriteFile(filepath.Join(out, "reference_verification.json"), append(encoded, '\n'), 0644); err != nil {
		t.Fatal(err)
	}
	t.Logf("Independent oracle: TP=%d FP=%d TN=%d FN=%d; mean error %.3g, variance error %.3g, standardized-score error %.3g, quantile error %.3g", tp, fp, tn, fn, maxMeanError, maxVarianceError, maxDistanceError, thresholdError)
}

func stripPipelineLabels(t *testing.T, input, output string) {
	t.Helper()
	entries, _ := pipelineRecords(t, input)
	f, err := os.Create(output)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range entries {
		e = proto.Clone(e).(*modelpb.AuditLogEntry)
		e.SetIsGroundTruthAnomalous(false)
		e.SetGroundTruthProfile("")
		e.SetGroundTruthSegmentId("")
		if _, err := protodelim.MarshalTo(f, e); err != nil {
			f.Close()
			t.Fatal(err)
		}
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
}

func verifyPipelineNormalControl(t *testing.T, out, normalDir string, config *clientpb.MarkovExperimentConfig) (int, int) {
	t.Helper()
	_, training := pipelineRecords(t, filepath.Join(out, "logs.binproto"))
	_, calibration := pipelineRecords(t, filepath.Join(out, "calibration", "logs.binproto"))
	records, actions := pipelineRecords(t, filepath.Join(normalDir, "logs.binproto"))
	for _, e := range records {
		if e.GetIsGroundTruthAnomalous() || e.GetGroundTruthSegmentId() != "" {
			t.Fatal("normal control is not normal")
		}
	}
	width, maxN := int(config.GetTrainingConfig().GetWindowLength()), int(config.GetTrainingConfig().GetMaxN())
	threshold := pipelineQuantile(pipelineReferenceScores(training, calibration, width, maxN, pipelineFloor(config)), config.GetCalibrationPercentile())
	reference := pipelineReferenceScores(training, actions, width, maxN, pipelineFloor(config))
	report := &predictionpb.WindowScoreReport{}
	readPipelineProto(t, filepath.Join(normalDir, "scored", "validation_scores.json"), report, true)
	comparePipelineScores(t, report, reference, len(actions), width, &threshold)
	var flags int
	for _, w := range reference {
		if w.score > threshold {
			flags++
		}
	}
	t.Logf("Independent normal-only control: %d/%d windows flagged (false positives)", flags, len(reference))
	return flags, len(reference)
}

func recordPipelineControls(t *testing.T, out string, falsePositives, normalWindows int) {
	t.Helper()
	path := filepath.Join(out, "reference_verification.json")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var summary map[string]any
	if err := json.Unmarshal(data, &summary); err != nil {
		t.Fatal(err)
	}
	summary["actual_client_cli_and_service_processes"] = true
	summary["validation_only_matches_full_pipeline"] = true
	summary["ground_truth_label_isolation_passed"] = true
	summary["uncalibrated_model_rejected"] = true
	summary["saved_model_bytes_unchanged"] = true
	summary["normal_only_false_positives"] = falsePositives
	summary["normal_only_windows"] = normalWindows
	summary["child_process_race_instrumentation"] = os.Getenv("MARKOV_E2E_RACE") == "1"
	data, err = json.MarshalIndent(summary, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, append(data, '\n'), 0644); err != nil {
		t.Fatal(err)
	}
}
