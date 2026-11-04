package service

import (
	"context"
	"fmt"
	"math"
	"sort"

	"github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/timestamppb"

	modelpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common/proto"
	predictionpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/prediction_service/proto"
)

const NGramScoringVersion = "variance_rms_v1"

type ngramBaseline struct{ mean, variance float64 }

// NGramWindowScorer uses frozen per-category window statistics, not overall
// histogram probabilities. Counts and training metadata are never updated here.
type NGramWindowScorer struct {
	windowLength int
	baseline     []map[[3]string]ngramBaseline
	config       *modelpb.NGramScoringConfig
	threshold    *float64
}

func resolvedScoringConfig(config *modelpb.NGramScoringConfig) (*modelpb.NGramScoringConfig, error) {
	if config == nil {
		config = &modelpb.NGramScoringConfig{}
		config.SetStandardDeviationFloorCount(1)
	}
	floor := config.GetStandardDeviationFloorCount()
	if !finite(floor) || floor <= 0 {
		return nil, fmt.Errorf("%w: standard_deviation_floor_count must be finite and positive", common.ErrInvalidArgument)
	}
	return proto.Clone(config).(*modelpb.NGramScoringConfig), nil
}

func NewNGramWindowScorer(model *modelpb.Model) (*NGramWindowScorer, error) {
	var config *modelpb.NGramScoringConfig
	if model != nil && model.HasNgramHistogram() {
		config = model.GetNgramHistogram().GetCalibration().GetScoringConfig()
	}
	return NewNGramWindowScorerWithConfig(model, config)
}

func NewNGramWindowScorerWithConfig(model *modelpb.Model, config *modelpb.NGramScoringConfig) (*NGramWindowScorer, error) {
	if model == nil || !model.HasNgramHistogram() {
		return nil, fmt.Errorf("%w: expected NgramHistogram scenario, got %v", common.ErrInvalidArgument, model.WhichScenario())
	}
	resolved, err := resolvedScoringConfig(config)
	if err != nil {
		return nil, err
	}
	scenario := model.GetNgramHistogram()
	cfg := scenario.GetConfig()
	maxN := int(cfg.GetMaxN())
	length := 300
	if cfg.HasWindowLength() {
		length = int(cfg.GetWindowLength())
	}
	if maxN < 1 || maxN > 3 || length < maxN {
		return nil, fmt.Errorf("%w: invalid histogram window configuration", common.ErrInvalidArgument)
	}
	if scenario.GetModelData().GetCompleteWindows() < 2 {
		return nil, fmt.Errorf("%w: variance scoring needs at least two complete training windows; retrain this model", common.ErrInvalidArgument)
	}
	scorer := &NGramWindowScorer{windowLength: length, baseline: make([]map[[3]string]ngramBaseline, maxN), config: resolved}
	if len(scenario.GetModelData().GetHistograms()) != maxN {
		return nil, fmt.Errorf("%w: missing histogram orders", common.ErrInvalidArgument)
	}
	for _, d := range scenario.GetModelData().GetHistograms() {
		n := int(d.GetN())
		if n < 1 || n > maxN || scorer.baseline[n-1] != nil || d.GetTotalWindows() <= 0 || len(d.GetEntries()) == 0 {
			return nil, fmt.Errorf("%w: invalid histogram order or empty baseline", common.ErrInvalidArgument)
		}
		categories := map[[3]string]ngramBaseline{}
		var probabilitySum, meanSum float64
		var count int64
		for _, e := range d.GetEntries() {
			probability := e.GetProbability()
			stats := e.GetWindowStatistics()
			if len(e.GetActions()) != n || e.GetCount() <= 0 || !finite(probability) || probability <= 0 || probability > 1 || math.Abs(probability-float64(e.GetCount())/float64(d.GetTotalWindows())) > 1e-9 {
				return nil, fmt.Errorf("%w: invalid histogram entry", common.ErrInvalidArgument)
			}
			if stats == nil || !stats.HasVariance() || !finite(stats.GetMean()) || stats.GetMean() < 0 || stats.GetMean() > 1 || !finite(stats.GetVariance()) || stats.GetVariance() < 0 {
				return nil, fmt.Errorf("%w: missing or invalid window mean/variance; retrain this model", common.ErrInvalidArgument)
			}
			var key [3]string
			for i, a := range e.GetActions() {
				if a == "" {
					return nil, fmt.Errorf("%w: empty action", common.ErrInvalidArgument)
				}
				key[i] = a
			}
			if _, exists := categories[key]; exists {
				return nil, fmt.Errorf("%w: duplicate histogram entry", common.ErrInvalidArgument)
			}
			categories[key] = ngramBaseline{stats.GetMean(), stats.GetVariance()}
			probabilitySum += probability
			meanSum += stats.GetMean()
			count += e.GetCount()
		}
		if math.Abs(probabilitySum-1) > 1e-9 || math.Abs(meanSum-1) > 1e-9 || count != d.GetTotalWindows() {
			return nil, fmt.Errorf("%w: baseline is not normalized", common.ErrInvalidArgument)
		}
		scorer.baseline[n-1] = categories
	}
	if calibration := scenario.GetCalibration(); calibration != nil {
		if calibration.GetScoringVersion() != NGramScoringVersion || !finite(calibration.GetThreshold()) || calibration.GetThreshold() < 0 || calibration.GetCompleteWindows() < 2 || !calibration.HasScoringConfig() {
			return nil, fmt.Errorf("%w: invalid or obsolete calibration; recalibrate for variance scoring", common.ErrInvalidArgument)
		}
		saved, err := resolvedScoringConfig(calibration.GetScoringConfig())
		if err != nil {
			return nil, err
		}
		if !proto.Equal(saved, resolved) {
			return nil, fmt.Errorf("%w: scoring configuration differs from calibration", common.ErrInvalidArgument)
		}
		threshold := calibration.GetThreshold()
		scorer.threshold = &threshold
	}
	return scorer, nil
}

func finite(value float64) bool { return !math.IsNaN(value) && !math.IsInf(value, 0) }

// WindowLength returns the number of actions per scored window.
func (s *NGramWindowScorer) WindowLength() int { return s.windowLength }

// Threshold returns the calibrated anomaly threshold. It must only be called
// when the scorer is calibrated; callers should guard with HasThreshold (or the
// presence of a threshold reported elsewhere, e.g. the calibration metadata on
// the model).
func (s *NGramWindowScorer) Threshold() float64 { return *s.threshold }

// HasThreshold reports whether the scorer carries a calibrated anomaly
// threshold. It is false for a scorer built from a model without calibration,
// in which case Threshold must not be called. Callers use this to guard the
// threshold dereference performed by Threshold.
func (s *NGramWindowScorer) HasThreshold() bool { return s.threshold != nil }

// Score compares each known category's window proportion with its learned mean
// using max(learned sigma, floor_count/positions). Unknown sequences share one
// additional bucket per order. The bucket is included even when its count is zero.
func (s *NGramWindowScorer) Score(actions []string) (*predictionpb.WindowScore, error) {
	if len(actions) != s.windowLength {
		return nil, fmt.Errorf("%w: expected %d actions", common.ErrInvalidArgument, s.windowLength)
	}
	for _, a := range actions {
		if a == "" {
			return nil, fmt.Errorf("%w: empty action", common.ErrInvalidArgument)
		}
	}
	window := &predictionpb.WindowScore{}
	var orders []*predictionpb.NGramOrderScore
	var deviations []*predictionpb.NGramDeviation
	var squaredTotal float64
	for order, baseline := range s.baseline {
		n := order + 1
		counts := map[[3]string]int{}
		for start := 0; start+n <= len(actions); start++ {
			var key [3]string
			copy(key[:], actions[start:start+n])
			counts[key]++
		}
		positions := float64(len(actions) - n + 1)
		floor := s.config.GetStandardDeviationFloorCount() / positions
		if floor == 0 || !finite(floor) {
			return nil, fmt.Errorf("%w: standard-deviation floor cannot be represented", common.ErrInvalidArgument)
		}
		keys := make([][3]string, 0, len(baseline))
		for key := range baseline {
			keys = append(keys, key)
		}
		sort.Slice(keys, func(i, j int) bool {
			for k := 0; k < n; k++ {
				if keys[i][k] != keys[j][k] {
					return keys[i][k] < keys[j][k]
				}
			}
			return false
		})
		categories := len(baseline) + 1 // Fixed denominator includes the unknown bucket.
		var squaredSum float64
		addDeviation := func(key [3]string, b ngramBaseline, count int, unknown bool) error {
			observed := float64(count) / positions
			sigma := math.Max(math.Sqrt(b.variance), floor)
			z := (observed - b.mean) / sigma
			squared := z * z
			if !finite(squared) {
				return fmt.Errorf("%w: scoring overflow; increase standard-deviation floor", common.ErrInvalidArgument)
			}
			squaredSum += squared
			if squared > 0 {
				d := &predictionpb.NGramDeviation{}
				d.SetN(int32(n))
				if !unknown {
					d.SetActions(append([]string(nil), key[:n]...))
				}
				d.SetIsUnknownBucket(unknown)
				d.SetObservedProportion(observed)
				d.SetLearnedMean(b.mean)
				d.SetLearnedVariance(b.variance)
				d.SetEffectiveStandardDeviation(sigma)
				d.SetStandardizedDeviation(z)
				d.SetSquaredDeviation(squared)
				d.SetScoreContribution(squared / float64(categories) / float64(len(s.baseline)))
				deviations = append(deviations, d)
			}
			return nil
		}
		for _, key := range keys {
			if err := addDeviation(key, baseline[key], counts[key], false); err != nil {
				return nil, err
			}
		}
		unknown := 0
		for key, count := range counts {
			if _, known := baseline[key]; !known {
				unknown += count
			}
		}
		if err := addDeviation([3]string{}, ngramBaseline{}, unknown, true); err != nil {
			return nil, err
		}
		meanSquared := squaredSum / float64(categories)
		component := &predictionpb.NGramOrderScore{}
		component.SetN(int32(n))
		component.SetScore(math.Sqrt(meanSquared))
		component.SetCategoryCount(int64(categories))
		component.SetUnknownOccurrences(int64(unknown))
		orders = append(orders, component)
		squaredTotal += meanSquared / float64(len(s.baseline))
	}
	if !finite(squaredTotal) {
		return nil, fmt.Errorf("%w: score overflow", common.ErrInvalidArgument)
	}
	window.SetOrderScores(orders)
	window.SetScore(math.Sqrt(squaredTotal))
	sort.Slice(deviations, func(i, j int) bool {
		a, b := deviations[i], deviations[j]
		if a.GetScoreContribution() != b.GetScoreContribution() {
			return a.GetScoreContribution() > b.GetScoreContribution()
		}
		if a.GetN() != b.GetN() {
			return a.GetN() < b.GetN()
		}
		if a.GetIsUnknownBucket() != b.GetIsUnknownBucket() {
			return !a.GetIsUnknownBucket()
		}
		for k := range a.GetActions() {
			if a.GetActions()[k] != b.GetActions()[k] {
				return a.GetActions()[k] < b.GetActions()[k]
			}
		}
		return false
	})
	if len(deviations) > 10 {
		deviations = deviations[:10]
	}
	window.SetTopDeviations(deviations)
	if s.threshold != nil {
		window.SetIsAnomalous(window.GetScore() > *s.threshold)
	}
	return window, nil
}

// ScoreFile resets window positions for each call. Only complete windows are
// scored; trailing actions are counted and reported. Memory holds one input
// window plus the returned report, not the full log stream.
func (s *NGramWindowScorer) ScoreFile(ctx context.Context, path string) (*predictionpb.WindowScoreReport, error) {
	report := &predictionpb.WindowScoreReport{}
	report.SetWindowLength(int32(s.windowLength))
	report.SetScoringVersion(NGramScoringVersion)
	report.SetScoringConfig(proto.Clone(s.config).(*modelpb.NGramScoringConfig))
	if s.threshold != nil {
		report.SetThreshold(*s.threshold)
	}
	var windows []*predictionpb.WindowScore
	actions := make([]string, 0, s.windowLength)
	var total int64
	err := common.IngestBinProtoStream(ctx, path, func(entry *modelpb.AuditLogEntry) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		if entry.GetNormalizedQueryTemplate() == "" {
			return fmt.Errorf("%w: empty action at record %d", common.ErrInvalidArgument, total)
		}
		total++
		actions = append(actions, entry.GetNormalizedQueryTemplate())
		if len(actions) == s.windowLength {
			window, err := s.Score(actions)
			if err != nil {
				return err
			}
			window.SetWindowIndex(int64(len(windows)))
			window.SetStartAction(total - int64(s.windowLength))
			window.SetEndAction(total)
			windows = append(windows, window)
			actions = actions[:0]
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	report.SetTotalActions(total)
	report.SetCompleteWindows(int64(len(windows)))
	report.SetTrailingActions(int64(len(actions)))
	report.SetWindows(windows)
	return report, nil
}

// FitNGramCalibration clones the baseline and adds only calibration metadata.
// Threshold = sortedScores[ceil(percentile*K)-1]; equality is not anomalous.
func FitNGramCalibration(model *modelpb.Model, report *predictionpb.WindowScoreReport, percentile float64) (*modelpb.Model, error) {
	scorer, err := NewNGramWindowScorerWithConfig(model, report.GetScoringConfig())
	if err != nil {
		return nil, err
	}
	if !finite(percentile) || percentile <= 0 || percentile >= 1 || report == nil || len(report.GetWindows()) < 2 || report.GetCompleteWindows() != int64(len(report.GetWindows())) || report.GetScoringVersion() != NGramScoringVersion || report.GetWindowLength() != int32(scorer.windowLength) || !report.HasScoringConfig() || !proto.Equal(scorer.config, report.GetScoringConfig()) || report.HasThreshold() || report.GetTrailingActions() < 0 || report.GetTrailingActions() >= int64(scorer.windowLength) || report.GetTotalActions() != report.GetCompleteWindows()*int64(scorer.windowLength)+report.GetTrailingActions() {
		return nil, fmt.Errorf("%w: calibration needs a percentile in (0,1) and at least two complete windows", common.ErrInvalidArgument)
	}
	scores := make([]float64, len(report.GetWindows()))
	var mean, m2 float64
	for i, w := range report.GetWindows() {
		v := w.GetScore()
		if !finite(v) || v < 0 {
			return nil, fmt.Errorf("%w: invalid calibration score", common.ErrInvalidArgument)
		}
		scores[i] = v
		delta := v - mean
		mean += delta / float64(i+1)
		m2 += delta * (v - mean)
		if !finite(mean) || !finite(m2) {
			return nil, fmt.Errorf("%w: calibration variance overflow; increase the standard-deviation floor", common.ErrInvalidArgument)
		}
	}
	sort.Float64s(scores)
	calibration := &modelpb.NGramCalibration{}
	calibration.SetPercentile(percentile)
	calibration.SetThreshold(scores[int(math.Ceil(percentile*float64(len(scores))))-1])
	calibration.SetCompleteWindows(report.GetCompleteWindows())
	calibration.SetTotalActions(report.GetTotalActions())
	calibration.SetTrailingActions(report.GetTrailingActions())
	calibration.SetScoreMean(mean)
	calibration.SetScoreVariance(math.Max(0, m2/float64(len(scores)-1)))
	calibration.SetScoringVersion(NGramScoringVersion)
	calibration.SetScoringConfig(proto.Clone(scorer.config).(*modelpb.NGramScoringConfig))
	calibration.SetCalibrationTime(timestamppb.Now())
	calibrated := proto.Clone(model).(*modelpb.Model)
	calibrated.GetNgramHistogram().SetCalibration(calibration)
	return calibrated, nil
}
