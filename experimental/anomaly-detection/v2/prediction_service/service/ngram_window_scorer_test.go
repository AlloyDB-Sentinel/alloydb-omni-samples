package service

import (
	"github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common"
	modelpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common/proto"
	predictionpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/prediction_service/proto"
	training "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/training_service/service"
	"google.golang.org/protobuf/proto"
	"math"
	"testing"
)

func histogramBaseline(t *testing.T, actions []string, maxN, length int) *modelpb.Model {
	t.Helper()
	trainer, err := training.NewNGramHistogramTrainer(maxN, length)
	if err != nil {
		t.Fatal(err)
	}
	// Repeat the fixture so its sample variance is defined (at least two windows).
	for repeat := 0; repeat < 2; repeat++ {
		for _, a := range actions {
			trainer.Count(common.ExtractedLogFeatures{QueryTemplate: a})
		}
	}
	return trainer.BuildModel()
}
func scoringConfig(floor float64) *modelpb.NGramScoringConfig {
	c := &modelpb.NGramScoringConfig{}
	c.SetStandardDeviationFloorCount(floor)
	return c
}

func TestNGramVarianceScores(t *testing.T) {
	baseline := histogramBaseline(t, []string{"A", "B"}, 1, 2)
	saved := proto.Clone(baseline)
	scorer, err := NewNGramWindowScorer(baseline)
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		actions []string
		want    float64
	}{
		{[]string{"A", "B"}, 0},
		// Known A/B plus a fixed unknown bucket: mean(z^2)=(1+1+0)/3.
		{[]string{"A", "A"}, math.Sqrt(2.0 / 3)},
		// Known categories both absent, and 2 unknown occurrences: (1+1+4)/3.
		{[]string{"C", "C"}, math.Sqrt(2)},
		{[]string{"C", "D"}, math.Sqrt(2)},
	} {
		result, err := scorer.Score(test.actions)
		if err != nil {
			t.Fatal(err)
		}
		if math.Abs(result.GetScore()-test.want) > 1e-12 {
			t.Fatalf("%v score=%g want=%g", test.actions, result.GetScore(), test.want)
		}
		if result.HasIsAnomalous() {
			t.Fatal("uncalibrated score must not classify")
		}
		if result.GetOrderScores()[0].GetCategoryCount() != 3 {
			t.Fatal("unknown bucket changed denominator")
		}
	}
	novel, _ := scorer.Score([]string{"C", "D"})
	top := novel.GetTopDeviations()
	if len(top) != 3 || !top[0].GetIsUnknownBucket() || top[0].GetStandardizedDeviation() != 2 || novel.GetOrderScores()[0].GetUnknownOccurrences() != 2 {
		t.Fatalf("incorrect unknown explanation: %v", novel)
	}
	var contribution float64
	for _, d := range top {
		contribution += d.GetScoreContribution()
		if d.GetEffectiveStandardDeviation() != 0.5 {
			t.Fatal("incorrect one-count floor")
		}
	}
	if math.Abs(contribution-novel.GetScore()*novel.GetScore()) > 1e-12 {
		t.Fatal("explanations do not explain squared score")
	}
	if !proto.Equal(baseline, saved) {
		t.Fatal("scoring mutated model")
	}
	for _, actions := range [][]string{nil, {"A"}, {"A", "B", "C"}, {"A", ""}} {
		if _, err := scorer.Score(actions); err == nil {
			t.Fatal("accepted incomplete/invalid window")
		}
	}
	ordered := histogramBaseline(t, []string{"A", "A", "B", "B"}, 3, 4)
	scorer, err = NewNGramWindowScorer(ordered)
	if err != nil {
		t.Fatal(err)
	}
	unchanged, _ := scorer.Score([]string{"A", "A", "B", "B"})
	if unchanged.GetScore() != 0 {
		t.Fatal("must use learned batch means, not global boundary-spanning probabilities")
	}
	result, _ := scorer.Score([]string{"A", "B", "A", "B"})
	orders := result.GetOrderScores()
	if len(orders) != 3 || orders[0].GetScore() != 0 || orders[1].GetScore() <= 0 || orders[2].GetScore() <= 0 {
		t.Fatal("ordering change was not detected")
	}
	var squared float64
	for _, o := range orders {
		squared += o.GetScore() * o.GetScore() / 3
	}
	if math.Abs(result.GetScore()-math.Sqrt(squared)) > 1e-12 {
		t.Fatal("incorrect order aggregation")
	}
}

func TestNGramLearnedVarianceAndFloor(t *testing.T) {
	stable := histogramBaseline(t, []string{"A", "B"}, 1, 2)
	variable := histogramBaseline(t, []string{"A", "A", "B", "B"}, 1, 2)
	stableScorer, _ := NewNGramWindowScorer(stable)
	variableScorer, _ := NewNGramWindowScorer(variable)
	a, _ := stableScorer.Score([]string{"A", "A"})
	b, _ := variableScorer.Score([]string{"A", "A"})
	if b.GetScore() >= a.GetScore() {
		t.Fatal("learned variation must reduce standardized deviation")
	}
	floored, err := NewNGramWindowScorerWithConfig(stable, scoringConfig(2))
	if err != nil {
		t.Fatal(err)
	}
	c, _ := floored.Score([]string{"A", "A"})
	if math.Abs(c.GetScore()-a.GetScore()/2) > 1e-12 {
		t.Fatal("configured floor not used")
	}
	for _, floor := range []float64{0, -1, math.NaN(), math.Inf(1)} {
		if _, err := NewNGramWindowScorerWithConfig(stable, scoringConfig(floor)); err == nil {
			t.Fatal("invalid floor accepted")
		}
	}
}

func calibrationReport(values ...float64) *predictionpb.WindowScoreReport {
	r := &predictionpb.WindowScoreReport{}
	r.SetWindowLength(2)
	r.SetCompleteWindows(int64(len(values)))
	r.SetTotalActions(int64(2 * len(values)))
	r.SetScoringVersion(NGramScoringVersion)
	r.SetScoringConfig(scoringConfig(1))
	var windows []*predictionpb.WindowScore
	for _, v := range values {
		w := &predictionpb.WindowScore{}
		w.SetScore(v)
		windows = append(windows, w)
	}
	r.SetWindows(windows)
	return r
}

func TestNGramCalibrationPreservesBaseline(t *testing.T) {
	baseline := histogramBaseline(t, []string{"A", "B"}, 1, 2)
	before := proto.Clone(baseline)
	report := calibrationReport(8, 0, 4, 2)
	calibrated, err := FitNGramCalibration(baseline, report, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	cal := calibrated.GetNgramHistogram().GetCalibration()
	if cal.GetThreshold() != 2 || math.Abs(cal.GetScoreMean()-3.5) > 1e-12 || math.Abs(cal.GetScoreVariance()-11.666666666666666) > 1e-12 || cal.GetScoringConfig().GetStandardDeviationFloorCount() != 1 {
		t.Fatalf("incorrect unbounded-score calibration: %v", cal)
	}
	if !proto.Equal(baseline, before) {
		t.Fatal("calibration mutated baseline")
	}
	clone := proto.Clone(calibrated).(*modelpb.Model)
	clone.GetNgramHistogram().ClearCalibration()
	if !proto.Equal(baseline, clone) {
		t.Fatal("calibration changed training statistics")
	}
	if _, err := NewNGramWindowScorerWithConfig(calibrated, scoringConfig(2)); err == nil {
		t.Fatal("calibrated floor may not be changed without recalibration")
	}
	for _, pct := range []float64{0, 1, -0.1, math.NaN(), math.Inf(1)} {
		if _, err := FitNGramCalibration(baseline, report, pct); err == nil {
			t.Fatal("invalid percentile accepted")
		}
	}
	if _, err := FitNGramCalibration(baseline, calibrationReport(0.1), 0.99); err == nil {
		t.Fatal("insufficient calibration accepted")
	}
	if _, err := FitNGramCalibration(baseline, calibrationReport(0, 1.7e154, 0, 1.7e154), 0.99); err == nil {
		t.Fatal("nonfinite calibration variance accepted")
	}
	calibrated, err = FitNGramCalibration(baseline, calibrationReport(0, 0), 0.99)
	if err != nil {
		t.Fatal(err)
	}
	data, err := proto.Marshal(calibrated)
	if err != nil {
		t.Fatal(err)
	}
	loaded := &modelpb.Model{}
	if err := proto.Unmarshal(data, loaded); err != nil {
		t.Fatal(err)
	}
	scorer, err := NewNGramWindowScorer(loaded)
	if err != nil {
		t.Fatal(err)
	}
	normal, _ := scorer.Score([]string{"A", "B"})
	anomaly, _ := scorer.Score([]string{"C", "C"})
	if !normal.HasIsAnomalous() || normal.GetIsAnomalous() || !anomaly.GetIsAnomalous() {
		t.Fatal("threshold equality or classification incorrect")
	}
	loaded.GetNgramHistogram().GetCalibration().SetScoringVersion("js_distance_base2_mean_v1")
	if _, err := NewNGramWindowScorer(loaded); err == nil {
		t.Fatal("old Jensen-Shannon threshold accepted")
	}
}

func TestNGramScorerRejectsInvalidBaseline(t *testing.T) {
	base := histogramBaseline(t, []string{"A", "B"}, 1, 2)
	for _, mutate := range []func(*modelpb.Model){
		func(m *modelpb.Model) { m.GetNgramHistogram().GetConfig().SetWindowLength(0) },
		func(m *modelpb.Model) { m.GetNgramHistogram().GetModelData().SetHistograms(nil) },
		func(m *modelpb.Model) { m.GetNgramHistogram().GetModelData().SetCompleteWindows(1) },
		func(m *modelpb.Model) {
			m.GetNgramHistogram().GetModelData().GetHistograms()[0].GetEntries()[0].SetProbability(math.NaN())
		},
		func(m *modelpb.Model) {
			m.GetNgramHistogram().GetModelData().GetHistograms()[0].GetEntries()[0].SetProbability(0.9)
		},
		func(m *modelpb.Model) {
			m.GetNgramHistogram().GetModelData().GetHistograms()[0].GetEntries()[0].ClearWindowStatistics()
		},
		func(m *modelpb.Model) {
			m.GetNgramHistogram().GetModelData().GetHistograms()[0].GetEntries()[0].GetWindowStatistics().ClearVariance()
		},
		func(m *modelpb.Model) {
			m.GetNgramHistogram().GetModelData().GetHistograms()[0].GetEntries()[0].GetWindowStatistics().SetVariance(-1)
		},
		func(m *modelpb.Model) {
			m.GetNgramHistogram().GetModelData().GetHistograms()[0].GetEntries()[0].GetWindowStatistics().SetMean(math.NaN())
		},
	} {
		m := proto.Clone(base).(*modelpb.Model)
		mutate(m)
		if _, err := NewNGramWindowScorer(m); err == nil {
			t.Fatal("accepted invalid model")
		}
	}
	if _, err := NewNGramWindowScorer(nil); err == nil {
		t.Fatal("nil model accepted")
	}
}
