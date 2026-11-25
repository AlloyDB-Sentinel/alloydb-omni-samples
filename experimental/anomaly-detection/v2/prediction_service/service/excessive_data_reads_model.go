package service

import (
	"fmt"
	"math"

	"github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common"

	modelpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common/proto"
)

// ExcessiveDataReadsModel flags a read when either rolling row total exceeds its baseline.
// Each instance tracks one prediction stream, separate from training state.
type ExcessiveDataReadsModel struct {
	baselines   map[common.ExcessiveDataReadsKey][2]VolumetricBaseline
	shortWindow *common.SlidingRowWindow
	longWindow  *common.SlidingRowWindow
	err         error
}

// NewExcessiveDataReadsModel loads and validates the saved hourly baselines.
func NewExcessiveDataReadsModel(pb *modelpb.Model) (*ExcessiveDataReadsModel, error) {
	if !pb.HasExcessiveDataReads() || len(pb.GetExcessiveDataReads().GetModelData().GetEntries()) == 0 {
		return nil, fmt.Errorf("excessive data reads model requires trained baselines")
	}
	m := &ExcessiveDataReadsModel{baselines: make(map[common.ExcessiveDataReadsKey][2]VolumetricBaseline)}
	counts := make(map[[2]string]int)
	for _, entry := range pb.GetExcessiveDataReads().GetModelData().GetEntries() {
		key := common.ExcessiveDataReadsKey{
			DbName:     entry.GetDbName(),
			DbUser:     entry.GetDbUser(),
			HourOfWeek: entry.GetHourOfWeek(),
		}
		if _, exists := m.baselines[key]; exists || key.HourOfWeek < 0 || key.HourOfWeek >= common.ExcessiveDataReadsWeeklyHours {
			return nil, fmt.Errorf("invalid or duplicate excessive data reads baseline: %v", key)
		}
		var pair [2]VolumetricBaseline
		for i, baseline := range []*modelpb.VolumetricBaseline{entry.GetShortWindow(), entry.GetLongWindow()} {
			if baseline == nil {
				return nil, fmt.Errorf("missing excessive data reads baseline for %v", key)
			}
			for _, value := range []float64{
				baseline.GetMean(), baseline.GetStdDev(),
				baseline.GetUpperThreshold(), baseline.GetDeviationMargin(),
			} {
				if value < 0 || math.IsNaN(value) || math.IsInf(value, 0) {
					return nil, fmt.Errorf("invalid excessive data reads baseline for %v", key)
				}
			}
			if baseline.GetUpperThreshold() != baseline.GetMean()+baseline.GetDeviationMargin() {
				return nil, fmt.Errorf("inconsistent excessive data reads threshold for %v", key)
			}
			pair[i] = VolumetricBaseline{
				Mean:            baseline.GetMean(),
				StdDev:          baseline.GetStdDev(),
				UpperThreshold:  baseline.GetUpperThreshold(),
				DeviationMargin: baseline.GetDeviationMargin(),
			}
		}
		m.baselines[key] = pair
		counts[[2]string{key.DbName, key.DbUser}]++
	}
	for key, count := range counts {
		if count != common.ExcessiveDataReadsWeeklyHours {
			return nil, fmt.Errorf("excessive data reads model needs 168 hourly baselines for %v, got %d", key, count)
		}
	}
	m.ResetLiveCounts()
	return m, nil
}

// Predict returns nil on invalid input; Err reports the failure until reset.
func (m *ExcessiveDataReadsModel) Predict(entry *modelpb.AuditLogEntry) *modelpb.AnomalyFinding {
	if m.err != nil {
		return nil
	}
	feature, err := common.Extract(entry)
	if err != nil {
		m.err = err
		return nil
	}
	short, err := m.shortWindow.Add(feature)
	if err != nil {
		m.err = err
		return nil
	}
	long, err := m.longWindow.Add(feature)
	if err != nil {
		m.err = err
		return nil
	}
	key := common.ExcessiveDataReadsKey{
		DbName:     feature.DbName,
		DbUser:     feature.DbUser,
		HourOfWeek: common.ExcessiveDataReadsHourOfWeek(feature.TimestampMs),
	}
	pair, ok := m.baselines[key]
	finding := generateAnomalyFinding(entry, !ok, 1)
	finding.SetTimestampMs(feature.TimestampMs)
	if !ok {
		return finding
	}
	anomalous, confidence := false, 0.0
	for i, rows := range [2]int64{short, long} {
		baseline := pair[i]
		if float64(rows) > baseline.UpperThreshold {
			anomalous = true
			score := 1.0
			if baseline.DeviationMargin > 0 {
				score -= baseline.DeviationMargin / (float64(rows) - baseline.Mean)
			}
			confidence = math.Max(confidence, score)
		}
	}
	if !anomalous {
		confidence = 1
	}
	finding.SetIsAnomalousPredict(anomalous)
	finding.SetConfidenceScore(confidence)
	return finding
}

// Err returns the first input error since the last reset.
func (m *ExcessiveDataReadsModel) Err() error {
	return m.err
}

// ResetLiveCounts clears both windows and the input error.
func (m *ExcessiveDataReadsModel) ResetLiveCounts() {
	m.shortWindow, _ = common.NewSlidingRowWindow(common.ExcessiveDataReadsShortWindow)
	m.longWindow, _ = common.NewSlidingRowWindow(common.ExcessiveDataReadsLongWindow)
	m.err = nil
}
