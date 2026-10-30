package service

import (
	"fmt"
	"math"
	"sort"
	"time"

	"github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common"
	"google.golang.org/protobuf/types/known/timestamppb"

	modelpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common/proto"
)

// NGramHistogramTrainer counts overlapping n-grams and measures their proportions
// across complete, nonoverlapping action windows. Every supplied action is counted.
// File order defines one continuous sequence, including Markov walk restarts.
// Users, timing and ground-truth labels do not partition or filter the sequence.
// Count and BuildModel do not support concurrent calls.
type NGramHistogramTrainer struct {
	maxN              int
	counts            []map[[3]string]int64
	window            []string
	total             int64
	minTimestampMs    int64
	maxTimestampMs    int64
	windowLength      int
	actionsInWindow   int
	completeWindows   int64
	windowCounts      []map[[3]string]int64
	proportionMoments []map[[3]string]proportionMoments
}

// Only nonzero proportions need updates. BuildModel merges the missing zero
// observations, including windows before a category's first appearance.
type proportionMoments struct {
	count    int64
	mean, m2 float64
}

func (m *proportionMoments) add(value float64) {
	m.count++
	delta := value - m.mean
	m.mean += delta / float64(m.count)
	m.m2 += delta * (value - m.mean)
}

func (m proportionMoments) statistics(total int64) *modelpb.NGramWindowStatistics {
	stats := &modelpb.NGramWindowStatistics{}
	stats.SetMean(m.mean * float64(m.count) / float64(total))
	if total > 1 {
		// Parallel Welford merge with a block of total-count zeros.
		m2 := m.m2 + m.mean*m.mean*float64(m.count)*float64(total-m.count)/float64(total)
		stats.SetVariance(math.Max(0, m2/float64(total-1)))
	}
	return stats
}

var _ Trainer = (*NGramHistogramTrainer)(nil)

func NewNGramHistogramTrainer(maxN, windowLength int) (*NGramHistogramTrainer, error) {
	if maxN < 1 || maxN > 3 {
		return nil, fmt.Errorf("%w: max_n must be between 1 and 3", ErrInvalidArgument)
	}
	if windowLength < maxN {
		return nil, fmt.Errorf("%w: window_length must be at least max_n", ErrInvalidArgument)
	}
	t := &NGramHistogramTrainer{
		maxN:              maxN,
		windowLength:      windowLength,
		counts:            make([]map[[3]string]int64, maxN),
		windowCounts:      make([]map[[3]string]int64, maxN),
		proportionMoments: make([]map[[3]string]proportionMoments, maxN),
		minTimestampMs:    math.MaxInt64,
		maxTimestampMs:    math.MinInt64,
	}
	for i := range t.counts {
		t.counts[i] = make(map[[3]string]int64)
		t.windowCounts[i] = make(map[[3]string]int64)
		t.proportionMoments[i] = make(map[[3]string]proportionMoments)
	}
	return t, nil
}

func (t *NGramHistogramTrainer) Count(feature common.ExtractedLogFeatures) {
	t.total++
	t.actionsInWindow++
	if feature.TimestampMs < t.minTimestampMs {
		t.minTimestampMs = feature.TimestampMs
	}
	if feature.TimestampMs > t.maxTimestampMs {
		t.maxTimestampMs = feature.TimestampMs
	}
	t.window = append(t.window, feature.QueryTemplate)
	if len(t.window) > t.maxN {
		t.window = t.window[1:]
	}
	for n := 1; n <= len(t.window); n++ {
		// Fixed-size keys preserve boundaries even for templates with separators.
		var key [3]string
		copy(key[:], t.window[len(t.window)-n:])
		t.counts[n-1][key]++
		if n <= t.actionsInWindow {
			t.windowCounts[n-1][key]++
		}
	}
	if t.actionsInWindow == t.windowLength {
		t.completeWindows++
		for i, counts := range t.windowCounts {
			denominator := float64(t.windowLength - i)
			for key, count := range counts {
				moments := t.proportionMoments[i][key]
				moments.add(float64(count) / denominator)
				t.proportionMoments[i][key] = moments
			}
			clear(counts)
		}
		t.actionsInWindow = 0
	}
}

func (t *NGramHistogramTrainer) BuildModel() *modelpb.Model {
	data := &modelpb.NGramHistogramModel{}
	data.SetCompleteWindows(t.completeWindows)
	data.SetTrailingActions(int64(t.actionsInWindow))
	var histograms []*modelpb.NGramHistogramDistribution
	for n := 1; n <= t.maxN; n++ {
		d := &modelpb.NGramHistogramDistribution{}
		d.SetN(int32(n))
		if t.total >= int64(n) {
			d.SetTotalWindows(t.total - int64(n) + 1)
		}
		var entries []*modelpb.NGramHistogramEntry
		for key, count := range t.counts[n-1] {
			e := &modelpb.NGramHistogramEntry{}
			e.SetActions(append([]string(nil), key[:n]...))
			e.SetCount(count)
			e.SetProbability(float64(count) / float64(d.GetTotalWindows()))
			if t.completeWindows > 0 {
				e.SetWindowStatistics(t.proportionMoments[n-1][key].statistics(t.completeWindows))
			}
			entries = append(entries, e)
		}
		sort.Slice(entries, func(i, j int) bool {
			for k := 0; k < n; k++ {
				if entries[i].GetActions()[k] != entries[j].GetActions()[k] {
					return entries[i].GetActions()[k] < entries[j].GetActions()[k]
				}
			}
			return false
		})
		d.SetEntries(entries)
		histograms = append(histograms, d)
	}
	data.SetHistograms(histograms)
	config := &modelpb.NGramHistogramTrainingConfig{}
	config.SetMaxN(int32(t.maxN))
	config.SetWindowLength(int32(t.windowLength))
	scenario := &modelpb.NGramHistogramScenario{}
	scenario.SetConfig(config)
	scenario.SetModelData(data)
	model := &modelpb.Model{}
	model.SetNgramHistogram(scenario)
	metadata := &modelpb.ModelMetadata{}
	metadata.SetTrainingTime(timestamppb.Now())
	metadata.SetTotalLogEntriesProcessed(t.total)
	if t.total > 0 {
		metadata.SetDataStartTime(timestamppb.New(time.UnixMilli(t.minTimestampMs)))
		metadata.SetDataEndTime(timestamppb.New(time.UnixMilli(t.maxTimestampMs)))
	}
	model.SetMetadata(metadata)
	return model
}
