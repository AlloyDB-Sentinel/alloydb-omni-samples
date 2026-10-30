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

// NGramHistogramTrainer counts overlapping windows of normalized query templates.
// The caller selects the training prefix; every supplied action is counted.
// File order defines one continuous sequence, including Markov walk restarts.
// Users, timing and ground-truth labels do not partition or filter the sequence.
// Count and BuildModel do not support concurrent calls.
type NGramHistogramTrainer struct {
	maxN           int
	counts         []map[[3]string]int64
	window         []string
	total          int64
	minTimestampMs int64
	maxTimestampMs int64
}

var _ Trainer = (*NGramHistogramTrainer)(nil)

func NewNGramHistogramTrainer(maxN int) (*NGramHistogramTrainer, error) {
	if maxN < 1 || maxN > 3 {
		return nil, fmt.Errorf("%w: max_n must be between 1 and 3", ErrInvalidArgument)
	}
	t := &NGramHistogramTrainer{maxN: maxN, counts: make([]map[[3]string]int64, maxN), minTimestampMs: math.MaxInt64, maxTimestampMs: math.MinInt64}
	for i := range t.counts {
		t.counts[i] = make(map[[3]string]int64)
	}
	return t, nil
}

func (t *NGramHistogramTrainer) Count(feature common.ExtractedLogFeatures) {
	t.total++
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
	}
}

func (t *NGramHistogramTrainer) BuildModel() *modelpb.Model {
	data := &modelpb.NGramHistogramModel{}
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
