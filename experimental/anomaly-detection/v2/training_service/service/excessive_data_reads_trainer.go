package service

import (
	"fmt"
	"math"
	"sort"
	"time"

	"github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/timestamppb"

	modelpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common/proto"
)

type excessiveDataReadsIdentity struct {
	dbName, dbUser string
}

// ExcessiveDataReadsTrainer learns weekday/hour baselines from hourly peak row totals.
// Training data must be clean and ordered by timestamp per database/user.
type ExcessiveDataReadsTrainer struct {
	config      *modelpb.ExcessiveDataReadsTrainingConfig
	shortWindow *common.SlidingRowWindow
	longWindow  *common.SlidingRowWindow
	peaks       map[excessiveDataReadsIdentity]map[int64][2]int64 // Absolute hour -> short and long window peaks.
	startMs     int64
	endMs       int64
	count       int64
	err         error
}

// NewExcessiveDataReadsTrainer validates the training interval and threshold multipliers.
func NewExcessiveDataReadsTrainer(config *modelpb.ExcessiveDataReadsTrainingConfig) (*ExcessiveDataReadsTrainer, error) {
	for _, c := range []float64{config.GetShortWindowStdDevMultiplier(), config.GetLongWindowStdDevMultiplier()} {
		if c <= 0 || math.IsNaN(c) || math.IsInf(c, 0) {
			return nil, fmt.Errorf("%w: excessive data reads multipliers must be finite and positive", ErrInvalidArgument)
		}
	}
	start, end := config.GetTrainingStartTime(), config.GetTrainingEndTime()
	if start == nil || end == nil || start.CheckValid() != nil || end.CheckValid() != nil {
		return nil, fmt.Errorf("%w: training start and end times are required", ErrInvalidArgument)
	}
	startMs, endMs := start.AsTime().UnixMilli(), end.AsTime().UnixMilli()
	hourMs := time.Hour.Milliseconds()
	if startMs < 0 || start.GetNanos() != 0 || end.GetNanos() != 0 ||
		startMs%hourMs != 0 || endMs%hourMs != 0 || endMs-startMs < 7*24*hourMs {
		return nil, fmt.Errorf("%w: training interval must span at least a week and align to UTC hours", ErrInvalidArgument)
	}
	short, _ := common.NewSlidingRowWindow(common.ExcessiveDataReadsShortWindow)
	long, _ := common.NewSlidingRowWindow(common.ExcessiveDataReadsLongWindow)
	return &ExcessiveDataReadsTrainer{
		config:      proto.Clone(config).(*modelpb.ExcessiveDataReadsTrainingConfig),
		shortWindow: short,
		longWindow:  long,
		peaks:       make(map[excessiveDataReadsIdentity]map[int64][2]int64),
		startMs:     startMs,
		endMs:       endMs,
	}, nil
}

// Count records hourly peaks. Reads before the training interval only warm the windows.
// Invalid input stops training; Err reports the failure.
func (t *ExcessiveDataReadsTrainer) Count(feature common.ExtractedLogFeatures) {
	if t.err != nil {
		return
	}
	if feature.TimestampMs >= t.endMs {
		t.err = fmt.Errorf("%w: read is outside the training interval", common.ErrInvalidArgument)
		return
	}
	short, err := t.shortWindow.Add(feature)
	if err != nil {
		t.err = err
		return
	}
	long, err := t.longWindow.Add(feature)
	if err != nil {
		t.err = err
		return
	}
	if feature.TimestampMs < t.startMs {
		return
	}
	key := excessiveDataReadsIdentity{feature.DbName, feature.DbUser}
	if t.peaks[key] == nil {
		t.peaks[key] = make(map[int64][2]int64)
	}
	hour := feature.TimestampMs / time.Hour.Milliseconds()
	peak := t.peaks[key][hour]
	t.peaks[key][hour] = [2]int64{max(peak[0], short), max(peak[1], long)}
	t.count++
}

// Err returns the first training error.
func (t *ExcessiveDataReadsTrainer) Err() error {
	return t.err
}

// BuildModel returns the hourly baselines, or nil if training failed.
func (t *ExcessiveDataReadsTrainer) BuildModel() *modelpb.Model {
	if t.err == nil && t.count == 0 {
		t.err = fmt.Errorf("%w: no reads in the training interval", common.ErrInvalidArgument)
	}
	if t.err != nil {
		return nil
	}
	keys := make([]excessiveDataReadsIdentity, 0, len(t.peaks))
	for key := range t.peaks {
		keys = append(keys, key)
	}
	sort.Slice(keys, func(i, j int) bool {
		if keys[i].dbName != keys[j].dbName {
			return keys[i].dbName < keys[j].dbName
		}
		return keys[i].dbUser < keys[j].dbUser
	})
	var entries []*modelpb.ExcessiveDataReadsBaselineEntry
	for _, key := range keys {
		var stats [common.ExcessiveDataReadsWeeklyHours][2]welfordAccumulator
		// Hours without reads contribute a zero peak.
		for ms := t.startMs; ms < t.endMs; ms += time.Hour.Milliseconds() {
			peak := t.peaks[key][ms/time.Hour.Milliseconds()]
			slot := common.ExcessiveDataReadsHourOfWeek(ms)
			stats[slot][0].update(float64(peak[0]))
			stats[slot][1].update(float64(peak[1]))
		}
		for slot, pair := range stats {
			entry := &modelpb.ExcessiveDataReadsBaselineEntry{}
			entry.SetDbName(key.dbName)
			entry.SetDbUser(key.dbUser)
			entry.SetHourOfWeek(int32(slot))
			entry.SetShortWindow(excessiveDataReadsBaseline(pair[0], t.config.GetShortWindowStdDevMultiplier()))
			entry.SetLongWindow(excessiveDataReadsBaseline(pair[1], t.config.GetLongWindowStdDevMultiplier()))
			entries = append(entries, entry)
		}
	}
	data := &modelpb.ExcessiveDataReadsModel{}
	data.SetEntries(entries)
	scenario := &modelpb.ExcessiveDataReadsScenario{}
	scenario.SetModelData(data)
	scenario.SetConfig(proto.Clone(t.config).(*modelpb.ExcessiveDataReadsTrainingConfig))
	metadata := &modelpb.ModelMetadata{}
	metadata.SetTrainingTime(timestamppb.Now())
	metadata.SetDataStartTime(timestamppb.New(time.UnixMilli(t.startMs)))
	metadata.SetDataEndTime(timestamppb.New(time.UnixMilli(t.endMs)))
	metadata.SetTotalLogEntriesProcessed(t.count)
	model := &modelpb.Model{}
	model.SetExcessiveDataReads(scenario)
	model.SetMetadata(metadata)
	return model
}

func excessiveDataReadsBaseline(stats welfordAccumulator, multiplier float64) *modelpb.VolumetricBaseline {
	b := stats.toVolumetricBaseline(multiplier)
	baseline := &modelpb.VolumetricBaseline{}
	baseline.SetMean(b.Mean)
	baseline.SetStdDev(b.StdDev)
	baseline.SetUpperThreshold(b.UpperThreshold)
	baseline.SetDeviationMargin(b.DeviationMargin)
	return baseline
}
