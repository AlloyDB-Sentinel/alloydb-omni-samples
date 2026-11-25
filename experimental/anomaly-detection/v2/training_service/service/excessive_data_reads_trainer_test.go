package service

import (
	"errors"
	"math"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common"
	"google.golang.org/protobuf/types/known/timestamppb"

	modelpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common/proto"
)

func excessiveDataReadsConfig() *modelpb.ExcessiveDataReadsTrainingConfig {
	start := time.Date(2026, 1, 5, 0, 0, 0, 0, time.UTC)
	config := &modelpb.ExcessiveDataReadsTrainingConfig{}
	config.SetShortWindowStdDevMultiplier(6)
	config.SetLongWindowStdDevMultiplier(4)
	config.SetTrainingStartTime(timestamppb.New(start))
	config.SetTrainingEndTime(timestamppb.New(start.Add(14 * 24 * time.Hour)))
	return config
}

func TestNewExcessiveDataReadsTrainer(t *testing.T) {
	tests := []struct {
		name string
		edit func(*modelpb.ExcessiveDataReadsTrainingConfig)
	}{
		{"zero_multiplier", func(c *modelpb.ExcessiveDataReadsTrainingConfig) { c.SetShortWindowStdDevMultiplier(0) }},
		{"nan_multiplier", func(c *modelpb.ExcessiveDataReadsTrainingConfig) { c.SetLongWindowStdDevMultiplier(math.NaN()) }},
		{"infinite_multiplier", func(c *modelpb.ExcessiveDataReadsTrainingConfig) { c.SetShortWindowStdDevMultiplier(math.Inf(1)) }},
		{"missing_start", func(c *modelpb.ExcessiveDataReadsTrainingConfig) { c.SetTrainingStartTime(nil) }},
		{"short_interval", func(c *modelpb.ExcessiveDataReadsTrainingConfig) { c.SetTrainingEndTime(c.GetTrainingStartTime()) }},
		{"partial_hour", func(c *modelpb.ExcessiveDataReadsTrainingConfig) {
			c.SetTrainingStartTime(timestamppb.New(c.GetTrainingStartTime().AsTime().Add(time.Minute)))
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			config := excessiveDataReadsConfig()
			test.edit(config)
			if _, err := NewExcessiveDataReadsTrainer(config); !errors.Is(err, ErrInvalidArgument) {
				t.Fatalf("NewExcessiveDataReadsTrainer() error = %v, want ErrInvalidArgument", err)
			}
		})
	}
	if _, err := NewExcessiveDataReadsTrainer(nil); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("nil config error = %v, want ErrInvalidArgument", err)
	}
}

func TestExcessiveDataReadsTrainerHourlyPeaks(t *testing.T) {
	config := excessiveDataReadsConfig()
	trainer, err := NewExcessiveDataReadsTrainer(config)
	if err != nil {
		t.Fatal(err)
	}
	start := config.GetTrainingStartTime().AsTime()
	for _, read := range []struct {
		db, user string
		offset   time.Duration
		rows     int64
	}{
		{"db", "alice", -30 * time.Minute, 100}, // Warmup affects only the long window.
		{"db", "alice", 0, 10},
		{"db", "alice", 30 * time.Minute, 20},
		{"db", "alice", 7 * 24 * time.Hour, 30},
		{"db", "bob", 0, 8},
		{"other", "alice", 0, 4},
	} {
		trainer.Count(common.ExtractedLogFeatures{
			DbName:       read.db,
			DbUser:       read.user,
			TimestampMs:  start.Add(read.offset).UnixMilli(),
			RowsReturned: read.rows,
		})
	}
	if err := trainer.Err(); err != nil {
		t.Fatal(err)
	}
	model := trainer.BuildModel()
	entries := model.GetExcessiveDataReads().GetModelData().GetEntries()
	if len(entries) != 3*168 || model.GetMetadata().GetTotalLogEntriesProcessed() != 5 {
		t.Fatalf("entries = %d, scored reads = %d; want 504 and 5", len(entries), model.GetMetadata().GetTotalLogEntriesProcessed())
	}
	// The two Monday peaks are (20, 130) and (30, 30), using population stddev.
	short, long := entries[0].GetShortWindow(), entries[0].GetLongWindow()
	if short.GetMean() != 25 || short.GetStdDev() != 5 || short.GetUpperThreshold() != 55 {
		t.Errorf("short baseline = %v, want mean 25, stddev 5, threshold 55", short)
	}
	if long.GetMean() != 80 || long.GetStdDev() != 50 || long.GetUpperThreshold() != 280 {
		t.Errorf("long baseline = %v, want mean 80, stddev 50, threshold 280", long)
	}
	if entries[1].GetLongWindow().GetUpperThreshold() != 0 {
		t.Error("an hour without reads must contribute a zero peak")
	}
	// Quiet weeks count as zero, with identities kept separate.
	if entries[168].GetShortWindow().GetMean() != 4 || entries[336].GetShortWindow().GetMean() != 2 {
		t.Error("incorrect per-identity baselines or missing zero-filled weeks")
	}
	config.SetShortWindowStdDevMultiplier(99)
	if trainer.BuildModel().GetExcessiveDataReads().GetConfig().GetShortWindowStdDevMultiplier() != 6 {
		t.Error("trainer retained mutable caller config")
	}
}

func TestExcessiveDataReadsTrainerRejectsInvalidStream(t *testing.T) {
	for _, offset := range []time.Duration{-time.Millisecond, 14 * 24 * time.Hour} {
		config := excessiveDataReadsConfig()
		trainer, err := NewExcessiveDataReadsTrainer(config)
		if err != nil {
			t.Fatal(err)
		}
		start := config.GetTrainingStartTime().AsTime()
		trainer.Count(common.ExtractedLogFeatures{TimestampMs: start.UnixMilli(), RowsReturned: 10})
		trainer.Count(common.ExtractedLogFeatures{TimestampMs: start.Add(offset).UnixMilli(), RowsReturned: 20})
		if !errors.Is(trainer.Err(), common.ErrInvalidArgument) || trainer.BuildModel() != nil {
			t.Fatalf("invalid stream: error = %v; must not build a partial model", trainer.Err())
		}
	}
}
