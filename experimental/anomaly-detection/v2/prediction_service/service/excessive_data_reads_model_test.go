package service

import (
	"errors"
	"math"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common"
	training "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/training_service/service"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/timestamppb"

	modelpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common/proto"
)

func excessiveDataReadsModelProto() *modelpb.Model {
	var entries []*modelpb.ExcessiveDataReadsBaselineEntry
	for slot := int32(0); slot < 168; slot++ {
		short, long := &modelpb.VolumetricBaseline{}, &modelpb.VolumetricBaseline{}
		short.SetMean(100)
		short.SetUpperThreshold(100)
		long.SetMean(200)
		long.SetUpperThreshold(200)
		entry := &modelpb.ExcessiveDataReadsBaselineEntry{}
		entry.SetDbName("db")
		entry.SetDbUser("alice")
		entry.SetHourOfWeek(slot)
		entry.SetShortWindow(short)
		entry.SetLongWindow(long)
		entries = append(entries, entry)
	}
	data := &modelpb.ExcessiveDataReadsModel{}
	data.SetEntries(entries)
	scenario := &modelpb.ExcessiveDataReadsScenario{}
	scenario.SetModelData(data)
	model := &modelpb.Model{}
	model.SetExcessiveDataReads(scenario)
	return model
}

func excessiveDataReadsRead(offset time.Duration, rows int64) *modelpb.AuditLogEntry {
	entry := &modelpb.AuditLogEntry{}
	entry.SetDbName("db")
	entry.SetDbUser("alice")
	entry.SetTimestampMs(time.Date(2026, 1, 5, 0, 0, 0, 0, time.UTC).Add(offset).UnixMilli())
	entry.SetRowsReturned(rows)
	return entry
}

func TestExcessiveDataReadsModelPredict(t *testing.T) {
	model, err := NewExcessiveDataReadsModel(excessiveDataReadsModelProto())
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		offset time.Duration
		rows   int64
		want   bool
	}{
		{0, 100, false},        // Equality is normal.
		{time.Minute, 1, true}, // Short window alone triggers.
		{15 * time.Minute, 99, false},
		{30 * time.Minute, 1, true}, // Long window alone triggers.
		{4 * time.Hour, 1, false},   // The first read expires at the boundary.
	} {
		finding := model.Predict(excessiveDataReadsRead(test.offset, test.rows))
		if model.Err() != nil || finding.GetIsAnomalousPredict() != test.want {
			t.Fatalf("Predict(%v, %d) = %v, error %v; want anomalous %v", test.offset, test.rows, finding, model.Err(), test.want)
		}
		if finding.GetTimestampMs() != excessiveDataReadsRead(test.offset, test.rows).GetTimestampMs() {
			t.Error("finding lost the read timestamp needed for reporting")
		}
	}
	model.ResetLiveCounts()
	if model.Predict(excessiveDataReadsRead(0, 100)).GetIsAnomalousPredict() {
		t.Error("reset retained earlier reads")
	}
	for _, change := range []func(*modelpb.AuditLogEntry){
		func(e *modelpb.AuditLogEntry) { e.SetDbName("other") },
		func(e *modelpb.AuditLogEntry) { e.SetDbUser("bob") },
	} {
		entry := excessiveDataReadsRead(0, 1)
		change(entry)
		if !model.Predict(entry).GetIsAnomalousPredict() {
			t.Error("an unseen identity should be anomalous")
		}
	}
	if model.Predict(excessiveDataReadsRead(15*time.Minute, 100)).GetIsAnomalousPredict() {
		t.Error("other identities contaminated the known user's window")
	}
}

func TestExcessiveDataReadsModelInvalidInput(t *testing.T) {
	model, err := NewExcessiveDataReadsModel(excessiveDataReadsModelProto())
	if err != nil {
		t.Fatal(err)
	}
	model.Predict(excessiveDataReadsRead(time.Minute, 1))
	if model.Predict(excessiveDataReadsRead(0, 1)) != nil || !errors.Is(model.Err(), common.ErrInvalidArgument) {
		t.Fatal("out-of-order input must fail without returning a normal finding")
	}
	model.ResetLiveCounts()
	if model.Predict(excessiveDataReadsRead(0, 1)) == nil || model.Err() != nil {
		t.Fatal("reset did not clear the stream error")
	}
}

func TestExcessiveDataReadsModelUsesHourlyBaseline(t *testing.T) {
	pb := excessiveDataReadsModelProto()
	// Monday 01:00 is quiet, unlike the preceding hour.
	baseline := pb.GetExcessiveDataReads().GetModelData().GetEntries()[1].GetShortWindow()
	baseline.SetMean(0)
	baseline.SetUpperThreshold(0)
	model, err := NewExcessiveDataReadsModel(pb)
	if err != nil {
		t.Fatal(err)
	}
	if model.Predict(excessiveDataReadsRead(0, 1)).GetIsAnomalousPredict() || !model.Predict(excessiveDataReadsRead(time.Hour, 1)).GetIsAnomalousPredict() {
		t.Error("prediction did not switch weekday/hour baselines")
	}
}

func TestNewExcessiveDataReadsModelInvalidBaselines(t *testing.T) {
	for _, test := range []struct {
		name string
		edit func(*modelpb.ExcessiveDataReadsModel)
	}{
		{"missing_hour", func(m *modelpb.ExcessiveDataReadsModel) { m.SetEntries(m.GetEntries()[1:]) }},
		{"duplicate_hour", func(m *modelpb.ExcessiveDataReadsModel) { m.GetEntries()[1].SetHourOfWeek(0) }},
		{"invalid_hour", func(m *modelpb.ExcessiveDataReadsModel) { m.GetEntries()[0].SetHourOfWeek(168) }},
		{"missing_window", func(m *modelpb.ExcessiveDataReadsModel) { m.GetEntries()[0].SetShortWindow(nil) }},
		{"nan_threshold", func(m *modelpb.ExcessiveDataReadsModel) {
			m.GetEntries()[0].GetShortWindow().SetUpperThreshold(math.NaN())
		}},
		{"inconsistent_threshold", func(m *modelpb.ExcessiveDataReadsModel) { m.GetEntries()[0].GetShortWindow().SetUpperThreshold(99) }},
	} {
		t.Run(test.name, func(t *testing.T) {
			pb := excessiveDataReadsModelProto()
			test.edit(pb.GetExcessiveDataReads().GetModelData())
			if _, err := NewExcessiveDataReadsModel(pb); err == nil {
				t.Fatal("accepted invalid model")
			}
		})
	}
	if _, err := NewExcessiveDataReadsModel(nil); err == nil {
		t.Fatal("accepted nil model")
	}
}

func TestExcessiveDataReadsTrainSerializePredict(t *testing.T) {
	start := time.Date(2026, 1, 5, 0, 0, 0, 0, time.UTC)
	config := &modelpb.ExcessiveDataReadsTrainingConfig{}
	config.SetShortWindowStdDevMultiplier(6)
	config.SetLongWindowStdDevMultiplier(4)
	config.SetTrainingStartTime(timestamppb.New(start))
	config.SetTrainingEndTime(timestamppb.New(start.Add(14 * 24 * time.Hour)))
	trainer, err := training.NewExcessiveDataReadsTrainer(config)
	if err != nil {
		t.Fatal(err)
	}
	for week, rows := range []int64{10, 30} {
		trainer.Count(common.ExtractedLogFeatures{
			DbName:       "db",
			DbUser:       "alice",
			TimestampMs:  start.Add(time.Duration(week) * 7 * 24 * time.Hour).UnixMilli(),
			RowsReturned: rows,
		})
	}
	data, err := proto.Marshal(trainer.BuildModel())
	if err != nil {
		t.Fatal(err)
	}
	pb := &modelpb.Model{}
	if err := proto.Unmarshal(data, pb); err != nil {
		t.Fatal(err)
	}
	model, err := NewExcessiveDataReadsModel(pb)
	if err != nil {
		t.Fatal(err)
	}
	// Short threshold = 80; long threshold = 60. The long window fires alone.
	finding := model.Predict(excessiveDataReadsRead(14*24*time.Hour, 61))
	if !finding.GetIsAnomalousPredict() || math.Abs(finding.GetConfidenceScore()-1.0/41) > 1e-12 {
		t.Fatalf("reloaded model returned %v", finding)
	}
}
