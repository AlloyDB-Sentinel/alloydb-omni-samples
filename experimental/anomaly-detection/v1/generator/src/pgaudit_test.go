package main

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	genpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v1/generator/proto"
)

func TestGeneratePGAuditReadLogs(t *testing.T) {
	backupFlags(t)
	start := time.Date(2026, time.January, 5, 0, 0, 0, 0, time.UTC)
	*startTimeMs = start.UnixMilli()
	*trainEndTimeMs = start.Add(30 * time.Minute).UnixMilli()
	*endTimeMs = start.Add(90 * time.Minute).UnixMilli()
	*bucketSizeMinutes = 60

	for _, test := range []struct {
		name          string
		anomalyConfig string
		wantTestReads int
		wantTestRows  int64
	}{
		{"extra rows", "anomalous_rows_probability: 1 anomalous_rows_multiplier: 4", 4, 100},
		{"query-count spike", "anomalous_spike_probability: 1 anomalous_spike_multiplier: 3", 12, 30},
	} {
		t.Run(test.name, func(t *testing.T) {
			configs := parseProto(t, fmt.Sprintf(`
bucket_sampling {
  query_template_configs {
    template { normalized_query_template: "SELECT 1" is_read_query: true }
    bucket_config {
      pgaudit_scenario: true
      db_user: "reader"
      queries_per_bucket_mean: 4
      rows_returned_mean: 30
      rows_returned_max: 100
      %s
    }
  }
  query_template_configs {
    template { normalized_query_template: "UPDATE t SET x = 1" is_read_query: false }
    bucket_config { pgaudit_scenario: true queries_per_bucket_mean: 4 }
  }
}
`, test.anomalyConfig), &genpb.QueryTemplateConfigs{})
			outputDir := t.TempDir()

			if err := GenerateLogs(context.Background(), configs, 0, 42, outputDir); err != nil {
				t.Fatal(err)
			}
			file, err := os.Open(filepath.Join(outputDir, "logs.binproto"))
			if err != nil {
				t.Fatal(err)
			}
			defer file.Close()
			logs := decodeLogs(t, file)

			const wantTrainingReads = 2
			if len(logs) != wantTrainingReads+test.wantTestReads {
				t.Fatalf("generated %d logs, want %d training reads and %d test reads", len(logs), wantTrainingReads, test.wantTestReads)
			}
			for i, entry := range logs {
				if entry.GetQueryStatement() != "SELECT 1" || entry.GetDbUser() != "reader" || entry.GetDatabaseSessionId() == "" {
					t.Errorf("log %d: expected a read from reader with a session ID, got %v", i, entry)
				}
				if i > 0 && entry.GetTimestampMs() < logs[i-1].GetTimestampMs() {
					t.Errorf("log %d: timestamp precedes the previous log", i)
				}
			}
			for _, entry := range logs[:wantTrainingReads] {
				if entry.GetTimestampMs() < *startTimeMs || entry.GetTimestampMs() >= *trainEndTimeMs ||
					entry.GetIsGroundTruthAnomalous() || entry.GetRowsReturned() != 30 {
					t.Errorf("training log: expected 30 rows and a normal label during training, got %v", entry)
				}
			}
			for _, entry := range logs[wantTrainingReads:] {
				if entry.GetTimestampMs() < *trainEndTimeMs || entry.GetTimestampMs() >= *endTimeMs ||
					!entry.GetIsGroundTruthAnomalous() || entry.GetRowsReturned() != test.wantTestRows {
					t.Errorf("test log: expected %d rows and an anomaly label during testing, got %v", test.wantTestRows, entry)
				}
			}
		})
	}
}

func TestGeneratePGAuditRejectsInvalidConfigBeforeWriting(t *testing.T) {
	for _, test := range []struct {
		name       string
		readField  string
		rowOptions string
		wantError  string
	}{
		{"missing read declaration", "", "", "set is_read_query"},
		{"write templates only", "is_read_query: false", "", "requires a read template"},
		{"missing row limit", "is_read_query: true", "rows_returned_mean: 30", "rows_returned_max"},
		{
			"missing anomaly multiplier",
			"is_read_query: true",
			"rows_returned_mean: 30 rows_returned_max: 100 anomalous_rows_probability: 1",
			"anomalous_rows_multiplier must be greater than 1",
		},
		{
			"neutral anomaly multiplier",
			"is_read_query: true",
			"rows_returned_mean: 30 rows_returned_max: 100 anomalous_rows_probability: 1 anomalous_rows_multiplier: 1",
			"anomalous_rows_multiplier must be greater than 1",
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			configs := parseProto(t, fmt.Sprintf(`
bucket_sampling {
  query_template_configs {
    template { normalized_query_template: "SELECT 1" %s }
    bucket_config { pgaudit_scenario: true queries_per_bucket_mean: 4 %s }
  }
}
`, test.readField, test.rowOptions), &genpb.QueryTemplateConfigs{})
			outputDir := t.TempDir()

			err := GenerateLogs(context.Background(), configs, 0, 42, outputDir)

			if err == nil || !strings.Contains(err.Error(), test.wantError) {
				t.Fatalf("GenerateLogs() error = %v, want %q", err, test.wantError)
			}
			if _, err := os.Stat(filepath.Join(outputDir, "logs.binproto")); !os.IsNotExist(err) {
				t.Fatalf("expected no output file after invalid config, stat error = %v", err)
			}
		})
	}
}
