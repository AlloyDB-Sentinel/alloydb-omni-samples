package main

import (
	"bytes"
	"context"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"testing"

	genpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v1/generator/proto"
)

func TestGenerateLogsRejectsInvalidPGAuditRowsBeforeWriting(t *testing.T) {
	backupFlags(t)
	*startTimeMs = 1719878400000
	*trainEndTimeMs = 1719882000000
	*endTimeMs = 1719885600000
	*bucketSizeMinutes = 60

	for _, mode := range []struct {
		name       string
		flag       bool
		configFlag bool
	}{
		{"cli", true, false},
		{"config", false, true},
		{"both", true, true},
	} {
		t.Run(mode.name, func(t *testing.T) {
			*pgauditScenario = mode.flag
			configs := parseProto(t, `bucket_sampling {
  query_template_configs {
    template { normalized_query_template: "SELECT 1" is_read_query: true }
    bucket_config { queries_per_bucket_mean: 2 rows_returned_mean: 25 rows_returned_max: 100 }
  }
  query_template_configs {
    template { normalized_query_template: "SELECT 2" is_read_query: true }
    bucket_config { queries_per_bucket_mean: 2 rows_returned_mean: 25 rows_returned_max: -1 }
  }
}`, &genpb.QueryTemplateConfigs{})
			configs.GetBucketSampling().GetQueryTemplateConfigs()[1].GetBucketConfig().SetPgauditScenario(mode.configFlag)
			dir := t.TempDir()
			path := filepath.Join(dir, "logs.binproto")
			original := []byte("previous experiment")
			if err := os.WriteFile(path, original, 0600); err != nil {
				t.Fatal(err)
			}

			err := GenerateLogs(context.Background(), configs, 0, 42, dir)
			if err == nil || !strings.Contains(err.Error(), "config[1]: rows_returned_max") {
				t.Fatalf("GenerateLogs() error = %v, want invalid row limit for second template", err)
			}
			got, err := os.ReadFile(path)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.Equal(got, original) {
				t.Fatal("invalid settings overwrote the previous experiment")
			}

			var output bytes.Buffer
			if err := GenerateLogsBucketByBucket(configs, rand.New(rand.NewSource(42)), &output); err == nil {
				t.Fatal("stream generation accepted invalid row settings")
			}
			if output.Len() != 0 {
				t.Fatal("stream generation wrote events before rejecting invalid settings")
			}
		})
	}
}

func TestGeneratePGAuditRequiresReadDeclaration(t *testing.T) {
	backupFlags(t)
	*startTimeMs = 1719878400000
	*trainEndTimeMs = 1719882000000
	*endTimeMs = 1719885600000
	*bucketSizeMinutes = 60

	for _, useFlag := range []bool{false, true} {
		*pgauditScenario = useFlag
		configs := parseProto(t, `bucket_sampling { query_template_configs {
  template { normalized_query_template: "SELECT 1" }
  bucket_config { queries_per_bucket_mean: 2 rows_returned_mean: 25 rows_returned_max: 100 }
} }`, &genpb.QueryTemplateConfigs{})
		configs.GetBucketSampling().GetQueryTemplateConfigs()[0].GetBucketConfig().SetPgauditScenario(!useFlag)
		dir := t.TempDir()
		err := GenerateLogs(context.Background(), configs, 0, 42, dir)
		if err == nil || !strings.Contains(err.Error(), "template.is_read_query must be set") {
			t.Fatalf("CLI flag %v: error = %v, want missing read declaration", useFlag, err)
		}
		if _, err := os.Stat(filepath.Join(dir, "logs.binproto")); !os.IsNotExist(err) {
			t.Fatalf("output should not exist after validation failure: %v", err)
		}
	}
}

func TestGeneratePGAuditReadStatements(t *testing.T) {
	backupFlags(t)
	*pgauditScenario = true
	*startTimeMs = 1719878400000
	*trainEndTimeMs = 1719882000000
	*endTimeMs = 1719882000000
	*bucketSizeMinutes = 60

	for _, test := range []struct {
		statement string
		isRead    bool
		wantCount int
	}{
		{"/* dashboard */ SELECT * FROM accounts", true, 2},
		{"WITH ids AS (SELECT 1) SELECT * FROM ids", true, 2},
		{"WITH ids AS (SELECT 1) DELETE FROM accounts", false, 0},
		{"WITH removed AS (DELETE FROM accounts RETURNING *) SELECT * FROM removed", false, 0},
	} {
		t.Run(test.statement, func(t *testing.T) {
			configs := parseProto(t, `bucket_sampling { query_template_configs {
  template { normalized_query_template: "SELECT 1" is_read_query: true }
  bucket_config { queries_per_bucket_mean: 2 rows_returned_mean: 25 rows_returned_max: 100 }
} }`, &genpb.QueryTemplateConfigs{})
			template := configs.GetBucketSampling().GetQueryTemplateConfigs()[0].GetTemplate()
			template.SetNormalizedQueryTemplate(test.statement)
			template.SetIsReadQuery(test.isRead)
			var output bytes.Buffer
			if err := GenerateLogsBucketByBucket(configs, rand.New(rand.NewSource(42)), &output); err != nil {
				t.Fatal(err)
			}
			entries := decodeLogs(t, &output)
			if len(entries) != test.wantCount {
				t.Fatalf("generated %d entries, want %d", len(entries), test.wantCount)
			}
			for _, entry := range entries {
				if entry.GetRowsReturned() != 25 || entry.GetQueryStatement() != test.statement || entry.GetIsGroundTruthAnomalous() {
					t.Errorf("unexpected training read: %v", entry)
				}
			}
		})
	}
}
