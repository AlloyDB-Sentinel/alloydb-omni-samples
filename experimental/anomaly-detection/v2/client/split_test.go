package main

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common"
	"google.golang.org/protobuf/encoding/protodelim"

	genpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v1/generator/proto"
	auditpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common/proto"
)

func TestSplitLogsPreservesPrefixAndExports(t *testing.T) {
	dir := t.TempDir()
	f, err := os.Create(filepath.Join(dir, "logs.binproto"))
	if err != nil {
		t.Fatal(err)
	}
	// Write v1 records, then verify the v2 pipeline preserves all fields.
	var want []*auditpb.AuditLogEntry
	for i, action := range []string{"A", "B", "A", "B", "HELD_OUT", "HELD_OUT", "HELD_OUT", "HELD_OUT", "HELD_OUT", "HELD_OUT"} {
		e := &genpb.AuditLogEntry{}
		e.SetNormalizedQueryTemplate(action)
		e.SetQueryStatement(action + " statement")
		e.SetDbUser("user")
		e.SetDbName("db")
		e.SetTimestampMs(int64(i))
		e.SetIsGroundTruthAnomalous(i%2 == 0)
		if _, err := protodelim.MarshalTo(f, e); err != nil {
			t.Fatal(err)
		}
		v := &auditpb.AuditLogEntry{}
		v.SetNormalizedQueryTemplate(action)
		want = append(want, v)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()
	if err := splitLogs(ctx, dir, 4, 10); err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		path string
		want []*auditpb.AuditLogEntry
	}{{"train.binproto", want[:4]}, {"test.binproto", want[4:]}} {
		var got []string
		err := common.IngestBinProtoStream(ctx, filepath.Join(dir, test.path), func(e *auditpb.AuditLogEntry) error {
			got = append(got, e.GetNormalizedQueryTemplate())
			if e.GetQueryStatement() != e.GetNormalizedQueryTemplate()+" statement" || e.GetDbUser() != "user" || e.GetDbName() != "db" || e.GetIsGroundTruthAnomalous() != (e.GetTimestampMs()%2 == 0) {
				t.Fatal("record fields changed")
			}
			return nil
		})
		if err != nil {
			t.Fatal(err)
		}
		var expected []string
		for _, e := range test.want {
			expected = append(expected, e.GetNormalizedQueryTemplate())
		}
		if !reflect.DeepEqual(got, expected) {
			t.Fatalf("%s: got %v want %v", test.path, got, expected)
		}
	}
	data, err := os.ReadFile(filepath.Join(dir, "logs.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) != 10 {
		t.Fatalf("JSONL rows=%d", len(lines))
	}
	for _, line := range lines {
		if !json.Valid([]byte(line)) {
			t.Fatal("invalid JSONL")
		}
	}
	if err := splitLogs(ctx, dir, 4, 11); err == nil {
		t.Fatal("expected count mismatch error")
	}
}
