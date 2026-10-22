package common

import (
	"context"
	"errors"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	"google.golang.org/protobuf/encoding/protodelim"

	genpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v1/generator/proto"
	auditpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common/proto"
)

func TestNewSlidingRowWindow(t *testing.T) {
	for _, duration := range []time.Duration{0, -time.Minute, time.Microsecond} {
		if _, err := NewSlidingRowWindow(duration); !errors.Is(err, ErrInvalidArgument) {
			t.Errorf("NewSlidingRowWindow(%v) error = %v, want ErrInvalidArgument", duration, err)
		}
	}
}

func TestSlidingRowWindowAdd(t *testing.T) {
	window, err := NewSlidingRowWindow(5 * time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name      string
		feature   ExtractedLogFeatures
		wantTotal int64
	}{
		{"first_read", ExtractedLogFeatures{DbName: "db", DbUser: "alice", TimestampMs: 0, RowsReturned: 100}, 100},
		{"another_template", ExtractedLogFeatures{DbName: "db", DbUser: "alice", QueryTemplate: "SELECT * FROM orders", TimestampMs: 60000, RowsReturned: 200}, 300},
		{"another_user", ExtractedLogFeatures{DbName: "db", DbUser: "bob", TimestampMs: 0, RowsReturned: 50}, 50},
		{"another_database", ExtractedLogFeatures{DbName: "other", DbUser: "alice", TimestampMs: 0, RowsReturned: 25}, 25},
		{"just_inside_boundary", ExtractedLogFeatures{DbName: "db", DbUser: "alice", TimestampMs: 299999, RowsReturned: 10}, 310},
		{"at_boundary", ExtractedLogFeatures{DbName: "db", DbUser: "alice", TimestampMs: 300000, RowsReturned: 300}, 510},
		{"equal_timestamp", ExtractedLogFeatures{DbName: "db", DbUser: "alice", TimestampMs: 300000, RowsReturned: 40}, 550},
		{"zero_rows_expires_old_read", ExtractedLogFeatures{DbName: "db", DbUser: "alice", TimestampMs: 360000, RowsReturned: 0}, 350},
		{"long_gap", ExtractedLogFeatures{DbName: "db", DbUser: "alice", TimestampMs: 1200000, RowsReturned: 70}, 70},
	}
	for _, test := range tests {
		got, err := window.Add(test.feature)
		if err != nil {
			t.Fatalf("%s: Add() error = %v", test.name, err)
		}
		if got != test.wantTotal {
			t.Errorf("%s: Add() = %d, want %d", test.name, got, test.wantTotal)
		}
	}
}

func TestSlidingRowWindowRejectsInvalidRead(t *testing.T) {
	tests := []struct {
		name string
		read ExtractedLogFeatures
	}{
		{"negative_timestamp", ExtractedLogFeatures{TimestampMs: -1, RowsReturned: 1}},
		{"negative_rows", ExtractedLogFeatures{TimestampMs: 400000, RowsReturned: -1}},
		{"out_of_order", ExtractedLogFeatures{TimestampMs: 59999, RowsReturned: 1}},
		{"overflow", ExtractedLogFeatures{TimestampMs: 300000, RowsReturned: math.MaxInt64}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			window, err := NewSlidingRowWindow(5 * time.Minute)
			if err != nil {
				t.Fatal(err)
			}
			for _, timestamp := range []int64{0, 60000} {
				if _, err := window.Add(ExtractedLogFeatures{TimestampMs: timestamp, RowsReturned: 100}); err != nil {
					t.Fatal(err)
				}
			}
			if _, err := window.Add(test.read); !errors.Is(err, ErrInvalidArgument) {
				t.Fatalf("Add() error = %v, want ErrInvalidArgument", err)
			}
			got, err := window.Add(ExtractedLogFeatures{TimestampMs: 60000, RowsReturned: 1})
			if err != nil || got != 201 {
				t.Errorf("Add() after rejected read = (%d, %v), want (201, nil)", got, err)
			}
		})
	}
}

func TestSlidingRowWindowIngestGeneratorLogs(t *testing.T) {
	path := filepath.Join(t.TempDir(), "logs.binproto")
	file, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { file.Close() })
	// Use the generator's message type to catch wire-format differences in v2 ingestion.
	entries := []genpb.AuditLogEntry_builder{
		{DbName: "db", DbUser: "alice", TimestampMs: 0, RowsReturned: 100, DatabaseSessionId: "session-1", NormalizedQueryTemplate: "SELECT * FROM accounts"},
		{DbName: "db", DbUser: "alice", TimestampMs: 60000, RowsReturned: 200, DatabaseSessionId: "session-2", NormalizedQueryTemplate: "SELECT * FROM orders"},
		{DbName: "db", DbUser: "alice", TimestampMs: 300000, RowsReturned: 300, DatabaseSessionId: "session-2", NormalizedQueryTemplate: "SELECT * FROM orders"},
	}
	for _, entry := range entries {
		if _, err := protodelim.MarshalTo(file, entry.Build()); err != nil {
			t.Fatal(err)
		}
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	window, err := NewSlidingRowWindow(5 * time.Minute)
	if err != nil {
		t.Fatal(err)
	}
	var totals []int64
	err = IngestBinProtoStream(context.Background(), path, func(entry *auditpb.AuditLogEntry) error {
		feature, err := Extract(entry)
		if err != nil {
			return err
		}
		total, err := window.Add(feature)
		totals = append(totals, total)
		return err
	})
	if err != nil {
		t.Fatal(err)
	}
	if want := []int64{100, 300, 500}; !reflect.DeepEqual(totals, want) {
		t.Errorf("window totals = %v, want %v", totals, want)
	}
}
