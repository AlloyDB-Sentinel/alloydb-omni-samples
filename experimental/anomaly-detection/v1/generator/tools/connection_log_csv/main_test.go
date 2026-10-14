package main

import (
	"bufio"
	"bytes"
	"encoding/csv"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/protobuf/encoding/protodelim"
	"google.golang.org/protobuf/encoding/prototext"

	genpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v1/generator/proto"
)

func mustParseConnectionLog(t *testing.T, text string) *genpb.ConnectionLogEntry {
	t.Helper()
	entry := &genpb.ConnectionLogEntry{}
	if err := prototext.Unmarshal([]byte(text), entry); err != nil {
		t.Fatalf("Failed to parse textproto: %v", err)
	}
	return entry
}

func encodeConnectionLogs(t *testing.T, entries ...*genpb.ConnectionLogEntry) []byte {
	t.Helper()
	var encoded bytes.Buffer
	for _, entry := range entries {
		if _, err := protodelim.MarshalTo(&encoded, entry); err != nil {
			t.Fatalf("Failed to encode connection log: %v", err)
		}
	}
	return encoded.Bytes()
}

func TestExportConnectionLogsToCSV(t *testing.T) {
	timestampMs := time.Date(2026, time.September, 17, 0, 4, 32, 716_000_000, time.UTC).UnixMilli()
	authorizedEntry := mustParseConnectionLog(t, `
client_address: "10.0.0.12"
client_port: 44031
db_user: "billing_app"
db_name: "billing"
application_name: "billing-api"
auth_method: "scram-sha-256"
outcome: CONNECTION_OUTCOME_AUTHORIZED
source: { project: "test-project" cluster_id: "test-cluster" instance_id: "primary" node_id: "v1" }
backend_pid: 828701`)
	authorizedEntry.SetTimestampMs(timestampMs)
	droppedEntry := mustParseConnectionLog(t, `
client_address: "203.0.113.217"
client_port: 58203
outcome: CONNECTION_OUTCOME_DROPPED_BEFORE_AUTH
backend_pid: 854855
is_ground_truth_anomalous: true
ground_truth_profile: ATTACKER_PROFILE_INTERNET_SCANNER`)
	droppedEntry.SetTimestampMs(timestampMs + 1)

	var csvOutput bytes.Buffer
	rowCount, err := exportConnectionLogsToCSV(bufio.NewReader(bytes.NewReader(encodeConnectionLogs(t, authorizedEntry, droppedEntry))), &csvOutput)
	if err != nil {
		t.Fatalf("exportConnectionLogsToCSV() error = %v", err)
	}
	if rowCount != 2 {
		t.Errorf("exportConnectionLogsToCSV() rowCount = %d, want 2", rowCount)
	}

	gotRows, err := csv.NewReader(&csvOutput).ReadAll()
	if err != nil {
		t.Fatalf("Failed to parse CSV output: %v", err)
	}
	wantRows := [][]string{
		csvHeader,
		{"2026-09-17 00:04:32.716", "10.0.0.12", "44031", "CONNECTION_OUTCOME_AUTHORIZED", "billing_app", "billing", "billing-api", "scram-sha-256", "828701", "false", "ATTACKER_PROFILE_UNSPECIFIED", "test-project", "test-cluster", "primary", "v1"},
		{"2026-09-17 00:04:32.717", "203.0.113.217", "58203", "CONNECTION_OUTCOME_DROPPED_BEFORE_AUTH", "", "", "", "", "854855", "true", "ATTACKER_PROFILE_INTERNET_SCANNER", "", "", "", ""},
	}
	if diff := cmp.Diff(wantRows, gotRows); diff != "" {
		t.Errorf("CSV output mismatch (-want +got):\n%s", diff)
	}
}

func TestExportConnectionLogsToCSVEmptyInput(t *testing.T) {
	var csvOutput bytes.Buffer
	rowCount, err := exportConnectionLogsToCSV(bufio.NewReader(bytes.NewReader(nil)), &csvOutput)
	if err != nil {
		t.Fatalf("exportConnectionLogsToCSV() error = %v", err)
	}
	if rowCount != 0 {
		t.Errorf("exportConnectionLogsToCSV() rowCount = %d, want 0", rowCount)
	}
	if got, want := strings.TrimSpace(csvOutput.String()), strings.Join(csvHeader, ","); got != want {
		t.Errorf("CSV output = %q, want header only %q", got, want)
	}
}

func TestExportConnectionLogsToCSVCorruptInput(t *testing.T) {
	// A length prefix of 10 bytes followed by only 3 bytes of data.
	corruptInput := []byte{10, 1, 2, 3}
	var csvOutput bytes.Buffer
	if _, err := exportConnectionLogsToCSV(bufio.NewReader(bytes.NewReader(corruptInput)), &csvOutput); err == nil {
		t.Errorf("exportConnectionLogsToCSV() error = nil, want error for truncated input")
	}
}
