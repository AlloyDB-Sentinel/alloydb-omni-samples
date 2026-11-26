package main

import (
	"bufio"
	"bytes"
	"errors"
	"io"
	"math"
	"math/rand"
	"net/netip"
	"testing"
	"time"

	"google.golang.org/protobuf/encoding/protodelim"

	genpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v1/generator/proto"
)

const testConnectionConfig = `
source: { project: "test-project" cluster_id: "test-cluster" instance_id: "primary" node_id: "v1" }
client_profiles: {
	profile_name: "app-server"
	client_address_pool: { addresses: "10.0.0.12" }
	db_user: "app"
	db_name: "appdb"
	application_name: "app-api"
	auth_method: "scram-sha-256"
	bucket_traffic: { queries_per_bucket_mean: 10.0 queries_variation_std_dev: 0.0 }
	active_window: ACTIVE_WINDOW_ALL
}
client_profiles: {
	profile_name: "internet-scanner"
	client_address_pool: { cidr_ranges: "203.0.113.0/24" }
	outcome_weights: { outcome: CONNECTION_OUTCOME_DROPPED_BEFORE_AUTH weight: 1.0 }
	periodic: { interval: { seconds: 600 } jitter: { seconds: 60 } }
	active_window: ACTIVE_WINDOW_TEST_ONLY
	attacker_profile: ATTACKER_PROFILE_INTERNET_SCANNER
}
client_profiles: {
	profile_name: "burst-attacker"
	client_address_pool: { addresses: "198.51.100.23" }
	db_user: "postgres"
	db_name: "postgres"
	auth_method: "scram-sha-256"
	outcome_weights: { outcome: CONNECTION_OUTCOME_AUTH_FAILED weight: 1.0 }
	burst: {
		bursts_per_day: 6.0
		connections_per_burst_mean: 50.0
		connections_per_burst_std_dev: 0.0
		burst_duration: { seconds: 300 }
	}
	active_window: ACTIVE_WINDOW_TEST_ONLY
	attacker_profile: ATTACKER_PROFILE_BURST_ATTACKER
}`

var (
	testStartTime    = time.Date(2026, time.October, 5, 0, 0, 0, 0, time.UTC)
	testTrainEndTime = testStartTime.Add(48 * time.Hour)
	testEndTime      = testStartTime.Add(96 * time.Hour)
)

// generateTestConnectionLogs runs GenerateConnectionLogs over 2 training days and 2 test days.
func generateTestConnectionLogs(t *testing.T, configText string, seed int64) []byte {
	t.Helper()
	connectionCfg := parseProto(t, configText, &genpb.ConnectionSamplingConfig{})
	var output bytes.Buffer
	rng := rand.New(rand.NewSource(seed))
	if err := GenerateConnectionLogs(connectionCfg, testStartTime, testTrainEndTime, testEndTime, time.Hour, rng, &output); err != nil {
		t.Fatalf("GenerateConnectionLogs() error = %v", err)
	}
	return output.Bytes()
}

// readConnectionLogs decodes length-delimited ConnectionLogEntry records.
func readConnectionLogs(t *testing.T, data []byte) []*genpb.ConnectionLogEntry {
	t.Helper()
	reader := bufio.NewReader(bytes.NewReader(data))
	var entries []*genpb.ConnectionLogEntry
	for {
		entry := &genpb.ConnectionLogEntry{}
		err := protodelim.UnmarshalFrom(reader, entry)
		if errors.Is(err, io.EOF) {
			return entries
		}
		if err != nil {
			t.Fatalf("failed to read connection log %d: %v", len(entries), err)
		}
		entries = append(entries, entry)
	}
}

func TestGenerateConnectionLogs(t *testing.T) {
	entries := readConnectionLogs(t, generateTestConnectionLogs(t, testConnectionConfig, 42))

	scannerRange := netip.MustParsePrefix("203.0.113.0/24")
	backendPids := make(map[int32]bool)
	entriesByProfile := make(map[genpb.AttackerProfile][]*genpb.ConnectionLogEntry)
	var previousTimestampMs int64
	for i, entry := range entries {
		if entry.GetTimestampMs() < previousTimestampMs {
			t.Fatalf("entry %d: timestamp %d is before previous %d, want chronological order", i, entry.GetTimestampMs(), previousTimestampMs)
		}
		previousTimestampMs = entry.GetTimestampMs()
		if entry.GetTimestampMs() < testStartTime.UnixMilli() || entry.GetTimestampMs() >= testEndTime.UnixMilli() {
			t.Errorf("entry %d: timestamp %d outside [start, end)", i, entry.GetTimestampMs())
		}
		if backendPids[entry.GetBackendPid()] {
			t.Errorf("entry %d: backend_pid %d reused", i, entry.GetBackendPid())
		}
		backendPids[entry.GetBackendPid()] = true
		if entry.GetSource().GetClusterId() != "test-cluster" {
			t.Errorf("entry %d: source cluster_id = %q, want %q", i, entry.GetSource().GetClusterId(), "test-cluster")
		}
		if entry.GetClientPort() < minClientPort || entry.GetClientPort() > maxClientPort {
			t.Errorf("entry %d: client_port %d outside ephemeral range", i, entry.GetClientPort())
		}
		entriesByProfile[entry.GetGroundTruthProfile()] = append(entriesByProfile[entry.GetGroundTruthProfile()], entry)
	}

	t.Run("normal_client", func(t *testing.T) {
		normalEntries := entriesByProfile[genpb.AttackerProfile_ATTACKER_PROFILE_UNSPECIFIED]
		// 96 hourly buckets at a constant 10 connections per bucket.
		if got, want := len(normalEntries), 960; got != want {
			t.Errorf("normal connections = %d, want %d", got, want)
		}
		for _, entry := range normalEntries {
			if entry.GetIsGroundTruthAnomalous() {
				t.Fatalf("normal connection labeled anomalous: %v", entry)
			}
			if entry.GetClientAddress() != "10.0.0.12" || entry.GetDbUser() != "app" || entry.GetApplicationName() != "app-api" {
				t.Fatalf("normal connection has unexpected fields: %v", entry)
			}
			if entry.GetOutcome() != genpb.ConnectionOutcome_CONNECTION_OUTCOME_AUTHORIZED {
				t.Fatalf("normal connection outcome = %v, want AUTHORIZED", entry.GetOutcome())
			}
		}
	})

	t.Run("internet_scanner", func(t *testing.T) {
		scannerEntries := entriesByProfile[genpb.AttackerProfile_ATTACKER_PROFILE_INTERNET_SCANNER]
		// 48 test hours at one connection every 10 minutes.
		if got := len(scannerEntries); got < 287 || got > 289 {
			t.Errorf("scanner connections = %d, want about 288", got)
		}
		for i, entry := range scannerEntries {
			if entry.GetTimestampMs() < testTrainEndTime.UnixMilli() {
				t.Fatalf("scanner connection at %d, before training end", entry.GetTimestampMs())
			}
			if !entry.GetIsGroundTruthAnomalous() {
				t.Fatalf("scanner connection not labeled anomalous: %v", entry)
			}
			if !scannerRange.Contains(netip.MustParseAddr(entry.GetClientAddress())) {
				t.Fatalf("scanner address %s outside %s", entry.GetClientAddress(), scannerRange)
			}
			if entry.GetDbUser() != "" || entry.GetDbName() != "" || entry.GetAuthMethod() != "" {
				t.Fatalf("dropped connection has session fields: %v", entry)
			}
			if i > 0 {
				gap := time.Duration(entry.GetTimestampMs()-scannerEntries[i-1].GetTimestampMs()) * time.Millisecond
				if gap < 9*time.Minute || gap > 11*time.Minute {
					t.Errorf("scanner gap = %v, want 10m +/- 1m", gap)
				}
			}
		}
	})

	t.Run("burst_attacker", func(t *testing.T) {
		burstEntries := entriesByProfile[genpb.AttackerProfile_ATTACKER_PROFILE_BURST_ATTACKER]
		// Bursts are random, but each has exactly 50 connections and about 12 are expected.
		if len(burstEntries) == 0 || len(burstEntries)%50 != 0 {
			t.Errorf("burst connections = %d, want a positive multiple of 50", len(burstEntries))
		}
		for _, entry := range burstEntries {
			if entry.GetTimestampMs() < testTrainEndTime.UnixMilli() {
				t.Fatalf("burst connection at %d, before training end", entry.GetTimestampMs())
			}
			if entry.GetOutcome() != genpb.ConnectionOutcome_CONNECTION_OUTCOME_AUTH_FAILED {
				t.Fatalf("burst outcome = %v, want AUTH_FAILED", entry.GetOutcome())
			}
			if entry.GetDbUser() != "postgres" || entry.GetApplicationName() != "" {
				t.Fatalf("failed connection has unexpected fields: %v", entry)
			}
		}
	})
}

func TestGenerateConnectionLogsIsDeterministicForSeed(t *testing.T) {
	firstOutput := generateTestConnectionLogs(t, testConnectionConfig, 7)
	secondOutput := generateTestConnectionLogs(t, testConnectionConfig, 7)
	if !bytes.Equal(firstOutput, secondOutput) {
		t.Errorf("GenerateConnectionLogs() produced different output for the same seed")
	}
}

func TestGenerateConnectionLogsTrainingOnlyProfile(t *testing.T) {
	entries := readConnectionLogs(t, generateTestConnectionLogs(t, `
client_profiles: {
	client_address_pool: { addresses: "10.0.0.99" }
	db_user: "retired"
	periodic: { interval: { seconds: 3600 } }
	active_window: ACTIVE_WINDOW_TRAINING_ONLY
}`, 1))
	if len(entries) == 0 {
		t.Fatalf("GenerateConnectionLogs() produced no connections, want training connections")
	}
	for _, entry := range entries {
		if entry.GetTimestampMs() >= testTrainEndTime.UnixMilli() {
			t.Fatalf("training-only connection at %d, after training end", entry.GetTimestampMs())
		}
	}
}

func TestIsProfileActive(t *testing.T) {
	tests := []struct {
		activeWindow    genpb.ActiveWindow
		isTrainingPhase bool
		want            bool
	}{
		{genpb.ActiveWindow_ACTIVE_WINDOW_UNSPECIFIED, true, true},
		{genpb.ActiveWindow_ACTIVE_WINDOW_UNSPECIFIED, false, true},
		{genpb.ActiveWindow_ACTIVE_WINDOW_ALL, true, true},
		{genpb.ActiveWindow_ACTIVE_WINDOW_ALL, false, true},
		{genpb.ActiveWindow_ACTIVE_WINDOW_TEST_ONLY, true, false},
		{genpb.ActiveWindow_ACTIVE_WINDOW_TEST_ONLY, false, true},
		{genpb.ActiveWindow_ACTIVE_WINDOW_TRAINING_ONLY, true, true},
		{genpb.ActiveWindow_ACTIVE_WINDOW_TRAINING_ONLY, false, false},
	}
	for _, test := range tests {
		if got := isProfileActive(test.activeWindow, test.isTrainingPhase); got != test.want {
			t.Errorf("isProfileActive(%v, isTrainingPhase=%v) = %v, want %v", test.activeWindow, test.isTrainingPhase, got, test.want)
		}
	}
}

func TestSamplePoissonCount(t *testing.T) {
	for _, mean := range []float64{0.25, 3.0, 100.0} {
		rng := rand.New(rand.NewSource(42))
		const draws = 20000
		total := 0
		for i := 0; i < draws; i++ {
			count := samplePoissonCount(mean, rng)
			if count < 0 {
				t.Fatalf("samplePoissonCount(%v) = %d, want non-negative", mean, count)
			}
			total += count
		}
		if average := float64(total) / draws; math.Abs(average-mean) > 0.05*mean+0.02 {
			t.Errorf("samplePoissonCount(%v) average = %v, want about %v", mean, average, mean)
		}
	}
	if got := samplePoissonCount(0, rand.New(rand.NewSource(1))); got != 0 {
		t.Errorf("samplePoissonCount(0) = %d, want 0", got)
	}
}
