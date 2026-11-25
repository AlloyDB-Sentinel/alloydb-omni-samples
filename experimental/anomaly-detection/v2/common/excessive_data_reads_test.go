package common

import (
	"testing"
	"time"
)

func TestExcessiveDataReadsHourOfWeek(t *testing.T) {
	for _, test := range []struct {
		timestamp string
		want      int32
	}{
		{"2026-01-05T00:00:00Z", 0},
		{"2026-01-11T23:59:59Z", 167},
		{"2026-01-12T00:00:00Z", 0},
		{"2026-01-05T19:00:00-05:00", 24},
	} {
		timestamp, err := time.Parse(time.RFC3339, test.timestamp)
		if err != nil {
			t.Fatal(err)
		}
		if got := ExcessiveDataReadsHourOfWeek(timestamp.UnixMilli()); got != test.want {
			t.Errorf("ExcessiveDataReadsHourOfWeek(%s) = %d, want %d", test.timestamp, got, test.want)
		}
	}
}
