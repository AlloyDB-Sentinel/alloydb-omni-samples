package common

import "time"

const (
	ExcessiveDataReadsShortWindow = 15 * time.Minute
	ExcessiveDataReadsLongWindow  = 4 * time.Hour
	ExcessiveDataReadsWeeklyHours = 7 * 24
)

// ExcessiveDataReadsKey identifies an hourly baseline across all query templates.
type ExcessiveDataReadsKey struct {
	DbName     string
	DbUser     string
	HourOfWeek int32
}

// ExcessiveDataReadsHourOfWeek returns the UTC hour of the week, starting at Monday 00:00.
func ExcessiveDataReadsHourOfWeek(timestampMs int64) int32 {
	t := time.UnixMilli(timestampMs).UTC()
	return int32((int(t.Weekday())+6)%7*24 + t.Hour())
}
