package common

import (
	"fmt"
	"math"
	"time"
)

type rowWindowKey struct {
	dbName string
	dbUser string
}

type rowRead struct {
	timestampMs  int64
	rowsReturned int64
}

type userRowWindow struct {
	reads []rowRead
	total int64
}

// SlidingRowWindow counts rows per database/user across all query templates.
// Create separate instances for training and prediction to keep their state apart.
type SlidingRowWindow struct {
	durationMs int64
	windows    map[rowWindowKey]*userRowWindow
}

// NewSlidingRowWindow creates an empty window with millisecond timestamp precision.
func NewSlidingRowWindow(duration time.Duration) (*SlidingRowWindow, error) {
	if duration < time.Millisecond {
		return nil, fmt.Errorf("%w: window duration must be at least 1 millisecond", ErrInvalidArgument)
	}
	return &SlidingRowWindow{
		durationMs: duration.Milliseconds(),
		windows:    make(map[rowWindowKey]*userRowWindow),
	}, nil
}

// Add returns the row total in (timestamp - duration, timestamp], including this read.
// Events must arrive in timestamp order per database/user; equal timestamps are allowed.
// Totals include partial windows at startup. Add does not support concurrent calls.
func (w *SlidingRowWindow) Add(feature ExtractedLogFeatures) (int64, error) {
	if feature.TimestampMs < 0 || feature.RowsReturned < 0 {
		return 0, fmt.Errorf("%w: timestamp and rows returned must be non-negative", ErrInvalidArgument)
	}
	key := rowWindowKey{dbName: feature.DbName, dbUser: feature.DbUser}
	window := w.windows[key]
	if window == nil {
		window = &userRowWindow{}
	}
	if len(window.reads) > 0 && feature.TimestampMs < window.reads[len(window.reads)-1].timestampMs {
		return 0, fmt.Errorf("%w: timestamp %d precedes the last read for database %q, user %q", ErrInvalidArgument, feature.TimestampMs, feature.DbName, feature.DbUser)
	}

	cutoff := feature.TimestampMs - w.durationMs
	total := window.total
	expired := 0
	for expired < len(window.reads) && window.reads[expired].timestampMs <= cutoff {
		total -= window.reads[expired].rowsReturned
		expired++
	}
	if feature.RowsReturned > math.MaxInt64-total {
		return 0, fmt.Errorf("%w: row total overflows for database %q, user %q", ErrInvalidArgument, feature.DbName, feature.DbUser)
	}

	// Validate before changing the queue so a rejected event leaves the window intact.
	window.reads = window.reads[expired:]
	if len(window.reads) == 0 {
		window.reads = nil
	}
	window.reads = append(window.reads, rowRead{timestampMs: feature.TimestampMs, rowsReturned: feature.RowsReturned})
	window.total = total + feature.RowsReturned
	w.windows[key] = window
	return window.total, nil
}
