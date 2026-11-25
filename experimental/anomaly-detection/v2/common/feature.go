// Package common provides shared utilities for AlloyDB anomaly detection.
package common

import (
	"errors"

	auditpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common/proto"
)

// ErrInvalidArgument represents an invalid argument error.
var ErrInvalidArgument = errors.New("invalid argument")

// ErrNotFound represents a resource not found error.
var ErrNotFound = errors.New("not found")

// ExtractedLogFeatures contains the audit fields used by the trainers and models.
type ExtractedLogFeatures struct {
	DbUser        string
	DbName        string
	QueryTemplate string
	TimestampMs   int64
	RowsReturned  int64
}

// Extract reads the model inputs from an audit log entry.
func Extract(logEntry *auditpb.AuditLogEntry) (ExtractedLogFeatures, error) {
	if logEntry == nil {
		return ExtractedLogFeatures{}, ErrInvalidArgument
	}
	return ExtractedLogFeatures{
		DbUser:        logEntry.GetDbUser(),
		DbName:        logEntry.GetDbName(),
		QueryTemplate: logEntry.GetNormalizedQueryTemplate(),
		TimestampMs:   logEntry.GetTimestampMs(),
		RowsReturned:  logEntry.GetRowsReturned(),
	}, nil
}
