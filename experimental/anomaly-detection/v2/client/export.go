package main

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"path/filepath"

	"github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common"
	"google.golang.org/protobuf/encoding/protojson"

	auditpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common/proto"
)

// exportLogs validates the generated count and exports every action as JSONL.
// The v1 generator and v2 ingestion AuditLogEntry schemas are wire-compatible.
func exportLogs(ctx context.Context, out string, totalCount int64) (err error) {
	file, err := os.Create(filepath.Join(out, "logs.jsonl"))
	if err != nil {
		return err
	}
	defer func() {
		if closeErr := file.Close(); err == nil {
			err = closeErr
		}
	}()
	writer := bufio.NewWriter(file)
	defer func() {
		if flushErr := writer.Flush(); err == nil {
			err = flushErr
		}
	}()
	var count int64
	jsonOptions := protojson.MarshalOptions{UseProtoNames: true, EmitUnpopulated: true}
	err = common.IngestBinProtoStream(ctx, filepath.Join(out, "logs.binproto"), func(entry *auditpb.AuditLogEntry) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		if entry.GetNormalizedQueryTemplate() == "" {
			return fmt.Errorf("log %d has an empty normalized query template", count+1)
		}
		data, err := jsonOptions.Marshal(entry)
		if err != nil {
			return err
		}
		if _, err := writer.Write(append(data, '\n')); err != nil {
			return err
		}
		count++
		return nil
	})
	if err != nil {
		return err
	}
	if count != totalCount {
		return fmt.Errorf("generator wrote %d actions, expected %d", count, totalCount)
	}
	return nil
}
