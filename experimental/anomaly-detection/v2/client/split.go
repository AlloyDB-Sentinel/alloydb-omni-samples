package main

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"path/filepath"

	"github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common"
	"google.golang.org/protobuf/encoding/protodelim"
	"google.golang.org/protobuf/encoding/protojson"

	auditpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common/proto"
)

// splitLogs preserves file order and exports every generated action as JSONL.
// The v1 generator and v2 ingestion AuditLogEntry schemas are wire-compatible.
func splitLogs(ctx context.Context, out string, trainCount, totalCount int) (err error) {
	paths := []string{"train.binproto", "test.binproto", "logs.jsonl"}
	var writers []*bufio.Writer
	for _, path := range paths {
		file, openErr := os.Create(filepath.Join(out, path))
		if openErr != nil {
			return openErr
		}
		defer func() {
			if closeErr := file.Close(); err == nil {
				err = closeErr
			}
		}()
		writer := bufio.NewWriter(file)
		writers = append(writers, writer)
		defer func() {
			if flushErr := writer.Flush(); err == nil {
				err = flushErr
			}
		}()
	}
	count := 0
	jsonOptions := protojson.MarshalOptions{UseProtoNames: true, EmitUnpopulated: true}
	err = common.IngestBinProtoStream(ctx, filepath.Join(out, "logs.binproto"), func(entry *auditpb.AuditLogEntry) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		if entry.GetNormalizedQueryTemplate() == "" {
			return fmt.Errorf("log %d has an empty normalized query template", count+1)
		}
		writer := writers[1]
		if count < trainCount {
			writer = writers[0]
		}
		if _, err := protodelim.MarshalTo(writer, entry); err != nil {
			return err
		}
		data, err := jsonOptions.Marshal(entry)
		if err != nil {
			return err
		}
		if _, err := writers[2].Write(append(data, '\n')); err != nil {
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
