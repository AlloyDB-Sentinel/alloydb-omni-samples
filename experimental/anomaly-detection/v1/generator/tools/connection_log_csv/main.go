// Command connection_log_csv converts a generated connection log file (.binproto) into a CSV file
// that can be opened in a spreadsheet.
//
// Usage:
//
//	go run ./v1/generator/tools/connection_log_csv --input=output/logs.binproto --output=output/logs.csv
package main

import (
	"bufio"
	"encoding/csv"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"strconv"
	"strings"
	"time"

	"google.golang.org/protobuf/encoding/protodelim"

	genpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v1/generator/proto"
)

var (
	inputPath  = flag.String("input", "", "Path to the generated connection log file (.binproto). Required.")
	outputPath = flag.String("output", "", "Path to the CSV file to write. Defaults to the input path with a .csv extension.")
)

// csvHeader lists the CSV columns in the order exportConnectionLogsToCSV writes them.
var csvHeader = []string{
	"time_utc",
	"client_address",
	"client_port",
	"outcome",
	"db_user",
	"db_name",
	"application_name",
	"auth_method",
	"backend_pid",
	"is_ground_truth_anomalous",
	"ground_truth_profile",
	"project",
	"cluster_id",
	"instance_id",
	"node_id",
}

func main() {
	flag.Parse()
	if *inputPath == "" {
		log.Fatalf("required flag --input is not set")
	}
	csvPath := *outputPath
	if csvPath == "" {
		csvPath = strings.TrimSuffix(*inputPath, ".binproto") + ".csv"
	}

	inputFile, err := os.Open(*inputPath)
	if err != nil {
		log.Fatalf("Failed to open input file: %v", err)
	}
	defer inputFile.Close()

	csvFile, err := os.Create(csvPath)
	if err != nil {
		log.Fatalf("Failed to create output file: %v", err)
	}

	rowCount, exportErr := exportConnectionLogsToCSV(bufio.NewReader(inputFile), csvFile)
	closeErr := csvFile.Close()
	if exportErr != nil {
		log.Fatalf("Failed to convert %q: %v", *inputPath, exportErr)
	}
	if closeErr != nil {
		log.Fatalf("Failed to write %q: %v", csvPath, closeErr)
	}
	log.Printf("Wrote %d connection logs to %q", rowCount, csvPath)
}

// exportConnectionLogsToCSV reads length-delimited ConnectionLogEntry records and writes one CSV row
// per record, after a header row. It returns the number of records written.
func exportConnectionLogsToCSV(reader protodelim.Reader, writer io.Writer) (int, error) {
	csvWriter := csv.NewWriter(writer)
	if err := csvWriter.Write(csvHeader); err != nil {
		return 0, err
	}

	rowCount := 0
	for {
		entry := &genpb.ConnectionLogEntry{}
		err := protodelim.UnmarshalFrom(reader, entry)
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return rowCount, fmt.Errorf("failed to read connection log %d: %w", rowCount+1, err)
		}
		if err := csvWriter.Write(formatConnectionLogRow(entry)); err != nil {
			return rowCount, err
		}
		rowCount++
	}

	csvWriter.Flush()
	return rowCount, csvWriter.Error()
}

// formatConnectionLogRow converts one record into CSV cells matching csvHeader.
func formatConnectionLogRow(entry *genpb.ConnectionLogEntry) []string {
	source := entry.GetSource()
	return []string{
		time.UnixMilli(entry.GetTimestampMs()).UTC().Format("2006-01-02 15:04:05.000"),
		entry.GetClientAddress(),
		strconv.Itoa(int(entry.GetClientPort())),
		entry.GetOutcome().String(),
		entry.GetDbUser(),
		entry.GetDbName(),
		entry.GetApplicationName(),
		entry.GetAuthMethod(),
		strconv.Itoa(int(entry.GetBackendPid())),
		strconv.FormatBool(entry.GetIsGroundTruthAnomalous()),
		entry.GetGroundTruthProfile().String(),
		source.GetProject(),
		source.GetClusterId(),
		source.GetInstanceId(),
		source.GetNodeId(),
	}
}
