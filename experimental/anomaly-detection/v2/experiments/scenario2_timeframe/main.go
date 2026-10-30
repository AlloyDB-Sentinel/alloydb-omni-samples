// Command scenario2_timeframe compares training durations at their best integer constant.
package main

import (
	"bufio"
	"encoding/csv"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	"github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common"
	auditpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common/proto"
	"google.golang.org/protobuf/encoding/protodelim"
)

const (
	week                 = 7 * 24 * time.Hour
	timeframe            = 4 * time.Hour
	slotsPerWeek         = int(week / timeframe)
	historyWeeks         = 50
	validationWeeks      = 4
	originalHistoryWeeks = 21
	earlierHistoryWeeks  = 1 + historyWeeks - originalHistoryWeeks
	totalWeeks           = 1 + historyWeeks + validationWeeks // First week warms the rolling window.
	trainingEndBucket    = (1 + historyWeeks) * slotsPerWeek
	maxConstant          = 8
	firstSeed            = 42
	seedCount            = 5
)

var trainingWeeks = [...]int{1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 30, 40, 50}

type bucket struct {
	peak    int64
	anomaly bool
}
type confusion struct{ tp, fp, tn, fn int }
type sweep [len(trainingWeeks)][maxConstant]confusion
type metricSet struct{ accuracy, precision, recall float64 }

// CSV columns keep the same order in the summary and per-seed audit.
func (m metricSet) csvValues() [3]float64 {
	return [3]float64{m.accuracy, m.precision, m.recall}
}

type result struct {
	weeks, constant int
	metrics         metricSet
}

func (c confusion) metrics() metricSet {
	ratio := func(a, b int) float64 {
		if b == 0 {
			return 0
		}
		return float64(a) / float64(b)
	}
	return metricSet{
		accuracy:  ratio(c.tp+c.tn, c.tp+c.fp+c.tn+c.fn),
		precision: ratio(c.tp, c.tp+c.fp),
		recall:    ratio(c.tp, c.tp+c.fn),
	}
}

// Each slot gets one observation per training week; day/night slots stay separate.
func thresholds(training []bucket, constant int) [slotsPerWeek]float64 {
	var limits [slotsPerWeek]float64
	for slot := range limits {
		var mean, m2 float64
		n := 0
		for i := slot; i < len(training); i += slotsPerWeek {
			n++
			delta := float64(training[i].peak) - mean
			mean += delta / float64(n)
			m2 += delta * (float64(training[i].peak) - mean)
		}
		limits[slot] = mean + float64(constant)*math.Sqrt(m2/float64(n))
	}
	return limits
}

func score(validation []bucket, limits [slotsPerWeek]float64) confusion {
	var counts confusion
	for i, b := range validation {
		predicted := float64(b.peak) > limits[i%slotsPerWeek]
		switch {
		case predicted && b.anomaly:
			counts.tp++
		case predicted:
			counts.fp++
		case b.anomaly:
			counts.fn++
		default:
			counts.tn++
		}
	}
	return counts
}

func evaluate(data []bucket) (sweep, error) {
	var scores sweep
	if len(data) != totalWeeks*slotsPerWeek {
		return scores, fmt.Errorf("expected %d buckets, got %d", totalWeeks*slotsPerWeek, len(data))
	}
	for _, b := range data[slotsPerWeek:trainingEndBucket] {
		if b.anomaly {
			return scores, fmt.Errorf("training contains an anomaly")
		}
	}
	positives := 0
	for _, b := range data[trainingEndBucket:] {
		if b.anomaly {
			positives++
		}
	}
	if positives == 0 || positives == len(data)-trainingEndBucket {
		return scores, fmt.Errorf("validation must contain both classes")
	}
	for i, weeks := range trainingWeeks {
		training := data[trainingEndBucket-weeks*slotsPerWeek : trainingEndBucket]
		for c := 1; c <= maxConstant; c++ {
			scores[i][c-1] = score(data[trainingEndBucket:], thresholds(training, c))
		}
	}
	return scores, nil
}

func summarize(runs []sweep) []result {
	results := make([]result, len(trainingWeeks))
	for i, weeks := range trainingWeeks {
		best, mostCorrect := 0, -1
		for c := 0; c < maxConstant; c++ {
			correct := 0
			for _, run := range runs {
				correct += run[i][c].tp + run[i][c].tn
			}
			// Equal validation sizes make total correct equivalent to mean accuracy.
			// Ascending candidates retain the smaller constant on ties.
			if correct > mostCorrect {
				best, mostCorrect = c, correct
			}
		}
		results[i] = result{weeks: weeks, constant: best + 1}
		for _, run := range runs {
			metrics := run[i][best].metrics()
			results[i].metrics.accuracy += metrics.accuracy / float64(len(runs))
			results[i].metrics.precision += metrics.precision / float64(len(runs))
			results[i].metrics.recall += metrics.recall / float64(len(runs))
		}
	}
	return results
}

func aggregate(input io.Reader, start time.Time) ([]bucket, error) {
	data := make([]bucket, totalWeeks*slotsPerWeek)
	window, err := common.NewSlidingRowWindow(timeframe)
	if err != nil {
		return nil, err
	}
	reader := bufio.NewReader(input)
	lastTimestampMs, lastAttackMs := start.UnixMilli(), int64(-1)
	var identity [2]string
	seen := false
	for record := 0; ; record++ {
		entry := &auditpb.AuditLogEntry{}
		if err := protodelim.UnmarshalFrom(reader, entry); err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return nil, fmt.Errorf("decode record %d: %w", record, err)
		}
		feature, err := common.Extract(entry)
		if err != nil {
			return nil, fmt.Errorf("extract record %d: %w", record, err)
		}
		elapsedMs := feature.TimestampMs - start.UnixMilli()
		if feature.TimestampMs < lastTimestampMs || elapsedMs < 0 || elapsedMs >= (totalWeeks*week).Milliseconds() {
			return nil, fmt.Errorf("record %d: timestamp %d precedes %d or is outside [%d, %d)", record, feature.TimestampMs, lastTimestampMs, start.UnixMilli(), start.Add(totalWeeks*week).UnixMilli())
		}
		key := [2]string{feature.DbName, feature.DbUser}
		if seen && key != identity {
			return nil, fmt.Errorf("record %d: expected database/user %q, got %q", record, identity, key)
		}
		identity, seen, lastTimestampMs = key, true, feature.TimestampMs
		rows, err := window.Add(feature)
		if err != nil {
			return nil, fmt.Errorf("rolling rows at record %d: %w", record, err)
		}
		if entry.GetIsGroundTruthAnomalous() {
			lastAttackMs = feature.TimestampMs
		}
		b := &data[elapsedMs/timeframe.Milliseconds()]
		b.peak = max(b.peak, rows)
		// Ground truth follows the same open-left rolling window, including spillover.
		b.anomaly = b.anomaly || (lastAttackMs >= 0 && lastAttackMs > feature.TimestampMs-timeframe.Milliseconds())
	}
	if !seen {
		return nil, fmt.Errorf("empty input")
	}
	return data, nil
}

func generate(generator, config, output string, seed int, start, trainEnd, end time.Time) error {
	cmd := exec.Command(generator, "--config_file="+config, "--output_dir="+output, "--pgaudit_scenario", "--bucket_size_minutes=60",
		fmt.Sprintf("--seed=%d", seed), fmt.Sprintf("--start_time_ms=%d", start.UnixMilli()), fmt.Sprintf("--train_end_time_ms=%d", trainEnd.UnixMilli()), fmt.Sprintf("--end_time_ms=%d", end.UnixMilli()))
	cmd.Env = append(os.Environ(), "TZ=UTC")
	if output, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("generate seed %d: %w\n%s", seed, err, output)
	}
	return nil
}

func runSeed(generator, config, dir string, seed int) (sweep, error) {
	anchor := time.Date(2026, time.January, 5, 0, 0, 0, 0, time.UTC)
	start := anchor.Add(-earlierHistoryWeeks * week)
	// Prepend independent clean history while preserving the original 1–20-week
	// training and validation samples. Re-seeding a single longer run would change them.
	segments := []struct {
		name                 string
		seed                 int
		start, trainEnd, end time.Time
	}{
		{"earlier", seed + 3000, start, anchor, anchor},
		{"original", seed, anchor, anchor.Add(originalHistoryWeeks * week), anchor.Add((originalHistoryWeeks + validationWeeks) * week)},
	}
	var readers []io.Reader
	for _, segment := range segments {
		output := filepath.Join(dir, fmt.Sprint(seed), segment.name)
		if err := generate(generator, config, output, segment.seed, segment.start, segment.trainEnd, segment.end); err != nil {
			return sweep{}, err
		}
		file, err := os.Open(filepath.Join(output, "logs.binproto"))
		if err != nil {
			return sweep{}, err
		}
		defer file.Close()
		readers = append(readers, file)
	}
	data, err := aggregate(io.MultiReader(readers...), start)
	if err != nil {
		return sweep{}, err
	}
	return evaluate(data)
}

func writeCSV(path string, rows [][]string) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	writer := csv.NewWriter(file)
	err = writer.WriteAll(rows)
	closeErr := file.Close()
	return errors.Join(err, closeErr)
}

func writeResults(output string, runs []sweep) error {
	raw := [][]string{{"seed", "weeks", "c", "tp", "fp", "tn", "fn", "accuracy", "precision", "recall"}}
	for seed, run := range runs {
		for i, weeks := range trainingWeeks {
			for c, counts := range run[i] {
				row := []string{fmt.Sprint(firstSeed + seed), fmt.Sprint(weeks), fmt.Sprint(c + 1), fmt.Sprint(counts.tp), fmt.Sprint(counts.fp), fmt.Sprint(counts.tn), fmt.Sprint(counts.fn)}
				for _, metric := range counts.metrics().csvValues() {
					row = append(row, fmt.Sprint(metric))
				}
				raw = append(raw, row)
			}
		}
	}
	if err := writeCSV(filepath.Join(output, "per_seed.csv"), raw); err != nil {
		return err
	}
	results := summarize(runs)
	summary := [][]string{{"training_weeks", "best_constant", "accuracy_percent", "precision_percent", "recall_percent"}}
	for _, r := range results {
		row := []string{fmt.Sprint(r.weeks), fmt.Sprint(r.constant)}
		for _, metric := range r.metrics.csvValues() {
			row = append(row, fmt.Sprintf("%.2f", 100*metric))
		}
		summary = append(summary, row)
	}
	if err := writeCSV(filepath.Join(output, "training_duration.csv"), summary); err != nil {
		return err
	}
	return writeChart(filepath.Join(output, "training_duration.png"), results)
}

func run(output string) error {
	// Run from the anomaly-detection module, as documented in README.md.
	config := filepath.Join("v2", "experiments", "scenario2_timeframe", "config.textproto")
	if _, err := os.Stat(config); err != nil {
		return fmt.Errorf("run from experimental/anomaly-detection: %w", err)
	}
	if err := os.MkdirAll(output, 0755); err != nil {
		return err
	}
	temp, err := os.MkdirTemp("", "scenario2-duration-")
	if err != nil {
		return err
	}
	defer os.RemoveAll(temp)
	generator := filepath.Join(temp, "generator")
	build := exec.Command("go", "build", "-o", generator, "./v1/generator/src")
	if output, err := build.CombinedOutput(); err != nil {
		return fmt.Errorf("build generator: %w\n%s", err, output)
	}
	runs := make([]sweep, seedCount)
	for i := range runs {
		runs[i], err = runSeed(generator, config, temp, firstSeed+i)
		if err != nil {
			return fmt.Errorf("seed %d: %w", firstSeed+i, err)
		}
		fmt.Fprintf(os.Stderr, "Completed seed %d (%d/%d)\n", firstSeed+i, i+1, seedCount)
	}
	return writeResults(output, runs)
}

func main() {
	output := flag.String("output", filepath.Join("v2", "experiments", "scenario2_timeframe", "results"), "Directory for the sweep CSV, summary CSV, and chart")
	flag.Parse()
	if err := run(*output); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
