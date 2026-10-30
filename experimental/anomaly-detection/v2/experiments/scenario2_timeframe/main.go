package main

import (
	"bufio"
	"encoding/csv"
	"errors"
	"flag"
	"fmt"
	"html"
	"io"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common"
	auditpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common/proto"
	"google.golang.org/protobuf/encoding/protodelim"
)

var frames = []timeframe{
	{name: "1m", duration: time.Minute},
	{name: "5m", duration: 5 * time.Minute},
	{name: "15m", duration: 15 * time.Minute},
	{name: "1h", duration: time.Hour},
	{name: "4h", duration: 4 * time.Hour},
	{name: "1d", duration: 24 * time.Hour},
	{name: "7d", duration: 7 * 24 * time.Hour},
}

var constants = []float64{0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 40, 60, 80, 100, 150, 200, 300, 500, 1000, 2000, 5000}

const (
	seedCount       = 10
	firstSeed       = 42
	trainDays       = 28
	validationDays  = 7
	testDays        = 7
	generatorBucket = 60
)

type timeframe struct {
	name     string
	duration time.Duration
}

type moments struct {
	n    int64
	mean float64
	m2   float64
}

func (m *moments) add(value float64) {
	m.n++
	delta := value - m.mean
	m.mean += delta / float64(m.n)
	m.m2 += delta * (value - m.mean)
}

func (m moments) stddev() float64 {
	if m.n == 0 {
		return 0
	}
	return math.Sqrt(m.m2 / float64(m.n))
}

type confusion struct {
	tp int64
	fp int64
	tn int64
	fn int64
}

func (c *confusion) add(predicted, truth bool) {
	switch {
	case predicted && truth:
		c.tp++
	case predicted && !truth:
		c.fp++
	case !predicted && truth:
		c.fn++
	default:
		c.tn++
	}
}

func (c *confusion) merge(other confusion) {
	c.tp += other.tp
	c.fp += other.fp
	c.tn += other.tn
	c.fn += other.fn
}

func (c confusion) n() int64 { return c.tp + c.fp + c.tn + c.fn }

func (c confusion) metrics() metricSet {
	accuracy := divide(float64(c.tp+c.tn), float64(c.n()))
	precision := divide(float64(c.tp), float64(c.tp+c.fp))
	recall := divide(float64(c.tp), float64(c.tp+c.fn))
	f1 := divide(2*precision*recall, precision+recall)
	return metricSet{accuracy: accuracy, precision: precision, recall: recall, f1: f1}
}

type metricSet struct {
	accuracy  float64
	precision float64
	recall    float64
	f1        float64
}

func divide(numerator, denominator float64) float64 {
	if denominator == 0 {
		return 0
	}
	return numerator / denominator
}

type bucketObservation struct {
	maxRowsReturned int64
	anomalous       bool
}

func (b *bucketObservation) add(rowsReturned int64, anomalous bool) {
	if rowsReturned > b.maxRowsReturned {
		b.maxRowsReturned = rowsReturned
	}
	b.anomalous = b.anomalous || anomalous
}

type seedResult struct {
	validation []confusion // frame index, then candidate index
	testing    []confusion
	baselines  []moments
}

type frameResult struct {
	frame             timeframe
	trainingBuckets   int
	validationBuckets int
	testBuckets       int
	constant          float64
	selectedIndex     int
	validation        confusion
	testing           confusion
	testMean          metricSet
	testStddev        metricSet
	baselineMean      float64
	baselineStd       float64
}

func main() {
	configFlag := flag.String("config", "", "Scenario 2 textproto (defaults to v1/client/query_template_config_s2.textproto).")
	outputFlag := flag.String("output", "", "Directory for the experiment report, CSV, and SVG charts.")
	flag.Parse()

	root, err := findModuleRoot()
	if err != nil {
		fatal(err)
	}
	configPath := *configFlag
	if configPath == "" {
		configPath = filepath.Join(root, "v1", "client", "query_template_config_s2.textproto")
	} else if !filepath.IsAbs(configPath) {
		configPath = filepath.Join(root, configPath)
	}
	outputDir := *outputFlag
	if outputDir == "" {
		outputDir = filepath.Join(root, "v2", "experiments", "scenario2_timeframe", "results")
	} else if !filepath.IsAbs(outputDir) {
		outputDir = filepath.Join(root, outputDir)
	}
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		fatal(err)
	}

	results, err := runExperiment(root, configPath)
	if err != nil {
		fatal(err)
	}
	if err := writeOutputs(outputDir, results); err != nil {
		fatal(err)
	}
	fmt.Printf("Wrote Scenario 2 timeframe experiment to %s\n", outputDir)
}

func runExperiment(root, configPath string) ([]frameResult, error) {
	start := time.Date(2026, time.January, 5, 0, 0, 0, 0, time.UTC)
	trainEnd := start.Add(trainDays * 24 * time.Hour)
	validationEnd := trainEnd.Add(validationDays * 24 * time.Hour)
	end := validationEnd.Add(testDays * 24 * time.Hour)

	tmpDir, err := os.MkdirTemp("", "scenario2-timeframe-")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(tmpDir)

	generatorPath := filepath.Join(tmpDir, "scenario2-generator.exe")
	build := exec.Command("go", "build", "-o", generatorPath, "./v1/generator/src")
	build.Dir = root
	build.Stdout = os.Stdout
	build.Stderr = os.Stderr
	if err := build.Run(); err != nil {
		return nil, fmt.Errorf("build Scenario 2 generator: %w", err)
	}

	seedResults := make([]seedResult, 0, seedCount)
	for seed := int64(firstSeed); seed < firstSeed+seedCount; seed++ {
		fmt.Printf("Generating and evaluating seed %d (%d/%d)\n", seed, seed-firstSeed+1, seedCount)
		seedDir := filepath.Join(tmpDir, fmt.Sprintf("seed-%d", seed))
		if err := os.MkdirAll(seedDir, 0755); err != nil {
			return nil, err
		}
		if err := generateSeed(generatorPath, configPath, seedDir, start, trainEnd, end, seed); err != nil {
			return nil, err
		}
		result, err := evaluateSeed(filepath.Join(seedDir, "logs.binproto"), start, trainEnd, validationEnd, end)
		if err != nil {
			return nil, fmt.Errorf("evaluate seed %d: %w", seed, err)
		}
		seedResults = append(seedResults, result)
	}
	return summarize(seedResults), nil
}

func generateSeed(generatorPath, configPath, outputDir string, start, trainEnd, end time.Time, seed int64) error {
	args := []string{
		"--config_file=" + configPath,
		"--output_dir=" + outputDir,
		"--start_time_ms=" + strconv.FormatInt(start.UnixMilli(), 10),
		"--train_end_time_ms=" + strconv.FormatInt(trainEnd.UnixMilli(), 10),
		"--end_time_ms=" + strconv.FormatInt(end.UnixMilli(), 10),
		"--bucket_size_minutes=" + strconv.Itoa(generatorBucket),
		"--pgaudit_scenario",
		"--seed=" + strconv.FormatInt(seed, 10),
	}
	cmd := exec.Command(generatorPath, args...)
	cmd.Stdout = io.Discard
	cmd.Stderr = os.Stderr
	cmd.Env = withEnv(os.Environ(), "TZ", "UTC")
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("generate seed %d: %w", seed, err)
	}
	return nil
}

func evaluateSeed(path string, start, trainEnd, validationEnd, end time.Time) (seedResult, error) {
	result := seedResult{
		validation: make([]confusion, len(frames)*len(constants)),
		testing:    make([]confusion, len(frames)*len(constants)),
		baselines:  make([]moments, len(frames)),
	}
	windows := make([]*common.SlidingRowWindow, len(frames))
	buckets := make([][]bucketObservation, len(frames))
	for i, frame := range frames {
		window, err := common.NewSlidingRowWindow(frame.duration)
		if err != nil {
			return seedResult{}, err
		}
		windows[i] = window
		if end.Sub(start)%frame.duration != 0 {
			return seedResult{}, fmt.Errorf("simulation duration must be divisible by timeframe %s", frame.name)
		}
		buckets[i] = make([]bucketObservation, int(end.Sub(start)/frame.duration))
	}

	file, err := os.Open(path)
	if err != nil {
		return seedResult{}, err
	}
	defer file.Close()
	reader := bufio.NewReader(file)
	var previousTimestamp int64
	for record := 0; ; record++ {
		entry := &auditpb.AuditLogEntry{}
		err := protodelim.UnmarshalFrom(reader, entry)
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return seedResult{}, fmt.Errorf("decode record %d: %w", record, err)
		}
		feature, err := common.Extract(entry)
		if err != nil {
			return seedResult{}, fmt.Errorf("extract record %d: %w", record, err)
		}
		if record > 0 && feature.TimestampMs < previousTimestamp {
			return seedResult{}, fmt.Errorf("input is not timestamp ordered at record %d", record)
		}
		previousTimestamp = feature.TimestampMs
		timestamp := time.UnixMilli(feature.TimestampMs)
		if timestamp.Before(start) || !timestamp.Before(end) {
			return seedResult{}, fmt.Errorf("record %d timestamp %s is outside simulation interval [%s, %s)", record, timestamp, start, end)
		}
		truth := entry.GetIsGroundTruthAnomalous()
		for fi, frame := range frames {
			total, err := windows[fi].Add(feature)
			if err != nil {
				return seedResult{}, fmt.Errorf("record %d, frame %s: %w", record, frame.name, err)
			}
			bucketIndex := int(timestamp.Sub(start) / frame.duration)
			buckets[fi][bucketIndex].add(total, truth)
		}
	}

	for fi, frame := range frames {
		if trainEnd.Sub(start)%frame.duration != 0 || validationEnd.Sub(trainEnd)%frame.duration != 0 {
			return seedResult{}, fmt.Errorf("training and validation boundaries must align to timeframe %s", frame.name)
		}
		trainingBucketCount := int(trainEnd.Sub(start) / frame.duration)
		validationBucketCount := int(validationEnd.Sub(trainEnd) / frame.duration)
		baseline, validation, testing := evaluateBuckets(buckets[fi], trainingBucketCount, validationBucketCount)
		if baseline.n == 0 {
			return seedResult{}, fmt.Errorf("frame %s has no training buckets", frame.name)
		}
		result.baselines[fi] = baseline
		for ci := range constants {
			flatIndex := fi*len(constants) + ci
			result.validation[flatIndex] = validation[ci]
			result.testing[flatIndex] = testing[ci]
		}
	}
	return result, nil
}

func evaluateBuckets(buckets []bucketObservation, trainingBucketCount, validationBucketCount int) (moments, []confusion, []confusion) {
	var baseline moments
	for _, bucket := range buckets[:trainingBucketCount] {
		baseline.add(float64(bucket.maxRowsReturned))
	}
	validation := make([]confusion, len(constants))
	testing := make([]confusion, len(constants))
	validationEnd := trainingBucketCount + validationBucketCount
	stddev := baseline.stddev()
	for bucketIndex, bucket := range buckets[trainingBucketCount:] {
		for ci, constant := range constants {
			predicted := float64(bucket.maxRowsReturned) > baseline.mean+constant*stddev
			if trainingBucketCount+bucketIndex < validationEnd {
				validation[ci].add(predicted, bucket.anomalous)
			} else {
				testing[ci].add(predicted, bucket.anomalous)
			}
		}
	}
	return baseline, validation, testing
}

func summarize(seeds []seedResult) []frameResult {
	results := make([]frameResult, len(frames))
	allValidation = make([][]confusion, len(frames))
	for fi, frame := range frames {
		allValidation[fi] = make([]confusion, len(constants))
		bestIndex := 0
		var bestValidation confusion
		for ci := range constants {
			var pooled confusion
			for _, seed := range seeds {
				pooled.merge(seed.validation[fi*len(constants)+ci])
			}
			allValidation[fi][ci] = pooled
			if ci == 0 || betterValidation(pooled.metrics(), bestValidation.metrics()) {
				bestIndex = ci
				bestValidation = pooled
			}
		}

		var pooledTest confusion
		seedMetrics := make([]metricSet, 0, len(seeds))
		var baselineMean, baselineStd float64
		for _, seed := range seeds {
			flatIndex := fi*len(constants) + bestIndex
			pooledTest.merge(seed.testing[flatIndex])
			seedMetrics = append(seedMetrics, seed.testing[flatIndex].metrics())
			baselineMean += seed.baselines[fi].mean
			baselineStd += seed.baselines[fi].stddev()
		}
		n := float64(len(seeds))
		baselineMean /= n
		baselineStd /= n
		mean, std := meanAndStd(seedMetrics)
		results[fi] = frameResult{
			frame:             frame,
			trainingBuckets:   int(time.Duration(trainDays) * 24 * time.Hour / frame.duration),
			validationBuckets: int(time.Duration(validationDays) * 24 * time.Hour / frame.duration),
			testBuckets:       int(time.Duration(testDays) * 24 * time.Hour / frame.duration),
			constant:          constants[bestIndex],
			selectedIndex:     bestIndex,
			validation:        bestValidation,
			testing:           pooledTest,
			testMean:          mean,
			testStddev:        std,
			baselineMean:      baselineMean,
			baselineStd:       baselineStd,
		}
	}
	return results
}

func betterValidation(candidate, current metricSet) bool {
	if candidate.f1 > current.f1+1e-12 {
		return true
	}
	if math.Abs(candidate.f1-current.f1) <= 1e-12 {
		if candidate.precision > current.precision+1e-12 {
			return true
		}
		if math.Abs(candidate.precision-current.precision) <= 1e-12 && candidate.accuracy > current.accuracy+1e-12 {
			return true
		}
	}
	return false
}

func meanAndStd(values []metricSet) (metricSet, metricSet) {
	var mean metricSet
	for _, value := range values {
		mean.accuracy += value.accuracy
		mean.precision += value.precision
		mean.recall += value.recall
		mean.f1 += value.f1
	}
	n := float64(len(values))
	mean.accuracy /= n
	mean.precision /= n
	mean.recall /= n
	mean.f1 /= n
	variance := metricSet{}
	for _, value := range values {
		variance.accuracy += math.Pow(value.accuracy-mean.accuracy, 2)
		variance.precision += math.Pow(value.precision-mean.precision, 2)
		variance.recall += math.Pow(value.recall-mean.recall, 2)
		variance.f1 += math.Pow(value.f1-mean.f1, 2)
	}
	std := metricSet{
		accuracy:  math.Sqrt(variance.accuracy / n),
		precision: math.Sqrt(variance.precision / n),
		recall:    math.Sqrt(variance.recall / n),
		f1:        math.Sqrt(variance.f1 / n),
	}
	return mean, std
}

func writeOutputs(dir string, results []frameResult) error {
	if err := writeCSV(filepath.Join(dir, "constant_sweep.csv"), results); err != nil {
		return err
	}
	if err := writeValidationSVG(filepath.Join(dir, "validation_metrics.svg"), results); err != nil {
		return err
	}
	if err := writeTestSVG(filepath.Join(dir, "test_metrics.svg"), results); err != nil {
		return err
	}
	return writeReport(filepath.Join(dir, "experiment_report.md"), results)
}

func writeCSV(path string, results []frameResult) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()
	writer := csv.NewWriter(file)
	if err := writer.Write([]string{"timeframe", "window", "constant", "validation_n", "validation_tp", "validation_fp", "validation_tn", "validation_fn", "validation_accuracy", "validation_precision", "validation_recall", "validation_f1", "selected_test_n", "selected_test_tp", "selected_test_fp", "selected_test_tn", "selected_test_fn", "test_accuracy_mean", "test_accuracy_sd", "test_precision_mean", "test_precision_sd", "test_recall_mean", "test_recall_sd", "test_f1_mean", "test_f1_sd"}); err != nil {
		return err
	}
	for fi, result := range results {
		for ci, constant := range constants {
			validation := allValidation[fi][ci]
			metrics := validation.metrics()
			row := make([]string, 25)
			row[0] = result.frame.name
			row[1] = result.frame.duration.String()
			row[2] = strconv.FormatFloat(constant, 'f', -1, 64)
			row[3] = strconv.FormatInt(validation.n(), 10)
			row[4] = strconv.FormatInt(validation.tp, 10)
			row[5] = strconv.FormatInt(validation.fp, 10)
			row[6] = strconv.FormatInt(validation.tn, 10)
			row[7] = strconv.FormatInt(validation.fn, 10)
			row[8] = formatMetric(metrics.accuracy)
			row[9] = formatMetric(metrics.precision)
			row[10] = formatMetric(metrics.recall)
			row[11] = formatMetric(metrics.f1)
			if ci == result.selectedIndex {
				row[12] = strconv.FormatInt(result.testing.n(), 10)
				row[13] = strconv.FormatInt(result.testing.tp, 10)
				row[14] = strconv.FormatInt(result.testing.fp, 10)
				row[15] = strconv.FormatInt(result.testing.tn, 10)
				row[16] = strconv.FormatInt(result.testing.fn, 10)
				row[17] = formatMetric(result.testMean.accuracy)
				row[18] = formatMetric(result.testStddev.accuracy)
				row[19] = formatMetric(result.testMean.precision)
				row[20] = formatMetric(result.testStddev.precision)
				row[21] = formatMetric(result.testMean.recall)
				row[22] = formatMetric(result.testStddev.recall)
				row[23] = formatMetric(result.testMean.f1)
				row[24] = formatMetric(result.testStddev.f1)
			}
			if err := writer.Write(row); err != nil {
				return err
			}
		}
	}
	writer.Flush()
	return writer.Error()
}

var allValidation [][]confusion

func writeReport(path string, results []frameResult) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()
	var b strings.Builder
	b.WriteString("# Scenario 2: optimal constant by sliding-window timeframe\n\n")
	b.WriteString("## Summary\n\n")
	b.WriteString(fmt.Sprintf("This experiment independently sweeps the tested standard-deviation constants `c` for each requested rolling timeframe. The candidate grid is `[%s]`; the selected value is the best candidate in that grid, not a continuous optimization. Constants are selected using pooled validation F1; the held-out test period is not used to select them. These results are for the checked-in **synthetic generator and row-count sliding-window implementation**, not production traffic.\n\n", constantList()))
	b.WriteString("## Protocol\n\n")
	b.WriteString(fmt.Sprintf("- Signal: sum of `rows_returned` for the same database and database user over the trailing window `(timestamp - duration, timestamp]`, using `common.SlidingRowWindow`.\n- Timeframes: %s.\n- Bucket semantics: each timeframe is divided into fixed, non-overlapping buckets aligned to the simulation start. A bucket's signal is the maximum rolling `rows_returned` total observed on reads in that bucket; its ground truth is anomalous if any read in that bucket is labeled anomalous. Buckets with no reads are included with signal zero and normal ground truth.\n- Threshold: predict a bucket anomalous when its bucket signal is strictly greater than `training_mean + c * training_stddev`. The mean and population standard deviation are calculated from the training buckets' maximum rolling totals, including empty buckets.\n- Constant selection: maximize pooled validation F1 (harmonic mean of precision and recall); ties prefer higher precision, then higher accuracy, with the lower `c` retained for any remaining tie. Accuracy, precision, and recall are reported alongside F1.\n- Simulation: %d UTC days of training, %d validation days, and %d held-out test days; legacy generator buckets are %d minutes. Start time is 2026-01-05 00:00 UTC. Ten fixed seeds (%d–%d) use `v1/client/query_template_config_s2.textproto`.\n- Rolling row totals still advance at each read using `common.SlidingRowWindow`; only the reported prediction and metric sample are aggregated to one per non-overlapping timeframe bucket. Rolling state carries across the train/validation/test boundaries.\n- Test summary: arithmetic mean and population standard deviation of each metric across the ten per-seed test runs. Pooled test confusion counts are also included in the CSV.\n\n", durationList(), trainDays, validationDays, testDays, generatorBucket, firstSeed, firstSeed+seedCount-1))
	b.WriteString("## Bucket counts per seed\n\n| Timeframe | Training buckets | Validation buckets | Test buckets |\n|---:|---:|---:|---:|\n")
	for _, result := range results {
		b.WriteString(fmt.Sprintf("| %s | %d | %d | %d |\n", result.frame.name, result.trainingBuckets, result.validationBuckets, result.testBuckets))
	}
	b.WriteString("\nAt 7d there is only one validation and one test bucket per seed (10 of each pooled across seeds); treat those weekly metrics as highly uncertain despite the displayed seed-to-seed spread.\n\n")
	b.WriteString("## Results\n\n")
	b.WriteString("| Timeframe | Selected `c` | Validation F1 | Validation accuracy | Validation precision | Validation recall | Test accuracy (mean ± SD) | Test precision (mean ± SD) | Test recall (mean ± SD) | Test F1 (mean ± SD) |\n")
	b.WriteString("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
	for _, result := range results {
		val := result.validation.metrics()
		b.WriteString(fmt.Sprintf("| %s | %s | %.3f | %.3f | %.3f | %.3f | %s | %s | %s | %s |\n",
			result.frame.name, selectedConstant(result), val.f1, val.accuracy, val.precision, val.recall,
			meanSd(result.testMean.accuracy, result.testStddev.accuracy),
			meanSd(result.testMean.precision, result.testStddev.precision),
			meanSd(result.testMean.recall, result.testStddev.recall),
			meanSd(result.testMean.f1, result.testStddev.f1)))
	}
	b.WriteString("\nValidation metrics by candidate constant (all seven timeframes):\n\n![Validation metrics by timeframe and constant](validation_metrics.svg)\n\n")
	b.WriteString("Held-out test metrics at the validation-selected constant:\n\n![Test metrics by timeframe](test_metrics.svg)\n\n")
	b.WriteString("Full validation sweep is in [`constant_sweep.csv`](constant_sweep.csv). Test metric columns are populated only for the validation-selected constant of each timeframe, to keep the held-out set out of constant selection.\n\n")
	b.WriteString("## Interpretation and limitations\n\n")
	b.WriteString("This branch's implemented `SlidingRowWindow` sums `rows_returned`, so this experiment tests **rolling returned-row volume**, not query count. Evaluation is per fixed timeframe bucket: the maximum rolling total predicts the bucket, and any generator-labeled anomalous read makes its bucket ground-truth anomalous. These bucket aggregation rules are explicit experiment choices because the progress doc did not define how event labels and read-level scores map to time buckets. The generator emits synthetic, deliberately injected anomalies; selected constants are specific to this configuration, aggregation rules, split, candidate grid, and seed set. Results should not be interpreted as production thresholds or compared numerically with the intern's results until their data, split, aggregation, and optimization objective are aligned. The next phase can combine timeframe signals after these single-frame constants are reviewed.\n")
	_, err = file.WriteString(b.String())
	return err
}

func writeValidationSVG(path string, results []frameResult) error {
	const width = 1060
	const panelHeight = 178
	const top = 56
	height := top + panelHeight*len(frames) + 30
	var b strings.Builder
	b.WriteString(svgStart(width, height, "Validation metrics by timeframe and constant"))
	b.WriteString(fmt.Sprintf(`<text x="%d" y="28" class="title">Validation metrics by timeframe and constant (pooled across seeds)</text>`, width/2))
	colors := []string{"#2563eb", "#dc2626", "#059669"}
	labels := []string{"accuracy", "precision", "recall"}
	for j, label := range labels {
		b.WriteString(fmt.Sprintf(`<line x1="%d" y1="42" x2="%d" y2="42" stroke="%s" stroke-width="3"/><text x="%d" y="46" class="legend">%s</text>`, 30+j*140, 55+j*140, colors[j], 62+j*140, label))
	}
	for fi, result := range results {
		panelY := top + fi*panelHeight
		left, right := 76.0, float64(width-32)
		plotTop, plotBottom := float64(panelY+24), float64(panelY+132)
		b.WriteString(fmt.Sprintf(`<text x="14" y="%d" class="frame">%s</text>`, panelY+20, result.frame.name))
		for _, tick := range []float64{0, 0.5, 1} {
			y := plotBottom - tick*(plotBottom-plotTop)
			b.WriteString(fmt.Sprintf(`<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" class="grid"/><text x="53" y="%.1f" class="tick">%.1f</text>`, left, y, right, y, y+4, tick))
		}
		minLog, maxLog := math.Log10(constants[0]), math.Log10(constants[len(constants)-1])
		xFor := func(c float64) float64 { return left + (math.Log10(c)-minLog)/(maxLog-minLog)*(right-left) }
		selectedX := xFor(result.constant)
		b.WriteString(fmt.Sprintf(`<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#6b7280" stroke-dasharray="4 4"/><text x="%.1f" y="%.1f" class="selected">c=%.2g</text>`, selectedX, plotTop, selectedX, plotBottom, math.Min(selectedX+4, right-44), plotTop+12, result.constant))
		for _, c := range []float64{0.1, 0.5, 1, 2, 5, 10, 20, 50, 100, 500, 1000, 5000} {
			x := xFor(c)
			b.WriteString(fmt.Sprintf(`<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" class="axis"/><text x="%.1f" y="%.1f" class="tick">%g</text>`, x, plotBottom, x, plotBottom+4, x, plotBottom+17, c))
		}
		for mi, metric := range []string{"accuracy", "precision", "recall"} {
			var points strings.Builder
			for ci, c := range constants {
				value := allValidation[fi][ci].metrics().value(metric)
				x := xFor(c)
				y := plotBottom - value*(plotBottom-plotTop)
				if points.Len() > 0 {
					points.WriteByte(' ')
				}
				points.WriteString(fmt.Sprintf("%.1f,%.1f", x, y))
				b.WriteString(fmt.Sprintf(`<circle cx="%.1f" cy="%.1f" r="2.7" fill="%s"/>`, x, y, colors[mi]))
			}
			b.WriteString(fmt.Sprintf(`<polyline points="%s" fill="none" stroke="%s" stroke-width="2"/>`, points.String(), colors[mi]))
		}
	}
	b.WriteString(`</svg>`)
	return os.WriteFile(path, []byte(b.String()), 0644)
}

func writeTestSVG(path string, results []frameResult) error {
	const width = 1000
	const panelHeight = 205
	const top = 50
	height := top + panelHeight*3 + 25
	var b strings.Builder
	b.WriteString(svgStart(width, height, "Held-out test metrics by timeframe"))
	b.WriteString(fmt.Sprintf(`<text x="%d" y="27" class="title">Held-out test metrics at validation-selected constants (mean ± SD across seeds)</text>`, width/2))
	metrics := []struct {
		name  string
		color string
		get   func(metricSet) float64
	}{
		{"accuracy", "#2563eb", func(m metricSet) float64 { return m.accuracy }},
		{"precision", "#dc2626", func(m metricSet) float64 { return m.precision }},
		{"recall", "#059669", func(m metricSet) float64 { return m.recall }},
	}
	for mi, metric := range metrics {
		panelY := top + mi*panelHeight
		left, right := 70.0, float64(width-28)
		plotTop, plotBottom := float64(panelY+18), float64(panelY+138)
		b.WriteString(fmt.Sprintf(`<text x="15" y="%d" class="frame">%s</text>`, panelY+16, metric.name))
		for _, tick := range []float64{0, 0.5, 1} {
			y := plotBottom - tick*(plotBottom-plotTop)
			b.WriteString(fmt.Sprintf(`<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" class="grid"/><text x="47" y="%.1f" class="tick">%.1f</text>`, left, y, right, y, y+4, tick))
		}
		var points strings.Builder
		for fi, result := range results {
			x := left + (float64(fi)+0.5)*(right-left)/float64(len(results))
			value := metric.get(result.testMean)
			sd := metric.get(result.testStddev)
			y := plotBottom - value*(plotBottom-plotTop)
			yHigh := plotBottom - math.Min(1, value+sd)*(plotBottom-plotTop)
			yLow := plotBottom - math.Max(0, value-sd)*(plotBottom-plotTop)
			b.WriteString(fmt.Sprintf(`<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="1.5"/><line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s"/><circle cx="%.1f" cy="%.1f" r="4" fill="%s"/><text x="%.1f" y="%.1f" class="tick">%s</text>`, x, yHigh, x, yLow, metric.color, x-4, yHigh, x+4, yHigh, metric.color, x, y, metric.color, x, plotBottom+20, result.frame.name))
			if points.Len() > 0 {
				points.WriteByte(' ')
			}
			points.WriteString(fmt.Sprintf("%.1f,%.1f", x, y))
		}
		b.WriteString(fmt.Sprintf(`<polyline points="%s" fill="none" stroke="%s" stroke-width="2"/>`, points.String(), metric.color))
	}
	b.WriteString(`</svg>`)
	return os.WriteFile(path, []byte(b.String()), 0644)
}

func svgStart(width, height int, title string) string {
	return fmt.Sprintf(`<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d"><title>%s</title><style>text{font-family:Arial,sans-serif;fill:#1f2937}.title{text-anchor:middle;font-size:17px;font-weight:bold}.legend{font-size:12px}.frame{font-size:13px;font-weight:bold}.tick{font-size:10px;text-anchor:middle}.grid{stroke:#d1d5db;stroke-width:1}.axis{stroke:#6b7280;stroke-width:1}.selected{font-size:10px;fill:#374151}</style>`, width, height, width, height, html.EscapeString(title))
}

func durationList() string {
	values := make([]string, len(frames))
	for i, frame := range frames {
		values[i] = frame.name
	}
	return strings.Join(values, ", ")
}

func selectedConstant(result frameResult) string {
	value := fmt.Sprintf("%.2f", result.constant)
	if result.selectedIndex == len(constants)-1 {
		return value + " **(upper grid boundary)**"
	}
	return value
}

func constantList() string {
	values := make([]string, len(constants))
	for i, constant := range constants {
		values[i] = strconv.FormatFloat(constant, 'f', -1, 64)
	}
	return strings.Join(values, ", ")
}

func (m metricSet) value(name string) float64 {
	switch name {
	case "accuracy":
		return m.accuracy
	case "precision":
		return m.precision
	default:
		return m.recall
	}
}

func formatMetric(value float64) string { return strconv.FormatFloat(value, 'f', 6, 64) }

func meanSd(mean, sd float64) string { return fmt.Sprintf("%.3f ± %.3f", mean, sd) }

func withEnv(env []string, key, value string) []string {
	prefix := key + "="
	for i, item := range env {
		if strings.HasPrefix(item, prefix) {
			env[i] = prefix + value
			return env
		}
	}
	return append(env, prefix+value)
}

func findModuleRoot() (string, error) {
	cwd, err := os.Getwd()
	if err != nil {
		return "", err
	}
	for dir := cwd; ; dir = filepath.Dir(dir) {
		data, err := os.ReadFile(filepath.Join(dir, "go.mod"))
		if err == nil && strings.Contains(string(data), "module github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection") {
			return dir, nil
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
	}
	return "", fmt.Errorf("could not find the experimental/anomaly-detection Go module from %s", cwd)
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "scenario2 timeframe experiment:", err)
	os.Exit(1)
}
