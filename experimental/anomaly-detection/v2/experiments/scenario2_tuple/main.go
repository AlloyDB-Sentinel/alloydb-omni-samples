// Command scenario2_tuple evaluates OR- and AND-combined subsets of rolling
// timeframes at a fixed 20-week training duration.
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
	"runtime"
	"sort"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common"
	auditpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common/proto"
	"google.golang.org/protobuf/encoding/protodelim"
)

const (
	week               = 7 * 24 * time.Hour
	reporting          = 4 * time.Hour
	slotsPerWeek       = int(week / reporting) // 42 four-hour slots per week
	warmupWeeks        = 1
	trainingWeeks      = 20
	validationWeeks    = 4
	testWeeks          = 4
	totalWeeks         = warmupWeeks + trainingWeeks + validationWeeks + testWeeks
	trainingStart      = warmupWeeks * slotsPerWeek
	trainingEnd        = trainingStart + trainingWeeks*slotsPerWeek
	validationEnd      = trainingEnd + validationWeeks*slotsPerWeek
	totalBuckets       = totalWeeks * slotsPerWeek
	firstSeed          = 42
	seedCount          = 10
	generatorBucketMin = 60
)

var frames = []frameSpec{
	{name: "1m", duration: time.Minute},
	{name: "5m", duration: 5 * time.Minute},
	{name: "15m", duration: 15 * time.Minute},
	{name: "1h", duration: time.Hour},
	{name: "4h", duration: 4 * time.Hour},
	{name: "1d", duration: 24 * time.Hour},
	{name: "7d", duration: 7 * 24 * time.Hour},
}

// The earlier timeframe sweep's grid is used here because long windows need
// sub-1 constants. The training-duration experiment's integer 1-8 grid was
// specific to the fixed 4-hour frame.
var constants = []float64{0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 15, 20, 30, 40, 60, 80, 100, 150, 200, 300, 500, 1000, 2000, 5000}

type frameSpec struct {
	name     string
	duration time.Duration
}

type combiner string

const (
	orCombination  combiner = "OR"
	andCombination combiner = "AND"
)

func (c combiner) order() int {
	if c == orCombination {
		return 0
	}
	return 1
}

type dataset struct {
	peaks  [][]int64 // [frame][reporting bucket] maximum rolling total in the bucket
	labels []bool    // [reporting bucket] shared trailing-4-hour attack label
}

type moments struct {
	n    int
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

func (c confusion) metrics() metricSet {
	ratio := func(numerator, denominator int64) float64 {
		if denominator == 0 {
			return 0
		}
		return float64(numerator) / float64(denominator)
	}
	precision := ratio(c.tp, c.tp+c.fp)
	recall := ratio(c.tp, c.tp+c.fn)
	return metricSet{
		accuracy:  ratio(c.tp+c.tn, c.tp+c.fp+c.tn+c.fn),
		precision: precision,
		recall:    recall,
		f1:        ratio(2*c.tp, 2*c.tp+c.fp+c.fn),
	}
}

type metricSet struct {
	accuracy  float64
	precision float64
	recall    float64
	f1        float64
}

// better reports whether a should be selected over b. F1 is the primary
// objective; ties prefer precision, then accuracy, then the earlier candidate.
func better(a, b metricSet) bool {
	if a.f1 != b.f1 {
		return a.f1 > b.f1
	}
	if a.precision != b.precision {
		return a.precision > b.precision
	}
	if a.accuracy != b.accuracy {
		return a.accuracy > b.accuracy
	}
	return false
}

// limitsFor learns one mean + c*stddev threshold per weekly slot from the
// training buckets, so day/night and weekday/weekend slots stay separate.
func limitsFor(training []int64, constant float64) [slotsPerWeek]float64 {
	var limits [slotsPerWeek]float64
	for slot := range limits {
		var m moments
		for i := slot; i < len(training); i += slotsPerWeek {
			m.add(float64(training[i]))
		}
		limits[slot] = m.mean + constant*m.stddev()
	}
	return limits
}

func scoreFrame(peaks []int64, labels []bool, limits [slotsPerWeek]float64, start, end int) confusion {
	var c confusion
	for i := start; i < end; i++ {
		c.add(float64(peaks[i]) > limits[i%slotsPerWeek], labels[i])
	}
	return c
}

// scoreSubset evaluates one combiner for a non-empty subset of frames. OR
// fires when any member exceeds its threshold; AND fires only when every
// member does.
func scoreSubset(data dataset, members []int, limits [][slotsPerWeek]float64, combination combiner, start, end int) confusion {
	var c confusion
	for i := start; i < end; i++ {
		predicted := combination == andCombination
		for _, fi := range members {
			hit := float64(data.peaks[fi][i]) > limits[fi][i%slotsPerWeek]
			if combination == orCombination {
				if hit {
					predicted = true
					break
				}
			} else if !hit {
				predicted = false
				break
			}
		}
		c.add(predicted, data.labels[i])
	}
	return c
}

func aggregate(input io.Reader, start time.Time) (dataset, error) {
	data := dataset{
		peaks:  make([][]int64, len(frames)),
		labels: make([]bool, totalBuckets),
	}
	for i := range data.peaks {
		data.peaks[i] = make([]int64, totalBuckets)
	}
	windows := make([]*common.SlidingRowWindow, len(frames))
	for i, frame := range frames {
		window, err := common.NewSlidingRowWindow(frame.duration)
		if err != nil {
			return dataset{}, err
		}
		windows[i] = window
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
			return dataset{}, fmt.Errorf("decode record %d: %w", record, err)
		}
		feature, err := common.Extract(entry)
		if err != nil {
			return dataset{}, fmt.Errorf("extract record %d: %w", record, err)
		}
		elapsedMs := feature.TimestampMs - start.UnixMilli()
		if feature.TimestampMs < lastTimestampMs || elapsedMs < 0 || elapsedMs >= (totalWeeks*week).Milliseconds() {
			return dataset{}, fmt.Errorf("record %d: timestamp %d precedes %d or is outside [%d, %d)", record, feature.TimestampMs, lastTimestampMs, start.UnixMilli(), start.Add(totalWeeks*week).UnixMilli())
		}
		key := [2]string{feature.DbName, feature.DbUser}
		if seen && key != identity {
			return dataset{}, fmt.Errorf("record %d: expected database/user %q, got %q", record, identity, key)
		}
		identity, seen, lastTimestampMs = key, true, feature.TimestampMs

		bucketIndex := int(elapsedMs / reporting.Milliseconds())
		for fi, window := range windows {
			rows, err := window.Add(feature)
			if err != nil {
				return dataset{}, fmt.Errorf("rolling rows at record %d, frame %s: %w", record, frames[fi].name, err)
			}
			if rows > data.peaks[fi][bucketIndex] {
				data.peaks[fi][bucketIndex] = rows
			}
		}
		if entry.GetIsGroundTruthAnomalous() {
			lastAttackMs = feature.TimestampMs
		}
		// Ground truth follows the trailing 4-hour window, including spillover.
		data.labels[bucketIndex] = data.labels[bucketIndex] || (lastAttackMs >= 0 && lastAttackMs > feature.TimestampMs-reporting.Milliseconds())
	}
	if !seen {
		return dataset{}, fmt.Errorf("empty input")
	}
	return data, nil
}

func trainingPeaks(data dataset, frame int) []int64 {
	return data.peaks[frame][trainingStart:trainingEnd]
}

func validateDataset(data dataset) error {
	for i := trainingStart; i < trainingEnd; i++ {
		if data.labels[i] {
			return fmt.Errorf("training contains an anomaly at bucket %d", i)
		}
	}
	for _, split := range []struct {
		name       string
		start, end int
	}{
		{"validation", trainingEnd, validationEnd},
		{"test", validationEnd, totalBuckets},
	} {
		positives := 0
		for i := split.start; i < split.end; i++ {
			if data.labels[i] {
				positives++
			}
		}
		if positives == 0 || positives == split.end-split.start {
			return fmt.Errorf("%s must contain both classes, got %d anomalous of %d buckets", split.name, positives, split.end-split.start)
		}
	}
	return nil
}

type frameResult struct {
	frame           frameSpec
	constant        float64
	validation      confusion
	test            confusion
	validationSweep []confusion
	testSweep       []confusion
}

type subsetSpec struct {
	members     []int
	combination combiner
}

type subsetResult struct {
	members     []int
	combination combiner
	validation  confusion
	test        confusion
}

func evaluate(datasets []dataset) ([]frameResult, []subsetResult, error) {
	for i, data := range datasets {
		if err := validateDataset(data); err != nil {
			return nil, nil, fmt.Errorf("seed %d: %w", firstSeed+i, err)
		}
	}

	frameResults := make([]frameResult, len(frames))
	for fi, frame := range frames {
		validation := make([]confusion, len(constants))
		test := make([]confusion, len(constants))
		for ci, constant := range constants {
			for _, data := range datasets {
				limits := limitsFor(trainingPeaks(data, fi), constant)
				validation[ci].merge(scoreFrame(data.peaks[fi], data.labels, limits, trainingEnd, validationEnd))
				test[ci].merge(scoreFrame(data.peaks[fi], data.labels, limits, validationEnd, totalBuckets))
			}
		}
		best := 0
		for ci := 1; ci < len(constants); ci++ {
			if better(validation[ci].metrics(), validation[best].metrics()) {
				best = ci
			}
		}
		frameResults[fi] = frameResult{
			frame:           frame,
			constant:        constants[best],
			validation:      validation[best],
			test:            test[best],
			validationSweep: validation,
			testSweep:       test,
		}
	}

	limits := make([][][slotsPerWeek]float64, len(datasets))
	for si, data := range datasets {
		limits[si] = make([][slotsPerWeek]float64, len(frames))
		for fi := range frames {
			limits[si][fi] = limitsFor(trainingPeaks(data, fi), frameResults[fi].constant)
		}
	}

	specs := subsetSpecs()
	subsets := make([]subsetResult, 0, len(specs))
	for _, spec := range specs {
		result := subsetResult{members: spec.members, combination: spec.combination}
		for si, data := range datasets {
			result.validation.merge(scoreSubset(data, spec.members, limits[si], spec.combination, trainingEnd, validationEnd))
			result.test.merge(scoreSubset(data, spec.members, limits[si], spec.combination, validationEnd, totalBuckets))
		}
		subsets = append(subsets, result)
	}
	return frameResults, subsets, nil
}

// subsetSpecs returns every non-empty subset of frames under both combiners,
// ordered by size, then member indices, then OR before AND. This ordering is
// the final tie-break when validation metrics are identical.
func subsetSpecs() []subsetSpec {
	specs := make([]subsetSpec, 0, 2*((1<<len(frames))-1))
	for mask := 1; mask < 1<<len(frames); mask++ {
		var members []int
		for fi := range frames {
			if mask&(1<<fi) != 0 {
				members = append(members, fi)
			}
		}
		specs = append(specs,
			subsetSpec{members: members, combination: orCombination},
			subsetSpec{members: members, combination: andCombination},
		)
	}
	sort.SliceStable(specs, func(i, j int) bool {
		a, b := specs[i], specs[j]
		if len(a.members) != len(b.members) {
			return len(a.members) < len(b.members)
		}
		for k := range a.members {
			if a.members[k] != b.members[k] {
				return a.members[k] < b.members[k]
			}
		}
		return a.combination.order() < b.combination.order()
	})
	return specs
}

func subsetName(members []int) string {
	names := make([]string, len(members))
	for i, fi := range members {
		names[i] = frames[fi].name
	}
	return strings.Join(names, "+")
}

func subsetConstants(members []int, frameResults []frameResult) string {
	values := make([]string, len(members))
	for i, fi := range members {
		values[i] = fmt.Sprint(frameResults[fi].constant)
	}
	return strings.Join(values, "|")
}

func constantDescription(members []int, frameResults []frameResult) string {
	parts := make([]string, len(members))
	for i, fi := range members {
		parts[i] = fmt.Sprintf("`c = %g` for %s", frameResults[fi].constant, frames[fi].name)
	}
	return strings.Join(parts, ", ")
}

func sortedSubsets(subsets []subsetResult) []subsetResult {
	sorted := append([]subsetResult(nil), subsets...)
	sort.SliceStable(sorted, func(i, j int) bool {
		return better(sorted[i].validation.metrics(), sorted[j].validation.metrics())
	})
	return sorted
}

func bestForCombination(subsets []subsetResult, combination combiner) subsetResult {
	var best subsetResult
	found := false
	for _, r := range subsets {
		if r.combination != combination {
			continue
		}
		if !found || better(r.validation.metrics(), best.validation.metrics()) {
			best, found = r, true
		}
	}
	return best
}

func bestForSize(subsets []subsetResult, size int, combination combiner) subsetResult {
	var best subsetResult
	found := false
	for _, r := range subsets {
		if len(r.members) != size || r.combination != combination {
			continue
		}
		if !found || better(r.validation.metrics(), best.validation.metrics()) {
			best, found = r, true
		}
	}
	return best
}

func bestForMinSize(subsets []subsetResult, minSize int, combination combiner) subsetResult {
	var best subsetResult
	found := false
	for _, r := range subsets {
		if len(r.members) < minSize || r.combination != combination {
			continue
		}
		if !found || better(r.validation.metrics(), best.validation.metrics()) {
			best, found = r, true
		}
	}
	return best
}

func writeFrameCSV(path string, results []frameResult) error {
	rows := [][]string{{"frame", "constant", "validation_tp", "validation_fp", "validation_tn", "validation_fn", "validation_accuracy", "validation_precision", "validation_recall", "validation_f1", "test_tp", "test_fp", "test_tn", "test_fn", "test_accuracy", "test_precision", "test_recall", "test_f1"}}
	for _, r := range results {
		v, t := r.validation.metrics(), r.test.metrics()
		rows = append(rows, []string{
			r.frame.name, fmt.Sprint(r.constant),
			fmt.Sprint(r.validation.tp), fmt.Sprint(r.validation.fp), fmt.Sprint(r.validation.tn), fmt.Sprint(r.validation.fn),
			fmt.Sprint(v.accuracy), fmt.Sprint(v.precision), fmt.Sprint(v.recall), fmt.Sprint(v.f1),
			fmt.Sprint(r.test.tp), fmt.Sprint(r.test.fp), fmt.Sprint(r.test.tn), fmt.Sprint(r.test.fn),
			fmt.Sprint(t.accuracy), fmt.Sprint(t.precision), fmt.Sprint(t.recall), fmt.Sprint(t.f1),
		})
	}
	return writeCSV(path, rows)
}

func writeFrameSweepCSV(path string, results []frameResult) error {
	rows := [][]string{{"frame", "constant", "validation_tp", "validation_fp", "validation_tn", "validation_fn", "validation_accuracy", "validation_precision", "validation_recall", "validation_f1", "test_tp", "test_fp", "test_tn", "test_fn", "test_accuracy", "test_precision", "test_recall", "test_f1"}}
	for _, r := range results {
		for ci, constant := range constants {
			v, t := r.validationSweep[ci].metrics(), r.testSweep[ci].metrics()
			rows = append(rows, []string{
				r.frame.name, fmt.Sprint(constant),
				fmt.Sprint(r.validationSweep[ci].tp), fmt.Sprint(r.validationSweep[ci].fp), fmt.Sprint(r.validationSweep[ci].tn), fmt.Sprint(r.validationSweep[ci].fn),
				fmt.Sprint(v.accuracy), fmt.Sprint(v.precision), fmt.Sprint(v.recall), fmt.Sprint(v.f1),
				fmt.Sprint(r.testSweep[ci].tp), fmt.Sprint(r.testSweep[ci].fp), fmt.Sprint(r.testSweep[ci].tn), fmt.Sprint(r.testSweep[ci].fn),
				fmt.Sprint(t.accuracy), fmt.Sprint(t.precision), fmt.Sprint(t.recall), fmt.Sprint(t.f1),
			})
		}
	}
	return writeCSV(path, rows)
}

func writeSubsetCSV(path string, subsets []subsetResult, frameResults []frameResult) error {
	rows := [][]string{{"subset", "size", "combination", "constants", "validation_tp", "validation_fp", "validation_tn", "validation_fn", "validation_accuracy", "validation_precision", "validation_recall", "validation_f1", "test_tp", "test_fp", "test_tn", "test_fn", "test_accuracy", "test_precision", "test_recall", "test_f1"}}
	for _, r := range sortedSubsets(subsets) {
		v, t := r.validation.metrics(), r.test.metrics()
		rows = append(rows, []string{
			subsetName(r.members), fmt.Sprint(len(r.members)), string(r.combination), subsetConstants(r.members, frameResults),
			fmt.Sprint(r.validation.tp), fmt.Sprint(r.validation.fp), fmt.Sprint(r.validation.tn), fmt.Sprint(r.validation.fn),
			fmt.Sprint(v.accuracy), fmt.Sprint(v.precision), fmt.Sprint(v.recall), fmt.Sprint(v.f1),
			fmt.Sprint(r.test.tp), fmt.Sprint(r.test.fp), fmt.Sprint(r.test.tn), fmt.Sprint(r.test.fn),
			fmt.Sprint(t.accuracy), fmt.Sprint(t.precision), fmt.Sprint(t.recall), fmt.Sprint(t.f1),
		})
	}
	return writeCSV(path, rows)
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

func writeReport(path, config string, frameResults []frameResult, subsets []subsetResult) error {
	sorted := sortedSubsets(subsets)
	if len(sorted) == 0 {
		return fmt.Errorf("no subset results")
	}
	best := sorted[0]
	bestOR := bestForCombination(sorted, orCombination)
	bestAND := bestForCombination(sorted, andCombination)
	bestMultiAND := bestForMinSize(sorted, 2, andCombination)
	bestSingle := bestForSize(sorted, 1, orCombination)
	bestValidation, bestTest := best.validation.metrics(), best.test.metrics()
	orValidation, orTest := bestOR.validation.metrics(), bestOR.test.metrics()
	andValidation, andTest := bestAND.validation.metrics(), bestAND.test.metrics()
	multiANDValidation, multiANDTest := bestMultiAND.validation.metrics(), bestMultiAND.test.metrics()
	singleValidation, singleTest := bestSingle.validation.metrics(), bestSingle.test.metrics()

	orValidationBeats, orTestBeats := 0, 0
	andValidationBeats, andTestBeats := 0, 0
	for _, r := range sorted {
		v, t := r.validation.metrics(), r.test.metrics()
		switch r.combination {
		case orCombination:
			if v.f1 > singleValidation.f1 {
				orValidationBeats++
			}
			if t.f1 > singleTest.f1 {
				orTestBeats++
			}
		case andCombination:
			if v.f1 > singleValidation.f1 {
				andValidationBeats++
			}
			if t.f1 > singleTest.f1 {
				andTestBeats++
			}
		}
	}

	var b strings.Builder
	b.WriteString("# Scenario 2: OR- and AND-combined timeframe subsets\n\n")
	b.WriteString("## Summary\n\n")
	b.WriteString(fmt.Sprintf("This experiment fixes clean training history at **%d weeks** and evaluates all **%d non-empty subsets** of the seven rolling timeframes under both **OR** and **AND** combination (%d candidates). Constants are selected independently per timeframe by pooled validation F1 across the %d seeds; each subset/combiner pair is then scored at those constants, and the best candidate is selected by pooled validation F1. Test metrics are reported at the validation-selected settings and are not used for selection.\n\n", trainingWeeks, (1<<len(frames))-1, len(sorted), seedCount))
	b.WriteString("These are synthetic results from the checked-in generator and should not be read as production thresholds.\n\n")

	b.WriteString("## Protocol\n\n")
	b.WriteString(fmt.Sprintf("- Signal: sum of `rows_returned` for the same database and database user over each frame's trailing window, using `common.SlidingRowWindow`: %s.\n", frameList()))
	b.WriteString("- Reporting grid: fixed, non-overlapping **4-hour** buckets aligned to the simulation start. Each frame contributes the **maximum** rolling total observed among reads in the bucket; buckets with no reads contribute zero.\n")
	b.WriteString("- Ground truth: a 4-hour bucket is anomalous if any observed trailing 4-hour window contains an attack read, including spillover after an episode.\n")
	b.WriteString(fmt.Sprintf("- Training: %d warm-up week, then %d clean training weeks. Baselines learn one `mean + c × population stddev` threshold per weekly slot (%d slots), using the 20 training observations for that slot.\n", warmupWeeks, trainingWeeks, slotsPerWeek))
	b.WriteString(fmt.Sprintf("- Split: %d validation weeks, then %d held-out test weeks. %d deterministic seeds (%d–%d) use `%s`.\n", validationWeeks, testWeeks, seedCount, firstSeed, firstSeed+seedCount-1, filepath.ToSlash(config)))
	b.WriteString("- Timezone: all timestamps and day/night peak windows are evaluated in UTC. The generator uses UTC explicitly because Windows ignores `TZ=UTC`.\n")
	b.WriteString(fmt.Sprintf("- Candidate constants: `[%s]`. This is the earlier timeframe sweep's grid, which includes sub-1 values needed by long windows; the training-duration experiment's integer 1–8 grid was specific to the fixed 4-hour frame.\n", constantList()))
	b.WriteString("- Selection: maximize pooled validation F1 across all subset/combiner candidates; ties prefer higher precision, then higher accuracy, then the earlier candidate (smaller subset, lexicographic member order, OR before AND).\n")
	b.WriteString("- Combination: **OR** flags a bucket when at least one member exceeds its threshold; **AND** flags only when every member does. Singletons are identical under both combiners.\n\n")
	b.WriteString(fmt.Sprintf("Each seed has %d training buckets, %d validation buckets, and %d test buckets. The full %d-row candidate table is in `subset_results.csv`.\n\n", trainingEnd-trainingStart, validationEnd-trainingEnd, totalBuckets-validationEnd, len(sorted)))

	b.WriteString("## Individual timeframe constants at 20 weeks\n\n")
	b.WriteString("| Timeframe | Selected `c` | Validation accuracy | Validation precision | Validation recall | Validation F1 | Test accuracy | Test precision | Test recall | Test F1 |\n")
	b.WriteString("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
	for _, r := range frameResults {
		v, t := r.validation.metrics(), r.test.metrics()
		b.WriteString(fmt.Sprintf("| %s | %g | %.2f%% | %.2f%% | %.2f%% | %.2f%% | %.2f%% | %.2f%% | %.2f%% | %.2f%% |\n", r.frame.name, r.constant, 100*v.accuracy, 100*v.precision, 100*v.recall, 100*v.f1, 100*t.accuracy, 100*t.precision, 100*t.recall, 100*t.f1))
	}
	b.WriteString("\n")

	b.WriteString("## Best subset by size\n\n")
	b.WriteString("For each subset size, the best OR and AND candidate by pooled validation F1.\n\n")
	b.WriteString("| Size | OR subset | OR validation F1 | OR test F1 | AND subset | AND validation F1 | AND test F1 |\n")
	b.WriteString("|---:|---|---:|---:|---|---:|---:|\n")
	for size := 1; size <= len(frames); size++ {
		or := bestForSize(sorted, size, orCombination)
		and := bestForSize(sorted, size, andCombination)
		orV, orT := or.validation.metrics(), or.test.metrics()
		andV, andT := and.validation.metrics(), and.test.metrics()
		b.WriteString(fmt.Sprintf("| %d | %s | %.2f%% | %.2f%% | %s | %.2f%% | %.2f%% |\n", size, subsetName(or.members), 100*orV.f1, 100*orT.f1, subsetName(and.members), 100*andV.f1, 100*andT.f1))
	}
	b.WriteString("\n")

	b.WriteString("## Best OR and best AND\n\n")
	b.WriteString(fmt.Sprintf("- **Best OR:** %s at %s — validation F1 %.2f%%, test F1 %.2f%%.\n", subsetName(bestOR.members), constantDescription(bestOR.members, frameResults), 100*orValidation.f1, 100*orTest.f1))
	b.WriteString(fmt.Sprintf("- **Best AND:** %s at %s — validation F1 %.2f%%, test F1 %.2f%%.\n", subsetName(bestAND.members), constantDescription(bestAND.members, frameResults), 100*andValidation.f1, 100*andTest.f1))
	b.WriteString(fmt.Sprintf("- **Best multi-member AND:** %s at %s — validation F1 %.2f%%, test F1 %.2f%%.\n\n", subsetName(bestMultiAND.members), constantDescription(bestMultiAND.members, frameResults), 100*multiANDValidation.f1, 100*multiANDTest.f1))

	b.WriteString("## Comparison with the best single frame\n\n")
	b.WriteString(fmt.Sprintf("The best single frame is **%s** at %s (validation F1 %.2f%%, test F1 %.2f%%). %d OR candidates beat it on validation F1 and %d on test F1; %d AND candidates beat it on validation F1 and %d on test F1.\n\n", subsetName(bestSingle.members), constantDescription(bestSingle.members, frameResults), 100*singleValidation.f1, 100*singleTest.f1, orValidationBeats, orTestBeats, andValidationBeats, andTestBeats))

	b.WriteString("## Top candidates\n\n")
	b.WriteString("The top 20 candidates overall, sorted by pooled validation F1. The selected candidate is marked.\n\n")
	b.WriteString("| Rank | Combination | Subset | Size | Validation accuracy | Validation precision | Validation recall | Validation F1 | Test accuracy | Test precision | Test recall | Test F1 |\n")
	b.WriteString("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
	limit := 20
	if len(sorted) < limit {
		limit = len(sorted)
	}
	for rank, r := range sorted[:limit] {
		v, t := r.validation.metrics(), r.test.metrics()
		name := subsetName(r.members)
		if r.combination == best.combination && sameMembers(r.members, best.members) {
			name = "**" + name + "**"
		}
		b.WriteString(fmt.Sprintf("| %d | %s | %s | %d | %.2f%% | %.2f%% | %.2f%% | %.2f%% | %.2f%% | %.2f%% | %.2f%% | %.2f%% |\n", rank+1, r.combination, name, len(r.members), 100*v.accuracy, 100*v.precision, 100*v.recall, 100*v.f1, 100*t.accuracy, 100*t.precision, 100*t.recall, 100*t.f1))
	}
	b.WriteString("\n")

	b.WriteString("## Selected candidate\n\n")
	b.WriteString(fmt.Sprintf("**%s (%s)** with %s.\n\n", subsetName(best.members), best.combination, constantDescription(best.members, frameResults)))
	b.WriteString(fmt.Sprintf("- Validation: accuracy %.2f%%, precision %.2f%%, recall %.2f%%, F1 %.2f%%\n", 100*bestValidation.accuracy, 100*bestValidation.precision, 100*bestValidation.recall, 100*bestValidation.f1))
	b.WriteString(fmt.Sprintf("- Held-out test: accuracy %.2f%%, precision %.2f%%, recall %.2f%%, F1 %.2f%%\n\n", 100*bestTest.accuracy, 100*bestTest.precision, 100*bestTest.recall, 100*bestTest.f1))
	b.WriteString("If another candidate ties the selected candidate on pooled validation F1, the earlier candidate in size-then-lexicographic order is retained; the test column is not used for that decision.\n\n")

	b.WriteString("## Interpretation and limitations\n\n")
	b.WriteString(fmt.Sprintf("OR admits a bucket when any member fires, so it raises recall; AND requires every member, so it raises precision but can suppress recall. In this run, %d OR candidates beat the best single frame on validation F1 and %d on test, while %d AND candidates beat it on validation and %d on test. The best multi-member AND is %s: it reaches %.2f%% validation precision and %.2f%% recall, so it is only attractive if a false positive is much more costly than a miss.\n\n", orValidationBeats, orTestBeats, andValidationBeats, andTestBeats, subsetName(bestMultiAND.members), 100*multiANDValidation.precision, 100*multiANDValidation.recall))

	var outcome strings.Builder
	if tied := tiedCandidates(sorted, best); len(tied) > 0 {
		tiedTestF1 := tied[0].test.metrics().f1
		for _, r := range tied[1:] {
			tiedTestF1 = math.Max(tiedTestF1, r.test.metrics().f1)
		}
		outcome.WriteString(fmt.Sprintf("Other candidates reach the same pooled validation F1 (%.2f%%): %s. The size-then-lexicographic tie-break retains %s. ", 100*bestValidation.f1, candidateNames(tied), subsetName(best.members)))
		if tiedTestF1 > bestTest.f1 {
			outcome.WriteString(fmt.Sprintf("The tied candidates reach up to %.2f%% held-out test F1 against %.2f%% for the selected candidate, so the tie-break rather than the held-out evidence decided the selection. ", 100*tiedTestF1, 100*bestTest.f1))
		}
	}
	if superset, ok := bestStrictSuperset(sorted, best); ok {
		supersetValidation := superset.validation.metrics()
		if supersetValidation.f1 > bestValidation.f1 {
			outcome.WriteString(fmt.Sprintf("The best strict %s superset, %s, raises validation F1 to %.2f%%. ", best.combination, subsetName(superset.members), 100*supersetValidation.f1))
		} else {
			outcome.WriteString(fmt.Sprintf("No strict %s superset of %s improves on it; the best is %s at %.2f%% validation F1. ", best.combination, subsetName(best.members), subsetName(superset.members), 100*supersetValidation.f1))
		}
	}
	for _, r := range sorted {
		if r.combination == orCombination && len(r.members) == len(frames) {
			v := r.validation.metrics()
			outcome.WriteString(fmt.Sprintf("All %d frames under OR reach %.2f%% validation recall at %.2f%% validation precision. ", len(frames), 100*v.recall, 100*v.precision))
			break
		}
	}
	if outcome.Len() > 0 {
		b.WriteString(strings.TrimSpace(outcome.String()))
		b.WriteString("\n\n")
	}
	b.WriteString("Because every member uses the constant selected for that frame alone, neither combiner is jointly optimized. Long frames (1d and 7d) produce rolling totals that change slowly, so their 4-hour bucket signals are highly autocorrelated and their labels can remain positive for a long tail after an episode; their apparent performance should be read with that in mind. Constants and the candidate are both selected on the same four validation weeks, which makes the validation figures optimistic; the four test weeks are held out and are the better guide. The generator emits synthetic, deliberately injected anomalies with one database/user identity, and this runner does not exercise the RPC services. The next step would be to validate the selected candidate on a second independent period and to test a joint constant sweep for the winning subset.\n")
	return os.WriteFile(path, []byte(b.String()), 0644)
}

// tiedCandidates returns the candidates other than best that the selection rule
// treats as tied with it: equal pooled validation F1, precision, and accuracy.
func tiedCandidates(sorted []subsetResult, best subsetResult) []subsetResult {
	bestValidation := best.validation.metrics()
	var tied []subsetResult
	for _, r := range sorted {
		if r.combination == best.combination && sameMembers(r.members, best.members) {
			continue
		}
		v := r.validation.metrics()
		if v.f1 == bestValidation.f1 && v.precision == bestValidation.precision && v.accuracy == bestValidation.accuracy {
			tied = append(tied, r)
		}
	}
	return tied
}

// bestStrictSuperset returns the best-scoring candidate that uses the same
// combiner as best and contains all of its members plus at least one more.
func bestStrictSuperset(sorted []subsetResult, best subsetResult) (subsetResult, bool) {
	var found subsetResult
	ok := false
	for _, r := range sorted {
		if r.combination != best.combination || len(r.members) <= len(best.members) || !containsAll(r.members, best.members) {
			continue
		}
		if !ok || better(r.validation.metrics(), found.validation.metrics()) {
			found, ok = r, true
		}
	}
	return found, ok
}

func containsAll(members, required []int) bool {
	for _, want := range required {
		found := false
		for _, member := range members {
			if member == want {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}

// candidateNames joins subset names for prose, truncating very long tie sets.
func candidateNames(candidates []subsetResult) string {
	const limit = 8
	names := make([]string, 0, len(candidates))
	for i, r := range candidates {
		if i == limit {
			return fmt.Sprintf("%s, and %d more", strings.Join(names, ", "), len(candidates)-limit)
		}
		names = append(names, subsetName(r.members))
	}
	return strings.Join(names, ", ")
}

func sameMembers(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func frameList() string {
	names := make([]string, len(frames))
	for i, frame := range frames {
		names[i] = frame.name
	}
	return strings.Join(names, ", ")
}

func constantList() string {
	values := make([]string, len(constants))
	for i, constant := range constants {
		values[i] = fmt.Sprint(constant)
	}
	return strings.Join(values, ", ")
}

func generate(generator, config, output string, seed int, start, trainEnd, end time.Time) error {
	cmd := exec.Command(generator, "--config_file="+config, "--output_dir="+output, "--pgaudit_scenario", fmt.Sprintf("--bucket_size_minutes=%d", generatorBucketMin),
		fmt.Sprintf("--seed=%d", seed), fmt.Sprintf("--start_time_ms=%d", start.UnixMilli()), fmt.Sprintf("--train_end_time_ms=%d", trainEnd.UnixMilli()), fmt.Sprintf("--end_time_ms=%d", end.UnixMilli()))
	cmd.Env = append(os.Environ(), "TZ=UTC")
	if output, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("generate seed %d: %w\n%s", seed, err, output)
	}
	return nil
}

func runSeed(generator, config, dir string, seed int) (dataset, error) {
	anchor := time.Date(2026, time.January, 5, 0, 0, 0, 0, time.UTC)
	start := anchor
	trainEnd := anchor.Add((warmupWeeks + trainingWeeks) * week)
	end := anchor.Add(totalWeeks * week)
	output := filepath.Join(dir, fmt.Sprint(seed))
	if err := generate(generator, config, output, seed, start, trainEnd, end); err != nil {
		return dataset{}, err
	}
	file, err := os.Open(filepath.Join(output, "logs.binproto"))
	if err != nil {
		return dataset{}, err
	}
	defer file.Close()
	return aggregate(file, start)
}

func run(config, output string) error {
	if _, err := os.Stat(config); err != nil {
		return fmt.Errorf("run from experimental/anomaly-detection and keep the config path: %w", err)
	}
	if err := os.MkdirAll(output, 0755); err != nil {
		return err
	}
	temp, err := os.MkdirTemp("", "scenario2-tuple-")
	if err != nil {
		return err
	}
	defer os.RemoveAll(temp)
	generator := filepath.Join(temp, "generator")
	if runtime.GOOS == "windows" {
		generator += ".exe"
	}
	build := exec.Command("go", "build", "-o", generator, "./v1/generator/src")
	if buildOutput, err := build.CombinedOutput(); err != nil {
		return fmt.Errorf("build generator: %w\n%s", err, buildOutput)
	}

	datasets := make([]dataset, seedCount)
	for i := range datasets {
		datasets[i], err = runSeed(generator, config, temp, firstSeed+i)
		if err != nil {
			return fmt.Errorf("seed %d: %w", firstSeed+i, err)
		}
		fmt.Fprintf(os.Stderr, "Completed seed %d (%d/%d)\n", firstSeed+i, i+1, seedCount)
	}

	frameResults, subsets, err := evaluate(datasets)
	if err != nil {
		return err
	}
	if err := writeFrameCSV(filepath.Join(output, "frame_results.csv"), frameResults); err != nil {
		return err
	}
	if err := writeFrameSweepCSV(filepath.Join(output, "frame_sweep.csv"), frameResults); err != nil {
		return err
	}
	if err := writeSubsetCSV(filepath.Join(output, "subset_results.csv"), subsets, frameResults); err != nil {
		return err
	}
	return writeReport(filepath.Join(output, "experiment_report.md"), config, frameResults, subsets)
}

func main() {
	config := flag.String("config", filepath.Join("v2", "experiments", "scenario2_timeframe", "config.textproto"), "Scenario 2 textproto path, relative to the anomaly-detection module root.")
	output := flag.String("output", filepath.Join("v2", "experiments", "scenario2_tuple", "results"), "Directory for the frame CSV, subset CSV, and report.")
	flag.Parse()
	if err := run(*config, *output); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
