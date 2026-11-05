package main

import (
	"context"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"image/color"
	"log"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	clientpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/client/proto"
	predictionpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/prediction_service/proto"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
	"gonum.org/v1/plot/vg/vgimg"
	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/encoding/prototext"
	"google.golang.org/protobuf/proto"
)

// Metrics are recomputed from counts; undefined ratios remain absent, not zero.
type sweepMetrics struct {
	Precision, Recall, Accuracy, FPR, F1, F2, SegmentRecall *float64
	TP, FP, TN, FN, CompleteWindows, TrailingActions        int64
	Threshold                                               float64
}
type sweepRow struct {
	Parameter string
	Value     float64
	Repeat    int
	Directory string
	Metrics   sweepMetrics
}

func ratio(a, b int64) *float64 {
	if b == 0 {
		return nil
	}
	v := float64(a) / float64(b)
	return &v
}
func metricsFromReport(r *clientpb.WindowEvaluationReport, s *predictionpb.WindowScoreReport) sweepMetrics {
	tp, fp, tn, fn := r.GetTruePositives(), r.GetFalsePositives(), r.GetTrueNegatives(), r.GetFalseNegatives()
	return sweepMetrics{
		Precision: ratio(tp, tp+fp), Recall: ratio(tp, tp+fn), Accuracy: ratio(tp+tn, tp+fp+tn+fn),
		FPR: ratio(fp, fp+tn), F1: ratio(2*tp, 2*tp+fp+fn), F2: ratio(5*tp, 5*tp+4*fn+fp),
		SegmentRecall: ratio(r.GetDetectedSegments(), r.GetScoredSegments()), TP: tp, FP: fp, TN: tn, FN: fn,
		CompleteWindows: s.GetCompleteWindows(), TrailingActions: s.GetTrailingActions(), Threshold: s.GetThreshold(),
	}
}

func sweepValues(name string, points int) ([]float64, error) {
	var lo, hi float64
	logarithmic := true
	switch name {
	case "window_length":
		lo, hi = 30, 3000
	case "training_actions":
		lo, hi = 50000, 2000000
	case "calibration_actions":
		lo, hi = 20000, 1000000
	case "calibration_percentile":
		lo, hi = .9, .9999
		logarithmic = false
	case "standard_deviation_floor_count":
		lo, hi = .1, 10
	case "max_n":
		return []float64{1, 2, 3}, nil
	default:
		return nil, fmt.Errorf("unknown sweep parameter %q", name)
	}
	values := make([]float64, points)
	for i := range values {
		t := float64(i) / float64(points-1)
		v := lo + (hi-lo)*t
		if logarithmic {
			v = math.Exp(math.Log(lo) + (math.Log(hi)-math.Log(lo))*t)
		}
		if name == "window_length" || strings.HasSuffix(name, "_actions") {
			v = math.Round(v)
		}
		if i > 0 && v <= values[i-1] {
			return nil, fmt.Errorf("too many points for %s", name)
		}
		values[i] = v
	}
	// Include the control setting in every grid so effect sizes have an anchor.
	control := map[string]float64{"window_length": 300, "training_actions": 600000, "calibration_actions": 200000, "calibration_percentile": .99, "standard_deviation_floor_count": 1}[name]
	nearest := 1
	for i := 1; i < len(values)-1; i++ {
		if math.Abs(values[i]-control) < math.Abs(values[nearest]-control) {
			nearest = i
		}
	}
	values[nearest] = control
	sort.Float64s(values)
	return values, nil
}

// Preserve the supplied anomaly profiles and their relative positions/durations.
// The million-record baseline is action-based, not a simulated ten-week duration.
func sweepBaseline(base *clientpb.MarkovExperimentConfig) *clientpb.MarkovExperimentConfig {
	c := proto.Clone(base).(*clientpb.MarkovExperimentConfig)
	old := c.GetValidationActions()
	c.SetTrainingActions(600000)
	c.SetCalibrationActions(200000)
	c.SetValidationActions(200000)
	c.GetTrainingConfig().SetMaxN(3)
	c.GetTrainingConfig().SetWindowLength(300)
	c.SetCalibrationPercentile(.99)
	c.GetScoringConfig().SetStandardDeviationFloorCount(1)
	for _, s := range c.GetInjections() {
		start := s.GetStartAction() * 200000 / old
		end := (s.GetStartAction() + s.GetNumActions()) * 200000 / old
		s.SetStartAction(start)
		s.SetNumActions(end - start)
	}
	return c
}
func setSweepValue(c *clientpb.MarkovExperimentConfig, name string, value float64) {
	switch name {
	case "window_length":
		c.GetTrainingConfig().SetWindowLength(int32(value))
	case "max_n":
		c.GetTrainingConfig().SetMaxN(int32(value))
	case "training_actions":
		c.SetTrainingActions(int64(value))
	case "calibration_actions":
		c.SetCalibrationActions(int64(value))
	case "calibration_percentile":
		c.SetCalibrationPercentile(value)
	case "standard_deviation_floor_count":
		c.GetScoringConfig().SetStandardDeviationFloorCount(value)
	}
}

func readTrial(dir string) (sweepMetrics, error) {
	b, err := os.ReadFile(filepath.Join(dir, "validation_evaluation.json"))
	if err != nil {
		return sweepMetrics{}, err
	}
	r := &clientpb.WindowEvaluationReport{}
	if err = protojson.Unmarshal(b, r); err != nil {
		return sweepMetrics{}, err
	}
	b, err = os.ReadFile(filepath.Join(dir, "validation_scores.json"))
	if err != nil {
		return sweepMetrics{}, err
	}
	s := &predictionpb.WindowScoreReport{}
	if err = protojson.Unmarshal(b, s); err != nil {
		return sweepMetrics{}, err
	}
	if r.GetTruePositives()+r.GetFalsePositives()+r.GetTrueNegatives()+r.GetFalseNegatives() != s.GetCompleteWindows() {
		return sweepMetrics{}, fmt.Errorf("inconsistent window counts in %s", dir)
	}
	return metricsFromReport(r, s), nil
}

func runSweep(o options, parameters string, points, repeats int, fprLimit float64, timeout time.Duration) error {
	if points < 20 || repeats < 1 || math.IsNaN(fprLimit) || fprLimit < 0 || fprLimit > 1 || timeout <= 0 {
		return fmt.Errorf("require points>=20, repeats>=1, FPR limit in [0,1], and positive timeout")
	}
	if o.modelPath != "" || o.validationLogPath != "" {
		return fmt.Errorf("sweeps generate independent datasets; omit model_path and validation_log_path")
	}
	names := strings.Split(parameters, ",")
	if parameters == "all" {
		names = []string{"window_length", "training_actions", "calibration_actions", "calibration_percentile", "standard_deviation_floor_count", "max_n"}
	}
	seen := map[string]bool{}
	grids := map[string][]float64{}
	for _, n := range names {
		if seen[n] {
			return fmt.Errorf("duplicate parameter %s", n)
		}
		seen[n] = true
		values, err := sweepValues(n, points)
		if err != nil {
			return err
		}
		if n == "window_length" && (o.sweepWindowMin != 0 || o.sweepWindowMax != 0) {
			values, err = linearWindowGrid(o.sweepWindowMin, o.sweepWindowMax, points)
			if err != nil {
				return err
			}
		}
		grids[n] = values
	}
	base, err := readExperiment(o.experimentPath)
	if err != nil {
		return err
	}
	if len(base.GetInjections()) == 0 && base.GetIndependentAnomalies() == nil {
		return fmt.Errorf("recall sweeps require labeled validation injections")
	}
	base = sweepBaseline(base)
	out, err := filepath.Abs(o.outputDir)
	if err != nil {
		return err
	}
	if err = os.MkdirAll(out, 0755); err != nil {
		return err
	}
	manifest := filepath.Join(out, "sweep.json")
	baselineBytes, err := (proto.MarshalOptions{Deterministic: true}).Marshal(base)
	if err != nil {
		return err
	}
	graphBytes, err := os.ReadFile(base.GetGeneratorConfigPath())
	if err != nil {
		return err
	}
	spec := struct {
		Experiment                string
		Parameters                []string
		Points, Repeats           int
		FPRLimit                  float64
		Baseline, GeneratorConfig []byte
		Grids                     map[string][]float64
	}{o.experimentPath, names, points, repeats, fprLimit, baselineBytes, graphBytes, grids}
	specBytes, err := json.MarshalIndent(spec, "", "  ")
	if err != nil {
		return err
	}
	if previous, e := os.ReadFile(manifest); e == nil {
		if string(previous) != string(specBytes) {
			return fmt.Errorf("output contains a different sweep; choose another output_dir")
		}
	} else if !os.IsNotExist(e) {
		return e
	} else if err = os.WriteFile(manifest, specBytes, 0644); err != nil {
		return err
	}

	workers := o.sweepWorkers
	if workers == 0 {
		workers = 1
	}
	if workers < 1 || workers > 8 {
		return fmt.Errorf("sweep_workers must be in [1,8]")
	}
	tasks := make(chan sweepTask)
	type result struct {
		row sweepRow
		err error
	}
	results := make(chan result, workers)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for task := range tasks {
				if ctx.Err() != nil {
					return
				}
				row, err := executeSweepTrial(ctx, o, base, out, task, timeout)
				results <- result{row, err}
				if err != nil {
					return
				}
			}
		}()
	}
	go func() {
		defer close(tasks)
		for _, name := range names {
			for index, value := range grids[name] {
				for repeat := 0; repeat < repeats; repeat++ {
					select {
					case tasks <- sweepTask{name, value, index, repeat}:
					case <-ctx.Done():
						return
					}
				}
			}
		}
	}()
	go func() { wg.Wait(); close(results) }()
	var rows []sweepRow
	var firstErr error
	for result := range results {
		if firstErr != nil {
			continue
		}
		if result.err != nil {
			firstErr = result.err
			cancel()
			continue
		}
		rows = append(rows, result.row)
		sort.Slice(rows, func(i, j int) bool {
			if rows[i].Parameter != rows[j].Parameter {
				return rows[i].Parameter < rows[j].Parameter
			}
			if rows[i].Value != rows[j].Value {
				return rows[i].Value < rows[j].Value
			}
			return rows[i].Repeat < rows[j].Repeat
		})
		if err = writeSweepResults(out, rows, fprLimit); err != nil {
			firstErr = err
			cancel()
		}
	}
	return firstErr
}

type sweepTask struct {
	name          string
	value         float64
	index, repeat int
}

func executeSweepTrial(ctx context.Context, o options, base *clientpb.MarkovExperimentConfig, out string, task sweepTask, timeout time.Duration) (sweepRow, error) {
	dir := filepath.Join(out, task.name, fmt.Sprintf("point-%02d-repeat-%02d", task.index, task.repeat))
	c := proto.Clone(base).(*clientpb.MarkovExperimentConfig)
	setSweepValue(c, task.name, task.value)
	c.SetTrainingSeed(base.GetTrainingSeed() + int64(task.repeat)*1000)
	c.SetCalibrationSeed(base.GetCalibrationSeed() + int64(task.repeat)*1000)
	c.SetValidationSeed(base.GetValidationSeed() + int64(task.repeat)*1000)
	saved, e := os.ReadFile(filepath.Join(dir, "experiment.textproto"))
	if e == nil {
		previous := &clientpb.MarkovExperimentConfig{}
		if prototext.Unmarshal(saved, previous) != nil || !proto.Equal(previous, c) {
			return sweepRow{}, fmt.Errorf("config changed for %s; use a fresh output directory", dir)
		}
	}
	m, e := readTrial(dir)
	if e != nil {
		if err := os.MkdirAll(dir, 0755); err != nil {
			return sweepRow{}, err
		}
		configPath := filepath.Join(dir, "input.textproto")
		if err := os.WriteFile(configPath, []byte(prototext.Format(c)), 0644); err != nil {
			return sweepRow{}, err
		}
		trial := o
		trial.experimentPath = configPath
		trial.outputDir = dir
		log.Printf("SWEEP %s=%g repeat=%d (%d total actions)", task.name, task.value, task.repeat, c.GetTrainingActions()+c.GetCalibrationActions()+c.GetValidationActions())
		trialCtx, cancel := context.WithTimeout(ctx, timeout)
		err := run(trialCtx, trial)
		cancel()
		if err != nil {
			return sweepRow{}, fmt.Errorf("%s=%g repeat=%d: %w", task.name, task.value, task.repeat, err)
		}
		m, err = readTrial(dir)
		if err != nil {
			return sweepRow{}, err
		}
	} else {
		log.Printf("Resuming completed %s=%g repeat=%d", task.name, task.value, task.repeat)
	}
	return sweepRow{task.name, task.value, task.repeat, dir, m}, nil
}

func metricValues(m sweepMetrics) []*float64 {
	return []*float64{m.Precision, m.Recall, m.Accuracy, m.FPR, m.F1, m.F2, m.SegmentRecall}
}

var metricNames = []string{"precision", "recall", "accuracy", "false_positive_rate", "f1", "f2", "segment_recall"}

func number(v *float64) string {
	if v == nil {
		return ""
	}
	return strconv.FormatFloat(*v, 'g', 10, 64)
}
func writeSweepResults(out string, rows []sweepRow, fprLimit float64) error {
	f, err := os.Create(filepath.Join(out, "results.csv"))
	if err != nil {
		return err
	}
	w := csv.NewWriter(f)
	header := []string{"parameter", "value", "repeat", "directory", "tp", "fp", "tn", "fn", "complete_windows", "trailing_actions", "threshold"}
	if err = w.Write(append(header, metricNames...)); err != nil {
		f.Close()
		return err
	}
	for _, r := range rows {
		m := r.Metrics
		record := []string{r.Parameter, fmt.Sprint(r.Value), fmt.Sprint(r.Repeat), r.Directory, fmt.Sprint(m.TP), fmt.Sprint(m.FP), fmt.Sprint(m.TN), fmt.Sprint(m.FN), fmt.Sprint(m.CompleteWindows), fmt.Sprint(m.TrailingActions), fmt.Sprint(m.Threshold)}
		for _, v := range metricValues(m) {
			record = append(record, number(v))
		}
		if err = w.Write(record); err != nil {
			f.Close()
			return err
		}
	}
	w.Flush()
	err = w.Error()
	closeErr := f.Close()
	if err != nil {
		return err
	}
	if closeErr != nil {
		return closeErr
	}
	return plotSweep(out, rows, fprLimit)
}

type aggregatePoint struct {
	Value     float64
	Means, SD []float64
	N         []int
}

func aggregate(rows []sweepRow) map[string][]aggregatePoint {
	grouped := map[string]map[float64][]sweepRow{}
	for _, r := range rows {
		if grouped[r.Parameter] == nil {
			grouped[r.Parameter] = map[float64][]sweepRow{}
		}
		grouped[r.Parameter][r.Value] = append(grouped[r.Parameter][r.Value], r)
	}
	result := map[string][]aggregatePoint{}
	for name, values := range grouped {
		for value, trials := range values {
			p := aggregatePoint{Value: value, Means: make([]float64, 7), SD: make([]float64, 7), N: make([]int, 7)}
			for _, r := range trials {
				for j, v := range metricValues(r.Metrics) {
					if v != nil {
						p.Means[j] += *v
						p.N[j]++
					}
				}
			}
			for j := range p.Means {
				if p.N[j] > 0 {
					p.Means[j] /= float64(p.N[j])
				}
			}
			for _, r := range trials {
				for j, v := range metricValues(r.Metrics) {
					if v != nil {
						p.SD[j] += math.Pow(*v-p.Means[j], 2)
					}
				}
			}
			for j := range p.SD {
				if p.N[j] > 1 {
					p.SD[j] = math.Sqrt(p.SD[j] / float64(p.N[j]-1))
				} else {
					p.SD[j] = 0
				}
			}
			result[name] = append(result[name], p)
		}
		sort.Slice(result[name], func(i, j int) bool { return result[name][i].Value < result[name][j].Value })
	}
	return result
}
func plotSweep(out string, rows []sweepRow, fprLimit float64) error {
	grouped := aggregate(rows)
	summary, err := os.Create(filepath.Join(out, "summary.csv"))
	if err != nil {
		return err
	}
	w := csv.NewWriter(summary)
	if err = w.Write([]string{"parameter", "value", "metric", "mean", "sample_sd", "defined_repeats"}); err != nil {
		summary.Close()
		return err
	}
	var report strings.Builder
	fmt.Fprintf(&report, "Recall-first candidates with mean FPR <= %.3f. Exploratory results, not held-out optimality.\nError bars: +/- one sample standard deviation across seeds (not confidence intervals).\n", fprLimit)
	names := make([]string, 0, len(grouped))
	for name := range grouped {
		names = append(names, name)
	}
	sort.Strings(names)
	colors := []color.Color{color.RGBA{40, 90, 200, 255}, color.RGBA{220, 65, 45, 255}, color.RGBA{30, 150, 85, 255}, color.RGBA{150, 70, 170, 255}, color.RGBA{180, 120, 0, 255}}
	for _, name := range names {
		points := grouped[name]
		var best *aggregatePoint
		for i := range points {
			p := &points[i]
			for j, metric := range metricNames {
				if p.N[j] > 0 {
					if err = w.Write([]string{name, fmt.Sprint(p.Value), metric, fmt.Sprint(p.Means[j]), fmt.Sprint(p.SD[j]), fmt.Sprint(p.N[j])}); err != nil {
						summary.Close()
						return err
					}
				}
			}
			if p.N[1] > 0 && p.N[3] > 0 && p.Means[3] <= fprLimit && (best == nil || p.Means[1] > best.Means[1] || (p.Means[1] == best.Means[1] && p.Means[3] < best.Means[3])) {
				best = p
			}
		}
		low, high := math.Inf(1), math.Inf(-1)
		for _, point := range points {
			if point.N[1] > 0 {
				low = math.Min(low, point.Means[1])
				high = math.Max(high, point.Means[1])
			}
		}
		if !math.IsInf(low, 1) {
			fmt.Fprintf(&report, "%s: %d settings; recall range %.4f–%.4f (%.2f percentage points across the tested range)\n", name, len(points), low, high, 100*(high-low))
		}
		if best == nil {
			fmt.Fprintf(&report, "%s: no setting meets the false-alarm limit\n", name)
		} else {
			fmt.Fprintf(&report, "%s=%g: recall %s, precision %s, accuracy %s, FPR %s, segment recall %s\n", name, best.Value, aggregateMetric(*best, 1), aggregateMetric(*best, 0), aggregateMetric(*best, 2), aggregateMetric(*best, 3), aggregateMetric(*best, 6))
		}
		p := plot.New()
		p.Title.Text = name + " (single seed; exploratory)"
		if points[0].N[1] > 1 {
			p.Title.Text = name + " (mean; error bars = seed SD)"
		}
		p.X.Label.Text = name
		p.Y.Label.Text = "Fraction"
		p.Y.Min = 0
		p.Y.Max = 1
		p.Legend.Top = true
		p.Legend.YOffs = -vg.Points(40)
		if name != "calibration_percentile" && name != "max_n" && name != "window_length" {
			p.X.Scale = plot.LogScale{}
			p.X.Tick.Marker = plot.LogTicks{}
		}
		for k, j := range []int{0, 1, 2, 3, 6} {
			xy := plotter.XYs{}
			errs := plotter.YErrors{}
			for _, point := range points {
				if point.N[j] > 0 {
					xy = append(xy, plotter.XY{X: point.Value, Y: point.Means[j]})
					errs = append(errs, struct{ Low, High float64 }{math.Min(point.SD[j], point.Means[j]), math.Min(point.SD[j], 1-point.Means[j])})
				}
			}
			if len(xy) == 0 {
				continue
			}
			line, dots, e := plotter.NewLinePoints(xy)
			if e != nil {
				return e
			}
			line.Color = colors[k]
			dots.Color = colors[k]
			dots.Radius = vg.Points(2)
			bars, e := plotter.NewYErrorBars(struct {
				plotter.XYs
				plotter.YErrors
			}{xy, errs})
			if e != nil {
				return e
			}
			bars.LineStyle.Color = colors[k]
			p.Add(line, dots, bars)
			p.Legend.Add(metricNames[j], line)
		}
		// A single fractional x-value otherwise gets expanded by +/-1 by
		// gonum, crossing zero and making logarithmic ticks panic.
		if name != "calibration_percentile" && name != "max_n" && name != "window_length" {
			p.X.Min = points[0].Value * 0.9
			p.X.Max = points[len(points)-1].Value * 1.1
		}
		for _, ext := range []string{"png", "svg"} {
			if err = p.Save(10*vg.Inch, 5*vg.Inch, filepath.Join(out, name+"."+ext)); err != nil {
				return err
			}
		}
	}
	w.Flush()
	err = w.Error()
	closeErr := summary.Close()
	if err != nil {
		return err
	}
	if closeErr != nil {
		return closeErr
	}
	if points, ok := grouped["window_length"]; ok {
		if err = plotWindowStudy(out, points); err != nil {
			return err
		}
	}
	return os.WriteFile(filepath.Join(out, "candidates.txt"), []byte(report.String()), 0644)
}

func aggregateMetric(p aggregatePoint, index int) string {
	if p.N[index] == 0 {
		return "undefined"
	}
	return fmt.Sprintf("%.4f", p.Means[index])
}

func linearWindowGrid(low, high, points int) ([]float64, error) {
	if low < 3 || high <= low || points < 2 || high-low < points-1 {
		return nil, fmt.Errorf("window range requires min>=3, max>min, and enough distinct integer settings")
	}
	values := make([]float64, points)
	for i := range values {
		values[i] = math.Round(float64(low) + float64(high-low)*float64(i)/float64(points-1))
	}
	return values, nil
}

// Separate false alarms from detection rates so small FPR changes stay visible.
func plotWindowStudy(out string, points []aggregatePoint) error {
	plots := [][]*plot.Plot{{plot.New()}, {plot.New()}}
	colors := []color.Color{color.RGBA{37, 99, 235, 255}, color.RGBA{234, 88, 12, 255}, color.RGBA{22, 130, 74, 255}, color.RGBA{124, 58, 237, 255}}
	for row, panel := range plots {
		p := panel[0]
		p.X.Label.Text = "Window length (actions)"
		p.Y.Label.Text = "Fraction of windows"
		p.X.Min = points[0].Value
		p.X.Max = points[len(points)-1].Value
		if p.X.Min == p.X.Max {
			p.X.Min--
			p.X.Max++
		}
		p.Y.Min = 0
		p.Y.Max = 1
		p.Y.Min = .7
		p.Title.Text = "Detection quality — single seed, exploratory (zoomed axis)"
		if points[0].N[1] > 1 {
			p.Title.Text = "Detection quality — means; bars: ±1 seed SD (zoomed axis)"
		}
		indices := []int{0, 1, 2}
		if row == 1 {
			p.Y.Min = 0
			p.Title.Text = "False alarms among normal windows"
			if points[0].N[3] > 1 {
				p.Title.Text += " — means; bars: ±1 seed SD"
			}
			indices = []int{3}
			p.Y.Max = .02
		}
		p.Add(plotter.NewGrid())
		p.Legend.Left = true
		for k, j := range indices {
			xy := plotter.XYs{}
			errs := plotter.YErrors{}
			for _, point := range points {
				if point.N[j] > 0 {
					xy = append(xy, plotter.XY{X: point.Value, Y: point.Means[j]})
					errs = append(errs, struct{ Low, High float64 }{math.Min(point.SD[j], point.Means[j]), math.Min(point.SD[j], 1-point.Means[j])})
					if row == 1 {
						p.Y.Max = math.Max(p.Y.Max, math.Min(1, point.Means[j]+point.SD[j])*1.2)
					}
				}
			}
			if len(xy) == 0 {
				continue
			}
			line, dots, err := plotter.NewLinePoints(xy)
			if err != nil {
				return err
			}
			c := colors[k]
			if row == 1 {
				c = colors[3]
			}
			line.Color = c
			line.Width = vg.Points(2)
			dots.Color = c
			dots.Radius = vg.Points(3)
			p.Add(line, dots)
			bars, err := plotter.NewYErrorBars(struct {
				plotter.XYs
				plotter.YErrors
			}{xy, errs})
			if err != nil {
				return err
			}
			bars.LineStyle.Color = c
			p.Add(bars)
			p.Legend.Add(metricNames[j], line)
		}
	}
	canvas := vgimg.New(11*vg.Inch, 7*vg.Inch)
	dc := draw.New(canvas)
	tiles := draw.Tiles{Rows: 2, Cols: 1, PadX: vg.Points(12), PadY: vg.Points(20), PadTop: vg.Points(12), PadBottom: vg.Points(8), PadLeft: vg.Points(8), PadRight: vg.Points(12)}
	aligned := plot.Align(plots, tiles, dc)
	for row := range plots {
		plots[row][0].Draw(aligned[row][0])
	}
	f, err := os.Create(filepath.Join(out, "window_length_detail.png"))
	if err != nil {
		return err
	}
	png := vgimg.PngCanvas{Canvas: canvas}
	_, err = png.WriteTo(f)
	closeErr := f.Close()
	if err != nil {
		return err
	}
	return closeErr
}
