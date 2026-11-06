// Command chi_squared_benchmark benchmarks the existing rolling ChiSquaredModel
// through its file Predict RPC. It performs no training or calibration.
package main

import (
	"context"
	"crypto/sha256"
	"encoding/csv"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"image/color"
	"io"
	"log"
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"

	modelpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common/proto"
	predictionpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/prediction_service/proto"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/local"
	"google.golang.org/protobuf/encoding/prototext"
	"google.golang.org/protobuf/proto"
)

type options struct {
	Input        string        `json:"input"`
	Output       string        `json:"output"`
	Address      string        `json:"-"`
	Sweep        string        `json:"sweep"`
	M            int           `json:"historical_m"`
	N            int           `json:"fixed_recent_n"`
	Threshold    float64       `json:"fixed_threshold"`
	Points       int           `json:"points"`
	WindowMin    int           `json:"window_min"`
	WindowMax    int           `json:"window_max"`
	ThresholdMin float64       `json:"threshold_min"`
	ThresholdMax float64       `json:"threshold_max"`
	Warmup       int           `json:"warmup_actions"`
	Timeout      time.Duration `json:"timeout_ns"`
}
type trial struct {
	Parameter string  `json:"parameter"`
	M         int     `json:"historical_m"`
	N         int     `json:"recent_n"`
	Threshold float64 `json:"threshold"`
}
type result struct {
	Trial      trial      `json:"trial"`
	Evaluation evaluation `json:"evaluation"`
	Directory  string     `json:"directory"`
}

func main() {
	var o options
	flag.StringVar(&o.Input, "validation_log_path", "", "Existing length-delimited AuditLogEntry stream with ground-truth labels (required).")
	flag.StringVar(&o.Output, "output_dir", "output/chi-squared-benchmark", "Fresh output directory.")
	flag.StringVar(&o.Address, "prediction_server_addr", "localhost:50052", "Existing prediction service.")
	flag.StringVar(&o.Sweep, "sweep", "both", "window, threshold, or both (isolated sweeps).")
	flag.IntVar(&o.M, "historical_m", 1000, "Fixed historical rolling-window length.")
	flag.IntVar(&o.N, "recent_n", 100, "Recent-window length held fixed during threshold sweep.")
	flag.Float64Var(&o.Threshold, "threshold", 10, "Raw statistic threshold held fixed during window sweep.")
	flag.IntVar(&o.Points, "points", 20, "Settings per sweep (>=2).")
	flag.IntVar(&o.WindowMin, "window_min", 20, "Minimum recent-window length N.")
	flag.IntVar(&o.WindowMax, "window_max", 500, "Maximum recent-window length N.")
	flag.Float64Var(&o.ThresholdMin, "threshold_min", 1, "Minimum raw statistic threshold.")
	flag.Float64Var(&o.ThresholdMax, "threshold_max", 50, "Maximum raw statistic threshold.")
	flag.IntVar(&o.Warmup, "warmup_actions", -1, "Initial decisions excluded from evaluation; -1 means M.")
	flag.DurationVar(&o.Timeout, "timeout", 5*time.Minute, "Deadline per prediction/evaluation trial.")
	flag.Parse()
	if err := run(context.Background(), o); err != nil {
		log.Fatal(err)
	}
}
func finitePositive(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) && v > 0 }
func trialGrid(o options) ([]trial, error) {
	if o.M <= o.N || o.N <= 0 || !finitePositive(o.Threshold) || o.Points < 2 || o.Timeout <= 0 {
		return nil, fmt.Errorf("require M>N>0, positive finite threshold, points>=2, and positive timeout")
	}
	if o.Warmup < 0 || o.Warmup < o.N {
		return nil, fmt.Errorf("warmup must be at least fixed N")
	}
	var trials []trial
	if o.Sweep == "window" || o.Sweep == "both" {
		if o.WindowMin < 1 || o.WindowMax >= o.M || o.WindowMax-o.WindowMin < o.Points-1 || o.Warmup < o.WindowMax {
			return nil, fmt.Errorf("require 0<window_min<window_max<M, sufficient distinct settings, and warmup>=window_max")
		}
		for i := 0; i < o.Points; i++ {
			n := int(math.Round(float64(o.WindowMin) + float64(o.WindowMax-o.WindowMin)*float64(i)/float64(o.Points-1)))
			trials = append(trials, trial{"window_length", o.M, n, o.Threshold})
		}
	}
	if o.Sweep == "threshold" || o.Sweep == "both" {
		if !finitePositive(o.ThresholdMin) || !finitePositive(o.ThresholdMax) || o.ThresholdMax <= o.ThresholdMin {
			return nil, fmt.Errorf("require 0<threshold_min<threshold_max, both finite")
		}
		for i := 0; i < o.Points; i++ {
			threshold := o.ThresholdMin + (o.ThresholdMax-o.ThresholdMin)*float64(i)/float64(o.Points-1)
			trials = append(trials, trial{"threshold", o.M, o.N, threshold})
		}
	}
	if len(trials) == 0 {
		return nil, fmt.Errorf("sweep must be window, threshold, or both")
	}
	return trials, nil
}
func digest(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err = io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
func writeJSON(path string, v any) error {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(b, '\n'), 0644)
}
func run(ctx context.Context, o options) error {
	if o.Input == "" {
		return fmt.Errorf("validation_log_path is required")
	}
	var err error
	o.Input, err = filepath.Abs(o.Input)
	if err != nil {
		return err
	}
	o.Output, err = filepath.Abs(o.Output)
	if err != nil {
		return err
	}
	if o.Warmup == -1 {
		o.Warmup = o.M
	}
	trials, err := trialGrid(o)
	if err != nil {
		return err
	}
	hash, err := digest(o.Input)
	if err != nil {
		return err
	}
	if err = os.MkdirAll(o.Output, 0755); err != nil {
		return err
	}
	manifest := struct {
		Options options `json:"options"`
		SHA256  string  `json:"input_sha256"`
		Policy  string  `json:"ground_truth_policy"`
		State   string  `json:"state_policy"`
	}{o, hash, "any_anomalous_action_in_recent_window", "fresh model per full-file request; first warmup_actions decisions excluded"}
	manifestPath := filepath.Join(o.Output, "benchmark.json")
	f, err := os.OpenFile(manifestPath, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0644)
	if err != nil {
		return fmt.Errorf("use a fresh output directory: %w", err)
	}
	b, err := json.MarshalIndent(manifest, "", "  ")
	if err != nil {
		f.Close()
		return err
	}
	_, err = f.Write(b)
	closeErr := f.Close()
	if err != nil {
		return err
	}
	if closeErr != nil {
		return closeErr
	}
	conn, err := grpc.NewClient(o.Address, grpc.WithTransportCredentials(local.NewCredentials()))
	if err != nil {
		return err
	}
	defer conn.Close()
	client := predictionpb.NewPredictionServiceClient(conn)
	var results []result
	for index, t := range trials {
		log.Printf("Trial %d/%d: %s M=%d N=%d threshold=%.6g", index+1, len(trials), t.Parameter, t.M, t.N, t.Threshold)
		dir := filepath.Join(o.Output, t.Parameter, fmt.Sprintf("point-%02d", index))
		if err = os.MkdirAll(dir, 0755); err != nil {
			return err
		}
		model := &modelpb.Model{}
		scenario := &modelpb.ChiSquaredScenario{}
		cfg := &modelpb.ChiSquaredTrainingConfig{}
		cfg.SetM(int64(t.M))
		cfg.SetN(int64(t.N))
		cfg.SetAnomalyThreshold(t.Threshold)
		scenario.SetConfig(cfg)
		model.SetChiSquared(scenario)
		modelPath := filepath.Join(dir, "model.bin")
		data, err := proto.Marshal(model)
		if err != nil {
			return err
		}
		if err = os.WriteFile(modelPath, data, 0644); err != nil {
			return err
		}
		if err = os.WriteFile(filepath.Join(dir, "model.textproto"), []byte(prototext.Format(model)), 0644); err != nil {
			return err
		}
		request := &predictionpb.PredictRequest{}
		request.SetModelLocation(modelPath)
		request.SetPredictLogPath(o.Input)
		if err = os.WriteFile(filepath.Join(dir, "request.textproto"), []byte(prototext.Format(request)), 0644); err != nil {
			return err
		}
		trialCtx, cancel := context.WithTimeout(ctx, o.Timeout)
		response, rpcErr := client.Predict(trialCtx, request)
		if rpcErr != nil {
			cancel()
			return fmt.Errorf("Predict: %w", rpcErr)
		}
		findings := response.GetFindingsPath()
		if findings == "" {
			cancel()
			return fmt.Errorf("Predict did not return a findings file")
		}
		eval, evalErr := evaluate(trialCtx, o.Input, findings, t.N, o.Warmup)
		cancel()
		if evalErr != nil {
			return evalErr
		}
		// Keep the actual server output with the model/config used to produce it.
		if err = copyFile(findings, filepath.Join(dir, "findings.textproto")); err != nil {
			return err
		}
		if err = os.Remove(findings); err != nil {
			return err
		}
		row := result{t, eval, dir}
		if err = writeJSON(filepath.Join(dir, "evaluation.json"), row); err != nil {
			return err
		}
		results = append(results, row)
		if err = writeResults(o.Output, results); err != nil {
			return err
		}
	}
	after, err := digest(o.Input)
	if err != nil {
		return err
	}
	if after != hash {
		return fmt.Errorf("input changed during benchmark; results are invalid")
	}
	if err = plotResults(o.Output, results); err != nil {
		return err
	}
	return writeJSON(filepath.Join(o.Output, "completed.json"), struct {
		Trials      int    `json:"trials"`
		InputSHA256 string `json:"input_sha256"`
		Verified    bool   `json:"input_unchanged"`
	}{len(results), hash, true})
}
func copyFile(source, destination string) error {
	in, err := os.Open(source)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.Create(destination)
	if err != nil {
		return err
	}
	_, err = io.Copy(out, in)
	closeErr := out.Close()
	if err != nil {
		return err
	}
	return closeErr
}
func metric(v *float64) string {
	if v == nil {
		return ""
	}
	return fmt.Sprintf("%.10g", *v)
}
func writeResults(out string, rows []result) error {
	f, err := os.Create(filepath.Join(out, "results.csv"))
	if err != nil {
		return err
	}
	w := csv.NewWriter(f)
	if err = w.Write([]string{"parameter", "historical_m", "recent_n", "threshold", "warmup_actions", "total_actions", "evaluated_decisions", "anomalous_actions", "warmup_anomalous_actions", "tp", "fp", "tn", "fn", "precision", "recall", "false_positive_rate", "directory"}); err != nil {
		f.Close()
		return err
	}
	for _, r := range rows {
		e := r.Evaluation
		if err = w.Write([]string{r.Trial.Parameter, fmt.Sprint(r.Trial.M), fmt.Sprint(r.Trial.N), fmt.Sprint(r.Trial.Threshold), fmt.Sprint(e.WarmupActions), fmt.Sprint(e.TotalActions), fmt.Sprint(e.EvaluatedDecisions), fmt.Sprint(e.AnomalousActions), fmt.Sprint(e.WarmupAnomalousActions), fmt.Sprint(e.TP), fmt.Sprint(e.FP), fmt.Sprint(e.TN), fmt.Sprint(e.FN), metric(e.Precision), metric(e.Recall), metric(e.FalsePositiveRate), r.Directory}); err != nil {
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
	return closeErr
}
func plotResults(out string, rows []result) error {
	for _, parameter := range []string{"window_length", "threshold"} {
		p := plot.New()
		p.Title.Text = "Chi-squared rolling decisions: " + parameter
		p.Y.Label.Text = "Fraction"
		p.Y.Min = 0
		p.Y.Max = 1
		p.X.Label.Text = "Recent-window length N (actions)"
		if parameter == "threshold" {
			p.X.Label.Text = "Raw chi-squared threshold"
		}
		p.Add(plotter.NewGrid())
		p.Legend.Top = true
		p.Legend.Left = true
		count := 0
		for i, name := range []string{"precision", "recall"} {
			xy := plotter.XYs{}
			for _, r := range rows {
				if r.Trial.Parameter != parameter {
					continue
				}
				v := r.Evaluation.Precision
				if i == 1 {
					v = r.Evaluation.Recall
				}
				if v == nil {
					continue
				}
				x := float64(r.Trial.N)
				if parameter == "threshold" {
					x = r.Trial.Threshold
				}
				xy = append(xy, plotter.XY{X: x, Y: *v})
			}
			if len(xy) == 0 {
				continue
			}
			count += len(xy)
			line, dots, err := plotter.NewLinePoints(xy)
			if err != nil {
				return err
			}
			c := color.RGBA{37, 99, 235, 255}
			if i == 1 {
				c = color.RGBA{234, 88, 12, 255}
			}
			line.Color = c
			line.Width = vg.Points(2)
			dots.Color = c
			dots.Radius = vg.Points(3)
			p.Add(line, dots)
			p.Legend.Add(name, line)
		}
		if count == 0 {
			continue
		}
		for _, extension := range strings.Fields("png svg") {
			if err := p.Save(10*vg.Inch, 5*vg.Inch, filepath.Join(out, parameter+"."+extension)); err != nil {
				return err
			}
		}
	}
	return nil
}
