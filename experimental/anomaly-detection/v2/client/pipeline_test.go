package main

import (
	"bytes"
	"context"
	"fmt"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/encoding/prototext"
	"google.golang.org/protobuf/proto"

	genpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v1/generator/proto"
	clientpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/client/proto"
	modelpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common/proto"
	predictionpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/prediction_service/proto"
)

// TestMarkovPipeline exercises the actual generator process, training RPC,
// calibration RPC, mixed generation, validation RPC and ground-truth evaluation.
// Workload cases load the profiles under v1/client directly. A deterministic
// graph is retained as an analytically predictable correctness fixture.
func TestMarkovPipeline(t *testing.T) {
	if testing.Short() {
		t.Skip("end-to-end test builds and starts local services")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	dir := t.TempDir()
	binDir := filepath.Join(dir, "bin")
	moduleRoot, err := filepath.Abs("../..")
	if err != nil {
		t.Fatal(err)
	}
	args := []string{"build"}
	if os.Getenv("MARKOV_E2E_RACE") == "1" {
		args = append(args, "-race")
	}
	args = append(args, "-o", binDir+string(os.PathSeparator), "./v1/generator/src", "./v2/training_service", "./v2/prediction_service", "./v2/client")
	build := exec.CommandContext(ctx, "go", args...)
	build.Dir = moduleRoot
	if output, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build services: %v\n%s", err, output)
	}
	models := filepath.Join(dir, "models")
	if err := os.MkdirAll(models, 0755); err != nil {
		t.Fatal(err)
	}
	trainingAddr := startPipelineServer(t, ctx, filepath.Join(binDir, "training_service"), "--model_dir="+models)
	predictionAddr := startPipelineServer(t, ctx, filepath.Join(binDir, "prediction_service"), "--models_dir="+models)
	scenarios := []string{"deterministic_known_answer"}
	for _, profile := range []string{"standard_user", "maintenance_user", "service_usage"} {
		for _, seed := range []int{42, 142, 242} {
			scenarios = append(scenarios, fmt.Sprintf("%s_seed_%d", profile, seed))
		}
	}
	for _, scenario := range scenarios {
		t.Run(scenario, func(t *testing.T) {
			config := pipelineExperiment(t, moduleRoot, scenario)
			caseDir := t.TempDir()
			if artifacts := os.Getenv("MARKOV_E2E_ARTIFACT_DIR"); artifacts != "" {
				artifacts, err = filepath.Abs(artifacts)
				if err != nil {
					t.Fatal(err)
				}
				if err := os.MkdirAll(artifacts, 0755); err != nil {
					t.Fatal(err)
				}
				caseDir, err = os.MkdirTemp(artifacts, scenario+"-")
				if err != nil {
					t.Fatal(err)
				}
				t.Logf("Retained test artifacts: %s", caseDir)
			}
			graphPath := config.GetGeneratorConfigPath()
			if scenario == "deterministic_known_answer" {
				graphPath = filepath.Join(caseDir, "normal.textproto")
				graph := `markov_sampling {
 nodes { id:"a" template:{normalized_query_template:"A"}
 completion_time:{fixed:{nanos:1000000}}
 edges:{destination_id:"a" weight:0} edges:{destination_id:"b" weight:1} }
 nodes { id:"b" template:{normalized_query_template:"B"}
 completion_time:{fixed:{nanos:1000000}} edges:{destination_id:"a" weight:1} }
 start:{node_id:"a"}
}`
				if err := os.WriteFile(graphPath, []byte(graph), 0644); err != nil {
					t.Fatal(err)
				}
				config.SetGeneratorConfigPath(graphPath)
			}
			experimentPath := filepath.Join(caseDir, "input_experiment.textproto")
			if err := os.WriteFile(experimentPath, []byte(prototext.Format(config)), 0644); err != nil {
				t.Fatal(err)
			}
			out := filepath.Join(caseDir, "experiment")
			client := filepath.Join(binDir, "client")
			runPipelineCLI(t, ctx, client, "--experiment_config="+experimentPath, "--generator_bin="+filepath.Join(binDir, "src"), "--server_addr="+trainingAddr, "--prediction_server_addr="+predictionAddr, "--output_dir="+out)
			baseline := &modelpb.Model{}
			readPipelineProto(t, filepath.Join(out, "histogram.bin"), baseline, false)
			calibrated := &modelpb.Model{}
			readPipelineProto(t, filepath.Join(out, "calibrated_histogram.bin"), calibrated, false)
			if baseline.GetNgramHistogram().HasCalibration() || !calibrated.GetNgramHistogram().HasCalibration() {
				t.Fatal("training/calibration stages not separated")
			}
			clone := proto.Clone(calibrated).(*modelpb.Model)
			clone.GetNgramHistogram().ClearCalibration()
			if !proto.Equal(baseline, clone) {
				t.Fatal("calibration changed learned baseline")
			}
			scores := &predictionpb.WindowScoreReport{}
			readPipelineProto(t, filepath.Join(out, "validation_scores.json"), scores, true)
			evaluation := &clientpb.WindowEvaluationReport{}
			readPipelineProto(t, filepath.Join(out, "validation_evaluation.json"), evaluation, true)
			verifyPipelineReference(t, out, config, baseline, calibrated, scores, evaluation)
			if scenario == "deterministic_known_answer" && (evaluation.GetTruePositives() != 4 || evaluation.GetTrueNegatives() != 16 || evaluation.GetFalsePositives() != 0 || evaluation.GetFalseNegatives() != 0 || evaluation.GetAnomalousActions() != 103 || evaluation.GetTrailingAnomalousActions() != 3 || evaluation.GetDetectedSegments() != 2 || evaluation.GetScoredSegments() != 2) {
				t.Fatalf("known-answer classification failed: %v", evaluation)
			}
			modelPath := filepath.Join(out, "calibrated_histogram.bin")
			before, err := os.ReadFile(modelPath)
			if err != nil {
				t.Fatal(err)
			}
			validationPath := filepath.Join(out, "validation", "logs.binproto")
			only := filepath.Join(out, "validation-only")
			validationArgs := func(model, logs, output string) []string {
				return []string{"--model_path=" + model, "--validation_log_path=" + logs, "--prediction_server_addr=" + predictionAddr, "--server_addr=localhost:1", "--generator_bin=/unused", "--experiment_config=/unused", "--output_dir=" + output}
			}
			runPipelineCLI(t, ctx, client, validationArgs(modelPath, validationPath, only)...)
			repeated := &predictionpb.WindowScoreReport{}
			readPipelineProto(t, filepath.Join(only, "validation_scores.json"), repeated, true)
			if !proto.Equal(scores, repeated) {
				t.Fatal("validation-only results changed")
			}
			// Repeat through the actual CLI/RPC after removing every truth label.
			// Evaluation should change; predictions and scores must not.
			stripped := filepath.Join(out, "stripped.binproto")
			stripPipelineLabels(t, validationPath, stripped)
			strippedOut := filepath.Join(out, "label-isolation")
			runPipelineCLI(t, ctx, client, validationArgs(modelPath, stripped, strippedOut)...)
			withoutLabels := &predictionpb.WindowScoreReport{}
			readPipelineProto(t, filepath.Join(strippedOut, "validation_scores.json"), withoutLabels, true)
			if !proto.Equal(scores, withoutLabels) {
				t.Fatal("ground truth affected predictions")
			}
			// Negative control: the same normal graph without injections.
			normalDir := filepath.Join(out, "normal-control")
			generate := exec.CommandContext(ctx, filepath.Join(binDir, "src"), "--config_file="+filepath.Join(out, "generator_config.textproto"), "--output_dir="+normalDir, fmt.Sprintf("--num_logs=%d", config.GetValidationActions()), fmt.Sprintf("--seed=%d", config.GetValidationSeed()))
			if output, err := generate.CombinedOutput(); err != nil {
				t.Fatalf("normal control: %v\n%s", err, output)
			}
			runPipelineCLI(t, ctx, client, validationArgs(modelPath, filepath.Join(normalDir, "logs.binproto"), filepath.Join(normalDir, "scored"))...)
			falsePositives, normalWindows := verifyPipelineNormalControl(t, out, normalDir, config)
			// A baseline without a calibrated threshold must not classify windows.
			uncalibrated := exec.CommandContext(ctx, client, validationArgs(filepath.Join(out, "histogram.bin"), validationPath, filepath.Join(out, "uncalibrated-control"))...)
			output, err := uncalibrated.CombinedOutput()
			if err == nil || !bytes.Contains(output, []byte("calibrate")) {
				t.Fatalf("uncalibrated model was not rejected: %v %s", err, output)
			}
			after, err := os.ReadFile(modelPath)
			if err != nil || !bytes.Equal(before, after) {
				t.Fatal("validation modified calibrated model")
			}
			recordPipelineControls(t, out, falsePositives, normalWindows)
			t.Logf("PASS: actual CLI + RPCs; independent counts, window means/variances, standardized scores, quantile and metrics; normal-only and label-isolation controls; immutable model")
		})
	}
}

func pipelineExperiment(t *testing.T, root, scenario string) *clientpb.MarkovExperimentConfig {
	t.Helper()
	config := &clientpb.MarkovExperimentConfig{}
	if scenario == "deterministic_known_answer" {
		text := `generator_config_path:"set-by-test" training_actions:600 training_seed:1
 training_config:{max_n:3 window_length:30}
 calibration_actions:600 calibration_seed:2 calibration_percentile:0.99
 validation_actions:607 validation_seed:3
 anomaly_profiles {profile_id:"repeat-a" transitions {node_id:"a" edges:{destination_id:"a" weight:1} edges:{destination_id:"b" weight:0}}}
 injections {segment_id:"aligned" profile_id:"repeat-a" start_action:120 num_actions:60}
 injections {segment_id:"crossing" profile_id:"repeat-a" start_action:275 num_actions:40}
 injections {segment_id:"unscored-tail" profile_id:"repeat-a" start_action:601 num_actions:3}`
		if err := prototext.Unmarshal([]byte(text), config); err != nil {
			t.Fatal(err)
		}
	} else {
		profile, seedText, ok := strings.Cut(scenario, "_seed_")
		if !ok {
			t.Fatalf("invalid profile scenario %q", scenario)
		}
		seed, err := strconv.ParseInt(seedText, 10, 64)
		if err != nil {
			t.Fatal(err)
		}
		config, err = readExperiment(filepath.Join(root, "v2/client/configs/"+profile+".textproto"))
		if err != nil {
			t.Fatal(err)
		}
		config.SetGeneratorConfigPath(filepath.Join(root, config.GetGeneratorConfigPath()))
		if config.GetGeneratorConfigPath() != filepath.Join(root, "v1/client/markov_"+profile+"_config.textproto") {
			t.Fatal("test must use upstream workload file directly")
		}
		// Reuse the runnable configuration; shrink only dataset sizes and ranges.
		// Normal and anomalous transition probabilities remain file-defined.
		config.SetTrainingActions(3000)
		config.SetCalibrationActions(30000)
		config.SetValidationActions(3307)
		variant := config.GetAnomalyProfiles()[0].GetProfileId()
		var segments []*genpb.MarkovInjection
		for _, spec := range []struct {
			id            string
			start, length int64
		}{{"aligned", 300, 300}, {"crossing", 1450, 450}, {"aligned-again", 2700, 300}, {"unscored-tail", 3301, 3}} {
			segment := &genpb.MarkovInjection{}
			segment.SetSegmentId(spec.id)
			segment.SetProfileId(variant)
			segment.SetStartAction(spec.start)
			segment.SetNumActions(spec.length)
			segments = append(segments, segment)
		}
		config.SetInjections(segments)
		config.SetTrainingSeed(seed)
		config.SetCalibrationSeed(seed + 1)
		config.SetValidationSeed(seed + 2)
	}
	return config
}

func runPipelineCLI(t *testing.T, ctx context.Context, binary string, args ...string) {
	t.Helper()
	command := exec.CommandContext(ctx, binary, args...)
	output, err := command.CombinedOutput()
	if err != nil {
		t.Fatalf("client CLI: %v\n%s", err, output)
	}
	t.Logf("Client CLI:\n%s", output)
}

func readPipelineProto(t *testing.T, path string, message proto.Message, jsonFormat bool) {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if jsonFormat {
		err = protojson.Unmarshal(data, message)
	} else {
		err = proto.Unmarshal(data, message)
	}
	if err != nil {
		t.Fatal(err)
	}
}

func startPipelineServer(t *testing.T, ctx context.Context, binary string, args ...string) string {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	address := listener.Addr().String()
	port := listener.Addr().(*net.TCPAddr).Port
	if err := listener.Close(); err != nil {
		t.Fatal(err)
	}
	logPath := filepath.Join(t.TempDir(), "server.log")
	if artifacts := os.Getenv("MARKOV_E2E_ARTIFACT_DIR"); artifacts != "" {
		artifacts, err = filepath.Abs(artifacts)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.MkdirAll(artifacts, 0755); err != nil {
			t.Fatal(err)
		}
		logDir, err := os.MkdirTemp(artifacts, "server-logs-")
		if err != nil {
			t.Fatal(err)
		}
		logPath = filepath.Join(logDir, filepath.Base(binary)+".log")
		t.Logf("Retained %s log: %s", filepath.Base(binary), logPath)
	}
	logFile, err := os.Create(logPath)
	if err != nil {
		t.Fatal(err)
	}
	cmd := exec.CommandContext(ctx, binary, append([]string{fmt.Sprintf("--port=%d", port)}, args...)...)
	cmd.Stdout = logFile
	cmd.Stderr = logFile
	if err := cmd.Start(); err != nil {
		logFile.Close()
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	t.Cleanup(func() {
		_ = cmd.Process.Kill()
		select {
		case <-done:
		case <-time.After(5 * time.Second):
			t.Error("server did not stop")
		}
		_ = logFile.Close()
		data, _ := os.ReadFile(logPath)
		if bytes.Contains(data, []byte("WARNING: DATA RACE")) {
			t.Errorf("%s child process reported a data race:\n%s", filepath.Base(binary), data)
		}
		if t.Failed() {
			t.Logf("%s log:\n%s", filepath.Base(binary), data)
		}
	})
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		connection, err := net.DialTimeout("tcp", address, 100*time.Millisecond)
		if err == nil {
			connection.Close()
			return address
		}
		select {
		case err := <-done:
			done <- err
			t.Fatalf("server stopped before readiness: %v", err)
		case <-ctx.Done():
			t.Fatal(ctx.Err())
		case <-time.After(10 * time.Millisecond):
		}
	}
	t.Fatal("server did not become ready")
	return ""
}
