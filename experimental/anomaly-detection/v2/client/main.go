// Command client orchestrates reproducible Markov generation and baseline training.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/local"
	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/encoding/prototext"
	"google.golang.org/protobuf/proto"

	genpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v1/generator/proto"
	clientpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/client/proto"
	modelpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common/proto"
	trainingpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/training_service/proto"
)

type options struct {
	experimentPath, generatorBin, outputDir, serverAddr string
	predictionAddr, modelPath, validationLogPath        string
}

func main() {
	var o options
	flag.StringVar(&o.experimentPath, "experiment_config", "v2/client/configs/standard_user.textproto", "Experiment configuration (.textproto).")
	flag.StringVar(&o.generatorBin, "generator_bin", "./bin/generator", "Generator executable.")
	flag.StringVar(&o.outputDir, "output_dir", "output/experiment", "Directory for training logs, experiment configuration and model exports.")
	flag.StringVar(&o.serverAddr, "server_addr", "localhost:50051", "Training service address.")
	flag.StringVar(&o.predictionAddr, "prediction_server_addr", "localhost:50052", "Prediction service for calibration and validation scoring.")
	flag.StringVar(&o.modelPath, "model_path", "", "Validation-only mode: existing calibrated model; requires --validation_log_path.")
	flag.StringVar(&o.validationLogPath, "validation_log_path", "", "Existing validation logs. Otherwise a separate normal validation stream is generated.")
	flag.Parse()
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	if err := run(ctx, o); err != nil {
		log.Fatal(err)
	}
}

func run(ctx context.Context, o options) error {
	if o.modelPath != "" {
		if o.validationLogPath == "" {
			return fmt.Errorf("--model_path requires --validation_log_path for validation-only mode")
		}
		out, err := filepath.Abs(o.outputDir)
		if err != nil {
			return err
		}
		if err := os.MkdirAll(out, 0755); err != nil {
			return err
		}
		return validateWindows(ctx, o, out, o.modelPath, o.validationLogPath)
	}
	experiment, err := readExperiment(o.experimentPath)
	if err != nil {
		return err
	}
	configData, err := os.ReadFile(experiment.GetGeneratorConfigPath())
	if err != nil {
		return err
	}
	config := &genpb.QueryTemplateConfigs{}
	if err := prototext.Unmarshal(configData, config); err != nil {
		return err
	}
	if config.GetMarkovSampling() == nil {
		return fmt.Errorf("this pipeline requires a markov_sampling configuration")
	}
	if len(config.GetMarkovSampling().GetInjections()) > 0 {
		return fmt.Errorf("training/calibration generator config must not contain injections; put them in the experiment's validation specification")
	}
	for _, node := range config.GetMarkovSampling().GetNodes() {
		if node.GetTemplate().GetIsGroundTruthAnomalous() {
			return fmt.Errorf("training/calibration templates must be normal")
		}
	}
	out, err := filepath.Abs(o.outputDir)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(out, 0755); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(out, "experiment.textproto"), []byte(prototext.Format(experiment)), 0644); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(out, "generator_config.textproto"), configData, 0644); err != nil {
		return err
	}
	cmd := exec.CommandContext(ctx, o.generatorBin,
		"--config_file="+filepath.Join(out, "generator_config.textproto"), "--output_dir="+out,
		fmt.Sprintf("--num_logs=%d", experiment.GetTrainingActions()), fmt.Sprintf("--start_time_ms=%d", experiment.GetStartTimeMs()),
		fmt.Sprintf("--seed=%d", experiment.GetTrainingSeed()))
	cmd.Stdout, cmd.Stderr = os.Stdout, os.Stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("generator: %w", err)
	}
	if err := exportLogs(ctx, out, experiment.GetTrainingActions()); err != nil {
		return err
	}
	log.Printf("Generated %d actions; training on every record", experiment.GetTrainingActions())

	conn, err := grpc.NewClient(o.serverAddr, grpc.WithTransportCredentials(local.NewCredentials()))
	if err != nil {
		return err
	}
	defer conn.Close()
	trainingConfig := &trainingpb.ModelTrainingConfig{}
	trainingConfig.SetNgramHistogramConfig(experiment.GetTrainingConfig())
	req := &trainingpb.TrainModelsRequest{}
	req.SetTrainLogPath(filepath.Join(out, "logs.binproto"))
	req.SetTrainingConfigs([]*trainingpb.ModelTrainingConfig{trainingConfig})
	if err := os.WriteFile(filepath.Join(out, "training_request.textproto"), []byte(prototext.Format(req)), 0644); err != nil {
		return err
	}
	resp, err := trainingpb.NewTrainingServiceClient(conn).TrainModels(ctx, req)
	if err != nil {
		return fmt.Errorf("training RPC: %w", err)
	}
	if err := writeJSON(filepath.Join(out, "training_response.json"), resp); err != nil {
		return err
	}
	if len(resp.GetResults()) != 1 {
		return fmt.Errorf("expected one trained model, got %d", len(resp.GetResults()))
	}
	// Like train_log_path, model_location uses a filesystem shared with the service.
	modelBytes, err := os.ReadFile(resp.GetResults()[0].GetModelLocation())
	if err != nil {
		return err
	}
	model := &modelpb.Model{}
	if err := proto.Unmarshal(modelBytes, model); err != nil {
		return err
	}
	if !model.HasNgramHistogram() || model.GetMetadata().GetTotalLogEntriesProcessed() != experiment.GetTrainingActions() {
		return fmt.Errorf("training returned an unexpected model or action count")
	}
	if err := os.WriteFile(filepath.Join(out, "histogram.bin"), modelBytes, 0644); err != nil {
		return err
	}
	if err := writeJSON(filepath.Join(out, "histogram.json"), model); err != nil {
		return err
	}
	for _, d := range model.GetNgramHistogram().GetModelData().GetHistograms() {
		log.Printf("n=%d: %d n-gram occurrences, %d observed sequences", d.GetN(), d.GetTotalWindows(), len(d.GetEntries()))
	}
	data := model.GetNgramHistogram().GetModelData()
	log.Printf("Window statistics: %d complete windows, %d trailing actions", data.GetCompleteWindows(), data.GetTrailingActions())
	return calibrateAndValidate(ctx, o, experiment, out, model)
}

func readExperiment(path string) (*clientpb.MarkovExperimentConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	config := &clientpb.MarkovExperimentConfig{}
	if err := prototext.Unmarshal(data, config); err != nil {
		return nil, fmt.Errorf("experiment config: %w", err)
	}
	training := config.GetTrainingConfig()
	if training == nil {
		return nil, fmt.Errorf("training_config is required")
	}
	if !training.HasWindowLength() {
		training.SetWindowLength(300)
	}
	if config.GetGeneratorConfigPath() == "" || config.GetTrainingActions() <= 0 || config.GetTrainingActions() > int64(^uint(0)>>1) || training.GetMaxN() < 1 || training.GetMaxN() > 3 || training.GetWindowLength() < training.GetMaxN() {
		return nil, fmt.Errorf("require a generator config path, positive training_actions, max_n in [1,3], and window_length >= max_n")
	}
	if config.GetTrainingActions() < 2*int64(training.GetWindowLength()) {
		return nil, fmt.Errorf("variance scoring requires at least two complete training windows")
	}
	if config.GetScoringConfig() == nil {
		scoring := &modelpb.NGramScoringConfig{}
		scoring.SetStandardDeviationFloorCount(1)
		config.SetScoringConfig(scoring)
	}
	floor := config.GetScoringConfig().GetStandardDeviationFloorCount()
	if math.IsNaN(floor) || math.IsInf(floor, 0) || floor <= 0 {
		return nil, fmt.Errorf("standard_deviation_floor_count must be finite and positive")
	}
	if !config.HasCalibrationActions() {
		config.SetCalibrationActions(300000)
	}
	if !config.HasCalibrationSeed() {
		config.SetCalibrationSeed(43)
	}
	if !config.HasValidationActions() {
		config.SetValidationActions(30000)
	}
	if !config.HasValidationSeed() {
		config.SetValidationSeed(44)
	}
	if config.GetCalibrationActions() < 2*int64(training.GetWindowLength()) || config.GetCalibrationActions() > int64(^uint(0)>>1) || config.GetValidationActions() < int64(training.GetWindowLength()) || config.GetValidationActions() > int64(^uint(0)>>1) {
		return nil, fmt.Errorf("calibration needs at least two complete windows and validation at least one")
	}
	if config.GetTrainingSeed() == config.GetCalibrationSeed() || config.GetTrainingSeed() == config.GetValidationSeed() || config.GetCalibrationSeed() == config.GetValidationSeed() {
		return nil, fmt.Errorf("training, calibration and validation seeds must differ")
	}
	percentile := config.GetCalibrationPercentile()
	if math.IsNaN(percentile) || percentile <= 0 || percentile >= 1 {
		return nil, fmt.Errorf("calibration_percentile must be in (0,1)")
	}
	if err := validateExperimentInjections(config); err != nil {
		return nil, err
	}
	return config, nil
}

func writeJSON(path string, message proto.Message) error {
	data, err := (protojson.MarshalOptions{Indent: "  ", UseProtoNames: true, EmitUnpopulated: true}).Marshal(message)
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(data, '\n'), 0644)
}
