package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/local"
	"google.golang.org/protobuf/encoding/prototext"
	"google.golang.org/protobuf/proto"

	clientpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/client/proto"
	modelpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common/proto"
	predictionpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/prediction_service/proto"
)

// Detailed reports grow with the number of windows and n-gram categories.
// Million-action sweeps exceed gRPC's default 4 MiB receive limit.
const maxWindowReportBytes = 256 << 20

func generateEvaluationStream(ctx context.Context, o options, experiment *clientpb.MarkovExperimentConfig, out, stage string, count, seed int64) (string, error) {
	if stage == "validation" && experiment.GetIndependentAnomalies() != nil {
		return generateIndependentValidation(ctx, o, experiment, out)
	}
	dir := filepath.Join(out, stage)
	configPath := filepath.Join(out, "generator_config.textproto")
	if stage == "validation" {
		var err error
		configPath, err = writeValidationGeneratorConfig(out, dir, experiment)
		if err != nil {
			return "", err
		}
	}
	cmd := exec.CommandContext(ctx, o.generatorBin, "--config_file="+configPath, "--output_dir="+dir, fmt.Sprintf("--num_logs=%d", count), fmt.Sprintf("--seed=%d", seed), fmt.Sprintf("--start_time_ms=%d", experiment.GetStartTimeMs()))
	cmd.Stdout, cmd.Stderr = os.Stdout, os.Stderr
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("%s generation: %w", stage, err)
	}
	if err := exportLogs(ctx, dir, count); err != nil {
		return "", err
	}
	return filepath.Join(dir, "logs.binproto"), nil
}

func calibrateAndValidate(ctx context.Context, o options, experiment *clientpb.MarkovExperimentConfig, out string, baseline *modelpb.Model) error {
	calibrationPath, err := generateEvaluationStream(ctx, o, experiment, out, "calibration", experiment.GetCalibrationActions(), experiment.GetCalibrationSeed())
	if err != nil {
		return err
	}
	conn, err := grpc.NewClient(o.predictionAddr, grpc.WithTransportCredentials(local.NewCredentials()), grpc.WithDefaultCallOptions(grpc.MaxCallRecvMsgSize(maxWindowReportBytes)))
	if err != nil {
		return err
	}
	defer conn.Close()
	req := &predictionpb.CalibrateWindowsRequest{}
	req.SetModelLocation(filepath.Join(out, "histogram.bin"))
	req.SetCalibrationLogPath(calibrationPath)
	req.SetPercentile(experiment.GetCalibrationPercentile())
	req.SetScoringConfig(experiment.GetScoringConfig())
	req.SetOutputModelLocation(filepath.Join(out, "calibrated_histogram.bin"))
	if err := os.WriteFile(filepath.Join(out, "calibration_request.textproto"), []byte(prototext.Format(req)), 0644); err != nil {
		return err
	}
	resp, err := predictionpb.NewPredictionServiceClient(conn).CalibrateWindows(ctx, req)
	if err != nil {
		return fmt.Errorf("calibration RPC: %w", err)
	}
	if err := writeJSON(filepath.Join(out, "calibration_response.json"), resp); err != nil {
		return err
	}
	if err := writeJSON(filepath.Join(out, "calibration_scores.json"), resp.GetScores()); err != nil {
		return err
	}
	bytes, err := os.ReadFile(resp.GetModelLocation())
	if err != nil {
		return err
	}
	calibrated := &modelpb.Model{}
	if err := proto.Unmarshal(bytes, calibrated); err != nil {
		return err
	}
	// Verify calibration preserved all learned statistics and training metadata.
	withoutCalibration := proto.Clone(calibrated).(*modelpb.Model)
	withoutCalibration.GetNgramHistogram().ClearCalibration()
	if !proto.Equal(baseline, withoutCalibration) {
		return fmt.Errorf("calibration unexpectedly changed the trained baseline")
	}
	if err := writeJSON(filepath.Join(out, "calibrated_histogram.json"), calibrated); err != nil {
		return err
	}
	log.Printf("Calibrated percentile %.3f on %d windows: threshold %.9f", resp.GetCalibration().GetPercentile(), resp.GetCalibration().GetCompleteWindows(), resp.GetCalibration().GetThreshold())
	validationPath := o.validationLogPath
	if validationPath == "" {
		validationPath, err = generateEvaluationStream(ctx, o, experiment, out, "validation", experiment.GetValidationActions(), experiment.GetValidationSeed())
		if err != nil {
			return err
		}
	}
	return validateWindows(ctx, o, out, resp.GetModelLocation(), validationPath)
}

func validateWindows(ctx context.Context, o options, out, modelPath, validationPath string) error {
	modelPath, err := filepath.Abs(modelPath)
	if err != nil {
		return err
	}
	validationPath, err = filepath.Abs(validationPath)
	if err != nil {
		return err
	}
	conn, err := grpc.NewClient(o.predictionAddr, grpc.WithTransportCredentials(local.NewCredentials()), grpc.WithDefaultCallOptions(grpc.MaxCallRecvMsgSize(maxWindowReportBytes)))
	if err != nil {
		return err
	}
	defer conn.Close()
	req := &predictionpb.ScoreWindowsRequest{}
	req.SetModelLocation(modelPath)
	req.SetValidationLogPath(validationPath)
	if err := os.WriteFile(filepath.Join(out, "validation_request.textproto"), []byte(prototext.Format(req)), 0644); err != nil {
		return err
	}
	report, err := predictionpb.NewPredictionServiceClient(conn).ScoreWindows(ctx, req)
	if err != nil {
		return fmt.Errorf("validation scoring RPC: %w", err)
	}
	if err := writeJSON(filepath.Join(out, "validation_scores.json"), report); err != nil {
		return err
	}
	evaluation, err := evaluateWindows(ctx, validationPath, report)
	if err != nil {
		return err
	}
	if err := writeJSON(filepath.Join(out, "validation_evaluation.json"), evaluation); err != nil {
		return err
	}
	log.Printf("Ground truth: TP=%d FP=%d TN=%d FN=%d; detected %d/%d scored segments", evaluation.GetTruePositives(), evaluation.GetFalsePositives(), evaluation.GetTrueNegatives(), evaluation.GetFalseNegatives(), evaluation.GetDetectedSegments(), evaluation.GetScoredSegments())
	flagged := 0
	for _, window := range report.GetWindows() {
		if window.GetIsAnomalous() {
			flagged++
		}
	}
	log.Printf("Validation: %d windows, %d flagged, %d trailing actions not scored; threshold %.9f", report.GetCompleteWindows(), flagged, report.GetTrailingActions(), report.GetThreshold())
	log.Printf("Experiment outputs: %s", out)
	return nil
}
