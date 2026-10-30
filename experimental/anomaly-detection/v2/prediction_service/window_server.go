package main

import (
	"context"
	"errors"
	"math"
	"os"
	"path/filepath"

	"github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common"
	"github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/prediction_service/service"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"

	modelpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common/proto"
	predictionpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/prediction_service/proto"
)

func windowRPCError(err error) error {
	if errors.Is(err, context.Canceled) {
		return status.Error(codes.Canceled, err.Error())
	}
	if errors.Is(err, context.DeadlineExceeded) {
		return status.Error(codes.DeadlineExceeded, err.Error())
	}
	if errors.Is(err, common.ErrInvalidArgument) {
		return status.Errorf(codes.InvalidArgument, "%v", err)
	}
	if errors.Is(err, common.ErrNotFound) || os.IsNotExist(err) {
		return status.Errorf(codes.NotFound, "%v", err)
	}
	return status.Errorf(codes.Internal, "%v", err)
}

func (s *predictionServer) CalibrateWindows(ctx context.Context, req *predictionpb.CalibrateWindowsRequest) (*predictionpb.CalibrateWindowsResponse, error) {
	if req == nil || req.GetModelLocation() == "" || req.GetCalibrationLogPath() == "" || req.GetOutputModelLocation() == "" {
		return nil, status.Error(codes.InvalidArgument, "model_location, calibration_log_path and output_model_location are required")
	}
	if math.IsNaN(req.GetPercentile()) || req.GetPercentile() <= 0 || req.GetPercentile() >= 1 {
		return nil, status.Error(codes.InvalidArgument, "percentile must be in (0,1)")
	}
	input, err := filepath.Abs(common.ResolvePath(req.GetModelLocation()))
	if err != nil {
		return nil, windowRPCError(err)
	}
	output, err := filepath.Abs(common.ResolvePath(req.GetOutputModelLocation()))
	if err != nil {
		return nil, windowRPCError(err)
	}
	inputInfo, inputErr := os.Stat(input)
	outputInfo, outputErr := os.Stat(output)
	if input == output || (inputErr == nil && outputErr == nil && os.SameFile(inputInfo, outputInfo)) {
		return nil, status.Error(codes.InvalidArgument, "output model must differ from the training model")
	}
	model, err := loadModelProtoFromDisk(ctx, input)
	if err != nil {
		return nil, windowRPCError(err)
	}
	// Recalibration must also produce unclassified scores, independent of any
	// existing threshold. Neither the caller's model nor its file is modified.
	baseline := proto.Clone(model).(*modelpb.Model)
	if baseline.HasNgramHistogram() {
		baseline.GetNgramHistogram().ClearCalibration()
	}
	scorer, err := service.NewNGramWindowScorerWithConfig(baseline, req.GetScoringConfig())
	if err != nil {
		return nil, windowRPCError(err)
	}
	report, err := scorer.ScoreFile(ctx, common.ResolvePath(req.GetCalibrationLogPath()))
	if err != nil {
		return nil, windowRPCError(err)
	}
	calibrated, err := service.FitNGramCalibration(baseline, report, req.GetPercentile())
	if err != nil {
		return nil, windowRPCError(err)
	}
	if err := ctx.Err(); err != nil {
		return nil, windowRPCError(err)
	}
	if err := saveModelToDisk(ctx, output, calibrated); err != nil {
		return nil, windowRPCError(err)
	}
	resp := &predictionpb.CalibrateWindowsResponse{}
	resp.SetModelLocation(output)
	resp.SetCalibration(calibrated.GetNgramHistogram().GetCalibration())
	resp.SetScores(report)
	return resp, nil
}

func (s *predictionServer) ScoreWindows(ctx context.Context, req *predictionpb.ScoreWindowsRequest) (*predictionpb.WindowScoreReport, error) {
	if req == nil || req.GetModelLocation() == "" || req.GetValidationLogPath() == "" {
		return nil, status.Error(codes.InvalidArgument, "model_location and validation_log_path are required")
	}
	model, err := loadModelProtoFromDisk(ctx, common.ResolvePath(req.GetModelLocation()))
	if err != nil {
		return nil, windowRPCError(err)
	}
	if model.HasNgramHistogram() && !model.GetNgramHistogram().HasCalibration() {
		return nil, status.Error(codes.FailedPrecondition, "calibrate the histogram model before validation")
	}
	if model.HasNgramHistogram() && model.GetNgramHistogram().GetCalibration().GetScoringVersion() != service.NGramScoringVersion {
		return nil, status.Error(codes.FailedPrecondition, "calibration is from a different scorer; recalibrate before validation")
	}
	scorer, err := service.NewNGramWindowScorer(model)
	if err != nil {
		return nil, windowRPCError(err)
	}
	report, err := scorer.ScoreFile(ctx, common.ResolvePath(req.GetValidationLogPath()))
	if err != nil {
		return nil, windowRPCError(err)
	}
	if report.GetCompleteWindows() == 0 {
		return nil, status.Error(codes.InvalidArgument, "validation input contains no complete windows")
	}
	return report, nil
}
