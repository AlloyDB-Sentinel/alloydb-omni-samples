package main

import (
	"context"
	"errors"
	"path/filepath"
	"testing"

	"github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common"
	"github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/prediction_service/service"
	training "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/training_service/service"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	auditpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common/proto"
	modelpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common/proto"
	predictionpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/prediction_service/proto"
)

// Server wiring example tests for task 6.3 (Requirements 6.1, 6.2, 6.4).
//
// These tests verify the server wiring added in tasks 6.1 and 6.2:
//   - modelFromProto returns a working *service.NGramScoringModel (typed as
//     service.Model) for a valid NgramHistogram proto and propagates the
//     constructor error for an invalid one (R6.1, R6.2).
//   - The Predict path no longer rejects NgramHistogram models with the old
//     "histogram models require the ScoreWindows RPC" FailedPrecondition (R6.4).
//
// Every new top-level symbol is prefixed with "Wiring" so it cannot collide
// with symbols authored by sibling test tasks in this same package main.
// Shared helpers (createTestLogFile, makeSingleLogPredictRequest, etc.) and the
// trainer-based baseline construction from window_server_test.go are reused,
// never redeclared.

// wiringNGramBaseline builds an uncalibrated NgramHistogram baseline model via
// the training package, mirroring window_server_test.go
// (TestWindowCalibrationAndValidationRPC). maxN=3, windowLength=4, and 8
// repeated actions yield two complete training windows, which the scorer
// construction requires. An uncalibrated baseline still constructs a usable
// scorer (calibration is optional in NewNGramWindowScorer), so it is sufficient
// for the modelFromProto wiring assertion.
func wiringNGramBaseline(t *testing.T) *modelpb.Model {
	t.Helper()
	trainer, err := training.NewNGramHistogramTrainer(3, 4)
	if err != nil {
		t.Fatalf("failed to create ngram trainer: %v", err)
	}
	for _, a := range []string{"A", "A", "B", "B", "A", "A", "B", "B"} {
		trainer.Count(common.ExtractedLogFeatures{QueryTemplate: a})
	}
	return trainer.BuildModel()
}

// wiringCalibratedNGramModelPath builds the baseline on disk, then calibrates it
// through the CalibrateWindows RPC (the same flow exercised by
// window_server_test.go), returning the path to the calibrated model file. A
// calibrated model is required so the Predict path can score real windows.
func wiringCalibratedNGramModelPath(ctx context.Context, t *testing.T, dir string) string {
	t.Helper()
	baseline := wiringNGramBaseline(t)
	basePath := filepath.Join(dir, "wiring_baseline.bin")
	createTestModelFile(ctx, t, basePath, baseline)

	calPath := filepath.Join(dir, "wiring_calibrated.bin")
	logs := filepath.Join(dir, "wiring_calibration.binproto")
	writeWindowLogs(t, logs, []string{"A", "A", "B", "B", "B", "B", "A", "A", "A", "B", "A", "B"}, false)

	srv := NewPredictionServer()
	req := &predictionpb.CalibrateWindowsRequest{}
	req.SetModelLocation(basePath)
	req.SetCalibrationLogPath(logs)
	req.SetOutputModelLocation(calPath)
	req.SetPercentile(0.99)
	if _, err := srv.CalibrateWindows(ctx, req); err != nil {
		t.Fatalf("failed to calibrate ngram model: %v", err)
	}
	return calPath
}

// TestWiringModelFromProtoNgram verifies modelFromProto wiring for the
// NgramHistogram scenario (R6.1, R6.2).
func TestWiringModelFromProtoNgram(t *testing.T) {
	t.Run("valid_proto_returns_ngram_scoring_model", func(t *testing.T) {
		baseline := wiringNGramBaseline(t)

		model, err := modelFromProto(baseline)
		if err != nil {
			t.Fatalf("modelFromProto returned error for valid ngram proto: %v", err)
		}
		if model == nil {
			t.Fatal("modelFromProto returned a nil model for valid ngram proto")
		}
		// The returned value is typed as service.Model but must be an
		// *service.NGramScoringModel under the hood (R6.1).
		if _, ok := model.(*service.NGramScoringModel); !ok {
			t.Fatalf("modelFromProto returned %T, want *service.NGramScoringModel", model)
		}
	})

	t.Run("invalid_proto_propagates_constructor_error", func(t *testing.T) {
		// An NgramHistogram scenario that the scorer construction rejects: fewer
		// than two complete training windows. This exercises error propagation
		// from NewNGramScoringModel through modelFromProto (R6.2), and confirms
		// the error is the scorer's ErrInvalidArgument surfaced unchanged.
		trainer, err := training.NewNGramHistogramTrainer(3, 4)
		if err != nil {
			t.Fatalf("failed to create ngram trainer: %v", err)
		}
		// Only four actions => a single complete window; construction requires
		// at least two.
		for _, a := range []string{"A", "A", "B", "B"} {
			trainer.Count(common.ExtractedLogFeatures{QueryTemplate: a})
		}
		invalid := trainer.BuildModel()

		model, err := modelFromProto(invalid)
		if err == nil {
			t.Fatal("modelFromProto returned nil error for invalid ngram proto")
		}
		if !errors.Is(err, common.ErrInvalidArgument) {
			t.Errorf("modelFromProto error = %v, want it to wrap common.ErrInvalidArgument", err)
		}
		// No usable model instance must be returned on failure (R6.2). The
		// failing constructor yields a nil *service.NGramScoringModel; because
		// modelFromProto widens it to the service.Model interface, the interface
		// value is non-nil but wraps a nil pointer, so assert the concrete
		// pointer is nil rather than comparing the interface to nil.
		if ngram, ok := model.(*service.NGramScoringModel); !ok || ngram != nil {
			t.Fatalf("modelFromProto returned a usable model %#v alongside error; want a nil *service.NGramScoringModel", model)
		}
	})
}

// TestWiringPredictAcceptsNgram verifies that the Predict RPC no longer rejects
// NgramHistogram models with the old precondition failure (R6.4). A Predict RPC
// against a calibrated ngram model must not return codes.FailedPrecondition for
// the "require the ScoreWindows RPC" reason; it flows through the shared predict
// path and succeeds.
func TestWiringPredictAcceptsNgram(t *testing.T) {
	ctx := context.Background()
	dir := t.TempDir()

	calPath := wiringCalibratedNGramModelPath(ctx, t, dir)

	entry := &auditpb.AuditLogEntry{}
	entry.SetNormalizedQueryTemplate("A")
	entry.SetDbUser("db_user")
	entry.SetDbName("db_name")

	srv := &predictionServer{}
	resp, err := srv.Predict(ctx, makeSingleLogPredictRequest(entry, calPath))

	gotCode := status.Code(err)
	if gotCode == codes.FailedPrecondition {
		t.Fatalf("Predict rejected an ngram model with FailedPrecondition (old ScoreWindows gate still present): %v", err)
	}
	if err != nil {
		t.Fatalf("Predict returned unexpected error for ngram model: %v", err)
	}
	if resp.GetSingleFinding() == nil {
		t.Fatal("Predict returned no single finding for a single-log ngram prediction")
	}
}
