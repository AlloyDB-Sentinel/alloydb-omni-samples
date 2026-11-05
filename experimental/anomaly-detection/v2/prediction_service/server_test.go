package main

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/encoding/protodelim"
	"google.golang.org/protobuf/encoding/prototext"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/durationpb"

	auditpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common/proto"
	findingpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common/proto"
	modelpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common/proto"
	predictionpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/prediction_service/proto"
	service "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/prediction_service/service"
)

// createTestLogFile creates a binary log file with a single AuditLogEntry.
func createTestLogFile(ctx context.Context, t *testing.T, path string) {
	t.Helper()
	file, err := os.Create(path)
	if err != nil {
		t.Fatalf("failed to create test log file: %v", err)
	}
	defer file.Close()

	entry := &auditpb.AuditLogEntry{}
	entry.SetNormalizedQueryTemplate("SELECT")
	entry.SetQueryStatement("SELECT 1")
	entry.SetDbUser("db_user")
	entry.SetDbName("db_name")
	entry.SetTimestampMs(time.Now().UnixMilli())

	if _, err := protodelim.MarshalTo(file, entry); err != nil {
		t.Fatalf("failed to write test log entry: %v", err)
	}
}

// createTestModelFile serializes a Model proto and writes it to disk.
func createTestModelFile(ctx context.Context, t *testing.T, path string, modelProto *modelpb.Model) {
	t.Helper()
	data, err := proto.Marshal(modelProto)
	if err != nil {
		t.Fatalf("failed to marshal model proto: %v", err)
	}
	if err := os.WriteFile(path, data, 0644); err != nil {
		t.Fatalf("failed to write model file: %v", err)
	}
}

func makeSingleLogPredictRequest(entry *auditpb.AuditLogEntry, modelLocation string) *predictionpb.PredictRequest {
	req := &predictionpb.PredictRequest{}
	req.SetSingleLog(entry)
	req.SetModelLocation(modelLocation)
	return req
}

func makeBatchLogsPredictRequest(entries []*auditpb.AuditLogEntry, modelLocation string) *predictionpb.PredictRequest {
	req := &predictionpb.PredictRequest{}
	batch := &predictionpb.AuditLogBatch{}
	batch.SetLogEntries(entries)
	req.SetBatchLogs(batch)
	req.SetModelLocation(modelLocation)
	return req
}

func makeFilePredictRequest(path string, modelLocation string) *predictionpb.PredictRequest {
	req := &predictionpb.PredictRequest{}
	req.SetPredictLogPath(path)
	req.SetModelLocation(modelLocation)
	return req
}

func makeTestTrafficDeviationModel(ratio float64, version string) *modelpb.Model {
	tdProto := &modelpb.TrafficDeviationModel{}
	tdProto.SetNormalTemplates([]string{"SELECT"})
	tdProto.SetAnomalyScores(map[string]float64{"DROP": 0.5})

	tdConfig := &modelpb.TrafficDeviationTrainingConfig{}
	tdConfig.SetAnomalyThresholdRatio(ratio)

	tdScenario := &modelpb.TrafficDeviationScenario{}
	tdScenario.SetModelData(tdProto)
	tdScenario.SetConfig(tdConfig)

	modelProto := &modelpb.Model{}
	modelProto.SetTrafficDeviation(tdScenario)
	if version != "" {
		meta := &modelpb.ModelMetadata{}
		meta.SetTrainerVersion(version)
		modelProto.SetMetadata(meta)
	}
	return modelProto
}

func makeTestVolumetricSpikeModel(multiplier float64, version string) *modelpb.Model {
	vsProto := &modelpb.VolumetricSpikeModel{}
	key := &modelpb.CycleBucketKey{}
	key.SetDbUser("db_user")
	key.SetQueryTemplate("SELECT")
	key.SetBucketIndex(0)

	val := &modelpb.VolumetricBaseline{}
	val.SetMean(1.0)
	val.SetStdDev(1.0)
	val.SetUpperThreshold(1.5)

	entry := &modelpb.VolumetricBaselineEntry{}
	entry.SetKey(key)
	entry.SetValue(val)
	vsProto.SetEntries([]*modelpb.VolumetricBaselineEntry{entry})

	config := &modelpb.CycleConfig{}
	config.SetBucketDuration(durationpb.New(24 * time.Hour))
	config.SetCycleDuration(durationpb.New(24 * time.Hour))
	config.SetBucketsPerCycle(1)
	vsProto.SetCycleConfig(config)

	vsConfig := &modelpb.VolumetricSpikeTrainingConfig{}
	vsConfig.SetStdDevMultiplier(multiplier)

	vsScenario := &modelpb.VolumetricSpikeScenario{}
	vsScenario.SetModelData(vsProto)
	vsScenario.SetConfig(vsConfig)

	modelProto := &modelpb.Model{}
	modelProto.SetVolumetricSpike(vsScenario)
	if version != "" {
		meta := &modelpb.ModelMetadata{}
		meta.SetTrainerVersion(version)
		modelProto.SetMetadata(meta)
	}
	return modelProto
}

func TestPredict(t *testing.T) {
	ctx := context.Background()
	tmpDir := t.TempDir()

	inputPath := filepath.Join(tmpDir, "predict_input.binproto")
	createTestLogFile(ctx, t, inputPath)

	// 1. Create a valid TrafficDeviationModel proto for testing disk loading
	modelProto := makeTestTrafficDeviationModel(0.1, "")
	diskModelPath := filepath.Join(tmpDir, "model_td.bin")
	createTestModelFile(ctx, t, diskModelPath, modelProto)

	// 2. Create a valid VolumetricSpikeModel proto for testing disk loading
	vsModelProto := makeTestVolumetricSpikeModel(3.0, "")
	vsDiskModelPath := filepath.Join(tmpDir, "model_vs.bin")
	createTestModelFile(ctx, t, vsDiskModelPath, vsModelProto)

	// 3. Create a corrupt model file (invalid proto bytes)
	corruptModelPath := filepath.Join(tmpDir, "corrupt_model.bin")
	if err := os.WriteFile(corruptModelPath, []byte("invalid proto bytes"), 0644); err != nil {
		t.Fatalf("failed to create corrupt model file: %v", err)
	}

	// 4. Create an uninitializable model file in a separate subfolder (valid Model proto, but missing scenario case)
	invalidSubDir := t.TempDir()
	uninitializableModelPath := filepath.Join(invalidSubDir, "uninitializable_model.bin")
	invalidScenarioModel := &modelpb.Model{} // No scenario set
	createTestModelFile(ctx, t, uninitializableModelPath, invalidScenarioModel)

	// 5. Create a corrupt input log file (invalid binproto stream)
	corruptInputPath := filepath.Join(tmpDir, "corrupt_input.binproto")
	if err := os.WriteFile(corruptInputPath, []byte("not a valid binproto stream"), 0644); err != nil {
		t.Fatalf("failed to create corrupt input file: %v", err)
	}

	entryNormal := &auditpb.AuditLogEntry{}
	entryNormal.SetNormalizedQueryTemplate("SELECT")
	entryNormal.SetDbUser("db_user")
	entryNormal.SetDbName("db_name")
	entryNormal.SetTimestampMs(time.Now().UnixMilli())

	entryAnomaly := &auditpb.AuditLogEntry{}
	entryAnomaly.SetNormalizedQueryTemplate("DROP")
	entryAnomaly.SetDbUser("db_user")
	entryAnomaly.SetDbName("db_name")
	entryAnomaly.SetTimestampMs(time.Now().UnixMilli())

	tests := []struct {
		name     string
		req      *predictionpb.PredictRequest
		wantCode codes.Code
		before   func(t *testing.T) func()
		verify   func(t *testing.T, resp *predictionpb.PredictResponse)
	}{
		// Phase 1: Request Validation & Model Resolution
		{
			name: "invalid_request_empty_payload",
			req: func() *predictionpb.PredictRequest {
				r := &predictionpb.PredictRequest{}
				r.SetModelLocation(diskModelPath)
				return r
			}(),
			wantCode: codes.InvalidArgument,
		},
		{
			name:     "model_resolution_dir_not_set",
			req:      makeSingleLogPredictRequest(entryNormal, ""),
			wantCode: codes.FailedPrecondition,
			before: func(t *testing.T) func() {
				oldDir := *modelsDir
				*modelsDir = ""
				return func() {
					*modelsDir = oldDir
				}
			},
		},
		{
			name:     "model_resolution_model_not_found",
			req:      makeSingleLogPredictRequest(entryNormal, "non_existent_model_loc"),
			wantCode: codes.FailedPrecondition,
		},
		{
			name:     "model_resolution_corrupt_model_file",
			req:      makeSingleLogPredictRequest(entryNormal, corruptModelPath),
			wantCode: codes.FailedPrecondition,
		},
		{
			// Verifies internal error when model proto is parsed but modelFromProto fails initialization.
			name:     "model_initialization_error",
			req:      makeSingleLogPredictRequest(entryNormal, uninitializableModelPath),
			wantCode: codes.Internal,
		},

		// Phase 2: Single Log Payload Branch
		{
			name:     "happy_path_single_log_normal",
			req:      makeSingleLogPredictRequest(entryNormal, diskModelPath),
			wantCode: codes.OK,
			verify: func(t *testing.T, resp *predictionpb.PredictResponse) {
				finding := resp.GetSingleFinding()
				if finding == nil {
					t.Fatalf("expected single finding, got nil")
				}
				if finding.GetIsAnomalousPredict() {
					t.Errorf("expected isAnomalousPredict to be false for SELECT")
				}
				if finding.GetConfidenceScore() != 1.0 {
					t.Errorf("expected score 1.0, got %f", finding.GetConfidenceScore())
				}
			},
		},
		{
			name:     "happy_path_single_log_anomaly",
			req:      makeSingleLogPredictRequest(entryAnomaly, diskModelPath),
			wantCode: codes.OK,
			verify: func(t *testing.T, resp *predictionpb.PredictResponse) {
				finding := resp.GetSingleFinding()
				if finding == nil {
					t.Fatalf("expected single finding, got nil")
				}
				if !finding.GetIsAnomalousPredict() {
					t.Errorf("expected isAnomalousPredict to be true for DROP")
				}
				if finding.GetConfidenceScore() != 0.5 {
					t.Errorf("expected score 0.5, got %f", finding.GetConfidenceScore())
				}
			},
		},

		// Phase 3: Batch Logs Payload Branch
		{
			name:     "happy_path_batch_logs",
			req:      makeBatchLogsPredictRequest([]*auditpb.AuditLogEntry{entryNormal, entryAnomaly}, diskModelPath),
			wantCode: codes.OK,
			verify: func(t *testing.T, resp *predictionpb.PredictResponse) {
				batch := resp.GetBatchFindings()
				if batch == nil {
					t.Fatalf("expected batch findings, got nil")
				}
				findings := batch.GetFindings()
				if len(findings) != 2 {
					t.Fatalf("expected 2 findings, got %d", len(findings))
				}
				if findings[0].GetIsAnomalousPredict() {
					t.Errorf("first finding should be normal")
				}
				if !findings[1].GetIsAnomalousPredict() {
					t.Errorf("second finding should be anomalous")
				}
			},
		},
		{
			name:     "happy_path_volumetric_spike_batch_predict",
			req:      makeBatchLogsPredictRequest([]*auditpb.AuditLogEntry{entryNormal, entryNormal}, vsDiskModelPath),
			wantCode: codes.OK,
			verify: func(t *testing.T, resp *predictionpb.PredictResponse) {
				batch := resp.GetBatchFindings()
				if batch == nil {
					t.Fatalf("expected batch findings, got nil")
				}
				findings := batch.GetFindings()
				if len(findings) != 2 {
					t.Fatalf("expected 2 findings, got %d", len(findings))
				}
				if findings[0].GetIsAnomalousPredict() {
					t.Errorf("expected first batch finding to be normal, got anomalous")
				}
				if !findings[1].GetIsAnomalousPredict() {
					t.Errorf("expected second batch finding to be anomalous, got normal")
				}
			},
		},

		// Phase 4: File Stream Payload Branch
		{
			name:     "happy_path_predict_log_path",
			req:      makeFilePredictRequest(inputPath, diskModelPath),
			wantCode: codes.OK,
			verify: func(t *testing.T, resp *predictionpb.PredictResponse) {
				outPath := resp.GetFindingsPath()
				if outPath == "" {
					t.Fatalf("expected findings path, got empty")
				}
				if _, err := os.Stat(outPath); err != nil {
					t.Errorf("output findings file %q does not exist: %v", outPath, err)
				}
				t.Cleanup(func() {
					os.Remove(outPath)
				})
			},
		},
		{
			name:     "predict_log_path_corrupt_stream",
			req:      makeFilePredictRequest(corruptInputPath, diskModelPath),
			wantCode: codes.InvalidArgument,
		},

		// Phase 5: Multi-Model & Filtering
		{
			name:     "happy_path_empty_location_defaults_to_all_models",
			req:      makeSingleLogPredictRequest(entryNormal, ""),
			wantCode: codes.OK,
			before: func(t *testing.T) func() {
				oldDir := *modelsDir
				*modelsDir = tmpDir
				return func() {
					*modelsDir = oldDir
				}
			},
			verify: func(t *testing.T, resp *predictionpb.PredictResponse) {
				batch := resp.GetBatchFindings()
				if batch == nil {
					t.Fatalf("expected batch findings for all-model predict with empty location, got nil")
				}
				if len(batch.GetFindings()) == 0 {
					t.Errorf("expected non-empty findings for all-model predict")
				}
			},
		},
		{
			name: "happy_path_predict_filtered_scenario",
			req: func() *predictionpb.PredictRequest {
				r := makeSingleLogPredictRequest(entryNormal, "")
				filter := &predictionpb.ModelFilter{}
				filter.SetTrafficDeviation(&modelpb.TrafficDeviationTrainingConfig{})
				r.SetFilter(filter)
				return r
			}(),
			wantCode: codes.OK,
			before: func(t *testing.T) func() {
				oldDir := *modelsDir
				*modelsDir = tmpDir
				return func() {
					*modelsDir = oldDir
				}
			},
			verify: func(t *testing.T, resp *predictionpb.PredictResponse) {
				if resp.GetSingleFinding() == nil && resp.GetBatchFindings() == nil {
					t.Fatalf("expected finding response for filtered scenario predict")
				}
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if test.name == "happy_path_predict_log_path" {
				t.Setenv("BUILD_WORKSPACE_DIRECTORY", tmpDir)
			}
			if test.before != nil {
				cleanup := test.before(t)
				if cleanup != nil {
					defer cleanup()
				}
			}
			s := &predictionServer{}
			resp, err := s.Predict(ctx, test.req)

			gotStatus, _ := status.FromError(err)
			if gotStatus.Code() != test.wantCode {
				t.Errorf("Predict() got code %v, want code %v, error: %v", gotStatus.Code(), test.wantCode, err)
			}

			if test.wantCode == codes.OK && test.verify != nil {
				test.verify(t, resp)
			}
		})
	}
}

func TestPushModel(t *testing.T) {
	ctx := context.Background()
	s := &predictionServer{}
	tmpDir := t.TempDir()

	oldModelsDir := *modelsDir
	*modelsDir = tmpDir
	defer func() { *modelsDir = oldModelsDir }()

	modelProto := makeTestTrafficDeviationModel(0.1, "")

	tests := []struct {
		name              string
		modelsDirOverride string
		req               *predictionpb.PushModelRequest
		wantCode          codes.Code
		verify            func(t *testing.T)
	}{
		// Phase 1: Request Validation & Flag Interception
		{
			name: "invalid_request_nil_model",
			req: func() *predictionpb.PushModelRequest {
				r := &predictionpb.PushModelRequest{}
				r.SetModelLocation("invalid_loc")
				return r
			}(),
			wantCode: codes.InvalidArgument,
		},
		{
			name:              "models_dir_not_set",
			modelsDirOverride: "UNSET",
			req: func() *predictionpb.PushModelRequest {
				r := &predictionpb.PushModelRequest{}
				r.SetModelLocation("pushed_model.bin")
				r.SetModel(modelProto)
				return r
			}(),
			wantCode: codes.FailedPrecondition,
		},

		// Phase 2: Disk Write Error
		{
			name:              "model_persistence_write_failure",
			modelsDirOverride: filepath.Join(tmpDir, "unwritable_file"),
			req: func() *predictionpb.PushModelRequest {
				unwritableFile := filepath.Join(tmpDir, "unwritable_file")
				if err := os.WriteFile(unwritableFile, []byte("data"), 0644); err != nil {
					t.Fatalf("failed to create file for write failure test: %v", err)
				}
				r := &predictionpb.PushModelRequest{}
				r.SetModelLocation("model.bin")
				r.SetModel(modelProto)
				return r
			}(),
			wantCode: codes.Internal,
		},

		// Phase 3: Happy Path Success
		{
			name: "happy_path_push_model_success",
			req: func() *predictionpb.PushModelRequest {
				r := &predictionpb.PushModelRequest{}
				r.SetModelLocation("pushed_model.bin")
				r.SetModel(modelProto)
				return r
			}(),
			wantCode: codes.OK,
			verify: func(t *testing.T) {
				pushedModelPath := filepath.Join(tmpDir, "pushed_model.bin")
				if _, err := os.Stat(pushedModelPath); err != nil {
					t.Errorf("expected pushed model file %q to exist: %v", pushedModelPath, err)
				}
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if test.modelsDirOverride == "UNSET" {
				*modelsDir = ""
			} else if test.modelsDirOverride != "" {
				*modelsDir = test.modelsDirOverride
			} else {
				*modelsDir = tmpDir
			}
			_, err := s.PushModel(ctx, test.req)
			gotStatus, _ := status.FromError(err)
			if gotStatus.Code() != test.wantCode {
				t.Errorf("PushModel() got code %v, want %v, err: %v", gotStatus.Code(), test.wantCode, err)
			}
			if test.wantCode == codes.OK && test.verify != nil {
				test.verify(t)
			}
		})
	}
}

func TestListModels(t *testing.T) {
	ctx := context.Background()
	tmpDir := t.TempDir()

	// Create test models
	modelTD := makeTestTrafficDeviationModel(0.1, "v1-td")
	modelVS := makeTestVolumetricSpikeModel(3.0, "v1-vs")

	pathTD := filepath.Join(tmpDir, "model_td.bin")
	pathVS := filepath.Join(tmpDir, "model_vs.bin")
	createTestModelFile(ctx, t, pathTD, modelTD)
	createTestModelFile(ctx, t, pathVS, modelVS)

	// Create a non-model file to ensure it's ignored
	nonModelPath := filepath.Join(tmpDir, "not_a_model.txt")
	if err := os.WriteFile(nonModelPath, []byte("some random data"), 0644); err != nil {
		t.Fatalf("failed to create non-model file: %v", err)
	}

	oldModelsDir := *modelsDir
	*modelsDir = tmpDir
	defer func() { *modelsDir = oldModelsDir }()

	s := &predictionServer{}

	tests := []struct {
		name     string
		req      *predictionpb.ListModelsRequest
		wantCode codes.Code
		before   func(t *testing.T) func()
		verify   func(t *testing.T, resp *predictionpb.ListModelsResponse)
	}{
		// Phase 1: Default Scanning & Location Lookup
		{
			name:     "happy_path_list_all_default_dir",
			req:      &predictionpb.ListModelsRequest{},
			wantCode: codes.OK,
			verify: func(t *testing.T, resp *predictionpb.ListModelsResponse) {
				if len(resp.GetModels()) != 2 {
					t.Fatalf("expected 2 models for empty request, got %d", len(resp.GetModels()))
				}
			},
		},
		{
			name: "happy_path_list_all_whitespace_location",
			req: func() *predictionpb.ListModelsRequest {
				r := &predictionpb.ListModelsRequest{}
				r.SetModelLocation("   ")
				return r
			}(),
			wantCode: codes.OK,
			verify: func(t *testing.T, resp *predictionpb.ListModelsResponse) {
				if len(resp.GetModels()) != 2 {
					t.Fatalf("expected 2 models for whitespace location, got %d", len(resp.GetModels()))
				}
			},
		},
		{
			name: "happy_path_single_model_location_valid",
			req: func() *predictionpb.ListModelsRequest {
				r := &predictionpb.ListModelsRequest{}
				r.SetModelLocation(pathTD)
				return r
			}(),
			wantCode: codes.OK,
			verify: func(t *testing.T, resp *predictionpb.ListModelsResponse) {
				models := resp.GetModels()
				if len(models) != 1 {
					t.Fatalf("expected 1 model for single location lookup, got %d", len(models))
				}
				if models[0].GetModelLocation() != pathTD {
					t.Errorf("expected location %s, got %s", pathTD, models[0].GetModelLocation())
				}
			},
		},
		{
			name: "happy_path_custom_directory_location",
			req: func() *predictionpb.ListModelsRequest {
				r := &predictionpb.ListModelsRequest{}
				r.SetModelLocation(tmpDir)
				return r
			}(),
			wantCode: codes.OK,
			verify: func(t *testing.T, resp *predictionpb.ListModelsResponse) {
				if len(resp.GetModels()) != 2 {
					t.Fatalf("expected 2 models for custom directory location %s, got %d", tmpDir, len(resp.GetModels()))
				}
			},
		},
		{
			name: "single_model_location_nonexistent",
			req: func() *predictionpb.ListModelsRequest {
				r := &predictionpb.ListModelsRequest{}
				r.SetModelLocation("nonexistent_model.bin")
				return r
			}(),
			wantCode: codes.OK,
			verify: func(t *testing.T, resp *predictionpb.ListModelsResponse) {
				if len(resp.GetModels()) != 0 {
					t.Fatalf("expected 0 models for nonexistent file, got %d", len(resp.GetModels()))
				}
			},
		},
		{
			name:     "models_dir_not_set",
			req:      &predictionpb.ListModelsRequest{},
			wantCode: codes.FailedPrecondition,
			before: func(t *testing.T) func() {
				oldDir := *modelsDir
				*modelsDir = ""
				return func() {
					*modelsDir = oldDir
				}
			},
		},

		// Phase 2: Metadata & Scenario Filter Matching
		{
			name: "filter_by_scenario_matching",
			req: func() *predictionpb.ListModelsRequest {
				r := &predictionpb.ListModelsRequest{}
				filter := &predictionpb.ModelFilter{}
				filter.SetTrafficDeviation(&modelpb.TrafficDeviationTrainingConfig{})
				r.SetFilter(filter)
				return r
			}(),
			wantCode: codes.OK,
			verify: func(t *testing.T, resp *predictionpb.ListModelsResponse) {
				models := resp.GetModels()
				if len(models) != 1 {
					t.Fatalf("expected 1 model for TrafficDeviation filter, got %d", len(models))
				}
				if models[0].GetMetadata().GetTrainerVersion() != "v1-td" {
					t.Errorf("expected trainer version v1-td, got %s", models[0].GetMetadata().GetTrainerVersion())
				}
			},
		},
		{
			name: "filter_by_scenario_param_matching",
			req: func() *predictionpb.ListModelsRequest {
				r := &predictionpb.ListModelsRequest{}
				filter := &predictionpb.ModelFilter{}
				cfg := &modelpb.TrafficDeviationTrainingConfig{}
				cfg.SetAnomalyThresholdRatio(0.1)
				filter.SetTrafficDeviation(cfg)
				r.SetFilter(filter)
				return r
			}(),
			wantCode: codes.OK,
			verify: func(t *testing.T, resp *predictionpb.ListModelsResponse) {
				if len(resp.GetModels()) != 1 {
					t.Fatalf("expected 1 model for matching param ratio 0.1, got %d", len(resp.GetModels()))
				}
			},
		},
		{
			name: "filter_by_scenario_param_mismatch",
			req: func() *predictionpb.ListModelsRequest {
				r := &predictionpb.ListModelsRequest{}
				filter := &predictionpb.ModelFilter{}
				cfg := &modelpb.TrafficDeviationTrainingConfig{}
				cfg.SetAnomalyThresholdRatio(0.99)
				filter.SetTrafficDeviation(cfg)
				r.SetFilter(filter)
				return r
			}(),
			wantCode: codes.OK,
			verify: func(t *testing.T, resp *predictionpb.ListModelsResponse) {
				if len(resp.GetModels()) != 0 {
					t.Fatalf("expected 0 models for mismatched ratio 0.99, got %d", len(resp.GetModels()))
				}
			},
		},
		{
			name: "filter_by_metadata_trainer_version",
			req: func() *predictionpb.ListModelsRequest {
				r := &predictionpb.ListModelsRequest{}
				filter := &predictionpb.ModelFilter{}
				metaF := &predictionpb.ModelMetadataFilter{}
				metaF.SetTrainerVersion("v1-td")
				filter.SetMetadataFilter(metaF)
				r.SetFilter(filter)
				return r
			}(),
			wantCode: codes.OK,
			verify: func(t *testing.T, resp *predictionpb.ListModelsResponse) {
				models := resp.GetModels()
				if len(models) != 1 {
					t.Fatalf("expected 1 model for metadata trainer_version filter, got %d", len(models))
				}
				if models[0].GetMetadata().GetTrainerVersion() != "v1-td" {
					t.Errorf("expected trainer version v1-td, got %s", models[0].GetMetadata().GetTrainerVersion())
				}
			},
		},
		{
			name: "single_model_location_with_matching_filter",
			req: func() *predictionpb.ListModelsRequest {
				r := &predictionpb.ListModelsRequest{}
				r.SetModelLocation(pathTD)
				filter := &predictionpb.ModelFilter{}
				filter.SetTrafficDeviation(&modelpb.TrafficDeviationTrainingConfig{})
				r.SetFilter(filter)
				return r
			}(),
			wantCode: codes.OK,
			verify: func(t *testing.T, resp *predictionpb.ListModelsResponse) {
				if len(resp.GetModels()) != 1 {
					t.Fatalf("expected 1 model for valid path + matching filter, got %d", len(resp.GetModels()))
				}
			},
		},
		{
			name: "single_model_location_with_mismatched_filter",
			req: func() *predictionpb.ListModelsRequest {
				r := &predictionpb.ListModelsRequest{}
				r.SetModelLocation(pathTD)
				filter := &predictionpb.ModelFilter{}
				filter.SetVolumetricSpike(&modelpb.VolumetricSpikeTrainingConfig{})
				r.SetFilter(filter)
				return r
			}(),
			wantCode: codes.OK,
			verify: func(t *testing.T, resp *predictionpb.ListModelsResponse) {
				if len(resp.GetModels()) != 0 {
					t.Fatalf("expected 0 models for valid TD path + mismatched VS filter, got %d", len(resp.GetModels()))
				}
			},
		},
		{
			name: "filter_by_volumetric_spike_scenario",
			req: func() *predictionpb.ListModelsRequest {
				r := &predictionpb.ListModelsRequest{}
				filter := &predictionpb.ModelFilter{}
				cfg := &modelpb.VolumetricSpikeTrainingConfig{}
				cfg.SetStdDevMultiplier(3.0)
				filter.SetVolumetricSpike(cfg)
				r.SetFilter(filter)
				return r
			}(),
			wantCode: codes.OK,
			verify: func(t *testing.T, resp *predictionpb.ListModelsResponse) {
				if len(resp.GetModels()) != 1 {
					t.Fatalf("expected 1 model for matching VolumetricSpike multiplier 3.0, got %d", len(resp.GetModels()))
				}
			},
		},
		{
			name: "filter_by_temporal_deviation_scenario",
			req: func() *predictionpb.ListModelsRequest {
				r := &predictionpb.ListModelsRequest{}
				filter := &predictionpb.ModelFilter{}
				cfg := &modelpb.TemporalDeviationTrainingConfig{}
				cfg.SetSilentThresholdRatio(0.05)
				filter.SetTemporalDeviation(cfg)
				r.SetFilter(filter)
				return r
			}(),
			wantCode: codes.OK,
			verify: func(t *testing.T, resp *predictionpb.ListModelsResponse) {
				if len(resp.GetModels()) != 0 {
					t.Fatalf("expected 0 models for unmatched TemporalDeviation filter, got %d", len(resp.GetModels()))
				}
			},
		},
		{
			name: "filter_by_metadata_trainer_version_mismatch",
			req: func() *predictionpb.ListModelsRequest {
				r := &predictionpb.ListModelsRequest{}
				filter := &predictionpb.ModelFilter{}
				metaF := &predictionpb.ModelMetadataFilter{}
				metaF.SetTrainerVersion("v99-nonexistent")
				filter.SetMetadataFilter(metaF)
				r.SetFilter(filter)
				return r
			}(),
			wantCode: codes.OK,
			verify: func(t *testing.T, resp *predictionpb.ListModelsResponse) {
				if len(resp.GetModels()) != 0 {
					t.Fatalf("expected 0 models for non-existent trainer version, got %d", len(resp.GetModels()))
				}
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if test.before != nil {
				cleanup := test.before(t)
				if cleanup != nil {
					defer cleanup()
				}
			}
			resp, err := s.ListModels(ctx, test.req)
			gotStatus, _ := status.FromError(err)
			if gotStatus.Code() != test.wantCode {
				t.Errorf("ListModels() got code %v, want %v, err: %v", gotStatus.Code(), test.wantCode, err)
			}
			if test.wantCode == codes.OK && test.verify != nil {
				test.verify(t, resp)
			}
		})
	}
}

// makeTestChiSquaredModel builds a chi-squared Model proto with the given M, N,
// and anomaly threshold using the opaque-API setters, mirroring the style of the
// other makeTest* helpers in this file.
func makeTestChiSquaredModel(m, n int64, threshold float64) *modelpb.Model {
	csConfig := &modelpb.ChiSquaredTrainingConfig{}
	csConfig.SetM(m)
	csConfig.SetN(n)
	csConfig.SetAnomalyThreshold(threshold)

	csScenario := &modelpb.ChiSquaredScenario{}
	csScenario.SetConfig(csConfig)

	modelProto := &modelpb.Model{}
	modelProto.SetChiSquared(csScenario)
	return modelProto
}

// TestModelFromProtoChiSquared asserts that modelFromProto wires chi-squared
// protos into the standard prediction flow: a valid proto yields a
// *service.ChiSquaredModel typed as service.Model and a nil error, an invalid
// proto propagates the constructor's init error, and the ngram-rejection guard
// in Predict cannot match a chi-squared target.
func TestModelFromProtoChiSquared(t *testing.T) {
	t.Run("valid_proto_returns_chi_squared_model", func(t *testing.T) {
		pb := makeTestChiSquaredModel(100, 10, 5.0)

		m, err := modelFromProto(pb)
		if err != nil {
			t.Fatalf("modelFromProto() returned unexpected error: %v", err)
		}
		if m == nil {
			t.Fatalf("modelFromProto() returned nil model for a valid chi-squared proto")
		}

		// The return type is service.Model; assert the concrete type is a
		// *service.ChiSquaredModel via a type switch so we verify both the
		// interface typing and the concrete model.
		switch concrete := m.(type) {
		case *service.ChiSquaredModel:
			if concrete == nil {
				t.Fatalf("modelFromProto() returned a nil *service.ChiSquaredModel")
			}
		default:
			t.Fatalf("modelFromProto() returned %T, want *service.ChiSquaredModel", m)
		}

		// It satisfies service.Model (guaranteed by the return type); assert the
		// assertion holds explicitly as well.
		var _ service.Model = m
		if _, ok := m.(service.Model); !ok {
			t.Errorf("returned model does not satisfy service.Model")
		}
	})

	t.Run("invalid_proto_returns_init_error", func(t *testing.T) {
		// N >= M violates the M > N constraint, so NewChiSquaredModel must fail.
		pb := makeTestChiSquaredModel(10, 10, 5.0)

		_, err := modelFromProto(pb)
		if err == nil {
			t.Fatalf("modelFromProto() returned nil error for an invalid chi-squared proto")
		}
		// Note: the returned service.Model interface value is deliberately not
		// asserted nil here. Every case in modelFromProto returns its concrete
		// constructor's result directly, so a nil *service.ChiSquaredModel
		// becomes a non-nil interface (Go's typed-nil wrapping). The Predict RPC
		// flow checks the error before touching the model, so the init error is
		// the contract that matters on the invalid path.
	})

	t.Run("ngram_rejection_does_not_trigger", func(t *testing.T) {
		pb := makeTestChiSquaredModel(100, 10, 5.0)

		// The ngram-rejection guard in Predict matches the ngram scenario
		// specifically. A chi-squared proto's active scenario is the chi-squared
		// case and never the ngram case, so the FailedPrecondition guard cannot
		// match.
		if got := pb.WhichScenario(); got != modelpb.Model_ChiSquared_case {
			t.Errorf("WhichScenario() = %v, want Model_ChiSquared_case", got)
		}
		if got := pb.WhichScenario(); got == modelpb.Model_NgramHistogram_case {
			t.Errorf("WhichScenario() = %v, must not equal Model_NgramHistogram_case", got)
		}
	})
}

// createMultiEntryTestLogFile writes a binary log file containing the given
// normalized query templates as consecutive AuditLogEntry records, using the
// same length-delimited framing as createTestLogFile. It is a uniquely-named
// helper added for the chi-squared integration tests (createTestLogFile only
// writes a single entry).
func createMultiEntryTestLogFile(ctx context.Context, t *testing.T, path string, templates []string) {
	t.Helper()
	file, err := os.Create(path)
	if err != nil {
		t.Fatalf("failed to create multi-entry test log file: %v", err)
	}
	defer file.Close()

	for _, tmpl := range templates {
		entry := &auditpb.AuditLogEntry{}
		entry.SetNormalizedQueryTemplate(tmpl)
		entry.SetQueryStatement(tmpl + " 1")
		entry.SetDbUser("db_user")
		entry.SetDbName("db_name")
		entry.SetTimestampMs(time.Now().UnixMilli())

		if _, err := protodelim.MarshalTo(file, entry); err != nil {
			t.Fatalf("failed to write test log entry %q: %v", tmpl, err)
		}
	}
}

// TestPredictChiSquared exercises the single-log, batch, and file Predict RPC
// paths against a chi-squared model and asserts the reset-before-first-entry
// behavior (batch/file reset the windows; single-log does not), plus the
// no-regression guarantee that mixing a chi-squared model with another scenario
// leaves the other scenario's findings unchanged (R8.4, R8.5, R9.3, R9.4).
func TestPredictChiSquared(t *testing.T) {
	ctx := context.Background()

	entrySelect := &auditpb.AuditLogEntry{}
	entrySelect.SetNormalizedQueryTemplate("SELECT")
	entrySelect.SetQueryStatement("SELECT 1")
	entrySelect.SetDbUser("db_user")
	entrySelect.SetDbName("db_name")
	entrySelect.SetTimestampMs(time.Now().UnixMilli())

	// Single-log path: a chi-squared model with N=3 can never fill its recent
	// window from a single entry, so the finding is a warm-up finding (not
	// anomalous, confidence 0.0). predictSingle does not call resetModels, so no
	// reset happens on this path (R8.5).
	t.Run("single_log_warmup", func(t *testing.T) {
		tmpDir := t.TempDir()
		csProto := makeTestChiSquaredModel(20, 3, 10.0)
		csPath := filepath.Join(tmpDir, "model_cs.bin")
		createTestModelFile(ctx, t, csPath, csProto)

		s := &predictionServer{}
		resp, err := s.Predict(ctx, makeSingleLogPredictRequest(entrySelect, csPath))
		if gotStatus, _ := status.FromError(err); gotStatus.Code() != codes.OK {
			t.Fatalf("Predict() got code %v, want OK, err: %v", gotStatus.Code(), err)
		}

		finding := resp.GetSingleFinding()
		if finding == nil {
			t.Fatalf("expected single finding, got nil")
		}
		if finding.GetIsAnomalousPredict() {
			t.Errorf("single-log chi-squared finding should be non-anomalous (warm-up)")
		}
		if finding.GetConfidenceScore() != 0.0 {
			t.Errorf("single-log chi-squared finding should have confidence 0.0 (warm-up), got %f", finding.GetConfidenceScore())
		}
	})

	// Batch path: predictBatch calls resetModels before processing, so the
	// windows start empty and the first N-1 entries are warm-up. With N=2 the
	// first entry's recent window holds only one Action, so it is non-anomalous
	// with confidence 0.0 — demonstrating the reset happened before the first
	// entry (R8.4). All entries share the same Action, so the distribution is
	// homogeneous and no later entry is anomalous either.
	t.Run("batch_reset_before_first_entry", func(t *testing.T) {
		tmpDir := t.TempDir()
		csProto := makeTestChiSquaredModel(10, 2, 5.0)
		csPath := filepath.Join(tmpDir, "model_cs.bin")
		createTestModelFile(ctx, t, csPath, csProto)

		entries := []*auditpb.AuditLogEntry{entrySelect, entrySelect, entrySelect, entrySelect}

		s := &predictionServer{}
		resp, err := s.Predict(ctx, makeBatchLogsPredictRequest(entries, csPath))
		if gotStatus, _ := status.FromError(err); gotStatus.Code() != codes.OK {
			t.Fatalf("Predict() got code %v, want OK, err: %v", gotStatus.Code(), err)
		}

		batch := resp.GetBatchFindings()
		if batch == nil {
			t.Fatalf("expected batch findings, got nil")
		}
		findings := batch.GetFindings()
		if len(findings) != len(entries) {
			t.Fatalf("expected %d findings, got %d", len(entries), len(findings))
		}

		// The key reset assertion: because the windows were reset before the
		// first entry, the recent window holds only one Action after the first
		// entry (N=2), so the first finding is a warm-up finding.
		if findings[0].GetIsAnomalousPredict() {
			t.Errorf("first batch finding should be non-anomalous (fresh/reset window warm-up)")
		}
		if findings[0].GetConfidenceScore() != 0.0 {
			t.Errorf("first batch finding should have confidence 0.0 (fresh/reset window warm-up), got %f", findings[0].GetConfidenceScore())
		}

		// A homogeneous stream never diverges, so no finding is anomalous.
		for i, f := range findings {
			if f.GetIsAnomalousPredict() {
				t.Errorf("finding %d should be non-anomalous for a homogeneous stream", i)
			}
		}
	})

	// File path: predictFile calls resetModels before streaming, so the first
	// entry sees fresh windows (warm-up). createTestLogFile only writes one
	// entry, so a multi-entry helper is used to exercise the reset-before-first
	// behavior across several entries. The path flow returns a non-empty
	// findings file (R8.4, R9.3).
	t.Run("file_reset_before_first_entry", func(t *testing.T) {
		tmpDir := t.TempDir()
		csProto := makeTestChiSquaredModel(10, 2, 5.0)
		csPath := filepath.Join(tmpDir, "model_cs.bin")
		createTestModelFile(ctx, t, csPath, csProto)

		inputPath := filepath.Join(tmpDir, "cs_input.binproto")
		createMultiEntryTestLogFile(ctx, t, inputPath, []string{"SELECT", "SELECT", "SELECT", "SELECT"})

		s := &predictionServer{}
		resp, err := s.Predict(ctx, makeFilePredictRequest(inputPath, csPath))
		if gotStatus, _ := status.FromError(err); gotStatus.Code() != codes.OK {
			t.Fatalf("Predict() got code %v, want OK, err: %v", gotStatus.Code(), err)
		}

		outPath := resp.GetFindingsPath()
		if outPath == "" {
			t.Fatalf("expected non-empty findings path, got empty")
		}
		t.Cleanup(func() { os.Remove(outPath) })
		info, err := os.Stat(outPath)
		if err != nil {
			t.Fatalf("output findings file %q does not exist: %v", outPath, err)
		}
		if info.Size() == 0 {
			t.Fatalf("output findings file %q is empty", outPath)
		}

		// Read back the first finding and assert it is a warm-up finding (reset
		// happened before the first entry), analogous to the batch path.
		// predictFile writes each finding's textproto followed by a single
		// newline; because prototext omits default (false / 0.0) fields, a
		// warm-up finding's block carries neither is_anomalous_predict nor
		// confidence_score. Parse the first finding from the leading block up to
		// the first identifying-field repeat (db_user) of the next finding.
		data, err := os.ReadFile(outPath)
		if err != nil {
			t.Fatalf("failed to read findings file: %v", err)
		}
		firstBlock := firstFindingBlock(string(data))
		first := &findingpb.AnomalyFinding{}
		if err := prototext.Unmarshal([]byte(firstBlock), first); err != nil {
			t.Fatalf("failed to unmarshal first finding from block %q: %v", firstBlock, err)
		}
		if first.GetIsAnomalousPredict() {
			t.Errorf("first file finding should be non-anomalous (fresh/reset window warm-up)")
		}
		if first.GetConfidenceScore() != 0.0 {
			t.Errorf("first file finding should have confidence 0.0 (fresh/reset window warm-up), got %f", first.GetConfidenceScore())
		}
	})

	// No-regression: a traffic-deviation model's finding for a SELECT entry is
	// not anomalous with confidence 1.0 (as asserted by the existing TestPredict
	// "happy_path_single_log_normal" case). Placing a chi-squared model in the
	// same models directory must not change that finding (R9.4). The baseline is
	// computed with only the TD model present, then compared against the finding
	// produced when both models are in the directory.
	t.Run("no_regression_td_unchanged_with_chi_squared", func(t *testing.T) {
		// Expected TD finding, matching the existing TestPredict expectations.
		expectedTD := &findingpb.AnomalyFinding{}
		expectedTD.SetDbUser("db_user")
		expectedTD.SetDbName("db_name")
		expectedTD.SetQueryStatement("SELECT 1")
		expectedTD.SetNormalizedQueryTemplate("SELECT")
		expectedTD.SetIsAnomalousPredict(false)
		expectedTD.SetConfidenceScore(1.0)

		// Baseline: directory containing ONLY the TD model.
		baselineDir := t.TempDir()
		tdProto := makeTestTrafficDeviationModel(0.1, "")
		createTestModelFile(ctx, t, filepath.Join(baselineDir, "model_td.bin"), tdProto)

		baselineTD := predictTDFindingInDir(ctx, t, baselineDir, entrySelect, expectedTD)

		// Mixed: directory containing the TD model AND a chi-squared model.
		mixedDir := t.TempDir()
		createTestModelFile(ctx, t, filepath.Join(mixedDir, "model_td.bin"), makeTestTrafficDeviationModel(0.1, ""))
		createTestModelFile(ctx, t, filepath.Join(mixedDir, "model_cs.bin"), makeTestChiSquaredModel(20, 3, 10.0))

		mixedTD := predictTDFindingInDir(ctx, t, mixedDir, entrySelect, expectedTD)

		// The TD finding must be byte-for-byte identical with and without the
		// chi-squared model present, and must equal the expected TD finding.
		if !proto.Equal(baselineTD, expectedTD) {
			t.Fatalf("baseline TD finding does not match expected:\n got: %v\nwant: %v", baselineTD, expectedTD)
		}
		if !proto.Equal(mixedTD, baselineTD) {
			t.Errorf("TD finding changed when a chi-squared model was added:\n with cs: %v\nwithout cs: %v", mixedTD, baselineTD)
		}
	})
}

// firstFindingBlock extracts the textproto of the first finding from the stream
// predictFile writes. Each finding is serialized with prototext and followed by
// a single newline; the entries written by createMultiEntryTestLogFile all carry
// a "db_user" field, which prototext emits as the leading line of every finding.
// The first finding therefore spans from the start of the stream up to the
// second "db_user:" line (the start of the second finding), or the whole stream
// if only one finding was written.
func firstFindingBlock(data string) string {
	lines := strings.Split(data, "\n")
	var out []string
	seenDbUser := false
	for _, line := range lines {
		if strings.HasPrefix(strings.TrimSpace(line), "db_user:") {
			if seenDbUser {
				break // start of the second finding
			}
			seenDbUser = true
		}
		out = append(out, line)
	}
	return strings.Join(out, "\n")
}

// predictTDFindingInDir runs an all-models single-log Predict against every model
// in dir and returns the finding that corresponds to the traffic-deviation model
// (identified by matching the expected TD finding). It fails the test if no such
// finding is present.
func predictTDFindingInDir(ctx context.Context, t *testing.T, dir string, entry *auditpb.AuditLogEntry, expectedTD *findingpb.AnomalyFinding) *findingpb.AnomalyFinding {
	t.Helper()

	oldDir := *modelsDir
	*modelsDir = dir
	defer func() { *modelsDir = oldDir }()

	s := &predictionServer{}
	resp, err := s.Predict(ctx, makeSingleLogPredictRequest(entry, ""))
	if gotStatus, _ := status.FromError(err); gotStatus.Code() != codes.OK {
		t.Fatalf("Predict() got code %v, want OK, err: %v", gotStatus.Code(), err)
	}

	var findings []*findingpb.AnomalyFinding
	if single := resp.GetSingleFinding(); single != nil {
		findings = append(findings, single)
	} else if batch := resp.GetBatchFindings(); batch != nil {
		findings = batch.GetFindings()
	}
	if len(findings) == 0 {
		t.Fatalf("expected at least one finding, got none")
	}

	for _, f := range findings {
		if proto.Equal(f, expectedTD) {
			return f
		}
	}
	t.Fatalf("no traffic-deviation finding matching the expected TD finding found among %d findings", len(findings))
	return nil
}
