// Command client orchestrates Markov generation, splitting and histogram training.
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
	modelpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common/proto"
	trainingpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/training_service/proto"
)

type options struct {
	configPath, generatorBin, outputDir, serverAddr string
	numLogs, maxN                                   int
	trainFraction                                   float64
	startTimeMs                                     int64
}

func main() {
	var o options
	flag.StringVar(&o.configPath, "config_path", "v1/client/markov_generator_config.textproto", "Markov generator configuration.")
	flag.StringVar(&o.generatorBin, "generator_bin", "./v1/generator/src/generator", "Generator executable.")
	flag.StringVar(&o.outputDir, "output_dir", "output/markov-histogram", "Directory for generated logs, split logs and trained model exports.")
	flag.StringVar(&o.serverAddr, "server_addr", "localhost:50051", "Training service address.")
	flag.IntVar(&o.numLogs, "num_logs", 10000, "Number of actions to generate.")
	flag.IntVar(&o.maxN, "max_n", 3, "Count n-grams of sizes 1 through this value (1-3).")
	flag.Float64Var(&o.trainFraction, "train_fraction", 0.4, "Ordered prefix used for training, in (0, 1].")
	flag.Int64Var(&o.startTimeMs, "start_time_ms", 0, "Simulated initial timestamp in milliseconds.")
	flag.Parse()
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	if err := run(ctx, o); err != nil {
		log.Fatal(err)
	}
}

func run(ctx context.Context, o options) error {
	if o.numLogs <= 0 || o.maxN < 1 || o.maxN > 3 || math.IsNaN(o.trainFraction) || o.trainFraction <= 0 || o.trainFraction > 1 {
		return fmt.Errorf("require num_logs > 0, max_n in [1, 3] and train_fraction in (0, 1]")
	}
	trainCount := int(math.Floor(float64(o.numLogs) * o.trainFraction))
	if trainCount == 0 {
		return fmt.Errorf("training prefix is empty")
	}
	configData, err := os.ReadFile(o.configPath)
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
	out, err := filepath.Abs(o.outputDir)
	if err != nil {
		return err
	}
	if err := os.MkdirAll(out, 0755); err != nil {
		return err
	}
	cmd := exec.CommandContext(ctx, o.generatorBin,
		"--config_file="+o.configPath, "--output_dir="+out,
		fmt.Sprintf("--num_logs=%d", o.numLogs), fmt.Sprintf("--start_time_ms=%d", o.startTimeMs))
	cmd.Stdout, cmd.Stderr = os.Stdout, os.Stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("generator: %w", err)
	}
	if err := splitLogs(ctx, out, trainCount, o.numLogs); err != nil {
		return err
	}
	log.Printf("Generated %d actions: %d training, %d held out", o.numLogs, trainCount, o.numLogs-trainCount)

	conn, err := grpc.NewClient(o.serverAddr, grpc.WithTransportCredentials(local.NewCredentials()))
	if err != nil {
		return err
	}
	defer conn.Close()
	configPB := &modelpb.NGramHistogramTrainingConfig{}
	configPB.SetMaxN(int32(o.maxN))
	trainingConfig := &trainingpb.ModelTrainingConfig{}
	trainingConfig.SetNgramHistogramConfig(configPB)
	req := &trainingpb.TrainModelsRequest{}
	req.SetTrainLogPath(filepath.Join(out, "train.binproto"))
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
	if !model.HasNgramHistogram() || model.GetMetadata().GetTotalLogEntriesProcessed() != int64(trainCount) {
		return fmt.Errorf("training returned an unexpected model or action count")
	}
	if err := os.WriteFile(filepath.Join(out, "histogram.bin"), modelBytes, 0644); err != nil {
		return err
	}
	if err := writeJSON(filepath.Join(out, "histogram.json"), model); err != nil {
		return err
	}
	for _, d := range model.GetNgramHistogram().GetModelData().GetHistograms() {
		log.Printf("n=%d: %d windows, %d observed sequences", d.GetN(), d.GetTotalWindows(), len(d.GetEntries()))
	}
	log.Printf("Generation and training outputs: %s", out)
	return nil
}

func writeJSON(path string, message proto.Message) error {
	data, err := (protojson.MarshalOptions{Indent: "  ", UseProtoNames: true, EmitUnpopulated: true}).Marshal(message)
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(data, '\n'), 0644)
}
