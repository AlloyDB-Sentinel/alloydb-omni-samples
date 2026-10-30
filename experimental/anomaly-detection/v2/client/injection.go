package main

import (
	"fmt"
	"os"
	"path/filepath"

	"google.golang.org/protobuf/encoding/prototext"
	"google.golang.org/protobuf/proto"

	genpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v1/generator/proto"
	clientpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/client/proto"
)

func validateExperimentInjections(config *clientpb.MarkovExperimentConfig) error {
	profiles := map[string]bool{}
	for _, profile := range config.GetAnomalyProfiles() {
		if profile.GetProfileId() == "" || profile.GetProfileId() == "normal" || profiles[profile.GetProfileId()] {
			return fmt.Errorf("anomaly_profiles require unique nonempty profile IDs, excluding 'normal'")
		}
		profiles[profile.GetProfileId()] = true
	}
	segments := map[string]bool{}
	var end int64
	for _, segment := range config.GetInjections() {
		if segment.GetSegmentId() == "" || segments[segment.GetSegmentId()] || !profiles[segment.GetProfileId()] {
			return fmt.Errorf("injections require unique segment IDs and defined profiles")
		}
		segments[segment.GetSegmentId()] = true
		if segment.GetStartAction() < end || segment.GetNumActions() <= 0 || segment.GetNumActions() > config.GetValidationActions() || segment.GetStartAction() > config.GetValidationActions()-segment.GetNumActions() {
			return fmt.Errorf("injections must be ordered, nonoverlapping and fit validation_actions")
		}
		end = segment.GetStartAction() + segment.GetNumActions()
	}
	return nil
}

func validationGeneratorConfig(normal *genpb.QueryTemplateConfigs, experiment *clientpb.MarkovExperimentConfig) *genpb.QueryTemplateConfigs {
	config := proto.Clone(normal).(*genpb.QueryTemplateConfigs)
	config.GetMarkovSampling().SetAnomalyProfiles(experiment.GetAnomalyProfiles())
	config.GetMarkovSampling().SetInjections(experiment.GetInjections())
	return config
}

func writeValidationGeneratorConfig(out, dir string, experiment *clientpb.MarkovExperimentConfig) (string, error) {
	data, err := os.ReadFile(filepath.Join(out, "generator_config.textproto"))
	if err != nil {
		return "", err
	}
	normal := &genpb.QueryTemplateConfigs{}
	if err := prototext.Unmarshal(data, normal); err != nil {
		return "", err
	}
	if err := os.MkdirAll(dir, 0755); err != nil {
		return "", err
	}
	path := filepath.Join(dir, "generator_config.textproto")
	return path, os.WriteFile(path, []byte(prototext.Format(validationGeneratorConfig(normal, experiment))), 0644)
}
