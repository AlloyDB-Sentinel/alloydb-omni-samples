package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestExperimentConfig(t *testing.T) {
	config, err := readExperiment("configs/standard_user.textproto")
	if err != nil {
		t.Fatal(err)
	}
	if config.GetTrainingActions() != 30000 || config.GetTrainingConfig().GetWindowLength() != 300 || config.GetTrainingConfig().GetMaxN() != 3 || config.GetCalibrationPercentile() != 0.99 {
		t.Fatal("unexpected experiment defaults")
	}
	valid := `generator_config_path:"normal.textproto" training_actions:30000 training_seed:42 training_config:{max_n:3 window_length:300} calibration_percentile:0.99`
	for _, test := range []struct {
		name, contents string
		valid          bool
	}{
		{"valid", valid, true},
		{"legacy_default", strings.Replace(valid, "window_length:300", "", 1), true},
		{"missing_training", `generator_config_path:"normal" training_actions:30000 calibration_percentile:0.99`, false},
		{"zero_count", strings.Replace(valid, "training_actions:30000", "training_actions:0", 1), false},
		{"short_training", strings.Replace(valid, "training_actions:30000", "training_actions:299", 1), false},
		{"invalid_n", strings.Replace(valid, "max_n:3", "max_n:4", 1), false},
		{"zero_window", strings.Replace(valid, "window_length:300", "window_length:0", 1), false},
		{"short_window", strings.Replace(valid, "window_length:300", "window_length:2", 1), false},
		{"invalid_percentile", strings.Replace(valid, "0.99", "1.0", 1), false},
		{"nan_percentile", strings.Replace(valid, "0.99", "nan", 1), false},
		{"missing_graph", strings.Replace(valid, `generator_config_path:"normal.textproto"`, "", 1), false},
		{"short_calibration", valid + ` calibration_actions:300`, false},
		{"short_validation", valid + ` validation_actions:299`, false},
		{"reused_training_seed", valid + ` calibration_seed:42`, false},
		{"reused_evaluation_seed", valid + ` validation_seed:43`, false},
		{"single_training_window", strings.Replace(valid, "training_actions:30000", "training_actions:300", 1), false},
		{"zero_floor", valid + ` scoring_config:{standard_deviation_floor_count:0}`, false},
		{"negative_floor", valid + ` scoring_config:{standard_deviation_floor_count:-1}`, false},
		{"nan_floor", valid + ` scoring_config:{standard_deviation_floor_count:nan}`, false},
		{"custom_floor", valid + ` scoring_config:{standard_deviation_floor_count:2}`, true},
	} {
		t.Run(test.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "experiment.textproto")
			if err := os.WriteFile(path, []byte(test.contents), 0644); err != nil {
				t.Fatal(err)
			}
			_, err := readExperiment(path)
			if (err == nil) != test.valid {
				t.Fatalf("valid=%v err=%v", test.valid, err)
			}
		})
	}
}
