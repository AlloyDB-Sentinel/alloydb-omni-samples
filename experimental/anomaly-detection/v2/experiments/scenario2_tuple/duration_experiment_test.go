package main

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
	"time"
)

// TestControlledDurationSweep reproduces the duration comparison using the
// runner's aggregation and scoring. Opt in because it generates ten long logs.
func TestControlledDurationSweep(t *testing.T) {
	output := os.Getenv("SCENARIO2_DURATION_OUTPUT")
	if output == "" {
		t.Skip("set SCENARIO2_DURATION_OUTPUT to a scratch directory to reproduce the duration experiment")
	}
	if !filepath.IsAbs(output) {
		t.Fatal("SCENARIO2_DURATION_OUTPUT must be an absolute path")
	}
	if err := os.MkdirAll(output, 0755); err != nil {
		t.Fatal(err)
	}
	t.Logf("Results: %s", output)
	moduleRoot, err := filepath.Abs(filepath.Join("..", "..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	config := filepath.Join(moduleRoot, "v1", "client", "query_template_config_s2.textproto")
	temp := t.TempDir()
	generator := filepath.Join(temp, "generator")
	if runtime.GOOS == "windows" {
		generator += ".exe"
	}
	build := exec.Command("go", "build", "-o", generator, "./v1/generator/src")
	build.Dir = moduleRoot
	if log, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build generator: %v\n%s", err, log)
	}

	// Restore the CLI schedule so this opt-in run can share a test process.
	savedTrainingWeeks, savedTotalWeeks := trainingWeeks, totalWeeks
	savedTrainingStart, savedTrainingEnd := trainingStart, trainingEnd
	savedValidationEnd, savedTotalBuckets := validationEnd, totalBuckets
	t.Cleanup(func() {
		trainingWeeks, totalWeeks = savedTrainingWeeks, savedTotalWeeks
		trainingStart, trainingEnd = savedTrainingStart, savedTrainingEnd
		validationEnd, totalBuckets = savedValidationEnd, savedTotalBuckets
	})
	const historyWeeks = 50
	trainingWeeks = historyWeeks
	totalWeeks = warmupWeeks + historyWeeks + validationWeeks + testWeeks
	trainingStart = warmupWeeks * bucketsPerWeek
	trainingEnd = (warmupWeeks + historyWeeks) * bucketsPerWeek
	validationEnd = trainingEnd + validationWeeks*bucketsPerWeek
	totalBuckets = totalWeeks * bucketsPerWeek

	anchor := time.Date(2026, time.January, 5, 0, 0, 0, 0, time.UTC)
	start := anchor.Add(-26 * week)
	trainEnd := anchor.Add(25 * week)
	end := trainEnd.Add(time.Duration(validationWeeks+testWeeks) * week)
	datasets := make([]dataset, seedCount)
	for i := range datasets {
		seed := firstSeed + i
		datasets[i] = generateDurationDataset(t, generator, config, temp, seed, start, anchor, trainEnd, end)
		if err := validateDataset(datasets[i]); err != nil {
			t.Fatalf("seed %d: %v", seed, err)
		}
		for _, segment := range []string{"earlier", "original"} {
			if err := os.RemoveAll(filepath.Join(temp, segment)); err != nil {
				t.Fatal(err)
			}
		}
		t.Logf("Aggregated seed %d (%d/%d)", seed, i+1, seedCount)
	}

	for _, weeks := range []int{1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 30, 40, 50} {
		trainingWeeks = weeks
		trainingStart = trainingEnd - weeks*bucketsPerWeek
		frameResults, subsets, err := evaluate(datasets)
		if err != nil {
			t.Fatalf("%d weeks: %v", weeks, err)
		}
		dir := filepath.Join(output, fmt.Sprintf("weeks-%02d", weeks))
		if err := os.MkdirAll(dir, 0755); err != nil {
			t.Fatal(err)
		}
		if err := writeFrameCSV(filepath.Join(dir, "frame_results.csv"), frameResults); err != nil {
			t.Fatal(err)
		}
		if err := writeFrameSweepCSV(filepath.Join(dir, "frame_sweep.csv"), frameResults); err != nil {
			t.Fatal(err)
		}
		if err := writeSubsetCSV(filepath.Join(dir, "subset_results.csv"), subsets, frameResults); err != nil {
			t.Fatal(err)
		}
		if weeks == 20 || weeks == 24 {
			for _, name := range []string{"frame_results.csv", "frame_sweep.csv", "subset_results.csv"} {
				got, err := os.ReadFile(filepath.Join(dir, name))
				if err != nil {
					t.Fatal(err)
				}
				want, err := os.ReadFile(filepath.Join("results", "controlled-duration", fmt.Sprintf("weeks-%02d", weeks), name))
				if err != nil {
					t.Fatal(err)
				}
				if !bytes.Equal(got, want) {
					t.Fatalf("%d-week replay differs from checked-in %s; results are in %s", weeks, name, dir)
				}
			}
		}
		t.Logf("Scored %d weeks: %s (%s)", weeks, subsetName(sortedSubsets(subsets)[0].members), sortedSubsets(subsets)[0].combination)
	}
}

func generateDurationDataset(t *testing.T, generator, config, dir string, seed int, start, anchor, trainEnd, end time.Time) dataset {
	t.Helper()
	var readers []io.Reader
	// Prepend independent clean history without changing the original 24-week
	// run's random draws. Every duration then uses exactly the same evaluation.
	for _, segment := range []struct {
		name                 string
		seed                 int
		start, trainEnd, end time.Time
	}{
		{"earlier", seed + 3000, start, anchor, anchor},
		{"original", seed, anchor, trainEnd, end},
	} {
		segmentDir := filepath.Join(dir, segment.name)
		if err := generate(generator, config, segmentDir, segment.seed, segment.start, segment.trainEnd, segment.end); err != nil {
			t.Fatal(err)
		}
		file, err := os.Open(filepath.Join(segmentDir, "logs.binproto"))
		if err != nil {
			t.Fatal(err)
		}
		defer file.Close()
		readers = append(readers, file)
	}
	data, err := aggregate(io.MultiReader(readers...), start)
	if err != nil {
		t.Fatalf("aggregate seed %d: %v", seed, err)
	}
	return data
}
