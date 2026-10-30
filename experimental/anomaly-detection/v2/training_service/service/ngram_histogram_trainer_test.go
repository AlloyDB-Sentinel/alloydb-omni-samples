package service

import (
	"encoding/json"
	"errors"
	"math"
	"math/rand"
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common"
	"google.golang.org/protobuf/proto"

	modelpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common/proto"
)

// Compare streaming statistics with explicit batch vectors and a two-pass
// mean/variance calculation. This reference includes zeros rather than merging
// sparse observations, and excludes boundary-crossing and trailing n-grams.
func TestNGramWindowStatisticsAgainstBatchReference(t *testing.T) {
	rng := rand.New(rand.NewSource(20261030))
	checked := 0
	for _, windowLength := range []int{3, 4, 300} {
		for _, length := range []int{0, 1, 2, 3, 4, 6, 7, 299, 300, 301, 600, 601, 900, 901} {
			actions := make([]string, length)
			trainer, err := NewNGramHistogramTrainer(3, windowLength)
			if err != nil {
				t.Fatal(err)
			}
			for i := range actions {
				actions[i] = []string{"A", "B", "C"}[rng.Intn(3)]
				// A late category exercises zero observations before its debut,
				// including categories seen only in a trailing partial batch.
				if i >= length-2 {
					actions[i] = "LATE"
				}
				trainer.Count(common.ExtractedLogFeatures{QueryTemplate: actions[i]})
			}
			data := trainer.BuildModel().GetNgramHistogram().GetModelData()
			batches := length / windowLength
			if data.GetCompleteWindows() != int64(batches) || data.GetTrailingActions() != int64(length%windowLength) {
				t.Fatal("incorrect window metadata")
			}
			for _, d := range data.GetHistograms() {
				n := int(d.GetN())
				for _, entry := range d.GetEntries() {
					if batches == 0 {
						if entry.HasWindowStatistics() {
							t.Fatal("statistics from incomplete batch")
						}
						continue
					}
					proportions := make([]float64, batches)
					for batch := range proportions {
						for start := batch * windowLength; start+n <= (batch+1)*windowLength; start++ {
							if reflect.DeepEqual(actions[start:start+n], entry.GetActions()) {
								proportions[batch]++
							}
						}
						proportions[batch] /= float64(windowLength - n + 1)
					}
					var mean float64
					for _, v := range proportions {
						mean += v
					}
					mean /= float64(batches)
					stats := entry.GetWindowStatistics()
					if stats == nil || math.Abs(stats.GetMean()-mean) > 1e-12 {
						t.Fatalf("length=%d W=%d n=%d actions=%v: mean=%v expected=%g", length, windowLength, n, entry.GetActions(), stats, mean)
					}
					if stats.HasVariance() != (batches > 1) {
						t.Fatal("incorrect variance presence")
					}
					if batches > 1 {
						var variance float64
						for _, v := range proportions {
							variance += (v - mean) * (v - mean)
						}
						variance /= float64(batches - 1)
						if math.Abs(stats.GetVariance()-variance) > 1e-12 {
							t.Fatalf("length=%d W=%d n=%d actions=%v: variance=%g expected=%g", length, windowLength, n, entry.GetActions(), stats.GetVariance(), variance)
						}
					}
					checked++
				}
			}
		}
	}
	t.Logf("Verified %d category statistics against explicit batch vectors", checked)
}

// This reference counts complete slices from left to right, independently of
// the trainer's streaming suffix counter and fixed-array keys. Check every
// sequence over three categories through length six, plus real window boundaries.
func TestNGramCountsAgainstIndependentScan(t *testing.T) {
	checked := 0
	check := func(actions []string, windowLength, maxN int) {
		t.Helper()
		trainer, err := NewNGramHistogramTrainer(maxN, windowLength)
		if err != nil {
			t.Fatal(err)
		}
		for i, action := range actions {
			trainer.Count(common.ExtractedLogFeatures{QueryTemplate: action})
			if i == len(actions)/2 {
				trainer.BuildModel()
			} // A snapshot must not reset counting.
		}
		model := trainer.BuildModel()
		if model.GetMetadata().GetTotalLogEntriesProcessed() != int64(len(actions)) {
			t.Fatal("incorrect total action count")
		}
		for _, d := range model.GetNgramHistogram().GetModelData().GetHistograms() {
			n := int(d.GetN())
			want := map[string]int64{}
			for start := 0; start+n <= len(actions); start++ {
				key, err := json.Marshal(actions[start : start+n])
				if err != nil {
					t.Fatal(err)
				}
				want[string(key)]++
			}
			got := map[string]int64{}
			var total int64
			var probabilitySum float64
			for _, entry := range d.GetEntries() {
				key, err := json.Marshal(entry.GetActions())
				if err != nil {
					t.Fatal(err)
				}
				if _, exists := got[string(key)]; exists {
					t.Fatal("duplicate output category")
				}
				got[string(key)] = entry.GetCount()
				total += entry.GetCount()
				if entry.GetProbability() != float64(want[string(key)])/float64(d.GetTotalWindows()) {
					t.Fatalf("incorrect probability: %v", entry)
				}
				probabilitySum += entry.GetProbability()
			}
			expectedTotal := len(actions) - n + 1
			if expectedTotal < 0 {
				expectedTotal = 0
			}
			if !reflect.DeepEqual(got, want) || total != int64(expectedTotal) || d.GetTotalWindows() != int64(expectedTotal) {
				t.Fatalf("length=%d window_length=%d n=%d: got=%v want=%v total=%d expected=%d", len(actions), windowLength, n, got, want, total, expectedTotal)
			}
			if expectedTotal > 0 && math.Abs(probabilitySum-1) > 1e-12 {
				t.Fatal("probabilities do not sum to one")
			}
		}
		checked++
	}
	var enumerate func([]string)
	enumerate = func(actions []string) {
		for _, windowLength := range []int{3, 4, 300} {
			for maxN := 1; maxN <= 3; maxN++ {
				check(actions, windowLength, maxN)
			}
		}
		if len(actions) == 6 {
			return
		}
		for _, action := range []string{"A", "B", "C"} {
			enumerate(append(actions, action))
		}
	}
	enumerate(nil)
	// Delimiters and SQL placeholders are part of an action, not token separators.
	alphabet := []string{"SELECT * FROM t WHERE id = %v", "INSERT INTO t VALUES (%v)", "a|b", "c", "a", "b|c", "a\x00b", "SELECT\n1"}
	for _, length := range []int{299, 300, 301, 599, 600, 601} {
		actions := make([]string, length)
		for i := range actions {
			actions[i] = alphabet[(i*i+3*i+i/11)%len(alphabet)]
		}
		for _, windowLength := range []int{3, 300} {
			for maxN := 1; maxN <= 3; maxN++ {
				check(actions, windowLength, maxN)
			}
		}
	}
	t.Logf("Verified %d sequence/configuration combinations against an independent count", checked)
}

func TestNGramHistogramCounts(t *testing.T) {
	trainer, err := NewNGramHistogramTrainer(3, 300)
	if err != nil {
		t.Fatal(err)
	}
	for i, action := range []string{"A", "B", "A", "B"} {
		trainer.Count(common.ExtractedLogFeatures{QueryTemplate: action, TimestampMs: int64(100 + i)})
	}
	model := trainer.BuildModel()
	want := []map[string]int64{{"A": 2, "B": 2}, {"AB": 2, "BA": 1}, {"ABA": 1, "BAB": 1}}
	for i, d := range model.GetNgramHistogram().GetModelData().GetHistograms() {
		if d.GetTotalWindows() != int64(4-i) {
			t.Fatalf("n=%d: windows=%d", i+1, d.GetTotalWindows())
		}
		got := map[string]int64{}
		var sum float64
		for _, e := range d.GetEntries() {
			key := ""
			for _, a := range e.GetActions() {
				key += a
			}
			got[key] = e.GetCount()
			if e.GetProbability() != float64(e.GetCount())/float64(d.GetTotalWindows()) {
				t.Fatal("wrong probability")
			}
			sum += e.GetProbability()
		}
		if !reflect.DeepEqual(got, want[i]) || math.Abs(sum-1) > 1e-12 {
			t.Fatalf("n=%d: counts=%v probability sum=%g", i+1, got, sum)
		}
	}
	if model.GetMetadata().GetTotalLogEntriesProcessed() != 4 || model.GetMetadata().GetDataStartTime().AsTime().UnixMilli() != 100 || model.GetMetadata().GetDataEndTime().AsTime().UnixMilli() != 103 {
		t.Fatal("incorrect metadata")
	}
	data, err := proto.Marshal(model)
	if err != nil {
		t.Fatal(err)
	}
	loaded := &modelpb.Model{}
	if err := proto.Unmarshal(data, loaded); err != nil {
		t.Fatal(err)
	}
	if !proto.Equal(model, loaded) {
		t.Fatal("model round trip changed data")
	}
	before := proto.Clone(model)
	trainer.Count(common.ExtractedLogFeatures{QueryTemplate: "C"})
	if !proto.Equal(model, before) {
		t.Fatal("count mutated previously built model")
	}
}

func TestNGramHistogramShortAndEmpty(t *testing.T) {
	for size := 0; size < 3; size++ {
		trainer, _ := NewNGramHistogramTrainer(3, 300)
		for i := 0; i < size; i++ {
			trainer.Count(common.ExtractedLogFeatures{QueryTemplate: "A"})
		}
		for _, d := range trainer.BuildModel().GetNgramHistogram().GetModelData().GetHistograms() {
			want := int64(size) - int64(d.GetN()) + 1
			if want < 0 {
				want = 0
			}
			if d.GetTotalWindows() != want {
				t.Fatal("incorrect short sequence window count")
			}
			if want == 0 && len(d.GetEntries()) != 0 {
				t.Fatal("unexpected entries")
			}
		}
	}
	for _, n := range []int{-1, 0, 4} {
		if _, err := NewNGramHistogramTrainer(n, 300); !errors.Is(err, ErrInvalidArgument) {
			t.Fatalf("n=%d: expected invalid argument, got %v", n, err)
		}
	}
}

func TestNGramHistogramActionBoundaries(t *testing.T) {
	trainer, _ := NewNGramHistogramTrainer(2, 300)
	for _, a := range []string{"a|b", "c", "a", "b|c"} {
		trainer.Count(common.ExtractedLogFeatures{QueryTemplate: a})
	}
	d := trainer.BuildModel().GetNgramHistogram().GetModelData().GetHistograms()[1]
	if len(d.GetEntries()) != 3 {
		t.Fatal("distinct action sequences collided")
	}
	for _, e := range d.GetEntries() {
		if e.GetCount() != 1 {
			t.Fatal("distinct pairs merged")
		}
	}
}

func TestNGramWindowStatistics(t *testing.T) {
	trainer, err := NewNGramHistogramTrainer(3, 3)
	if err != nil {
		t.Fatal(err)
	}
	// Two complete windows AAA / ABB, followed by a trailing C.
	// Global n-grams cross window boundaries; window statistics must not.
	for _, action := range []string{"A", "A", "A", "A", "B", "B", "C"} {
		trainer.Count(common.ExtractedLogFeatures{QueryTemplate: action})
	}
	model := trainer.BuildModel()
	data := model.GetNgramHistogram().GetModelData()
	if data.GetCompleteWindows() != 2 || data.GetTrailingActions() != 1 || model.GetMetadata().GetTotalLogEntriesProcessed() != 7 {
		t.Fatal("incorrect action/window counts")
	}
	want := []map[string][2]float64{
		{"A": {2.0 / 3, 2.0 / 9}, "B": {1.0 / 3, 2.0 / 9}, "C": {0, 0}},
		{"AA": {0.5, 0.5}, "AB": {0.25, 0.125}, "BB": {0.25, 0.125}, "BC": {0, 0}},
		{"AAA": {0.5, 0.5}, "AAB": {0, 0}, "ABB": {0.5, 0.5}, "BBC": {0, 0}},
	}
	for i, d := range data.GetHistograms() {
		if len(d.GetEntries()) != len(want[i]) {
			t.Fatalf("n=%d: wrong category count", i+1)
		}
		var meanSum float64
		for _, e := range d.GetEntries() {
			key := ""
			for _, a := range e.GetActions() {
				key += a
			}
			expected, ok := want[i][key]
			if !ok {
				t.Fatalf("unexpected key %q", key)
			}
			s := e.GetWindowStatistics()
			if s == nil || !s.HasVariance() || math.Abs(s.GetMean()-expected[0]) > 1e-12 || math.Abs(s.GetVariance()-expected[1]) > 1e-12 {
				t.Fatalf("%q: got %v want %v", key, s, expected)
			}
			meanSum += s.GetMean()
		}
		if math.Abs(meanSum-1) > 1e-12 {
			t.Fatal("mean proportions do not sum to one")
		}
	}
	encoded, err := proto.Marshal(model)
	if err != nil {
		t.Fatal(err)
	}
	loaded := &modelpb.Model{}
	if err := proto.Unmarshal(encoded, loaded); err != nil {
		t.Fatal(err)
	}
	if !proto.Equal(model, loaded) {
		t.Fatal("statistics changed after persistence")
	}
	// BuildModel is a snapshot, and must not flush the incomplete window.
	again := trainer.BuildModel()
	if !proto.Equal(data, again.GetNgramHistogram().GetModelData()) {
		t.Fatal("BuildModel mutated statistics")
	}
	trainer.Count(common.ExtractedLogFeatures{QueryTemplate: "C"})
	trainer.Count(common.ExtractedLogFeatures{QueryTemplate: "C"})
	if trainer.BuildModel().GetNgramHistogram().GetModelData().GetCompleteWindows() != 3 {
		t.Fatal("count after snapshot lost trailing actions")
	}
	if !proto.Equal(model, loaded) {
		t.Fatal("count mutated saved snapshot")
	}
}

func TestNGramStatisticsInsufficientWindows(t *testing.T) {
	trainer, _ := NewNGramHistogramTrainer(1, 3)
	trainer.Count(common.ExtractedLogFeatures{QueryTemplate: "A"})
	entry := trainer.BuildModel().GetNgramHistogram().GetModelData().GetHistograms()[0].GetEntries()[0]
	if entry.HasWindowStatistics() {
		t.Fatal("partial window must not produce statistics")
	}
	trainer.Count(common.ExtractedLogFeatures{QueryTemplate: "A"})
	trainer.Count(common.ExtractedLogFeatures{QueryTemplate: "A"})
	entry = trainer.BuildModel().GetNgramHistogram().GetModelData().GetHistograms()[0].GetEntries()[0]
	if entry.GetWindowStatistics().GetMean() != 1 || entry.GetWindowStatistics().HasVariance() {
		t.Fatal("one window has a mean but no estimated variance")
	}
	for _, length := range []int{-1, 0, 2} {
		if _, err := NewNGramHistogramTrainer(3, length); !errors.Is(err, ErrInvalidArgument) {
			t.Fatalf("invalid length %d: %v", length, err)
		}
	}
}
