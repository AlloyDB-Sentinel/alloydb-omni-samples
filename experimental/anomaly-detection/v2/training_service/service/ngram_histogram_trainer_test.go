package service

import (
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common"
	"google.golang.org/protobuf/proto"

	modelpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common/proto"
)

func TestNGramHistogramCounts(t *testing.T) {
	trainer, err := NewNGramHistogramTrainer(3)
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
		trainer, _ := NewNGramHistogramTrainer(3)
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
		if _, err := NewNGramHistogramTrainer(n); !errors.Is(err, ErrInvalidArgument) {
			t.Fatalf("n=%d: expected invalid argument, got %v", n, err)
		}
	}
}

func TestNGramHistogramActionBoundaries(t *testing.T) {
	trainer, _ := NewNGramHistogramTrainer(2)
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
