package main

import (
	"math"
	"testing"
)

func TestLimitsForLearnsPerWeeklySlot(t *testing.T) {
	training := make([]int64, 2*slotsPerWeek)
	for i := range training {
		training[i] = int64(100*(i%slotsPerWeek+1) + 20*(i/slotsPerWeek))
	}
	for slot, limit := range limitsFor(training, 5) {
		if want := float64(100*(slot+1) + 60); math.Abs(limit-want) > 1e-9 {
			t.Fatalf("slot %d: got %v, want %v", slot, limit, want)
		}
	}
}

func TestScoreSubsetUsesORAndAND(t *testing.T) {
	labels := []bool{true, true, false, false}
	a := []int64{10, 0, 0, 0}
	b := []int64{0, 10, 0, 0}
	data := dataset{peaks: [][]int64{a, b}, labels: labels}
	limits := make([][slotsPerWeek]float64, 2)
	for i := range limits {
		for j := range limits[i] {
			limits[i][j] = 5
		}
	}

	if got, want := scoreFrame(a, labels, limits[0], 0, len(labels)), (confusion{tp: 1, fn: 1, tn: 2}); got != want {
		t.Fatalf("single frame = %+v, want %+v", got, want)
	}
	if got, want := scoreSubset(data, []int{0, 1}, limits, orCombination, 0, len(labels)), (confusion{tp: 2, tn: 2}); got != want {
		t.Fatalf("OR subset = %+v, want %+v", got, want)
	}
	if got, want := scoreSubset(data, []int{0, 1}, limits, andCombination, 0, len(labels)), (confusion{tn: 2, fn: 2}); got != want {
		t.Fatalf("AND subset = %+v, want %+v", got, want)
	}

	// Both members must fire for AND to produce a true positive.
	data.peaks = [][]int64{{10, 0}, {10, 0}}
	data.labels = []bool{true, false}
	if got, want := scoreSubset(data, []int{0, 1}, limits, andCombination, 0, 2), (confusion{tp: 1, tn: 1}); got != want {
		t.Fatalf("AND joint hit = %+v, want %+v", got, want)
	}
}

func TestSubsetSpecsCoversAllNonEmptySubsets(t *testing.T) {
	specs := subsetSpecs()
	if got, want := len(specs), 2*((1<<len(frames))-1); got != want {
		t.Fatalf("specs = %d, want %d", got, want)
	}

	type key struct {
		subset string
		comb   combiner
	}
	seen := make(map[key]int)
	bySize := make(map[int]int)
	for _, spec := range specs {
		if len(spec.members) == 0 {
			t.Fatal("empty subset")
		}
		seen[key{subsetName(spec.members), spec.combination}]++
		bySize[len(spec.members)]++
	}
	for mask := 1; mask < 1<<len(frames); mask++ {
		var members []int
		for fi := range frames {
			if mask&(1<<fi) != 0 {
				members = append(members, fi)
			}
		}
		for _, comb := range []combiner{orCombination, andCombination} {
			if got := seen[key{subsetName(members), comb}]; got != 1 {
				t.Fatalf("%s/%s seen %d times, want 1", subsetName(members), comb, got)
			}
		}
	}
	for size, want := range map[int]int{1: 14, 2: 42, 3: 70, 4: 70, 5: 42, 6: 14, 7: 2} {
		if got := bySize[size]; got != want {
			t.Fatalf("size %d: got %d candidates, want %d", size, got, want)
		}
	}

	if specs[0].combination != orCombination || len(specs[0].members) != 1 || specs[0].members[0] != 0 {
		t.Fatalf("first spec = %+v, want 1m/OR", specs[0])
	}
	if last := specs[len(specs)-1]; last.combination != andCombination || len(last.members) != len(frames) {
		t.Fatalf("last spec = %+v, want all frames/AND", last)
	}
}

func TestBetterPrefersF1ThenPrecisionThenAccuracy(t *testing.T) {
	base := metricSet{accuracy: .9, precision: .8, recall: .7, f1: .75}
	if !better(metricSet{accuracy: .5, precision: .1, recall: .9, f1: .8}, base) {
		t.Fatal("higher F1 must win")
	}
	if !better(metricSet{accuracy: .5, precision: .9, recall: .5, f1: .75}, base) {
		t.Fatal("equal F1 must prefer precision")
	}
	if !better(metricSet{accuracy: .95, precision: .8, recall: .5, f1: .75}, base) {
		t.Fatal("equal F1 and precision must prefer accuracy")
	}
	if better(base, base) {
		t.Fatal("equal metrics must keep the earlier candidate")
	}
}
