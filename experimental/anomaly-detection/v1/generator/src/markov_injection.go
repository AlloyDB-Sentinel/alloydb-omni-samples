package main

import (
	"fmt"
	"io"
	"math"
	"math/rand"

	"google.golang.org/protobuf/proto"

	genpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v1/generator/proto"
)

func graphWithProfile(base *genpb.MarkovSamplingConfig, profile *genpb.MarkovAnomalyProfile) *genpb.MarkovSamplingConfig {
	graph := proto.Clone(base).(*genpb.MarkovSamplingConfig)
	graph.SetAnomalyProfiles(nil)
	graph.SetInjections(nil)
	nodes := map[string]*genpb.MarkovNode{}
	for _, node := range graph.GetNodes() {
		nodes[node.GetId()] = node
	}
	for _, change := range profile.GetTransitions() {
		nodes[change.GetNodeId()].SetEdges(proto.Clone(change).(*genpb.MarkovTransitionOverride).GetEdges())
	}
	return graph
}

func validateMarkovProfiles(cfg *genpb.MarkovSamplingConfig, nodes map[string]*genpb.MarkovNode) error {
	profiles := map[string]bool{}
	for _, profile := range cfg.GetAnomalyProfiles() {
		id := profile.GetProfileId()
		if id == "" || id == "normal" || profiles[id] {
			return fmt.Errorf("markov_sampling: anomaly profile IDs must be unique, nonempty and not 'normal'")
		}
		profiles[id] = true
		seen := map[string]bool{}
		changed := false
		for _, override := range profile.GetTransitions() {
			node := nodes[override.GetNodeId()]
			if node == nil || seen[override.GetNodeId()] || len(node.GetEdges()) == 0 {
				return fmt.Errorf("profile %q: override requires a unique, nonterminal existing node", id)
			}
			seen[override.GetNodeId()] = true
			original := map[string]float64{}
			for _, e := range node.GetEdges() {
				original[e.GetDestinationId()] = e.GetWeight()
			}
			if len(override.GetEdges()) != len(original) {
				return fmt.Errorf("profile %q: must preserve all existing edge destinations", id)
			}
			destinations := map[string]bool{}
			for _, e := range override.GetEdges() {
				weight, exists := original[e.GetDestinationId()]
				if !exists || destinations[e.GetDestinationId()] {
					return fmt.Errorf("profile %q: must preserve unique existing edge destinations", id)
				}
				destinations[e.GetDestinationId()] = true
				if e.GetWeight() != weight {
					changed = true
				}
			}
		}
		if !changed {
			return fmt.Errorf("profile %q: must change at least one transition probability", id)
		}
		// Reuse graph validation for finite weights, normalization and references.
		if err := validateMarkovSampling(graphWithProfile(cfg, profile)); err != nil {
			return fmt.Errorf("profile %q: %w", id, err)
		}
	}
	segments := map[string]bool{}
	var previousEnd int64
	for i, injection := range cfg.GetInjections() {
		if injection.GetSegmentId() == "" || segments[injection.GetSegmentId()] {
			return fmt.Errorf("injections require unique nonempty segment_id values")
		}
		segments[injection.GetSegmentId()] = true
		if !profiles[injection.GetProfileId()] {
			return fmt.Errorf("injection references undefined profile %q", injection.GetProfileId())
		}
		start, length := injection.GetStartAction(), injection.GetNumActions()
		if start < 0 || length <= 0 || start > math.MaxInt64-length || (i > 0 && start < previousEnd) {
			return fmt.Errorf("injections must be ordered, nonoverlapping ranges with nonnegative starts and positive lengths")
		}
		previousEnd = start + length
	}
	return nil
}

func validateMarkovInjectionBounds(cfg *genpb.MarkovSamplingConfig, total int) error {
	for _, injection := range cfg.GetInjections() {
		if injection.GetStartAction() < 0 || injection.GetNumActions() <= 0 || injection.GetNumActions() > int64(total) || injection.GetStartAction() > int64(total)-injection.GetNumActions() {
			return fmt.Errorf("injection %q exceeds num_logs=%d", injection.GetSegmentId(), total)
		}
	}
	return nil
}

type markovCursor struct {
	walker *markovWalker
	nextID string
}

func (c *markovCursor) emitNext(clock int64, out io.Writer) (int64, error) {
	w := c.walker
	if c.nextID == "" {
		c.nextID = w.selectStartID()
	}
	node := w.graph.nodes[c.nextID]
	w.clockMs = clock
	if err := w.emit(node, out); err != nil {
		return clock, err
	}
	clock += sampleCompletionMs(node.node.GetCompletionTime(), w.rng)
	if len(node.destIDs) == 0 {
		c.nextID = ""
	} else {
		c.nextID = w.selectNextID(node)
	}
	return clock, nil
}

// The normal cursor is paused during each injection. An anomaly starts a fresh
// walk at its configured start; terminal nodes restart within the segment.
// All cursors share one seeded RNG and one nondecreasing simulated clock.
func generateLogsMarkovInjected(cfg *genpb.MarkovSamplingConfig, total int, start int64, rng *rand.Rand, out io.Writer) error {
	if err := validateMarkovInjectionBounds(cfg, total); err != nil {
		return err
	}
	profiles := map[string]*markovGraph{}
	for _, profile := range cfg.GetAnomalyProfiles() {
		profiles[profile.GetProfileId()] = newMarkovGraph(graphWithProfile(cfg, profile))
	}
	newCursor := func(graph *markovGraph, profile, segment string) *markovCursor {
		return &markovCursor{walker: &markovWalker{graph: graph, generator: NewLogGenerator(rng), rng: rng, labelFromInjection: true, groundTruthProfile: profile, groundTruthSegmentID: segment}}
	}
	normal := newCursor(newMarkovGraph(cfg), "normal", "")
	var anomalous *markovCursor
	segmentIndex := 0
	clock := start
	for position := 0; position < total; position++ {
		if segmentIndex < len(cfg.GetInjections()) {
			segment := cfg.GetInjections()[segmentIndex]
			if int64(position) >= segment.GetStartAction()+segment.GetNumActions() {
				segmentIndex++
				anomalous = nil
			}
		}
		active := normal
		if segmentIndex < len(cfg.GetInjections()) {
			segment := cfg.GetInjections()[segmentIndex]
			if int64(position) >= segment.GetStartAction() {
				if anomalous == nil {
					anomalous = newCursor(profiles[segment.GetProfileId()], segment.GetProfileId(), segment.GetSegmentId())
				}
				active = anomalous
			}
		}
		var err error
		clock, err = active.emitNext(clock, out)
		if err != nil {
			return err
		}
	}
	return nil
}
