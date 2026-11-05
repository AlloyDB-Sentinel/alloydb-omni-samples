package main

import (
	"fmt"
	"io"
	"math"
	"math/rand"

	genpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v1/generator/proto"
)

// Anomaly injection splices anomalous bursts into an otherwise normal Markov walk.
//
// Each MarkovInjection names a half-open action-index range [start_action,
// start_action+num_actions) and references a SEPARATE, complete markov_sampling graph by
// anomaly_graph_path. That referenced graph is resolved, validated, and inlined by loadConfigs
// (resolveAnomalyGraphs, reusing resolveBaseGraphFile) before generation, exactly like the
// subtype/individual base graphs. During a segment the normal cursor is paused and an anomaly
// cursor walks the referenced full graph on its own start spec; the two cursors share one seeded
// rng and one nondecreasing clock. Injections do not chain: a referenced graph may not itself
// carry injections (enforced by resolveBaseGraphFile requiring a plain markov_sampling config).

// validateMarkovInjections checks the injection segments of a markov_sampling config. The
// referenced anomaly graphs are resolved and validated separately (resolveAnomalyGraphs in
// loadConfigs); this function enforces the segment-level invariants that need no file I/O:
//   - every segment_id is nonempty and unique
//   - every injection carries an anomaly_graph_path
//   - start_action >= 0, num_actions > 0, and start_action+num_actions does not overflow int64
//   - segments are ordered and non-overlapping (each starts at or after the previous one ends)
func validateMarkovInjections(cfg *genpb.MarkovSamplingConfig) error {
	segments := map[string]bool{}
	var previousEnd int64
	for i, injection := range cfg.GetInjections() {
		if injection.GetSegmentId() == "" || segments[injection.GetSegmentId()] {
			return fmt.Errorf("markov_sampling: injections require unique nonempty segment_id values")
		}
		segments[injection.GetSegmentId()] = true
		if injection.GetAnomalyGraphPath() == "" {
			return fmt.Errorf("markov_sampling: injection %q requires an anomaly_graph_path", injection.GetSegmentId())
		}
		start, length := injection.GetStartAction(), injection.GetNumActions()
		if start < 0 || length <= 0 || start > math.MaxInt64-length || (i > 0 && start < previousEnd) {
			return fmt.Errorf("markov_sampling: injections must be ordered, nonoverlapping ranges with nonnegative starts and positive lengths")
		}
		previousEnd = start + length
	}
	return nil
}

// validateMarkovInjectionBounds verifies every injection's action-index range fits within the
// total number of actions (--num_logs). Called both at flag-validation time and before generation.
func validateMarkovInjectionBounds(cfg *genpb.MarkovSamplingConfig, total int) error {
	for _, injection := range cfg.GetInjections() {
		if injection.GetStartAction() < 0 || injection.GetNumActions() <= 0 || injection.GetNumActions() > int64(total) || injection.GetStartAction() > int64(total)-injection.GetNumActions() {
			return fmt.Errorf("markov_sampling: injection %q exceeds num_logs=%d", injection.GetSegmentId(), total)
		}
	}
	return nil
}

// markovCursor wraps a walker with a persisted next-node id so a paused cursor resumes exactly
// where it left off. A nextID of "" means the next emit starts a fresh walk at the graph's start.
type markovCursor struct {
	walker *markovWalker
	nextID string
}

// emitNext emits one AuditLogEntry at the shared clock value and returns the advanced clock. A
// fresh walk starts when nextID is empty; reaching a terminal node resets nextID so the following
// emit restarts the walk (within the same segment for an anomaly cursor).
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

// generateLogsMarkovInjected emits exactly total AuditLogEntry records, splicing anomaly segments
// into a normal stream. The normal cursor walks the base graph; during each injection segment it is
// paused and an anomaly cursor walks that segment's referenced full anomaly graph (anomalyGraphs,
// keyed by segment_id) from the anomaly graph's own start spec, restarting at terminal nodes until
// the segment ends. All cursors share the single seeded rng and a single nondecreasing clock.
func generateLogsMarkovInjected(cfg *genpb.MarkovSamplingConfig, anomalyGraphs map[string]*genpb.MarkovSamplingConfig, total int, start int64, rng *rand.Rand, out io.Writer) error {
	if err := validateMarkovInjectionBounds(cfg, total); err != nil {
		return err
	}

	newCursor := func(graph *markovGraph, profile, segment string) *markovCursor {
		return &markovCursor{walker: &markovWalker{
			graph:                graph,
			generator:            NewLogGenerator(rng),
			rng:                  rng,
			labelFromInjection:   true,
			groundTruthProfile:   profile,
			groundTruthSegmentID: segment,
		}}
	}

	normal := newCursor(newMarkovGraph(cfg), "normal", "")
	var anomalous *markovCursor
	segmentIndex := 0
	clock := start
	for position := 0; position < total; position++ {
		// Advance past a segment the current position has moved beyond.
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
					graph := newMarkovGraph(anomalyGraphs[segment.GetSegmentId()])
					anomalous = newCursor(graph, segment.GetAnomalyGraphPath(), segment.GetSegmentId())
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
