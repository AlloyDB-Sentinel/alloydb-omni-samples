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

// Automatic anomaly injection (auto_injection): the generator positions non-overlapping anomaly
// segments to a target anomalous fraction instead of the author enumerating each one. Placement is
// planned entirely up front from the single seeded rng in a fixed draw order (per segment: gap,
// then length, then graph selection), so the realized placement is deterministic. The planned
// segments are then run through the exact same two-cursor loop as manual injection
// (generateLogsMarkovInjected) by synthesizing an equivalent MarkovInjection list.

// autoSelector precomputes a cumulative-weight table over an auto_injection's anomaly_refs so a
// referenced graph can be chosen per segment with the same idiom as edge/start selection. When no
// ref carries a positive selection_weight, selection is uniform.
type autoSelector struct {
	paths             []string
	cumulativeWeights []float64 // last element == total weight
}

func newAutoSelector(refs []*genpb.MarkovAutoAnomalyRef) *autoSelector {
	s := &autoSelector{}
	var total float64
	var anyWeight bool
	for _, ref := range refs {
		w := ref.GetSelectionWeight()
		if w > 0 {
			anyWeight = true
		}
	}
	for _, ref := range refs {
		w := ref.GetSelectionWeight()
		if !anyWeight {
			w = 1.0 // uniform when no positive weights are configured
		}
		total += w
		s.paths = append(s.paths, ref.GetAnomalyGraphPath())
		s.cumulativeWeights = append(s.cumulativeWeights, total)
	}
	return s
}

// selectPath draws one referenced anomaly-graph path using the cumulative-weight idiom.
func (s *autoSelector) selectPath(rng *rand.Rand) string {
	total := s.cumulativeWeights[len(s.cumulativeWeights)-1]
	r := rng.Float64() * total
	for i, cw := range s.cumulativeWeights {
		if r < cw {
			return s.paths[i]
		}
	}
	return s.paths[len(s.paths)-1]
}

// plannedSegment is one auto-placed anomaly segment: a half-open [start, start+length) range, the
// referenced anomaly graph path, and the generator-assigned "auto-N" segment id.
type plannedSegment struct {
	start     int64
	length    int64
	graphPath string
	segmentID string
}

// planAutoInjections positions non-overlapping anomaly segments across a stream of total actions,
// targeting an anomalous budget of round(anomalous_fraction * total). For each segment it draws, in
// a fixed order, a gap of normal actions, a length uniform in [min_actions, max_actions] clamped to
// the remaining budget and stream, and a referenced graph by selection weight. Planning stops when
// the budget is exhausted or the stream ends. All draws come from rng, so the plan is deterministic.
func planAutoInjections(auto *genpb.MarkovAutoInjection, total int, rng *rand.Rand) []plannedSegment {
	budget := int64(math.Round(auto.GetAnomalousFraction() * float64(total)))
	if budget <= 0 {
		return nil
	}
	selector := newAutoSelector(auto.GetAnomalyRefs())
	minLen := auto.GetMinActions()
	maxLen := auto.GetMaxActions()

	var planned []plannedSegment
	var pos int64
	remaining := budget
	index := 0
	for remaining > 0 && pos < int64(total) {
		// Draw a gap of normal actions. The gap is spread across the room left over after the
		// remaining budget is reserved, so segments stay distributed across the stream rather than
		// bunching at the front. maxGap is the slack between the stream end and the remaining budget.
		maxGap := int64(total) - pos - remaining
		if maxGap < 0 {
			maxGap = 0
		}
		var gap int64
		if maxGap > 0 {
			gap = rng.Int63n(maxGap + 1)
		}
		pos += gap
		if pos >= int64(total) {
			break
		}

		// Draw a segment length uniform in [minLen, maxLen], then clamp to the remaining budget and
		// the room left in the stream.
		length := minLen
		if maxLen > minLen {
			length = minLen + rng.Int63n(maxLen-minLen+1)
		}
		if length > remaining {
			length = remaining
		}
		if length > int64(total)-pos {
			length = int64(total) - pos
		}
		if length <= 0 {
			break
		}

		graphPath := selector.selectPath(rng)
		planned = append(planned, plannedSegment{
			start:     pos,
			length:    length,
			graphPath: graphPath,
			segmentID: fmt.Sprintf("auto-%d", index),
		})
		pos += length
		remaining -= length
		index++
	}
	return planned
}

// generateLogsMarkovAuto plans auto-injection segments, then runs the shared injected two-cursor
// loop. It synthesizes a MarkovSamplingConfig carrying the planned segments as MarkovInjections and
// a segment_id-keyed anomaly-graph map, so generation reuses generateLogsMarkovInjected verbatim.
func generateLogsMarkovAuto(cfg *genpb.MarkovSamplingConfig, autoAnomalyGraphs map[string]*genpb.MarkovSamplingConfig, total int, start int64, rng *rand.Rand, out io.Writer) error {
	if err := validateMarkovAutoInjectionBounds(cfg, total); err != nil {
		return err
	}
	planned := planAutoInjections(cfg.GetAutoInjection(), total, rng)

	// Build a synthetic config identical to cfg but with the planned segments as injections, plus a
	// segment_id -> resolved anomaly graph map. autoAnomalyGraphs is keyed by anomaly_graph_path.
	synthetic := genpb.MarkovSamplingConfig_builder{
		Nodes: cfg.GetNodes(),
		Start: cfg.GetStart(),
	}.Build()
	injections := make([]*genpb.MarkovInjection, 0, len(planned))
	bySegment := make(map[string]*genpb.MarkovSamplingConfig, len(planned))
	for _, seg := range planned {
		injections = append(injections, genpb.MarkovInjection_builder{
			SegmentId:        seg.segmentID,
			AnomalyGraphPath: seg.graphPath,
			StartAction:      seg.start,
			NumActions:       seg.length,
		}.Build())
		bySegment[seg.segmentID] = autoAnomalyGraphs[seg.graphPath]
	}
	synthetic.SetInjections(injections)

	return generateLogsMarkovInjected(synthetic, bySegment, total, start, rng, out)
}

// validateMarkovAutoInjection checks a markov_sampling config's auto_injection spec. The referenced
// anomaly graphs are resolved and validated separately (resolveBaseGraphFile in loadConfigs); this
// function enforces the spec-level invariants: mutual exclusion with manual injections, a valid
// target fraction, a valid segment-length range, at least one reference, and finite non-negative
// selection weights that (when any is positive) sum to a positive value.
func validateMarkovAutoInjection(cfg *genpb.MarkovSamplingConfig) error {
	auto := cfg.GetAutoInjection()
	if auto == nil {
		return nil
	}
	if len(cfg.GetInjections()) > 0 {
		return fmt.Errorf("markov_sampling: auto_injection and injections are mutually exclusive")
	}
	f := auto.GetAnomalousFraction()
	if math.IsNaN(f) || math.IsInf(f, 0) || f < 0.0 || f > 1.0 {
		return fmt.Errorf("markov_sampling: auto_injection anomalous_fraction must be in [0,1]")
	}
	if auto.GetMinActions() < 1 || auto.GetMaxActions() < auto.GetMinActions() {
		return fmt.Errorf("markov_sampling: auto_injection requires min_actions>=1 and max_actions>=min_actions")
	}
	if len(auto.GetAnomalyRefs()) == 0 {
		return fmt.Errorf("markov_sampling: auto_injection requires at least one anomaly_ref")
	}
	var totalWeight float64
	for _, ref := range auto.GetAnomalyRefs() {
		w := ref.GetSelectionWeight()
		if !isFiniteNonNegative(w) {
			return fmt.Errorf("markov_sampling: auto_injection selection weights must be finite, non-negative and sum to a positive value")
		}
		totalWeight += w
	}
	// Uniform selection (all weights zero/unset) is allowed; only reject a mix that sums to <= 0
	// while claiming to be weighted — which cannot happen here since all weights are non-negative
	// and "all zero" is treated as uniform. A negative total is impossible given the check above.
	_ = totalWeight
	return nil
}

// validateMarkovAutoInjectionBounds verifies the auto_injection target budget and minimum segment
// length fit within the total number of actions (--num_logs). Called at flag-validation time and
// before generation, mirroring validateMarkovInjectionBounds for the manual path.
func validateMarkovAutoInjectionBounds(cfg *genpb.MarkovSamplingConfig, total int) error {
	auto := cfg.GetAutoInjection()
	if auto == nil {
		return nil
	}
	budget := int64(math.Round(auto.GetAnomalousFraction() * float64(total)))
	if budget > int64(total) {
		return fmt.Errorf("markov_sampling: auto_injection anomalous budget %d exceeds num_logs=%d", budget, total)
	}
	if budget > 0 && auto.GetMinActions() > int64(total) {
		return fmt.Errorf("markov_sampling: auto_injection min_actions %d exceeds num_logs=%d", auto.GetMinActions(), total)
	}
	return nil
}
