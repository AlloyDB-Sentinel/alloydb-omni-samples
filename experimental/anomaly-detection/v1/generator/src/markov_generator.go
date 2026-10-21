package main

import (
	"fmt"
	"io"
	"math/rand"

	"google.golang.org/protobuf/encoding/protodelim"

	genpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v1/generator/proto"
)

// Markov sampling scenario.

// markovGraph is the parsed, ready-to-walk graph built once from a validated config.
type markovGraph struct {
	nodes map[string]*markovNode // keyed by node id
	start *startSelector         // resolves the first node of each walk
}

// markovNode holds one vertex plus a precomputed cumulative-weight edge table.
type markovNode struct {
	node *genpb.MarkovNode

	// Outgoing transitions. destIDs[i] is chosen when the draw falls in bucket i.
	// Empty for terminal nodes.
	destIDs           []string
	cumulativeWeights []float64 // monotonically increasing; last element == total weight
}

// startSelector resolves the start node for each walk.
// Exactly one of the two forms is populated (mirrors MarkovStartNode.start_spec).
type startSelector struct {
	fixedID string // non-empty when a single start node is configured

	// Weighted distribution form (same cumulative-weight idiom as edges).
	startIDs          []string
	cumulativeWeights []float64
}

// newMarkovGraph builds the in-memory graph from a validated MarkovSamplingConfig.
// It assumes validation has already run, so it does not re-check invariants.
func newMarkovGraph(cfg *genpb.MarkovSamplingConfig) *markovGraph {
	graph := &markovGraph{nodes: make(map[string]*markovNode)}

	for _, node := range cfg.GetNodes() {
		parsed := &markovNode{node: node}
		var total float64
		for _, edge := range node.GetEdges() {
			total += edge.GetWeight()
			parsed.destIDs = append(parsed.destIDs, edge.GetDestinationId())
			parsed.cumulativeWeights = append(parsed.cumulativeWeights, total)
		}
		graph.nodes[node.GetId()] = parsed
	}

	graph.start = &startSelector{}
	if cfg.GetStart().HasNodeId() {
		graph.start.fixedID = cfg.GetStart().GetNodeId()
	} else {
		var total float64
		for _, start := range cfg.GetStart().GetDistribution().GetStarts() {
			total += start.GetWeight()
			graph.start.startIDs = append(graph.start.startIDs, start.GetNodeId())
			graph.start.cumulativeWeights = append(graph.start.cumulativeWeights, total)
		}
	}

	return graph
}

// sampleCompletionMs returns the node's completion time in milliseconds for this visit.
// Fixed: the single value. Range: a uniform draw in the closed interval [minMs, maxMs].
func sampleCompletionMs(ct *genpb.CompletionTime, rng *rand.Rand) int64 {
	if ct.HasFixed() {
		return ct.GetFixed().AsDuration().Milliseconds()
	}
	r := ct.GetRange()
	minMs := r.GetMin().AsDuration().Milliseconds()
	maxMs := r.GetMax().AsDuration().Milliseconds()
	if maxMs == minMs {
		return minMs
	}
	return minMs + rng.Int63n(maxMs-minMs+1) // closed range, inclusive of both ends
}

// markovWalker performs weighted random walks over a markovGraph, emitting one
// log entry per visited node. All randomness flows from the single seeded rng.
type markovWalker struct {
	graph     *markovGraph
	generator *LogGenerator // reused: buildQueryStatement + constUser/constDbName
	clockMs   int64         // simulated clock; set once from --start_time_ms, never reset
	rng       *rand.Rand    // the single seeded source
}

// selectStartID resolves the first node of a walk. A fixed start node is returned
// directly; otherwise a cumulative-weight draw picks one of the configured starts.
func (w *markovWalker) selectStartID() string {
	s := w.graph.start
	if s.fixedID != "" {
		return s.fixedID
	}
	totalWeight := s.cumulativeWeights[len(s.cumulativeWeights)-1]
	randomWeight := w.rng.Float64() * totalWeight
	for i, cumulativeWeight := range s.cumulativeWeights {
		if randomWeight < cumulativeWeight {
			return s.startIDs[i]
		}
	}
	return s.startIDs[len(s.startIDs)-1]
}

// selectNextID picks the next node id according to n's outgoing edge weights.
// Only called for non-terminal nodes (non-empty destIDs/cumulativeWeights).
func (w *markovWalker) selectNextID(n *markovNode) string {
	totalWeight := n.cumulativeWeights[len(n.cumulativeWeights)-1]
	randomWeight := w.rng.Float64() * totalWeight
	for i, cumulativeWeight := range n.cumulativeWeights {
		if randomWeight < cumulativeWeight {
			return n.destIDs[i]
		}
	}
	return n.destIDs[len(n.destIDs)-1]
}

// emit builds one AuditLogEntry for the visited node and writes it to out as a
// length-delimited record. The query statement is rendered from the node's
// template, the db identity uses the shared package constants, and the entry is
// stamped with the walker's current simulated clock.
func (w *markovWalker) emit(n *markovNode, out io.Writer) error {
	entry := &genpb.AuditLogEntry{}
	template := n.node.GetTemplate()
	entry.SetNormalizedQueryTemplate(template.GetNormalizedQueryTemplate())
	entry.SetQueryStatement(w.generator.buildQueryStatement(template))
	entry.SetDbUser(constUser)
	entry.SetDbName(constDbName)
	entry.SetIsGroundTruthAnomalous(template.GetIsGroundTruthAnomalous())
	entry.SetTimestampMs(w.clockMs)

	if _, err := protodelim.MarshalTo(out, entry); err != nil {
		return fmt.Errorf("failed to write markov log: %w", err)
	}
	return nil
}

// GenerateLogsMarkov runs weighted random walks over the configured Markov graph,
// emitting exactly numLogs length-delimited AuditLogEntry records to w.
//
// Each walk starts at a node chosen by the start spec and follows weighted edges
// until it reaches a terminal node (no outgoing edges) or the numLogs cap is hit,
// whichever comes first. The simulated clock is initialized once from startTimeMs
// and advanced after each emission by that node's sampled completion time; because
// completion times are non-negative, emitted timestamps are non-decreasing.
// All randomness (start selection, transitions, completion sampling, and placeholder
// sampling inside buildQueryStatement) flows from the single seeded rng.
func GenerateLogsMarkov(configs *genpb.QueryTemplateConfigs, numLogs int, startTimeMs int64, rng *rand.Rand, w io.Writer) error {
	graph := newMarkovGraph(configs.GetMarkovSampling())
	walker := &markovWalker{
		graph:     graph,
		generator: NewLogGenerator(rng),
		clockMs:   startTimeMs,
		rng:       rng,
	}

	emitted := 0
	for emitted < numLogs {
		curID := walker.selectStartID()
		for {
			node := graph.nodes[curID]
			if err := walker.emit(node, w); err != nil {
				return err
			}
			emitted++
			// Advance the clock after emitting; completion is non-negative so
			// the timestamp stream never moves backwards.
			walker.clockMs += sampleCompletionMs(node.node.GetCompletionTime(), rng)

			// Stop immediately at the cap, even mid-walk.
			if emitted == numLogs {
				return nil
			}
			// Terminal node: end this walk; the outer loop restarts under the cap.
			if len(node.destIDs) == 0 {
				break
			}
			curID = walker.selectNextID(node)
		}
	}

	return nil
}
