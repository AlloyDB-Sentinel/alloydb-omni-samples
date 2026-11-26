package main

import (
	"fmt"
	"math/rand"

	"google.golang.org/protobuf/proto"

	genpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v1/generator/proto"
)

// edgeKey identifies a Markov_Edge by its (source_id, destination_id) pair, used to
// index per-individual mechanisms (overrides, variances) for O(1) lookup during derivation.
type edgeKey struct {
	src string
	dst string
}

// Graph derivation for subtypes and individuals.
//
// Derivation transforms a validated base MarkovSamplingConfig (optionally with
// overrides or per-individual mechanisms applied) into a plain, ready-to-walk
// MarkovSamplingConfig. renormalizeOutgoing is the single place outgoing edge
// weights are scaled to sum to 1.0; it is called exactly once after all weight
// edits for a derived graph.

// renormalizeOutgoing scales each non-terminal node's outgoing edge weights so
// they sum to 1.0. Terminal nodes (no outgoing edges) are left untouched.
//
// For each non-terminal node it sums the outgoing weights and rejects the graph
// when that sum is not positive; the `!(sum > 0)` test also rejects a NaN sum.
// Otherwise it scales every edge by 1.0 / sum using the opaque-API setter.
func renormalizeOutgoing(nodes []*genpb.MarkovNode) error {
	for _, node := range nodes {
		edges := node.GetEdges()
		if len(edges) == 0 {
			// Terminal node: no outgoing edges to renormalize.
			continue
		}

		var sum float64
		for _, edge := range edges {
			sum += edge.GetWeight()
		}

		// Reject a non-positive sum (which also catches NaN): such a node cannot
		// be renormalized to sum to 1.0 within FloatTolerance.
		if !(sum > 0) {
			return fmt.Errorf("markov derivation: node %q: outgoing edge weights sum to %v, must be positive to renormalize", node.GetId(), sum)
		}

		scale := 1.0 / sum
		for _, edge := range edges {
			edge.SetWeight(edge.GetWeight() * scale)
		}
	}
	return nil
}

// applyEdgeOverrides replaces edge weights in place for the edges named by overrides.
//
// It builds a source-id -> destination-id -> *MarkovEdge index over nodes, then for each
// override sets the matching edge's weight with edge.SetWeight(ov.GetWeight()). The index is
// over the edges passed in (expected to be a cloned graph, so the loaded base is never mutated);
// overrides naming a (source, destination) pair with no matching edge are ignored here, as
// validation (validateSubtype) is responsible for rejecting such overrides before derivation.
func applyEdgeOverrides(nodes []*genpb.MarkovNode, overrides []*genpb.EdgeWeightOverride) {
	index := make(map[string]map[string]*genpb.MarkovEdge, len(nodes))
	for _, node := range nodes {
		edges := node.GetEdges()
		if len(edges) == 0 {
			continue
		}
		byDest := make(map[string]*genpb.MarkovEdge, len(edges))
		for _, edge := range edges {
			byDest[edge.GetDestinationId()] = edge
		}
		index[node.GetId()] = byDest
	}

	for _, ov := range overrides {
		byDest, ok := index[ov.GetSourceId()]
		if !ok {
			continue
		}
		edge, ok := byDest[ov.GetDestinationId()]
		if !ok {
			continue
		}
		edge.SetWeight(ov.GetWeight())
	}
}

// deriveSubtypeGraph derives a plain MarkovSamplingConfig from a subtype config and its
// already-resolved base graph.
//
// base is the resolved base graph referenced by cfg.base_graph_path; the subtype message itself
// only carries the path, so the caller (dispatch wiring) supplies the resolved base. The base is
// cloned with proto.Clone before any edit so the loaded base is never mutated and all
// non-overridden structure (nodes, templates, completion times, un-named edges) is preserved
// exactly. Edge-weight overrides are applied by (source_id, destination_id), then
// renormalizeOutgoing is called exactly once so every non-terminal node's outgoing weights sum to
// 1.0. The returned MarkovSamplingConfig is ready for newMarkovGraph.
func deriveSubtypeGraph(cfg *genpb.MarkovSubtypeConfig, base *genpb.MarkovSamplingConfig) (*genpb.MarkovSamplingConfig, error) {
	derived := proto.Clone(base).(*genpb.MarkovSamplingConfig)

	applyEdgeOverrides(derived.GetNodes(), cfg.GetEdgeWeightOverrides())

	if err := renormalizeOutgoing(derived.GetNodes()); err != nil {
		return nil, fmt.Errorf("markov subtype derivation: %w", err)
	}

	return derived, nil
}

// deriveIndividualGraph derives a plain MarkovSamplingConfig for a single Individual from an
// individual config and its already-resolved base graph.
//
// base is the resolved base graph referenced by cfg.base_graph_path; the individual message itself
// only carries the path, so the caller (dispatch wiring) supplies the resolved base, mirroring
// deriveSubtypeGraph. The base is cloned with proto.Clone before any edit so the loaded base is
// never mutated. Any subtype_overrides are applied first with applyEdgeOverrides to form the
// *class graph* (R14.1); the class weight of an edge is its weight after that layer.
//
// The derivation realizes Parameterization_Precedence (R15.1) as a strict per-edge ladder: each
// edge is governed by exactly one mechanism — the highest-priority one that applies — never two
// stacked together. Nodes are iterated in declaration order and, within a node, edges in
// declaration order, so the rng draw order is fully determined for a fixed seed (R15.6, R11).
// For each edge, with classWeight being its post-subtype-override weight:
//
//   - pinned by individual_edge_overrides: use the override value, draw nothing (R15.2);
//   - else a variance entry with a positive down or up delta (down > 0 || up > 0): draw exactly
//     one value w = classWeight - down + rng.Float64()*(down+up) as the only mechanism, so jitter
//     is NOT also applied to this edge (R14.3, R15.3);
//   - else when weight_jitter.bound > 0: draw exactly one value
//     w = classWeight + (rng.Float64()*2-1)*bound (R14.4, R15.4);
//   - else keep the class weight and draw nothing (R14.6, R15.5).
//
// Because the branches are mutually exclusive, no edge is ever perturbed by both variance and
// jitter (R15.1, R15.6). Each non-pinned result is clamped to >= 0 before being written back with
// edge.SetWeight so no negative weight reaches renormalization (R16.9); pinned override values are
// written through unchanged (already validated finite and non-negative). renormalizeOutgoing then
// runs exactly once, surfacing the R16.8 non-positive-sum error. The returned MarkovSamplingConfig
// is ready for newMarkovGraph.
func deriveIndividualGraph(cfg *genpb.MarkovIndividualConfig, base *genpb.MarkovSamplingConfig, rng *rand.Rand) (*genpb.MarkovSamplingConfig, error) {
	// Clone the resolved base so the loaded base is never mutated, then apply any subtype
	// overrides to form the class graph whose edge weights are the per-edge class weights.
	derived := proto.Clone(base).(*genpb.MarkovSamplingConfig)
	applyEdgeOverrides(derived.GetNodes(), cfg.GetSubtypeOverrides())

	// Build the per-edge lookup maps used by the precedence ladder:
	//   - pinned overrides by (src, dst) from individual_edge_overrides;
	//   - variance entries by (src, dst) from edge_variances;
	//   - the single global weight-jitter bound.
	pinned := make(map[edgeKey]float64, len(cfg.GetIndividualEdgeOverrides()))
	for _, ov := range cfg.GetIndividualEdgeOverrides() {
		pinned[edgeKey{ov.GetSourceId(), ov.GetDestinationId()}] = ov.GetWeight()
	}

	variances := make(map[edgeKey]*genpb.EdgeVariance, len(cfg.GetEdgeVariances()))
	for _, v := range cfg.GetEdgeVariances() {
		variances[edgeKey{v.GetSourceId(), v.GetDestinationId()}] = v
	}

	// GetWeightJitter() is nil-safe and GetBound() on a nil receiver returns 0, so an unset
	// jitter config yields a zero bound that never triggers the jitter branch.
	jitterBound := cfg.GetWeightJitter().GetBound()

	// Iterate nodes in declaration order; within a node, edges in declaration order. This fixed
	// order plus at-most-one draw per edge fully determines the rng draw sequence (R15.6, R11).
	for _, node := range derived.GetNodes() {
		src := node.GetId()
		for _, edge := range node.GetEdges() {
			key := edgeKey{src, edge.GetDestinationId()}

			// (1) Pinned by an individual edge override: write the override value directly and
			// draw nothing. The override weight is validated non-negative already (R15.2).
			if w, ok := pinned[key]; ok {
				edge.SetWeight(w)
				continue
			}

			classWeight := edge.GetWeight()
			var w float64
			switch v, hasVar := variances[key]; {
			case hasVar && (v.GetDown() > 0 || v.GetUp() > 0):
				// (2) Edge_Weight_Variance only: one asymmetric draw in
				// [classWeight-down, classWeight+up]. Jitter is not also applied (R14.3, R15.3).
				down, up := v.GetDown(), v.GetUp()
				w = classWeight - down + rng.Float64()*(down+up)
			case jitterBound > 0:
				// (3) Weight_Jitter only: one draw in [classWeight-bound, classWeight+bound],
				// reached only because this edge has no override and no applicable variance
				// entry (R14.4, R15.4).
				w = classWeight + (rng.Float64()*2-1)*jitterBound
			default:
				// No mechanism applies: keep the class weight and draw nothing (R14.6, R15.5).
				w = classWeight
			}

			// Clamp each non-pinned result to >= 0 before writing it back so no negative weight
			// enters renormalization (R16.9).
			if w < 0 {
				w = 0
			}
			edge.SetWeight(w)
		}
	}

	// Renormalize exactly once after all per-edge mechanisms have been applied. This surfaces the
	// R16.8 non-positive-sum error for any non-terminal node (R15.3, R15.4, R15.5).
	if err := renormalizeOutgoing(derived.GetNodes()); err != nil {
		return nil, fmt.Errorf("markov individual derivation: %w", err)
	}

	return derived, nil
}
