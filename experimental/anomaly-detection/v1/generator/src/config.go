// Package main implements config loading logic for the generator.
package main

import (
	"context"
	"fmt"
	"math"
	"net/netip"
	"os"
	"strings"
	"time"

	"google.golang.org/protobuf/encoding/prototext"
	"google.golang.org/protobuf/types/known/durationpb"

	cfgpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v1/generator/proto"
)

// loadConfigs reads and validates the query template configurations from a textproto file.
func loadConfigs(ctx context.Context, filePath string) (*cfgpb.QueryTemplateConfigs, error) {
	// 1. Read config textproto
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, fmt.Errorf("failed to read file %s: %w", filePath, err)
	}

	// 2. Create an empty instance of the Go struct defined by config proto
	configs := &cfgpb.QueryTemplateConfigs{}

	// 3. Unmarshal config textproto data to configs
	err = prototext.Unmarshal(data, configs)
	if err != nil {
		return nil, fmt.Errorf("failed to unmarshal textproto: %w", err)
	}

	// 4. Validate configurations
	if err := validateConfigs(configs); err != nil {
		return nil, fmt.Errorf("invalid configuration: %w", err)
	}

	return configs, nil
}

func isFiniteNonNegative(v float64) bool {
	return v >= 0.0 && !math.IsNaN(v) && !math.IsInf(v, 0)
}

// validateConfigs ensures logical integrity of the loaded configurations.
func validateConfigs(configs *cfgpb.QueryTemplateConfigs) error {
	// Identify scenario: Weighted, Bucket or Connection sampling.
	if configs.GetWeightedSampling() != nil {
		cfgs := configs.GetWeightedSampling().GetQueryTemplateConfigs()
		if len(cfgs) == 0 {
			return fmt.Errorf("configuration contains no query templates")
		}
		var totalWeight float64
		for i, wCfg := range cfgs {
			if math.IsNaN(wCfg.GetProbabilityWeight()) {
				return fmt.Errorf("config[%d]: probability_weight cannot be NaN", i)
			}
			if wCfg.GetProbabilityWeight() < 0.0 {
				return fmt.Errorf("config[%d]: probability_weight cannot be negative", i)
			}
			totalWeight += wCfg.GetProbabilityWeight()

			if err := validateQueryTemplate(i, wCfg.GetTemplate()); err != nil {
				return err
			}
		}

		if totalWeight == 0 {
			return fmt.Errorf("at least one template must have a positive weight")
		}
		if math.IsInf(totalWeight, 1) {
			return fmt.Errorf("total probability weight exceeds maximum representable float64 value")
		}

	} else if configs.GetBucketSampling() != nil {
		cfgs := configs.GetBucketSampling().GetQueryTemplateConfigs()
		if len(cfgs) == 0 {
			return fmt.Errorf("configuration contains no query templates")
		}
		hasPositiveMean := false
		for i, bCfg := range cfgs {
			simCfg := bCfg.GetBucketConfig()
			if simCfg == nil {
				return fmt.Errorf("config[%d]: bucket_config cannot be nil under bucket_sampling scenario", i)
			}
			if err := validateBucketSimulationConfig(i, simCfg); err != nil {
				return err
			}
			if simCfg.GetQueriesPerBucketMean() > 0 {
				hasPositiveMean = true
			}

			if err := validateQueryTemplate(i, bCfg.GetTemplate()); err != nil {
				return err
			}
		}

		if !hasPositiveMean {
			return fmt.Errorf("at least one template must have a queries_per_bucket_mean > 0 for bucket_sampling")
		}

	} else if configs.GetConnectionSampling() != nil {
		if err := validateConnectionSampling(configs.GetConnectionSampling()); err != nil {
			return err
		}

	} else if configs.GetMarkovSampling() != nil {
		return validateMarkovSampling(configs.GetMarkovSampling())

	} else {
		return fmt.Errorf("configuration must specify either weighted_sampling, bucket_sampling, connection_sampling or markov_sampling")
	}

	return nil
}

// validateBucketSimulationConfig checks the traffic settings shared by bucket_sampling templates
// and connection_sampling profiles that use bucket_traffic.
func validateBucketSimulationConfig(i int, simCfg *cfgpb.BucketSimulationConfig) error {
	if !isFiniteNonNegative(simCfg.GetQueriesPerBucketMean()) {
		return fmt.Errorf("config[%d]: queries_per_bucket_mean must be finite and non-negative", i)
	}
	if !isFiniteNonNegative(simCfg.GetQueriesVariationStdDev()) {
		return fmt.Errorf("config[%d]: queries_variation_std_dev must be finite and non-negative", i)
	}
	if simCfg.GetAnomalousSpikeProbability() < 0.0 || simCfg.GetAnomalousSpikeProbability() > 1.0 || math.IsNaN(simCfg.GetAnomalousSpikeProbability()) {
		return fmt.Errorf("config[%d]: anomalous_spike_probability must be between 0.0 and 1.0", i)
	}
	if !isFiniteNonNegative(simCfg.GetAnomalousSpikeMultiplier()) {
		return fmt.Errorf("config[%d]: anomalous_spike_multiplier must be finite and non-negative", i)
	}
	if !isFiniteNonNegative(simCfg.GetWeekendTrafficMultiplier()) {
		return fmt.Errorf("config[%d]: weekend_traffic_multiplier must be finite and non-negative", i)
	}
	return validatePeakWindows(i, simCfg.GetPeakWindows())
}

// validateConnectionSampling ensures every client profile can produce connection log entries.
func validateConnectionSampling(connectionCfg *cfgpb.ConnectionSamplingConfig) error {
	profiles := connectionCfg.GetClientProfiles()
	if len(profiles) == 0 {
		return fmt.Errorf("configuration contains no client profiles")
	}
	for i, profile := range profiles {
		if err := validateClientAddressPool(i, profile.GetClientAddressPool()); err != nil {
			return err
		}
		if err := validateOutcomeWeights(i, profile); err != nil {
			return err
		}
		if err := validateConnectionSchedule(i, profile); err != nil {
			return err
		}
	}
	return nil
}

// validateClientAddressPool ensures a profile has at least one well-formed address or CIDR range.
func validateClientAddressPool(i int, pool *cfgpb.ClientAddressPool) error {
	if len(pool.GetAddresses()) == 0 && len(pool.GetCidrRanges()) == 0 {
		return fmt.Errorf("config[%d]: client_address_pool must contain at least one address or cidr_range", i)
	}
	for j, address := range pool.GetAddresses() {
		if _, err := netip.ParseAddr(address); err != nil {
			return fmt.Errorf("config[%d].client_address_pool.addresses[%d]: invalid IP address %q: %w", i, j, address, err)
		}
	}
	for j, cidrRange := range pool.GetCidrRanges() {
		if _, err := netip.ParsePrefix(cidrRange); err != nil {
			return fmt.Errorf("config[%d].client_address_pool.cidr_ranges[%d]: invalid CIDR range %q: %w", i, j, cidrRange, err)
		}
	}
	return nil
}

// validateOutcomeWeights ensures outcome weights are usable and that authorized connections name a db_user.
func validateOutcomeWeights(i int, profile *cfgpb.ClientProfile) error {
	weights := profile.GetOutcomeWeights()

	// An empty list defaults to authorized connections only.
	canBeAuthorized := len(weights) == 0
	var totalWeight float64
	for j, outcomeWeight := range weights {
		if outcomeWeight.GetOutcome() == cfgpb.ConnectionOutcome_CONNECTION_OUTCOME_UNSPECIFIED {
			return fmt.Errorf("config[%d].outcome_weights[%d]: outcome must be specified", i, j)
		}
		if !isFiniteNonNegative(outcomeWeight.GetWeight()) {
			return fmt.Errorf("config[%d].outcome_weights[%d]: weight must be finite and non-negative", i, j)
		}
		totalWeight += outcomeWeight.GetWeight()
		if outcomeWeight.GetOutcome() == cfgpb.ConnectionOutcome_CONNECTION_OUTCOME_AUTHORIZED && outcomeWeight.GetWeight() > 0 {
			canBeAuthorized = true
		}
	}
	if len(weights) > 0 && totalWeight == 0 {
		return fmt.Errorf("config[%d]: at least one outcome weight must be positive", i)
	}
	if math.IsInf(totalWeight, 1) {
		return fmt.Errorf("config[%d]: total outcome weight exceeds maximum representable float64 value", i)
	}

	// PostgreSQL only logs "connection authorized" after resolving the user.
	if canBeAuthorized && profile.GetDbUser() == "" {
		return fmt.Errorf("config[%d]: db_user is required when connections can be authorized", i)
	}
	return nil
}

// validateConnectionSchedule ensures exactly one schedule is set and its timing values are usable.
func validateConnectionSchedule(i int, profile *cfgpb.ClientProfile) error {
	if bucketTraffic := profile.GetBucketTraffic(); bucketTraffic != nil {
		return validateBucketSimulationConfig(i, bucketTraffic)
	}

	if periodic := profile.GetPeriodic(); periodic != nil {
		interval, err := durationOrError(periodic.GetInterval())
		if err != nil || interval <= 0 {
			return fmt.Errorf("config[%d].periodic: interval must be a positive duration", i)
		}
		// A missing jitter means connections land exactly on the interval.
		var jitter time.Duration
		if periodic.HasJitter() {
			jitter, err = durationOrError(periodic.GetJitter())
			if err != nil || jitter < 0 {
				return fmt.Errorf("config[%d].periodic: jitter must be a non-negative duration", i)
			}
		}
		if jitter >= interval {
			return fmt.Errorf("config[%d].periodic: jitter (%v) must be less than interval (%v)", i, jitter, interval)
		}
		return nil
	}

	if burst := profile.GetBurst(); burst != nil {
		if !isFiniteNonNegative(burst.GetBurstsPerDay()) || burst.GetBurstsPerDay() == 0 {
			return fmt.Errorf("config[%d].burst: bursts_per_day must be finite and positive", i)
		}
		if !isFiniteNonNegative(burst.GetConnectionsPerBurstMean()) || burst.GetConnectionsPerBurstMean() == 0 {
			return fmt.Errorf("config[%d].burst: connections_per_burst_mean must be finite and positive", i)
		}
		if !isFiniteNonNegative(burst.GetConnectionsPerBurstStdDev()) {
			return fmt.Errorf("config[%d].burst: connections_per_burst_std_dev must be finite and non-negative", i)
		}
		burstDuration, err := durationOrError(burst.GetBurstDuration())
		if err != nil || burstDuration <= 0 {
			return fmt.Errorf("config[%d].burst: burst_duration must be a positive duration", i)
		}
		return nil
	}

	return fmt.Errorf("config[%d]: one of bucket_traffic, periodic or burst must be set", i)
}

// durationOrError converts a protobuf Duration, rejecting missing or malformed values.
func durationOrError(protoDuration *durationpb.Duration) (time.Duration, error) {
	if protoDuration == nil {
		return 0, fmt.Errorf("duration is not set")
	}
	if err := protoDuration.CheckValid(); err != nil {
		return 0, err
	}
	return protoDuration.AsDuration(), nil
}

func validatePeakWindows(i int, windows []*cfgpb.PeakWindow) error {
	for wIdx, pw := range windows {
		if pw.GetHourStart() < 0 || pw.GetHourStart() > 23 {
			return fmt.Errorf("config[%d].peak_windows[%d]: hour_start (%d) must be in [0, 23]", i, wIdx, pw.GetHourStart())
		}
		if pw.GetHourEnd() < 0 || pw.GetHourEnd() > 23 {
			return fmt.Errorf("config[%d].peak_windows[%d]: hour_end (%d) must be in [0, 23]", i, wIdx, pw.GetHourEnd())
		}
		if !isFiniteNonNegative(pw.GetMultiplier()) {
			return fmt.Errorf("config[%d].peak_windows[%d]: multiplier must be finite and non-negative", i, wIdx)
		}
		if pw.GetMultiplier() > 0 && pw.GetHourStart() == pw.GetHourEnd() {
			return fmt.Errorf("config[%d].peak_windows[%d]: hour_start (%d) and hour_end (%d) cannot be equal when multiplier is active", i, wIdx, pw.GetHourStart(), pw.GetHourEnd())
		}
	}

	for w1 := 0; w1 < len(windows); w1++ {
		for w2 := w1 + 1; w2 < len(windows); w2++ {
			pw1 := windows[w1]
			pw2 := windows[w2]
			if pw1.GetMultiplier() <= 0 || pw2.GetMultiplier() <= 0 {
				continue
			}
			for h := 0; h < 24; h++ {
				if InPeakWindow(h, pw1) && InPeakWindow(h, pw2) {
					return fmt.Errorf("config[%d].peak_windows: peak window %d (%s) and peak window %d (%s) overlap at hour %d", i, w1, pw1.GetWindowName(), w2, pw2.GetWindowName(), h)
				}
			}
		}
	}
	return nil
}

func validateQueryTemplate(i int, cfg *cfgpb.QueryTemplateConfig) error {
	if cfg == nil || len(cfg.GetNormalizedQueryTemplate()) == 0 {
		return fmt.Errorf("config[%d]: normalized_query_template cannot be empty", i)
	}
	// The number of placeholder_values matches the number of %v placeholders in the normalized_query_template.
	expectedCount := strings.Count(cfg.GetNormalizedQueryTemplate(), "%v")
	actualCount := len(cfg.GetPlaceholderValues())
	if expectedCount != actualCount {
		return fmt.Errorf("config[%d]: placeholder count mismatch: template requires %d '%%v' placeholders, but got %d placeholder specs", i, expectedCount, actualCount)
	}

	// min <= max for all placeholder value ranges.
	for j, spec := range cfg.GetPlaceholderValues() {
		if intSpec := spec.GetIntPlaceholder(); intSpec != nil {
			if intSpec.GetMin() > intSpec.GetMax() {
				return fmt.Errorf("config[%d].placeholder_values[%d]: int min (%d) cannot be greater than max (%d)", i, j, intSpec.GetMin(), intSpec.GetMax())
			}
		} else if floatSpec := spec.GetFloatPlaceholder(); floatSpec != nil {
			if floatSpec.GetMin() > floatSpec.GetMax() {
				return fmt.Errorf("config[%d].placeholder_values[%d]: float min (%f) cannot be greater than max (%f)", i, j, floatSpec.GetMin(), floatSpec.GetMax())
			}
		} else if stringSpec := spec.GetStringPlaceholder(); stringSpec != nil {
			if len(stringSpec.GetAllowedValues()) == 0 {
				return fmt.Errorf("config[%d].placeholder_values[%d]: string_placeholder allowed_values cannot be empty", i, j)
			}
		} else {
			return fmt.Errorf("config[%d].placeholder_values[%d]: placeholder type must be specified", i, j)
		}
	}
	return nil
}

// InPeakWindow checks whether the given hour (0-23) falls within the peak window [hour_start, hour_end).
func InPeakWindow(hour int, pw *cfgpb.PeakWindow) bool {
	if pw.GetMultiplier() <= 0 {
		return false
	}
	start := int(pw.GetHourStart())
	end := int(pw.GetHourEnd())
	span := (end - start + 24) % 24
	if span == 0 {
		return false
	}
	offset := (hour - start + 24) % 24
	return offset < span
}

// FloatTolerance is the maximum permitted absolute deviation when comparing a node's
// outgoing edge-weight sum to 1.0.
const FloatTolerance = 1e-9

// validateMarkovSampling ensures a Markov sampling configuration forms a well-defined graph:
// unique node ids, valid templates and completion times, edges that reference defined nodes,
// outgoing weights that sum to 1.0 (or a terminal node with no edges), a resolvable start spec,
// and every node reachable from the start.
func validateMarkovSampling(cfg *cfgpb.MarkovSamplingConfig) error {
	nodes := cfg.GetNodes()

	// Non-empty graph.
	if len(nodes) == 0 {
		return fmt.Errorf("markov_sampling: configuration contains no nodes")
	}

	// Build an id->node map, rejecting empty or duplicate ids.
	byID := make(map[string]*cfgpb.MarkovNode, len(nodes))
	for _, node := range nodes {
		id := node.GetId()
		if id == "" {
			return fmt.Errorf("markov_sampling: node id must not be empty")
		}
		if _, exists := byID[id]; exists {
			return fmt.Errorf("markov_sampling: duplicate node id %q", id)
		}
		byID[id] = node
	}

	for i, node := range nodes {
		id := node.GetId()

		// Validate the query template.
		if err := validateQueryTemplate(i, node.GetTemplate()); err != nil {
			return err
		}

		// Require completion_time and validate its durations.
		ct := node.GetCompletionTime()
		if ct == nil || !ct.HasDurationSpec() {
			return fmt.Errorf("markov_sampling: node %q: completion_time must be set", id)
		}
		if ct.HasRange() {
			r := ct.GetRange()
			minDur, err := durationOrError(r.GetMin())
			if err != nil || minDur < 0 {
				return fmt.Errorf("markov_sampling: node %q: completion_time duration must be non-negative", id)
			}
			maxDur, err := durationOrError(r.GetMax())
			if err != nil || maxDur < 0 {
				return fmt.Errorf("markov_sampling: node %q: completion_time duration must be non-negative", id)
			}
			if minDur > maxDur {
				return fmt.Errorf("markov_sampling: node %q: completion_time min (%v) must not exceed max (%v)", id, minDur, maxDur)
			}
		} else {
			fixedDur, err := durationOrError(ct.GetFixed())
			if err != nil || fixedDur < 0 {
				return fmt.Errorf("markov_sampling: node %q: completion_time duration must be non-negative", id)
			}
		}

		// Validate edges: destinations must exist, weights must be finite and non-negative.
		var sum float64
		for e, edge := range node.GetEdges() {
			dest := edge.GetDestinationId()
			if _, ok := byID[dest]; !ok {
				return fmt.Errorf("markov_sampling: node %q: edge %d destination %q is not a defined node", id, e, dest)
			}
			if !isFiniteNonNegative(edge.GetWeight()) {
				return fmt.Errorf("markov_sampling: node %q: edge %d weight must be finite and non-negative", id, e)
			}
			sum += edge.GetWeight()
		}

		// A node with no outgoing edges is terminal; others must sum to 1.0 within tolerance.
		if len(node.GetEdges()) > 0 && math.Abs(sum-1.0) > FloatTolerance {
			return fmt.Errorf("markov_sampling: node %q: outgoing edge weights sum to %v, must equal 1.0 within %v", id, sum, FloatTolerance)
		}
	}

	// Validate the start spec and collect the set of seed node ids for reachability.
	start := cfg.GetStart()
	if start == nil || !start.HasStartSpec() {
		return fmt.Errorf("markov_sampling: start must be set")
	}

	var seeds []string
	if start.HasNodeId() {
		startID := start.GetNodeId()
		if _, ok := byID[startID]; !ok {
			return fmt.Errorf("markov_sampling: start references undefined node id %q", startID)
		}
		seeds = append(seeds, startID)
	} else {
		dist := start.GetDistribution()
		var total float64
		for _, ws := range dist.GetStarts() {
			nodeID := ws.GetNodeId()
			if _, ok := byID[nodeID]; !ok {
				return fmt.Errorf("markov_sampling: start references undefined node id %q", nodeID)
			}
			if !isFiniteNonNegative(ws.GetWeight()) {
				return fmt.Errorf("markov_sampling: start distribution weights must be finite, non-negative and sum to a positive value")
			}
			total += ws.GetWeight()
			seeds = append(seeds, nodeID)
		}
		if total <= 0 {
			return fmt.Errorf("markov_sampling: start distribution weights must be finite, non-negative and sum to a positive value")
		}
	}

	// Reachability: traverse outgoing edges from the start seed(s).
	visited := make(map[string]bool, len(byID))
	queue := make([]string, 0, len(seeds))
	for _, s := range seeds {
		if !visited[s] {
			visited[s] = true
			queue = append(queue, s)
		}
	}
	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]
		for _, edge := range byID[current].GetEdges() {
			dest := edge.GetDestinationId()
			if !visited[dest] {
				visited[dest] = true
				queue = append(queue, dest)
			}
		}
	}
	// Report the first unreachable node in declaration order.
	for _, node := range nodes {
		if !visited[node.GetId()] {
			return fmt.Errorf("markov_sampling: node %q is unreachable from the start node", node.GetId())
		}
	}

	return nil
}
