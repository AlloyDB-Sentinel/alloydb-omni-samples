package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"time"

	genpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v1/generator/proto"
	clientpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/client/proto"
	auditpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v2/common/proto"
	"google.golang.org/protobuf/encoding/protodelim"
	"google.golang.org/protobuf/encoding/prototext"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/durationpb"
)

// Exactly 1/100 of generated validation records are anomalous (1%).
const anomalyRecordDenominator int64 = 100

type anomalyChunk struct {
	SegmentID string `json:"segment_id"`
	Start     int64  `json:"start_action"`
	Length    int64  `json:"num_actions"`
}
type anomalySchedule struct {
	TotalActions     int64          `json:"total_actions"`
	AnomalousActions int64          `json:"anomalous_actions"`
	NormalSeed       int64          `json:"normal_seed"`
	AnomalySeed      int64          `json:"anomaly_seed"`
	GraphSeed        int64          `json:"graph_seed"`
	ScheduleSeed     int64          `json:"schedule_seed"`
	Chunks           []anomalyChunk `json:"chunks"`
}

func validateIndependentAnomalies(c *clientpb.MarkovExperimentConfig) error {
	a := c.GetIndependentAnomalies()
	if a == nil {
		return nil
	}
	if len(c.GetInjections()) != 0 || len(c.GetAnomalyProfiles()) != 0 {
		return fmt.Errorf("independent_anomalies cannot be combined with legacy profiles/injections")
	}
	if c.GetValidationActions()%anomalyRecordDenominator != 0 {
		return fmt.Errorf("independent_anomalies requires validation_actions divisible by 100 for exactly 1%% anomalous records")
	}
	if a.GetChunkActions() == 0 {
		a.SetChunkActions(30)
	}
	if !a.HasCompletionMinMs() {
		a.SetCompletionMinMs(1)
	}
	if !a.HasCompletionMaxMs() {
		a.SetCompletionMaxMs(10)
	}
	if len(a.GetPreferredTemplates()) > 0 && !a.HasPreferredProbability() {
		a.SetPreferredProbability(.9)
	}
	if a.HasPreferredProbability() && (len(a.GetPreferredTemplates()) == 0 || math.IsNaN(a.GetPreferredProbability()) || math.IsInf(a.GetPreferredProbability(), 0) || a.GetPreferredProbability() <= 0 || a.GetPreferredProbability() >= 1) {
		return fmt.Errorf("preferred_probability requires preferred_templates and must be finite and strictly between 0 and 1")
	}
	if a.GetChunkActions() < 1 || a.GetCompletionMinMs() < 0 || a.GetCompletionMaxMs() < a.GetCompletionMinMs() || a.GetCompletionMaxMs() > int64(math.MaxInt64/time.Millisecond) {
		return fmt.Errorf("invalid independent anomaly chunk length or completion range")
	}
	return nil
}

// Uniform weak composition of all normal actions into K+1 gaps (stars and bars).
// Every gap may be zero, including the leading and trailing gaps. No window
// length or minimum distance participates in placement.
func randomAnomalySchedule(total, chunk, seed int64) []anomalyChunk {
	budget := total / anomalyRecordDenominator
	if budget == 0 {
		return nil
	}
	k := (budget-1)/chunk + 1
	normal := total - budget
	slots := normal + k
	rng := rand.New(rand.NewSource(seed))
	bars := make(map[int64]bool, k)
	// Floyd sampling selects K unique bar positions in O(K) draws.
	for j := slots - k; j < slots; j++ {
		candidate := rng.Int63n(j + 1)
		if bars[candidate] {
			bars[j] = true
		} else {
			bars[candidate] = true
		}
	}
	positions := make([]int64, 0, k)
	for p := range bars {
		positions = append(positions, p)
	}
	sort.Slice(positions, func(i, j int) bool { return positions[i] < positions[j] })
	result := make([]anomalyChunk, 0, k)
	var cursor int64
	previous := int64(-1)
	for i, p := range positions {
		cursor += p - previous - 1
		length := chunk
		if length > budget {
			length = budget
		}
		result = append(result, anomalyChunk{fmt.Sprintf("random-%06d", i+1), cursor, length})
		cursor += length
		budget -= length
		previous = p
	}
	return result
}

func independentAnomalyGraph(normal *genpb.QueryTemplateConfigs, a *clientpb.IndependentAnomalyConfig, seed int64) (*genpb.QueryTemplateConfigs, error) {
	sources := []*genpb.QueryTemplateConfigs{normal}
	// Explicit alphabet paths replace the default normal-workload alphabet.
	// This lets every workload use the standard three tokens unless expanded.
	if len(a.GetAlphabetConfigPaths()) > 0 {
		sources = nil
	}
	for _, path := range a.GetAlphabetConfigPaths() {
		b, err := os.ReadFile(path)
		if err != nil {
			return nil, err
		}
		c := &genpb.QueryTemplateConfigs{}
		if err = prototext.Unmarshal(b, c); err != nil {
			return nil, err
		}
		sources = append(sources, c)
	}
	templates := map[string]*genpb.QueryTemplateConfig{}
	for _, source := range sources {
		var actions []*genpb.QueryTemplateConfig
		switch {
		case source.GetMarkovSampling() != nil:
			for _, node := range source.GetMarkovSampling().GetNodes() {
				actions = append(actions, node.GetTemplate())
			}
		case source.GetWeightedSampling() != nil:
			for _, entry := range source.GetWeightedSampling().GetQueryTemplateConfigs() {
				actions = append(actions, entry.GetTemplate())
			}
		case source.GetBucketSampling() != nil:
			for _, entry := range source.GetBucketSampling().GetQueryTemplateConfigs() {
				actions = append(actions, entry.GetTemplate())
			}
		default:
			return nil, fmt.Errorf("alphabet sources must contain Markov, weighted, or bucket query templates")
		}
		for _, t := range actions {
			if t.GetNormalizedQueryTemplate() == "" {
				return nil, fmt.Errorf("anomaly alphabet contains an empty template")
			}
			if _, exists := templates[t.GetNormalizedQueryTemplate()]; !exists {
				templates[t.GetNormalizedQueryTemplate()] = proto.Clone(t).(*genpb.QueryTemplateConfig)
			}
		}
	}
	keys := make([]string, 0, len(templates))
	for key := range templates {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	if len(keys) == 0 {
		return nil, fmt.Errorf("anomaly alphabet is empty")
	}
	preferred := make([]bool, len(keys))
	seenPreferred := map[string]bool{}
	for _, key := range a.GetPreferredTemplates() {
		index := sort.SearchStrings(keys, key)
		if index == len(keys) || keys[index] != key || seenPreferred[key] {
			return nil, fmt.Errorf("preferred template must be unique and present in the anomaly alphabet: %q", key)
		}
		preferred[index] = true
		seenPreferred[key] = true
	}
	if len(seenPreferred) == len(keys) {
		return nil, fmt.Errorf("preferred templates must leave at least one other action")
	}
	probability := a.GetPreferredProbability()
	if len(seenPreferred) > 0 && !a.HasPreferredProbability() {
		probability = .9
	}
	rng := rand.New(rand.NewSource(seed))
	startWeights := make([]float64, len(keys))
	for i := range startWeights {
		startWeights[i] = 1 / float64(len(keys))
	}
	if len(seenPreferred) > 0 {
		startWeights = sampleAnomalyWeights(rng, preferred, probability)
	}
	nodes := make([]*genpb.MarkovNode, 0, len(keys))
	starts := &genpb.WeightedStartNodes{}
	var startEntries []*genpb.WeightedStartNode
	for i, key := range keys {
		node := &genpb.MarkovNode{}
		node.SetId(fmt.Sprintf("action-%04d", i))
		template := templates[key]
		template.SetIsGroundTruthAnomalous(false)
		node.SetTemplate(template)
		completion := &genpb.CompletionTime{}
		r := &genpb.DurationRange{}
		r.SetMin(durationpb.New(time.Duration(a.GetCompletionMinMs()) * time.Millisecond))
		r.SetMax(durationpb.New(time.Duration(a.GetCompletionMaxMs()) * time.Millisecond))
		completion.SetRange(r)
		node.SetCompletionTime(completion)
		weights := sampleAnomalyWeights(rng, preferred, probability)
		var edges []*genpb.MarkovEdge
		for j, w := range weights {
			edge := &genpb.MarkovEdge{}
			edge.SetDestinationId(fmt.Sprintf("action-%04d", j))
			edge.SetWeight(w)
			edges = append(edges, edge)
		}
		node.SetEdges(edges)
		nodes = append(nodes, node)
		s := &genpb.WeightedStartNode{}
		s.SetNodeId(node.GetId())
		s.SetWeight(startWeights[i])
		startEntries = append(startEntries, s)
	}
	starts.SetStarts(startEntries)
	start := &genpb.MarkovStartNode{}
	start.SetDistribution(starts)
	graph := &genpb.MarkovSamplingConfig{}
	graph.SetNodes(nodes)
	graph.SetStart(start)
	result := &genpb.QueryTemplateConfigs{}
	result.SetMarkovSampling(graph)
	return result, nil
}

// Each source has a lookahead record, retaining the last emitted action's
// completion interval even when the interleaver switches sources.
type sourceCursor struct {
	reader  *bufio.Reader
	current *auditpb.AuditLogEntry
	next    *auditpb.AuditLogEntry
}

func newSourceCursor(r io.Reader) (*sourceCursor, error) {
	c := &sourceCursor{reader: bufio.NewReader(r), current: &auditpb.AuditLogEntry{}, next: &auditpb.AuditLogEntry{}}
	if err := protodelim.UnmarshalFrom(c.reader, c.current); err != nil {
		return nil, err
	}
	if err := protodelim.UnmarshalFrom(c.reader, c.next); err != nil {
		return nil, err
	}
	return c, nil
}
func (c *sourceCursor) emit(out io.Writer, clock int64, segment string, needAnother bool) (int64, error) {
	delta := c.next.GetTimestampMs() - c.current.GetTimestampMs()
	if delta < 0 || clock > math.MaxInt64-delta {
		return clock, fmt.Errorf("invalid source timestamp interval")
	}
	entry := proto.Clone(c.current).(*auditpb.AuditLogEntry)
	entry.SetTimestampMs(clock)
	entry.SetIsGroundTruthAnomalous(segment != "")
	entry.SetGroundTruthSegmentId(segment)
	profile := "normal"
	if segment != "" {
		profile = "independent-random-graph"
	}
	entry.SetGroundTruthProfile(profile)
	if _, err := protodelim.MarshalTo(out, entry); err != nil {
		return clock, err
	}
	c.current = c.next
	if needAnother {
		c.next = &auditpb.AuditLogEntry{}
		if err := protodelim.UnmarshalFrom(c.reader, c.next); err != nil {
			return clock, err
		}
	}
	return clock + delta, nil
}
func interleaveIndependent(ctx context.Context, normal, anomaly io.Reader, out io.Writer, schedule anomalySchedule, start int64) error {
	n, err := newSourceCursor(normal)
	if err != nil {
		return err
	}
	a, err := newSourceCursor(anomaly)
	if err != nil {
		return err
	}
	var normalUsed, anomalyUsed int64
	segment := 0
	clock := start
	for position := int64(0); position < schedule.TotalActions; position++ {
		if err = ctx.Err(); err != nil {
			return err
		}
		for segment < len(schedule.Chunks) && position >= schedule.Chunks[segment].Start+schedule.Chunks[segment].Length {
			segment++
		}
		cursor := n
		id := ""
		normalRemaining := schedule.TotalActions - schedule.AnomalousActions - normalUsed
		needAnother := normalRemaining > 1
		if segment < len(schedule.Chunks) && position >= schedule.Chunks[segment].Start {
			cursor = a
			id = schedule.Chunks[segment].SegmentID
			anomalyUsed++
			needAnother = anomalyUsed < schedule.AnomalousActions
		} else {
			normalUsed++
		}
		clock, err = cursor.emit(out, clock, id, needAnother)
		if err != nil {
			return err
		}
	}
	if anomalyUsed != schedule.AnomalousActions {
		return fmt.Errorf("schedule does not match anomaly budget")
	}
	return nil
}

func generateIndependentValidation(ctx context.Context, o options, c *clientpb.MarkovExperimentConfig, out string) (string, error) {
	dir := filepath.Join(out, "validation")
	if err := os.MkdirAll(dir, 0755); err != nil {
		return "", err
	}
	b, err := os.ReadFile(filepath.Join(out, "generator_config.textproto"))
	if err != nil {
		return "", err
	}
	normal := &genpb.QueryTemplateConfigs{}
	if err = prototext.Unmarshal(b, normal); err != nil {
		return "", err
	}
	seed := c.GetValidationSeed()
	schedule := anomalySchedule{TotalActions: c.GetValidationActions(), AnomalousActions: c.GetValidationActions() / anomalyRecordDenominator, NormalSeed: seed, AnomalySeed: seed ^ 0x243f6a8885a308d3, GraphSeed: seed ^ 0x13198a2e03707344, ScheduleSeed: seed ^ 0x452821e638d01377}
	schedule.Chunks = randomAnomalySchedule(schedule.TotalActions, c.GetIndependentAnomalies().GetChunkActions(), schedule.ScheduleSeed)
	graph, err := independentAnomalyGraph(normal, c.GetIndependentAnomalies(), schedule.GraphSeed)
	if err != nil {
		return "", err
	}
	graphPath := filepath.Join(dir, "anomaly_generator.textproto")
	if err = os.WriteFile(graphPath, []byte(prototext.Format(graph)), 0644); err != nil {
		return "", err
	}
	data, err := json.MarshalIndent(schedule, "", "  ")
	if err != nil {
		return "", err
	}
	if err = os.WriteFile(filepath.Join(dir, "schedule.json"), data, 0644); err != nil {
		return "", err
	}
	for _, source := range []struct {
		name, path  string
		count, seed int64
	}{
		{"normal", filepath.Join(out, "generator_config.textproto"), schedule.TotalActions - schedule.AnomalousActions, schedule.NormalSeed},
		{"anomaly", graphPath, schedule.AnomalousActions, schedule.AnomalySeed},
	} {
		cmd := exec.CommandContext(ctx, o.generatorBin, "--config_file="+source.path, "--output_dir="+filepath.Join(dir, source.name), fmt.Sprintf("--num_logs=%d", source.count+1), fmt.Sprintf("--seed=%d", source.seed), fmt.Sprintf("--start_time_ms=%d", c.GetStartTimeMs()))
		cmd.Stdout, cmd.Stderr = os.Stdout, os.Stderr
		if err = cmd.Run(); err != nil {
			return "", err
		}
	}
	n, err := os.Open(filepath.Join(dir, "normal", "logs.binproto"))
	if err != nil {
		return "", err
	}
	defer n.Close()
	a, err := os.Open(filepath.Join(dir, "anomaly", "logs.binproto"))
	if err != nil {
		return "", err
	}
	defer a.Close()
	path := filepath.Join(dir, "logs.binproto")
	f, err := os.Create(path)
	if err != nil {
		return "", err
	}
	w := bufio.NewWriter(f)
	err = interleaveIndependent(ctx, n, a, w, schedule, c.GetStartTimeMs())
	flushErr := w.Flush()
	closeErr := f.Close()
	if err != nil {
		return "", err
	}
	if flushErr != nil {
		return "", flushErr
	}
	if closeErr != nil {
		return "", closeErr
	}
	if err = exportLogs(ctx, dir, schedule.TotalActions); err != nil {
		return "", err
	}
	return path, nil
}

// Randomize probabilities within each group while fixing the group's total mass.
func sampleAnomalyWeights(rng *rand.Rand, preferred []bool, probability float64) []float64 {
	weights := make([]float64, len(preferred))
	var all, yes, no float64
	for i := range weights {
		weights[i] = rng.ExpFloat64()
		if weights[i] == 0 {
			weights[i] = 1e-12
		}
		all += weights[i]
		if preferred[i] {
			yes += weights[i]
		} else {
			no += weights[i]
		}
	}
	for i := range weights {
		if yes == 0 {
			weights[i] /= all
		} else if preferred[i] {
			weights[i] = weights[i] / yes * probability
		} else {
			weights[i] = weights[i] / no * (1 - probability)
		}
	}
	return weights
}
