package main

import (
	"fmt"
	"io"
	"math"
	"math/rand"
	"time"

	"google.golang.org/protobuf/encoding/protodelim"

	genpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v1/generator/proto"
)

const (
	constUser   = "db_user"
	constDbName = "db_name"
)

// LogGenerator constructs AuditLogEntry protobufs.
type LogGenerator struct {
	rng *rand.Rand
}

// NewLogGenerator initializes a LogGenerator with a random source.
func NewLogGenerator(rng *rand.Rand) *LogGenerator {
	return &LogGenerator{
		rng: rng,
	}
}

// GenerateLog builds a query statement, constructs the AuditLogEntry protobuf, and streams it to the writer.
func (g *LogGenerator) GenerateLog(cfg *genpb.WeightedQueryTemplateConfig, writer io.Writer) error {
	statement := g.buildQueryStatement(cfg.GetTemplate())

	entry := &genpb.AuditLogEntry{
		NormalizedQueryTemplate: cfg.GetTemplate().GetNormalizedQueryTemplate(),
		QueryStatement:          statement,
		DbUser:                  constUser,
		DbName:                  constDbName,
		IsGroundTruthAnomalous:  cfg.GetTemplate().GetIsGroundTruthAnomalous(),
	}

	_, err := protodelim.MarshalTo(writer, entry)
	return err
}

// BuildBucketLog constructs an AuditLogEntry protobuf with custom timestamp and anomaly label.
func (g *LogGenerator) BuildBucketLog(cfg *genpb.BucketQueryTemplateConfig, isSpike bool, timestampMs int64) *genpb.AuditLogEntry {
	statement := g.buildQueryStatement(cfg.GetTemplate())
	dbUser := cfg.GetBucketConfig().GetDbUser()
	if dbUser == "" {
		dbUser = constUser
	}

	return &genpb.AuditLogEntry{
		NormalizedQueryTemplate: cfg.GetTemplate().GetNormalizedQueryTemplate(),
		QueryStatement:          statement,
		DbUser:                  dbUser,
		DbName:                  constDbName,
		IsGroundTruthAnomalous:  isSpike,
		TimestampMs:             timestampMs,
	}
}

// buildQueryStatement injects randomized values into placeholders (%v) to create the query statement.
func (g *LogGenerator) buildQueryStatement(cfg *genpb.QueryTemplateConfig) string {
	var vals []any
	for _, v := range cfg.GetPlaceholderValues() {
		if intSpec := v.GetIntPlaceholder(); intSpec != nil {
			vals = append(vals, intSpec.GetMin()+g.rng.Int31n(intSpec.GetMax()-intSpec.GetMin()+1))
		} else if floatSpec := v.GetFloatPlaceholder(); floatSpec != nil {
			vals = append(vals, floatSpec.GetMin()+g.rng.Float64()*(floatSpec.GetMax()-floatSpec.GetMin()))
		} else if stringSpec := v.GetStringPlaceholder(); stringSpec != nil {
			allowed := stringSpec.GetAllowedValues()
			if len(allowed) > 0 {
				idx := g.rng.Intn(len(allowed))
				vals = append(vals, allowed[idx])
			} else {
				vals = append(vals, "")
			}
		}
	}
	return fmt.Sprintf(cfg.GetNormalizedQueryTemplate(), vals...)
}

func (g *LogGenerator) appendPGAuditReads(logs []*genpb.AuditLogEntry, template *genpb.BucketQueryTemplateConfig, bucketStart, bucketEnd, trainEnd time.Time, bucketSize time.Duration) []*genpb.AuditLogEntry {
	config := template.GetBucketConfig()
	var sessionID string
	for phaseStart := bucketStart; phaseStart.Before(bucketEnd); {
		isTraining := phaseStart.Before(trainEnd)
		phaseEnd := bucketEnd
		if isTraining && trainEnd.Before(bucketEnd) {
			phaseEnd = trainEnd
		}

		isSpike := !isTraining && g.rng.Float64() < config.GetAnomalousSpikeProbability()
		count := SampleQueryCount(template, isSpike, phaseStart.Hour(), phaseStart.Weekday(), g.rng)
		phaseSize := phaseEnd.Sub(phaseStart)
		if phaseSize < bucketSize {
			// Prorate the counts so a split bucket does not produce two full buckets of traffic.
			count = int(math.Round(float64(count) * float64(phaseSize) / float64(bucketSize)))
		}

		if sessionID == "" {
			sessionID = fmt.Sprintf("%016x", g.rng.Uint64())
		}
		for i := 0; i < count; i++ {
			timestampMs := SampleTimestampMs(template, phaseStart, phaseSize.Milliseconds(), g.rng)
			isAnomalous := isSpike || (!isTraining && g.rng.Float64() < config.GetAnomalousRowsProbability())
			entry := g.BuildBucketLog(template, isAnomalous, timestampMs)
			entry.DatabaseSessionId = sessionID
			entry.RowsReturned = samplePGAuditRows(config, isAnomalous, g.rng)
			logs = append(logs, entry)
		}
		phaseStart = phaseEnd
	}
	return logs
}

func samplePGAuditRows(config *genpb.BucketSimulationConfig, isAnomalous bool, rng *rand.Rand) int64 {
	mean := config.GetRowsReturnedMean()
	// A query-count spike can use the normal rows per read.
	if multiplier := config.GetAnomalousRowsMultiplier(); isAnomalous && multiplier > 0 {
		mean *= multiplier
	}
	rows := math.Round(mean + rng.NormFloat64()*config.GetRowsReturnedStdDev())
	maxRows := config.GetRowsReturnedMax()
	if rows >= float64(maxRows) {
		return maxRows
	}
	if rows <= 0 {
		return 0
	}
	return int64(rows)
}
