package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	cfgpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v1/generator/proto"
)

// Scenario 2 and 3: Bucket-based sampling.

// SampleQueryCount returns a Gaussian randomized query count for the template config, clamped to non-negative.
func SampleQueryCount(cfg *cfgpb.BucketQueryTemplateConfig, isSpike bool, currentHour int, currentWeekday time.Weekday, rng *rand.Rand) int {
	bCfg := cfg.GetBucketConfig()
	if bCfg == nil {
		return 0
	}
	rate := bCfg.GetQueriesPerBucketMean()
	profile := bCfg.GetWeeklyTrafficProfile()

	if profile != nil {
		// The weekly profile carries the full day-of-week x hour-of-day shape,
		// so it replaces the hour-only peaks and the weekend multiplier.
		// validateConfigs guarantees a well-formed profile; a malformed one
		// leaves the rate unshaped rather than failing mid-generation.
		multiplier := weeklyTrafficMultiplier(profile, currentWeekday, currentHour)
		// A silent slot stays silent for ordinary traffic, but an injected
		// attack still produces reads, matching the legacy silent weekend.
		if multiplier == 0.0 && isSpike {
			multiplier = 1.0
		}
		rate *= multiplier
	} else {
		isWeekend := (currentWeekday == time.Saturday || currentWeekday == time.Sunday)
		hasWeekendMult := bCfg.HasWeekendTrafficMultiplier()
		weekendMult := bCfg.GetWeekendTrafficMultiplier()
		isSilentWeekend := isWeekend && hasWeekendMult && weekendMult == 0.0

		// isSpike = false and silent weekend -> always 0 traffic
		if !isSpike && isSilentWeekend {
			return 0
		}

		// Apply Peak Windows on weekdays or busy weekends
		if !isSilentWeekend {
			for _, pw := range bCfg.GetPeakWindows() {
				rate = applyParabolicPeak(rate, int(pw.GetHourStart()), int(pw.GetHourEnd()), pw.GetMultiplier(), currentHour)
			}
		}

		// Apply Weekend Traffic Multiplier when weekendMult > 0
		if isWeekend && hasWeekendMult && weekendMult > 0.0 {
			rate *= weekendMult
		}
	}

	// Apply Anomalous Spike Multiplier for spike attacks
	if isSpike {
		rate *= bCfg.GetAnomalousSpikeMultiplier()
	}

	// A zero multiplier means no reads at all; do not let noise resurrect it.
	if profile != nil && rate == 0.0 {
		return 0
	}

	stdDev := bCfg.GetQueriesVariationStdDev()
	if profile != nil {
		// The profile changes the expected count, so the configured deviation
		// is interpreted at the baseline and scaled with the rate instead of
		// staying fixed in absolute reads.
		stdDev = scaledVariationStdDev(rate, bCfg.GetQueriesPerBucketMean(), stdDev)
	}

	if count := int(math.Round(rate + rng.NormFloat64()*stdDev)); count > 0 {
		return count
	}
	return 0
}

// scaledVariationStdDev returns the standard deviation for an expected count
// of rate, calibrated so that it equals stdDev at baseMean. The variance is
// Poisson at low counts plus the configured overdispersion, which grows with
// the rate. A non-positive stdDev keeps the old noiseless behavior.
func scaledVariationStdDev(rate, baseMean, stdDev float64) float64 {
	if stdDev <= 0 {
		return 0
	}
	if baseMean <= 0 {
		return stdDev
	}
	// Var(rate) = rate + overdispersion * rate^2, with the coefficient chosen
	// so Var(baseMean) = stdDev^2. If the configured deviation is below
	// Poisson noise, keep the Poisson term alone.
	overdispersion := (stdDev*stdDev - baseMean) / (baseMean * baseMean)
	if overdispersion < 0 {
		overdispersion = 0
	}
	return math.Sqrt(rate + overdispersion*rate*rate)
}

// weeklyTrafficMultiplier returns the profile multiplier for a UTC weekday and
// hour. Days are Monday-first and hours are 0..23; validateConfigs rejects
// profiles of the wrong shape, and out-of-range lookups leave the rate intact.
func weeklyTrafficMultiplier(profile *cfgpb.WeeklyTrafficProfile, weekday time.Weekday, hour int) float64 {
	dayIndex := (int(weekday) + 6) % 7 // time.Weekday starts at Sunday.
	days := profile.GetDays()
	if hour < 0 || hour > 23 || dayIndex >= len(days) {
		return 1.0
	}
	multipliers := days[dayIndex].GetHourlyMultipliers()
	if hour >= len(multipliers) {
		return 1.0
	}
	return multipliers[hour]
}

func applyParabolicPeak(rate float64, start, end int, multiplier float64, hour int) float64 {
	span := (end - start + 24) % 24
	offset := (hour - start + 24) % 24
	if multiplier <= 0 || span == 0 || offset >= span {
		return rate
	}
	x := float64(2*offset-span+1) / float64(span)
	return rate * (1.0 + (multiplier-1.0)*(1.0-x*x))
}

// SampleTimestampMs returns a timestamp (in milliseconds) within [bucketStart, bucketStart + bucketSizeMs).
func SampleTimestampMs(cfg *cfgpb.BucketQueryTemplateConfig, bucketStart time.Time, bucketSizeMs int64, rng *rand.Rand) int64 {
	startMs := bucketStart.UnixMilli()

	bCfg := cfg.GetBucketConfig()
	if bCfg == nil {
		return startMs + rng.Int63n(bucketSizeMs)
	}
	// Default: distribute timestamps uniformly within current bucket
	if !bCfg.GetSpikyTimestampsDistribution() {
		return startMs + rng.Int63n(bucketSizeMs)
	}

	const interval int64 = 1800000 // 30 minutes in milliseconds
	endMs := startMs + bucketSizeMs

	// Find the next 30-minute boundary.
	center := startMs + bucketSizeMs/2
	firstCenter := ((startMs + interval - 1) / interval) * interval

	// Align to a random 30-minute boundary if it falls within the bucket.
	if firstCenter < endMs {
		numCenters := (endMs-1-firstCenter)/interval + 1
		center = firstCenter + rng.Int63n(numCenters)*interval
	}

	// Add Gaussian jitter around the selected center.
	jitterStdDevMs := bucketSizeMs / 30
	if jitterStdDevMs <= 0 {
		jitterStdDevMs = 1
	}
	val := center + int64(rng.NormFloat64()*float64(jitterStdDevMs))

	// Clamp the timestamp to the bucket range.
	if val < startMs {
		return startMs
	}
	if val >= endMs {
		return endMs - 1
	}
	return val
}

// Scenario 1: Weighted random sampling.

// Sampler selects configs based on their probability weights.
type Sampler struct {
	configs     []*cfgpb.WeightedQueryTemplateConfig
	totalWeight float64
	randFloat   func() float64
}

// NewSampler initializes a Sampler with pre-validated configurations.
func NewSampler(configs []*cfgpb.WeightedQueryTemplateConfig, randFloat func() float64) *Sampler {
	if len(configs) == 0 {
		panic(fmt.Errorf("NewSampler: configs cannot be empty"))
	}

	var totalWeight float64
	for _, cfg := range configs {
		totalWeight += cfg.GetProbabilityWeight()
	}

	return &Sampler{
		configs:     configs,
		totalWeight: totalWeight,
		randFloat:   randFloat,
	}
}

// Sample returns a config using a O(k) linear scan based on accumulated weights.
func (s *Sampler) Sample() *cfgpb.WeightedQueryTemplateConfig {
	randomWeight := s.randFloat() * s.totalWeight

	var currentWeight float64
	for _, cfg := range s.configs {
		currentWeight += cfg.GetProbabilityWeight()

		// Use '<' to map correctly to the random interval [0.0, totalWeight), preventing boundary overlap.
		if randomWeight < currentWeight {
			return cfg
		}
	}

	// Fallback to the last config to catch rare floating-point precision underflows.
	return s.configs[len(s.configs)-1]
}
