package main

import (
	"fmt"
	"io"
	"math"
	"math/rand"
	"net/netip"
	"sort"
	"time"

	"google.golang.org/protobuf/encoding/protodelim"

	genpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v1/generator/proto"
)

// Connection sampling scenario.

const (
	// Linux default ephemeral port range, used by clients for outgoing connections.
	minClientPort = 32768
	maxClientPort = 60999

	// Linux default pid_max. PostgreSQL backend PIDs wrap around after this value.
	minBackendPid = 300
	maxBackendPid = 4194304

	millisecondsPerDay = int64(24 * time.Hour / time.Millisecond)

	// Above this mean, burst counts use a normal approximation instead of Poisson sampling.
	maxPoissonMean = 30.0
)

// clientAddressSource is one entry of a client address pool: either a fixed address or a CIDR range.
type clientAddressSource struct {
	fixedAddress netip.Addr
	rangeSampler *IPRangeSampler
}

// connectionProfileState holds one client profile's parsed settings and its schedule progress.
type connectionProfileState struct {
	profile        *genpb.ClientProfile
	addressSources []clientAddressSource

	outcomes                 []genpb.ConnectionOutcome
	cumulativeOutcomeWeights []float64

	// bucket_traffic schedule, wrapped so SampleQueryCount and SampleTimestampMs can be reused.
	bucketTrafficConfig *genpb.BucketQueryTemplateConfig

	// periodic schedule.
	periodicIntervalMs int64
	periodicJitterMs   int64
	nextScheduledMs    int64
	nextConnectionMs   int64

	// burst schedule.
	burstDurationMs int64
	// Burst connections that fall after the end of the bucket their burst started in.
	pendingBurstTimestampsMs []int64
}

// sampledConnection is a connection whose time and label are decided but which is not yet built.
type sampledConnection struct {
	profileState           *connectionProfileState
	timestampMs            int64
	isGroundTruthAnomalous bool
}

// GenerateConnectionLogs writes ConnectionLogEntry records bucket by bucket from startTime to endTime.
// Profiles are active according to their active_window, relative to trainEndTime.
func GenerateConnectionLogs(connectionCfg *genpb.ConnectionSamplingConfig, startTime, trainEndTime, endTime time.Time, bucketSize time.Duration, rng *rand.Rand, w io.Writer) error {
	if bucketSize < time.Millisecond {
		return fmt.Errorf("bucket size must be at least 1ms, got %v", bucketSize)
	}

	var profileStates []*connectionProfileState
	for i, profile := range connectionCfg.GetClientProfiles() {
		profileState, err := newConnectionProfileState(profile, startTime.UnixMilli(), rng)
		if err != nil {
			return fmt.Errorf("client_profiles[%d] (%q): %w", i, profile.GetProfileName(), err)
		}
		profileStates = append(profileStates, profileState)
	}

	builder := newConnectionLogBuilder(connectionCfg.GetSource(), rng)
	for bucketStart := startTime; bucketStart.Before(endTime); bucketStart = bucketStart.Add(bucketSize) {
		bucketEnd := bucketStart.Add(bucketSize)
		if bucketEnd.After(endTime) {
			bucketEnd = endTime
		}
		isTrainingPhase := bucketStart.Before(trainEndTime)

		var connections []sampledConnection
		for _, profileState := range profileStates {
			connections = append(connections, profileState.sampleConnections(bucketStart, bucketEnd, isTrainingPhase, rng)...)
		}

		// Sort chronologically to mirror the real PostgreSQL log sequence.
		sort.SliceStable(connections, func(i, j int) bool {
			return connections[i].timestampMs < connections[j].timestampMs
		})

		for _, connection := range connections {
			entry := builder.buildConnectionLog(connection, rng)
			if _, err := protodelim.MarshalTo(w, entry); err != nil {
				return fmt.Errorf("failed to write connection log: %w", err)
			}
		}
	}
	return nil
}

// newConnectionProfileState parses a validated client profile into ready-to-sample state.
func newConnectionProfileState(profile *genpb.ClientProfile, startMs int64, rng *rand.Rand) (*connectionProfileState, error) {
	profileState := &connectionProfileState{profile: profile}

	for _, address := range profile.GetClientAddressPool().GetAddresses() {
		parsedAddress, err := netip.ParseAddr(address)
		if err != nil {
			return nil, fmt.Errorf("invalid IP address %q: %w", address, err)
		}
		profileState.addressSources = append(profileState.addressSources, clientAddressSource{fixedAddress: parsedAddress})
	}
	for _, cidrRange := range profile.GetClientAddressPool().GetCidrRanges() {
		rangeSampler, err := NewIPRangeSampler(cidrRange)
		if err != nil {
			return nil, err
		}
		profileState.addressSources = append(profileState.addressSources, clientAddressSource{rangeSampler: rangeSampler})
	}
	if len(profileState.addressSources) == 0 {
		return nil, fmt.Errorf("client_address_pool is empty")
	}

	// An empty outcome list means every connection is authorized.
	var totalWeight float64
	for _, outcomeWeight := range profile.GetOutcomeWeights() {
		totalWeight += outcomeWeight.GetWeight()
		profileState.outcomes = append(profileState.outcomes, outcomeWeight.GetOutcome())
		profileState.cumulativeOutcomeWeights = append(profileState.cumulativeOutcomeWeights, totalWeight)
	}
	if len(profileState.outcomes) == 0 {
		profileState.outcomes = []genpb.ConnectionOutcome{genpb.ConnectionOutcome_CONNECTION_OUTCOME_AUTHORIZED}
		profileState.cumulativeOutcomeWeights = []float64{1.0}
	}

	switch {
	case profile.HasBucketTraffic():
		profileState.bucketTrafficConfig = &genpb.BucketQueryTemplateConfig{}
		profileState.bucketTrafficConfig.SetBucketConfig(profile.GetBucketTraffic())

	case profile.HasPeriodic():
		profileState.periodicIntervalMs = profile.GetPeriodic().GetInterval().AsDuration().Milliseconds()
		profileState.periodicJitterMs = profile.GetPeriodic().GetJitter().AsDuration().Milliseconds()
		if profileState.periodicIntervalMs < 1 {
			return nil, fmt.Errorf("periodic interval must be at least 1ms")
		}
		// Start at a random point in the first interval so profiles are not aligned to startMs.
		profileState.nextScheduledMs = startMs + rng.Int63n(profileState.periodicIntervalMs)
		profileState.nextConnectionMs = profileState.nextScheduledMs + rng.Int63n(profileState.periodicJitterMs+1)

	case profile.HasBurst():
		profileState.burstDurationMs = profile.GetBurst().GetBurstDuration().AsDuration().Milliseconds()
		if profileState.burstDurationMs < 1 {
			return nil, fmt.Errorf("burst_duration must be at least 1ms")
		}

	default:
		return nil, fmt.Errorf("one of bucket_traffic, periodic or burst must be set")
	}
	return profileState, nil
}

// isProfileActive reports whether a profile produces connections in the training or test phase.
func isProfileActive(activeWindow genpb.ActiveWindow, isTrainingPhase bool) bool {
	switch activeWindow {
	case genpb.ActiveWindow_ACTIVE_WINDOW_TEST_ONLY:
		return !isTrainingPhase
	case genpb.ActiveWindow_ACTIVE_WINDOW_TRAINING_ONLY:
		return isTrainingPhase
	default:
		return true
	}
}

// sampleConnections returns this profile's connections with timestamps in [bucketStart, bucketEnd).
func (s *connectionProfileState) sampleConnections(bucketStart, bucketEnd time.Time, isTrainingPhase bool, rng *rand.Rand) []sampledConnection {
	isActive := isProfileActive(s.profile.GetActiveWindow(), isTrainingPhase)
	isAttacker := s.profile.GetAttackerProfile() != genpb.AttackerProfile_ATTACKER_PROFILE_UNSPECIFIED
	bucketStartMs := bucketStart.UnixMilli()
	bucketEndMs := bucketEnd.UnixMilli()
	bucketDurationMs := bucketEndMs - bucketStartMs

	var connections []sampledConnection
	switch {
	case s.bucketTrafficConfig != nil:
		if !isActive {
			return nil
		}
		// As in bucket_sampling, spikes only occur in the test phase and are labeled anomalous.
		isSpike := !isTrainingPhase && rng.Float64() < s.bucketTrafficConfig.GetBucketConfig().GetAnomalousSpikeProbability()
		count := SampleQueryCount(s.bucketTrafficConfig, isSpike, bucketStart.Hour(), bucketStart.Weekday(), rng)
		for i := 0; i < count; i++ {
			connections = append(connections, sampledConnection{
				profileState:           s,
				timestampMs:            SampleTimestampMs(s.bucketTrafficConfig, bucketStart, bucketDurationMs, rng),
				isGroundTruthAnomalous: isAttacker || isSpike,
			})
		}

	case s.periodicIntervalMs > 0:
		// The schedule keeps advancing while inactive so it resumes on its own rhythm.
		// Jitter is less than the interval, so connection times are strictly increasing.
		for s.nextConnectionMs < bucketEndMs {
			if isActive && s.nextConnectionMs >= bucketStartMs {
				connections = append(connections, sampledConnection{
					profileState:           s,
					timestampMs:            s.nextConnectionMs,
					isGroundTruthAnomalous: isAttacker,
				})
			}
			s.nextScheduledMs += s.periodicIntervalMs
			s.nextConnectionMs = s.nextScheduledMs + rng.Int63n(s.periodicJitterMs+1)
		}

	case s.burstDurationMs > 0:
		if isActive {
			burstCfg := s.profile.GetBurst()
			expectedBursts := burstCfg.GetBurstsPerDay() * float64(bucketDurationMs) / float64(millisecondsPerDay)
			for burstCount := samplePoissonCount(expectedBursts, rng); burstCount > 0; burstCount-- {
				burstStartMs := bucketStartMs + rng.Int63n(bucketDurationMs)
				connectionCount := int(math.Round(burstCfg.GetConnectionsPerBurstMean() + rng.NormFloat64()*burstCfg.GetConnectionsPerBurstStdDev()))
				for i := 0; i < connectionCount; i++ {
					s.pendingBurstTimestampsMs = append(s.pendingBurstTimestampsMs, burstStartMs+rng.Int63n(s.burstDurationMs))
				}
			}
		}
		// Emit burst connections in this bucket; keep later ones for the buckets they fall in.
		var laterTimestampsMs []int64
		for _, timestampMs := range s.pendingBurstTimestampsMs {
			if timestampMs < bucketEndMs {
				connections = append(connections, sampledConnection{
					profileState:           s,
					timestampMs:            timestampMs,
					isGroundTruthAnomalous: isAttacker,
				})
			} else {
				laterTimestampsMs = append(laterTimestampsMs, timestampMs)
			}
		}
		s.pendingBurstTimestampsMs = laterTimestampsMs
	}
	return connections
}

// samplePoissonCount returns a random event count for the given expected count.
func samplePoissonCount(mean float64, rng *rand.Rand) int {
	if mean <= 0 {
		return 0
	}
	if mean > maxPoissonMean {
		count := int(math.Round(mean + rng.NormFloat64()*math.Sqrt(mean)))
		return max(count, 0)
	}
	// Knuth's algorithm: multiply uniform draws until the product drops below e^-mean.
	threshold := math.Exp(-mean)
	count := 0
	for product := rng.Float64(); product > threshold; product *= rng.Float64() {
		count++
	}
	return count
}

// connectionLogBuilder turns sampled connections into ConnectionLogEntry records.
type connectionLogBuilder struct {
	source         *genpb.SourceInstance
	nextBackendPid int32
}

func newConnectionLogBuilder(source *genpb.SourceInstance, rng *rand.Rand) *connectionLogBuilder {
	return &connectionLogBuilder{
		source:         source,
		nextBackendPid: int32(minBackendPid + rng.Intn(maxBackendPid-minBackendPid)),
	}
}

// buildConnectionLog fills one ConnectionLogEntry. Fields PostgreSQL would not have logged
// for the chosen outcome are left empty.
func (b *connectionLogBuilder) buildConnectionLog(connection sampledConnection, rng *rand.Rand) *genpb.ConnectionLogEntry {
	profileState := connection.profileState
	profile := profileState.profile
	outcome := profileState.sampleOutcome(rng)

	entry := &genpb.ConnectionLogEntry{}
	entry.SetTimestampMs(connection.timestampMs)
	entry.SetClientAddress(profileState.sampleClientAddress(rng).String())
	entry.SetClientPort(int32(minClientPort + rng.Intn(maxClientPort-minClientPort+1)))
	entry.SetOutcome(outcome)
	// The source is shared, read-only, by every entry; marshaling does not modify it.
	entry.SetSource(b.source)
	entry.SetBackendPid(b.takeBackendPid())
	entry.SetIsGroundTruthAnomalous(connection.isGroundTruthAnomalous)
	entry.SetGroundTruthProfile(profile.GetAttackerProfile())

	switch outcome {
	case genpb.ConnectionOutcome_CONNECTION_OUTCOME_AUTHORIZED:
		// "connection authenticated" and "connection authorized" lines carry all session details.
		entry.SetDbUser(profile.GetDbUser())
		entry.SetDbName(profile.GetDbName())
		entry.SetAuthMethod(profile.GetAuthMethod())
		entry.SetApplicationName(profile.GetApplicationName())
	case genpb.ConnectionOutcome_CONNECTION_OUTCOME_AUTH_FAILED:
		// The startup packet names the user and database, but the session is never authorized.
		entry.SetDbUser(profile.GetDbUser())
		entry.SetDbName(profile.GetDbName())
		entry.SetAuthMethod(profile.GetAuthMethod())
	}
	// CONNECTION_OUTCOME_DROPPED_BEFORE_AUTH: only "connection received" is logged.
	return entry
}

// takeBackendPid returns the next backend PID, increasing like PostgreSQL's forked backends.
func (b *connectionLogBuilder) takeBackendPid() int32 {
	backendPid := b.nextBackendPid
	b.nextBackendPid++
	if b.nextBackendPid >= maxBackendPid {
		b.nextBackendPid = minBackendPid
	}
	return backendPid
}

// sampleOutcome picks a connection outcome according to the profile's outcome weights.
func (s *connectionProfileState) sampleOutcome(rng *rand.Rand) genpb.ConnectionOutcome {
	totalWeight := s.cumulativeOutcomeWeights[len(s.cumulativeOutcomeWeights)-1]
	randomWeight := rng.Float64() * totalWeight
	for i, cumulativeWeight := range s.cumulativeOutcomeWeights {
		if randomWeight < cumulativeWeight {
			return s.outcomes[i]
		}
	}
	return s.outcomes[len(s.outcomes)-1]
}

// sampleClientAddress picks one pool entry uniformly, then an address within it if it is a range.
func (s *connectionProfileState) sampleClientAddress(rng *rand.Rand) netip.Addr {
	addressSource := s.addressSources[rng.Intn(len(s.addressSources))]
	if addressSource.rangeSampler != nil {
		return addressSource.rangeSampler.Sample(rng)
	}
	return addressSource.fixedAddress
}
