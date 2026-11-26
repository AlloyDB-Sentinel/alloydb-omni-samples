package main

import (
	"encoding/binary"
	"fmt"
	"math/rand"
	"net/netip"
)

// IPRangeSampler picks random host addresses from one IPv4 CIDR range, e.g. "129.97.134.0/24".
type IPRangeSampler struct {
	prefix          netip.Prefix
	firstHostNumber uint32
	hostCount       int64
}

// NewIPRangeSampler parses the CIDR range once so Sample can be called many times cheaply.
// A range with host bits set, e.g. "129.97.134.77/24", is treated as its network, "129.97.134.0/24".
func NewIPRangeSampler(cidrRange string) (*IPRangeSampler, error) {
	prefix, err := netip.ParsePrefix(cidrRange)
	if err != nil {
		return nil, fmt.Errorf("invalid CIDR range %q: %w", cidrRange, err)
	}
	if !prefix.Addr().Is4() {
		return nil, fmt.Errorf("CIDR range %q: only IPv4 ranges are supported", cidrRange)
	}
	prefix = prefix.Masked()

	networkAddress := prefix.Addr().As4()
	networkNumber := binary.BigEndian.Uint32(networkAddress[:])
	rangeSize := int64(1) << (32 - prefix.Bits())

	// Exclude the network and broadcast addresses, which never belong to a connecting client.
	// /31 and /32 ranges have no such addresses, so every address in them is a host.
	if prefix.Bits() <= 30 {
		return &IPRangeSampler{
			prefix:          prefix,
			firstHostNumber: networkNumber + 1,
			hostCount:       rangeSize - 2,
		}, nil
	}
	return &IPRangeSampler{
		prefix:          prefix,
		firstHostNumber: networkNumber,
		hostCount:       rangeSize,
	}, nil
}

// Prefix returns the normalized CIDR range being sampled.
func (s *IPRangeSampler) Prefix() netip.Prefix {
	return s.prefix
}

// HostCount returns the number of distinct addresses Sample can return.
func (s *IPRangeSampler) HostCount() int64 {
	return s.hostCount
}

// Sample returns a uniformly random host address within the range.
func (s *IPRangeSampler) Sample(rng *rand.Rand) netip.Addr {
	hostOffset := uint32(rng.Int63n(s.hostCount))
	var hostAddress [4]byte
	binary.BigEndian.PutUint32(hostAddress[:], s.firstHostNumber+hostOffset)
	return netip.AddrFrom4(hostAddress)
}
