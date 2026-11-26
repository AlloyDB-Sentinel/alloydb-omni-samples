package main

import (
	"math/rand"
	"net/netip"
	"testing"
)

func TestNewIPRangeSampler(t *testing.T) {
	tests := []struct {
		name          string
		cidrRange     string
		wantPrefix    string
		wantHostCount int64
		wantErr       bool
	}{
		{
			name:          "slash_24_excludes_network_and_broadcast",
			cidrRange:     "129.97.134.0/24",
			wantPrefix:    "129.97.134.0/24",
			wantHostCount: 254,
		},
		{
			name:          "slash_30_has_two_hosts",
			cidrRange:     "10.0.0.0/30",
			wantPrefix:    "10.0.0.0/30",
			wantHostCount: 2,
		},
		{
			name:          "slash_31_uses_both_addresses",
			cidrRange:     "10.0.0.0/31",
			wantPrefix:    "10.0.0.0/31",
			wantHostCount: 2,
		},
		{
			name:          "slash_32_single_address",
			cidrRange:     "10.0.0.7/32",
			wantPrefix:    "10.0.0.7/32",
			wantHostCount: 1,
		},
		{
			name:          "slash_0_whole_ipv4_space",
			cidrRange:     "0.0.0.0/0",
			wantPrefix:    "0.0.0.0/0",
			wantHostCount: 1<<32 - 2,
		},
		{
			// Host bits set: treated as the enclosing network.
			name:          "host_bits_set_are_masked",
			cidrRange:     "129.97.134.77/24",
			wantPrefix:    "129.97.134.0/24",
			wantHostCount: 254,
		},
		{
			name:      "not_a_cidr_range",
			cidrRange: "not-a-cidr",
			wantErr:   true,
		},
		{
			name:      "missing_prefix_length",
			cidrRange: "129.97.134.0",
			wantErr:   true,
		},
		{
			name:      "prefix_length_out_of_range",
			cidrRange: "129.97.134.0/33",
			wantErr:   true,
		},
		{
			name:      "ipv4_octet_out_of_range",
			cidrRange: "129.97.256.0/24",
			wantErr:   true,
		},
		{
			name:      "ipv6_not_supported",
			cidrRange: "2001:db8::/64",
			wantErr:   true,
		},
		{
			name:      "ipv4_mapped_ipv6_not_supported",
			cidrRange: "::ffff:129.97.134.0/120",
			wantErr:   true,
		},
		{
			name:      "empty_string",
			cidrRange: "",
			wantErr:   true,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			sampler, err := NewIPRangeSampler(test.cidrRange)
			if (err != nil) != test.wantErr {
				t.Fatalf("NewIPRangeSampler(%q) error = %v, wantErr %v", test.cidrRange, err, test.wantErr)
			}
			if test.wantErr {
				return
			}
			if got := sampler.Prefix().String(); got != test.wantPrefix {
				t.Errorf("NewIPRangeSampler(%q).Prefix() = %s, want %s", test.cidrRange, got, test.wantPrefix)
			}
			if got := sampler.HostCount(); got != test.wantHostCount {
				t.Errorf("NewIPRangeSampler(%q).HostCount() = %d, want %d", test.cidrRange, got, test.wantHostCount)
			}
		})
	}
}

func TestIPRangeSamplerSample(t *testing.T) {
	tests := []struct {
		name       string
		cidrRange  string
		numSamples int
		// Addresses that must never be returned.
		forbiddenAddresses []string
		// When set, every address in this list must be returned at least once.
		wantAllAddresses []string
	}{
		{
			name:               "slash_24_stays_in_range_and_skips_network_and_broadcast",
			cidrRange:          "129.97.134.0/24",
			numSamples:         20000,
			forbiddenAddresses: []string{"129.97.134.0", "129.97.134.255"},
		},
		{
			name:               "slash_30_returns_only_its_two_hosts",
			cidrRange:          "10.0.0.0/30",
			numSamples:         1000,
			forbiddenAddresses: []string{"10.0.0.0", "10.0.0.3"},
			wantAllAddresses:   []string{"10.0.0.1", "10.0.0.2"},
		},
		{
			name:             "slash_31_returns_both_addresses",
			cidrRange:        "10.0.0.0/31",
			numSamples:       1000,
			wantAllAddresses: []string{"10.0.0.0", "10.0.0.1"},
		},
		{
			name:             "slash_32_always_returns_its_address",
			cidrRange:        "10.0.0.7/32",
			numSamples:       100,
			wantAllAddresses: []string{"10.0.0.7"},
		},
		{
			name:               "range_crossing_octet_boundary",
			cidrRange:          "10.0.0.0/23",
			numSamples:         20000,
			forbiddenAddresses: []string{"10.0.0.0", "10.0.1.255"},
		},
		{
			name:               "whole_ipv4_space",
			cidrRange:          "0.0.0.0/0",
			numSamples:         1000,
			forbiddenAddresses: []string{"0.0.0.0", "255.255.255.255"},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			sampler, err := NewIPRangeSampler(test.cidrRange)
			if err != nil {
				t.Fatalf("NewIPRangeSampler(%q) error = %v", test.cidrRange, err)
			}
			rng := rand.New(rand.NewSource(42))

			sampledAddresses := make(map[netip.Addr]bool)
			for i := 0; i < test.numSamples; i++ {
				address := sampler.Sample(rng)
				if !sampler.Prefix().Contains(address) {
					t.Fatalf("Sample() = %s, outside range %s", address, sampler.Prefix())
				}
				sampledAddresses[address] = true
			}

			for _, forbidden := range test.forbiddenAddresses {
				if sampledAddresses[netip.MustParseAddr(forbidden)] {
					t.Errorf("Sample() returned %s, which must never be returned for %s", forbidden, test.cidrRange)
				}
			}
			for _, wanted := range test.wantAllAddresses {
				if !sampledAddresses[netip.MustParseAddr(wanted)] {
					t.Errorf("Sample() never returned %s in %d samples of %s", wanted, test.numSamples, test.cidrRange)
				}
			}
		})
	}
}

func TestIPRangeSamplerSampleCoversWholeRange(t *testing.T) {
	// With enough samples, every host in a small range should appear.
	sampler, err := NewIPRangeSampler("192.168.1.0/28")
	if err != nil {
		t.Fatalf("NewIPRangeSampler() error = %v", err)
	}
	rng := rand.New(rand.NewSource(7))

	sampledAddresses := make(map[netip.Addr]bool)
	for i := 0; i < 5000; i++ {
		sampledAddresses[sampler.Sample(rng)] = true
	}
	if int64(len(sampledAddresses)) != sampler.HostCount() {
		t.Errorf("Sample() returned %d distinct addresses, want all %d hosts", len(sampledAddresses), sampler.HostCount())
	}
}

func TestIPRangeSamplerSampleIsDeterministicForSeed(t *testing.T) {
	sampler, err := NewIPRangeSampler("129.97.134.0/24")
	if err != nil {
		t.Fatalf("NewIPRangeSampler() error = %v", err)
	}
	firstRng := rand.New(rand.NewSource(99))
	secondRng := rand.New(rand.NewSource(99))

	for i := 0; i < 100; i++ {
		firstAddress := sampler.Sample(firstRng)
		secondAddress := sampler.Sample(secondRng)
		if firstAddress != secondAddress {
			t.Fatalf("sample %d: got %s and %s from the same seed, want identical", i, firstAddress, secondAddress)
		}
	}
}
