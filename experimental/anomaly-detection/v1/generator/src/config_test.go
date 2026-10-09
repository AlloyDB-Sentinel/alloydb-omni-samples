package main

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/google/go-cmp/cmp"
	"google.golang.org/protobuf/encoding/prototext"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/testing/protocmp"

	cfgpb "github.com/GoogleCloudPlatform/alloydb-omni-samples/experimental/anomaly-detection/v1/generator/proto"
)

// parseProto unmarshals a textproto string into a protobuf message.
// It is shared as a package-level helper across all test files.
func parseProto[T proto.Message](t *testing.T, text string, msg T) T {
	t.Helper()
	if err := prototext.Unmarshal([]byte(text), msg); err != nil {
		t.Fatalf("Failed to parse textproto: %v", err)
	}
	return msg
}

func TestValidateConfigs(t *testing.T) {
	tests := []struct {
		name    string
		input   *cfgpb.QueryTemplateConfigs
		wantErr bool
	}{
		{
			name:    "no_scenario_selected",
			input:   &cfgpb.QueryTemplateConfigs{},
			wantErr: true,
		},

		// Scenario 1: Weighted Sampling Checks.
		{
			name: "no_templates_specified",
			input: parseProto(t, `
weighted_sampling {
}`, &cfgpb.QueryTemplateConfigs{}),
			wantErr: true,
		},
		{
			name: "nan_probability_weight",
			input: parseProto(t, `
weighted_sampling {
	query_template_configs: {
		template: {
			normalized_query_template: "SELECT 1"
		}
		probability_weight: nan
	}
}`, &cfgpb.QueryTemplateConfigs{}),
			wantErr: true,
		},
		{
			name: "negative_probability_weight",
			input: parseProto(t, `
weighted_sampling {
	query_template_configs: {
		template: {
			normalized_query_template: "SELECT 1"
		}
		probability_weight: -1.0
	}
}`, &cfgpb.QueryTemplateConfigs{}),
			wantErr: true,
		},
		{
			name: "placeholder_count_mismatch",
			input: parseProto(t, `
weighted_sampling {
	query_template_configs: {
		template: {
			normalized_query_template: "SELECT * FROM users WHERE age > %v AND salary > %v"
			placeholder_values: { int_placeholder: { min: 18 max: 65 } }
		}
		probability_weight: 1.0
	}
}`, &cfgpb.QueryTemplateConfigs{}),
			wantErr: true,
		},
		{
			name: "integer_placeholder_min_greater_than_max",
			input: parseProto(t, `
weighted_sampling {
	query_template_configs: {
		template: {
			normalized_query_template: "SELECT * FROM users WHERE age > %v"
			placeholder_values: { int_placeholder: { min: 65 max: 18 } }
		}
		probability_weight: 1.0
	}
}`, &cfgpb.QueryTemplateConfigs{}),
			wantErr: true,
		},
		{
			name: "float_placeholder_min_greater_than_max",
			input: parseProto(t, `
weighted_sampling {
	query_template_configs: {
		template: {
			normalized_query_template: "SELECT * FROM users WHERE salary > %v"
			placeholder_values: { float_placeholder: { min: 50.5 max: 10.2 } }
		}
		probability_weight: 1.0
	}
}`, &cfgpb.QueryTemplateConfigs{}),
			wantErr: true,
		},
		{
			name: "string_placeholder_empty_allowed_values",
			input: parseProto(t, `
weighted_sampling {
	query_template_configs: {
		template: {
			normalized_query_template: "SELECT * FROM users WHERE name = %v"
			placeholder_values: { string_placeholder: { } }
		}
		probability_weight: 1.0
	}
}`, &cfgpb.QueryTemplateConfigs{}),
			wantErr: true,
		},
		{
			name: "missing_placeholder_type_selection",
			input: parseProto(t, `
weighted_sampling {
	query_template_configs: {
		template: {
			normalized_query_template: "SELECT * FROM users WHERE name = %v"
			placeholder_values: { }
		}
		probability_weight: 1.0
	}
}`, &cfgpb.QueryTemplateConfigs{}),
			wantErr: true,
		},
		{
			name: "zero_probability_weight_all",
			input: parseProto(t, `
weighted_sampling {
	query_template_configs: {
		template: {
			normalized_query_template: "SELECT 1"
		}
		probability_weight: 0.0
	}
}`, &cfgpb.QueryTemplateConfigs{}),
			wantErr: true,
		},
		{
			name: "zero_probability_weight_one_active",
			input: parseProto(t, `
weighted_sampling {
	query_template_configs: {
		template: {
			normalized_query_template: "SELECT 1"
		}
		probability_weight: 0.0
	}
	query_template_configs: {
		template: {
			normalized_query_template: "SELECT 2"
		}
		probability_weight: 0.5
	}
}`, &cfgpb.QueryTemplateConfigs{}),
			wantErr: false,
		},
		{
			name: "infinite_probability_weight",
			input: parseProto(t, `
weighted_sampling {
	query_template_configs: {
		template: {
			normalized_query_template: "SELECT 1"
		}
		probability_weight: inf
	}
}`, &cfgpb.QueryTemplateConfigs{}),
			wantErr: true,
		},
		{
			name: "valid_weighted_sampling",
			input: parseProto(t, `
weighted_sampling {
	query_template_configs: {
		template: {
			normalized_query_template: "SELECT * FROM users WHERE age > %v AND salary > %v AND name = %v"
			placeholder_values: { int_placeholder: { min: 18 max: 65 } }
			placeholder_values: { float_placeholder: { min: 1000.0 max: 9999.5 } }
			placeholder_values: { string_placeholder: { allowed_values: ["alice", "bob"] } }
		}
		probability_weight: 1.0
	}
}`, &cfgpb.QueryTemplateConfigs{}),
			wantErr: false,
		},

		// Scenario 2: Bucket Sampling Checks.
		{
			name: "bucket_no_templates_specified",
			input: parseProto(t, `
bucket_sampling {
}`, &cfgpb.QueryTemplateConfigs{}),
			wantErr: true,
		},
		{
			name: "nil_bucket_config",
			input: parseProto(t, `
bucket_sampling {
	query_template_configs: {
		template: {
			normalized_query_template: "SELECT 1"
		}
	}
}`, &cfgpb.QueryTemplateConfigs{}),
			wantErr: true,
		},
		{
			name: "negative_queries_mean",
			input: parseProto(t, `
bucket_sampling {
	query_template_configs: {
		template: {
			normalized_query_template: "SELECT 1"
		}
		bucket_config: {
			queries_per_bucket_mean: -10.0
		}
	}
}`, &cfgpb.QueryTemplateConfigs{}),
			wantErr: true,
		},
		{
			name: "negative_queries_std_dev",
			input: parseProto(t, `
bucket_sampling {
	query_template_configs: {
		template: {
			normalized_query_template: "SELECT 1"
		}
		bucket_config: {
			queries_per_bucket_mean: 10.0
			queries_variation_std_dev: -2.0
		}
	}
}`, &cfgpb.QueryTemplateConfigs{}),
			wantErr: true,
		},
		{
			name: "invalid_spike_probability",
			input: parseProto(t, `
bucket_sampling {
	query_template_configs: {
		template: {
			normalized_query_template: "SELECT 1"
		}
		bucket_config: {
			queries_per_bucket_mean: 10.0
			anomalous_spike_probability: -0.1
		}
	}
}`, &cfgpb.QueryTemplateConfigs{}),
			wantErr: true,
		},
		{
			name: "negative_spike_multiplier",
			input: parseProto(t, `
bucket_sampling {
	query_template_configs: {
		template: {
			normalized_query_template: "SELECT 1"
		}
		bucket_config: {
			queries_per_bucket_mean: 10.0
			anomalous_spike_multiplier: -1.0
		}
	}
}`, &cfgpb.QueryTemplateConfigs{}),
			wantErr: true,
		},
		{
			name: "negative_peak_window_multiplier",
			input: parseProto(t, `
bucket_sampling {
	query_template_configs: {
		template: {
			normalized_query_template: "SELECT 1"
		}
		bucket_config: {
			queries_per_bucket_mean: 10.0
			peak_windows: {
				window_name: "day"
				multiplier: -0.5
			}
		}
	}
}`, &cfgpb.QueryTemplateConfigs{}),
			wantErr: true,
		},
		{
			name: "peak_window_hours_out_of_bounds",
			input: parseProto(t, `
bucket_sampling {
	query_template_configs: {
		template: {
			normalized_query_template: "SELECT 1"
		}
		bucket_config: {
			queries_per_bucket_mean: 10.0
			peak_windows: {
				window_name: "day"
				multiplier: 1.5
				hour_start: -1
				hour_end: 24
			}
		}
	}
}`, &cfgpb.QueryTemplateConfigs{}),
			wantErr: true,
		},
		{
			name: "peak_window_hours_equal",
			input: parseProto(t, `
bucket_sampling {
	query_template_configs: {
		template: {
			normalized_query_template: "SELECT 1"
		}
		bucket_config: {
			queries_per_bucket_mean: 10.0
			peak_windows: {
				window_name: "night"
				multiplier: 1.5
				hour_start: 22
				hour_end: 22
			}
		}
	}
}`, &cfgpb.QueryTemplateConfigs{}),
			wantErr: true,
		},
		{
			name: "peak_window_start_hour_greater_than_end_hour_midnight_crossing",
			input: parseProto(t, `
bucket_sampling {
	query_template_configs: {
		template: {
			normalized_query_template: "SELECT 1"
		}
		bucket_config: {
			queries_per_bucket_mean: 10.0
			anomalous_spike_probability: 0.05
			anomalous_spike_multiplier: 2.0
			peak_windows: {
				window_name: "day"
				multiplier: 1.5
				hour_start: 17
				hour_end: 9
			}
		}
	}
}`, &cfgpb.QueryTemplateConfigs{}),
			wantErr: false,
		},
		{
			name: "bucket_placeholder_count_mismatch",
			input: parseProto(t, `
bucket_sampling {
	query_template_configs: {
		template: {
			normalized_query_template: "SELECT * FROM users WHERE age > %v AND salary > %v"
			placeholder_values: { int_placeholder: { min: 18 max: 65 } }
		}
		bucket_config: {
			queries_per_bucket_mean: 10.0
		}
	}
}`, &cfgpb.QueryTemplateConfigs{}),
			wantErr: true,
		},
		{
			name: "negative_weekend_traffic_multiplier",
			input: parseProto(t, `
bucket_sampling {
	query_template_configs: {
		template: {
			normalized_query_template: "SELECT 1"
		}
		bucket_config: {
			queries_per_bucket_mean: 10.0
			weekend_traffic_multiplier: -1.0
		}
	}
}`, &cfgpb.QueryTemplateConfigs{}),
			wantErr: true,
		},
		{
			name: "nan_weekend_traffic_multiplier",
			input: parseProto(t, `
bucket_sampling {
	query_template_configs: {
		template: {
			normalized_query_template: "SELECT 1"
		}
		bucket_config: {
			queries_per_bucket_mean: 10.0
			weekend_traffic_multiplier: nan
		}
	}
}`, &cfgpb.QueryTemplateConfigs{}),
			wantErr: true,
		},
		{
			name: "infinite_weekend_traffic_multiplier",
			input: parseProto(t, `
bucket_sampling {
	query_template_configs: {
		template: {
			normalized_query_template: "SELECT 1"
		}
		bucket_config: {
			queries_per_bucket_mean: 10.0
			weekend_traffic_multiplier: inf
		}
	}
}`, &cfgpb.QueryTemplateConfigs{}),
			wantErr: true,
		},
		{
			name: "zero_queries_per_bucket_mean_all",
			input: parseProto(t, `
bucket_sampling {
	query_template_configs: {
		template: {
			normalized_query_template: "SELECT 1"
		}
		bucket_config: {
			queries_per_bucket_mean: 0.0
		}
	}
}`, &cfgpb.QueryTemplateConfigs{}),
			wantErr: true,
		},
		{
			name: "valid_bucket_sampling",
			input: parseProto(t, `
bucket_sampling {
	query_template_configs: {
		template: {
			normalized_query_template: "SELECT * FROM users WHERE id = %v"
			placeholder_values: { int_placeholder: { min: 1 max: 10 } }
		}
		bucket_config: {
			queries_per_bucket_mean: 10.0
			queries_variation_std_dev: 2.0
			anomalous_spike_probability: 0.1
			anomalous_spike_multiplier: 1.5
			peak_windows: {
				window_name: "day"
				hour_start: 9
				hour_end: 17
				multiplier: 2.0
			}
			peak_windows: {
				window_name: "night"
				hour_start: 18
				hour_end: 22
				multiplier: 1.5
			}
		}
	}
}`, &cfgpb.QueryTemplateConfigs{}),
			wantErr: false,
		},

		{
			name: "overlapping_peak_windows_error",
			input: parseProto(t, `
bucket_sampling {
	query_template_configs: {
		template: {
			normalized_query_template: "SELECT 1"
		}
		bucket_config: {
			queries_per_bucket_mean: 10.0
			anomalous_spike_probability: 0.05
			anomalous_spike_multiplier: 2.0
			peak_windows: { window_name: "w1" hour_start: 9 hour_end: 14 multiplier: 1.5 }
			peak_windows: { window_name: "w2" hour_start: 12 hour_end: 17 multiplier: 1.5 }
		}
	}
}`, &cfgpb.QueryTemplateConfigs{}),
			wantErr: true,
		},
		{
			name: "valid_scenario3",
			input: parseProto(t, `
bucket_sampling {
	query_template_configs: {
		template: {
			normalized_query_template: "SELECT 1"
		}
		bucket_config: {
			queries_per_bucket_mean: 10.0
			weekend_traffic_multiplier: 0.0
			peak_windows: { window_name: "day" hour_start: 9 hour_end: 17 multiplier: 1.5 }
		}
	}
}`, &cfgpb.QueryTemplateConfigs{}),
			wantErr: false,
		},

		// Connection Sampling Checks.
		{
			name: "connection_no_client_profiles",
			input: parseProto(t, `
connection_sampling {
}`, &cfgpb.QueryTemplateConfigs{}),
			wantErr: true,
		},
		{
			name: "connection_empty_address_pool",
			input: parseProto(t, `
connection_sampling {
	client_profiles: {
		db_user: "app"
		periodic: { interval: { seconds: 600 } }
	}
}`, &cfgpb.QueryTemplateConfigs{}),
			wantErr: true,
		},
		{
			name: "connection_invalid_address",
			input: parseProto(t, `
connection_sampling {
	client_profiles: {
		client_address_pool: { addresses: "10.0.0.256" }
		db_user: "app"
		periodic: { interval: { seconds: 600 } }
	}
}`, &cfgpb.QueryTemplateConfigs{}),
			wantErr: true,
		},
		{
			name: "connection_invalid_cidr_range",
			input: parseProto(t, `
connection_sampling {
	client_profiles: {
		client_address_pool: { cidr_ranges: "203.0.113.0/33" }
		db_user: "app"
		periodic: { interval: { seconds: 600 } }
	}
}`, &cfgpb.QueryTemplateConfigs{}),
			wantErr: true,
		},
		{
			name: "connection_unspecified_outcome",
			input: parseProto(t, `
connection_sampling {
	client_profiles: {
		client_address_pool: { addresses: "10.0.0.12" }
		db_user: "app"
		outcome_weights: { weight: 1.0 }
		periodic: { interval: { seconds: 600 } }
	}
}`, &cfgpb.QueryTemplateConfigs{}),
			wantErr: true,
		},
		{
			name: "connection_negative_outcome_weight",
			input: parseProto(t, `
connection_sampling {
	client_profiles: {
		client_address_pool: { addresses: "10.0.0.12" }
		db_user: "app"
		outcome_weights: { outcome: CONNECTION_OUTCOME_AUTHORIZED weight: -1.0 }
		periodic: { interval: { seconds: 600 } }
	}
}`, &cfgpb.QueryTemplateConfigs{}),
			wantErr: true,
		},
		{
			name: "connection_all_outcome_weights_zero",
			input: parseProto(t, `
connection_sampling {
	client_profiles: {
		client_address_pool: { addresses: "10.0.0.12" }
		outcome_weights: { outcome: CONNECTION_OUTCOME_AUTH_FAILED weight: 0.0 }
		periodic: { interval: { seconds: 600 } }
	}
}`, &cfgpb.QueryTemplateConfigs{}),
			wantErr: true,
		},
		{
			name: "connection_default_authorized_without_db_user",
			input: parseProto(t, `
connection_sampling {
	client_profiles: {
		client_address_pool: { addresses: "10.0.0.12" }
		periodic: { interval: { seconds: 600 } }
	}
}`, &cfgpb.QueryTemplateConfigs{}),
			wantErr: true,
		},
		{
			name: "connection_authorized_weight_without_db_user",
			input: parseProto(t, `
connection_sampling {
	client_profiles: {
		client_address_pool: { addresses: "10.0.0.12" }
		outcome_weights: { outcome: CONNECTION_OUTCOME_AUTHORIZED weight: 0.1 }
		outcome_weights: { outcome: CONNECTION_OUTCOME_AUTH_FAILED weight: 0.9 }
		periodic: { interval: { seconds: 600 } }
	}
}`, &cfgpb.QueryTemplateConfigs{}),
			wantErr: true,
		},
		{
			name: "connection_no_schedule",
			input: parseProto(t, `
connection_sampling {
	client_profiles: {
		client_address_pool: { addresses: "10.0.0.12" }
		db_user: "app"
	}
}`, &cfgpb.QueryTemplateConfigs{}),
			wantErr: true,
		},
		{
			name: "connection_bucket_traffic_invalid_spike_probability",
			input: parseProto(t, `
connection_sampling {
	client_profiles: {
		client_address_pool: { addresses: "10.0.0.12" }
		db_user: "app"
		bucket_traffic: { queries_per_bucket_mean: 10.0 anomalous_spike_probability: 1.5 }
	}
}`, &cfgpb.QueryTemplateConfigs{}),
			wantErr: true,
		},
		{
			name: "connection_bucket_traffic_overlapping_peak_windows",
			input: parseProto(t, `
connection_sampling {
	client_profiles: {
		client_address_pool: { addresses: "10.0.0.12" }
		db_user: "app"
		bucket_traffic: {
			queries_per_bucket_mean: 10.0
			peak_windows: { window_name: "w1" hour_start: 9 hour_end: 14 multiplier: 1.5 }
			peak_windows: { window_name: "w2" hour_start: 12 hour_end: 17 multiplier: 1.5 }
		}
	}
}`, &cfgpb.QueryTemplateConfigs{}),
			wantErr: true,
		},
		{
			name: "connection_periodic_missing_interval",
			input: parseProto(t, `
connection_sampling {
	client_profiles: {
		client_address_pool: { cidr_ranges: "203.0.113.0/24" }
		outcome_weights: { outcome: CONNECTION_OUTCOME_DROPPED_BEFORE_AUTH weight: 1.0 }
		periodic: { }
	}
}`, &cfgpb.QueryTemplateConfigs{}),
			wantErr: true,
		},
		{
			name: "connection_periodic_negative_jitter",
			input: parseProto(t, `
connection_sampling {
	client_profiles: {
		client_address_pool: { cidr_ranges: "203.0.113.0/24" }
		outcome_weights: { outcome: CONNECTION_OUTCOME_DROPPED_BEFORE_AUTH weight: 1.0 }
		periodic: { interval: { seconds: 600 } jitter: { seconds: -60 } }
	}
}`, &cfgpb.QueryTemplateConfigs{}),
			wantErr: true,
		},
		{
			name: "connection_periodic_jitter_not_less_than_interval",
			input: parseProto(t, `
connection_sampling {
	client_profiles: {
		client_address_pool: { cidr_ranges: "203.0.113.0/24" }
		outcome_weights: { outcome: CONNECTION_OUTCOME_DROPPED_BEFORE_AUTH weight: 1.0 }
		periodic: { interval: { seconds: 600 } jitter: { seconds: 600 } }
	}
}`, &cfgpb.QueryTemplateConfigs{}),
			wantErr: true,
		},
		{
			name: "connection_burst_zero_bursts_per_day",
			input: parseProto(t, `
connection_sampling {
	client_profiles: {
		client_address_pool: { addresses: "198.51.100.23" }
		outcome_weights: { outcome: CONNECTION_OUTCOME_AUTH_FAILED weight: 1.0 }
		burst: { connections_per_burst_mean: 500.0 burst_duration: { seconds: 300 } }
	}
}`, &cfgpb.QueryTemplateConfigs{}),
			wantErr: true,
		},
		{
			name: "connection_burst_zero_connections_per_burst",
			input: parseProto(t, `
connection_sampling {
	client_profiles: {
		client_address_pool: { addresses: "198.51.100.23" }
		outcome_weights: { outcome: CONNECTION_OUTCOME_AUTH_FAILED weight: 1.0 }
		burst: { bursts_per_day: 0.5 burst_duration: { seconds: 300 } }
	}
}`, &cfgpb.QueryTemplateConfigs{}),
			wantErr: true,
		},
		{
			name: "connection_burst_nan_std_dev",
			input: parseProto(t, `
connection_sampling {
	client_profiles: {
		client_address_pool: { addresses: "198.51.100.23" }
		outcome_weights: { outcome: CONNECTION_OUTCOME_AUTH_FAILED weight: 1.0 }
		burst: {
			bursts_per_day: 0.5
			connections_per_burst_mean: 500.0
			connections_per_burst_std_dev: nan
			burst_duration: { seconds: 300 }
		}
	}
}`, &cfgpb.QueryTemplateConfigs{}),
			wantErr: true,
		},
		{
			name: "connection_burst_missing_duration",
			input: parseProto(t, `
connection_sampling {
	client_profiles: {
		client_address_pool: { addresses: "198.51.100.23" }
		outcome_weights: { outcome: CONNECTION_OUTCOME_AUTH_FAILED weight: 1.0 }
		burst: { bursts_per_day: 0.5 connections_per_burst_mean: 500.0 }
	}
}`, &cfgpb.QueryTemplateConfigs{}),
			wantErr: true,
		},
		{
			name: "connection_second_profile_invalid",
			input: parseProto(t, `
connection_sampling {
	client_profiles: {
		client_address_pool: { addresses: "10.0.0.12" }
		db_user: "app"
		periodic: { interval: { seconds: 600 } }
	}
	client_profiles: {
		client_address_pool: { addresses: "not-an-ip" }
		db_user: "app"
		periodic: { interval: { seconds: 600 } }
	}
}`, &cfgpb.QueryTemplateConfigs{}),
			wantErr: true,
		},
		{
			name: "valid_connection_sampling",
			input: parseProto(t, `
connection_sampling {
	source: { project: "test-project" cluster_id: "test-cluster" instance_id: "primary" node_id: "v1" }
	client_profiles: {
		profile_name: "app-server"
		client_address_pool: { addresses: "10.0.0.12" addresses: "::1" }
		db_user: "app"
		bucket_traffic: {
			queries_per_bucket_mean: 40.0
			queries_variation_std_dev: 8.0
			peak_windows: { window_name: "day" hour_start: 9 hour_end: 17 multiplier: 3.0 }
		}
		active_window: ACTIVE_WINDOW_ALL
	}
	client_profiles: {
		profile_name: "internet-scanner"
		client_address_pool: { cidr_ranges: "203.0.113.0/24" }
		outcome_weights: { outcome: CONNECTION_OUTCOME_DROPPED_BEFORE_AUTH weight: 0.7 }
		outcome_weights: { outcome: CONNECTION_OUTCOME_AUTH_FAILED weight: 0.3 }
		periodic: { interval: { seconds: 600 } jitter: { seconds: 60 } }
		active_window: ACTIVE_WINDOW_TEST_ONLY
		attacker_profile: ATTACKER_PROFILE_INTERNET_SCANNER
	}
	client_profiles: {
		profile_name: "burst-attacker"
		client_address_pool: { addresses: "198.51.100.23" }
		db_user: "postgres"
		outcome_weights: { outcome: CONNECTION_OUTCOME_AUTH_FAILED weight: 0.98 }
		outcome_weights: { outcome: CONNECTION_OUTCOME_AUTHORIZED weight: 0.02 }
		burst: {
			bursts_per_day: 0.5
			connections_per_burst_mean: 500.0
			connections_per_burst_std_dev: 100.0
			burst_duration: { seconds: 300 }
		}
		active_window: ACTIVE_WINDOW_TEST_ONLY
		attacker_profile: ATTACKER_PROFILE_BURST_ATTACKER
	}
}`, &cfgpb.QueryTemplateConfigs{}),
			wantErr: false,
		},
		{
			name: "valid_connection_periodic_without_jitter",
			input: parseProto(t, `
connection_sampling {
	client_profiles: {
		client_address_pool: { cidr_ranges: "203.0.113.0/24" }
		outcome_weights: { outcome: CONNECTION_OUTCOME_DROPPED_BEFORE_AUTH weight: 1.0 }
		periodic: { interval: { seconds: 600 } }
	}
}`, &cfgpb.QueryTemplateConfigs{}),
			wantErr: false,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			err := validateConfigs(test.input)
			if (err != nil) != test.wantErr {
				t.Errorf("validateConfigs() error = %v, wantErr %v", err, test.wantErr)
			}
		})
	}
}

func TestLoadConfigs(t *testing.T) {
	wantValidConfigs := parseProto(t, `
weighted_sampling {
  query_template_configs: {
    template: {
      normalized_query_template: "SELECT * FROM table1 WHERE id = %v"
      placeholder_values: { int_placeholder: { min: 1 max: 10 } }
      is_ground_truth_anomalous: false
    }
    probability_weight: 0.4
  }
  query_template_configs: {
    template: {
      normalized_query_template: "DROP TABLE table2"
      is_ground_truth_anomalous: true
    }
    probability_weight: 0.6
  }
}`, &cfgpb.QueryTemplateConfigs{})

	wantValidBucketConfigs := parseProto(t, `
bucket_sampling {
  query_template_configs: {
    template: {
      normalized_query_template: "SELECT * FROM table1 WHERE id = %v"
      placeholder_values: { int_placeholder: { min: 1 max: 10 } }
    }
    bucket_config: {
      queries_per_bucket_mean: 10.0
      queries_variation_std_dev: 2.0
      anomalous_spike_probability: 0.1
      anomalous_spike_multiplier: 1.5
    }
  }
}`, &cfgpb.QueryTemplateConfigs{})

	tests := []struct {
		name       string
		writeFile  bool
		filename   string
		content    string
		wantErr    bool
		wantConfig *cfgpb.QueryTemplateConfigs
	}{
		{
			name:      "valid_multiple_configs",
			writeFile: true,
			filename:  "valid_config.textproto",
			content: `
weighted_sampling {
  query_template_configs: {
    template: {
      normalized_query_template: "SELECT * FROM table1 WHERE id = %v"
      placeholder_values: { int_placeholder: { min: 1 max: 10 } }
      is_ground_truth_anomalous: false
    }
    probability_weight: 0.4
  }
  query_template_configs: {
    template: {
      normalized_query_template: "DROP TABLE table2"
      is_ground_truth_anomalous: true
    }
    probability_weight: 0.6
  }
}`,
			wantErr:    false,
			wantConfig: wantValidConfigs,
		},
		{
			name:      "valid_bucket_sampling_configs",
			writeFile: true,
			filename:  "valid_bucket_config.textproto",
			content: `
bucket_sampling {
  query_template_configs: {
    template: {
      normalized_query_template: "SELECT * FROM table1 WHERE id = %v"
      placeholder_values: { int_placeholder: { min: 1 max: 10 } }
    }
    bucket_config: {
      queries_per_bucket_mean: 10.0
      queries_variation_std_dev: 2.0
      anomalous_spike_probability: 0.1
      anomalous_spike_multiplier: 1.5
    }
  }
}`,
			wantErr:    false,
			wantConfig: wantValidBucketConfigs,
		},
		{
			name:      "missing_file",
			writeFile: false,
			filename:  "non_existent_file.textproto",
			wantErr:   true,
		},
		{
			name:      "invalid_syntax",
			writeFile: true,
			filename:  "invalid_syntax.textproto",
			content: `
weighted_sampling {
  query_template_configs: {
    template: {
      normalized_query_template: "SELECT * FROM table"
    }
  }
}`,
			wantErr: true,
		},
		{
			name:      "genuine_unmarshal_syntax_error",
			writeFile: true,
			filename:  "genuine_unmarshal_syntax_error.textproto",
			content:   `invalid_key: {`,
			wantErr:   true,
		},
		{
			name:      "both_scenarios_selected_syntax_error",
			writeFile: true,
			filename:  "both_scenarios_selected.textproto",
			content: `
weighted_sampling {
  query_template_configs: {
    template: {
      normalized_query_template: "SELECT 1"
    }
  }
}
bucket_sampling {
  query_template_configs: {
    template: {
      normalized_query_template: "SELECT 2"
    }
    bucket_config: {
      queries_per_bucket_mean: 10.0
    }
  }
}`,
			wantErr: true,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var path string
			if test.writeFile {
				tempDir := t.TempDir()
				path = filepath.Join(tempDir, test.filename)
				if err := os.WriteFile(path, []byte(test.content), 0644); err != nil {
					t.Fatalf("Failed to write temporary test file: %v", err)
				}
			} else {
				path = test.filename
			}

			configs, err := loadConfigs(context.Background(), path)
			if (err != nil) != test.wantErr {
				t.Fatalf("loadConfigs() error = %v, wantErr %v", err, test.wantErr)
			}
			if !test.wantErr {
				if diff := cmp.Diff(test.wantConfig, configs, protocmp.Transform()); diff != "" {
					t.Errorf("loadConfigs() mismatch (-want +got):\n%s", diff)
				}
			}
		})
	}
}

func TestLoadConfigsConnectionExample(t *testing.T) {
	// Keeps the checked-in example connection config in sync with the schema and validation rules.
	examplePath := filepath.Join("..", "..", "client", "connection_generator_config.textproto")
	configs, err := loadConfigs(context.Background(), examplePath)
	if err != nil {
		t.Fatalf("loadConfigs(%q) error = %v, want nil", examplePath, err)
	}
	if got := len(configs.GetConnectionSampling().GetClientProfiles()); got == 0 {
		t.Errorf("loadConfigs(%q) returned %d client profiles, want at least 1", examplePath, got)
	}
}
