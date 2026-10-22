package generatorpb

import (
	"encoding/hex"
	"testing"

	"google.golang.org/protobuf/proto"
)

func TestBucketRowSettingsWireFormat(t *testing.T) {
	// Encoded with protoc --encode from query_template_config.proto, independently
	// of the Go bindings. This catches stale field numbers in generated code.
	wire, err := hex.DecodeString("490000000000006940510000000000004440597b14ae47e17a943f610000000000002e4068a08d067001")
	if err != nil {
		t.Fatal(err)
	}
	got := &BucketSimulationConfig{}
	if err := proto.Unmarshal(wire, got); err != nil {
		t.Fatal(err)
	}
	want := BucketSimulationConfig_builder{
		RowsReturnedMean:         200,
		RowsReturnedStdDev:       40,
		AnomalousRowsProbability: 0.02,
		AnomalousRowsMultiplier:  15,
		RowsReturnedMax:          100000,
		PgauditScenario:          true,
	}.Build()
	if !proto.Equal(got, want) {
		t.Errorf("decoded row settings = %v, want %v", got, want)
	}
}
