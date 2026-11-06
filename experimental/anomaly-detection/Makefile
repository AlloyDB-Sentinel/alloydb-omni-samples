.PHONY: build test test-race proto

PROTO_FILES := \
	v1/generator/proto/audit_log_entry.proto \
	v1/generator/proto/connection_log_entry.proto \
	v1/generator/proto/query_template_config.proto \
	v2/common/proto/audit_log_entry.proto \
	v2/common/proto/anomaly_finding_entry.proto \
	v2/common/proto/model.proto \
	v2/client/proto/experiment.proto \
	v2/client/proto/evaluation.proto \
	v2/training_service/proto/training_service.proto \
	v2/prediction_service/proto/prediction_service.proto

build:
	mkdir -p bin
	go build -o bin/generator ./v1/generator/src
	go build -o bin/training-service ./v2/training_service
	go build -o bin/prediction-service ./v2/prediction_service
	go build -o bin/client ./v2/client

test:
	go test ./v2/... ./v1/generator/...

test-race:
	MARKOV_E2E_RACE=1 go test -race ./v2/... ./v1/generator/...

proto:
	protoc -I . --go_out=. --go_opt=paths=source_relative \
		--go_opt=default_api_level=API_OPAQUE \
		--go-grpc_out=. --go-grpc_opt=paths=source_relative $(PROTO_FILES)

# Dedicated chi-squared runner: does not depend on the legacy n-gram client.
.PHONY: build-chi-squared-benchmark test-chi-squared-benchmark
build-chi-squared-benchmark:
	mkdir -p bin
	go build -o bin/prediction-service ./v2/prediction_service
	go build -o bin/chi-squared-benchmark ./v2/chi_squared_benchmark

test-chi-squared-benchmark:
	go test ./v2/chi_squared_benchmark ./v2/prediction_service/...
