#!/usr/bin/env python3
"""Run ChiSquaredModel sweeps against one saved labeled stream; no training."""
import argparse
from pathlib import Path
import socket
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Existing logs.binproto with ground truth")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/chi-squared-benchmark")
    parser.add_argument("--sweep", choices=["both", "window", "threshold"], default="both")
    parser.add_argument("--historical-m", type=int, default=1000)
    parser.add_argument("--recent-n", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=10)
    parser.add_argument("--points", type=int, default=20)
    parser.add_argument("--window-min", type=int, default=20)
    parser.add_argument("--window-max", type=int, default=500)
    parser.add_argument("--threshold-min", type=float, default=1)
    parser.add_argument("--threshold-max", type=float, default=50)
    parser.add_argument("--warmup-actions", type=int, default=-1)
    args = parser.parse_args()
    source = args.input.resolve()
    if not source.is_file():
        parser.error(f"input does not exist: {source}")
    out = args.output_dir.resolve()
    if (out / "benchmark.json").exists():
        parser.error("output already contains a benchmark; choose a fresh output directory")
    models = out / "models"
    models.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with (out / "prediction-service.log").open("w") as service_log:
        process = subprocess.Popen([str(ROOT / "bin/prediction-service"), f"--port={port}", f"--models_dir={models}"], cwd=ROOT, stdout=service_log, stderr=service_log)
        try:
            ready = False
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f"prediction service exited; inspect {out / 'prediction-service.log'}")
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                        ready = True
                        break
                except OSError:
                    time.sleep(0.1)
            if not ready:
                raise RuntimeError("prediction service did not become ready")
            command = [str(ROOT / "bin/chi-squared-benchmark"), f"--validation_log_path={source}", f"--output_dir={out}", f"--prediction_server_addr=localhost:{port}"]
            for key in ["sweep", "historical_m", "recent_n", "threshold", "points", "window_min", "window_max", "threshold_min", "threshold_max", "warmup_actions"]:
                command.append(f"--{key}={getattr(args, key)}")
            with (out / "runner.log").open("w") as runner_log:
                completed = subprocess.run(command, cwd=ROOT, stdout=runner_log, stderr=runner_log)
            if completed.returncode:
                raise RuntimeError(f"benchmark failed; inspect {out / 'runner.log'}")
            print(f"Completed chi-squared benchmark: {out}")
        finally:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    main()
