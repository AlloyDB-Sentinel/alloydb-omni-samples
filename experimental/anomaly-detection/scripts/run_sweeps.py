#!/usr/bin/env python3
"""Start isolated local services and run reproducible parameter sweeps (stdlib only)."""
import argparse
import contextlib
from pathlib import Path
import socket
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_ready(process, port):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Service exited; inspect service logs")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError("Service failed to start")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output/parameter-sweeps")
    parser.add_argument("--profiles", nargs="+", choices=["standard_user", "maintenance_user", "service_usage"], default=["standard_user"])
    parser.add_argument("--parameters", default="all")
    parser.add_argument("--points", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--fpr-limit", type=float, default=0.05)
    parser.add_argument("--window-min", type=int, default=0)
    parser.add_argument("--window-max", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    models = out / "models"
    models.mkdir(parents=True, exist_ok=True)
    processes = []
    try:
        with contextlib.ExitStack() as stack:
            ports = []
            for binary, directory_flag in [("training-service", "model_dir"), ("prediction-service", "models_dir")]:
                port = free_port()
                log = stack.enter_context((out / (binary + ".log")).open("a"))
                proc = subprocess.Popen([str(ROOT / "bin" / binary), f"--port={port}", f"--{directory_flag}={models}"], cwd=ROOT, stdout=log, stderr=log)
                processes.append(proc)
                wait_ready(proc, port)
                ports.append(port)
            for profile in args.profiles:
                with (out / (profile + ".log")).open("a") as log:
                    subprocess.run([
                        str(ROOT / "bin/client"), f"--generator_bin={ROOT / 'bin/generator'}",
                        f"--experiment_config=v2/client/configs/{profile}.textproto",
                        f"--output_dir={out / profile}", f"--server_addr=localhost:{ports[0]}",
                        f"--prediction_server_addr=localhost:{ports[1]}", f"--sweep={args.parameters}",
                        f"--sweep_points={args.points}", f"--sweep_repeats={args.repeats}", f"--sweep_workers={args.workers}",
                        f"--sweep_fpr_limit={args.fpr_limit}",
                        f"--sweep_window_min={args.window_min}", f"--sweep_window_max={args.window_max}",
                    ], cwd=ROOT, stdout=log, stderr=log, check=True)
                print(f"Completed {profile}: {out / profile}", flush=True)
    finally:
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
        for proc in processes:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


if __name__ == "__main__":
    main()
