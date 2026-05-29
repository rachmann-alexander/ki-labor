#!/usr/bin/env python3
"""
Host-side unattended grid runner for Jetson benchmarks where:
- Power/tegrastats are controlled on the HOST (needs sudo).
- The actual PyTorch benchmark runs INSIDE a Docker container via `docker exec`.

This version does NOT require a shared results mount between host and container.
It writes benchmark outputs inside the container (default: /tmp/ee_bench_out/<run_id>/)
and copies them back to the host via `docker cp` after each run.

Designed for: ResNet50 vs MobileNetV2, FP16/FP32/FP64, BS 32/64/128, power profiles slow/medium/fast.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PROFILES = ["slow", "medium", "fast"]
#PROFILES = ["fast"]
MODELS = ["resnet50", "mobilenet_v2"]
#MODELS = ["mobilenet_v2"]
PRECISIONS = ["fp16", "fp32", "fp64"]
#PRECISIONS = ["fp32"]
BATCH_SIZES = [32, 64, 128]
#BATCH_SIZES = [32]


def run(cmd: List[str], *, check: bool = True, capture: bool = False, text: bool = True, timeout: Optional[int] = None) -> subprocess.CompletedProcess:
    kwargs = {}
    if capture:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    return subprocess.run(cmd, check=check, text=text, timeout=timeout, **kwargs)


def is_sudo_noprompt_ok() -> bool:
    try:
        subprocess.run(["sudo", "-n", "true"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


class SudoKeepAlive:
    """Keep sudo timestamp alive so unattended runs don't get stuck on password prompts."""
    def __init__(self, interval_s: int = 60):
        self.interval_s = interval_s
        self._stop = False
        self._pid: Optional[int] = None

    def start(self) -> None:
        pid = os.fork()
        if pid == 0:
            # child
            while True:
                try:
                    subprocess.run(["sudo", "-n", "true"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception:
                    # cannot prompt; just exit to avoid hanging the parent
                    os._exit(2)
                time.sleep(self.interval_s)
        else:
            self._pid = pid

    def stop(self) -> None:
        if self._pid is None:
            return
        try:
            os.kill(self._pid, signal.SIGTERM)
        except Exception:
            pass
        self._pid = None


@dataclass(frozen=True)
class RunSpec:
    model: str
    precision: str
    profile: str
    batch: int

    @property
    def run_id(self) -> str:
        return f"{self.model}_{self.precision}_{self.profile}_BS{self.batch}"


def load_done_set(state_path: Path) -> set:
    done = set()
    if not state_path.exists():
        return done
    for line in state_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if rec.get("status") == "done":
            done.add(rec.get("run_id"))
    return done


def append_state(state_path: Path, rec: Dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with state_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def ensure_scripts_exist(project_root: Path) -> None:
    for rel in ["set_power.sh", "host_tegrastats.sh"]:
        p = project_root / rel
        if not p.exists():
            raise FileNotFoundError(f"Fehlt: {p}")
        if not os.access(p, os.X_OK):
            raise PermissionError(f"Nicht ausführbar: {p} (chmod +x {p})")


def docker_exec(container: str, cmd: List[str], *, capture: bool = False, timeout: Optional[int] = None, check: bool = True) -> subprocess.CompletedProcess:
    base = ["docker", "exec", container] + cmd
    return run(base, check=check, capture=capture, timeout=timeout)


def docker_cp_from(container: str, src: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    run(["docker", "cp", f"{container}:{src}", str(dst)], check=True, capture=False)


def read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_power_stats(power_csv: Path) -> Dict[str, float]:
    if not power_csv.exists():
        return {}
    samples = []
    with power_csv.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                samples.append(float(row["power_W"]))
            except (KeyError, ValueError):
                continue
    if not samples:
        return {}
    avg = sum(samples) / len(samples)
    return {"power_avg_W": avg, "power_samples": len(samples)}


def write_manifest_header(manifest_path: Path) -> None:
    if manifest_path.exists():
        return
    with manifest_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "prefix",
                "profile",
                "model",
                "precision",
                "batch",
                "power_csv",
                "perf_csv",
                "bench_meta",
                "avg_power_W",
                "power_samples",
                "throughput_img_s",
                "latency_avg_ms",
                "latency_p50_ms",
                "top1",
                "duration_s",
                "timestamp",
            ]
        )


def append_manifest(manifest_path: Path, row: List) -> None:
    with manifest_path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(row)



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=str, default=".", help="Host project root (contains set_power.sh, host_tegrastats.sh)")
    ap.add_argument("--results-dir", type=str, default="results", help="Host results dir")
    ap.add_argument("--docker-container", type=str, required=True, help="Docker container name (e.g. unruffled_kepler)")
    ap.add_argument("--container-project-dir", type=str, default="/workspace", help="Project root inside container")
    ap.add_argument("--container-bench", type=str, default="repro_automation/bench_classifier_repro.py", help="Path to bench script relative to container project dir (or absolute)")
    ap.add_argument("--container-out-base", type=str, default="/tmp/ee_bench_out", help="Base dir inside container for per-run outputs")
    ap.add_argument("--data-dir", type=str, required=True, help="ImageNet/val path AS SEEN INSIDE THE CONTAINER")
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=0, help="0 = full dataset")
    ap.add_argument("--measure-accuracy", action="store_true")
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--timeout-s", type=int, default=0, help="0 = no timeout per run")
    ap.add_argument("--unattended", action="store_true", help="Fail if sudo would prompt for password")

    args = ap.parse_args()

    project_root = Path(args.project_root).resolve()
    results_dir = (project_root / args.results_dir).resolve()
    state_path = results_dir / "grid_state.jsonl"
    manifest_path = results_dir / "grid_manifest.csv"

    ensure_scripts_exist(project_root)

    if args.unattended and not is_sudo_noprompt_ok():
        print("ERROR: sudo würde nach Passwort fragen. Für unattended: zuerst `sudo -v` oder NOPASSWD konfigurieren.", file=sys.stderr)
        sys.exit(2)

    keepalive = SudoKeepAlive(interval_s=60)
    keepalive.start()

    done_set = load_done_set(state_path) if args.skip_existing else set()

    runs: List[RunSpec] = [RunSpec(m, p, prof, b) for prof in PROFILES for m in MODELS for p in PRECISIONS for b in BATCH_SIZES]
    if args.shuffle:
        random.Random(args.seed).shuffle(runs)

    results_dir.mkdir(parents=True, exist_ok=True)

    # Manifest header
    write_manifest_header(manifest_path)


    for idx, spec in enumerate(runs, start=1):
        if args.skip_existing and spec.run_id in done_set:
            print(f"[{idx}/{len(runs)}] SKIP {spec.run_id} (done)")
            continue

        attempt = 0
        status = "failed"
        t0 = time.time()

        # Host result paths
        perf_csv_host = results_dir / f"{spec.run_id}_perf.csv"
        meta_json_host = results_dir / f"{spec.run_id}_bench_meta.json"
        tegra_log_host = results_dir / f"{spec.run_id}_tegrastats.log"
        tegra_power_host = results_dir / f"{spec.run_id}_tegrastats_power.csv"

        # Container paths
        out_dir_c = f"{args.container_out_base}/{spec.run_id}"
        perf_csv_c = f"{out_dir_c}/perf.csv"
        meta_json_c = f"{out_dir_c}/meta.json"

        bench_path = args.container_bench
        if not bench_path.startswith("/"):
            bench_path = f"{args.container_project_dir.rstrip('/')}/{bench_path}"

        while attempt <= max(0, int(args.retries)):
            attempt += 1
            append_state(state_path, {"run_id": spec.run_id, "status": "running", "attempt": attempt, "ts": time.time()})

            print(f"[{idx}/{len(runs)}] RUN {spec.run_id} (attempt {attempt})")

            try:
                # 1) set power on host
                run(["sudo", str(project_root / "set_power.sh"), spec.profile], check=True, capture=False)

                # 2) start tegrastats on host (it writes to results/...)
                run(["sudo", str(project_root / "host_tegrastats.sh"), "start", spec.run_id], check=True, capture=False)

                # 3) run bench in container, writing outputs to container tmp
                docker_exec(args.docker_container, ["bash", "-lc", f"mkdir -p {shlex.quote(out_dir_c)}"], check=True)

                bench_cmd = [
                    "python3",
                    bench_path,
                    "--model", spec.model,
                    "--precision", spec.precision,
                    "--batch", str(spec.batch),
                    "--data_dir", args.data_dir,
                    "--imgsz", str(args.imgsz),
                    "--warmup", str(args.warmup),
                    "--iters", str(args.iters),
                    "--out_csv", perf_csv_c,
                    "--meta_out", meta_json_c,
                    "--seed", str(args.seed),
                    "--pretrained",
                ]
                if args.measure_accuracy:
                    bench_cmd.append("--measure_accuracy")

                timeout = None if args.timeout_s <= 0 else int(args.timeout_s)
                docker_exec(args.docker_container, bench_cmd, check=True, timeout=timeout)

                # 4) stop tegrastats on host
                run(["sudo", str(project_root / "host_tegrastats.sh"), "stop"], check=True, capture=False)

                # 5) copy outputs from container to host
                # docker cp copies file or dir. We'll copy both files explicitly.
                docker_cp_from(args.docker_container, perf_csv_c, perf_csv_host)
                docker_cp_from(args.docker_container, meta_json_c, meta_json_host)

                status = "done"
                break

            except subprocess.TimeoutExpired:
                status = "timeout"
                try:
                    run(["sudo", str(project_root / "host_tegrastats.sh"), "stop"], check=False, capture=False)
                except Exception:
                    pass
                append_state(state_path, {"run_id": spec.run_id, "status": "timeout", "attempt": attempt, "ts": time.time()})
                print(f"TIMEOUT: {spec.run_id}", file=sys.stderr)

            except subprocess.CalledProcessError as e:
                status = "failed"
                try:
                    run(["sudo", str(project_root / "host_tegrastats.sh"), "stop"], check=False, capture=False)
                except Exception:
                    pass
                append_state(state_path, {"run_id": spec.run_id, "status": "failed", "attempt": attempt, "returncode": e.returncode, "ts": time.time()})
                print(f"FAILED: {spec.run_id} (rc={e.returncode})", file=sys.stderr)

        duration = round(time.time() - t0, 3)
        # host_tegrastats.sh writes into results/<prefix>_tegrastats.log and <prefix>_tegrastats_power.csv
        # We record expected paths (whether they exist depends on script success)
        meta = read_json(meta_json_host)

        power_csv_disk = tegra_power_host
        legacy_power_csv = tegra_power_host.parent / f"{spec.run_id}_power.csv"
        if not power_csv_disk.exists() and legacy_power_csv.exists():
            power_csv_disk = legacy_power_csv

        power_stats = read_power_stats(power_csv_disk)

        append_manifest(
            manifest_path,
            [
                spec.run_id,                         # prefix
                spec.profile,                        # profile
                spec.model,                          # model
                spec.precision,                      # precision
                spec.batch,                          # batch
                str(power_csv_disk),                 # power_csv
                str(perf_csv_host),                  # perf_csv
                str(meta_json_host),                 # bench_meta
                power_stats.get("power_avg_W"),      # avg_power_W
                power_stats.get("power_samples"),    # power_samples
                meta.get("metrics", {}).get("throughput_img_s"),
                meta.get("metrics", {}).get("latency_avg_ms"),
                meta.get("metrics", {}).get("latency_p50_ms"),
                meta.get("metrics", {}).get("top1"),
                meta.get("metrics", {}).get("duration_s") if meta.get("metrics") else None,
                meta.get("timestamp"),
            ],
        )
        append_state(state_path, {"run_id": spec.run_id, "status": status, "attempts": attempt, "duration_s": duration, "ts": time.time()})

    keepalive.stop()
    print(f"Fertig. Manifest: {manifest_path}")
    print(f"State: {state_path}")


if __name__ == "__main__":
    main()
