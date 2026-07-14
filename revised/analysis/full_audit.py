#!/usr/bin/env python3
"""Reproducible audit of every Ergebnisse_* file and the Plotcode directory.

All generated artefacts are written below ``revised/supplementary/audit``.  The
script is deliberately read-only with respect to the source experiment folders.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / "supplementary" / "audit"
OUT.mkdir(parents=True, exist_ok=True)

RUN_RE = re.compile(
    r"^(?P<model>mobilenet_v2|resnet50)_(?P<precision>fp16|fp32|fp64)_"
    r"(?P<profile>slow|medium|fast)_BS(?P<batch>\d+)_"
    r"(?P<kind>bench_meta\.json|perf\.csv|tegrastats\.log|tegrastats_power\.csv)$"
)
PREFIX_RE = re.compile(
    r"^(?P<model>mobilenet_v2|resnet50)_(?P<precision>fp16|fp32|fp64)_"
    r"(?P<profile>slow|medium|fast)_BS(?P<batch>\d+)$"
)

PERF_HEADER = ["iter", "batch", "latency_ms", "img_per_s", "acc_top1"]
POWER_HEADER = ["sample_index", "elapsed_s", "power_W"]
MANIFEST_HEADER = [
    "prefix", "profile", "model", "precision", "batch", "power_csv",
    "perf_csv", "bench_meta", "avg_power_W", "power_samples",
    "throughput_img_s", "latency_avg_ms", "latency_p50_ms", "top1",
    "duration_s", "timestamp",
]
ENRICHED_EXTRA = [
    "energy_J", "energy_per_img_J", "energy_per_inference_J",
    "energy_J_integrated", "energy_per_img_J_integrated", "edp_Js",
    "edp_per_img_Js",
]

LOG_TS_RE = re.compile(r"^(\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2})")
LOG_POWER_RE = re.compile(r"\bVDD_IN\s+(\d+(?:\.\d+)?)mW/(\d+(?:\.\d+)?)mW")
LOG_CPU_BLOCK_RE = re.compile(r"\bCPU \[(.*?)\]\s+EMC_FREQ")
LOG_CPU_FREQ_RE = re.compile(r"@(\d+)")
LOG_EMC_RE = re.compile(r"\bEMC_FREQ\s+\d+%@(\d+)")
LOG_GPU_RE = re.compile(r"\bGR3D_FREQ\s+\d+%@\[([^\]]+)\]")
LOG_TEMP_RE = re.compile(r"\b(?:gpu|tj)@(-?\d+(?:\.\d+)?)C")


DATASET_LABELS = {
    "Ergebnisse_13.03.2026 (feste Reihenfolge)": "fixed_2026-03",
    "Ergebnisse_23.03.2026 (zufällige Reihenfolge)": "random_2026-03",
    "Ergebnisse_Clone1_13.07.2026": "Clone1",
    "Ergebnisse_Clone2_13.07.2026": "Clone2",
    "Ergebnisse_Clone3_13.07.2026": "Clone3",
    "Ergebnisse_MasterJetson_07.07.2026": "MasterJetson",
}


issues: list[dict[str, Any]] = []


def add_issue(dataset: str, severity: str, category: str, detail: str, path: str = "") -> None:
    issues.append({
        "dataset": dataset,
        "severity": severity,
        "category": category,
        "detail": detail,
        "path": path,
    })


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def close(a: Any, b: Any, rtol: float = 1e-10, atol: float = 1e-10) -> bool:
    try:
        return bool(np.isclose(float(a), float(b), rtol=rtol, atol=atol, equal_nan=False))
    except Exception:
        return False


def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def finite_array(x: Iterable[Any]) -> np.ndarray:
    a = np.asarray(list(x), dtype=float)
    return a[np.isfinite(a)]


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    p = np.asarray(list(p_values), dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    vals = p[ok]
    if vals.size == 0:
        return out.tolist()
    order = np.argsort(vals)
    ranked = vals[order]
    adjusted = np.maximum.accumulate((len(vals) - np.arange(len(vals))) * ranked)
    adjusted = np.minimum(adjusted, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    out[np.where(ok)[0]] = restored
    return out.tolist()


def paired_summary(x_num: Iterable[float], x_den: Iterable[float]) -> dict[str, Any]:
    num = np.asarray(list(x_num), dtype=float)
    den = np.asarray(list(x_den), dtype=float)
    ok = np.isfinite(num) & np.isfinite(den) & (num > 0) & (den > 0)
    num, den = num[ok], den[ok]
    n = len(num)
    if n == 0:
        return {"n": 0}
    lr = np.log(num / den)
    mean_lr = float(np.mean(lr))
    if n > 1:
        se = float(stats.sem(lr))
        q = float(stats.t.ppf(0.975, n - 1))
        ci_lo, ci_hi = mean_lr - q * se, mean_lr + q * se
    else:
        ci_lo = ci_hi = float("nan")
    diff = num - den
    nz = diff[diff != 0]
    if len(nz):
        try:
            w = stats.wilcoxon(num, den, zero_method="wilcox", alternative="two-sided", method="auto")
            p_w = float(w.pvalue)
            ranks = stats.rankdata(np.abs(nz))
            wplus = float(ranks[nz > 0].sum())
            wminus = float(ranks[nz < 0].sum())
            rank_biserial = (wplus - wminus) / (wplus + wminus)
        except Exception:
            p_w = rank_biserial = float("nan")
    else:
        p_w, rank_biserial = 1.0, 0.0
    return {
        "n": n,
        "geomean_ratio": float(math.exp(mean_lr)),
        "geomean_ratio_ci95_low": float(math.exp(ci_lo)) if np.isfinite(ci_lo) else np.nan,
        "geomean_ratio_ci95_high": float(math.exp(ci_hi)) if np.isfinite(ci_hi) else np.nan,
        "median_ratio": float(np.median(num / den)),
        "median_percent_change": float(100 * np.median(num / den - 1)),
        "mean_num": float(np.mean(num)),
        "mean_den": float(np.mean(den)),
        "wilcoxon_p": p_w,
        "rank_biserial_num_gt_den": rank_biserial,
    }


def inventory_files(dataset_dirs: list[Path], plot_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for directory in dataset_dirs + [plot_dir]:
        label = DATASET_LABELS.get(directory.name, directory.name)
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            stat = path.stat()
            digest = sha256_file(path)
            rows.append({
                "dataset": label,
                "path": rel(path),
                "extension": path.suffix.lower(),
                "bytes": stat.st_size,
                "sha256": digest,
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "file_inventory_sha256.csv", index=False)
    dup = df[df.duplicated("sha256", keep=False)].sort_values(["sha256", "path"])
    dup.to_csv(OUT / "duplicate_file_content.csv", index=False)
    return df


def read_manifests(dataset_dirs: list[Path]) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    manifests: dict[str, pd.DataFrame] = {}
    enriched: dict[str, pd.DataFrame] = {}
    schema_rows: list[dict[str, Any]] = []
    for directory in dataset_dirs:
        label = DATASET_LABELS[directory.name]
        for filename, target, expected in [
            ("grid_manifest.csv", manifests, MANIFEST_HEADER),
            ("grid_manifest_enriched.csv", enriched, MANIFEST_HEADER + ENRICHED_EXTRA),
        ]:
            path = directory / filename
            if not path.exists():
                add_issue(label, "error", "missing_manifest", filename, rel(directory))
                continue
            with path.open("r", newline="", encoding="utf-8-sig") as fh:
                reader = csv.reader(fh)
                header = next(reader, [])
            df = pd.read_csv(path)
            target[label] = df
            schema_rows.append({
                "dataset": label,
                "file_type": filename,
                "header": "|".join(header),
                "rows": len(df),
                "header_exact": header == expected,
                "duplicate_rows": int(df.duplicated().sum()),
                "duplicate_prefixes": int(df.duplicated("prefix").sum()) if "prefix" in df else np.nan,
            })
            if header != expected:
                add_issue(label, "error", "manifest_schema", f"Unexpected header in {filename}", rel(path))
            if "prefix" in df and df["prefix"].duplicated().any():
                add_issue(label, "error", "manifest_duplicate", f"Duplicate prefix in {filename}", rel(path))
            # Absolute paths are historical Linux paths; record exact validity and basename recoverability.
            for col in ["power_csv", "perf_csv", "bench_meta"]:
                if col in df:
                    exact_exists = int(sum(Path(str(v)).exists() for v in df[col]))
                    base_exists = int(sum((directory / Path(str(v)).name).exists() for v in df[col]))
                    schema_rows.append({
                        "dataset": label,
                        "file_type": f"{filename}:{col}_path_check",
                        "header": "",
                        "rows": len(df),
                        "header_exact": np.nan,
                        "duplicate_rows": np.nan,
                        "duplicate_prefixes": np.nan,
                        "exact_paths_existing": exact_exists,
                        "basenames_existing_locally": base_exists,
                    })
    pd.DataFrame(schema_rows).to_csv(OUT / "manifest_schema_and_path_audit.csv", index=False)
    return manifests, enriched


def parse_state_files(dataset_dirs: list[Path], manifests: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    summary: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    running_orders: dict[str, list[str]] = {}
    for directory in dataset_dirs:
        label = DATASET_LABELS[directory.name]
        path = directory / "grid_state.jsonl"
        events: list[dict[str, Any]] = []
        malformed = 0
        with path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    events.append(rec)
                    event_rows.append({"dataset": label, "line": line_no, **rec})
                except Exception as exc:
                    malformed += 1
                    add_issue(label, "error", "state_json", str(exc), f"{rel(path)}:{line_no}")
        running_order = [str(e.get("run_id")) for e in events if e.get("status") == "running"]
        running_orders[label] = running_order
        final_by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for e in events:
            if e.get("status") in {"done", "failed", "timeout"} and "attempts" in e:
                final_by_run[str(e.get("run_id"))].append(e)
        manifest_order = manifests.get(label, pd.DataFrame()).get("prefix", pd.Series(dtype=str)).astype(str).tolist()
        summary.append({
            "dataset": label,
            "events": len(events),
            "malformed_lines": malformed,
            "running_events": sum(e.get("status") == "running" for e in events),
            "done_events_total": sum(e.get("status") == "done" for e in events),
            "failed_events_total": sum(e.get("status") == "failed" for e in events),
            "timeout_events_total": sum(e.get("status") == "timeout" for e in events),
            "unique_run_ids": len({str(e.get("run_id")) for e in events}),
            "unique_final_records": len(final_by_run),
            "multiple_final_records": sum(len(v) > 1 for v in final_by_run.values()),
            "running_order_equals_manifest": running_order == manifest_order,
        })
        if running_order != manifest_order:
            add_issue(label, "warning", "state_manifest_order", "running-event order differs from manifest order", rel(path))
    pd.DataFrame(event_rows).to_csv(OUT / "grid_state_events.csv", index=False)
    state_summary = pd.DataFrame(summary)
    state_summary.to_csv(OUT / "grid_state_summary.csv", index=False)
    return state_summary, running_orders


def parse_log(
    path: Path, power_values: np.ndarray, power_elapsed_values: np.ndarray
) -> dict[str, Any]:
    timestamps: list[datetime] = []
    log_power: list[float] = []
    temperature_timestamps: list[float] = []
    temperature_values: list[float] = []
    cpu_freqs = Counter()
    emc_freqs = Counter()
    gpu_freqs = Counter()
    max_temp = -np.inf
    lines = malformed_ts = malformed_power = replacement_lines = 0
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            lines += 1
            parsed_timestamp: datetime | None = None
            if "\ufffd" in line:
                replacement_lines += 1
            m = LOG_TS_RE.search(line)
            if m:
                try:
                    parsed_timestamp = datetime.strptime(m.group(1), "%m-%d-%Y %H:%M:%S")
                    timestamps.append(parsed_timestamp)
                except Exception:
                    malformed_ts += 1
            else:
                malformed_ts += 1
            m = LOG_POWER_RE.search(line)
            if m:
                log_power.append(float(m.group(1)) / 1000.0)
            else:
                malformed_power += 1
            m = LOG_CPU_BLOCK_RE.search(line)
            if m:
                for f in LOG_CPU_FREQ_RE.findall(m.group(1)):
                    cpu_freqs[int(f)] += 1
            m = LOG_EMC_RE.search(line)
            if m:
                emc_freqs[int(m.group(1))] += 1
            m = LOG_GPU_RE.search(line)
            if m:
                vals = [int(v) for v in re.findall(r"\d+", m.group(1))]
                for v in vals:
                    gpu_freqs[v] += 1
            temps = [float(v) for v in LOG_TEMP_RE.findall(line)]
            if temps:
                line_temp = max(temps)
                max_temp = max(max_temp, line_temp)
                if parsed_timestamp is not None:
                    temperature_timestamps.append(parsed_timestamp.timestamp())
                    temperature_values.append(line_temp)
    ts_seconds = np.array([t.timestamp() for t in timestamps], dtype=float)
    gaps = np.diff(ts_seconds) if len(ts_seconds) > 1 else np.array([], dtype=float)
    log_p = np.asarray(log_power, dtype=float)
    if len(ts_seconds) == len(log_p) and len(ts_seconds) >= 2:
        timestamp_energy = float(np.trapezoid(log_p, ts_seconds))
    else:
        timestamp_energy = np.nan
    if len(log_p) == len(power_values) and len(log_p):
        p_max_abs_diff = float(np.max(np.abs(log_p - power_values)))
    else:
        p_max_abs_diff = np.nan
    if len(ts_seconds) == len(power_elapsed_values) and len(ts_seconds):
        log_elapsed = ts_seconds - ts_seconds[0]
        elapsed_max_abs_diff = float(np.max(np.abs(log_elapsed - power_elapsed_values)))
    else:
        elapsed_max_abs_diff = np.nan

    temp_t = np.asarray(temperature_timestamps, dtype=float)
    temp_v = np.asarray(temperature_values, dtype=float)
    if len(temp_t) >= 2 and np.all(np.diff(temp_t) > 0):
        temp_span = float(temp_t[-1] - temp_t[0])
        temp_time_weighted_mean = (
            float(np.trapezoid(temp_v, temp_t) / temp_span) if temp_span > 0 else np.nan
        )
    else:
        temp_time_weighted_mean = np.nan

    def counter_mode(c: Counter) -> float:
        return float(c.most_common(1)[0][0]) if c else np.nan

    return {
        "log_lines": lines,
        "log_timestamp_count": len(timestamps),
        "log_power_count": len(log_power),
        "log_malformed_timestamp_lines": malformed_ts,
        "log_malformed_power_lines": malformed_power,
        "log_replacement_character_lines": replacement_lines,
        "log_wall_span_s": float(ts_seconds[-1] - ts_seconds[0]) if len(ts_seconds) > 1 else np.nan,
        "log_gap_count_gt1s": int(np.sum(gaps > 1.0)),
        "log_missing_seconds_from_gaps": float(np.sum(np.maximum(gaps - 1.0, 0))) if len(gaps) else 0.0,
        "log_max_gap_s": float(np.max(gaps)) if len(gaps) else np.nan,
        "log_nonpositive_time_steps": int(np.sum(gaps <= 0)) if len(gaps) else 0,
        "energy_J_integrated_log_timestamps": timestamp_energy,
        "log_vs_power_csv_max_abs_diff_W": p_max_abs_diff,
        "log_vs_power_csv_elapsed_max_abs_diff_s": elapsed_max_abs_diff,
        "log_temperature_count": len(temp_v),
        "log_start_gpu_or_tj_temp_C": float(temp_v[0]) if len(temp_v) else np.nan,
        "log_end_gpu_or_tj_temp_C": float(temp_v[-1]) if len(temp_v) else np.nan,
        "log_time_weighted_mean_gpu_or_tj_temp_C": temp_time_weighted_mean,
        "log_end_minus_start_gpu_or_tj_temp_C": (
            float(temp_v[-1] - temp_v[0]) if len(temp_v) else np.nan
        ),
        "cpu_freq_mode_MHz": counter_mode(cpu_freqs),
        "cpu_freq_max_MHz": float(max(cpu_freqs)) if cpu_freqs else np.nan,
        "emc_freq_mode_MHz": counter_mode(emc_freqs),
        "emc_freq_max_MHz": float(max(emc_freqs)) if emc_freqs else np.nan,
        "gpu_freq_mode_MHz": counter_mode(gpu_freqs),
        "gpu_freq_max_MHz": float(max(gpu_freqs)) if gpu_freqs else np.nan,
        "max_gpu_or_tj_temp_C": float(max_temp) if np.isfinite(max_temp) else np.nan,
    }


def audit_runs(
    dataset_dirs: list[Path],
    manifests: dict[str, pd.DataFrame],
    enriched: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_rows: list[dict[str, Any]] = []
    grid_rows: list[dict[str, Any]] = []
    schema_rows: list[dict[str, Any]] = []
    expected_kinds = {"bench_meta.json", "perf.csv", "tegrastats.log", "tegrastats_power.csv"}

    for directory in dataset_dirs:
        label = DATASET_LABELS[directory.name]
        grouped: dict[str, dict[str, Path]] = defaultdict(dict)
        for path in directory.iterdir():
            if not path.is_file():
                continue
            m = RUN_RE.match(path.name)
            if not m:
                continue
            prefix = path.name[: -(len(m.group("kind")) + 1)]
            grouped[prefix][m.group("kind")] = path

        levels = {
            "model": sorted({PREFIX_RE.match(p).group("model") for p in grouped}),
            "precision": sorted({PREFIX_RE.match(p).group("precision") for p in grouped}),
            "profile": sorted({PREFIX_RE.match(p).group("profile") for p in grouped}),
            "batch": sorted({int(PREFIX_RE.match(p).group("batch")) for p in grouped}),
        }
        expected_prefixes = {
            f"{m}_{prec}_{prof}_BS{b}"
            for m in levels["model"]
            for prec in levels["precision"]
            for prof in levels["profile"]
            for b in levels["batch"]
        }
        manifest_prefixes = set(manifests[label]["prefix"].astype(str))
        for prefix in sorted(expected_prefixes | set(grouped) | manifest_prefixes):
            kinds = set(grouped.get(prefix, {}))
            grid_rows.append({
                "dataset": label,
                "prefix": prefix,
                "in_cartesian_grid": prefix in expected_prefixes,
                "in_manifest": prefix in manifest_prefixes,
                "available_kinds": "|".join(sorted(kinds)),
                "missing_kinds": "|".join(sorted(expected_kinds - kinds)),
            })
            if prefix in expected_prefixes and kinds != expected_kinds:
                add_issue(label, "error", "missing_run_file", f"{prefix}: {sorted(expected_kinds-kinds)}", rel(directory))
        if expected_prefixes != set(grouped):
            add_issue(label, "error", "incomplete_grid", f"Expected {len(expected_prefixes)} prefixes; found {len(grouped)}", rel(directory))
        if manifest_prefixes != set(grouped):
            add_issue(label, "error", "manifest_file_set_mismatch", "Manifest prefix set differs from run files", rel(directory / "grid_manifest.csv"))

        manifest_lookup = manifests[label].set_index("prefix", drop=False).to_dict("index")
        enriched_lookup = enriched[label].set_index("prefix", drop=False).to_dict("index")

        for prefix in sorted(grouped):
            files = grouped[prefix]
            if set(files) != expected_kinds:
                continue
            spec_m = PREFIX_RE.match(prefix)
            assert spec_m
            spec = spec_m.groupdict()
            batch_cfg = int(spec["batch"])
            meta_path = files["bench_meta.json"]
            perf_path = files["perf.csv"]
            power_path = files["tegrastats_power.csv"]
            log_path = files["tegrastats.log"]
            mr = manifest_lookup[prefix]
            er = enriched_lookup[prefix]

            with meta_path.open("r", encoding="utf-8") as fh:
                meta = json.load(fh)
            meta_keys = sorted(meta.keys())
            metric_keys = sorted(meta.get("metrics", {}).keys())
            schema_rows.append({
                "dataset": label,
                "file_type": "bench_meta.json",
                "schema": "|".join(meta_keys) + " :: metrics=" + "|".join(metric_keys),
                "path": rel(meta_path),
            })

            with perf_path.open("r", newline="", encoding="utf-8-sig") as fh:
                perf_header = next(csv.reader(fh), [])
            perf = pd.read_csv(perf_path)
            schema_rows.append({
                "dataset": label,
                "file_type": "perf.csv",
                "schema": "|".join(perf_header),
                "path": rel(perf_path),
            })
            if perf_header != PERF_HEADER:
                add_issue(label, "error", "perf_schema", f"{prefix}: {perf_header}", rel(perf_path))

            with power_path.open("r", newline="", encoding="utf-8-sig") as fh:
                power_header = next(csv.reader(fh), [])
            power = pd.read_csv(power_path)
            schema_rows.append({
                "dataset": label,
                "file_type": "tegrastats_power.csv",
                "schema": "|".join(power_header),
                "path": rel(power_path),
            })
            if power_header != POWER_HEADER:
                add_issue(label, "error", "power_schema", f"{prefix}: {power_header}", rel(power_path))

            # Numeric coercion and integrity checks.
            perf_num = perf.apply(pd.to_numeric, errors="coerce")
            power_num = power.apply(pd.to_numeric, errors="coerce")
            perf_nan = int(perf_num.isna().sum().sum())
            power_nan = int(power_num.isna().sum().sum())
            perf_nonfinite = int((~np.isfinite(perf_num.to_numpy(dtype=float))).sum())
            power_nonfinite = int((~np.isfinite(power_num.to_numpy(dtype=float))).sum())
            iter_values = perf_num["iter"].to_numpy(dtype=float)
            batch_values = perf_num["batch"].to_numpy(dtype=float)
            latency = perf_num["latency_ms"].to_numpy(dtype=float)
            img_s = perf_num["img_per_s"].to_numpy(dtype=float)
            accuracy = perf_num["acc_top1"].to_numpy(dtype=float)
            sample_idx = power_num["sample_index"].to_numpy(dtype=float)
            elapsed = power_num["elapsed_s"].to_numpy(dtype=float)
            power_w = power_num["power_W"].to_numpy(dtype=float)

            total_images = float(np.sum(batch_values))
            inference_time_s = float(np.sum(latency) / 1000.0)
            calc_latency_mean = float(np.mean(latency))
            # The benchmark's stored "p50" is the upper order statistic for even n
            # (equivalent to quantile(method="higher")), not the conventional median
            # that averages the two central observations.
            calc_latency_median_conventional = float(np.median(latency))
            calc_latency_median = float(np.sort(latency)[len(latency) // 2])
            calc_throughput = total_images / inference_time_s
            calc_top1 = float(np.sum(accuracy * batch_values) / total_images)
            per_row_img_s_err = float(np.max(np.abs(img_s - batch_values * 1000.0 / latency)))
            latency_std = float(np.std(latency, ddof=1)) if len(latency) > 1 else np.nan
            latency_cv = latency_std / calc_latency_mean if calc_latency_mean else np.nan
            latency_p95 = float(np.quantile(latency, 0.95))
            latency_p99 = float(np.quantile(latency, 0.99))
            latency_max = float(np.max(latency))
            latency_gt2 = int(np.sum(latency > 2 * calc_latency_median))
            latency_gt5 = int(np.sum(latency > 5 * calc_latency_median))
            latency_gt10 = int(np.sum(latency > 10 * calc_latency_median))
            keep5 = latency <= 5 * calc_latency_median
            robust_throughput_no_gt5 = float(np.sum(batch_values[keep5]) / (np.sum(latency[keep5]) / 1000.0)) if np.any(keep5) else np.nan
            latency_lag1 = float(np.corrcoef(latency[:-1], latency[1:])[0, 1]) if len(latency) > 2 and np.std(latency[:-1]) > 0 and np.std(latency[1:]) > 0 else np.nan
            dec_n = max(1, len(latency) // 10)
            latency_last_first_decile_ratio = float(np.mean(latency[-dec_n:]) / np.mean(latency[:dec_n]))
            power_mean = float(np.mean(power_w))
            power_std = float(np.std(power_w, ddof=1)) if len(power_w) > 1 else np.nan
            power_cv = power_std / power_mean if power_mean else np.nan
            nominal_energy = float(np.trapezoid(power_w, elapsed)) if len(power_w) >= 2 else np.nan
            elapsed_span = float(elapsed[-1] - elapsed[0]) if len(elapsed) > 1 else np.nan
            power_time_weighted_mean = nominal_energy / elapsed_span if elapsed_span > 0 else np.nan
            power_lag1 = float(np.corrcoef(power_w[:-1], power_w[1:])[0, 1]) if len(power_w) > 2 and np.std(power_w[:-1]) > 0 and np.std(power_w[1:]) > 0 else np.nan
            power_dec_n = max(1, len(power_w) // 10)
            power_last_first_decile_ratio = float(np.mean(power_w[-power_dec_n:]) / np.mean(power_w[:power_dec_n]))

            log_stats = parse_log(log_path, power_w, elapsed)
            meta_metrics = meta.get("metrics", {})
            meta_duration = safe_float(meta_metrics.get("duration_s"))
            reported_energy = power_mean * meta_duration
            reported_energy_per_img = reported_energy / total_images
            integrated_per_img = nominal_energy / total_images
            timestamp_energy = log_stats["energy_J_integrated_log_timestamps"]
            timestamp_energy_per_img = timestamp_energy / total_images
            inference_proxy_epi = power_mean / calc_throughput
            response_edp_reported = integrated_per_img * calc_latency_mean / 1000.0
            response_edp_inference_proxy = inference_proxy_epi * calc_latency_mean / 1000.0
            end_to_end_time_per_image = meta_duration / total_images
            end_to_end_edp = reported_energy_per_img * end_to_end_time_per_image

            formula_checks = {
                "meta_model": meta.get("model") == spec["model"],
                "meta_precision": meta.get("precision") == spec["precision"],
                "meta_batch": safe_float(meta.get("batch")) == batch_cfg,
                "manifest_model": str(mr["model"]) == spec["model"],
                "manifest_precision": str(mr["precision"]) == spec["precision"],
                "manifest_profile": str(mr["profile"]) == spec["profile"],
                "manifest_batch": safe_float(mr["batch"]) == batch_cfg,
                "iterations": safe_float(meta_metrics.get("iterations")) == len(perf),
                "latency_avg": close(meta_metrics.get("latency_avg_ms"), calc_latency_mean),
                "latency_p50": close(meta_metrics.get("latency_p50_ms"), calc_latency_median),
                "throughput": close(meta_metrics.get("throughput_img_s"), calc_throughput),
                "top1": close(meta_metrics.get("top1"), calc_top1),
                "manifest_avg_power": close(mr["avg_power_W"], power_mean),
                "manifest_power_samples": safe_float(mr["power_samples"]) == len(power),
                "manifest_throughput": close(mr["throughput_img_s"], calc_throughput),
                "manifest_latency_avg": close(mr["latency_avg_ms"], calc_latency_mean),
                "manifest_latency_p50": close(mr["latency_p50_ms"], calc_latency_median),
                "manifest_top1": close(mr["top1"], calc_top1),
                "manifest_duration": close(mr["duration_s"], meta_duration),
                "enriched_energy": close(er["energy_J"], reported_energy),
                "enriched_epi": close(er["energy_per_img_J"], reported_energy_per_img),
                "enriched_energy_per_iteration": close(er["energy_per_inference_J"], reported_energy / len(perf)),
                "enriched_integrated_energy": close(er["energy_J_integrated"], nominal_energy),
                "enriched_integrated_epi": close(er["energy_per_img_J_integrated"], integrated_per_img),
                "enriched_edp_run": close(er["edp_Js"], nominal_energy * meta_duration),
                "enriched_edp_per_img": close(er["edp_per_img_Js"], response_edp_reported),
            }
            failed_checks = [k for k, ok in formula_checks.items() if not ok]
            if failed_checks:
                add_issue(label, "error", "formula_mismatch", f"{prefix}: {failed_checks}", rel(meta_path))

            expected_iters = math.ceil(50000 / batch_cfg)
            allowed_batches = set(batch_values[:-1]) <= {float(batch_cfg)} and 0 < batch_values[-1] <= batch_cfg
            perf_iter_sequence_ok = np.array_equal(iter_values, np.arange(len(perf), dtype=float))
            power_index_sequence_ok = np.array_equal(sample_idx, np.arange(len(power), dtype=float))
            elapsed_strict = bool(np.all(np.diff(elapsed) > 0)) if len(elapsed) > 1 else True
            duplicate_perf_rows = int(perf.duplicated().sum())
            duplicate_perf_iters = int(perf["iter"].duplicated().sum())
            duplicate_power_rows = int(power.duplicated().sum())
            duplicate_power_indices = int(power["sample_index"].duplicated().sum())

            integrity_failures = []
            if perf_nan or perf_nonfinite:
                integrity_failures.append("perf_nan_or_nonfinite")
            if power_nan or power_nonfinite:
                integrity_failures.append("power_nan_or_nonfinite")
            if not perf_iter_sequence_ok or duplicate_perf_iters:
                integrity_failures.append("perf_iter_sequence")
            if not power_index_sequence_ok or duplicate_power_indices or not elapsed_strict:
                integrity_failures.append("power_index_or_time_sequence")
            if total_images != 50000 or len(perf) != expected_iters or not allowed_batches:
                integrity_failures.append("dataset_coverage_or_batch")
            if per_row_img_s_err > 1e-8:
                integrity_failures.append("row_throughput_formula")
            if (
                log_stats["log_malformed_timestamp_lines"]
                or log_stats["log_malformed_power_lines"]
                or log_stats["log_timestamp_count"] != log_stats["log_lines"]
                or log_stats["log_power_count"] != log_stats["log_lines"]
                or log_stats["log_timestamp_count"] != log_stats["log_power_count"]
            ):
                integrity_failures.append("raw_log_timestamp_or_power_parse")
            if log_stats["log_nonpositive_time_steps"]:
                integrity_failures.append("raw_log_time_sequence")
            if not np.isfinite(log_stats["energy_J_integrated_log_timestamps"]):
                integrity_failures.append("raw_log_timestamp_energy")
            if (
                len(power) != log_stats["log_lines"]
                or not np.isfinite(log_stats["log_vs_power_csv_max_abs_diff_W"])
                or log_stats["log_vs_power_csv_max_abs_diff_W"] > 1e-9
                or not np.isfinite(log_stats["log_vs_power_csv_elapsed_max_abs_diff_s"])
                or log_stats["log_vs_power_csv_elapsed_max_abs_diff_s"] > 1e-9
            ):
                integrity_failures.append("log_power_or_elapsed_csv_mismatch")
            if integrity_failures:
                add_issue(label, "error", "run_integrity", f"{prefix}: {integrity_failures}", rel(perf_path))

            row = {
                "dataset": label,
                "source_directory": directory.name,
                "prefix": prefix,
                "model": spec["model"],
                "precision": spec["precision"],
                "profile": spec["profile"],
                "batch": batch_cfg,
                "seed": meta.get("seed"),
                "cuda_device": meta.get("cuda_device"),
                "torch_version": meta.get("torch_version"),
                "torchvision_version": meta.get("torchvision_version"),
                "timestamp_utc": meta.get("timestamp"),
                "perf_rows_iterations": len(perf),
                "total_images": total_images,
                "perf_nan_cells": perf_nan,
                "perf_nonfinite_cells": perf_nonfinite,
                "perf_duplicate_rows": duplicate_perf_rows,
                "perf_duplicate_iters": duplicate_perf_iters,
                "perf_iter_sequence_ok": perf_iter_sequence_ok,
                "batch_pattern_ok": allowed_batches,
                "row_img_per_s_max_abs_error": per_row_img_s_err,
                "latency_avg_ms": calc_latency_mean,
                "latency_p50_ms": calc_latency_median,
                "latency_p50_conventional_median_ms": calc_latency_median_conventional,
                "latency_p50_higher_minus_conventional_ms": calc_latency_median - calc_latency_median_conventional,
                "latency_p95_ms": latency_p95,
                "latency_p99_ms": latency_p99,
                "latency_max_ms": latency_max,
                "latency_max_to_p50_ratio": latency_max / calc_latency_median,
                "latency_iterations_gt2x_p50": latency_gt2,
                "latency_iterations_gt5x_p50": latency_gt5,
                "latency_iterations_gt10x_p50": latency_gt10,
                "throughput_img_s_excluding_gt5x_p50_sensitivity": robust_throughput_no_gt5,
                "throughput_sensitivity_percent_increase_excluding_gt5x_p50": 100 * (robust_throughput_no_gt5 / calc_throughput - 1),
                "latency_std_ms": latency_std,
                "latency_within_run_cv": latency_cv,
                "latency_lag1_autocorrelation": latency_lag1,
                "latency_last_vs_first_decile_mean_ratio": latency_last_first_decile_ratio,
                "throughput_img_s": calc_throughput,
                "top1_weighted": calc_top1,
                "inference_time_s": inference_time_s,
                "meta_duration_s": meta_duration,
                "inference_time_fraction_of_meta_duration": inference_time_s / meta_duration,
                "power_rows": len(power),
                "power_nan_cells": power_nan,
                "power_nonfinite_cells": power_nonfinite,
                "power_duplicate_rows": duplicate_power_rows,
                "power_duplicate_indices": duplicate_power_indices,
                "power_index_sequence_ok": power_index_sequence_ok,
                "power_elapsed_strictly_increasing": elapsed_strict,
                "power_elapsed_span_s": elapsed_span,
                "power_elapsed_dt_min_s": float(np.min(np.diff(elapsed))) if len(elapsed) > 1 else np.nan,
                "power_elapsed_dt_max_s": float(np.max(np.diff(elapsed))) if len(elapsed) > 1 else np.nan,
                "avg_power_W": power_mean,
                "time_weighted_avg_power_W": power_time_weighted_mean,
                "time_weighted_to_arithmetic_power_ratio": power_time_weighted_mean / power_mean,
                "power_std_W": power_std,
                "power_within_run_cv": power_cv,
                "power_lag1_autocorrelation": power_lag1,
                "power_last_vs_first_decile_mean_ratio": power_last_first_decile_ratio,
                "energy_J_avg_power_times_meta_duration": reported_energy,
                "energy_per_img_J_avg_power_times_meta_duration": reported_energy_per_img,
                "energy_J_integrated_nominal_elapsed": nominal_energy,
                "energy_per_img_J_integrated_nominal_elapsed": integrated_per_img,
                "energy_per_img_J_power_over_throughput_proxy": inference_proxy_epi,
                "reported_epi_to_power_over_throughput_ratio": reported_energy_per_img / inference_proxy_epi,
                "edp_per_img_Js_reported_formula": response_edp_reported,
                "edp_per_img_Js_power_throughput_proxy": response_edp_inference_proxy,
                "end_to_end_time_per_img_s": end_to_end_time_per_image,
                "end_to_end_edp_per_img_Js": end_to_end_edp,
                "formula_checks_failed": "|".join(failed_checks),
                **log_stats,
            }
            row["log_wall_span_to_meta_duration_ratio"] = row["log_wall_span_s"] / meta_duration
            row["nominal_elapsed_to_log_wall_span_ratio"] = row["power_elapsed_span_s"] / row["log_wall_span_s"] if row["log_wall_span_s"] else np.nan
            row["timestamp_energy_to_nominal_energy_ratio"] = timestamp_energy / nominal_energy if nominal_energy else np.nan
            row["energy_per_img_J_integrated_log_timestamps"] = timestamp_energy_per_img
            run_rows.append(row)

    run_df = pd.DataFrame(run_rows)
    grid_df = pd.DataFrame(grid_rows)
    schema_df = pd.DataFrame(schema_rows)
    run_df.to_csv(OUT / "run_level_audit_and_recomputed_metrics.csv", index=False)
    grid_df.to_csv(OUT / "grid_completeness.csv", index=False)
    (schema_df.groupby(["dataset", "file_type", "schema"], dropna=False)
     .size().reset_index(name="files")
     .to_csv(OUT / "run_file_schema_summary.csv", index=False))
    schema_df.to_csv(OUT / "run_file_schema_by_file.csv", index=False)
    return run_df, grid_df, schema_df


def audit_order(
    run_df: pd.DataFrame,
    manifests: dict[str, pd.DataFrame],
    running_orders: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, Any]] = []
    positions: list[dict[str, Any]] = []
    profile_order = ["slow", "medium", "fast"]
    model_order = ["resnet50", "mobilenet_v2"]
    precision_order = ["fp16", "fp32", "fp64"]
    for label, manifest in manifests.items():
        sub = run_df[run_df.dataset == label]
        batches = sorted(sub.batch.unique().tolist())
        base = [f"{m}_{p}_{prof}_BS{b}" for prof in profile_order for m in model_order for p in precision_order for b in batches]
        actual = manifest.prefix.astype(str).tolist()
        seeds = sorted(pd.to_numeric(sub.seed, errors="coerce").dropna().astype(int).unique().tolist())
        seed = seeds[0] if len(seeds) == 1 else None
        shuffled = base.copy()
        if seed is not None:
            random.Random(seed).shuffle(shuffled)
        exact_unshuffled = actual == base
        exact_seed_shuffle = actual == shuffled if seed is not None else False
        base_pos = {p: i + 1 for i, p in enumerate(base)}
        actual_pos = {p: i + 1 for i, p in enumerate(actual)}
        if set(actual) == set(base):
            rho = float(stats.spearmanr([base_pos[p] for p in actual], list(range(1, len(actual) + 1))).statistic)
        else:
            rho = np.nan
        summaries.append({
            "dataset": label,
            "n": len(actual),
            "batch_levels": "|".join(map(str, batches)),
            "seed_unique": "|".join(map(str, seeds)),
            "manifest_exact_unshuffled": exact_unshuffled,
            "manifest_exact_python_seed_shuffle": exact_seed_shuffle,
            "state_running_order_exact_manifest": running_orders.get(label, []) == actual,
            "spearman_actual_vs_fixed_base_order": rho,
            "profile_transitions": sum(a.split("_")[-2] != b.split("_")[-2] for a, b in zip(actual[:-1], actual[1:])),
        })
        for p in actual:
            positions.append({
                "dataset": label,
                "prefix": p,
                "actual_position": actual_pos[p],
                "base_fixed_position": base_pos.get(p),
                "previous_prefix": actual[actual_pos[p] - 2] if actual_pos[p] > 1 else "",
            })
    summary_df = pd.DataFrame(summaries)
    pos_df = pd.DataFrame(positions)
    summary_df.to_csv(OUT / "run_order_summary.csv", index=False)
    pos_df.to_csv(OUT / "run_order_positions.csv", index=False)
    return summary_df, pos_df


def clock_summary(run_df: pd.DataFrame) -> pd.DataFrame:
    cols = ["cpu_freq_mode_MHz", "cpu_freq_max_MHz", "emc_freq_mode_MHz", "emc_freq_max_MHz", "gpu_freq_mode_MHz", "gpu_freq_max_MHz"]
    rows: list[dict[str, Any]] = []
    for (dataset, profile), g in run_df.groupby(["dataset", "profile"]):
        row: dict[str, Any] = {"dataset": dataset, "profile": profile, "runs": len(g)}
        for c in cols:
            row[f"{c}_median_across_runs"] = float(g[c].median())
            vals = sorted(g[c].dropna().unique().tolist())
            row[f"{c}_unique"] = "|".join(f"{v:g}" for v in vals)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "clock_signature_by_dataset_profile.csv", index=False)
    signatures = (run_df.groupby([
        "dataset", "profile", "cpu_freq_mode_MHz", "cpu_freq_max_MHz",
        "emc_freq_mode_MHz", "emc_freq_max_MHz", "gpu_freq_mode_MHz", "gpu_freq_max_MHz",
    ], dropna=False).size().reset_index(name="runs"))
    signatures.to_csv(OUT / "clock_signature_counts.csv", index=False)
    return out


def state_timing_alignment(run_df: pd.DataFrame, state_events_path: Path) -> pd.DataFrame:
    events = pd.read_csv(state_events_path)
    final = events[(events["status"] == "done") & events["duration_s"].notna()][
        ["dataset", "run_id", "duration_s"]
    ].rename(columns={"run_id": "prefix", "duration_s": "state_runner_duration_s"})
    out = run_df.merge(final, on=["dataset", "prefix"], how="left", validate="one_to_one")
    out = out[[
        "dataset", "prefix", "model", "precision", "profile", "batch",
        "meta_duration_s", "log_wall_span_s", "state_runner_duration_s",
    ]].copy()
    out["state_to_meta_duration_ratio"] = out.state_runner_duration_s / out.meta_duration_s
    out["state_to_log_wall_span_ratio"] = out.state_runner_duration_s / out.log_wall_span_s
    out["log_to_meta_duration_ratio"] = out.log_wall_span_s / out.meta_duration_s
    out.to_csv(OUT / "state_power_meta_timing_alignment.csv", index=False)
    (out.groupby("dataset").agg(
        runs=("prefix", "size"),
        state_meta_ratio_min=("state_to_meta_duration_ratio", "min"),
        state_meta_ratio_median=("state_to_meta_duration_ratio", "median"),
        state_meta_ratio_p95=("state_to_meta_duration_ratio", lambda x: x.quantile(.95)),
        state_meta_ratio_max=("state_to_meta_duration_ratio", "max"),
        state_log_ratio_median=("state_to_log_wall_span_ratio", "median"),
        state_log_ratio_max=("state_to_log_wall_span_ratio", "max"),
    ).reset_index().to_csv(OUT / "state_power_meta_timing_summary.csv", index=False))
    return out


def methodological_summaries(run_df: pd.DataFrame) -> None:
    # The two definitions are intentionally both retained; they represent different
    # boundaries and must not be silently interchanged.
    (run_df.groupby(["dataset", "model", "precision"]).agg(
        runs=("prefix", "size"),
        epi_wall_to_P_over_T_min=("reported_epi_to_power_over_throughput_ratio", "min"),
        epi_wall_to_P_over_T_median=("reported_epi_to_power_over_throughput_ratio", "median"),
        epi_wall_to_P_over_T_p95=("reported_epi_to_power_over_throughput_ratio", lambda x: x.quantile(.95)),
        epi_wall_to_P_over_T_max=("reported_epi_to_power_over_throughput_ratio", "max"),
        inference_fraction_median=("inference_time_fraction_of_meta_duration", "median"),
        power_window_to_meta_median=("log_wall_span_to_meta_duration_ratio", "median"),
    ).reset_index().to_csv(OUT / "energy_definition_boundary_summary.csv", index=False))

    (run_df.groupby("dataset").agg(
        runs=("prefix", "size"),
        conventional_p50_differs=("latency_p50_higher_minus_conventional_ms", lambda x: int((x != 0).sum())),
        p50_difference_median_ms=("latency_p50_higher_minus_conventional_ms", "median"),
        p50_difference_max_ms=("latency_p50_higher_minus_conventional_ms", "max"),
        runs_with_gt5x_p50_stall=("latency_iterations_gt5x_p50", lambda x: int((x > 0).sum())),
        total_iterations_gt5x_p50=("latency_iterations_gt5x_p50", "sum"),
        median_throughput_sensitivity_percent=("throughput_sensitivity_percent_increase_excluding_gt5x_p50", "median"),
        p95_throughput_sensitivity_percent=("throughput_sensitivity_percent_increase_excluding_gt5x_p50", lambda x: x.quantile(.95)),
        max_throughput_sensitivity_percent=("throughput_sensitivity_percent_increase_excluding_gt5x_p50", "max"),
        median_latency_lag1=("latency_lag1_autocorrelation", "median"),
        median_power_lag1=("power_lag1_autocorrelation", "median"),
    ).reset_index().to_csv(OUT / "within_run_variability_and_stall_summary.csv", index=False))

    run_df.nlargest(30, "throughput_sensitivity_percent_increase_excluding_gt5x_p50")[
        [
            "dataset", "prefix", "latency_avg_ms", "latency_p50_ms", "latency_p95_ms",
            "latency_p99_ms", "latency_max_ms", "latency_iterations_gt5x_p50",
            "throughput_img_s", "throughput_img_s_excluding_gt5x_p50_sensitivity",
            "throughput_sensitivity_percent_increase_excluding_gt5x_p50",
        ]
    ].to_csv(OUT / "largest_latency_stall_sensitivity.csv", index=False)

    # Device variability separated by profile makes the unstable fast clock regime visible.
    devices = {"MasterJetson", "Clone1", "Clone2", "Clone3"}
    overlap = run_df[run_df.dataset.isin(devices) & run_df.batch.isin([1, 4, 8, 16])]
    rows: list[dict[str, Any]] = []
    metrics = ["throughput_img_s", "avg_power_W", "energy_per_img_J_avg_power_times_meta_duration"]
    for (prefix, profile), g in overlap.groupby(["prefix", "profile"]):
        if set(g.dataset) != devices:
            continue
        for metric in metrics:
            vals = g[metric].to_numpy(dtype=float)
            rows.append({
                "prefix": prefix, "profile": profile, "metric": metric,
                "cv": float(np.std(vals, ddof=1) / np.mean(vals)),
                "max_to_min_ratio": float(np.max(vals) / np.min(vals)),
            })
    prof = pd.DataFrame(rows)
    (prof.groupby(["profile", "metric"]).agg(
        configurations=("cv", "size"), median_cv=("cv", "median"),
        p75_cv=("cv", lambda x: x.quantile(.75)),
        p95_cv=("cv", lambda x: x.quantile(.95)),
        median_max_to_min=("max_to_min_ratio", "median"),
    ).reset_index().to_csv(OUT / "four_device_variability_by_profile_summary.csv", index=False))

    clone_set = {"Clone1", "Clone2", "Clone3"}
    clone_rows: list[dict[str, Any]] = []
    clone_data = run_df[run_df.dataset.isin(clone_set)]
    for keys, g in clone_data.groupby(["model", "precision", "profile", "batch"]):
        if set(g.dataset) != clone_set:
            continue
        for metric in metrics:
            vals = g[metric].to_numpy(dtype=float)
            clone_rows.append({
                "model": keys[0], "precision": keys[1], "profile": keys[2], "batch": keys[3],
                "metric": metric, "cv": float(np.std(vals, ddof=1) / np.mean(vals)),
                "max_to_min_ratio": float(np.max(vals) / np.min(vals)),
            })
    clone_prof = pd.DataFrame(clone_rows)
    (clone_prof.groupby(["profile", "metric"]).agg(
        configurations=("cv", "size"), median_cv=("cv", "median"),
        p75_cv=("cv", lambda x: x.quantile(.75)),
        p95_cv=("cv", lambda x: x.quantile(.95)),
        max_cv=("cv", "max"),
        median_max_to_min=("max_to_min_ratio", "median"),
    ).reset_index().to_csv(OUT / "three_clone_variability_by_profile_summary.csv", index=False))


def compare_fixed_random(run_df: pd.DataFrame, positions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fixed = run_df[run_df.dataset == "fixed_2026-03"].copy()
    random_df = run_df[run_df.dataset == "random_2026-03"].copy()
    merged = fixed.merge(random_df, on=["prefix", "model", "precision", "profile", "batch"], suffixes=("_fixed", "_random"), validate="one_to_one")
    pfix = positions[positions.dataset == "fixed_2026-03"][["prefix", "actual_position"]].rename(columns={"actual_position": "position_fixed"})
    pran = positions[positions.dataset == "random_2026-03"][["prefix", "actual_position"]].rename(columns={"actual_position": "position_random"})
    merged = merged.merge(pfix, on="prefix").merge(pran, on="prefix")
    merged["position_change_random_minus_fixed"] = merged.position_random - merged.position_fixed
    metrics = [
        "throughput_img_s", "latency_avg_ms", "avg_power_W",
        "energy_per_img_J_avg_power_times_meta_duration",
        "energy_per_img_J_integrated_log_timestamps",
        "energy_per_img_J_power_over_throughput_proxy",
        "inference_time_fraction_of_meta_duration",
    ]
    summaries: list[dict[str, Any]] = []
    for metric in metrics:
        for group_name, group_cols in [
            ("all", []), ("by_model", ["model"]), ("by_precision", ["precision"]), ("by_profile", ["profile"]),
        ]:
            iterator = [("all", merged)] if not group_cols else merged.groupby(group_cols[0])
            for level, g in iterator:
                s = paired_summary(g[f"{metric}_random"], g[f"{metric}_fixed"])
                summaries.append({"metric": metric, "grouping": group_name, "level": level, "numerator": "random", "denominator": "fixed", **s})
        merged[f"log_ratio_random_fixed__{metric}"] = np.log(merged[f"{metric}_random"] / merged[f"{metric}_fixed"])
    summary_df = pd.DataFrame(summaries)
    summary_df["wilcoxon_p_holm_within_grouping"] = np.nan
    for _, idx in summary_df.groupby(["grouping", "level"]).groups.items():
        summary_df.loc[idx, "wilcoxon_p_holm_within_grouping"] = holm_adjust(summary_df.loc[idx, "wilcoxon_p"])

    temp_diff = merged["max_gpu_or_tj_temp_C_random"] - merged["max_gpu_or_tj_temp_C_fixed"]
    t_ci = stats.t.interval(.95, len(temp_diff) - 1, loc=float(temp_diff.mean()), scale=float(stats.sem(temp_diff)))
    pd.DataFrame([{
        "n": len(temp_diff),
        "fixed_campaign_median_max_temp_C": float(merged.max_gpu_or_tj_temp_C_fixed.median()),
        "random_campaign_median_max_temp_C": float(merged.max_gpu_or_tj_temp_C_random.median()),
        "random_minus_fixed_mean_difference_C": float(temp_diff.mean()),
        "mean_difference_ci95_low_C": float(t_ci[0]),
        "mean_difference_ci95_high_C": float(t_ci[1]),
        "random_minus_fixed_median_difference_C": float(temp_diff.median()),
    }]).to_csv(OUT / "fixed_vs_random_temperature_difference.csv", index=False)

    order_assoc: list[dict[str, Any]] = []
    order_metrics = metrics + ["max_gpu_or_tj_temp_C"]
    for metric in order_metrics:
        if f"log_ratio_random_fixed__{metric}" in merged:
            y = merged[f"log_ratio_random_fixed__{metric}"]
        else:
            y = merged[f"{metric}_random"] - merged[f"{metric}_fixed"]
        for xname in ["position_random", "position_fixed", "position_change_random_minus_fixed"]:
            r = stats.spearmanr(merged[xname], y, nan_policy="omit")
            order_assoc.append({
                "metric": metric,
                "order_variable": xname,
                "n": int((np.isfinite(merged[xname]) & np.isfinite(y)).sum()),
                "spearman_rho": float(r.statistic),
                "p": float(r.pvalue),
            })
    order_df = pd.DataFrame(order_assoc)
    order_df["p_holm"] = holm_adjust(order_df.p)
    merged.to_csv(OUT / "fixed_vs_random_matched_runs.csv", index=False)
    summary_df.to_csv(OUT / "fixed_vs_random_paired_statistics.csv", index=False)
    order_df.to_csv(OUT / "fixed_vs_random_order_association.csv", index=False)
    return merged, summary_df, order_df


def device_comparison(run_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    devices = ["MasterJetson", "Clone1", "Clone2", "Clone3"]
    d = run_df[run_df.dataset.isin(devices)].copy()
    overlap = d[d.batch.isin([1, 4, 8, 16])].copy()
    key = ["prefix", "model", "precision", "profile", "batch"]
    metrics = [
        "throughput_img_s", "latency_avg_ms", "avg_power_W",
        "energy_per_img_J_avg_power_times_meta_duration",
        "energy_per_img_J_integrated_log_timestamps",
        "energy_per_img_J_power_over_throughput_proxy",
    ]
    variability: list[dict[str, Any]] = []
    for keys, g in overlap.groupby(key):
        if set(g.dataset) != set(devices):
            continue
        base = dict(zip(key, keys))
        for metric in metrics:
            vals = g.set_index("dataset")[metric].reindex(devices).to_numpy(dtype=float)
            variability.append({
                **base, "metric": metric, "n_devices": len(vals),
                "mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)),
                "cv": float(np.std(vals, ddof=1) / np.mean(vals)),
                "min": float(np.min(vals)), "max": float(np.max(vals)),
                "max_to_min_ratio": float(np.max(vals) / np.min(vals)),
            })
    var_df = pd.DataFrame(variability)

    friedman_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for metric in metrics:
        piv = overlap.pivot_table(index="prefix", columns="dataset", values=metric, aggfunc="first").dropna(subset=devices)
        arrays = [piv[x].to_numpy(dtype=float) for x in devices]
        fr = stats.friedmanchisquare(*arrays)
        n, k = len(piv), len(devices)
        friedman_rows.append({
            "metric": metric, "blocks": n, "devices": k,
            "friedman_chi2": float(fr.statistic), "p": float(fr.pvalue),
            "kendalls_W": float(fr.statistic / (n * (k - 1))),
        })
        metric_pair_start = len(pair_rows)
        for i, a in enumerate(devices):
            for b in devices[i + 1:]:
                s = paired_summary(piv[a], piv[b])
                pair_rows.append({"metric": metric, "numerator": a, "denominator": b, **s})
        pvals = [r["wilcoxon_p"] for r in pair_rows[metric_pair_start:]]
        for row, adj in zip(pair_rows[metric_pair_start:], holm_adjust(pvals)):
            row["wilcoxon_p_holm_within_metric"] = adj
    friedman_df = pd.DataFrame(friedman_rows)
    friedman_df["p_holm"] = holm_adjust(friedman_df.p)
    pair_df = pd.DataFrame(pair_rows)

    cv_summary = (var_df.groupby("metric").agg(
        configurations=("cv", "size"), median_cv=("cv", "median"),
        p25_cv=("cv", lambda x: x.quantile(.25)), p75_cv=("cv", lambda x: x.quantile(.75)),
        p95_cv=("cv", lambda x: x.quantile(.95)), max_cv=("cv", "max"),
        median_max_to_min=("max_to_min_ratio", "median"),
        p95_max_to_min=("max_to_min_ratio", lambda x: x.quantile(.95)),
    ).reset_index())
    var_df.to_csv(OUT / "four_device_variability_by_configuration.csv", index=False)
    cv_summary.to_csv(OUT / "four_device_variability_summary.csv", index=False)
    friedman_df.to_csv(OUT / "four_device_friedman_tests.csv", index=False)
    pair_df.to_csv(OUT / "four_device_pairwise_statistics.csv", index=False)
    return var_df, cv_summary, friedman_df, pair_df


def clone_analysis(run_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    clones = run_df[run_df.dataset.isin(["Clone1", "Clone2", "Clone3"])].copy()
    metrics = [
        "throughput_img_s", "latency_avg_ms", "avg_power_W",
        "energy_per_img_J_avg_power_times_meta_duration",
        "energy_per_img_J_integrated_log_timestamps",
        "energy_per_img_J_power_over_throughput_proxy",
    ]
    config = (clones.groupby(["model", "precision", "profile", "batch"])[metrics]
              .agg(["mean", "std", "min", "max"]).reset_index())
    config.columns = ["_".join([str(x) for x in c if x]).rstrip("_") if isinstance(c, tuple) else c for c in config.columns]
    config.to_csv(OUT / "clone_config_means_sd.csv", index=False)

    contrasts: list[dict[str, Any]] = []

    def factor_contrast(factor: str, numerator: Any, denominator: Any, groupby: list[str]) -> None:
        id_cols = ["dataset", "model", "precision", "profile", "batch"]
        keep = [c for c in id_cols if c != factor]
        a = clones[clones[factor] == numerator][keep + metrics].copy()
        b = clones[clones[factor] == denominator][keep + metrics].copy()
        m = a.merge(b, on=keep, suffixes=("_num", "_den"), validate="one_to_one")
        for metric in metrics:
            for levels, g in m.groupby(groupby, dropna=False):
                if not isinstance(levels, tuple):
                    levels = (levels,)
                level_text = "|".join(f"{k}={v}" for k, v in zip(groupby, levels))
                contrasts.append({
                    "factor": factor, "numerator": numerator, "denominator": denominator,
                    "metric": metric, "group": level_text,
                    **paired_summary(g[f"{metric}_num"], g[f"{metric}_den"]),
                })

    factor_contrast("precision", "fp32", "fp16", ["model"])
    factor_contrast("precision", "fp64", "fp32", ["model"])
    factor_contrast("profile", "fast", "slow", ["model", "precision"])
    factor_contrast("batch", 128, 1, ["model", "precision"])
    contrast_df = pd.DataFrame(contrasts)
    contrast_df["wilcoxon_p_holm"] = holm_adjust(contrast_df.wilcoxon_p)
    contrast_df.to_csv(OUT / "clone_factor_contrasts.csv", index=False)

    winners: list[dict[str, Any]] = []
    for (model, precision), g in config.groupby(["model", "precision"]):
        for metric, objective in [
            ("throughput_img_s_mean", "max"),
            ("energy_per_img_J_avg_power_times_meta_duration_mean", "min"),
            ("energy_per_img_J_integrated_log_timestamps_mean", "min"),
        ]:
            idx = g[metric].idxmax() if objective == "max" else g[metric].idxmin()
            row = g.loc[idx]
            winners.append({
                "model": model, "precision": precision, "objective": f"{objective} {metric}",
                "profile": row.profile, "batch": int(row.batch),
                "mean": row[metric], "sd": row[metric.replace("_mean", "_std")],
            })
    winner_df = pd.DataFrame(winners)
    winner_df.to_csv(OUT / "clone_best_mean_configurations.csv", index=False)
    return config, contrast_df, winner_df


def plotcode_audit(plot_dir: Path, inventory: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(plot_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix == ".py":
            source = path.read_text(encoding="utf-8", errors="replace")
            try:
                ast.parse(source, filename=str(path))
                syntax_ok, syntax_error = True, ""
            except SyntaxError as exc:
                syntax_ok, syntax_error = False, str(exc)
                add_issue("Plotcode", "error", "python_syntax", str(exc), rel(path))
            rows.append({
                "path": rel(path), "bytes": path.stat().st_size,
                "syntax_ok": syntax_ok, "syntax_error": syntax_error,
                "mentions_energy_per_img": "energy_per_img_J" in source,
                "mentions_integrated_energy": "energy_J_integrated" in source,
                "mentions_edp": "edp" in source.lower(),
                "batch_marker_support_1_4_8_16": all(f"{b}:" in source.replace(" ", "") for b in [1, 4, 8, 16]),
                "batch_marker_support_32_64_128": all(f"{b}:" in source.replace(" ", "") for b in [32, 64, 128]),
            })
        elif path.suffix == ".code-workspace":
            try:
                json.loads(path.read_text(encoding="utf-8"))
                ok, err = True, ""
            except Exception as exc:
                ok, err = False, str(exc)
                add_issue("Plotcode", "error", "workspace_json", err, rel(path))
            rows.append({"path": rel(path), "bytes": path.stat().st_size, "syntax_ok": ok, "syntax_error": err})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "plotcode_syntax_and_feature_audit.csv", index=False)
    return out


def create_summary_tables(
    inventory: pd.DataFrame,
    run_df: pd.DataFrame,
    grid_df: pd.DataFrame,
    state_summary: pd.DataFrame,
) -> None:
    file_summary = (inventory.groupby(["dataset", "extension"], dropna=False)
                    .agg(files=("path", "size"), bytes=("bytes", "sum")).reset_index())
    file_summary.to_csv(OUT / "file_count_summary.csv", index=False)
    run_summary_rows = []
    for dataset, g in run_df.groupby("dataset"):
        run_summary_rows.append({
            "dataset": dataset, "runs": len(g),
            "models": "|".join(sorted(g.model.unique())),
            "precisions": "|".join(sorted(g.precision.unique())),
            "profiles": "|".join(sorted(g.profile.unique())),
            "batches": "|".join(map(str, sorted(g.batch.unique()))),
            "seeds": "|".join(map(str, sorted(g.seed.unique()))),
            "total_images_all_runs": int(g.total_images.sum()),
            "formula_mismatch_runs": int((g.formula_checks_failed != "").sum()),
            "runs_with_perf_nan": int((g.perf_nan_cells > 0).sum()),
            "runs_with_power_nan": int((g.power_nan_cells > 0).sum()),
            "runs_with_duplicate_perf_iter": int((g.perf_duplicate_iters > 0).sum()),
            "runs_with_duplicate_power_index": int((g.power_duplicate_indices > 0).sum()),
            "runs_with_log_gaps": int((g.log_gap_count_gt1s > 0).sum()),
            "sum_missing_log_seconds": float(g.log_missing_seconds_from_gaps.sum()),
            "median_wallspan_meta_duration_ratio": float(g.log_wall_span_to_meta_duration_ratio.median()),
            "median_timestamp_to_nominal_energy_ratio": float(g.timestamp_energy_to_nominal_energy_ratio.median()),
            "median_reported_epi_to_P_over_T_ratio": float(g.reported_epi_to_power_over_throughput_ratio.median()),
            "median_inference_time_fraction_of_meta_duration": float(g.inference_time_fraction_of_meta_duration.median()),
        })
    pd.DataFrame(run_summary_rows).to_csv(OUT / "dataset_quality_summary.csv", index=False)
    pd.DataFrame(
        issues,
        columns=["dataset", "severity", "category", "detail", "path"],
    ).to_csv(OUT / "issues.csv", index=False)


def main() -> None:
    dataset_dirs = sorted([p for p in ROOT.iterdir() if p.is_dir() and p.name.startswith("Ergebnisse_")])
    plot_dir = ROOT / "Plotcode"
    inventory = inventory_files(dataset_dirs, plot_dir)
    manifests, enriched = read_manifests(dataset_dirs)
    state_summary, running_orders = parse_state_files(dataset_dirs, manifests)
    run_df, grid_df, _ = audit_runs(dataset_dirs, manifests, enriched)
    state_timing_alignment(run_df, OUT / "grid_state_events.csv")
    order_summary, positions = audit_order(run_df, manifests, running_orders)
    clock_summary(run_df)
    methodological_summaries(run_df)
    compare_fixed_random(run_df, positions)
    device_comparison(run_df)
    clone_analysis(run_df)
    plotcode_audit(plot_dir, inventory)
    create_summary_tables(inventory, run_df, grid_df, state_summary)
    error_count = sum(issue["severity"] == "error" for issue in issues)
    print(json.dumps({
        "files_read_and_hashed": int(len(inventory)),
        "runs_audited": int(len(run_df)),
        "issues": len(issues),
        "integrity_errors": error_count,
        "datasets": sorted(run_df.dataset.unique().tolist()),
        "out": str(OUT),
    }, ensure_ascii=False, indent=2))
    if error_count:
        raise RuntimeError(
            f"Audit failed with {error_count} integrity error(s); inspect {OUT / 'issues.csv'}"
        )


if __name__ == "__main__":
    main()
