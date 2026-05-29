#!/usr/bin/env python3
import csv
import json
import argparse
from pathlib import Path




def remap_path(p_str: str, base_dir: Path) -> Path:
    return base_dir / Path(p_str).name


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def read_perf_stats(perf_path: Path):
    total_images = 0
    iterations = 0
    with perf_path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            total_images += int(row["batch"])
            iterations += 1
    return total_images, iterations


def integrate_power_csv(power_path: Path):
    rows = []
    with power_path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                t = float(row["elapsed_s"])
                p = float(row["power_W"])
                rows.append((t, p))
            except Exception:
                continue

    if len(rows) < 2:
        return None

    energy_j = 0.0
    for i in range(len(rows) - 1):
        t0, p0 = rows[i]
        t1, p1 = rows[i + 1]
        dt = t1 - t0
        if dt <= 0:
            continue
        energy_j += 0.5 * (p0 + p1) * dt
    return energy_j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-dir",
        required=True,
        help="Ordner mit grid_manifest.csv, *_perf.csv, *_bench_meta.json und *_tegrastats_power.csv",
    )
    args = ap.parse_args()

    base = Path(args.base_dir).resolve()
    input_manifest = base / "grid_manifest.csv"
    output_manifest = base / "grid_manifest_enriched.csv"

    with input_manifest.open() as f:
        reader = csv.DictReader(f)
        base_rows = list(reader)

    original_fields = reader.fieldnames[:]
    extra_fields = [
        "energy_J",
        "energy_per_img_J",
        "energy_per_inference_J",
        "energy_J_integrated",
        "energy_per_img_J_integrated",
        "edp_Js",
        "edp_per_img_Js",
    ]
    out_fields = original_fields + extra_fields

    out_rows = []

    for row in base_rows:
        perf_path = remap_path(row["perf_csv"], base)
        meta_path = remap_path(row["bench_meta"], base)
        power_path = remap_path(row["power_csv"], base)

        avg_power_W = safe_float(row["avg_power_W"])
        duration_s = safe_float(row["duration_s"])
        latency_avg_ms = safe_float(row["latency_avg_ms"])

        total_images, iterations = read_perf_stats(perf_path)
        meta = json.load(meta_path.open())
        energy_J_integrated = integrate_power_csv(power_path)

        energy_J = None
        energy_per_img_J = None
        energy_per_inference_J = None
        energy_per_img_J_integrated = None
        edp_Js = None
        edp_per_img_Js = None

        if avg_power_W is not None and duration_s is not None:
            energy_J = avg_power_W * duration_s

        if energy_J is not None and total_images > 0:
            energy_per_img_J = energy_J / total_images

        if energy_J is not None and iterations > 0:
            energy_per_inference_J = energy_J / iterations

        if energy_J_integrated is not None and total_images > 0:
            energy_per_img_J_integrated = energy_J_integrated / total_images

        if energy_J_integrated is not None and duration_s is not None:
            edp_Js = energy_J_integrated * duration_s

        if energy_per_img_J_integrated is not None and latency_avg_ms is not None:
            edp_per_img_Js = energy_per_img_J_integrated * (latency_avg_ms / 1000.0)

        new_row = dict(row)

        # Pfade auf den aktuellen Basisordner umbiegen
        new_row["power_csv"] = str(power_path)
        new_row["perf_csv"] = str(perf_path)
        new_row["bench_meta"] = str(meta_path)

        # neue Kennzahlen anhängen
        new_row["energy_J"] = energy_J
        new_row["energy_per_img_J"] = energy_per_img_J
        new_row["energy_per_inference_J"] = energy_per_inference_J
        new_row["energy_J_integrated"] = energy_J_integrated
        new_row["energy_per_img_J_integrated"] = energy_per_img_J_integrated
        new_row["edp_Js"] = edp_Js
        new_row["edp_per_img_Js"] = edp_per_img_Js

        out_rows.append(new_row)

    with output_manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote: {output_manifest}")


if __name__ == "__main__":
    main()
