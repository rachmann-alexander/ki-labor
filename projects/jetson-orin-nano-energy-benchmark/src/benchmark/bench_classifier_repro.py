#!/usr/bin/env python3
"""Deterministic image classification benchmark for ResNet/MobileNet.

This script mirrors the behaviour of the original bench_*.py files but
enforces strict reproducibility (deterministic kernels, TF32 disabled,
single-threaded DataLoader). Use together with run_grid.py for automated
measurements.
"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import os
import random

import numpy as np

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # deterministische cuBLAS-Configs

import torch  # noqa: E402

# Reproduzierbarkeit erzwingen
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  # Repro > autotuned speed
# TF32 vollständig deaktivieren (ehrliches FP32, stabile Pfade)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def seed_all(seed: int = 123) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loader(data_dir: str, batch: int, imgsz: int, workers: int) -> "torch.utils.data.DataLoader":
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader

    if not os.path.isdir(data_dir):
        raise RuntimeError(f"DATA_DIR nicht gefunden: {data_dir}")

    tfm = transforms.Compose(
        [
            transforms.Resize(imgsz, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(imgsz),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    dataset = datasets.ImageFolder(root=data_dir, transform=tfm)
    if len(dataset) == 0:
        raise RuntimeError(f"ImageFolder ist leer: {data_dir}")

    loader = DataLoader(
        dataset,
        batch_size=batch,
        shuffle=False,  # feste Reihenfolge
        num_workers=0,  # erzwingt deterministische Latenzen
        pin_memory=True,  # leicht bessere Transfers, ohne Asynchronität
        drop_last=False,
        persistent_workers=False,
    )
    return loader


def load_model(name: str, pretrained: bool):
    try:
        import torchvision.models as models
        from torchvision.models import (
            MobileNet_V2_Weights,
            MobileNet_V3_Large_Weights,
            MobileNet_V3_Small_Weights,
            ResNet50_Weights,
        )
    except Exception:  # pragma: no cover - runtime installation hint
        print(
            "ERROR: torchvision fehlt. Bitte im Container installieren: "
            "pip3 install --no-cache-dir torchvision pillow",
            file=sys.stderr,
        )
        raise

    name = name.lower()
    weights = None
    if pretrained:
        if name in {"resnet50", "resnet"}:
            weights = ResNet50_Weights.DEFAULT
        elif name in {"mobilenet_v2", "mobilenet"}:
            weights = MobileNet_V2_Weights.DEFAULT
        elif name in {"mobilenet_v3_small", "mbv3s"}:
            weights = MobileNet_V3_Small_Weights.DEFAULT
        elif name in {"mobilenet_v3_large", "mbv3l"}:
            weights = MobileNet_V3_Large_Weights.DEFAULT

    if name in {"resnet50", "resnet"}:
        model = models.resnet50(weights=weights)
    elif name in {"mobilenet_v2", "mobilenet"}:
        model = models.mobilenet_v2(weights=weights)
    elif name in {"mobilenet_v3_small", "mbv3s"}:
        model = models.mobilenet_v3_small(weights=weights)
    elif name in {"mobilenet_v3_large", "mbv3l"}:
        model = models.mobilenet_v3_large(weights=weights)
    else:
        raise ValueError(f"Unbekanntes Modell: {name}")

    model.eval()
    return model


@torch.no_grad()
def run_benchmark(
    model: torch.nn.Module,
    device: torch.device,
    loader: "torch.utils.data.DataLoader",
    warmup: int,
    iters: Optional[int],
    out_csv: Optional[Path],
    measure_acc: bool,
    input_dtype: torch.dtype,
    progress_every: int,
    expected_images: Optional[int],
) -> Optional[Dict[str, float]]:
    # ersten Batch für Warmup vorbereiten
    it_loader = iter(loader)
    try:
        x0, y0 = next(it_loader)
    except StopIteration:
        print("Keine Daten im Loader gefunden.", file=sys.stderr)
        return None

    x0 = x0.to(device=device, dtype=input_dtype)
    y0 = y0.to(device)

    # Warmup
    for _ in range(warmup):
        _ = model(x0)
        if device.type == "cuda":
            torch.cuda.synchronize()

    writer = None
    csv_file = None
    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        csv_file = out_csv.open("w", newline="")
        writer = csv.writer(csv_file)
        writer.writerow(["iter", "batch", "latency_ms", "img_per_s", "acc_top1"])

    times = []
    total_images = 0
    acc_correct = 0
    acc_total = 0
    iteration = 0
    start = time.perf_counter()
    progress_marker = progress_every if progress_every > 0 else None

    try:
        while True:
            for xb, yb in loader:
                xb = xb.to(device=device, dtype=input_dtype)
                yb = yb.to(device)

                t0 = time.perf_counter()
                logits = model(xb)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                dt = time.perf_counter() - t0

                bs = xb.size(0)
                ips = bs / dt
                times.append(dt)
                total_images += bs

                batch_acc = float("nan")
                if measure_acc:
                    correct = (logits.argmax(1) == yb).sum().item()
                    acc_correct += correct
                    acc_total += bs
                    batch_acc = correct / bs

                if writer:
                    writer.writerow([iteration, bs, dt * 1000.0, ips, batch_acc])
                    csv_file.flush()

                iteration += 1

                if progress_marker is not None:
                    while total_images >= progress_marker:
                        if expected_images and expected_images > 0:
                            percent = min(100.0, (total_images / expected_images) * 100.0)
                            print(f"[PROGRESS] {total_images}/{expected_images} images ({percent:.1f}%)")
                        else:
                            print(f"[PROGRESS] {total_images} images verarbeitet")
                        sys.stdout.flush()
                        progress_marker += progress_every
                if iters is not None and iteration >= iters:
                    raise StopIteration
            if iters is None:
                break
    except StopIteration:
        pass
    finally:
        if csv_file:
            csv_file.close()

    if not times:
        print("Keine Messwerte erzeugt.", file=sys.stderr)
        return None

    duration = time.perf_counter() - start
    avg = sum(times) / len(times)
    p50 = sorted(times)[len(times) // 2]
    throughput = total_images / sum(times)
    top1 = (acc_correct / max(1, acc_total)) if measure_acc else float("nan")

    print("=== BENCH SUMMARY ===")
    print(f"Device={device.type}  Precision={str(input_dtype)}")
    print(f"Iters={iteration}  Warmup={warmup}")
    print(f"Latency_avg={avg * 1000:.3f} ms  Latency_p50={p50 * 1000:.3f} ms  Throughput={throughput:.2f} img/s")
    if measure_acc:
        print(f"Top1={top1 * 100:.2f} %")

    return {
        "latency_avg_ms": avg * 1000.0,
        "latency_p50_ms": p50 * 1000.0,
        "throughput_img_s": throughput,
        "top1": top1,
        "iterations": iteration,
        "duration_s": duration,
    }


def resolve_device(requested_precision: str) -> torch.dtype:
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "fp64": torch.float64}
    return dtype_map[requested_precision]


def get_git_info() -> Dict[str, Optional[str]]:
    try:
        import subprocess

        commit = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            )
            .stdout.strip()
        )
    except Exception:
        commit = None

    try:
        import subprocess

        status = (
            subprocess.run(
                ["git", "status", "--short", "--untracked-files=no"],
                check=True,
                capture_output=True,
                text=True,
            )
            .stdout.strip()
        )
    except Exception:
        status = None

    return {"git_commit": commit, "git_status": status}


def get_torchvision_version() -> Optional[str]:
    try:
        import torchvision

        return torchvision.__version__
    except Exception:
        return None


def write_meta(path: Path, meta: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser("Deterministic ResNet/MobileNet Benchmark")
    parser.add_argument("--model", default="resnet50")
    parser.add_argument("--precision", default="fp32", choices=["fp16", "fp32", "fp64"])
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=0, help="0 = gesamte Validation-Epoche")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--measure_accuracy", action="store_true")
    parser.add_argument("--out_csv", type=str, default=None)
    parser.add_argument("--meta_out", type=str, default=None)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=5000,
        help="Gibt alle X verarbeiteten Bilder einen Fortschritts-Hinweis aus (0=aus).",
    )

    args = parser.parse_args()

    seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = args.precision.lower()
    dtype = resolve_device(precision)
    if precision == "fp16" and device.type != "cuda":
        print("WARNUNG: FP16 nur auf CUDA verfügbar, falle auf FP32 zurück.", file=sys.stderr)
        precision = "fp32"
        dtype = torch.float32

    if precision == "fp64":
        print("HINWEIS: FP64 ist deutlich langsamer und benötigt mehr Speicher.", file=sys.stderr)

    model = load_model(args.model, pretrained=args.pretrained).to(device=device, dtype=dtype)
    loader = build_loader(args.data_dir, args.batch, args.imgsz, workers=0)
    dataset_images = len(loader.dataset)

    max_iters = None if args.iters <= 0 else args.iters
    if max_iters is None:
        expected_images = dataset_images
    else:
        expected_images = min(dataset_images, max_iters * args.batch)
    metrics = run_benchmark(
        model=model,
        device=device,
        loader=loader,
        warmup=args.warmup,
        iters=max_iters,
        out_csv=Path(args.out_csv) if args.out_csv else None,
        measure_acc=args.measure_accuracy,
        input_dtype=dtype,
        progress_every=max(0, args.progress_every),
        expected_images=expected_images,
    )

    if metrics is None:
        sys.exit(1)

    if args.meta_out:
        meta_path = Path(args.meta_out)
        meta = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "model": args.model,
            "precision": precision,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "warmup": args.warmup,
            "iters": max_iters,
            "measure_accuracy": args.measure_accuracy,
            "seed": args.seed,
            "device_type": device.type,
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torch_version": torch.__version__,
            "torchvision_version": get_torchvision_version(),
            "metrics": metrics,
        }
        meta.update(get_git_info())
        write_meta(meta_path, meta)


if __name__ == "__main__":
    main()
