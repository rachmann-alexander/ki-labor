# Energy-Efficient Image Classification on the NVIDIA Jetson Orin Nano

This folder contains the benchmark code, analysis scripts, manifests, and raw per-run measurement artifacts for the paper:

**Energy-Efficient Image Classification on the NVIDIA Jetson Orin Nano: A Full-Factorial Study of Model, Precision, Batch Size, and Power Profiles**

## Contents

- `src/benchmark/`: benchmark execution, power-profile control, and power logging scripts
- `src/preprocessing/`: dataset preparation scripts
- `src/analysis/`: scripts used to generate figures, tables, and comparison summaries
- `data/randomized_primary/`: primary randomized benchmark run used for the main energy conclusions
- `data/ordered_reference/`: ordered reference run used for rank-level robustness checks
- `results/figures/`: figures generated for the paper
- `results/tables/`: tables generated for the paper
- `docs/`: documentation of metrics, measurement boundaries, and reproduction steps
- `environment/`: software and hardware setup notes

## Dataset

The experiments use the ImageNet 2012 validation set. The image files are not redistributed in this repository and must be obtained separately from ImageNet.

## Primary Evidence

The randomized run is the primary basis for energy conclusions. The ordered reference run is provided as a robustness artifact for performance-rank comparisons.

## Measurement Boundary

Run-integrated energy includes warmup iterations, validation inference, data loading, preprocessing, and framework execution inside the logged benchmark command. It excludes power-profile setup, container startup, and post-run file copying.
