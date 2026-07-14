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

The publication-ready revision is based on a file-complete audit of six experimental campaigns. Its primary blocked factorial analysis uses the three physically distinct July board directories (`Clone1--3`); the March ordered and randomized campaigns and the smaller `MasterJetson` campaign are retained as historical or audit evidence. The complete raw records and revision materials are distributed through the tagged GitHub Release [`electronics-revision-data-v1`](https://github.com/weissbeck-lucas/ki-labor/releases/tag/electronics-revision-data-v1). Editable manuscript, analysis, audit, and figure sources are also available in [`revised/`](../../revised/).

The exact executable used for the July campaigns was not archived. Consequently, the code under `src/` is a later reference snapshot and must not be interpreted as artifact-level proof of the executed July code. The manuscript and audit report distinguish archived facts, author-reported protocol information, and reference-code reconstructions explicitly.

## Measurement Boundary

Run-integrated energy includes warmup iterations, validation inference, data loading, preprocessing, and framework execution inside the logged benchmark command. It excludes power-profile setup, container startup, and post-run file copying.

## License

The authors release their code, raw experimental records, derived tables, and figure sources under the repository-root MIT License. Third-party materials and datasets remain subject to their respective terms.
