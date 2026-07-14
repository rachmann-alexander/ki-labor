# Supplementary Evidence Package

This directory is the entry point for the manuscript's submission-ready supplementary material. It contains a complete read-only audit of the source archive, machine-readable primary analyses, robustness checks, and a data dictionary.

## Supplementary items

- **S1 — Complete archive and analysis audit:** `audit/AUDIT_REPORT.md`
- **S2 — Run-level measurement-boundary table:** `../generated/measurement_boundary_metrics.csv`
- **S3 — Full-grid and FP16/FP32 factorial decomposition:** complete-grid, FP16/FP32, and FP16/FP32 slow+medium factorial result and variance-sensitivity CSVs in `../generated/`
- **S4 — Board-blocked precision and batch contrasts:** primary and excluding-fast contrast CSVs in `../generated/`
- **S5 — Requested-fast realized-state fidelity and carry-over:** `fast_state_carryover_*.csv`, including identical-workload cross-block consistency, plus `fast_clock_regime_counts.csv` and `profile_repeatability.csv`
- **S6 — Precision-by-batch crossover and protocol workload:** complete-grid and excluding-fast crossover CSVs plus batch-dependent-work CSVs in `../generated/`
- **S7 — Primary configuration summaries and figure data:** configuration, batch-curve, and trace CSVs in `../generated/`
- **S8 — Campaign and run accounting:** campaign, grid, state, and run-order CSVs
- **S9 — Utilization and tail-latency diagnostics:** utilization, percentile, and stall-sensitivity CSVs
- **S10 — Complete 558-run reconstruction:** `audit/run_level_audit_and_recomputed_metrics.csv`
- **S11 — File, schema, and manifest integrity:** SHA-256 inventory and structural-audit CSVs
- **S12 — Historical-boundary, sequence, and campaign diagnostics:** boundary, March-campaign, four-folder, inter-run-gap, successor, and temporal-drift diagnostics in `audit/`; the long-run entry point is `audit/LONG_RUN_DRIFT_ANALYSIS.md`

The exact file map and column definitions are in `DATA_DICTIONARY.md`.

## Primary estimands

The revised manuscript separates:

1. synchronized model-forward throughput;
2. throughput over the complete timestamped telemetry interval;
3. `VDD_IN` energy integrated over that same telemetry interval.

Generated columns prefixed `task_*` and the few headers containing `whole_task_*` are legacy machine names. They mean **logged-window** quantities; they are not model-forward-only energy and not a complete deployed-service boundary. The historical duration-based and power/forward-throughput fields in `audit/` are retained solely to make the earlier inconsistency reproducible.

## Central robustness evidence

- The requested-fast level follows an exact sequence rule in all 126 primary fast runs: it realizes low after slow and high after medium, and consecutive fast runs retain the previous fast regime.
- Precision/batch conclusions are therefore accompanied by analyses excluding fast.
- The 71.5% full-grid precision share is explicitly identified as being driven largely by the FP64 stress condition. Separate FP16/FP32 and FP16/FP32 slow+medium decompositions show that, in the cross-block-consistent branches, batch size accounts for 54.1% of forward-throughput variation.
- Cell-level tables document the batch-one crossover: after excluding requested fast, all 12 cells without an FP16 forward advantage occur at batch size 1, while FP16 is faster in all 72 cells at batch sizes 4--128.
- The batch-protocol table separates evaluated iterations, warm-up work, and the conditional per-batch flush count inferred from the inspected reference implementation.
- The long-run diagnostic reports archived state-event gaps, fixed-effect-adjusted successors of FP64 and upper-duration-quartile runs, boundary-temperature continuity, and temporal slopes with leave-one-board-out sensitivity ranges, without p values or causal claims. Requested-fast successors are excluded from its conservative primary scope because their realized state is already predecessor-dependent.

## Reproduction

Run from the project root:

```text
python -m pip install -r revised/analysis/requirements.txt
python revised/analysis/full_audit.py
python revised/analysis/build_revision_results.py
python revised/analysis/long_run_drift_analysis.py
```

The recorded runtime is Python 3.11.9 with NumPy 2.0.2, pandas 2.2.3, and SciPy 1.14.1. The scripts read all original experiment folders and the original Plotcode. They do not modify raw measurements. Generated evidence is written under `revised/generated/` and `revised/supplementary/audit/`.

## Provenance limits

The authors confirm that `Clone1--3` are three different physical boards, but the archive does not contain their serial numbers. It also lacks the exact executed seven-batch July runner, a non-null Git commit/status, and complete JetPack/L4T/container provenance. The public repository commit cited by the manuscript is an immutable reference implementation, not artifact-level proof of the executed July code. The supplement consistently distinguishes archived facts from author-reported details and reference-code inferences.
