# Electronics revision data release

This tagged release contains the complete raw telemetry, per-batch records, metadata, state/manifest files, analysis code, machine-readable audit outputs, and editable figure sources supporting the revised manuscript. The six experimental campaigns comprise 2,268 files and 1,880,437,839 uncompressed bytes; the accompanying `Plotcode/` snapshot brings the audited experimental-source inventory to 2,282 files and 1,880,535,529 bytes (1.751 GiB).

## Assets and extraction

The release uses one ZIP per campaign so that every GitHub Release asset remains below the 2 GiB per-file limit. Extract all seven ZIP files into the same empty directory. The result must contain these common-root paths:

```text
Ergebnisse_13.03.2026 (feste Reihenfolge)/
Ergebnisse_23.03.2026 (zufällige Reihenfolge)/
Ergebnisse_MasterJetson_07.07.2026/
Ergebnisse_Clone1_13.07.2026/
Ergebnisse_Clone2_13.07.2026/
Ergebnisse_Clone3_13.07.2026/
Plotcode/
revised/
validation/
```

The original campaign directory names are preserved because the reproducibility scripts use these relative paths. Campaign coverage is not a uniform factorial matrix: the March runs use batch sizes 32/64/128, the MasterJetson July campaign uses 1/4/8/16, and the three clone campaigns use 1/4/8/16/32/64/128.

## Integrity and reproduction

Verify `SHA256SUMS.txt` before extraction. From the common extraction root, use Python 3.11.9 and run:

```text
python -m pip install -r revised/analysis/requirements.txt
python revised/analysis/full_audit.py
python revised/analysis/build_revision_results.py
python revised/analysis/long_run_drift_analysis.py
```

Expected invariants are recorded in `RELEASE_MANIFEST.json`. In particular, the audit must recover 2,282 inventoried source files and 558 runs, produce zero issue rows, and reproduce zero maximum raw/CSV differences for both power and relative elapsed time. `revised/supplementary/audit/file_inventory_sha256.csv` is the authoritative file-level inventory; `analysis_outputs_sha256.csv` records deterministic derived outputs.

## Scope and redistribution

The authors release their code, raw experimental records, derived tables, and figure sources under the MIT License supplied in `LICENSE`; users must also follow any third-party terms that apply. The ImageNet validation images are not redistributed and remain subject to the dataset's access conditions. No external power-meter measurements were collected. The supplied power observations are the Jetson-reported input-rail telemetry described in the manuscript. The exact July executable and some environment identifiers were not recoverable; the manuscript and audit report state these provenance limits explicitly rather than inferring them from the later reference implementation.
