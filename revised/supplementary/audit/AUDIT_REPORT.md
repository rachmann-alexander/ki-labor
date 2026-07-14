# Supplementary Item S1 — Complete Archive and Analysis Audit

**Audit date:** 13 July 2026  
**Scope:** Six archived experiment directories and the original `Plotcode` directory  
**Audit mode:** Read-only. No raw measurement file was changed.

This report documents the complete archive audit underlying the revised manuscript. It distinguishes the manuscript's compatible primary estimands from quantities reconstructed only to explain the historical analysis. Machine-readable results are indexed in `../DATA_DICTIONARY.md`.

## 1. Measurement boundaries and terminology

The following quantities are not interchangeable:

- **Forward throughput** (`throughput_img_s`) is based only on the sum of synchronized model-forward latencies recorded in the per-batch performance file:

  `Theta_forward = N / (sum_i latency_i_ms / 1000)`.

- **Logged-window duration** (`task_time_s` in the generated CSVs; described as `logged_window_duration_s` in prose) is the elapsed time between the first and last valid telemetry timestamps.
- **Logged-window throughput** (`task_throughput_img_s`) is:

  `Theta_log = N / logged_window_duration_s`.

- **Logged-window energy** (`task_energy_J`) is the trapezoidal integral of `VDD_IN` power over the same timestamped telemetry interval:

  `E_log = integral P_VDD_IN(t) dt`.

- **Logged energy per evaluated image** (`task_energy_per_eval_image_J`) is `E_log / N`, where `N = 50,000` evaluated images. The metadata record 50 warm-up batches within the monitored benchmark protocol. Under the inspected reference implementation's full-batch warm-up behavior, `task_energy_per_processed_image_J` additionally includes `50 × batch_size` inferred warm-up images in its denominator. This alternative denominator is a provenance-qualified sensitivity field.
- **Logged-window EDP** (`task_edp_Js`) is the secondary, boundary-compatible quantity `E_log × logged_window_duration_s`.
- **Board block** denotes one of the author-confirmed distinct physical boards represented by `Clone1`, `Clone2`, or `Clone3`. Serial numbers, hardware UUIDs, and host identifiers were not archived, so the identity cannot be independently authenticated from the artifacts and the block must not be read as an isolated silicon-effect estimate.

The `task_*` prefixes are legacy machine-readable column names in `revised/generated/`. In the manuscript and this report, they mean **logged-window**, not inference-only or application-end-to-end measurements.

Three historical quantities are retained in the audit tables solely to reconstruct the earlier analysis:

1. `avg_power_W × meta.duration_s / N`;
2. `avg_power_W / throughput_img_s`, a power–forward-throughput proxy;
3. an EDP that combined duration-based energy with forward latency.

They do not share a common boundary and are not used as primary revised-manuscript estimands.

## 2. Archive scope and file inventory

Every file in the six experiment directories and `Plotcode` was read and SHA-256 hashed.

| Archive area | Files | Bytes |
|---|---:|---:|
| Ordered campaign, March | 231 | 143,630,169 |
| Randomized campaign, March | 225 | 171,598,483 |
| Clone1 | 507 | 446,481,868 |
| Clone2 | 507 | 431,573,652 |
| Clone3 | 507 | 412,776,337 |
| MasterJetson | 291 | 274,377,330 |
| Plotcode | 14 | 97,690 |
| **Total** | **2,282** | **1,880,535,529** |

The inventory contains 2,232 run-specific artifacts for 558 runs: one `perf.csv`, `bench_meta.json`, `tegrastats.log`, and `tegrastats_power.csv` per run. The remainder comprises 18 manifest/state files, 18 SVG files, and 14 Plotcode files. The audit parsed 5,421,240 batch records, 4,177,590 monitoring records or raw log lines, and 27.9 million evaluated images. The 2,289-file figure stated in the original task does not match the files present in these source directories; no measurement file was omitted from the 2,282-file inventory.

Reproduction from the project root:

```text
python revised/analysis/full_audit.py
```

## 3. Grid completeness and structural integrity

### 3.1 Complete Cartesian grids

| Campaign or board folder | Runs | Observed grid |
|---|---:|---|
| Ordered March campaign | 54 | 2 models × 3 precisions × 3 requested profiles × batches 32/64/128 |
| Randomized March campaign | 54 | Same 54-cell grid |
| Clone1, Clone2, Clone3, each | 126 | 2 × 3 × 3 × batches 1/4/8/16/32/64/128 |
| MasterJetson | 72 | 2 × 3 × 3 × batches 1/4/8/16 |

Every expected run prefix has all four run-specific files and exactly one manifest row. The MasterJetson archive consistently contains only the four smallest batch sizes. Consequently, 72 configurations are common to the four archive folders, whereas the balanced primary factorial design is the 378-run `Clone1`–`Clone3` grid on three distinct physical boards.

### 3.2 Integrity checks

- No NaN, infinity, or nonnumeric cell occurs in any of the 558 performance or 558 power CSV files.
- No duplicate performance row, iteration identifier, power row, or sample index was found.
- All iteration and sample indices begin at zero and are contiguous.
- Every run evaluates exactly 50,000 images. The final partial batch is correctly represented; for example, batch size 128 contains `390 × 128 + 80 = 50,000` evaluated images.
- All 1,116 CSV schemas and all 558 JSON schemas are internally consistent across campaigns.
- Every raw telemetry line contains a parseable timestamp and `VDD_IN` value. Raw-log power and the corresponding power CSV agree sample by sample.
- All six `grid_state.jsonl` files are syntactically complete: 558 `running` and 558 final `done` events, no `failed` or `timeout` event, and no multiple final record.
- Manifest order and state-event order agree exactly within every campaign. Metadata timestamps are strictly increasing in that order.
- No measurement files have identical contents. The only duplicate SHA-256 pair is the Plotcode pair `make_plot_06_edp_vs_throughput_all.py` and `make_plot_06_edp_vs_throughput_grid_swapped.py`.

The absolute Linux paths stored in the two manifest formats are not available in the present workspace (0/558 paths exist in each path column), but every referenced basename is present locally. `build_enriched_manifest.py` uses basename remapping. The data are therefore recoverable in this workspace, but the original manifests are not location-independent without that remapping step.

## 4. Exact reconstruction of archived metrics

For performance iteration `i`, let `b_i`, `l_i`, and `a_i` denote evaluated batch size, forward latency in milliseconds, and batch accuracy. For telemetry sample `j`, let `t_j` and `P_j` denote timestamp in seconds and `VDD_IN` power in watts.

| Archived or audited field | Exact reconstruction |
|---|---|
| `total_images` | `N = sum_i b_i = 50,000` |
| `iterations` | Number of performance rows, `K = ceil(50,000 / requested_batch)` |
| `latency_avg_ms` | Unweighted arithmetic mean, `sum_i l_i / K` |
| `latency_p50_ms` | Upper central observation, `sort(l)[K//2]`; for even `K`, not the mean of the two central values |
| `throughput_img_s` | `N / (sum_i l_i / 1000)` |
| `top1` | Batch-size-weighted value, `sum_i (a_i b_i) / N` |
| `avg_power_W` | Unweighted sample mean, `sum_j P_j / M` |
| Historical `energy_J` | `avg_power_W × meta.duration_s` |
| Historical `energy_per_img_J` | Historical `energy_J / N` |
| Historical `energy_per_inference_J` | Historical `energy_J / K`; despite its name, this is energy per **batch iteration**, not per image |
| `energy_J_integrated` | Trapezoidal sum `sum_j 0.5(P_j + P_(j+1))(t_(j+1) - t_j)` over the power-CSV elapsed axis |
| `energy_J_integrated_log_timestamps` | Trapezoidal integration using the raw timestamped telemetry log; this is the basis for `E_log` |

All 558 reconstructions agree with the metadata, manifest, and enriched-manifest values to the expected numerical precision.

The archived upper-central p50 differs numerically from the conventional median in 284/558 runs, but the largest difference is only 0.05704 ms. The distinction is definitional and not material for the reported conclusions.

Every run contains at least one 2 s rather than 1 s telemetry step. The `elapsed_s` field nevertheless reproduces the actual log timestamps, so trapezoidal integration accounts for those intervals. The time-weighted and simple arithmetic mean powers differ by at most 0.1818% and by 0.0095% at the median.

## 5. Historical boundary inconsistency and its resolution

The original analysis mixed incompatible time intervals. `build_enriched_manifest.py` defined the stored energy per image as:

`avg_power_W × meta.duration_s / N`,

whereas text generated by `make_best_of.py` described energy per image as:

`avg_power_W / forward_throughput_img_s`.

Their ratio is `meta.duration_s / forward_time_s`. Across 558 runs, the median ratio is 1.934, the 95th percentile is 5.582, and the maximum is 6.879.

| Model | Precision | Median historical duration energy / power–forward-throughput proxy |
|---|---|---:|
| MobileNetV2 | FP16 | 4.911 |
| MobileNetV2 | FP32 | 3.358 |
| MobileNetV2 | FP64 | 1.289 |
| ResNet-50 | FP16 | 3.051 |
| ResNet-50 | FP32 | 1.910 |
| ResNet-50 | FP64 | 1.032 |

The earlier Plotcode therefore produced an `Energy per Image` panel using the duration-based field while its `Images per Joule` panel recomputed `throughput/avg_power` using forward throughput. Those displayed quantities were not reciprocals. The same mismatch propagated into `make_best_of.py`, and the earlier EDP scripts multiplied one boundary's energy by another boundary's latency.

The raw power-log span closely matches the outer runner duration: the median state/log ratio is 1.0015–1.0025 by campaign. Relative to `meta.duration_s`, however, the power log is 1.3–4.2% longer at the campaign medians and up to 12.56% longer. Consequently, timestamp-integrated log energy is globally 2.03% higher at the median, 8.81% higher at the 95th percentile, and 12.54% higher at the maximum than `avg_power_W × meta.duration_s`.

The revision resolves the mismatch as follows:

- forward throughput and forward-latency distributions remain forward-window metrics;
- energy per evaluated image uses timestamp-integrated `E_log/N`;
- the energy-compatible throughput is `N/logged_window_duration_s`;
- EDP, when shown, is `E_log × logged_window_duration_s` and is explicitly secondary;
- no inference-only energy is claimed, because the telemetry log has no synchronized phase markers that could isolate model-forward intervals.

The logged interval includes warm-up, setup, input-pipeline work, transfers, accuracy computation, per-batch bookkeeping, and any wrapper overhead between its first and last samples. It is an **instrumented finite-benchmark** boundary, not a model-only energy boundary and not a complete deployed-service boundary.

## 6. Ordered and randomized March campaigns

The recorded orders are internally correct:

- the ordered campaign follows the unshuffled loop order and contains two requested-profile transitions;
- the randomized campaign exactly matches `random.Random(123).shuffle` and contains 33 profile transitions;
- the Spearman correlation between randomized position and the base ordered position is −0.0950.

These campaigns do not constitute an isolated order experiment:

- the modal EMC frequency is predominantly 2133 MHz in the ordered campaign and 3199 MHz in all 54 randomized runs;
- all 18 ordered requested-fast runs have modal CPU/GPU clocks of 1190/815 MHz; the randomized requested-fast runs comprise 11 at 729/407 MHz, 3 at 1190/815 MHz, and 4 at higher regimes (CPU at least 1267 MHz and GPU at least 815 MHz);
- the median maximum GPU/TJ temperature is 49.734 °C in the ordered campaign and 41.203 °C in the randomized campaign. The paired random-minus-ordered mean difference is −6.634 °C (95% CI −8.012 to −5.257 °C), with a median of −9.016 °C.

The following paired random/ordered results are retained as descriptive campaign diagnostics. The energy rows in the original audit table included both the historical duration-based energy and the corrected log-integrated energy; they were numerically similar for this particular contrast, but only the latter has the manuscript's logged-window boundary.

| Metric | Geometric mean ratio | 95% CI | Holm-adjusted Wilcoxon p |
|---|---:|---:|---:|
| Forward throughput | 0.946 | 0.877–1.019 | 0.0123 |
| Mean forward latency | 1.058 | 0.981–1.140 | 0.501 |
| Arithmetic mean power | 1.153 | 1.118–1.189 | 6.01 × 10^-8 |
| Historical duration-based energy/image | 1.221 | 1.173–1.270 | 1.14 × 10^-19 |
| Logged-window energy/image | 1.222 | 1.174–1.271 | 1.14 × 10^-19 |

The forward-throughput ratio is heterogeneous by requested profile: 0.755 (95% CI 0.624–0.914) for requested fast, 1.087 (1.055–1.121) for requested medium, and 1.029 (1.012–1.046) for requested slow. After Holm adjustment across 24 order-association tests, no association between run position and a performance, power, or energy metric remains significant; only temperature associations remain. Clock state, EMC, temperature, date, and sequence differ, so none of these contrasts identifies a causal randomization or order effect.

## 7. Board blocks, identifiability, and realized profile state

### 7.1 Provenance limits

- All 558 metadata files identify only `cuda_device: "Orin"`; none contains a module serial number, UUID, hostname, JetPack/L4T release, or board identifier.
- `git_commit` and `git_status` are `null` in all 558 metadata files.
- Folder and randomization seed are perfectly associated: MasterJetson uses 123, Clone1 uses 42, Clone2 uses 789, and Clone3 uses 456.
- MasterJetson has modal EMC 3199 MHz, whereas all three primary Clone folders have modal EMC 2133 MHz.

The authors confirm that `Clone1--3` represent different physical boards, but their identities are not independently authenticated by archived serials. Board-blocked intervals describe variation among the three archived physical-board means under their associated seeds and sequences; they are not population intervals for arbitrary Jetson modules and do not isolate silicon variation.

### 7.2 Repeatability by requested profile

Across the three primary board blocks and 42 configurations per requested profile:

| Requested profile | Median CV, forward throughput | Median CV, power | Median CV, logged energy/image |
|---|---:|---:|---:|
| Slow | 0.057% | 3.881% | 3.872% |
| Medium | 0.500% | 3.919% | 4.084% |
| Fast | 28.690% | 12.139% | 14.857% |

The requested-fast state is operationally bimodal. The 126 fast runs comprise 57 low-clock runs with modal GPU clock 407 MHz and 69 high-clock runs with modal GPU clock 812–815 MHz. Counts by board folder are Clone1 19/23, Clone2 23/19, and Clone3 15/27 for low/high.

### 7.3 Deterministic state carry-over in the archived sequence

The bimodality is fully associated with the preceding realized sequence:

| Immediately preceding state | Current requested-fast outcome |
|---|---:|
| Requested slow | 42/42 low (407 MHz) |
| Requested medium | 41/41 high (812–815 MHz) |
| Requested fast, previously low | 15/15 remains low |
| Requested fast, previously high | 28/28 remains high |

Equivalently, using the last non-fast requested profile, all 57 runs following slow are low and all 69 runs following medium are high. There is no exception in the 126-run fast subset. This is strong evidence that requested fast was not an independent, idempotently realized treatment. It is consistent with hardware/software state carry-over, but the archive cannot identify the mechanism because the exact executed July runner and complete environment provenance are unavailable. Accordingly, the revision treats requested-fast effects as descriptive and reports a sensitivity analysis excluding fast.

The sequence reconstruction uses the archived `grid_state.jsonl` `running` events as the primary start-order source. Each primary board folder contains exactly 126 `running` and 126 `done` events, with no retry, duplicate final record, failure, or timeout. Metadata timestamps provide a concordant secondary ordering.

### 7.4 Four-folder overlap

For the 72 configurations common to MasterJetson and Clone1–3, median forward-throughput CV is 0.843% for slow, 1.604% for medium, and 26.902% for fast. Corresponding power CVs are 8.783%, 8.970%, and 12.576%. The MasterJetson folder increases power dispersion under its different EMC and campaign conditions.

Friedman diagnostics over 72 matched configuration blocks give Kendall's W = 0.291 for throughput/latency (Holm p = 2.75 × 10^-13), W = 0.581 for power (1.48 × 10^-26), and W = 0.716 for the historical duration-based energy field (1.52 × 10^-32). These are descriptive folder/campaign rank effects, not isolated hardware effects. Historical Master/Clone geometric mean ratios are retained in `four_device_pairwise_statistics.csv`; they must not be used as evidence of silicon-to-silicon differences.

For completeness, the historical Master/Clone ratios over those 72 matched configurations were:

| Historical comparison | Forward throughput | Power | Historical meta-duration energy/image |
|---|---:|---:|---:|
| Master/Clone1 | 0.990 [0.941, 1.042] | 1.185 [1.159, 1.211] | 1.198 [1.162, 1.235] |
| Master/Clone2 | 1.018 [0.962, 1.077] | 1.123 [1.099, 1.148] | 1.110 [1.074, 1.148] |
| Master/Clone3 | 0.979 [0.933, 1.028] | 1.175 [1.153, 1.198] | 1.202 [1.167, 1.238] |

The energy column is retained only as an audit reconstruction of the earlier meta-duration definition. It is not a revised-manuscript estimand.

## 8. Replicated primary findings and robustness analyses

The experimental unit is one run. Contrasts first average matched configuration ratios within each physical-board block and then summarize the three board-block estimates. Their intervals describe those three archived boards; they do not represent repeated runs within a cell or a hardware population.

### 8.1 Precision contrasts using compatible boundaries

| Model | Contrast | Forward-throughput GMR [95% interval] | Logged-energy/image GMR [95% interval] |
|---|---|---:|---:|
| MobileNetV2 | FP32/FP16 | 0.694 [0.655, 0.736] | 1.196 [1.156, 1.238] |
| ResNet-50 | FP32/FP16 | 0.519 [0.492, 0.547] | 1.625 [1.576, 1.675] |
| MobileNetV2 | FP64/FP32 | 0.136 [0.125, 0.148] | 2.996 [2.788, 3.219] |
| ResNet-50 | FP64/FP32 | 0.0330 [0.0272, 0.0401] | 12.41 [11.28, 13.66] |

FP32 and FP64 have identical archived Top-1 values within each model: 0.72140 for MobileNetV2 and 0.80724 for ResNet-50. FP16 ranges from 0.72106 to 0.72116 and from 0.80722 to 0.80736, respectively. The largest FP16 deviation from FP32 is 0.034 percentage points for MobileNetV2 and 0.012 percentage points for ResNet-50. These controls apply only to the tested pretrained weights and preprocessing.

### 8.2 Batch-size 128 versus batch-size 1

| Model | Precision | Forward-throughput GMR [95% interval] | Logged-energy/image GMR [95% interval] |
|---|---|---:|---:|
| MobileNetV2 | FP16 | 10.684 [8.396, 13.595] | 0.424 [0.377, 0.478] |
| MobileNetV2 | FP32 | 4.994 [3.509, 7.107] | 0.550 [0.450, 0.672] |
| MobileNetV2 | FP64 | 0.934 [0.826, 1.057] | 1.105 [0.971, 1.257] |
| ResNet-50 | FP16 | 6.323 [4.958, 8.064] | 0.452 [0.385, 0.531] |
| ResNet-50 | FP32 | 2.111 [1.377, 3.236] | 0.670 [0.581, 0.773] |
| ResNet-50 | FP64 | 1.260 [0.964, 1.648] | 0.991 [0.834, 1.178] |

Large batches strongly improve FP16 and FP32 throughput and amortize the logged finite-benchmark energy. FP64 does not show a general batching benefit. These energy ratios include the full logged-window protocol and are not isolated model-forward energy effects.

### 8.3 FP64 leverage and FP16/FP32-only sensitivity

The full three-precision sum-of-squares partition is dominated by the extreme FP64 stress condition: precision accounts for 71.55% of log forward-throughput, 66.41% of log logged-energy/image, and 67.56% of log logged-window-throughput variation. Those values describe the tested FP16/FP32/FP64 grid, not the practical FP16/FP32 subspace.

Restricting the same balanced decomposition to FP16 and FP32 changes the interpretation:

| Outcome | Precision share | Batch share | Model share |
|---|---:|---:|---:|
| Log forward throughput | 11.60% | 50.99% | 20.66% |
| Log logged-energy/image | 17.76% | 33.18% | 33.82% |
| Log logged-window throughput | 6.22% | 47.52% | 12.46% |

Thus precision is the largest full-grid factor only when the FP64 stress condition is included. In the deployment-relevant FP16/FP32 subset, batch size dominates throughput variation, while model and batch are at least as important as precision for logged energy.

Excluding the non-reproducible fast requested profile does not remove the paired FP16/FP32 effects. The forward-throughput FP32/FP16 GMR is 0.684 for MobileNetV2 and 0.548 for ResNet-50; the logged-energy FP32/FP16 GMR is 1.212 and 1.612, respectively.

### 8.4 Precision-by-batch crossover

FP16 is not faster in every matched cell. In 19/126 board × model × profile × batch cells, its forward-throughput point estimate is no greater than FP32; in 12/126 cells, its logged-energy point estimate is no lower. At batch size 1, the board/profile geometric mean FP16/FP32 forward-throughput ratio is 0.931 for MobileNetV2 and 0.845 for ResNet-50. The reduced-precision advantage therefore appears only after sufficient work is amortized per forward call in this software stack. Aggregate statements must say *on average across the tested grid*, not *consistently in every configuration*.

### 8.5 Lowest observed means within the tested grid

Using the corrected logged-window energy, the lowest three-board mean in each model–precision cell is:

| Model/precision | Requested profile and batch | Forward throughput, mean ± SD (image/s) | Logged-window throughput, mean ± SD (image/s) | Logged energy, mean ± SD |
|---|---|---:|---:|---:|
| MobileNetV2 FP16 | medium, 64 | 336.75 ± 1.68 | 61.08 ± 0.71 | 99.43 ± 2.77 mJ/image |
| MobileNetV2 FP32 | medium, 32 | 175.54 ± 0.98 | 51.37 ± 2.08 | 131.89 ± 8.48 mJ/image |
| MobileNetV2 FP64 | medium, 32 | 23.33 ± 0.03 | 17.02 ± 0.28 | 376.76 ± 11.11 mJ/image |
| ResNet-50 FP16 | medium, 128 | 182.06 ± 0.70 | 50.99 ± 0.32 | 143.08 ± 5.86 mJ/image |
| ResNet-50 FP32 | medium, 64 | 79.06 ± 0.25 | 36.29 ± 0.31 | 245.40 ± 9.91 mJ/image |
| ResNet-50 FP64 | medium, 16 | 2.434 ± 0.007 | 2.316 ± 0.013 | 2.713 ± 0.120 J/image |

These are lowest observed means within the tested grid, not global optima. Requested-fast cell means are especially unsuitable for optimization claims because their realized state depends on sequence.

The boundary choice changes which configuration has the lowest mean in two of six model–precision cells. The historical `avg_power × meta.duration` field selected MobileNetV2 FP32 requested fast/batch 64 and ResNet-50 FP32 requested medium/batch 128, whereas the corrected timestamp-integrated logged-window field selects requested medium/batch 32 and requested medium/batch 64, respectively. This is why no energy-optimal configuration is asserted without naming the boundary.

## 9. Protocol-level batch-size caveat

The archived metadata record a warm-up value of 50. If the inspected reference implementation's full-batch warm-up behavior matches the executed code, this corresponds to 50 additional images at batch size 1 and 6,400 at batch size 128. Under that provenance-qualified interpretation, those images contribute to the logged energy but are excluded from the primary 50,000-image denominator; the corresponding alternative processed-image denominator is provided in `measurement_boundary_metrics.csv`.

The referenced public benchmark implementation also calls `csv_file.flush()` once per evaluated batch. If that reference code is confirmed as the executed July benchmark, this implies 50,000 flush calls at batch size 1 versus 391 at batch size 128. Logged-window batch effects would then combine GPU batching, pipeline amortization, warm-up work, and batch-dependent logging I/O. Because the metadata do not identify the executed commit and the exact seven-batch July runner is missing, this code-level mechanism is a provenance-qualified inference, not a proven property of the executed runs. A short control experiment with buffered or disabled per-batch logging is required to isolate it.

## 10. Tail latency and within-run dependence

Fifteen of 558 runs contain 28 iterations with latency greater than five times the archived upper-central p50. Twelve of those runs are FP64/batch-128, two are FP16/batch-32, and one is FP32/batch-32. Batch latencies are temporally correlated, with median lag-1 autocorrelation between 0.078 and 0.237 by campaign; telemetry power has median lag-1 autocorrelation between 0.154 and 0.545. Neither batch rows nor power samples are independent experimental replicates.

The largest throughput sensitivity occurs in `Ergebnisse_Clone2_13.07.2026/mobilenet_v2_fp64_fast_BS128_perf.csv`: six batches exceed 5 × p50 and the maximum batch latency is 288.101 s. The reported throughput is 9.272 image/s; diagnostic exclusion of those six rows would give 10.815 image/s (+16.64%). Other large cases are:

- Clone3 ResNet-50 FP64 requested fast, batch 128: maximum 745.586 s; +8.25% diagnostic sensitivity;
- ordered campaign ResNet-50 FP64 requested fast, batch 128: maximum 639.061 s; +5.56%;
- ordered campaign MobileNetV2 FP64 requested medium, batch 128: maximum 126.119 s; +4.94%.

No observation was removed. These stalls may be genuine tail behavior. The audit reports p50, p95, p99, maximum, and the non-destructive >5 × p50 sensitivity. The 95th percentile sensitivity is zero or below 0.18% in every campaign except the ordered series; the issue is concentrated in a small number of FP64/batch-128 runs.

## 11. Code and environment provenance audit

All 13 Python files in the archived Plotcode are syntactically valid, and the workspace JSON is valid. The following limitations remain:

1. The earlier energy and images-per-joule code used incompatible boundaries (Section 5).
2. The earlier EDP calculation mixed duration-based energy with forward latency.
3. `energy_per_inference_J` is energy per batch iteration, despite its name.
4. The old plotting dictionaries explicitly distinguish only batches 32, 64, and 128. Batches 1, 4, 8, and 16 would share the default marker and are absent from those legends.
5. The archived runner `host_run_grid_unattended_v6.py` supports only batches 32, 64, and 128. It cannot reproduce the seven-batch Clone grid or the four-batch MasterJetson grid; the exact executed July runner is absent.
6. `git_commit` and `git_status` are null in every run metadata file, and the executed `bench_classifier_repro.py` was not included in the original experiment archive.
7. JetPack, L4T, Ubuntu, CUDA toolkit, cuDNN, container digest, exact Torchvision weight enums, host identifiers, and board serials are not fully captured in the run metadata. Torch and Torchvision version strings are present.
8. The archived runner computes an unweighted power-sample mean. The empirical difference from `integral/span` is small here (maximum 0.182%) but timestamp weighting is the principled estimator.
9. The two EDP plotting files named in Section 3 are byte-identical.

The public repository commit cited in the manuscript is an immutable reference implementation inspected after the experiment. The archive does not prove that this exact commit produced the July runs. Claims about the executed code must therefore be labelled **archived**, **author-reported**, or **reference-implementation inference**, as appropriate.

## 12. Statistical interpretation and claim limits

- The experimental unit is the run, not an iteration or telemetry sample.
- Primary precision and batch contrasts are matched within configuration and summarized through board-block means. With three physical-board blocks, intervals show dispersion across these archived boards and not a population sample of arbitrary devices.
- The four-factor full-grid sum-of-squares decomposition is descriptive. Main-effect shares must be interpreted together with interactions and with the FP16/FP32-only sensitivity analysis.
- The requested-fast level is a sequence-dependent realized treatment. Primary conclusions that depend on precision or batch are accompanied by an analysis excluding fast; no causal profile effect is claimed.
- The March order campaigns are descriptive historical sensitivity evidence because time, EMC, temperature, and realized clocks differ.
- Four-folder Friedman and pairwise results are archive-folder/campaign diagnostics, not silicon estimates.
- Energy and EDP must always be paired with the logged-window duration. Inference-only energy would require new measurements with synchronized phase markers or higher-resolution segmentable power data.
- Internal `VDD_IN` telemetry was not calibrated against an external power analyzer. Absolute joule values are internally computed benchmark measurements, not metrology-grade certification.
- The three primary campaigns shared an author-reported air-conditioned 24 °C room condition; no run-synchronized ambient-temperature series was archived.

## 13. Machine-readable evidence map

The complete item map and column definitions are provided in `../DATA_DICTIONARY.md`. The most important files are:

- `../../generated/measurement_boundary_metrics.csv`: compatible forward and logged-window metrics for all 558 runs;
- `../../generated/precision_device_block_contrasts.csv` and `batch_device_block_contrasts.csv`: primary board-blocked effect estimates;
- `../../generated/factorial_anova_results.csv` and `anova_variance_groups.csv`: full-grid decomposition;
- the FP16/FP32 sensitivity and carry-over CSVs listed in the data dictionary;
- `run_level_audit_and_recomputed_metrics.csv`: full 558-run reconstruction and integrity diagnostics;
- `grid_completeness.csv`, `grid_state_summary.csv`, and `run_order_summary.csv`: run completeness and sequence;
- `LONG_RUN_DRIFT_ANALYSIS.md` and `long_run_*.csv`: conservative inter-run-gap, successor, boundary-temperature, and temporal-drift diagnostics for Reviewer 2 Comment 7;
- `energy_definition_boundary_summary.csv` and `state_power_meta_timing_summary.csv`: historical boundary diagnostics;
- `within_run_variability_and_stall_summary.csv` and `largest_latency_stall_sensitivity.csv`: tail-latency diagnostics;
- `file_inventory_sha256.csv` and `duplicate_file_content.csv`: complete file inventory.
