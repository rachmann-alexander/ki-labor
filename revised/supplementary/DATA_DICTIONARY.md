# Supplementary Data Dictionary

**Version:** 13 July 2026  
**Encoding:** UTF-8  
**Delimiter:** Comma  
**Decimal mark:** Period  
**Missing values:** Empty CSV field unless stated otherwise

This dictionary assigns stable supplementary-item labels to the narrative report and machine-readable evidence. File names are retained so that the analysis scripts remain reproducible. All paths are relative to `revised/supplementary/`.

## 1. Supplementary-item index

| Item | Submission label | Files | Purpose |
|---|---|---|---|
| **S1** | Complete archive and analysis audit | `audit/AUDIT_REPORT.md` | Read-only narrative audit, estimand definitions, integrity findings, boundary correction, provenance limits, and claim constraints. |
| **S2** | Run-level measurement-boundary table | `../generated/measurement_boundary_metrics.csv` | Compatible forward-window and logged-window metrics for all 558 runs. This is the primary per-run boundary table. |
| **S3** | Full-grid and FP16/FP32 factorial decomposition | `../generated/factorial_anova_results.csv`; `../generated/anova_variance_groups.csv`; `../generated/anova_variance_plot.csv`; `../generated/factorial_anova_results_fp16_fp32.csv`; `../generated/factorial_anova_results_fp16_fp32_excluding_fast.csv`; `../generated/anova_variance_sensitivity.csv` | Log-scale sums of squares for the 378-run primary grid, the 252-run FP16/FP32 grid, and the 168-run FP16/FP32 slow+medium sensitivity grid. |
| **S4** | Board-blocked precision and batch contrasts | `../generated/precision_device_block_contrasts.csv`; `../generated/batch_device_block_contrasts.csv`; `../generated/precision_device_block_contrasts_excluding_fast.csv`; `../generated/batch_device_block_contrasts_excluding_fast.csv` | Matched geometric mean ratios and intervals across three author-confirmed physical-board blocks, with and without the sequence-dependent requested-fast level. |
| **S5** | Requested-fast realized-state fidelity and carry-over | `../generated/fast_clock_regime_counts.csv`; `../generated/profile_repeatability.csv`; `../generated/fast_state_carryover_runs.csv`; `../generated/fast_state_carryover_immediate_predecessor.csv`; `../generated/fast_state_carryover_last_nonfast.csv`; `../generated/fast_state_carryover_configuration_consistency.csv`; `../generated/state_carryover_macros.tex` | Per-run realized clock regime, exact transition summaries, and cross-block consistency of identical workload cells showing sequence-dependent state carry-over. |
| **S6** | Precision-by-batch crossover and protocol workload | `../generated/precision_batch_crossover_cells.csv`; `../generated/precision_batch_crossover_summary.csv`; `../generated/precision_batch_crossover_excluding_fast_cells.csv`; `../generated/precision_batch_crossover_excluding_fast_summary.csv`; `../generated/protocol_batch_dependent_work_summary.csv` | Cell-level FP16/FP32 comparisons, batch-level board-block summaries with and without requested fast, and batch-dependent iteration/warm-up/reference-logging work. |
| **S7** | Primary configuration summaries and figure data | `../generated/clone_configuration_summary.csv`; `../generated/clone_best_configurations.csv`; all `../generated/batch_throughput_*.csv`; all `../generated/batch_task_energy_*.csv`; `../generated/representative_cpu_gpu_temperature_trace.csv` | Three-board means/SDs, lowest observed means, and manuscript-figure source data, including the representative CPU/GPU-frequency and temperature traces. |
| **S8** | Campaign and run accounting | `../generated/campaign_summary.csv`; `audit/grid_completeness.csv`; `audit/grid_state_summary.csv`; `audit/run_order_summary.csv`; `audit/run_order_positions.csv` | Run counts, complete Cartesian grids, state completion, and realized run sequences. |
| **S9** | Utilization and tail-latency diagnostics | `../generated/run_utilization_summary.csv`; `../generated/utilization_by_model_precision.csv`; `../generated/tail_latency_summary.csv`; `../generated/tail_latency_largest_cases.csv`; `audit/within_run_variability_and_stall_summary.csv`; `audit/largest_latency_stall_sensitivity.csv` | Coarse telemetry utilization, forward-time fractions, latency percentiles, and non-destructive stall sensitivity. |
| **S10** | Complete 558-run reconstruction | `audit/run_level_audit_and_recomputed_metrics.csv` | Full run-level formulas, integrity flags, latency distributions, energy alternatives, timing, clocks, and temperature. |
| **S11** | File, schema, and manifest integrity | `audit/file_inventory_sha256.csv`; `audit/file_count_summary.csv`; `audit/duplicate_file_content.csv`; `audit/dataset_quality_summary.csv`; `audit/run_file_schema_by_file.csv`; `audit/run_file_schema_summary.csv`; `audit/manifest_schema_and_path_audit.csv`; `audit/grid_state_events.csv`; `audit/issues.csv` | Complete SHA-256 inventory and structural validation. An empty `issues.csv` contains no detected issue row. |
| **S12** | Historical-boundary, sequence, and campaign diagnostics | `audit/energy_definition_boundary_summary.csv`; `audit/state_power_meta_timing_alignment.csv`; `audit/state_power_meta_timing_summary.csv`; `audit/clone_config_means_sd.csv`; `audit/clone_best_mean_configurations.csv`; `audit/clone_factor_contrasts.csv`; `audit/fixed_vs_random_matched_runs.csv`; `audit/fixed_vs_random_paired_statistics.csv`; `audit/fixed_vs_random_order_association.csv`; `audit/fixed_vs_random_temperature_difference.csv`; `audit/four_device_friedman_tests.csv`; `audit/four_device_pairwise_statistics.csv`; `audit/four_device_variability_by_configuration.csv`; `audit/four_device_variability_by_profile_summary.csv`; `audit/four_device_variability_summary.csv`; `audit/three_clone_variability_by_profile_summary.csv`; `audit/clock_signature_by_dataset_profile.csv`; `audit/clock_signature_counts.csv`; `audit/plotcode_syntax_and_feature_audit.csv`; `audit/LONG_RUN_DRIFT_ANALYSIS.md`; all `audit/long_run_*.csv`; `audit/long_run_drift_results.tex` | Audit diagnostics retained for transparency, including conservative existing-data long-run, successor, and drift checks. Duration-based historical fields are not primary revised-manuscript estimands. |

## 2. Global conventions

### 2.1 Identifiers

| Column or token | Meaning |
|---|---|
| `dataset` | Archive campaign/folder identifier: `fixed_2026-03`, `random_2026-03`, `Clone1`, `Clone2`, `Clone3`, or `MasterJetson`. |
| `device_block` | Primary physical-board block. The authors confirm that `Clone1--3` are distinct boards, but serial numbers were not archived and the identity cannot be authenticated from the artifacts alone. |
| `prefix` | Unique run stem encoding model, precision, requested profile, and batch size within a dataset. |
| `model` | `mobilenet_v2` or `resnet50`. |
| `precision` | Requested PyTorch dtype: `fp16`, `fp32`, or `fp64`. |
| `profile` | Requested script profile: `slow`, `medium`, or `fast`. This is not a verified DVFS treatment. |
| `batch` | Requested evaluated batch size. The final evaluated batch may be partial. |
| `seed` | Archived campaign permutation seed; it is associated with dataset/board folder. |

### 2.2 Statistics and units

| Suffix or unit | Meaning |
|---|---|
| `_J`, `_W`, `_s`, `_ms`, `_MHz`, `_C` | Joules, watts, seconds, milliseconds, megahertz, and degrees Celsius. |
| `_img_s` | Evaluated images per second. |
| `_mean`, `_sd`, `_min`, `_max` | Arithmetic mean, sample standard deviation (`ddof=1` where applicable), minimum, and maximum. |
| `_p50`, `_p95`, `_p99` | 50th, 95th, and 99th percentile. The archived `latency_p50_ms` is the upper central value; the audit also provides a conventional median. |
| `n`, `n_runs`, `n_devices` | Number of contributing observations, runs, or physical-board blocks. `n_devices=3` is retained as a legacy column name and denotes the three distinct `Clone1--3` boards. |
| `geometric_mean_ratio` or `geomean_ratio` | Exponentiated mean of matched log ratios. |
| `ci95_*_device_block` | t interval over three board-block log-ratio averages. It is not a population interval for arbitrary Jetson devices. |
| `variance_share` | Term sum of squares divided by total corrected sum of squares on the natural-log outcome scale. |
| Boolean `True`/`False` | Whether the explicitly named audit rule is satisfied. |

### 2.3 Measurement-boundary aliases

Some file and column names predate the manuscript's final terminology.

| Legacy machine name | Interpret as |
|---|---|
| `task_time_s` | Logged-window duration. |
| `task_throughput_img_s` | Logged-window throughput, `50,000 / task_time_s`. |
| `task_energy_J` | Timestamp-integrated `VDD_IN` energy over the same logged window. |
| `task_energy_per_eval_image_J` | Logged-window energy divided by 50,000 evaluated images. |
| `task_energy_per_processed_image_J` | Provenance-qualified sensitivity field: logged-window energy divided by evaluated images plus conditionally inferred warm-up images, if the inspected reference implementation's full-batch warm-up behavior matches the executed code. |
| `task_edp_Js` | Logged-window energy × logged-window duration. |
| `whole_task_*` | Legacy header text for the same logged-window quantity in a small number of generated descriptive files. |
| `throughput_img_s` | Forward-only throughput based on synchronized per-batch model-forward latencies. |

No `task_*` or `whole_task_*` field should be described as inference-only energy or full deployed-service performance.

## 3. S2 — `measurement_boundary_metrics.csv`

One row represents one archived run.

| Column | Definition |
|---|---|
| `dataset`, `prefix`, `model`, `precision`, `batch`, `profile` | Run identifiers defined above. |
| `total_images` | Evaluated validation images; 50,000 in every run. |
| `warmup_images` | Conditional `50 × requested batch size`; the metadata record 50 warm-up batches, while the image count requires the unverified assumption that the inspected reference implementation's full-batch warm-up behavior matches the executed code. |
| `throughput_img_s` | Forward-only throughput from summed synchronized model-forward latencies. |
| `task_time_s` | First-to-last valid telemetry timestamp span; logged-window duration. |
| `task_throughput_img_s` | `total_images / task_time_s`. |
| `task_energy_J` | Trapezoidal integral of timestamped `VDD_IN` over `task_time_s`. |
| `task_energy_per_eval_image_J` | `task_energy_J / total_images`; primary energy-normalization field. |
| `task_energy_per_processed_image_J` | `task_energy_J / (total_images + warmup_images)`; alternative protocol denominator. |
| `task_edp_Js` | `task_energy_J × task_time_s`; secondary boundary-compatible EDP. |

## 4. S3 — factorial decompositions

### 4.1 Long-format result files

`factorial_anova_results.csv` contains the full FP16/FP32/FP64 grid; `factorial_anova_results_fp16_fp32.csv` contains the sensitivity grid without FP64; and `factorial_anova_results_fp16_fp32_excluding_fast.csv` contains the FP16/FP32 sensitivity restricted to the repeatable slow and medium branches.

| Column | Definition |
|---|---|
| `outcome` | Positive response analyzed after natural-log transformation. |
| `term` | Board block, main effect, interaction, or residual term. |
| `order` | Interaction order; board and residual terms use their scripted category. |
| `df` | Degrees of freedom. |
| `ss`, `ms` | Corrected sum of squares and mean square. |
| `f`, `p` | Classical F statistic and p value; included for completeness, not used as the primary evidential basis with three board blocks. |
| `variance_share` | `ss / total corrected ss`. |
| `partial_eta_squared`, `omega_squared` | Conventional effect-size diagnostics on the modeled log scale. |
| `scale` | Always `natural logarithm`. |

### 4.2 Wide-format sensitivity file

`anova_variance_sensitivity.csv` contains one row per analysis grid and outcome.

| Column | Definition |
|---|---|
| `analysis_grid` | `full_fp16_fp32_fp64` (378 runs), `practical_fp16_fp32` (252 runs), or `practical_fp16_fp32_slow_medium` (168 runs). |
| `precision_levels` | Included requested dtype levels. |
| `n_runs` | Number of primary-grid runs in the decomposition. |
| `outcome` | Modeled positive response. |
| `Model`, `Precision`, `Batch`, `Profile` | Main-effect percentage of corrected total sum of squares. |
| `Two-way interactions` | Sum of all second-order interaction shares, in percent. |
| `Higher-order interactions` | Sum of all third- and fourth-order interaction shares, in percent. |
| `Device block` | Legacy header for the physical-board block share, in percent. |
| `Residual` | Board-by-configuration residual share, in percent. |

`anova_variance_groups.csv` and `anova_variance_plot.csv` contain equivalent full-grid grouped values formatted for the manuscript table and TikZ plot.

## 5. S4 — board-blocked contrast files

The four files use the same core schema. Rows are separate outcomes.

| Column | Definition |
|---|---|
| `factor` | `precision` or `batch`. |
| `included_profiles` | Present only in `*_excluding_fast.csv`; `slow|medium`. |
| `comparison` | Numerator/denominator, e.g. `fp32/fp16` or `128/1`. |
| `model`, `precision` | Analysis stratum; `precision` is present for batch contrasts. |
| `metric` | Forward throughput, logged-window energy/image, logged-window throughput, or timestamp-weighted mean power. |
| `n_devices` | Three author-confirmed physical-board blocks; legacy name. |
| `matched_pairs_per_device` | Matched configuration ratios averaged within each board block. |
| `geometric_mean_ratio` | Exponentiated mean of the three board-block log-ratio means. |
| `ci95_low_device_block`, `ci95_high_device_block` | 95% t interval over the three board-block log-ratio means. |
| `sd_log_ratio_across_devices` | Sample SD of those three board-block log-ratio means. |
| `device_ratios` | Pipe-separated exponentiated board-block estimates; legacy name. |

## 6. S5 — realized-state and carry-over files

### 6.1 `fast_state_carryover_runs.csv`

One row represents one of the 126 requested-fast primary runs.

| Column | Definition |
|---|---|
| `device_block` | Clone archive folder. |
| `fast_sequence_index` | One-based position of the current requested-fast run in the board folder's archived `running`-event sequence. |
| `fast_start_timestamp_utc`, `fast_metadata_timestamp_utc`, `fast_prefix` | State-event start timestamp, metadata timestamp, and run identifier of the current requested-fast run. |
| `realized_fast_regime` | `low` for modal GPU 407 MHz; `high` for modal GPU 812–815 MHz. |
| `fast_cpu_mode_MHz`, `fast_gpu_mode_MHz` | Current run's modal observed CPU and GPU clocks. |
| `immediate_predecessor_sequence_index`, `immediate_predecessor_start_timestamp_utc`, `immediate_predecessor_prefix`, `immediate_predecessor_profile` | Immediately preceding archived `running` event in the same board-folder sequence. |
| `immediate_predecessor_fast_regime` | Realized regime if the predecessor was requested fast; otherwise `not_applicable`. |
| `immediate_predecessor_state` | `slow`, `medium`, `fast-low`, or `fast-high`. |
| `expected_regime_from_immediate_predecessor` | Low after slow/fast-low; high after medium/fast-high. |
| `immediate_rule_match` | Whether the current observed regime matches that deterministic transition rule. |
| `last_nonfast_sequence_index`, `last_nonfast_start_timestamp_utc`, `last_nonfast_prefix`, `last_nonfast_profile` | Most recent preceding slow or medium `running` event. |
| `expected_regime_from_last_nonfast` | Low after slow; high after medium. |
| `last_nonfast_rule_match` | Whether the current regime matches the last-non-fast rule. |

### 6.2 Transition summaries

`fast_state_carryover_immediate_predecessor.csv` and `fast_state_carryover_last_nonfast.csv` contain:

| Column | Definition |
|---|---|
| `immediate_predecessor_state` or `last_nonfast_profile` | Transition stratum. |
| `n_fast_runs` | Requested-fast runs in the stratum. |
| `observed_low`, `observed_high` | Counts by realized current regime. |
| `expected_regime` | Rule-predicted current regime. |
| `rule_matches` | Number agreeing with the rule. |
| `rule_accuracy_percent` | `100 × rule_matches / n_fast_runs`; 100% in every observed stratum. |

`fast_clock_regime_counts.csv` gives low/high counts per board folder. `profile_repeatability.csv` gives the median configuration-level coefficient of variation by requested profile and the archived temperature range.

## 7. S6 — crossover and protocol-work files

### 7.1 `precision_batch_crossover_cells.csv` and `precision_batch_crossover_excluding_fast_cells.csv`

One row is one matched `board folder × model × requested profile × batch` FP16/FP32 pair. The complete-grid file has 126 rows; the excluding-fast sensitivity file has 84 rows over slow and medium only.

| Column | Definition |
|---|---|
| Identifier columns | `dataset`, `model`, `batch`, and `profile`. |
| `*_fp16`, `*_fp32` | Cell values for logged energy/image, logged-window throughput, and forward throughput. |
| `forward_throughput_fp16_over_fp32` | FP16/FP32 forward-throughput ratio. |
| `logged_window_energy_fp16_over_fp32` | FP16/FP32 logged-energy ratio; values below 1 favor FP16. |
| `logged_window_throughput_fp16_over_fp32` | FP16/FP32 logged-window-throughput ratio. |
| `fp16_forward_faster` | `True` when the forward-throughput ratio is greater than 1. |
| `fp16_logged_window_energy_lower` | `True` when the energy ratio is below 1. |
| `fp16_logged_window_throughput_faster` | `True` when the logged-window-throughput ratio is greater than 1. |

### 7.2 `precision_batch_crossover_summary.csv` and `precision_batch_crossover_excluding_fast_summary.csv`

One row is one model × batch summary across three board blocks and either all three requested profiles or the repeatable slow and medium branches only.

| Column family | Definition |
|---|---|
| `n_matched_device_profile_cells`, `n_devices`, `n_profiles` | Contributing matched cells, physical-board blocks, and requested profiles. Legacy `device` names refer to the `Clone1--3` board folders. |
| `*_gmr` | FP16/FP32 geometric mean ratio after board-block averaging. |
| `*_ci95_low_device_block`, `*_ci95_high_device_block` | 95% interval over three board-block log-ratio means. |
| `*_sd_log_ratio_across_devices` | SD of the three board-block log-ratio means. |
| `*_favorable_count`, `*_not_favorable_count` | Number of matched board/profile cells favoring or not favoring FP16 for that outcome: nine in the complete-grid file and six in the slow+medium file. |

### 7.3 `protocol_batch_dependent_work_summary.csv`

| Column | Definition |
|---|---|
| `batch` | Requested batch size. |
| `n_primary_runs` | Runs at that batch size in the 378-run primary grid; 54. |
| `evaluated_images_per_run` | Always 50,000. |
| `timed_batch_iterations_per_run` | `ceil(50,000/batch)`. |
| `last_timed_batch_images` | Images in the final evaluated partial or complete batch. |
| `warmup_batches_from_metadata` | Archived warm-up count; 50. |
| `warmup_images_if_full_batch` | `50 × batch`; conditional on the inspected reference implementation matching executed behavior. |
| `images_processed_including_warmup` | Evaluated plus conditionally inferred warm-up images under the reference-code interpretation above. |
| `warmup_share_of_processed_images_percent` | Conditional warm-up image share under the reference-code interpretation above. |
| `reference_writer_flush_calls_if_one_per_timed_batch` | Conditional per-batch flush count from the inspected reference implementation. |
| `flush_evidence_scope`, `warmup_evidence_scope` | Textual provenance qualification. |
| `flush_calls_relative_to_batch128` | Conditional flush count divided by 391 at batch 128. |

## 8. S7–S9 — summaries used by tables and figures

### 8.1 Configuration summaries

`clone_configuration_summary.csv` has one row per model × precision × batch × requested profile. `n_devices` denotes the three distinct physical-board blocks. For each metric, `_mean` and `_sd` are the arithmetic mean and sample SD across those three blocks. `task_energy_mJ_*` and `task_throughput_*` are logged-window values despite their legacy prefixes. `max_temperature_C_mean` is the mean of per-run maximum GPU/TJ temperature.

`clone_best_configurations.csv` selects the highest or lowest observed three-board mean within each model–precision cell. `objective` states the selection rule; no row is a global optimum. Requested-fast selections inherit the carry-over limitation.

`batch_throughput_*.csv` and `batch_task_energy_*.csv` use:

- `batch`: requested batch size;
- `{slow|medium|fast}_mean`: three-board mean at that requested profile;
- `{slow|medium|fast}_sd`: corresponding sample SD.

Energy files report logged-window millijoules per evaluated image.

### 8.2 Utilization

`run_utilization_summary.csv` contains one row per run:

| Column | Definition |
|---|---|
| `samples` | Parsed telemetry samples. |
| `gpu_active_mean_percent`, `gpu_active_p95_percent` | Mean and 95th percentile GPU activity over the telemetry log. |
| `emc_bandwidth_mean_percent`, `emc_bandwidth_p95_percent` | Mean and 95th percentile EMC bandwidth-use field. |
| `cpu_mean_across_cores_percent`, `cpu_p95_across_cores_percent` | Per-sample mean CPU activity across reported cores, summarized over telemetry samples. |
| `fan_fields`, `throttle_fields` | Counts of parsed corresponding fields; zero means unavailable, not proof that no throttling occurred. |
| `forward_time_fraction_of_meta_duration` | Sum of forward latencies divided by archived benchmark `meta.duration_s`. This has a different boundary from the telemetry utilization columns and is labelled as such. |

`utilization_by_model_precision.csv` contains medians across the 63 primary runs in each model–precision stratum.

### 8.3 Tail latency

`tail_latency_summary.csv` gives archive-wide counts and maxima. `tail_latency_largest_cases.csv` and the two audit-level stall files provide p50, p95, p99, maximum latency, number of iterations above 5 × p50, reported forward throughput, and the diagnostic percentage increase obtained by excluding only those >5 × p50 rows. No row was excluded from manuscript estimates.

## 9. S10 — full run-level audit table

`audit/run_level_audit_and_recomputed_metrics.csv` contains one row per run and is intentionally wide. Column families are:

| Family | Representative columns | Meaning |
|---|---|---|
| Provenance | `dataset`, `source_directory`, `prefix`, factor columns, software versions, `timestamp_utc` | Archived identifiers and available environment metadata. |
| Performance integrity | `perf_rows_iterations`, NaN/nonfinite/duplicate counts, sequence and batch-pattern flags | Structural checks on `perf.csv`. |
| Latency | archived/conventional p50, p95, p99, maximum, >2×/>5×/>10× counts, SD, CV, lag-1 correlation | Run-level latency distribution and dependence. |
| Forward outcome | `throughput_img_s`, `top1_weighted`, `inference_time_s` | Synchronized forward-window results. |
| Archived benchmark duration | `meta_duration_s`, `inference_time_fraction_of_meta_duration` | Metadata boundary; not identical to the telemetry log span. |
| Power integrity | sample counts, NaN/nonfinite/duplicate/index/timestamp flags | Structural checks on telemetry CSVs. |
| Power summary | arithmetic and timestamp-weighted means, SD, CV, lag-1 correlation | `VDD_IN` power over the logged interval. |
| Historical energy alternatives | columns containing `avg_power_times_meta_duration`, `power_over_throughput_proxy`, or `reported_formula` | Retained only to audit the earlier boundary mismatch. |
| Logged-window energy | `energy_J_integrated_log_timestamps`, `energy_per_img_J_integrated_log_timestamps` | Timestamp-integrated primary energy source. |
| Raw-log validation | log-line, parse-error, gap, span, and raw-vs-CSV agreement fields | Timestamp and power-log integrity. |
| Realized state | modal/maximum CPU, EMC, and GPU clocks; start, end, time-weighted mean, and maximum GPU/TJ temperature | Read-back diagnostics, not requested settings. Start/end values are the first/last retained internal-telemetry observations, not ambient-temperature measurements. |
| Timing alignment | log/meta/state ratios and nominal-vs-timestamp energy ratio | Boundary-alignment diagnostics. |

## 10. S11–S12 — integrity and historical diagnostics

- `file_inventory_sha256.csv`: one row per audited source file with archive area, relative path, extension, byte count, and SHA-256 digest.
- `grid_completeness.csv`: expected run prefix, Cartesian-grid membership, manifest membership, available artifact types, and missing artifact types.
- `grid_state_events.csv`: every parsed state line with run identifier, status, attempt, timestamp, and runner duration.
- `manifest_schema_and_path_audit.csv`: header/schema consistency, duplicate checks, and absolute-path/basename availability.
- `clock_signature_counts.csv`: exact modal/maximum CPU, EMC, and GPU signature with run count; `clock_signature_by_dataset_profile.csv` aggregates those signatures.
- `fixed_vs_random_*`: matched March-campaign rows, log-ratio summaries, order-position associations, and temperature differences. These are campaign comparisons, not causal order estimates.
- `four_device_*`: four-folder overlap variability and rank diagnostics. These are folder/campaign effects, not serial-number-verified silicon effects.
- `energy_definition_boundary_summary.csv` and `state_power_meta_timing_*`: historical energy-definition ratios and timestamp-span alignment.
- `plotcode_syntax_and_feature_audit.csv`: Python syntax validation and detection of legacy energy/EDP/batch-marker behavior.
- `LONG_RUN_DRIFT_ANALYSIS.md`: conservative interpretation of archived inter-run gaps, successors of FP64/upper-duration-quartile runs, and temporal slopes with successor-configuration and physical-board fixed effects.
- `long_run_successor_pairs.csv`: all 552 adjacent archived run pairs, including predecessor duration/state, state-event gap, boundary temperatures, and primary adjusted residuals.
- `long_run_successor_binary_effects.csv`: successor contrasts for FP64 and upper-duration-quartile predecessors with successor-configuration and physical-board fixed effects plus leave-one-board-out sensitivity ranges; requested-fast successors are excluded.
- `long_run_successor_continuous_associations.csv`: pooled and board-specific descriptive Spearman associations after residualizing both the predecessor predictor and successor outcome for successor configuration and physical-board block; no p values are used.
- `long_run_successor_temperature_transitions.csv`: raw predecessor-end to successor-start internal-temperature continuity.
- `long_run_state_event_gaps.csv`: campaign-wise gaps from predecessor `done` to successor `running` state events.
- `long_run_temporal_drift_summary.csv`: primary temporal slopes with configuration and physical-board fixed effects, leave-one-board-out sensitivity ranges, and confounded historical matched diagnostics.
- `long_run_drift_results.tex`: generated descriptive-result macros for response/supplement use.

## 11. Reproduction and precedence

From the project root:

```text
python -m pip install -r revised/analysis/requirements.txt
python revised/analysis/full_audit.py
python revised/analysis/build_revision_results.py
python revised/analysis/long_run_drift_analysis.py
```

The recorded reproduction runtime is Python 3.11.9 with NumPy 2.0.2, pandas 2.2.3, and SciPy 1.14.1. The scripts read the original archives and overwrite only generated evidence files in `revised/`. They do not alter raw measurements.

If labels conflict, use this precedence order:

1. mathematical estimand definitions in Supplementary Item S1 and the revised manuscript;
2. this data dictionary's mapping of legacy column names;
3. machine-readable numerical values;
4. historical names embedded in the original scripts or CSV headers.
