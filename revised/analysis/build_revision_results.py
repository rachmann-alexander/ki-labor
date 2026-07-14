#!/usr/bin/env python3
"""Build the statistical tables and PGFPlots inputs used by the revision.

Run ``full_audit.py`` first.  This script consumes only the audited run-level
table and the archived raw tegrastats logs; it never modifies source data.
"""

from __future__ import annotations

import itertools
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
REV = Path(__file__).resolve().parents[1]
AUDIT = REV / "supplementary" / "audit"
OUT = REV / "generated"
OUT.mkdir(parents=True, exist_ok=True)

RUN_TABLE = AUDIT / "run_level_audit_and_recomputed_metrics.csv"
CLONE_DEVICES = ["Clone1", "Clone2", "Clone3"]
FACTORS = ["model", "precision", "batch", "profile"]
FACTOR_LABEL = {
    "model": "Model",
    "precision": "Precision",
    "batch": "Batch",
    "profile": "Profile",
}
LEVELS = {
    "model": ["mobilenet_v2", "resnet50"],
    "precision": ["fp16", "fp32", "fp64"],
    "batch": [1, 4, 8, 16, 32, 64, 128],
    "profile": ["slow", "medium", "fast"],
}

OUTCOMES = {
    "forward_throughput": "throughput_img_s",
    "task_energy_per_image": "task_energy_per_eval_image_J",
    "task_throughput": "task_throughput_img_s",
    "average_input_power": "time_weighted_avg_power_W",
}

LOG_TS_RE = re.compile(r"^(\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2})")
LOG_GPU_RE = re.compile(r"\bGR3D_FREQ\s+\d+%@\[([^\]]+)\]")
LOG_GPU_UTIL_RE = re.compile(r"\bGR3D_FREQ\s+(\d+(?:\.\d+)?)%@")
LOG_EMC_UTIL_RE = re.compile(r"\bEMC_FREQ\s+(\d+(?:\.\d+)?)%@")
LOG_CPU_BLOCK_RE = re.compile(r"\bCPU \[(.*?)\]\s+EMC_FREQ")
LOG_PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)%")
LOG_TEMP_RE = re.compile(r"\b(?:gpu|tj)@(-?\d+(?:\.\d+)?)C")
LOG_POWER_RE = re.compile(r"\bVDD_IN\s+(\d+(?:\.\d+)?)mW/")


def load_runs() -> pd.DataFrame:
    df = pd.read_csv(RUN_TABLE)
    df["task_energy_J"] = df["energy_J_integrated_log_timestamps"]
    df["task_energy_per_eval_image_J"] = df["task_energy_J"] / df["total_images"]
    df["task_time_s"] = df["log_wall_span_s"]
    df["task_throughput_img_s"] = df["total_images"] / df["task_time_s"]
    df["task_edp_Js"] = df["task_energy_J"] * df["task_time_s"]
    df["warmup_images"] = 50 * df["batch"]
    df["task_energy_per_processed_image_J"] = (
        df["task_energy_J"] / (df["total_images"] + df["warmup_images"])
    )
    return df


def subset_name(indices: tuple[int, ...]) -> str:
    return ":".join(FACTOR_LABEL[FACTORS[i]] for i in indices)


def blocked_factorial_anova(
    clones: pd.DataFrame,
    metric: str,
    factor_levels: dict[str, list[str | int]] | None = None,
) -> pd.DataFrame:
    """Orthogonal full-factorial ANOVA with nominal device as a block.

    The design is balanced: 3 devices x 2 x 3 x 7 x 3.  Sum-to-zero
    factorial effects are obtained by marginal-mean decomposition.  The
    residual is the device-by-treatment interaction; no batch or power sample
    is treated as an independent replicate.
    """

    devices = CLONE_DEVICES
    levels = LEVELS if factor_levels is None else factor_levels
    shape = [len(devices)] + [len(levels[f]) for f in FACTORS]
    arr = np.full(shape, np.nan, dtype=float)
    index = {
        "dataset": {value: i for i, value in enumerate(devices)},
        **{factor: {value: i for i, value in enumerate(levels[factor])} for factor in FACTORS},
    }
    for row in clones.itertuples(index=False):
        pos = [
            index["dataset"][row.dataset],
            index["model"][row.model],
            index["precision"][row.precision],
            index["batch"][row.batch],
            index["profile"][row.profile],
        ]
        arr[tuple(pos)] = math.log(float(getattr(row, metric)))
    if np.isnan(arr).any():
        raise RuntimeError(f"Incomplete primary clone grid for {metric}")

    grand = float(arr.mean())
    treatment_mean = arr.mean(axis=0)
    treatment_shape = treatment_mean.shape
    effects: dict[tuple[int, ...], np.ndarray] = {}
    rows: list[dict[str, float | str | int]] = []
    n_devices = len(devices)

    for order in range(1, len(FACTORS) + 1):
        for subset in itertools.combinations(range(len(FACTORS)), order):
            complement = tuple(i for i in range(len(FACTORS)) if i not in subset)
            marginal = treatment_mean.mean(axis=complement, keepdims=True) if complement else treatment_mean.copy()
            effect = np.broadcast_to(marginal, treatment_shape).astype(float).copy() - grand
            for lower_order in range(1, order):
                for lower in itertools.combinations(subset, lower_order):
                    effect -= effects[lower]
            effects[subset] = effect
            ss = n_devices * float(np.square(effect).sum())
            df_term = int(np.prod([len(levels[FACTORS[i]]) - 1 for i in subset]))
            rows.append({"term": subset_name(subset), "order": order, "df": df_term, "ss": ss})

    treatment_fit = grand + sum(effects.values())
    device_effect = arr.mean(axis=(1, 2, 3, 4)) - grand
    device_ss = int(np.prod(treatment_shape)) * float(np.square(device_effect).sum())
    residual = arr - treatment_fit[np.newaxis, ...] - device_effect[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    residual_ss = float(np.square(residual).sum())
    residual_df = (len(devices) - 1) * (int(np.prod(treatment_shape)) - 1)
    total_ss = float(np.square(arr - grand).sum())
    partition_ss = sum(float(row["ss"]) for row in rows) + device_ss + residual_ss
    if not np.isclose(partition_ss, total_ss, rtol=1e-10, atol=1e-10):
        raise RuntimeError(f"ANOVA partition failed for {metric}: {partition_ss} != {total_ss}")

    ms_error = residual_ss / residual_df
    for row in rows:
        ms = float(row["ss"]) / int(row["df"])
        f_value = ms / ms_error
        row.update(
            {
                "ms": ms,
                "f": f_value,
                "p": float(stats.f.sf(f_value, int(row["df"]), residual_df)),
                "variance_share": float(row["ss"]) / total_ss,
                "partial_eta_squared": float(row["ss"]) / (float(row["ss"]) + residual_ss),
                "omega_squared": max(0.0, (float(row["ss"]) - int(row["df"]) * ms_error) / (total_ss + ms_error)),
            }
        )
    rows.extend(
        [
            {
                "term": "Device block",
                "order": 0,
                "df": len(devices) - 1,
                "ss": device_ss,
                "ms": device_ss / (len(devices) - 1),
                "f": (device_ss / (len(devices) - 1)) / ms_error,
                "p": float(stats.f.sf((device_ss / (len(devices) - 1)) / ms_error, len(devices) - 1, residual_df)),
                "variance_share": device_ss / total_ss,
                "partial_eta_squared": device_ss / (device_ss + residual_ss),
                "omega_squared": max(0.0, (device_ss - (len(devices) - 1) * ms_error) / (total_ss + ms_error)),
            },
            {
                "term": "Device-by-configuration residual",
                "order": -1,
                "df": residual_df,
                "ss": residual_ss,
                "ms": ms_error,
                "f": np.nan,
                "p": np.nan,
                "variance_share": residual_ss / total_ss,
                "partial_eta_squared": np.nan,
                "omega_squared": np.nan,
            },
        ]
    )
    out = pd.DataFrame(rows)
    out.insert(0, "outcome", metric)
    out["scale"] = "natural logarithm"
    return out


def summarize_anova(anova: pd.DataFrame) -> pd.DataFrame:
    categories = ["Model", "Precision", "Batch", "Profile", "Two-way interactions", "Higher-order interactions", "Device block", "Residual"]
    rows = []
    for metric, group in anova.groupby("outcome", sort=False):
        values: dict[str, float] = {}
        for category in categories:
            if category in {"Model", "Precision", "Batch", "Profile"}:
                value = group.loc[group.term == category, "variance_share"].sum()
            elif category == "Two-way interactions":
                value = group.loc[group.order == 2, "variance_share"].sum()
            elif category == "Higher-order interactions":
                value = group.loc[group.order >= 3, "variance_share"].sum()
            elif category == "Device block":
                value = group.loc[group.term == "Device block", "variance_share"].sum()
            else:
                value = group.loc[group.term == "Device-by-configuration residual", "variance_share"].sum()
            values[category] = 100.0 * float(value)
        rows.append({"outcome": metric, **values})
    return pd.DataFrame(rows)


def device_level_ci(log_ratios: pd.Series) -> tuple[float, float, float, float]:
    values = np.asarray(log_ratios, dtype=float)
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    if len(values) > 1:
        half = float(stats.t.ppf(0.975, len(values) - 1) * sd / math.sqrt(len(values)))
    else:
        half = 0.0
    return math.exp(mean), math.exp(mean - half), math.exp(mean + half), sd


def precision_contrasts(clones: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in LEVELS["model"]:
        model_df = clones[clones.model == model]
        for numerator, denominator in [("fp32", "fp16"), ("fp64", "fp32")]:
            for metric_key, metric in OUTCOMES.items():
                pivot = model_df.pivot_table(index=["dataset", "batch", "profile"], columns="precision", values=metric, aggfunc="first")
                pair = np.log(pivot[numerator] / pivot[denominator]).rename("log_ratio").reset_index()
                device_logs = pair.groupby("dataset", sort=False).log_ratio.mean().reindex(CLONE_DEVICES)
                gmr, low, high, sd_log = device_level_ci(device_logs)
                rows.append(
                    {
                        "factor": "precision",
                        "comparison": f"{numerator}/{denominator}",
                        "model": model,
                        "metric": metric,
                        "n_devices": len(device_logs),
                        "matched_pairs_per_device": int(len(pair) / len(device_logs)),
                        "geometric_mean_ratio": gmr,
                        "ci95_low_device_block": low,
                        "ci95_high_device_block": high,
                        "sd_log_ratio_across_devices": sd_log,
                        "device_ratios": "|".join(f"{math.exp(v):.6g}" for v in device_logs),
                    }
                )
    return pd.DataFrame(rows)


def batch_contrasts(clones: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in LEVELS["model"]:
        for precision in LEVELS["precision"]:
            cell = clones[(clones.model == model) & (clones.precision == precision)]
            for metric_key, metric in OUTCOMES.items():
                pivot = cell.pivot_table(index=["dataset", "profile"], columns="batch", values=metric, aggfunc="first")
                pair = np.log(pivot[128] / pivot[1]).rename("log_ratio").reset_index()
                device_logs = pair.groupby("dataset", sort=False).log_ratio.mean().reindex(CLONE_DEVICES)
                gmr, low, high, sd_log = device_level_ci(device_logs)
                rows.append(
                    {
                        "factor": "batch",
                        "comparison": "128/1",
                        "model": model,
                        "precision": precision,
                        "metric": metric,
                        "n_devices": len(device_logs),
                        "matched_pairs_per_device": int(len(pair) / len(device_logs)),
                        "geometric_mean_ratio": gmr,
                        "ci95_low_device_block": low,
                        "ci95_high_device_block": high,
                        "sd_log_ratio_across_devices": sd_log,
                        "device_ratios": "|".join(f"{math.exp(v):.6g}" for v in device_logs),
                    }
                )
    return pd.DataFrame(rows)


def write_batch_plot_data(clones: pd.DataFrame) -> pd.DataFrame:
    summary = (
        clones.groupby(["model", "precision", "batch", "profile"], observed=True)
        .agg(
            n_devices=("dataset", "size"),
            throughput_mean=("throughput_img_s", "mean"),
            throughput_sd=("throughput_img_s", "std"),
            task_energy_mJ_mean=("task_energy_per_eval_image_J", lambda x: 1000.0 * x.mean()),
            task_energy_mJ_sd=("task_energy_per_eval_image_J", lambda x: 1000.0 * x.std(ddof=1)),
            task_throughput_mean=("task_throughput_img_s", "mean"),
            task_throughput_sd=("task_throughput_img_s", "std"),
            max_temperature_C_mean=("max_gpu_or_tj_temp_C", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(OUT / "clone_configuration_summary.csv", index=False)
    for model in LEVELS["model"]:
        for precision in LEVELS["precision"]:
            cell = summary[(summary.model == model) & (summary.precision == precision)]
            for metric_name, mean_col, sd_col in [
                ("throughput", "throughput_mean", "throughput_sd"),
                ("task_energy", "task_energy_mJ_mean", "task_energy_mJ_sd"),
            ]:
                wide = pd.DataFrame({"batch": LEVELS["batch"]})
                for profile in LEVELS["profile"]:
                    series = cell[cell.profile == profile].set_index("batch").reindex(LEVELS["batch"])
                    wide[f"{profile}_mean"] = series[mean_col].to_numpy()
                    wide[f"{profile}_sd"] = series[sd_col].to_numpy()
                wide.to_csv(OUT / f"batch_{metric_name}_{model}_{precision}.csv", index=False)
    return summary


def profile_repeatability(clones: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_map = {
        "throughput_cv_percent": "throughput_img_s",
        "power_cv_percent": "time_weighted_avg_power_W",
        "task_energy_cv_percent": "task_energy_per_eval_image_J",
    }
    config_keys = ["model", "precision", "batch", "profile"]
    config_rows = []
    for keys, group in clones.groupby(config_keys, observed=True):
        row = dict(zip(config_keys, keys))
        for out_name, metric in metric_map.items():
            row[out_name] = 100.0 * float(group[metric].std(ddof=1) / group[metric].mean())
        config_rows.append(row)
    config_cv = pd.DataFrame(config_rows)
    summary = config_cv.groupby("profile", observed=True)[list(metric_map)].median().reset_index()
    temps = clones.groupby("profile", observed=True).max_gpu_or_tj_temp_C.agg(["median", "min", "max"]).reset_index()
    temps = temps.rename(columns={"median": "maximum_temperature_median_C", "min": "maximum_temperature_min_C", "max": "maximum_temperature_max_C"})
    summary = summary.merge(temps, on="profile")
    summary["profile_order"] = summary.profile.map({"slow": 1, "medium": 2, "fast": 3})
    summary = summary.sort_values("profile_order")
    summary.to_csv(OUT / "profile_repeatability.csv", index=False)

    fast = clones[clones.profile == "fast"].copy()
    fast["clock_regime"] = np.where(
        fast.gpu_freq_mode_MHz >= 600.0,
        "high",
        "low",
    )
    counts = fast.groupby(["dataset", "clock_regime"], observed=True).size().unstack(fill_value=0).reindex(CLONE_DEVICES)
    for col in ["low", "high"]:
        if col not in counts:
            counts[col] = 0
    counts = counts[["low", "high"]].reset_index().rename(columns={"dataset": "device"})
    counts["total"] = counts.low + counts.high
    counts.to_csv(OUT / "fast_clock_regime_counts.csv", index=False)
    return summary, counts


def variance_share_sensitivity(clones: pd.DataFrame, full_anova: pd.DataFrame) -> pd.DataFrame:
    """Compare the full grid with FP16/FP32 and repeatable-profile sub-grids.

    The latter prevents the deliberately severe FP64 stress condition from
    dominating the scientific interpretation of practically common precision
    choices.  Both decompositions retain the three nominal device folders as
    blocks and use the run as the experimental unit.
    """

    practical_levels = {factor: list(values) for factor, values in LEVELS.items()}
    practical_levels["precision"] = ["fp16", "fp32"]
    practical = clones[clones.precision.isin(practical_levels["precision"])].copy()
    if len(practical) != 252:
        raise RuntimeError(f"Unexpected FP16/FP32 sensitivity grid size: {len(practical)}")

    practical_anova = pd.concat(
        [blocked_factorial_anova(practical, metric, practical_levels) for metric in OUTCOMES.values()],
        ignore_index=True,
    )
    practical_anova.to_csv(OUT / "factorial_anova_results_fp16_fp32.csv", index=False)

    repeatable_levels = {factor: list(values) for factor, values in practical_levels.items()}
    repeatable_levels["profile"] = ["slow", "medium"]
    repeatable = practical[practical.profile.isin(repeatable_levels["profile"])].copy()
    if len(repeatable) != 168:
        raise RuntimeError(f"Unexpected FP16/FP32 slow+medium grid size: {len(repeatable)}")
    repeatable_anova = pd.concat(
        [blocked_factorial_anova(repeatable, metric, repeatable_levels) for metric in OUTCOMES.values()],
        ignore_index=True,
    )
    repeatable_anova.to_csv(
        OUT / "factorial_anova_results_fp16_fp32_excluding_fast.csv", index=False
    )

    full_groups = summarize_anova(full_anova)
    full_groups.insert(0, "analysis_grid", "full_fp16_fp32_fp64")
    full_groups.insert(1, "precision_levels", "fp16|fp32|fp64")
    full_groups.insert(2, "n_runs", len(clones))
    practical_groups = summarize_anova(practical_anova)
    practical_groups.insert(0, "analysis_grid", "practical_fp16_fp32")
    practical_groups.insert(1, "precision_levels", "fp16|fp32")
    practical_groups.insert(2, "n_runs", len(practical))
    repeatable_groups = summarize_anova(repeatable_anova)
    repeatable_groups.insert(0, "analysis_grid", "practical_fp16_fp32_slow_medium")
    repeatable_groups.insert(1, "precision_levels", "fp16|fp32")
    repeatable_groups.insert(2, "n_runs", len(repeatable))
    out = pd.concat([full_groups, practical_groups, repeatable_groups], ignore_index=True)
    out.to_csv(OUT / "anova_variance_sensitivity.csv", index=False)

    # Keep panel (b) of the manuscript figure data-driven.  The two plotted
    # outcomes are ordered explicitly so that PGFPlots can use the CSV row
    # index as its categorical y coordinate.
    plot_outcomes = ["throughput_img_s", "task_energy_per_eval_image_J"]
    plot = repeatable_groups.loc[
        repeatable_groups["outcome"].isin(plot_outcomes),
        ["outcome", "Model", "Precision", "Batch"],
    ].copy()
    plot["outcome_order"] = plot["outcome"].map(
        {outcome: index for index, outcome in enumerate(plot_outcomes)}
    )
    plot = plot.sort_values("outcome_order").drop(columns="outcome_order")
    plot["remaining_pct"] = 100.0 - plot[["Model", "Precision", "Batch"]].sum(axis=1)
    plot = plot.rename(
        columns={
            "Model": "model_pct",
            "Precision": "precision_pct",
            "Batch": "batch_pct",
        }
    )
    plot.to_csv(OUT / "anova_variance_slow_medium_plot.csv", index=False)
    return out


def state_carryover_analysis(clones: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reconstruct the realized nominal-fast state from chronological order.

    Sequence is taken from the archived ``grid_state.jsonl`` running events;
    the analysis is descriptive.  It identifies a deterministic association
    between prior nominal profile and subsequent read-back clocks, but it does
    not assign a software or hardware mechanism without the executed runner.
    """

    sequence_rows: list[dict[str, object]] = []
    for device in CLONE_DEVICES:
        device_runs = clones[clones.dataset == device]
        directories = device_runs.source_directory.unique()
        if len(directories) != 1:
            raise RuntimeError(f"Expected one source directory for {device}, found {directories}")
        state_path = ROOT / str(directories[0]) / "grid_state.jsonl"
        events = [json.loads(line) for line in state_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        running = [event for event in events if event.get("status") == "running"]
        done = [event for event in events if event.get("status") == "done"]
        if len(running) != 126 or len(done) != 126:
            raise RuntimeError(
                f"Expected 126 running and 126 done events for {device}, got {len(running)} and {len(done)}"
            )
        running_ids = [str(event["run_id"]) for event in running]
        done_ids = [str(event["run_id"]) for event in done]
        if len(set(running_ids)) != 126 or set(running_ids) != set(done_ids):
            raise RuntimeError(f"Retries, duplicates, or unmatched state events in {state_path}")
        if set(running_ids) != set(device_runs.prefix):
            raise RuntimeError(f"State-log and audited-run identifiers differ for {device}")
        for sequence_index, event in enumerate(running, start=1):
            sequence_rows.append(
                {
                    "dataset": device,
                    "prefix": str(event["run_id"]),
                    "sequence_index": sequence_index,
                    "sequence_start_utc": pd.to_datetime(float(event["ts"]), unit="s", utc=True),
                }
            )

    sequence = pd.DataFrame(sequence_rows)
    ordered = clones.merge(sequence, on=["dataset", "prefix"], how="inner", validate="one_to_one")
    ordered["metadata_timestamp_utc"] = pd.to_datetime(ordered["timestamp_utc"], utc=True, errors="raise")
    ordered = ordered.sort_values(["dataset", "sequence_index"]).reset_index(drop=True)

    ordered["realized_fast_regime"] = np.where(
        ordered.profile.ne("fast"),
        "not_fast",
        np.where(
            ordered.gpu_freq_mode_MHz >= 600.0,
            "high",
            "low",
        ),
    )

    rows: list[dict[str, object]] = []
    for device, group in ordered.groupby("dataset", sort=False):
        previous: pd.Series | None = None
        last_nonfast: pd.Series | None = None
        for _, run in group.iterrows():
            if run.profile == "fast":
                if previous is None or last_nonfast is None:
                    raise RuntimeError(f"Fast run lacks an observable predecessor in {device}: {run.prefix}")
                if previous.profile == "slow":
                    immediate_state = "slow"
                    expected_immediate = "low"
                elif previous.profile == "medium":
                    immediate_state = "medium"
                    expected_immediate = "high"
                elif previous.profile == "fast":
                    immediate_state = f"fast-{previous.realized_fast_regime}"
                    expected_immediate = str(previous.realized_fast_regime)
                else:
                    raise RuntimeError(f"Unexpected predecessor profile: {previous.profile}")
                expected_last_nonfast = {"slow": "low", "medium": "high"}[str(last_nonfast.profile)]
                rows.append(
                    {
                        "device_block": device,
                        "fast_sequence_index": run.sequence_index,
                        "fast_start_timestamp_utc": run.sequence_start_utc.isoformat(),
                        "fast_metadata_timestamp_utc": run.metadata_timestamp_utc.isoformat(),
                        "fast_prefix": run.prefix,
                        "model": run.model,
                        "precision": run.precision,
                        "batch": int(run.batch),
                        "realized_fast_regime": run.realized_fast_regime,
                        "fast_cpu_mode_MHz": run.cpu_freq_mode_MHz,
                        "fast_gpu_mode_MHz": run.gpu_freq_mode_MHz,
                        "immediate_predecessor_sequence_index": previous.sequence_index,
                        "immediate_predecessor_start_timestamp_utc": previous.sequence_start_utc.isoformat(),
                        "immediate_predecessor_prefix": previous.prefix,
                        "immediate_predecessor_profile": previous.profile,
                        "immediate_predecessor_fast_regime": (
                            previous.realized_fast_regime if previous.profile == "fast" else "not_applicable"
                        ),
                        "immediate_predecessor_state": immediate_state,
                        "expected_regime_from_immediate_predecessor": expected_immediate,
                        "immediate_rule_match": run.realized_fast_regime == expected_immediate,
                        "last_nonfast_sequence_index": last_nonfast.sequence_index,
                        "last_nonfast_start_timestamp_utc": last_nonfast.sequence_start_utc.isoformat(),
                        "last_nonfast_prefix": last_nonfast.prefix,
                        "last_nonfast_profile": last_nonfast.profile,
                        "expected_regime_from_last_nonfast": expected_last_nonfast,
                        "last_nonfast_rule_match": run.realized_fast_regime == expected_last_nonfast,
                    }
                )
            else:
                last_nonfast = run
            previous = run

    run_level = pd.DataFrame(rows)
    if len(run_level) != 126:
        raise RuntimeError(f"Expected 126 nominal-fast transitions, found {len(run_level)}")

    state_order = ["slow", "medium", "fast-low", "fast-high"]
    immediate_rows = []
    for state in state_order:
        group = run_level[run_level.immediate_predecessor_state == state]
        immediate_rows.append(
            {
                "immediate_predecessor_state": state,
                "n_fast_runs": len(group),
                "observed_low": int((group.realized_fast_regime == "low").sum()),
                "observed_high": int((group.realized_fast_regime == "high").sum()),
                "expected_regime": "low" if state in {"slow", "fast-low"} else "high",
                "rule_matches": int(group.immediate_rule_match.sum()),
                "rule_accuracy_percent": 100.0 * float(group.immediate_rule_match.mean()),
            }
        )
    immediate = pd.DataFrame(immediate_rows)

    last_rows = []
    for profile in ["slow", "medium"]:
        group = run_level[run_level.last_nonfast_profile == profile]
        last_rows.append(
            {
                "last_nonfast_profile": profile,
                "n_fast_runs": len(group),
                "observed_low": int((group.realized_fast_regime == "low").sum()),
                "observed_high": int((group.realized_fast_regime == "high").sum()),
                "expected_regime": "low" if profile == "slow" else "high",
                "rule_matches": int(group.last_nonfast_rule_match.sum()),
                "rule_accuracy_percent": 100.0 * float(group.last_nonfast_rule_match.mean()),
            }
        )
    last_nonfast = pd.DataFrame(last_rows)

    if not run_level.immediate_rule_match.all() or not run_level.last_nonfast_rule_match.all():
        raise RuntimeError("The state-carry-over rule is no longer deterministic; revise the manuscript claim")

    configuration_rows: list[dict[str, object]] = []
    for (model, precision, batch), group in run_level.groupby(
        ["model", "precision", "batch"], observed=True, sort=True
    ):
        low_blocks = int((group.realized_fast_regime == "low").sum())
        high_blocks = int((group.realized_fast_regime == "high").sum())
        configuration_rows.append(
            {
                "model": model,
                "precision": precision,
                "batch": int(batch),
                "n_device_blocks": group.device_block.nunique(),
                "low_blocks": low_blocks,
                "high_blocks": high_blocks,
                "regime_changes_across_blocks": low_blocks > 0 and high_blocks > 0,
                "regime_pattern": (
                    "mixed_across_blocks"
                    if low_blocks > 0 and high_blocks > 0
                    else "all_low"
                    if low_blocks == 3
                    else "all_high"
                ),
            }
        )
    configuration_consistency = pd.DataFrame(configuration_rows)
    pattern_counts = configuration_consistency.regime_pattern.value_counts().to_dict()
    expected_patterns = {"mixed_across_blocks": 35, "all_low": 3, "all_high": 4}
    if len(configuration_consistency) != 42 or pattern_counts != expected_patterns:
        raise RuntimeError(
            "Unexpected workload-cell regime consistency: "
            f"cells={len(configuration_consistency)}, patterns={pattern_counts}"
        )

    run_level.to_csv(OUT / "fast_state_carryover_runs.csv", index=False)
    immediate.to_csv(OUT / "fast_state_carryover_immediate_predecessor.csv", index=False)
    last_nonfast.to_csv(OUT / "fast_state_carryover_last_nonfast.csv", index=False)
    configuration_consistency.to_csv(
        OUT / "fast_state_carryover_configuration_consistency.csv", index=False
    )

    immediate_slow = run_level.immediate_predecessor_profile.eq("slow").sum()
    immediate_medium = run_level.immediate_predecessor_profile.eq("medium").sum()
    immediate_fast = run_level.immediate_predecessor_profile.eq("fast").sum()
    fast_retained = (
        run_level.loc[run_level.immediate_predecessor_profile.eq("fast"), "immediate_rule_match"].sum()
    )
    macro_values = {
        "FastLastSlowCount": int((run_level.last_nonfast_profile == "slow").sum()),
        "FastLastMediumCount": int((run_level.last_nonfast_profile == "medium").sum()),
        "FastImmediateSlowCount": int(immediate_slow),
        "FastImmediateMediumCount": int(immediate_medium),
        "FastImmediateFastCount": int(immediate_fast),
        "FastImmediateFastRetained": int(fast_retained),
        "FastTransitionTotal": len(run_level),
        "FastWorkloadCellTotal": len(configuration_consistency),
        "FastWorkloadMixedCells": int(pattern_counts["mixed_across_blocks"]),
        "FastWorkloadAllLowCells": int(pattern_counts["all_low"]),
        "FastWorkloadAllHighCells": int(pattern_counts["all_high"]),
    }
    macro_lines = ["% Generated by analysis/build_revision_results.py; do not edit manually."]
    macro_lines += [f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in macro_values.items()]
    (OUT / "state_carryover_macros.tex").write_text("\n".join(macro_lines) + "\n", encoding="utf-8")
    return run_level, immediate, last_nonfast


def precision_batch_crossover(
    clones: pd.DataFrame,
    *,
    output_stem: str = "precision_batch_crossover",
    expected_forward_not_faster: int | None = None,
    expected_energy_not_lower: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Quantify FP16/FP32 crossovers at matched device/profile/model/batch cells."""

    paired = clones[clones.precision.isin(["fp16", "fp32"])].pivot_table(
        index=["dataset", "model", "batch", "profile"],
        columns="precision",
        values=["throughput_img_s", "task_energy_per_eval_image_J", "task_throughput_img_s"],
        aggfunc="first",
    )
    expected_cells = (
        clones.dataset.nunique()
        * clones.model.nunique()
        * clones.batch.nunique()
        * clones.profile.nunique()
    )
    if paired.isna().any().any() or len(paired) != expected_cells:
        raise RuntimeError(
            f"Incomplete FP16/FP32 matched-cell table: {len(paired)} rows; expected {expected_cells}"
        )
    paired.columns = [f"{metric}_{precision}" for metric, precision in paired.columns]
    cells = paired.reset_index()
    cells["forward_throughput_fp16_over_fp32"] = (
        cells.throughput_img_s_fp16 / cells.throughput_img_s_fp32
    )
    cells["logged_window_energy_fp16_over_fp32"] = (
        cells.task_energy_per_eval_image_J_fp16 / cells.task_energy_per_eval_image_J_fp32
    )
    cells["logged_window_throughput_fp16_over_fp32"] = (
        cells.task_throughput_img_s_fp16 / cells.task_throughput_img_s_fp32
    )
    cells["fp16_forward_faster"] = cells.forward_throughput_fp16_over_fp32 > 1.0
    cells["fp16_logged_window_energy_lower"] = cells.logged_window_energy_fp16_over_fp32 < 1.0
    cells["fp16_logged_window_throughput_faster"] = (
        cells.logged_window_throughput_fp16_over_fp32 > 1.0
    )

    rows = []
    for (model, batch), group in cells.groupby(["model", "batch"], observed=True, sort=False):
        row: dict[str, object] = {
            "model": model,
            "batch": int(batch),
            "n_matched_device_profile_cells": len(group),
            "n_devices": group.dataset.nunique(),
            "n_profiles": group.profile.nunique(),
        }
        metric_specs = [
            ("forward_throughput", "forward_throughput_fp16_over_fp32", "fp16_forward_faster"),
            ("logged_window_energy", "logged_window_energy_fp16_over_fp32", "fp16_logged_window_energy_lower"),
            (
                "logged_window_throughput",
                "logged_window_throughput_fp16_over_fp32",
                "fp16_logged_window_throughput_faster",
            ),
        ]
        for label, ratio_column, favorable_column in metric_specs:
            device_logs = group.assign(log_ratio=np.log(group[ratio_column])).groupby("dataset").log_ratio.mean()
            gmr, low, high, sd_log = device_level_ci(device_logs.reindex(CLONE_DEVICES))
            row[f"{label}_fp16_over_fp32_gmr"] = gmr
            row[f"{label}_ci95_low_device_block"] = low
            row[f"{label}_ci95_high_device_block"] = high
            row[f"{label}_sd_log_ratio_across_devices"] = sd_log
            row[f"{label}_favorable_count"] = int(group[favorable_column].sum())
            row[f"{label}_not_favorable_count"] = int((~group[favorable_column]).sum())
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(["model", "batch"]).reset_index(drop=True)

    forward_not_faster = int((~cells.fp16_forward_faster).sum())
    energy_not_lower = int((~cells.fp16_logged_window_energy_lower).sum())
    if expected_forward_not_faster is not None and forward_not_faster != expected_forward_not_faster:
        raise RuntimeError(
            f"Unexpected forward crossover count: {forward_not_faster}; "
            f"expected {expected_forward_not_faster}"
        )
    if expected_energy_not_lower is not None and energy_not_lower != expected_energy_not_lower:
        raise RuntimeError(
            f"Unexpected energy crossover count: {energy_not_lower}; expected {expected_energy_not_lower}"
        )
    cells.to_csv(OUT / f"{output_stem}_cells.csv", index=False)
    summary.to_csv(OUT / f"{output_stem}_summary.csv", index=False)
    return cells, summary


def protocol_batch_summary(clones: pd.DataFrame) -> pd.DataFrame:
    """Describe batch-dependent work and I/O implied by data and reference code."""

    rows = []
    for batch in LEVELS["batch"]:
        group = clones[clones.batch == batch]
        iterations = sorted(group.perf_rows_iterations.unique())
        total_images = sorted(group.total_images.unique())
        if len(iterations) != 1 or len(total_images) != 1:
            raise RuntimeError(f"Non-constant protocol counts for batch {batch}")
        iteration_count = int(iterations[0])
        evaluated_images = int(total_images[0])
        expected_iterations = math.ceil(evaluated_images / batch)
        if iteration_count != expected_iterations:
            raise RuntimeError(f"Unexpected iteration count for batch {batch}: {iteration_count}")
        warmup_batches = 50
        warmup_images = warmup_batches * batch
        rows.append(
            {
                "batch": batch,
                "n_primary_runs": len(group),
                "evaluated_images_per_run": evaluated_images,
                "timed_batch_iterations_per_run": iteration_count,
                "last_timed_batch_images": evaluated_images % batch or batch,
                "warmup_batches_from_metadata": warmup_batches,
                "warmup_images_if_full_batch": warmup_images,
                "images_processed_including_warmup": evaluated_images + warmup_images,
                "warmup_share_of_processed_images_percent": 100.0
                * warmup_images
                / (evaluated_images + warmup_images),
                "reference_writer_flush_calls_if_one_per_timed_batch": iteration_count,
                "flush_evidence_scope": (
                    "inspected reference implementation calls csv_file.flush() after each written timed batch; "
                    "executed July runner/code commit is not archived"
                ),
                "warmup_evidence_scope": (
                    "all primary metadata report warmup=50; full-batch warmup follows the inspected reference implementation"
                ),
            }
        )
    out = pd.DataFrame(rows)
    out["flush_calls_relative_to_batch128"] = (
        out.reference_writer_flush_calls_if_one_per_timed_batch
        / float(out.loc[out.batch == 128, "reference_writer_flush_calls_if_one_per_timed_batch"].iloc[0])
    )
    out.to_csv(OUT / "protocol_batch_dependent_work_summary.csv", index=False)
    return out


def top_configurations(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in LEVELS["model"]:
        for precision in LEVELS["precision"]:
            cell = summary[(summary.model == model) & (summary.precision == precision)]
            for objective, column, direction in [
                ("maximum forward throughput", "throughput_mean", "max"),
                ("minimum whole-task energy", "task_energy_mJ_mean", "min"),
                ("maximum whole-task throughput", "task_throughput_mean", "max"),
            ]:
                idx = cell[column].idxmax() if direction == "max" else cell[column].idxmin()
                selected = cell.loc[idx]
                rows.append(
                    {
                        "model": model,
                        "precision": precision,
                        "objective": objective,
                        "profile": selected.profile,
                        "batch": int(selected.batch),
                        "forward_throughput_mean": selected.throughput_mean,
                        "forward_throughput_sd": selected.throughput_sd,
                        "task_energy_mJ_mean": selected.task_energy_mJ_mean,
                        "task_energy_mJ_sd": selected.task_energy_mJ_sd,
                        "task_throughput_mean": selected.task_throughput_mean,
                        "task_throughput_sd": selected.task_throughput_sd,
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "clone_best_configurations.csv", index=False)
    return out


def historical_top_three(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset in ["fixed_2026-03", "random_2026-03"]:
        group = df[df.dataset == dataset]
        for objective, metric, ascending in [
            ("whole-task energy", "task_energy_per_eval_image_J", True),
            ("forward throughput", "throughput_img_s", False),
        ]:
            ranked = group.sort_values(metric, ascending=ascending).head(3)
            for rank, row in enumerate(ranked.itertuples(index=False), start=1):
                rows.append(
                    {
                        "campaign": dataset,
                        "objective": objective,
                        "rank": rank,
                        "configuration": row.prefix,
                        "forward_throughput_img_s": row.throughput_img_s,
                        "whole_task_energy_mJ_per_image": 1000.0 * row.task_energy_per_eval_image_J,
                        "whole_task_throughput_img_s": row.task_throughput_img_s,
                        "emc_mode_MHz": row.emc_freq_mode_MHz,
                        "gpu_mode_MHz": row.gpu_freq_mode_MHz,
                    }
                )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "historical_campaign_top_three_descriptive.csv", index=False)
    return out


def campaign_summary(df: pd.DataFrame) -> pd.DataFrame:
    def joined(values: pd.Series) -> str:
        return "|".join(str(v) for v in sorted(values.dropna().unique()))

    out = (
        df.groupby("dataset", observed=True)
        .agg(
            runs=("prefix", "size"),
            models=("model", joined),
            precisions=("precision", joined),
            profiles=("profile", joined),
            batches=("batch", joined),
            seeds=("seed", joined),
            total_evaluated_images=("total_images", "sum"),
            emc_mode_median_MHz=("emc_freq_mode_MHz", "median"),
            maximum_temperature_median_C=("max_gpu_or_tj_temp_C", "median"),
            maximum_temperature_max_C=("max_gpu_or_tj_temp_C", "max"),
        )
        .reset_index()
    )
    out.to_csv(OUT / "campaign_summary.csv", index=False)
    return out


def tail_latency_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame(
        [
            {
                "runs": len(df),
                "runs_with_gt5xp50": int((df.latency_iterations_gt5x_p50 > 0).sum()),
                "iterations_gt5xp50": int(df.latency_iterations_gt5x_p50.sum()),
                "fp64_bs128_runs_with_gt5xp50": int(
                    ((df.latency_iterations_gt5x_p50 > 0) & (df.precision == "fp64") & (df.batch == 128)).sum()
                ),
                "maximum_observed_batch_latency_s": float(df.latency_max_ms.max() / 1000.0),
                "maximum_throughput_sensitivity_percent": float(df.throughput_sensitivity_percent_increase_excluding_gt5x_p50.max()),
            }
        ]
    )
    summary.to_csv(OUT / "tail_latency_summary.csv", index=False)
    top = df.sort_values("throughput_sensitivity_percent_increase_excluding_gt5x_p50", ascending=False).head(15)
    top[[
        "dataset", "prefix", "latency_p50_ms", "latency_p95_ms", "latency_p99_ms", "latency_max_ms",
        "latency_iterations_gt5x_p50", "throughput_sensitivity_percent_increase_excluding_gt5x_p50",
    ]].to_csv(OUT / "tail_latency_largest_cases.csv", index=False)
    return summary


def parse_trace(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    timestamps = []
    cpu = []
    gpu = []
    temp = []
    power = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        ts_match = LOG_TS_RE.search(line)
        cpu_match = LOG_CPU_BLOCK_RE.search(line)
        gpu_match = LOG_GPU_RE.search(line)
        temps = [float(value) for value in LOG_TEMP_RE.findall(line)]
        power_match = LOG_POWER_RE.search(line)
        if not (ts_match and cpu_match and gpu_match and temps and power_match):
            continue
        timestamp = pd.to_datetime(ts_match.group(1), format="%m-%d-%Y %H:%M:%S")
        cpu_values = [float(value) for value in re.findall(r"@(\d+(?:\.\d+)?)", cpu_match.group(1))]
        gpu_values = [float(value) for value in re.findall(r"\d+(?:\.\d+)?", gpu_match.group(1))]
        if not cpu_values or not gpu_values:
            continue
        timestamps.append(timestamp.timestamp())
        cpu.append(max(cpu_values))
        gpu.append(max(gpu_values))
        temp.append(max(temps))
        power.append(float(power_match.group(1)) / 1000.0)
    t = np.asarray(timestamps, dtype=float)
    t -= t[0]
    return t, np.asarray(cpu), np.asarray(gpu), np.asarray(temp), np.asarray(power)


def representative_trace() -> pd.DataFrame:
    progress = np.linspace(0.0, 100.0, 101)
    out = pd.DataFrame({"progress_percent": progress})
    folder = ROOT / "Ergebnisse_Clone1_13.07.2026"
    for profile in LEVELS["profile"]:
        path = folder / f"mobilenet_v2_fp16_{profile}_BS32_tegrastats.log"
        t, cpu, gpu, temp, power = parse_trace(path)
        normalized = 100.0 * t / t[-1]
        out[f"{profile}_cpu_MHz"] = np.interp(progress, normalized, cpu)
        out[f"{profile}_gpu_MHz"] = np.interp(progress, normalized, gpu)
        out[f"{profile}_temperature_C"] = np.interp(progress, normalized, temp)
        out[f"{profile}_power_W"] = np.interp(progress, normalized, power)
    out.to_csv(OUT / "representative_cpu_gpu_temperature_trace.csv", index=False)
    return out


def utilization_summary(clones: pd.DataFrame) -> pd.DataFrame:
    """Summarize whole-log CPU, GPU-active, and EMC-bandwidth observations."""

    rows = []
    for run in clones.itertuples(index=False):
        path = ROOT / run.source_directory / f"{run.prefix}_tegrastats.log"
        gpu_values: list[float] = []
        emc_values: list[float] = []
        cpu_values: list[float] = []
        fan_fields = 0
        throttle_fields = 0
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                gpu_match = LOG_GPU_UTIL_RE.search(line)
                emc_match = LOG_EMC_UTIL_RE.search(line)
                cpu_match = LOG_CPU_BLOCK_RE.search(line)
                if gpu_match:
                    gpu_values.append(float(gpu_match.group(1)))
                if emc_match:
                    emc_values.append(float(emc_match.group(1)))
                if cpu_match:
                    core_values = [float(value) for value in LOG_PERCENT_RE.findall(cpu_match.group(1))]
                    if core_values:
                        cpu_values.append(float(np.mean(core_values)))
                fan_fields += int("FAN" in line.upper())
                throttle_fields += int("THROTTL" in line.upper())
        rows.append(
            {
                "dataset": run.dataset,
                "prefix": run.prefix,
                "model": run.model,
                "precision": run.precision,
                "batch": run.batch,
                "profile": run.profile,
                "samples": len(gpu_values),
                "gpu_active_mean_percent": float(np.mean(gpu_values)),
                "gpu_active_p95_percent": float(np.quantile(gpu_values, 0.95)),
                "emc_bandwidth_mean_percent": float(np.mean(emc_values)),
                "emc_bandwidth_p95_percent": float(np.quantile(emc_values, 0.95)),
                "cpu_mean_across_cores_percent": float(np.mean(cpu_values)),
                "cpu_p95_across_cores_percent": float(np.quantile(cpu_values, 0.95)),
                "fan_fields": fan_fields,
                "throttle_fields": throttle_fields,
                "forward_time_fraction_of_meta_duration": run.inference_time_fraction_of_meta_duration,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "run_utilization_summary.csv", index=False)
    grouped = (
        out.groupby(["model", "precision"], observed=True)
        .agg(
            n_runs=("prefix", "size"),
            median_gpu_active_percent=("gpu_active_mean_percent", "median"),
            median_emc_bandwidth_percent=("emc_bandwidth_mean_percent", "median"),
            median_cpu_across_cores_percent=("cpu_mean_across_cores_percent", "median"),
            median_forward_time_fraction=("forward_time_fraction_of_meta_duration", "median"),
        )
        .reset_index()
    )
    grouped.to_csv(OUT / "utilization_by_model_precision.csv", index=False)
    return out


def write_macros(
    df: pd.DataFrame,
    clones: pd.DataFrame,
    precision: pd.DataFrame,
    batch: pd.DataFrame,
    repeatability: pd.DataFrame,
    tails: pd.DataFrame,
) -> None:
    def contrast(table: pd.DataFrame, **filters: str) -> pd.Series:
        selected = table.copy()
        for key, value in filters.items():
            selected = selected[selected[key] == value]
        if len(selected) != 1:
            raise RuntimeError(f"Expected one contrast for {filters}, got {len(selected)}")
        return selected.iloc[0]

    p_m_fp32 = contrast(precision, comparison="fp32/fp16", model="mobilenet_v2", metric="throughput_img_s")
    p_r_fp32 = contrast(precision, comparison="fp32/fp16", model="resnet50", metric="throughput_img_s")
    e_m_fp32 = contrast(precision, comparison="fp32/fp16", model="mobilenet_v2", metric="task_energy_per_eval_image_J")
    e_r_fp32 = contrast(precision, comparison="fp32/fp16", model="resnet50", metric="task_energy_per_eval_image_J")
    p_m_fp64 = contrast(precision, comparison="fp64/fp32", model="mobilenet_v2", metric="throughput_img_s")
    p_r_fp64 = contrast(precision, comparison="fp64/fp32", model="resnet50", metric="throughput_img_s")
    e_m_fp64 = contrast(precision, comparison="fp64/fp32", model="mobilenet_v2", metric="task_energy_per_eval_image_J")
    e_r_fp64 = contrast(precision, comparison="fp64/fp32", model="resnet50", metric="task_energy_per_eval_image_J")
    fast = repeatability[repeatability.profile == "fast"].iloc[0]
    slow = repeatability[repeatability.profile == "slow"].iloc[0]
    medium = repeatability[repeatability.profile == "medium"].iloc[0]
    tail = tails.iloc[0]

    values = {
        "TotalRuns": f"{len(df)}",
        "PrimaryRuns": f"{len(clones)}",
        "PrimaryConfigurations": "126",
        "PrimaryDevices": "3",
        "TotalEvaluatedImagesMillions": f"{df.total_images.sum()/1e6:.1f}",
        "MobileFpThirtyTwoThroughputRatio": f"{p_m_fp32.geometric_mean_ratio:.3f}",
        "ResnetFpThirtyTwoThroughputRatio": f"{p_r_fp32.geometric_mean_ratio:.3f}",
        "MobileFpThirtyTwoEnergyRatio": f"{e_m_fp32.geometric_mean_ratio:.3f}",
        "ResnetFpThirtyTwoEnergyRatio": f"{e_r_fp32.geometric_mean_ratio:.3f}",
        "MobileFpSixtyFourThroughputRatio": f"{p_m_fp64.geometric_mean_ratio:.3f}",
        "ResnetFpSixtyFourThroughputRatio": f"{p_r_fp64.geometric_mean_ratio:.4f}",
        "MobileFpSixtyFourEnergyRatio": f"{e_m_fp64.geometric_mean_ratio:.3f}",
        "ResnetFpSixtyFourEnergyRatio": f"{e_r_fp64.geometric_mean_ratio:.2f}",
        "SlowThroughputCv": f"{slow.throughput_cv_percent:.3f}",
        "MediumThroughputCv": f"{medium.throughput_cv_percent:.3f}",
        "FastThroughputCv": f"{fast.throughput_cv_percent:.2f}",
        "FastEnergyCv": f"{fast.task_energy_cv_percent:.2f}",
        "MaximumObservedTemperature": f"{clones.max_gpu_or_tj_temp_C.max():.1f}",
        "RunsWithLargeStalls": f"{int(tail.runs_with_gt5xp50)}",
        "LargeStallIterations": f"{int(tail.iterations_gt5xp50)}",
        "MaximumStallSensitivity": f"{tail.maximum_throughput_sensitivity_percent:.2f}",
    }
    lines = ["% Generated by analysis/build_revision_results.py; do not edit manually."]
    lines += [f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in values.items()]
    (OUT / "revision_macros.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    df = load_runs()
    clones = df[df.dataset.isin(CLONE_DEVICES)].copy()
    if len(df) != 558 or len(clones) != 378:
        raise RuntimeError(f"Unexpected run counts: all={len(df)}, primary={len(clones)}")

    boundary_columns = [
        "dataset",
        "prefix",
        "model",
        "precision",
        "batch",
        "profile",
        "total_images",
        "warmup_images",
        "throughput_img_s",
        "task_time_s",
        "task_throughput_img_s",
        "task_energy_J",
        "task_energy_per_eval_image_J",
        "task_energy_per_processed_image_J",
        "task_edp_Js",
    ]
    df[boundary_columns].to_csv(OUT / "measurement_boundary_metrics.csv", index=False)

    anova_parts = [blocked_factorial_anova(clones, metric) for metric in OUTCOMES.values()]
    anova = pd.concat(anova_parts, ignore_index=True)
    anova.to_csv(OUT / "factorial_anova_results.csv", index=False)
    variance_groups = summarize_anova(anova)
    variance_groups.to_csv(OUT / "anova_variance_groups.csv", index=False)
    variance_sensitivity = variance_share_sensitivity(clones, anova)
    variance_groups.rename(
        columns={
            "Model": "model_pct",
            "Precision": "precision_pct",
            "Batch": "batch_pct",
            "Profile": "profile_pct",
            "Two-way interactions": "two_way_pct",
            "Higher-order interactions": "higher_order_pct",
            "Device block": "device_pct",
            "Residual": "residual_pct",
        }
    ).to_csv(OUT / "anova_variance_plot.csv", index=False)

    precision = precision_contrasts(clones)
    precision.to_csv(OUT / "precision_device_block_contrasts.csv", index=False)
    precision_excluding_fast = precision_contrasts(clones[clones.profile.isin(["slow", "medium"])])
    precision_excluding_fast.insert(1, "included_profiles", "slow|medium")
    precision_excluding_fast.to_csv(
        OUT / "precision_device_block_contrasts_excluding_fast.csv", index=False
    )
    batch = batch_contrasts(clones)
    batch.to_csv(OUT / "batch_device_block_contrasts.csv", index=False)
    batch_excluding_fast = batch_contrasts(clones[clones.profile.isin(["slow", "medium"])])
    batch_excluding_fast.insert(1, "included_profiles", "slow|medium")
    batch_excluding_fast.to_csv(OUT / "batch_device_block_contrasts_excluding_fast.csv", index=False)
    summary = write_batch_plot_data(clones)
    repeatability, fast_counts = profile_repeatability(clones)
    carryover_runs, carryover_immediate, carryover_last = state_carryover_analysis(clones)
    crossover_cells, crossover_summary = precision_batch_crossover(
        clones,
        expected_forward_not_faster=19,
        expected_energy_not_lower=12,
    )
    crossover_no_fast_cells, crossover_no_fast_summary = precision_batch_crossover(
        clones[clones.profile.isin(["slow", "medium"])],
        output_stem="precision_batch_crossover_excluding_fast",
        expected_forward_not_faster=12,
        expected_energy_not_lower=6,
    )
    no_fast_batch_one = crossover_no_fast_summary[crossover_no_fast_summary.batch == 1]
    if len(no_fast_batch_one) != 2 or not (
        no_fast_batch_one.forward_throughput_ci95_high_device_block < 1.0
    ).all():
        raise RuntimeError("Slow+medium batch-one FP16/FP32 crossover no longer excludes unity")
    no_fast_batch_four_plus = crossover_no_fast_cells[crossover_no_fast_cells.batch >= 4]
    if len(no_fast_batch_four_plus) != 72 or not no_fast_batch_four_plus.fp16_forward_faster.all():
        raise RuntimeError("Expected FP16 to be faster in all 72 slow+medium cells at batch >= 4")
    protocol_summary = protocol_batch_summary(clones)
    top_configurations(summary)
    historical_top_three(df)
    campaign_summary(df)
    tails = tail_latency_summary(df)
    representative_trace()
    utilization_summary(clones)
    write_macros(df, clones, precision, batch, repeatability, tails)

    print(
        {
            "all_runs": len(df),
            "primary_clone_runs": len(clones),
            "anova_rows": len(anova),
            "variance_sensitivity_rows": len(variance_sensitivity),
            "precision_contrasts": len(precision),
            "precision_contrasts_excluding_fast": len(precision_excluding_fast),
            "batch_contrasts": len(batch),
            "batch_contrasts_excluding_fast": len(batch_excluding_fast),
            "fast_carryover_runs": len(carryover_runs),
            "precision_batch_crossover_cells": len(crossover_cells),
            "precision_batch_crossover_excluding_fast_cells": len(crossover_no_fast_cells),
            "protocol_batch_rows": len(protocol_summary),
            "generated_dir": str(OUT),
        }
    )


if __name__ == "__main__":
    main()
