#!/usr/bin/env python3
"""Conservative long-run, successor, and temporal-drift diagnostics.

This secondary analysis addresses Reviewer 2, Comment 7 using only archived
run-level results and state-event chronology.  It deliberately avoids causal
or population-level inference:

* immediate state-event gaps document the archived wrapper chronology;
* successor effects use configuration and physical-board fixed effects after
  FP64 or upper-duration-quartile predecessors;
* drift slopes use different random permutations in the three primary board
  folders after removing additive configuration and board means;
* requested-fast successors are excluded from the conservative primary
  successor/drift summaries because their realized state is already known to
  depend deterministically on predecessor state;
* no p values are reported.  Leave-one-board-out estimates are retained
  because three physical-board blocks cannot support a robust population model.

Run ``full_audit.py`` first so that the run table contains first/end telemetry
temperatures and the hardened raw-log integrity checks.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


REV = Path(__file__).resolve().parents[1]
AUDIT = REV / "supplementary" / "audit"
RUN_TABLE = AUDIT / "run_level_audit_and_recomputed_metrics.csv"
EVENT_TABLE = AUDIT / "grid_state_events.csv"
PRIMARY_BLOCKS = ["Clone1", "Clone2", "Clone3"]

PAIR_OUT = AUDIT / "long_run_successor_pairs.csv"
BINARY_OUT = AUDIT / "long_run_successor_binary_effects.csv"
ASSOCIATION_OUT = AUDIT / "long_run_successor_continuous_associations.csv"
THERMAL_OUT = AUDIT / "long_run_successor_temperature_transitions.csv"
GAP_OUT = AUDIT / "long_run_state_event_gaps.csv"
DRIFT_OUT = AUDIT / "long_run_temporal_drift_summary.csv"
MARKDOWN_OUT = AUDIT / "LONG_RUN_DRIFT_ANALYSIS.md"
TEX_OUT = AUDIT / "long_run_drift_results.tex"


OUTCOMES: list[dict[str, str]] = [
    {
        "name": "forward_throughput",
        "column": "throughput_img_s",
        "transform": "log",
        "unit": "ratio",
    },
    {
        "name": "logged_energy_per_evaluated_image",
        "column": "energy_per_img_J_integrated_log_timestamps",
        "transform": "log",
        "unit": "ratio",
    },
    {
        "name": "logged_window_throughput",
        "column": "logged_window_throughput_img_s",
        "transform": "log",
        "unit": "ratio",
    },
    {
        "name": "average_input_power",
        "column": "time_weighted_avg_power_W",
        "transform": "log",
        "unit": "ratio",
    },
    {
        "name": "logged_duration",
        "column": "log_wall_span_s",
        "transform": "log",
        "unit": "ratio",
    },
    {
        "name": "start_temperature",
        "column": "log_start_gpu_or_tj_temp_C",
        "transform": "identity",
        "unit": "delta_C",
    },
    {
        "name": "maximum_temperature",
        "column": "max_gpu_or_tj_temp_C",
        "transform": "identity",
        "unit": "delta_C",
    },
]


def finite_spearman(x: pd.Series, y: pd.Series) -> float:
    mask = np.isfinite(x.to_numpy(dtype=float)) & np.isfinite(y.to_numpy(dtype=float))
    xv = x.to_numpy(dtype=float)[mask]
    yv = y.to_numpy(dtype=float)[mask]
    if len(xv) < 3 or len(np.unique(xv)) < 2 or len(np.unique(yv)) < 2:
        return float("nan")
    return float(stats.spearmanr(xv, yv).statistic)


def linear_slope(x: pd.Series, y: pd.Series) -> float:
    mask = np.isfinite(x.to_numpy(dtype=float)) & np.isfinite(y.to_numpy(dtype=float))
    xv = x.to_numpy(dtype=float)[mask]
    yv = y.to_numpy(dtype=float)[mask]
    if len(xv) < 3 or float(np.var(xv)) == 0.0:
        return float("nan")
    return float(np.polyfit(xv, yv, 1)[0])


def transformed_effect(effect: float, unit: str) -> float:
    return math.exp(effect) if unit == "ratio" else effect


def residualize_fixed_effects(
    frame: pd.DataFrame,
    values: pd.Series | np.ndarray,
    categorical_columns: tuple[str, ...] = ("configuration_id", "dataset"),
) -> np.ndarray:
    """Project values off an intercept and categorical nuisance effects."""

    matrices: list[np.ndarray] = [np.ones((len(frame), 1), dtype=float)]
    for column in categorical_columns:
        dummies = pd.get_dummies(frame[column], drop_first=True, dtype=float)
        if dummies.shape[1]:
            matrices.append(dummies.to_numpy(dtype=float))
    design = np.column_stack(matrices)
    y = np.asarray(values, dtype=float)
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    return y - design @ coefficients


def fixed_effect_slope(frame: pd.DataFrame, predictor: pd.Series, outcome: pd.Series) -> float:
    x_residual = residualize_fixed_effects(frame, predictor)
    y_residual = residualize_fixed_effects(frame, outcome)
    denominator = float(x_residual @ x_residual)
    if denominator <= 1e-15:
        return float("nan")
    return float((x_residual @ y_residual) / denominator)


def load_sequence() -> tuple[pd.DataFrame, float]:
    runs = pd.read_csv(RUN_TABLE)
    required = {
        "log_start_gpu_or_tj_temp_C",
        "log_end_gpu_or_tj_temp_C",
        "log_time_weighted_mean_gpu_or_tj_temp_C",
    }
    missing = sorted(required - set(runs.columns))
    if missing:
        raise RuntimeError(
            "Run revised/analysis/full_audit.py before this analysis; missing columns: "
            + ", ".join(missing)
        )
    if len(runs) != 558 or runs[["dataset", "prefix"]].duplicated().any():
        raise RuntimeError("Expected 558 unique archived runs")

    runs["logged_window_throughput_img_s"] = runs.total_images / runs.log_wall_span_s

    events = pd.read_csv(EVENT_TABLE)
    running = events[events.status.eq("running")].copy()
    done = events[events.status.eq("done")].copy()
    if len(running) != len(runs) or len(done) != len(runs):
        raise RuntimeError("Expected one running and one done event per archived run")
    if running[["dataset", "run_id"]].duplicated().any() or done[["dataset", "run_id"]].duplicated().any():
        raise RuntimeError("Duplicate running or done event")

    running = running.sort_values(["dataset", "line"]).copy()
    running["sequence_index"] = running.groupby("dataset", sort=False).cumcount() + 1
    running["campaign_run_count"] = running.groupby("dataset", sort=False).run_id.transform("size")
    running["sequence_fraction"] = np.where(
        running.campaign_run_count > 1,
        (running.sequence_index - 1) / (running.campaign_run_count - 1),
        0.0,
    )
    running = running.rename(columns={"run_id": "prefix", "ts": "state_running_ts"})
    done = done[["dataset", "run_id", "ts", "duration_s"]].rename(
        columns={
            "run_id": "prefix",
            "ts": "state_done_ts",
            "duration_s": "state_runner_duration_s",
        }
    )
    sequence = running.merge(done, on=["dataset", "prefix"], validate="one_to_one")
    sequence = sequence.merge(runs, on=["dataset", "prefix"], validate="one_to_one")
    sequence = sequence.sort_values(["dataset", "sequence_index"]).reset_index(drop=True)

    previous_columns = [
        "prefix",
        "model",
        "precision",
        "profile",
        "batch",
        "state_done_ts",
        "state_runner_duration_s",
        "log_wall_span_s",
        "max_gpu_or_tj_temp_C",
        "log_end_gpu_or_tj_temp_C",
        "time_weighted_avg_power_W",
    ]
    grouped = sequence.groupby("dataset", sort=False)
    for column in previous_columns:
        sequence[f"predecessor_{column}"] = grouped[column].shift(1)

    sequence["state_event_gap_s"] = sequence.state_running_ts - sequence.predecessor_state_done_ts
    observed_gaps = sequence.state_event_gap_s.dropna()
    if (observed_gaps < -1e-6).any():
        raise RuntimeError("A successor running event precedes its predecessor done event")

    primary_duration_q75 = float(
        runs.loc[runs.dataset.isin(PRIMARY_BLOCKS), "log_wall_span_s"].quantile(0.75)
    )
    sequence["predecessor_is_fp64"] = sequence.predecessor_precision.eq("fp64")
    sequence["predecessor_in_duration_upper_quartile"] = (
        sequence.predecessor_log_wall_span_s >= primary_duration_q75
    )
    sequence["successor_start_minus_predecessor_end_temp_C"] = (
        sequence.log_start_gpu_or_tj_temp_C - sequence.predecessor_log_end_gpu_or_tj_temp_C
    )

    primary_sm = sequence[
        sequence.dataset.isin(PRIMARY_BLOCKS) & sequence.profile.isin(["slow", "medium"])
    ].copy()
    if len(primary_sm) != 252:
        raise RuntimeError(f"Expected 252 slow/medium primary runs, found {len(primary_sm)}")
    primary_sm["configuration_id"] = (
        primary_sm.model.astype(str)
        + "|"
        + primary_sm.precision.astype(str)
        + "|B"
        + primary_sm.batch.astype(int).astype(str)
        + "|"
        + primary_sm.profile.astype(str)
    )
    if not (primary_sm.groupby("configuration_id").size() == 3).all():
        raise RuntimeError("Primary slow/medium configuration grid is not balanced across three blocks")

    for spec in OUTCOMES:
        values = primary_sm[spec["column"]].astype(float)
        if spec["transform"] == "log":
            if (values <= 0).any():
                raise RuntimeError(f"Nonpositive value for log outcome {spec['name']}")
            analysis_values = np.log(values)
        else:
            analysis_values = values
        primary_sm[f"analysis_value__{spec['name']}"] = analysis_values
        grand = float(analysis_values.mean())
        config_mean = primary_sm.groupby("configuration_id")[
            f"analysis_value__{spec['name']}"
        ].transform("mean")
        block_mean = primary_sm.groupby("dataset")[
            f"analysis_value__{spec['name']}"
        ].transform("mean")
        primary_sm[f"adjusted_residual__{spec['name']}"] = (
            analysis_values - config_mean - block_mean + grand
        )

    residual_columns = [
        "dataset",
        "prefix",
        "configuration_id",
        *[f"adjusted_residual__{spec['name']}" for spec in OUTCOMES],
    ]
    sequence = sequence.merge(
        primary_sm[residual_columns], on=["dataset", "prefix"], how="left", validate="one_to_one"
    )
    return sequence, primary_duration_q75


def state_event_gap_summary(sequence: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups: list[tuple[str, pd.DataFrame]] = [
        (name, group) for name, group in sequence.groupby("dataset", sort=False)
    ]
    groups.append(("primary_combined", sequence[sequence.dataset.isin(PRIMARY_BLOCKS)]))
    for name, group in groups:
        gaps = group.state_event_gap_s.dropna().astype(float)
        rows.append(
            {
                "scope": name,
                "n_successor_transitions": len(gaps),
                "median_gap_s": float(gaps.median()),
                "p95_gap_s": float(gaps.quantile(0.95)),
                "maximum_gap_s": float(gaps.max()),
                "interpretation": (
                    "gap from archived predecessor done event to successor running event; "
                    "does not measure unmarked waits inside the successor runner"
                ),
            }
        )
    return pd.DataFrame(rows)


def successor_binary_effects(sequence: pd.DataFrame) -> pd.DataFrame:
    pairs = sequence[
        sequence.dataset.isin(PRIMARY_BLOCKS)
        & sequence.profile.isin(["slow", "medium"])
        & sequence.predecessor_prefix.notna()
    ].copy()
    exposures = [
        ("predecessor_fp64", "predecessor_is_fp64"),
        (
            "predecessor_duration_upper_quartile",
            "predecessor_in_duration_upper_quartile",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for exposure_name, exposure_column in exposures:
        for spec in OUTCOMES:
            analysis = pairs[
                ["dataset", "configuration_id", exposure_column, spec["column"]]
            ].dropna().copy()
            outcome = analysis[spec["column"]].astype(float)
            if spec["transform"] == "log":
                if (outcome <= 0).any():
                    raise RuntimeError(f"Nonpositive value for log outcome {spec['name']}")
                outcome = np.log(outcome)
            predictor = analysis[exposure_column].astype(float)
            coefficient = fixed_effect_slope(analysis, predictor, outcome)
            if not np.isfinite(coefficient):
                raise RuntimeError(
                    f"Exposure {exposure_name} is not identifiable for {spec['name']}"
                )

            leave_one_block_out: list[float] = []
            board_counts: list[str] = []
            for board in PRIMARY_BLOCKS:
                retained = analysis[~analysis.dataset.eq(board)].copy()
                retained_outcome = retained[spec["column"]].astype(float)
                if spec["transform"] == "log":
                    retained_outcome = np.log(retained_outcome)
                retained_coefficient = fixed_effect_slope(
                    retained,
                    retained[exposure_column].astype(float),
                    retained_outcome,
                )
                if not np.isfinite(retained_coefficient):
                    raise RuntimeError(
                        f"Exposure {exposure_name} is not identifiable after omitting {board}"
                    )
                leave_one_block_out.append(retained_coefficient)

                group = analysis[analysis.dataset.eq(board)]
                exposed_count = int(group[exposure_column].astype(bool).sum())
                reference_count = int((~group[exposure_column].astype(bool)).sum())
                board_counts.append(f"{board}:{exposed_count}/{reference_count}")

            transformed_lobo = [
                transformed_effect(value, spec["unit"]) for value in leave_one_block_out
            ]
            exposed_total = int(analysis[exposure_column].astype(bool).sum())
            reference_total = int((~analysis[exposure_column].astype(bool)).sum())
            informative_configurations = int(
                analysis.groupby("configuration_id")[exposure_column]
                .nunique()
                .gt(1)
                .sum()
            )
            rows.append(
                {
                    "scope": "primary_successors_slow_medium_only",
                    "exposure": exposure_name,
                    "outcome": spec["name"],
                    "outcome_scale": spec["unit"],
                    "n_successors": len(analysis),
                    "n_exposed": exposed_total,
                    "n_reference": reference_total,
                    "n_informative_configurations": informative_configurations,
                    "n_nominal_blocks": len(PRIMARY_BLOCKS),
                    "fixed_effect_coefficient_analysis_scale": coefficient,
                    "reported_effect_exposed_vs_reference": transformed_effect(
                        coefficient, spec["unit"]
                    ),
                    "leave_one_block_out_effect_min_reported_scale": float(
                        min(transformed_lobo)
                    ),
                    "leave_one_block_out_effect_max_reported_scale": float(
                        max(transformed_lobo)
                    ),
                    "leave_one_block_out_effects_reported_scale": "|".join(
                        f"omit_{board}:{value:.9g}"
                        for board, value in zip(PRIMARY_BLOCKS, transformed_lobo)
                    ),
                    "board_exposed_reference_counts": "|".join(board_counts),
                    "interpretation": (
                        "configuration- and physical-board-fixed-effect ratio; range omits one physical board at a time"
                        if spec["unit"] == "ratio"
                        else "configuration- and physical-board-fixed-effect difference in degrees C; range omits one physical board at a time"
                    ),
                }
            )
    return pd.DataFrame(rows)


def successor_continuous_associations(sequence: pd.DataFrame) -> pd.DataFrame:
    pairs = sequence[
        sequence.dataset.isin(PRIMARY_BLOCKS)
        & sequence.profile.isin(["slow", "medium"])
        & sequence.predecessor_prefix.notna()
    ].copy()
    pairs["log_predecessor_duration_s"] = np.log(pairs.predecessor_log_wall_span_s)
    predictors = [
        ("log_predecessor_duration_s", "log_predecessor_duration_s"),
        ("predecessor_max_temperature_C", "predecessor_max_gpu_or_tj_temp_C"),
    ]
    rows: list[dict[str, Any]] = []
    for predictor_name, predictor_column in predictors:
        for spec in OUTCOMES:
            analysis = pairs[
                ["dataset", "configuration_id", predictor_column, spec["column"]]
            ].dropna().copy().reset_index(drop=True)
            predictor = analysis[predictor_column].astype(float)
            outcome = analysis[spec["column"]].astype(float)
            if spec["transform"] == "log":
                if (outcome <= 0).any():
                    raise RuntimeError(f"Nonpositive value for log outcome {spec['name']}")
                outcome = np.log(outcome)
            predictor_residual = pd.Series(
                residualize_fixed_effects(analysis, predictor), index=analysis.index
            )
            outcome_residual = pd.Series(
                residualize_fixed_effects(analysis, outcome), index=analysis.index
            )
            board_rhos = [
                finite_spearman(
                    predictor_residual[analysis.dataset.eq(board)],
                    outcome_residual[analysis.dataset.eq(board)],
                )
                for board in PRIMARY_BLOCKS
            ]
            finite_board_rhos = [value for value in board_rhos if np.isfinite(value)]
            rows.append(
                {
                    "scope": "primary_successors_slow_medium_only",
                    "predictor": predictor_name,
                    "outcome": spec["name"],
                    "n_successors": len(analysis),
                    "pooled_fixed_effect_residual_spearman_rho": finite_spearman(
                        predictor_residual, outcome_residual
                    ),
                    "median_board_spearman_rho": float(np.median(finite_board_rhos)),
                    "minimum_board_spearman_rho": float(min(finite_board_rhos)),
                    "maximum_board_spearman_rho": float(max(finite_board_rhos)),
                    "board_rhos": "|".join(
                        f"{board}:{rho:.9g}" for board, rho in zip(PRIMARY_BLOCKS, board_rhos)
                    ),
                    "interpretation": (
                        "descriptive rank association after residualizing both predictor and successor outcome for configuration and physical-board block; no p value or causal claim"
                    ),
                }
            )
    return pd.DataFrame(rows)


def temperature_transition_summary(sequence: pd.DataFrame) -> pd.DataFrame:
    pairs = sequence[
        sequence.dataset.isin(PRIMARY_BLOCKS)
        & sequence.profile.isin(["slow", "medium"])
        & sequence.predecessor_prefix.notna()
    ].copy()
    exposures = [
        ("predecessor_fp64", "predecessor_is_fp64"),
        (
            "predecessor_duration_upper_quartile",
            "predecessor_in_duration_upper_quartile",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for exposure_name, exposure_column in exposures:
        for level, label in [(True, "exposed"), (False, "reference")]:
            group = pairs[pairs[exposure_column].astype(bool).eq(level)]
            delta = group.successor_start_minus_predecessor_end_temp_C.dropna()
            rows.append(
                {
                    "scope": "primary_successors_slow_medium_only",
                    "exposure": exposure_name,
                    "level": label,
                    "n_transitions": len(delta),
                    "median_predecessor_end_temperature_C": float(
                        group.predecessor_log_end_gpu_or_tj_temp_C.median()
                    ),
                    "median_successor_start_temperature_C": float(
                        group.log_start_gpu_or_tj_temp_C.median()
                    ),
                    "median_successor_start_minus_predecessor_end_C": float(delta.median()),
                    "p25_successor_start_minus_predecessor_end_C": float(delta.quantile(0.25)),
                    "p75_successor_start_minus_predecessor_end_C": float(delta.quantile(0.75)),
                    "interpretation": (
                        "raw boundary-temperature continuity; start/end samples are coarse internal telemetry and not ambient temperature"
                    ),
                }
            )
    return pd.DataFrame(rows)


def temporal_drift_summary(sequence: pd.DataFrame) -> pd.DataFrame:
    primary = sequence[
        sequence.dataset.isin(PRIMARY_BLOCKS) & sequence.profile.isin(["slow", "medium"])
    ].copy()
    rows: list[dict[str, Any]] = []
    for spec in OUTCOMES:
        residual_column = f"adjusted_residual__{spec['name']}"
        analysis = primary[
            [
                "dataset",
                "configuration_id",
                "sequence_fraction",
                residual_column,
                spec["column"],
            ]
        ].dropna().copy().reset_index(drop=True)
        outcome = analysis[spec["column"]].astype(float)
        if spec["transform"] == "log":
            if (outcome <= 0).any():
                raise RuntimeError(f"Nonpositive value for log outcome {spec['name']}")
            outcome = np.log(outcome)
        predictor = analysis.sequence_fraction.astype(float)
        slope = fixed_effect_slope(analysis, predictor, outcome)
        predictor_residual = pd.Series(
            residualize_fixed_effects(analysis, predictor), index=analysis.index
        )
        outcome_residual = pd.Series(
            residualize_fixed_effects(analysis, outcome), index=analysis.index
        )

        leave_one_block_out: list[float] = []
        board_rhos: list[float] = []
        board_late_early: list[float] = []
        for board in PRIMARY_BLOCKS:
            retained = analysis[~analysis.dataset.eq(board)].copy()
            retained_outcome = retained[spec["column"]].astype(float)
            if spec["transform"] == "log":
                retained_outcome = np.log(retained_outcome)
            leave_one_block_out.append(
                fixed_effect_slope(
                    retained,
                    retained.sequence_fraction.astype(float),
                    retained_outcome,
                )
            )

            board_mask = analysis.dataset.eq(board)
            rho = finite_spearman(
                predictor_residual[board_mask], outcome_residual[board_mask]
            )
            group = analysis[board_mask]
            early = group[group.sequence_fraction <= 0.25][residual_column]
            late = group[group.sequence_fraction >= 0.75][residual_column]
            late_early = float(late.mean() - early.mean())
            board_rhos.append(rho)
            board_late_early.append(late_early)

        mean_late_early = float(np.mean(board_late_early))
        transformed_lobo = [
            transformed_effect(value, spec["unit"]) for value in leave_one_block_out
        ]
        rows.append(
            {
                "scope": "primary_slow_medium_configuration_and_block_centred",
                "outcome": spec["name"],
                "outcome_scale": spec["unit"],
                "n_runs": len(analysis),
                "n_nominal_blocks": 3,
                "fixed_effect_slope_per_full_sequence": slope,
                "reported_end_vs_start_effect": transformed_effect(slope, spec["unit"]),
                "leave_one_block_out_end_vs_start_effect_min": float(
                    min(transformed_lobo)
                ),
                "leave_one_block_out_end_vs_start_effect_max": float(
                    max(transformed_lobo)
                ),
                "leave_one_block_out_effects_reported_scale": "|".join(
                    f"omit_{board}:{value:.9g}"
                    for board, value in zip(PRIMARY_BLOCKS, transformed_lobo)
                ),
                "pooled_fixed_effect_residual_spearman_rho": finite_spearman(
                    predictor_residual, outcome_residual
                ),
                "median_board_spearman_rho": float(np.median(board_rhos)),
                "minimum_board_spearman_rho": float(min(board_rhos)),
                "maximum_board_spearman_rho": float(max(board_rhos)),
                "reported_late_vs_early_quartile_effect": transformed_effect(
                    mean_late_early, spec["unit"]
                ),
                "leave_one_block_out_slopes_analysis_scale": "|".join(
                    f"omit_{board}:{value:.9g}"
                    for board, value in zip(PRIMARY_BLOCKS, leave_one_block_out)
                ),
                "board_spearman_rhos": "|".join(
                    f"{board}:{value:.9g}" for board, value in zip(PRIMARY_BLOCKS, board_rhos)
                ),
                "interpretation": (
                    "descriptive sequence slope with configuration and physical-board fixed effects; range omits one physical board at a time; no repeated anchor or causal drift estimate"
                ),
            }
        )

    fixed = sequence[sequence.dataset.eq("fixed_2026-03")].copy()
    random_campaign = sequence[sequence.dataset.eq("random_2026-03")].copy()
    keys = ["prefix", "model", "precision", "profile", "batch"]
    historical = random_campaign.merge(
        fixed,
        on=keys,
        suffixes=("_random", "_ordered"),
        validate="one_to_one",
    )
    if len(historical) != 54:
        raise RuntimeError("Expected 54 matched historical configurations")
    for spec in OUTCOMES:
        random_values = historical[f"{spec['column']}_random"].astype(float)
        ordered_values = historical[f"{spec['column']}_ordered"].astype(float)
        if spec["transform"] == "log":
            adjusted = np.log(random_values / ordered_values)
        else:
            adjusted = random_values - ordered_values
        x = historical.sequence_fraction_random
        slope = linear_slope(x, adjusted)
        rows.append(
            {
                "scope": "historical_randomized_minus_ordered_matched_diagnostic",
                "outcome": spec["name"],
                "outcome_scale": spec["unit"],
                "n_runs": len(historical),
                "n_nominal_blocks": 1,
                "fixed_effect_slope_per_full_sequence": slope,
                "reported_end_vs_start_effect": transformed_effect(slope, spec["unit"]),
                "leave_one_block_out_end_vs_start_effect_min": np.nan,
                "leave_one_block_out_end_vs_start_effect_max": np.nan,
                "leave_one_block_out_effects_reported_scale": "not_applicable",
                "pooled_fixed_effect_residual_spearman_rho": finite_spearman(x, adjusted),
                "median_board_spearman_rho": finite_spearman(x, adjusted),
                "minimum_board_spearman_rho": np.nan,
                "maximum_board_spearman_rho": np.nan,
                "reported_late_vs_early_quartile_effect": np.nan,
                "leave_one_block_out_slopes_analysis_scale": "not_applicable",
                "board_spearman_rhos": "not_applicable",
                "interpretation": (
                    "configuration-matched randomized/ordered campaign difference versus randomized position; campaigns differ in date, EMC, temperature, and realized clocks, so this is not an isolated drift estimate"
                ),
            }
        )
    return pd.DataFrame(rows)


def select_row(table: pd.DataFrame, **filters: str) -> pd.Series:
    selected = table.copy()
    for column, value in filters.items():
        selected = selected[selected[column].eq(value)]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one row for {filters}, found {len(selected)}")
    return selected.iloc[0]


def write_markdown_and_tex(
    sequence: pd.DataFrame,
    threshold_s: float,
    gaps: pd.DataFrame,
    binary: pd.DataFrame,
    associations: pd.DataFrame,
    thermal: pd.DataFrame,
    drift: pd.DataFrame,
) -> None:
    primary_gap = select_row(gaps, scope="primary_combined")
    fp64_forward = select_row(binary, exposure="predecessor_fp64", outcome="forward_throughput")
    fp64_energy = select_row(
        binary,
        exposure="predecessor_fp64",
        outcome="logged_energy_per_evaluated_image",
    )
    fp64_start_temp = select_row(binary, exposure="predecessor_fp64", outcome="start_temperature")
    long_forward = select_row(
        binary,
        exposure="predecessor_duration_upper_quartile",
        outcome="forward_throughput",
    )
    long_energy = select_row(
        binary,
        exposure="predecessor_duration_upper_quartile",
        outcome="logged_energy_per_evaluated_image",
    )
    long_start_temp = select_row(
        binary,
        exposure="predecessor_duration_upper_quartile",
        outcome="start_temperature",
    )
    duration_forward_assoc = select_row(
        associations,
        predictor="log_predecessor_duration_s",
        outcome="forward_throughput",
    )
    duration_energy_assoc = select_row(
        associations,
        predictor="log_predecessor_duration_s",
        outcome="logged_energy_per_evaluated_image",
    )
    primary_forward_drift = select_row(
        drift,
        scope="primary_slow_medium_configuration_and_block_centred",
        outcome="forward_throughput",
    )
    primary_energy_drift = select_row(
        drift,
        scope="primary_slow_medium_configuration_and_block_centred",
        outcome="logged_energy_per_evaluated_image",
    )
    primary_start_temp_drift = select_row(
        drift,
        scope="primary_slow_medium_configuration_and_block_centred",
        outcome="start_temperature",
    )
    historical_forward = select_row(
        drift,
        scope="historical_randomized_minus_ordered_matched_diagnostic",
        outcome="forward_throughput",
    )
    historical_energy = select_row(
        drift,
        scope="historical_randomized_minus_ordered_matched_diagnostic",
        outcome="logged_energy_per_evaluated_image",
    )
    fp64_thermal = select_row(thermal, exposure="predecessor_fp64", level="exposed")
    ref_thermal = select_row(thermal, exposure="predecessor_fp64", level="reference")

    markdown = f"""# Long-Run, Successor, and Temporal-Drift Diagnostics

**Generated by:** `revised/analysis/long_run_drift_analysis.py`  
**Analysis unit:** archived run or adjacent run pair  
**Primary conservative scope:** Clone1--3 successors requested as slow or medium; requested-fast successors are excluded because their realized clock regime is already deterministically predecessor-dependent.

## 1. What can and cannot be identified

The archive contains one observation per physical-board × configuration cell and no repeated reference configuration, cooldown marker, or pre-specified stabilization criterion. Therefore these diagnostics cannot prove absence of drift, cannot identify a causal thermal or FP64-successor effect, and cannot provide a population estimate for arbitrary Jetson modules. No p values are reported.

Primary estimates use fixed effects for the successor configuration and physical-board block. Ratio outcomes are analysed on the log scale and exponentiated; temperature effects are adjusted differences in degrees Celsius. Robustness ranges refit the same model after omitting one physical board at a time. These leave-one-board-out ranges are sensitivity analyses, not confidence intervals.

## 2. Archived inter-run state-event gaps

Across the {int(primary_gap.n_successor_transitions)} adjacent transitions in Clone1--3, the median gap from a predecessor `done` event to the successor `running` event is {1000*primary_gap.median_gap_s:.3f} ms, the 95th percentile is {1000*primary_gap.p95_gap_s:.3f} ms, and the maximum is {1000*primary_gap.maximum_gap_s:.3f} ms. Thus, the archived wrapper chronology contains no explicit between-run rest period. These event gaps do not exclude an unmarked wait or profile-setting interval inside the successor runner.

## 3. Immediate successors of FP64 and upper-duration-quartile runs

The upper duration quartile in the 378-run primary grid begins at {threshold_s:.2f} s ({threshold_s/3600:.3f} h). The table reports fixed-effect exposed/reference estimates for slow/medium successors, controlling for successor configuration and physical-board block.

| Predecessor exposure | Successor outcome | Effect | Leave-one-block-out range | Exposed/reference successors |
|---|---|---:|---:|---:|
| FP64 | Forward throughput | {fp64_forward.reported_effect_exposed_vs_reference:.4f}× | {fp64_forward.leave_one_block_out_effect_min_reported_scale:.4f} to {fp64_forward.leave_one_block_out_effect_max_reported_scale:.4f}× | {int(fp64_forward.n_exposed)}/{int(fp64_forward.n_reference)} |
| FP64 | Logged energy/image | {fp64_energy.reported_effect_exposed_vs_reference:.4f}× | {fp64_energy.leave_one_block_out_effect_min_reported_scale:.4f} to {fp64_energy.leave_one_block_out_effect_max_reported_scale:.4f}× | {int(fp64_energy.n_exposed)}/{int(fp64_energy.n_reference)} |
| FP64 | Start temperature | {fp64_start_temp.reported_effect_exposed_vs_reference:+.3f} °C | {fp64_start_temp.leave_one_block_out_effect_min_reported_scale:+.3f} to {fp64_start_temp.leave_one_block_out_effect_max_reported_scale:+.3f} °C | {int(fp64_start_temp.n_exposed)}/{int(fp64_start_temp.n_reference)} |
| Duration upper quartile | Forward throughput | {long_forward.reported_effect_exposed_vs_reference:.4f}× | {long_forward.leave_one_block_out_effect_min_reported_scale:.4f} to {long_forward.leave_one_block_out_effect_max_reported_scale:.4f}× | {int(long_forward.n_exposed)}/{int(long_forward.n_reference)} |
| Duration upper quartile | Logged energy/image | {long_energy.reported_effect_exposed_vs_reference:.4f}× | {long_energy.leave_one_block_out_effect_min_reported_scale:.4f} to {long_energy.leave_one_block_out_effect_max_reported_scale:.4f}× | {int(long_energy.n_exposed)}/{int(long_energy.n_reference)} |
| Duration upper quartile | Start temperature | {long_start_temp.reported_effect_exposed_vs_reference:+.3f} °C | {long_start_temp.leave_one_block_out_effect_min_reported_scale:+.3f} to {long_start_temp.leave_one_block_out_effect_max_reported_scale:+.3f} °C | {int(long_start_temp.n_exposed)}/{int(long_start_temp.n_reference)} |

For continuous predecessor duration, the median board-specific Spearman association is {duration_forward_assoc.median_board_spearman_rho:+.3f} for adjusted successor forward throughput (board range {duration_forward_assoc.minimum_board_spearman_rho:+.3f} to {duration_forward_assoc.maximum_board_spearman_rho:+.3f}) and {duration_energy_assoc.median_board_spearman_rho:+.3f} for adjusted successor logged energy (range {duration_energy_assoc.minimum_board_spearman_rho:+.3f} to {duration_energy_assoc.maximum_board_spearman_rho:+.3f}).

The raw boundary-temperature diagnostic gives a median successor-start minus predecessor-end temperature of {fp64_thermal.median_successor_start_minus_predecessor_end_C:+.3f} °C after FP64 predecessors and {ref_thermal.median_successor_start_minus_predecessor_end_C:+.3f} °C after non-FP64 predecessors. These are coarse internal-telemetry boundary samples, not ambient-temperature or calibrated thermal-state measurements.

## 4. Configuration- and physical-board-adjusted temporal drift

Across the three primary slow/medium grids, the fixed-effect linear end-versus-start factor is {primary_forward_drift.reported_end_vs_start_effect:.4f}× for forward throughput (leave-one-board-out range {primary_forward_drift.leave_one_block_out_end_vs_start_effect_min:.4f} to {primary_forward_drift.leave_one_block_out_end_vs_start_effect_max:.4f}×), {primary_energy_drift.reported_end_vs_start_effect:.4f}× for logged energy/image (range {primary_energy_drift.leave_one_block_out_end_vs_start_effect_min:.4f} to {primary_energy_drift.leave_one_block_out_end_vs_start_effect_max:.4f}×), and {primary_start_temp_drift.reported_end_vs_start_effect:+.3f} °C for start temperature (range {primary_start_temp_drift.leave_one_block_out_end_vs_start_effect_min:+.3f} to {primary_start_temp_drift.leave_one_block_out_end_vs_start_effect_max:+.3f} °C). The ranges are sensitivity analyses, not confidence intervals; consult `long_run_temporal_drift_summary.csv` for all slopes and rank correlations.

For the single historical randomized campaign, configuration-matched randomized/ordered differences have Spearman rho {historical_forward.median_board_spearman_rho:+.3f} with randomized position for forward throughput and {historical_energy.median_board_spearman_rho:+.3f} for logged energy/image. These historical diagnostics remain confounded by campaign date, EMC state, temperature, and realized clocks and are not an isolated long-term-drift estimate.

## 5. Conservative conclusion

The archive establishes that wrapper-level transitions were effectively immediate and that one strong successor mechanism exists for requested-fast clock realization. For slow/medium successors, the effect-size tables quantify associations with FP64 and long predecessors, but the three board estimates and temporal slopes are not sufficient to exclude smaller drift, distinguish thermal from software/background-state effects, or establish causality. A confirmatory design still requires repeated reference configurations, explicit stabilization/washout, synchronized phase markers, verified device identities, and randomized temporal blocks.

## Machine-readable outputs

- `long_run_successor_pairs.csv`: complete adjacent-pair chronology and adjusted primary residuals;
- `long_run_state_event_gaps.csv`: campaign-specific archived wrapper gaps;
- `long_run_successor_binary_effects.csv`: FP64 and upper-duration-quartile successor effects;
- `long_run_successor_continuous_associations.csv`: board-specific predecessor-duration/temperature rank associations;
- `long_run_successor_temperature_transitions.csv`: raw end-to-start temperature continuity;
- `long_run_temporal_drift_summary.csv`: primary adjusted slopes and historical matched diagnostics;
- `long_run_drift_results.tex`: generated macros for a response or supplement.
"""
    MARKDOWN_OUT.write_text(markdown, encoding="utf-8")

    macros = {
        "PrimaryDurationUpperQuartileSeconds": f"{threshold_s:.2f}",
        "PrimaryDurationUpperQuartileHours": f"{threshold_s/3600:.3f}",
        "PrimaryAdjacentTransitionCount": f"{int(primary_gap.n_successor_transitions)}",
        "PrimaryStateEventGapMedianMilliseconds": f"{1000*primary_gap.median_gap_s:.3f}",
        "PrimaryStateEventGapPninetyfiveMilliseconds": f"{1000*primary_gap.p95_gap_s:.3f}",
        "PrimaryStateEventGapMaximumMilliseconds": f"{1000*primary_gap.maximum_gap_s:.3f}",
        "FpSixtyFourSuccessorForwardRatio": f"{fp64_forward.reported_effect_exposed_vs_reference:.4f}",
        "FpSixtyFourSuccessorForwardRatioLoboMinimum": f"{fp64_forward.leave_one_block_out_effect_min_reported_scale:.4f}",
        "FpSixtyFourSuccessorForwardRatioLoboMaximum": f"{fp64_forward.leave_one_block_out_effect_max_reported_scale:.4f}",
        "FpSixtyFourSuccessorEnergyRatio": f"{fp64_energy.reported_effect_exposed_vs_reference:.4f}",
        "FpSixtyFourSuccessorEnergyRatioLoboMinimum": f"{fp64_energy.leave_one_block_out_effect_min_reported_scale:.4f}",
        "FpSixtyFourSuccessorEnergyRatioLoboMaximum": f"{fp64_energy.leave_one_block_out_effect_max_reported_scale:.4f}",
        "FpSixtyFourSuccessorStartTemperatureDeltaC": f"{fp64_start_temp.reported_effect_exposed_vs_reference:.3f}",
        "FpSixtyFourSuccessorStartTemperatureDeltaCLoboMinimum": f"{fp64_start_temp.leave_one_block_out_effect_min_reported_scale:.3f}",
        "FpSixtyFourSuccessorStartTemperatureDeltaCLoboMaximum": f"{fp64_start_temp.leave_one_block_out_effect_max_reported_scale:.3f}",
        "FpSixtyFourSuccessorExposedCount": f"{int(fp64_forward.n_exposed)}",
        "FpSixtyFourSuccessorReferenceCount": f"{int(fp64_forward.n_reference)}",
        "LongPredecessorSuccessorForwardRatio": f"{long_forward.reported_effect_exposed_vs_reference:.4f}",
        "LongPredecessorSuccessorForwardRatioLoboMinimum": f"{long_forward.leave_one_block_out_effect_min_reported_scale:.4f}",
        "LongPredecessorSuccessorForwardRatioLoboMaximum": f"{long_forward.leave_one_block_out_effect_max_reported_scale:.4f}",
        "LongPredecessorSuccessorEnergyRatio": f"{long_energy.reported_effect_exposed_vs_reference:.4f}",
        "LongPredecessorSuccessorEnergyRatioLoboMinimum": f"{long_energy.leave_one_block_out_effect_min_reported_scale:.4f}",
        "LongPredecessorSuccessorEnergyRatioLoboMaximum": f"{long_energy.leave_one_block_out_effect_max_reported_scale:.4f}",
        "LongPredecessorSuccessorStartTemperatureDeltaC": f"{long_start_temp.reported_effect_exposed_vs_reference:.3f}",
        "LongPredecessorSuccessorStartTemperatureDeltaCLoboMinimum": f"{long_start_temp.leave_one_block_out_effect_min_reported_scale:.3f}",
        "LongPredecessorSuccessorStartTemperatureDeltaCLoboMaximum": f"{long_start_temp.leave_one_block_out_effect_max_reported_scale:.3f}",
        "LongPredecessorSuccessorExposedCount": f"{int(long_forward.n_exposed)}",
        "LongPredecessorSuccessorReferenceCount": f"{int(long_forward.n_reference)}",
        "PrimaryForwardDriftEndStartRatio": f"{primary_forward_drift.reported_end_vs_start_effect:.4f}",
        "PrimaryForwardDriftEndStartRatioLoboMinimum": f"{primary_forward_drift.leave_one_block_out_end_vs_start_effect_min:.4f}",
        "PrimaryForwardDriftEndStartRatioLoboMaximum": f"{primary_forward_drift.leave_one_block_out_end_vs_start_effect_max:.4f}",
        "PrimaryEnergyDriftEndStartRatio": f"{primary_energy_drift.reported_end_vs_start_effect:.4f}",
        "PrimaryEnergyDriftEndStartRatioLoboMinimum": f"{primary_energy_drift.leave_one_block_out_end_vs_start_effect_min:.4f}",
        "PrimaryEnergyDriftEndStartRatioLoboMaximum": f"{primary_energy_drift.leave_one_block_out_end_vs_start_effect_max:.4f}",
        "PrimaryStartTemperatureDriftEndStartDeltaC": f"{primary_start_temp_drift.reported_end_vs_start_effect:.3f}",
        "PrimaryStartTemperatureDriftEndStartDeltaCLoboMinimum": f"{primary_start_temp_drift.leave_one_block_out_end_vs_start_effect_min:.3f}",
        "PrimaryStartTemperatureDriftEndStartDeltaCLoboMaximum": f"{primary_start_temp_drift.leave_one_block_out_end_vs_start_effect_max:.3f}",
        "HistoricalRandomForwardPositionRho": f"{historical_forward.median_board_spearman_rho:.3f}",
        "HistoricalRandomEnergyPositionRho": f"{historical_energy.median_board_spearman_rho:.3f}",
    }
    tex_lines = [
        "% Generated by analysis/long_run_drift_analysis.py; do not edit manually.",
        "% Descriptive existing-data diagnostics only; no causal or population inference.",
    ]
    tex_lines.extend(
        f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in macros.items()
    )
    TEX_OUT.write_text("\n".join(tex_lines) + "\n", encoding="utf-8")


def main() -> None:
    sequence, threshold_s = load_sequence()
    gaps = state_event_gap_summary(sequence)
    binary = successor_binary_effects(sequence)
    associations = successor_continuous_associations(sequence)
    thermal = temperature_transition_summary(sequence)
    drift = temporal_drift_summary(sequence)

    pair_columns = [
        "dataset",
        "sequence_index",
        "campaign_run_count",
        "sequence_fraction",
        "prefix",
        "model",
        "precision",
        "profile",
        "batch",
        "state_running_ts",
        "state_done_ts",
        "state_runner_duration_s",
        "log_wall_span_s",
        "log_start_gpu_or_tj_temp_C",
        "log_end_gpu_or_tj_temp_C",
        "max_gpu_or_tj_temp_C",
        "predecessor_prefix",
        "predecessor_model",
        "predecessor_precision",
        "predecessor_profile",
        "predecessor_batch",
        "predecessor_state_done_ts",
        "predecessor_state_runner_duration_s",
        "predecessor_log_wall_span_s",
        "predecessor_max_gpu_or_tj_temp_C",
        "predecessor_log_end_gpu_or_tj_temp_C",
        "state_event_gap_s",
        "predecessor_is_fp64",
        "predecessor_in_duration_upper_quartile",
        "successor_start_minus_predecessor_end_temp_C",
        "configuration_id",
        *[f"adjusted_residual__{spec['name']}" for spec in OUTCOMES],
    ]
    sequence.loc[sequence.predecessor_prefix.notna(), pair_columns].to_csv(
        PAIR_OUT, index=False
    )
    gaps.to_csv(GAP_OUT, index=False)
    binary.to_csv(BINARY_OUT, index=False)
    associations.to_csv(ASSOCIATION_OUT, index=False)
    thermal.to_csv(THERMAL_OUT, index=False)
    drift.to_csv(DRIFT_OUT, index=False)
    write_markdown_and_tex(sequence, threshold_s, gaps, binary, associations, thermal, drift)

    print(
        {
            "archived_runs": len(sequence),
            "adjacent_pairs": int(sequence.predecessor_prefix.notna().sum()),
            "primary_duration_upper_quartile_s": threshold_s,
            "binary_effect_rows": len(binary),
            "continuous_association_rows": len(associations),
            "drift_rows": len(drift),
            "out": str(AUDIT),
        }
    )


if __name__ == "__main__":
    main()
