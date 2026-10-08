#!/usr/bin/env python3
"""
MotionStage Stage 18C
Build a window-level movement-difference dataset for unsupervised ML.

The script:
- scans data/comparisons/*/final_summary.json
- follows each summary's DTW alignment output
- loads the reference/comparison motion-feature CSVs
- reconstructs aligned per-step absolute feature differences
- aggregates them into overlapping time windows
- carries leakage_group_id from Stage 18B
- marks overlap with existing deterministic divergence windows for later
  evaluation only (not as a training target)

Default outputs:
    data/ml/window_features.csv
    data/ml/window_features_manifest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REF_INDEX_CANDIDATES = (
    "reference_index",
    "reference_idx",
    "ref_index",
    "ref_idx",
    "reference_frame_index",
    "ref_frame_index",
    "reference_frame",
    "ref_frame",
)

COMP_INDEX_CANDIDATES = (
    "comparison_index",
    "comparison_idx",
    "comp_index",
    "comp_idx",
    "comparison_frame_index",
    "comp_frame_index",
    "comparison_frame",
    "comp_frame",
)

REF_TIME_CANDIDATES = (
    "reference_timestamp_seconds",
    "reference_time_seconds",
    "reference_timestamp",
    "reference_time",
    "ref_timestamp_seconds",
    "ref_time_seconds",
    "ref_timestamp",
    "ref_time",
)

COMP_TIME_CANDIDATES = (
    "comparison_timestamp_seconds",
    "comparison_time_seconds",
    "comparison_timestamp",
    "comparison_time",
    "comp_timestamp_seconds",
    "comp_time_seconds",
    "comp_timestamp",
    "comp_time",
)

FEATURE_TIME_CANDIDATES = (
    "timestamp_seconds",
    "time_seconds",
    "timestamp",
    "time",
)

FRAME_CANDIDATES = (
    "frame",
    "frame_index",
    "frame_number",
)

EXCLUDED_FEATURE_TOKENS = (
    "frame",
    "index",
    "timestamp",
    "time",
    "visibility",
    "presence",
    "confidence",
    "score",
)


def first_present(
    columns: list[str],
    candidates: tuple[str, ...],
) -> str | None:
    lowered = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def fuzzy_side_column(
    columns: list[str],
    side_tokens: tuple[str, ...],
    kind_tokens: tuple[str, ...],
) -> str | None:
    for column in columns:
        low = column.lower()
        if any(side in low for side in side_tokens) and any(
            kind in low for kind in kind_tokens
        ):
            return column
    return None


def resolve_project_path(
    project_root: Path,
    value: str | None,
    fallback: Path,
) -> Path:
    if not value:
        return fallback

    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path


def load_stage18b_groups(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    df = pd.read_csv(path)
    required = {"comparison_id", "leakage_group_id"}

    if not required.issubset(df.columns):
        return {}

    return {
        str(row.comparison_id): str(row.leakage_group_id)
        for row in df.itertuples()
    }


def feature_csv_score(path: Path) -> int:
    name = path.name.lower()

    score = 0

    if name == "motion_features.csv":
        score += 300
    if "motion" in name:
        score += 120
    if "feature" in name:
        score += 100
    if "smooth" in name:
        score += 15

    if "pose3d" in name:
        score -= 500
    if "world" in name:
        score -= 300
    if "landmark" in name:
        score -= 250
    if "keypoint" in name:
        score -= 200
    if "tracking" in name:
        score -= 100
    if "summary" in name:
        score -= 100

    return score


def find_feature_csv(
    processed_root: Path,
    performance_name: str,
) -> Path:
    directory = processed_root / performance_name

    if not directory.exists():
        raise FileNotFoundError(
            f"Processed performance directory not found: {directory}"
        )

    preferred = (
        "motion_features.csv",
        "smoothed_motion_features.csv",
        "motion_features_smoothed.csv",
        "features.csv",
    )

    for filename in preferred:
        path = directory / filename
        if path.exists():
            return path

    candidates = sorted(
        directory.glob("*.csv"),
        key=lambda p: (feature_csv_score(p), p.name),
        reverse=True,
    )

    if not candidates:
        raise FileNotFoundError(
            f"No CSV files found in {directory}"
        )

    usable: list[tuple[int, Path]] = []

    for path in candidates:
        try:
            sample = pd.read_csv(path, nrows=25)
        except Exception:
            continue

        numeric = sample.select_dtypes(include=[np.number]).columns.tolist()

        feature_like = [
            c
            for c in numeric
            if not any(
                token in c.lower()
                for token in EXCLUDED_FEATURE_TOKENS
            )
        ]

        if len(feature_like) >= 3:
            usable.append(
                (feature_csv_score(path), path)
            )

    if not usable:
        raise RuntimeError(
            f"Could not identify a motion-feature CSV in {directory}. "
            f"Candidates: {[p.name for p in candidates]}"
        )

    usable.sort(
        key=lambda item: (item[0], item[1].name),
        reverse=True,
    )
    return usable[0][1]


def detect_alignment_columns(
    alignment: pd.DataFrame,
) -> tuple[str, str, str | None, str | None]:
    columns = alignment.columns.tolist()

    ref_index = first_present(
        columns,
        REF_INDEX_CANDIDATES,
    )
    comp_index = first_present(
        columns,
        COMP_INDEX_CANDIDATES,
    )

    if ref_index is None:
        ref_index = fuzzy_side_column(
            columns,
            ("reference", "ref"),
            ("index", "idx", "frame"),
        )

    if comp_index is None:
        comp_index = fuzzy_side_column(
            columns,
            ("comparison", "comp"),
            ("index", "idx", "frame"),
        )

    ref_time = first_present(
        columns,
        REF_TIME_CANDIDATES,
    )
    comp_time = first_present(
        columns,
        COMP_TIME_CANDIDATES,
    )

    if ref_time is None:
        ref_time = fuzzy_side_column(
            columns,
            ("reference", "ref"),
            ("timestamp", "time"),
        )

    if comp_time is None:
        comp_time = fuzzy_side_column(
            columns,
            ("comparison", "comp"),
            ("timestamp", "time"),
        )

    if ref_index is None or comp_index is None:
        raise RuntimeError(
            "Could not detect DTW reference/comparison index columns. "
            f"Alignment columns were: {columns}"
        )

    return (
        ref_index,
        comp_index,
        ref_time,
        comp_time,
    )


def feature_time_column(df: pd.DataFrame) -> str | None:
    return first_present(
        df.columns.tolist(),
        FEATURE_TIME_CANDIDATES,
    )


def feature_frame_column(df: pd.DataFrame) -> str | None:
    return first_present(
        df.columns.tolist(),
        FRAME_CANDIDATES,
    )


def resolve_feature_rows(
    df: pd.DataFrame,
    values: pd.Series,
    alignment_column: str,
) -> pd.DataFrame:
    numeric = pd.to_numeric(
        values,
        errors="coerce",
    )

    if numeric.isna().any():
        raise RuntimeError(
            f"Non-numeric values found in alignment column "
            f"{alignment_column}"
        )

    integer_values = numeric.round().astype(int).to_numpy()
    column_low = alignment_column.lower()

    frame_column = feature_frame_column(df)

    # Prefer explicit frame mapping when the alignment column is frame-based.
    if "frame" in column_low and frame_column is not None:
        lookup = (
            df.drop_duplicates(frame_column)
            .set_index(frame_column)
        )

        resolved = lookup.reindex(
            integer_values
        )

        if not resolved.isna().all(axis=1).any():
            return resolved.reset_index(drop=False)

    # Index-based DTW paths normally refer to zero-based row positions.
    if (
        len(integer_values) > 0
        and integer_values.min() >= 0
        and integer_values.max() < len(df)
    ):
        return (
            df.iloc[integer_values]
            .reset_index(drop=True)
        )

    # Fallback: try matching the values to a frame column.
    if frame_column is not None:
        lookup = (
            df.drop_duplicates(frame_column)
            .set_index(frame_column)
        )

        resolved = lookup.reindex(
            integer_values
        )

        if not resolved.isna().all(axis=1).any():
            return resolved.reset_index(drop=False)

    raise RuntimeError(
        f"Could not map alignment column {alignment_column} "
        f"onto feature rows. Feature rows={len(df)}, "
        f"alignment range={integer_values.min()}..{integer_values.max()}"
    )


def common_numeric_features(
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
) -> list[str]:
    ref_numeric = set(
        reference.select_dtypes(
            include=[np.number]
        ).columns
    )

    comp_numeric = set(
        comparison.select_dtypes(
            include=[np.number]
        ).columns
    )

    common = sorted(
        ref_numeric.intersection(
            comp_numeric
        )
    )

    features = [
        column
        for column in common
        if not any(
            token in column.lower()
            for token in EXCLUDED_FEATURE_TOKENS
        )
    ]

    return features


def safe_window(
    value: Any,
) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None

    start = value.get("start_seconds")
    end = value.get("end_seconds")

    if not isinstance(start, (int, float)):
        return None
    if not isinstance(end, (int, float)):
        return None

    return {
        "start_seconds": float(start),
        "end_seconds": float(end),
    }


def windows_overlap(
    start_a: float,
    end_a: float,
    window_b: dict[str, float] | None,
) -> bool:
    if window_b is None:
        return False

    start_b = window_b["start_seconds"]
    end_b = window_b["end_seconds"]

    return max(start_a, start_b) <= min(end_a, end_b)


def deterministic_windows(
    summary: dict[str, Any],
) -> tuple[
    dict[str, float] | None,
    dict[str, float] | None,
]:
    robust = summary.get(
        "robust_localisation",
        {},
    )

    primary = None

    if isinstance(robust, dict):
        divergence = robust.get(
            "divergence",
            {},
        )

        if isinstance(divergence, dict):
            primary = safe_window(
                divergence.get(
                    "comparison_window"
                )
            )

    if primary is None:
        overall = summary.get(
            "overall_divergence",
            {},
        )

        if isinstance(overall, dict):
            primary = safe_window(
                overall.get(
                    "comparison_window"
                )
            )

    isolated = summary.get(
        "isolated_arm_divergence",
        {},
    )

    isolated_window = None

    if isinstance(isolated, dict):
        isolated_window = safe_window(
            isolated.get(
                "comparison_window"
            )
        )

    return primary, isolated_window


def build_aligned_differences(
    alignment: pd.DataFrame,
    reference_features: pd.DataFrame,
    comparison_features: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    list[str],
    dict[str, str | None],
]:
    (
        ref_index_col,
        comp_index_col,
        ref_time_col,
        comp_time_col,
    ) = detect_alignment_columns(
        alignment
    )

    ref_rows = resolve_feature_rows(
        reference_features,
        alignment[ref_index_col],
        ref_index_col,
    )

    comp_rows = resolve_feature_rows(
        comparison_features,
        alignment[comp_index_col],
        comp_index_col,
    )

    feature_columns = common_numeric_features(
        ref_rows,
        comp_rows,
    )

    if len(feature_columns) < 3:
        raise RuntimeError(
            "Fewer than three common numeric motion features "
            f"were found. Common features: {feature_columns}"
        )

    aligned = pd.DataFrame(
        {
            "alignment_step":
                np.arange(len(alignment)),
        }
    )

    if ref_time_col is not None:
        aligned["reference_time_seconds"] = pd.to_numeric(
            alignment[ref_time_col],
            errors="coerce",
        )
    else:
        ref_feature_time = feature_time_column(
            ref_rows
        )

        if ref_feature_time is None:
            raise RuntimeError(
                "No reference time column found in either "
                "the alignment CSV or feature CSV."
            )

        aligned["reference_time_seconds"] = pd.to_numeric(
            ref_rows[ref_feature_time],
            errors="coerce",
        )

    if comp_time_col is not None:
        aligned["comparison_time_seconds"] = pd.to_numeric(
            alignment[comp_time_col],
            errors="coerce",
        )
    else:
        comp_feature_time = feature_time_column(
            comp_rows
        )

        if comp_feature_time is None:
            raise RuntimeError(
                "No comparison time column found in either "
                "the alignment CSV or feature CSV."
            )

        aligned["comparison_time_seconds"] = pd.to_numeric(
            comp_rows[comp_feature_time],
            errors="coerce",
        )

    for column in feature_columns:
        ref_values = pd.to_numeric(
            ref_rows[column],
            errors="coerce",
        ).to_numpy()

        comp_values = pd.to_numeric(
            comp_rows[column],
            errors="coerce",
        ).to_numpy()

        aligned[
            f"absdiff__{column}"
        ] = np.abs(
            ref_values - comp_values
        )

    aligned = (
        aligned
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna(
            subset=[
                "reference_time_seconds",
                "comparison_time_seconds",
            ]
        )
        .sort_values(
            "comparison_time_seconds"
        )
        .reset_index(drop=True)
    )

    detected = {
        "reference_alignment_index_column":
            ref_index_col,
        "comparison_alignment_index_column":
            comp_index_col,
        "reference_alignment_time_column":
            ref_time_col,
        "comparison_alignment_time_column":
            comp_time_col,
    }

    return (
        aligned,
        feature_columns,
        detected,
    )


def aggregate_windows(
    aligned: pd.DataFrame,
    base_features: list[str],
    comparison_id: str,
    leakage_group_id: str,
    reference_name: str,
    comparison_name: str,
    window_seconds: float,
    stride_seconds: float,
    min_window_points: int,
    primary_window: dict[str, float] | None,
    isolated_window: dict[str, float] | None,
) -> list[dict[str, Any]]:
    time = aligned[
        "comparison_time_seconds"
    ].to_numpy(dtype=float)

    if len(time) == 0:
        return []

    start_time = float(np.nanmin(time))
    end_time = float(np.nanmax(time))

    if end_time - start_time < window_seconds:
        starts = np.array([start_time])
    else:
        final_start = (
            end_time
            - window_seconds
        )

        starts = np.arange(
            start_time,
            final_start + stride_seconds * 0.5,
            stride_seconds,
        )

    rows: list[dict[str, Any]] = []

    for window_number, start in enumerate(
        starts,
        start=1,
    ):
        end = float(
            start + window_seconds
        )

        mask = (
            (
                aligned[
                    "comparison_time_seconds"
                ] >= start
            )
            & (
                aligned[
                    "comparison_time_seconds"
                ] < end
            )
        )

        current = aligned.loc[
            mask
        ].copy()

        if len(current) < min_window_points:
            continue

        reference_start = float(
            current[
                "reference_time_seconds"
            ].min()
        )

        reference_end = float(
            current[
                "reference_time_seconds"
            ].max()
        )

        row: dict[str, Any] = {
            "window_id":
                f"{comparison_id}__w{window_number:04d}",
            "comparison_id":
                comparison_id,
            "leakage_group_id":
                leakage_group_id,
            "reference_name":
                reference_name,
            "comparison_name":
                comparison_name,
            "comparison_window_start_seconds":
                round(float(start), 6),
            "comparison_window_end_seconds":
                round(float(end), 6),
            "comparison_window_mid_seconds":
                round(
                    float(
                        start
                        + window_seconds / 2.0
                    ),
                    6,
                ),
            "reference_window_start_seconds":
                round(
                    reference_start,
                    6,
                ),
            "reference_window_end_seconds":
                round(
                    reference_end,
                    6,
                ),
            "aligned_points":
                int(len(current)),
            "overlaps_primary_divergence":
                windows_overlap(
                    float(start),
                    float(end),
                    primary_window,
                ),
            "overlaps_isolated_divergence":
                windows_overlap(
                    float(start),
                    float(end),
                    isolated_window,
                ),
        }

        all_values: list[float] = []

        for feature in base_features:
            column = f"absdiff__{feature}"

            values = pd.to_numeric(
                current[column],
                errors="coerce",
            ).dropna()

            if values.empty:
                row[
                    f"{feature}__mean_absdiff"
                ] = np.nan

                row[
                    f"{feature}__std_absdiff"
                ] = np.nan

                row[
                    f"{feature}__max_absdiff"
                ] = np.nan

                continue

            array = values.to_numpy(
                dtype=float
            )

            row[
                f"{feature}__mean_absdiff"
            ] = float(
                np.mean(array)
            )

            row[
                f"{feature}__std_absdiff"
            ] = float(
                np.std(
                    array,
                    ddof=0,
                )
            )

            row[
                f"{feature}__max_absdiff"
            ] = float(
                np.max(array)
            )

            all_values.extend(
                array.tolist()
            )

        if all_values:
            pooled = np.asarray(
                all_values,
                dtype=float,
            )

            row[
                "pooled_mean_absdiff"
            ] = float(
                np.mean(pooled)
            )

            row[
                "pooled_std_absdiff"
            ] = float(
                np.std(
                    pooled,
                    ddof=0,
                )
            )

            row[
                "pooled_max_absdiff"
            ] = float(
                np.max(pooled)
            )

            row[
                "pooled_median_absdiff"
            ] = float(
                np.median(pooled)
            )

        rows.append(row)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build MotionStage Stage 18C "
            "window-level ML feature dataset."
        )
    )

    parser.add_argument(
        "--project-root",
        default=".",
    )

    parser.add_argument(
        "--comparisons-dir",
        default="data/comparisons",
    )

    parser.add_argument(
        "--processed-dir",
        default="data/processed",
    )

    parser.add_argument(
        "--stage18b-csv",
        default=(
            "data/ml/"
            "comparison_features_validated.csv"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "data/ml/"
            "window_features.csv"
        ),
    )

    parser.add_argument(
        "--manifest",
        default=(
            "data/ml/"
            "window_features_manifest.json"
        ),
    )

    parser.add_argument(
        "--window-seconds",
        type=float,
        default=0.75,
    )

    parser.add_argument(
        "--stride-seconds",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--min-window-points",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--include-duplicate-groups",
        action="store_true",
        help=(
            "Process every stored comparison instead "
            "of one representative per Stage 18B "
            "leakage group."
        ),
    )

    args = parser.parse_args()

    project_root = Path(
        args.project_root
    ).resolve()

    comparisons_dir = (
        project_root
        / args.comparisons_dir
    )

    processed_dir = (
        project_root
        / args.processed_dir
    )

    stage18b_path = (
        project_root
        / args.stage18b_csv
    )

    output_path = (
        project_root
        / args.output
    )

    manifest_path = (
        project_root
        / args.manifest
    )

    group_lookup = load_stage18b_groups(
        stage18b_path
    )

    summary_paths = sorted(
        comparisons_dir.glob(
            "*/final_summary.json"
        )
    )

    if not summary_paths:
        raise FileNotFoundError(
            f"No final_summary.json files "
            f"found under {comparisons_dir}"
        )

    seen_groups: set[str] = set()

    all_windows: list[
        dict[str, Any]
    ] = []

    processed: list[
        dict[str, Any]
    ] = []

    skipped_duplicates: list[
        dict[str, Any]
    ] = []

    errors: list[
        dict[str, Any]
    ] = []

    base_feature_union: set[str] = set()

    for summary_path in summary_paths:
        comparison_id = (
            summary_path.parent.name
        )

        leakage_group = (
            group_lookup.get(
                comparison_id,
                comparison_id,
            )
        )

        if (
            not args.include_duplicate_groups
            and leakage_group in seen_groups
        ):
            skipped_duplicates.append(
                {
                    "comparison_id":
                        comparison_id,
                    "leakage_group_id":
                        leakage_group,
                }
            )
            continue

        try:
            summary = json.loads(
                summary_path.read_text(
                    encoding="utf-8"
                )
            )

            reference_name = str(
                summary.get(
                    "reference",
                    "",
                )
            )

            comparison_name = str(
                summary.get(
                    "comparison",
                    "",
                )
            )

            if (
                not reference_name
                or not comparison_name
            ):
                raise RuntimeError(
                    "Summary is missing reference "
                    "or comparison name."
                )

            outputs = summary.get(
                "outputs",
                {},
            )

            alignment_value = (
                outputs.get("alignment")
                if isinstance(
                    outputs,
                    dict,
                )
                else None
            )

            alignment_path = (
                resolve_project_path(
                    project_root,
                    alignment_value,
                    summary_path.parent
                    / "dtw_alignment.csv",
                )
            )

            if not alignment_path.exists():
                raise FileNotFoundError(
                    "DTW alignment not found: "
                    f"{alignment_path}"
                )

            reference_feature_path = (
                find_feature_csv(
                    processed_dir,
                    reference_name,
                )
            )

            comparison_feature_path = (
                find_feature_csv(
                    processed_dir,
                    comparison_name,
                )
            )

            alignment = pd.read_csv(
                alignment_path
            )

            reference_features = pd.read_csv(
                reference_feature_path
            )

            comparison_features = pd.read_csv(
                comparison_feature_path
            )

            (
                aligned,
                base_features,
                detected_columns,
            ) = build_aligned_differences(
                alignment,
                reference_features,
                comparison_features,
            )

            (
                primary_window,
                isolated_window,
            ) = deterministic_windows(
                summary
            )

            windows = aggregate_windows(
                aligned=aligned,
                base_features=base_features,
                comparison_id=comparison_id,
                leakage_group_id=leakage_group,
                reference_name=reference_name,
                comparison_name=comparison_name,
                window_seconds=args.window_seconds,
                stride_seconds=args.stride_seconds,
                min_window_points=args.min_window_points,
                primary_window=primary_window,
                isolated_window=isolated_window,
            )

            if not windows:
                raise RuntimeError(
                    "No valid windows were produced."
                )

            all_windows.extend(
                windows
            )

            base_feature_union.update(
                base_features
            )

            processed.append(
                {
                    "comparison_id":
                        comparison_id,
                    "leakage_group_id":
                        leakage_group,
                    "reference":
                        reference_name,
                    "comparison":
                        comparison_name,
                    "alignment_csv":
                        str(
                            alignment_path
                        ),
                    "reference_feature_csv":
                        str(
                            reference_feature_path
                        ),
                    "comparison_feature_csv":
                        str(
                            comparison_feature_path
                        ),
                    "aligned_steps":
                        int(len(aligned)),
                    "windows":
                        int(len(windows)),
                    "base_feature_count":
                        int(
                            len(base_features)
                        ),
                    "base_features":
                        base_features,
                    "detected_alignment_columns":
                        detected_columns,
                }
            )

            seen_groups.add(
                leakage_group
            )

        except Exception as exc:
            errors.append(
                {
                    "comparison_id":
                        comparison_id,
                    "error_type":
                        type(exc).__name__,
                    "message":
                        str(exc),
                }
            )

    if not all_windows:
        print("=" * 72)
        print(
            "MotionStage Stage 18C - "
            "Window Dataset"
        )
        print("=" * 72)
        print(
            "No windows were produced."
        )

        if errors:
            print("\nErrors:")
            for error in errors:
                print(
                    f"  - "
                    f"{error['comparison_id']}: "
                    f"{error['error_type']}: "
                    f"{error['message']}"
                )

        raise RuntimeError(
            "Stage 18C could not build "
            "the window dataset."
        )

    window_df = pd.DataFrame(
        all_windows
    )

    metadata_columns = [
        "window_id",
        "comparison_id",
        "leakage_group_id",
        "reference_name",
        "comparison_name",
        "comparison_window_start_seconds",
        "comparison_window_end_seconds",
        "comparison_window_mid_seconds",
        "reference_window_start_seconds",
        "reference_window_end_seconds",
        "aligned_points",
        "overlaps_primary_divergence",
        "overlaps_isolated_divergence",
    ]

    ordered = [
        column
        for column in metadata_columns
        if column in window_df.columns
    ]

    ordered += sorted(
        column
        for column in window_df.columns
        if column not in ordered
    )

    window_df = window_df[
        ordered
    ]

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    window_df.to_csv(
        output_path,
        index=False,
    )

    ml_feature_columns = [
        column
        for column in window_df.columns
        if (
            column.endswith(
                "__mean_absdiff"
            )
            or column.endswith(
                "__std_absdiff"
            )
            or column.endswith(
                "__max_absdiff"
            )
            or column.startswith(
                "pooled_"
            )
        )
    ]

    report = {
        "stage":
            "18C",
        "purpose":
            (
                "Window-level movement-difference "
                "dataset for unsupervised anomaly "
                "detection."
            ),
        "window_seconds":
            args.window_seconds,
        "stride_seconds":
            args.stride_seconds,
        "min_window_points":
            args.min_window_points,
        "stored_comparisons_found":
            len(summary_paths),
        "comparisons_processed":
            len(processed),
        "duplicate_comparisons_skipped":
            len(skipped_duplicates),
        "comparison_errors":
            len(errors),
        "total_windows":
            int(
                len(window_df)
            ),
        "unique_leakage_groups":
            int(
                window_df[
                    "leakage_group_id"
                ].nunique()
            ),
        "base_feature_union_count":
            len(
                base_feature_union
            ),
        "ml_feature_column_count":
            len(
                ml_feature_columns
            ),
        "ml_feature_columns":
            ml_feature_columns,
        "evaluation_only_columns": [
            "overlaps_primary_divergence",
            "overlaps_isolated_divergence",
        ],
        "processed_comparisons":
            processed,
        "skipped_duplicate_comparisons":
            skipped_duplicates,
        "errors":
            errors,
        "notes": [
            (
                "One representative per Stage 18B "
                "leakage group is processed by "
                "default."
            ),
            (
                "The divergence-overlap flags are "
                "evaluation metadata only and must "
                "not be used as Isolation Forest "
                "training features."
            ),
            (
                "Stage 18D should standardise/select "
                "the numeric difference features "
                "before anomaly modelling."
            ),
        ],
    }

    manifest_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 72)
    print(
        "MotionStage Stage 18C - "
        "Window-Level ML Dataset"
    )
    print("=" * 72)

    print(
        f"Stored comparisons found     : "
        f"{len(summary_paths)}"
    )

    print(
        f"Comparisons processed        : "
        f"{len(processed)}"
    )

    print(
        "Duplicate comparisons skipped: "
        f"{len(skipped_duplicates)}"
    )

    print(
        f"Comparison errors            : "
        f"{len(errors)}"
    )

    print(
        f"Total windows                : "
        f"{len(window_df)}"
    )

    print(
        f"Unique leakage groups        : "
        f"{window_df['leakage_group_id'].nunique()}"
    )

    print(
        f"Base motion features         : "
        f"{len(base_feature_union)}"
    )

    print(
        f"Window ML feature columns    : "
        f"{len(ml_feature_columns)}"
    )

    print(
        f"Output CSV                   : "
        f"{output_path}"
    )

    print(
        f"Manifest                     : "
        f"{manifest_path}"
    )

    if skipped_duplicates:
        print(
            "\nSkipped duplicate comparisons:"
        )
        for item in skipped_duplicates:
            print(
                f"  - {item['comparison_id']} "
                f"({item['leakage_group_id']})"
            )

    if errors:
        print(
            "\nComparisons needing attention:"
        )

        for error in errors:
            print(
                f"  - "
                f"{error['comparison_id']}: "
                f"{error['error_type']}: "
                f"{error['message']}"
            )

    print("\nImportant:")
    print(
        "  overlaps_primary_divergence and "
        "overlaps_isolated_divergence are "
        "evaluation-only metadata."
    )

    print(
        "  They are NOT target labels and must "
        "not be supplied to the anomaly model."
    )

    print("=" * 72)


if __name__ == "__main__":
    main()
