#!/usr/bin/env python3
"""
MotionStage Stage 18A
Build a comparison-level feature dataset from data/comparisons/*/final_summary.json.

Each row represents one completed comparison. The script exports existing
MotionStage measurements without inventing a target label or treating prototype
similarity indices as probabilities.

Default outputs:
    data/ml/comparison_features.csv
    data/ml/comparison_features_manifest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


REGIONS = ("left_arm", "right_arm", "lower_body", "torso")
SIMILARITY_FIELDS = (
    "pose_similarity",
    "position_similarity",
    "speed_similarity",
    "composite_similarity",
)


def put(row: dict[str, Any], key: str, value: Any) -> None:
    if isinstance(value, (str, int, float, bool)) or value is None:
        row[key] = value


def add_window(
    row: dict[str, Any],
    prefix: str,
    window: dict[str, Any] | None,
) -> None:
    window = window or {}
    start = window.get("start_seconds")
    end = window.get("end_seconds")

    put(row, f"{prefix}_start_seconds", start)
    put(row, f"{prefix}_end_seconds", end)

    if isinstance(start, (int, float)) and isinstance(end, (int, float)):
        put(row, f"{prefix}_duration_seconds", end - start)
    else:
        put(row, f"{prefix}_duration_seconds", None)


def add_region_similarity(
    row: dict[str, Any],
    prefix: str,
    regions: dict[str, Any] | None,
) -> None:
    regions = regions or {}
    for region in REGIONS:
        values = regions.get(region, {})
        if not isinstance(values, dict):
            values = {}

        for field in SIMILARITY_FIELDS:
            put(row, f"{prefix}_{region}_{field}", values.get(field))


def flatten_summary(path: Path, project_root: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))

    row: dict[str, Any] = {
        "comparison_id": path.parent.name,
        "summary_path": str(path.relative_to(project_root)),
        "reference_name": data.get("reference"),
        "comparison_name": data.get("comparison"),
    }

    global_similarity = data.get("global_similarity", {})
    if not isinstance(global_similarity, dict):
        global_similarity = {}

    for field in (
        "overall_similarity",
        "pose_similarity",
        "position_similarity",
        "movement_speed_similarity",
        "regional_composite_similarity",
        "duration_similarity",
        "alignment_steps",
        "position_max_distance",
    ):
        put(row, f"global_{field}", global_similarity.get(field))

    add_region_similarity(row, "global", global_similarity.get("regions"))

    divergence = data.get("overall_divergence")
    if not isinstance(divergence, dict):
        divergence = data.get("divergence", {})
    if not isinstance(divergence, dict):
        divergence = {}

    put(row, "primary_most_divergent_region", divergence.get("most_divergent_region"))
    put(
        row,
        "primary_worst_regional_window_similarity",
        divergence.get("worst_regional_window_similarity"),
    )
    put(row, "primary_window_seconds", divergence.get("window_seconds"))

    add_window(row, "primary_reference_window", divergence.get("reference_window"))
    add_window(row, "primary_comparison_window", divergence.get("comparison_window"))

    region_similarity = divergence.get("region_similarity", {})
    if not isinstance(region_similarity, dict):
        region_similarity = {}

    for region in REGIONS:
        put(row, f"primary_window_{region}_similarity", region_similarity.get(region))

    divergent_components = divergence.get("divergent_region_components", {})
    if not isinstance(divergent_components, dict):
        divergent_components = {}

    for field in ("pose_similarity", "position_similarity", "speed_similarity"):
        put(
            row,
            f"primary_divergent_region_{field}",
            divergent_components.get(field),
        )

    robust = data.get("robust_localisation", {})
    if not isinstance(robust, dict):
        robust = {}

    robust_similarity = robust.get("similarity", {})
    if not isinstance(robust_similarity, dict):
        robust_similarity = {}

    for field in (
        "overall_similarity",
        "pose_similarity",
        "position_similarity",
        "movement_speed_similarity",
        "regional_composite_similarity",
        "duration_similarity",
        "alignment_steps",
        "position_max_distance",
    ):
        put(row, f"robust_{field}", robust_similarity.get(field))

    add_region_similarity(row, "robust", robust_similarity.get("regions"))

    robust_divergence = robust.get("divergence", {})
    if isinstance(robust_divergence, dict):
        put(
            row,
            "robust_most_divergent_region",
            robust_divergence.get("most_divergent_region"),
        )
        put(
            row,
            "robust_worst_regional_window_similarity",
            robust_divergence.get("worst_regional_window_similarity"),
        )

    isolated = data.get("isolated_arm_divergence", {})
    if not isinstance(isolated, dict):
        isolated = {}

    put(row, "isolated_region", isolated.get("region"))
    put(row, "isolated_isolation_gap", isolated.get("isolation_gap"))
    put(row, "isolated_region_similarity", isolated.get("isolated_region_similarity"))
    put(row, "isolated_peer_arm_similarity", isolated.get("peer_arm_similarity"))
    put(row, "isolated_context_similarity", isolated.get("context_similarity"))

    add_window(row, "isolated_reference_window", isolated.get("reference_window"))
    add_window(row, "isolated_comparison_window", isolated.get("comparison_window"))

    isolated_components = isolated.get("components", {})
    if not isinstance(isolated_components, dict):
        isolated_components = {}

    for field in ("pose_similarity", "position_similarity", "speed_similarity"):
        put(row, f"isolated_{field}", isolated_components.get(field))

    methods = data.get("methods", {})
    if isinstance(methods, dict):
        for key, value in methods.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                put(row, f"method_{key}", value)

    return row


def build_dataset(
    project_root: Path,
    comparisons_dir: Path,
) -> tuple[pd.DataFrame, list[str]]:
    summary_paths = sorted(comparisons_dir.glob("*/final_summary.json"))

    if not summary_paths:
        raise FileNotFoundError(
            f"No final_summary.json files found under {comparisons_dir}"
        )

    rows: list[dict[str, Any]] = []
    errors: list[str] = []

    for path in summary_paths:
        try:
            rows.append(flatten_summary(path, project_root))
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")

    if not rows:
        raise RuntimeError("No valid comparison summaries could be parsed.")

    df = pd.DataFrame(rows)

    identifier_columns = [
        "comparison_id",
        "summary_path",
        "reference_name",
        "comparison_name",
    ]
    other_columns = sorted(
        column
        for column in df.columns
        if column not in identifier_columns
    )

    ordered_columns = [
        column for column in identifier_columns if column in df.columns
    ] + other_columns

    return df[ordered_columns], errors


def dataset_report(df: pd.DataFrame) -> dict[str, Any]:
    id_columns = {
        "comparison_id",
        "summary_path",
        "reference_name",
        "comparison_name",
    }

    feature_columns = [
        column
        for column in df.columns
        if column not in id_columns and not column.startswith("method_")
    ]

    numeric_feature_columns = [
        column
        for column in feature_columns
        if pd.api.types.is_numeric_dtype(df[column])
    ]

    missing = {
        column: int(df[column].isna().sum())
        for column in df.columns
        if df[column].isna().any()
    }

    duplicate_feature_rows = 0
    duplicate_groups: list[list[str]] = []

    if numeric_feature_columns:
        duplicate_mask = df.duplicated(
            subset=numeric_feature_columns,
            keep=False,
        )
        duplicate_feature_rows = int(duplicate_mask.sum())

        if duplicate_feature_rows:
            grouped = (
                df.loc[duplicate_mask]
                .groupby(numeric_feature_columns, dropna=False, sort=False)
            )
            for _, group in grouped:
                if len(group) > 1:
                    duplicate_groups.append(
                        group["comparison_id"].astype(str).tolist()
                    )

    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "feature_columns": int(len(feature_columns)),
        "numeric_feature_columns": int(len(numeric_feature_columns)),
        "missing_values_by_column": missing,
        "exact_duplicate_numeric_feature_rows": duplicate_feature_rows,
        "exact_duplicate_numeric_feature_groups": duplicate_groups,
        "training_note": (
            "Feature dataset only. No supervised-learning target label "
            "has been invented."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build MotionStage Stage 18A comparison-level feature dataset."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--comparisons-dir", default="data/comparisons")
    parser.add_argument("--output", default="data/ml/comparison_features.csv")
    parser.add_argument(
        "--manifest",
        default="data/ml/comparison_features_manifest.json",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    comparisons_dir = (
        project_root / args.comparisons_dir
        if not Path(args.comparisons_dir).is_absolute()
        else Path(args.comparisons_dir)
    )
    output_path = (
        project_root / args.output
        if not Path(args.output).is_absolute()
        else Path(args.output)
    )
    manifest_path = (
        project_root / args.manifest
        if not Path(args.manifest).is_absolute()
        else Path(args.manifest)
    )

    df, errors = build_dataset(project_root, comparisons_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    report = dataset_report(df)
    report["source_directory"] = str(comparisons_dir)
    report["output_csv"] = str(output_path)
    report["parse_errors"] = errors

    manifest_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("=" * 68)
    print("MotionStage Stage 18A - Comparison Feature Dataset")
    print("=" * 68)
    print(f"Comparison summaries parsed : {len(df)}")
    print(f"Dataset columns             : {len(df.columns)}")
    print(f"Output CSV                  : {output_path}")
    print(f"Manifest                    : {manifest_path}")
    print(f"Parse errors                : {len(errors)}")
    print(
        "Exact duplicate feature rows: "
        f"{report['exact_duplicate_numeric_feature_rows']}"
    )

    if report["exact_duplicate_numeric_feature_groups"]:
        print("\nPotential duplicate comparison groups:")
        for group in report["exact_duplicate_numeric_feature_groups"]:
            print("  - " + " | ".join(group))

    print("\nImportant:")
    print("  This dataset contains MotionStage features only.")
    print("  No ML target label has been created.")
    print(
        "  Repeated runs of the same underlying video pair must not be "
        "split across train/test folds."
    )
    print("=" * 68)


if __name__ == "__main__":
    main()
