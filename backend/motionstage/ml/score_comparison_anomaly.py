#!/usr/bin/env python3
"""
MotionStage Stage 18F
Score one completed comparison with the Stage 18D Isolation Forest model.

This script reuses the Stage 18C window-building logic, creates window-level
features for a single comparison, applies the saved unsupervised model, and
writes both detailed window scores and a compact JSON summary.

It is designed to be called by FastAPI in the .venv_pose3d environment, where
scikit-learn and joblib are installed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from build_window_dataset import (
    aggregate_windows,
    build_aligned_differences,
    deterministic_windows,
    find_feature_csv,
    resolve_project_path,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score one MotionStage comparison with Isolation Forest."
    )

    parser.add_argument("reference")
    parser.add_argument("comparison")

    parser.add_argument(
        "--summary",
        default=None,
    )

    parser.add_argument(
        "--model",
        default="data/ml/isolation_forest_model.joblib",
    )

    parser.add_argument(
        "--output",
        default=None,
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

    args = parser.parse_args()

    reference_name = args.reference
    comparison_name = args.comparison

    comparison_dir = (
        PROJECT_ROOT
        / "data"
        / "comparisons"
        / f"{reference_name}_vs_{comparison_name}"
    )

    summary_path = (
        Path(args.summary)
        if args.summary
        else comparison_dir / "final_summary.json"
    )

    if not summary_path.is_absolute():
        summary_path = PROJECT_ROOT / summary_path

    model_path = Path(args.model)

    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    output_path = (
        Path(args.output)
        if args.output
        else comparison_dir / "ml_anomaly_summary.json"
    )

    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    detailed_scores_path = (
        output_path.parent
        / "ml_anomaly_scores.csv"
    )

    if not summary_path.exists():
        raise FileNotFoundError(
            f"Comparison summary not found: {summary_path}"
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Isolation Forest model not found: {model_path}"
        )

    summary = json.loads(
        summary_path.read_text(
            encoding="utf-8"
        )
    )

    outputs = summary.get("outputs", {})

    alignment_value = (
        outputs.get("alignment")
        if isinstance(outputs, dict)
        else None
    )

    alignment_path = resolve_project_path(
        PROJECT_ROOT,
        alignment_value,
        comparison_dir / "dtw_alignment.csv",
    )

    if not alignment_path.exists():
        raise FileNotFoundError(
            f"DTW alignment not found: {alignment_path}"
        )

    processed_dir = (
        PROJECT_ROOT
        / "data"
        / "processed"
    )

    reference_feature_path = find_feature_csv(
        processed_dir,
        reference_name,
    )

    comparison_feature_path = find_feature_csv(
        processed_dir,
        comparison_name,
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
        _,
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

    window_rows = aggregate_windows(
        aligned=aligned,
        base_features=base_features,
        comparison_id=f"{reference_name}_vs_{comparison_name}",
        leakage_group_id="runtime",
        reference_name=reference_name,
        comparison_name=comparison_name,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
        min_window_points=args.min_window_points,
        primary_window=primary_window,
        isolated_window=isolated_window,
    )

    if not window_rows:
        raise RuntimeError(
            "No valid ML windows were produced for this comparison."
        )

    windows = pd.DataFrame(
        window_rows
    )

    artifact: dict[str, Any] = joblib.load(
        model_path
    )

    pipeline = artifact.get("pipeline")
    model_feature_columns = artifact.get(
        "model_feature_columns",
        [],
    )

    if pipeline is None:
        raise RuntimeError(
            "Saved model artifact does not contain a pipeline."
        )

    if not model_feature_columns:
        raise RuntimeError(
            "Saved model artifact does not contain model feature columns."
        )

    # The saved pipeline has learned median values from Stage 18D. Any feature
    # absent from a runtime comparison is added as NaN so the fitted imputer
    # can apply those training medians.
    for column in model_feature_columns:
        if column not in windows.columns:
            windows[column] = np.nan

    X = windows[
        model_feature_columns
    ].copy()

    normality_score = pipeline.decision_function(
        X
    )

    windows["ml_anomaly_score"] = (
        -normality_score
    )

    windows["ml_anomaly_flag"] = (
        pipeline.predict(X) == -1
    )

    windows[
        "ml_within_comparison_percentile"
    ] = (
        windows[
            "ml_anomaly_score"
        ]
        .rank(
            method="average",
            pct=True,
            ascending=True,
        )
        * 100.0
    )

    windows[
        "ml_within_comparison_rank"
    ] = (
        windows[
            "ml_anomaly_score"
        ]
        .rank(
            method="first",
            ascending=False,
        )
        .astype(int)
    )

    ranked = windows.sort_values(
        "ml_anomaly_score",
        ascending=False,
    ).reset_index(drop=True)

    top_rows = ranked.head(
        min(5, len(ranked))
    )

    top_windows: list[
        dict[str, Any]
    ] = []

    for row in top_rows.itertuples():
        top_windows.append(
            {
                "window_id":
                    str(row.window_id),

                "start_seconds":
                    float(
                        row.comparison_window_start_seconds
                    ),

                "end_seconds":
                    float(
                        row.comparison_window_end_seconds
                    ),

                "anomaly_score":
                    float(
                        row.ml_anomaly_score
                    ),

                "percentile":
                    float(
                        row.ml_within_comparison_percentile
                    ),

                "rank":
                    int(
                        row.ml_within_comparison_rank
                    ),

                "anomaly_flag":
                    bool(
                        row.ml_anomaly_flag
                    ),

                "overlaps_primary_divergence":
                    bool(
                        row.overlaps_primary_divergence
                    ),

                "overlaps_isolated_divergence":
                    bool(
                        row.overlaps_isolated_divergence
                    ),
            }
        )

    top_window = top_windows[0]

    result = {
        "available": True,

        "model":
            "IsolationForest",

        "model_stage":
            artifact.get(
                "stage",
                "18D",
            ),

        "reference":
            reference_name,

        "comparison":
            comparison_name,

        "windows_analyzed":
            int(len(windows)),

        "window_seconds":
            args.window_seconds,

        "stride_seconds":
            args.stride_seconds,

        "model_feature_count":
            int(
                len(
                    model_feature_columns
                )
            ),

        "flagged_windows":
            int(
                windows[
                    "ml_anomaly_flag"
                ].sum()
            ),

        "top_window":
            top_window,

        "top_windows":
            top_windows,

        "scores_path":
            str(
                detailed_scores_path.relative_to(
                    PROJECT_ROOT
                )
            ),

        "summary_path":
            str(
                output_path.relative_to(
                    PROJECT_ROOT
                )
            ),

        "interpretation":
            (
                "Higher anomaly scores indicate movement-difference windows "
                "that are more unusual relative to the prototype training "
                "distribution. This is an exploratory unsupervised signal, "
                "not a validated good/bad movement label."
            ),
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    windows.to_csv(
        detailed_scores_path,
        index=False,
    )

    output_path.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 72)
    print(
        "MotionStage Stage 18F - "
        "Runtime ML Anomaly Scoring"
    )
    print("=" * 72)
    print(
        f"Comparison       : "
        f"{reference_name} vs {comparison_name}"
    )
    print(
        f"Windows analyzed : "
        f"{len(windows)}"
    )
    print(
        f"Flagged windows  : "
        f"{int(windows['ml_anomaly_flag'].sum())}"
    )
    print(
        f"Top anomaly      : "
        f"{top_window['start_seconds']:.3f}-"
        f"{top_window['end_seconds']:.3f}s"
    )
    print(
        f"Top score        : "
        f"{top_window['anomaly_score']:.4f}"
    )
    print(
        f"Summary          : "
        f"{output_path}"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
