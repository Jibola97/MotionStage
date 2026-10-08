#!/usr/bin/env python3
"""
MotionStage Stage 18D
Train an unsupervised Isolation Forest over Stage 18C window-level features.

Input:
    data/ml/window_features.csv

Outputs:
    data/ml/window_anomaly_scores.csv
    data/ml/isolation_forest_manifest.json
    data/ml/isolation_forest_model.joblib

Notes:
- Deterministic divergence overlap columns are NEVER used for training.
- Metadata columns are excluded from the model.
- Missing numeric values are median-imputed.
- Numeric features are standardised before Isolation Forest fitting.
- The model produces anomaly scores/ranks for exploration; it does not create
  a supervised "good/bad movement" label.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


METADATA_COLUMNS = {
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
}

EVALUATION_ONLY_COLUMNS = {
    "overlaps_primary_divergence",
    "overlaps_isolated_divergence",
}


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()

    selected = [
        column
        for column in numeric
        if column not in METADATA_COLUMNS
        and column not in EVALUATION_ONLY_COLUMNS
        and (
            column.endswith("__mean_absdiff")
            or column.endswith("__std_absdiff")
            or column.endswith("__max_absdiff")
            or column.startswith("pooled_")
        )
    ]

    return sorted(selected)


def percentile_rank_descending(values: pd.Series) -> pd.Series:
    """
    Convert anomaly scores into a 0..100 percentile where larger means
    more anomalous.
    """
    return values.rank(
        method="average",
        pct=True,
        ascending=True,
    ) * 100.0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "MotionStage Stage 18D - Isolation Forest anomaly model"
        )
    )

    parser.add_argument(
        "--input",
        default="data/ml/window_features.csv",
    )

    parser.add_argument(
        "--scores-output",
        default="data/ml/window_anomaly_scores.csv",
    )

    parser.add_argument(
        "--manifest",
        default="data/ml/isolation_forest_manifest.json",
    )

    parser.add_argument(
        "--model-output",
        default="data/ml/isolation_forest_model.joblib",
    )

    parser.add_argument(
        "--n-estimators",
        type=int,
        default=400,
    )

    parser.add_argument(
        "--contamination",
        type=float,
        default=0.10,
        help=(
            "Prototype anomaly fraction used only for the binary flag. "
            "The continuous anomaly score should remain the main output."
        ),
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    scores_path = Path(args.scores_output)
    manifest_path = Path(args.manifest)
    model_path = Path(args.model_output)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Stage 18C dataset not found: {input_path}"
        )

    if not (0.0 < args.contamination <= 0.5):
        raise ValueError(
            "--contamination must be greater than 0 and at most 0.5"
        )

    df = pd.read_csv(input_path)

    if df.empty:
        raise RuntimeError("Stage 18C dataset is empty.")

    feature_columns = select_feature_columns(df)

    if not feature_columns:
        raise RuntimeError(
            "No Stage 18C ML feature columns were found."
        )

    X = df[feature_columns].copy()

    all_null_features = [
        column
        for column in feature_columns
        if X[column].isna().all()
    ]

    if all_null_features:
        X = X.drop(columns=all_null_features)
        feature_columns = [
            column
            for column in feature_columns
            if column not in all_null_features
        ]

    constant_features = [
        column
        for column in feature_columns
        if X[column].nunique(dropna=True) <= 1
    ]

    model_features = [
        column
        for column in feature_columns
        if column not in constant_features
    ]

    if len(model_features) < 2:
        raise RuntimeError(
            "Fewer than two usable numeric features remain after cleaning."
        )

    X_model = X[model_features]

    pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "isolation_forest",
                IsolationForest(
                    n_estimators=args.n_estimators,
                    contamination=args.contamination,
                    random_state=args.random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    pipeline.fit(X_model)

    forest = pipeline.named_steps["isolation_forest"]

    # sklearn decision_function:
    # larger = more normal, smaller = more anomalous.
    # Negate it so MotionStage's anomaly score has the intuitive direction:
    # larger = more anomalous.
    normality_score = pipeline.decision_function(X_model)
    anomaly_score = -normality_score

    sklearn_prediction = pipeline.predict(X_model)
    binary_anomaly = sklearn_prediction == -1

    result = df.copy()

    result["ml_anomaly_score"] = anomaly_score

    result["ml_anomaly_percentile"] = (
        percentile_rank_descending(
            result["ml_anomaly_score"]
        )
    )

    result["ml_anomaly_flag"] = binary_anomaly

    # Rank windows independently inside each comparison as well. This makes
    # the score easier to interpret on the dashboard even when comparisons
    # have different absolute score ranges.
    result["ml_within_comparison_percentile"] = (
        result.groupby(
            "comparison_id"
        )["ml_anomaly_score"]
        .rank(
            method="average",
            pct=True,
            ascending=True,
        )
        * 100.0
    )

    result["ml_within_comparison_rank"] = (
        result.groupby(
            "comparison_id"
        )["ml_anomaly_score"]
        .rank(
            method="first",
            ascending=False,
        )
        .astype(int)
    )

    scores_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        scores_path,
        index=False,
    )

    joblib.dump(
        {
            "pipeline": pipeline,
            "model_feature_columns": model_features,
            "stage": "18D",
        },
        model_path,
    )

    comparison_summary: list[dict[str, Any]] = []

    for comparison_id, group in result.groupby(
        "comparison_id",
        sort=False,
    ):
        top = group.sort_values(
            "ml_anomaly_score",
            ascending=False,
        ).head(3)

        comparison_summary.append(
            {
                "comparison_id": str(comparison_id),
                "windows": int(len(group)),
                "flagged_windows": int(
                    group["ml_anomaly_flag"].sum()
                ),
                "top_3_anomaly_windows": [
                    {
                        "window_id": str(row.window_id),
                        "comparison_start_seconds": float(
                            row.comparison_window_start_seconds
                        ),
                        "comparison_end_seconds": float(
                            row.comparison_window_end_seconds
                        ),
                        "anomaly_score": float(
                            row.ml_anomaly_score
                        ),
                        "global_percentile": float(
                            row.ml_anomaly_percentile
                        ),
                        "within_comparison_percentile": float(
                            row.ml_within_comparison_percentile
                        ),
                        "overlaps_primary_divergence": bool(
                            row.overlaps_primary_divergence
                        ),
                        "overlaps_isolated_divergence": bool(
                            row.overlaps_isolated_divergence
                        ),
                    }
                    for row in top.itertuples()
                ],
            }
        )

    manifest = {
        "stage": "18D",
        "model": "IsolationForest",
        "purpose": (
            "Exploratory unsupervised detection of unusually large "
            "movement-difference windows."
        ),
        "input_csv": str(input_path),
        "scores_csv": str(scores_path),
        "model_file": str(model_path),
        "rows": int(len(result)),
        "comparisons": int(
            result["comparison_id"].nunique()
        ),
        "leakage_groups": int(
            result["leakage_group_id"].nunique()
            if "leakage_group_id" in result.columns
            else 0
        ),
        "candidate_feature_columns": int(
            len(feature_columns)
        ),
        "model_feature_columns": int(
            len(model_features)
        ),
        "all_null_features_removed": all_null_features,
        "constant_features_removed": constant_features,
        "n_estimators": args.n_estimators,
        "contamination": args.contamination,
        "random_state": args.random_state,
        "flagged_windows": int(
            result["ml_anomaly_flag"].sum()
        ),
        "flagged_fraction": float(
            result["ml_anomaly_flag"].mean()
        ),
        "evaluation_only_columns_excluded_from_training": sorted(
            EVALUATION_ONLY_COLUMNS
        ),
        "comparison_summary": comparison_summary,
        "notes": [
            (
                "This is an unsupervised prototype. The binary anomaly flag "
                "depends on the chosen contamination setting and should not "
                "be interpreted as a validated good/bad movement label."
            ),
            (
                "The continuous anomaly score and within-comparison ranking "
                "are the preferred outputs for Stage 18E evaluation."
            ),
            (
                "Deterministic divergence-overlap columns are preserved only "
                "for evaluation and were not used during model fitting."
            ),
        ],
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 72)
    print(
        "MotionStage Stage 18D - Isolation Forest Anomaly Detection"
    )
    print("=" * 72)
    print(f"Windows                    : {len(result)}")
    print(
        f"Comparisons                : "
        f"{result['comparison_id'].nunique()}"
    )
    print(
        f"Leakage groups             : "
        f"{result['leakage_group_id'].nunique()}"
    )
    print(
        f"Candidate ML features      : "
        f"{len(feature_columns)}"
    )
    print(
        f"Model features             : "
        f"{len(model_features)}"
    )
    print(
        f"Constant features removed  : "
        f"{len(constant_features)}"
    )
    print(
        f"Flagged anomaly windows    : "
        f"{int(result['ml_anomaly_flag'].sum())}"
    )
    print(
        f"Flagged fraction           : "
        f"{result['ml_anomaly_flag'].mean():.3f}"
    )
    print(f"Scores CSV                 : {scores_path}")
    print(f"Model                      : {model_path}")
    print(f"Manifest                   : {manifest_path}")

    print("\nTop anomaly window per comparison:")

    for item in comparison_summary:
        top = item["top_3_anomaly_windows"][0]

        print(
            f"  - {item['comparison_id']}: "
            f"{top['comparison_start_seconds']:.3f}"
            f"-{top['comparison_end_seconds']:.3f}s "
            f"(score={top['anomaly_score']:.4f}, "
            f"primary_overlap="
            f"{top['overlaps_primary_divergence']}, "
            f"isolated_overlap="
            f"{top['overlaps_isolated_divergence']})"
        )

    print("\nImportant:")
    print(
        "  ml_anomaly_flag is a prototype thresholded indicator, "
        "not a ground-truth movement label."
    )
    print(
        "  Stage 18E will evaluate whether high-scoring windows align "
        "with the existing deterministic divergence system."
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
