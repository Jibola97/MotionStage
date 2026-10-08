#!/usr/bin/env python3
"""
MotionStage Stage 18E
Evaluate whether unsupervised anomaly peaks align with MotionStage's existing
deterministic divergence windows.

This uses leave-one-leakage-group-out evaluation:
- hold out one comparison/leakage group
- fit preprocessing + Isolation Forest on the remaining groups only
- score the held-out windows
- rank anomaly scores within the held-out comparison
- compare high-ranked windows with the existing primary/isolated divergence
  windows

The deterministic divergence flags are evaluation-only. They are never supplied
to the model as features or targets.

Inputs:
    data/ml/window_features.csv

Outputs:
    data/ml/anomaly_alignment_scores.csv
    data/ml/anomaly_alignment_report.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

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


def select_candidate_features(df: pd.DataFrame) -> list[str]:
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()

    return sorted(
        column
        for column in numeric
        if column not in METADATA_COLUMNS
        and (
            column.endswith("__mean_absdiff")
            or column.endswith("__std_absdiff")
            or column.endswith("__max_absdiff")
            or column.startswith("pooled_")
        )
    )


def safe_mean(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def safe_median(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.median())


def build_fold_pipeline(
    n_estimators: int,
    random_state: int,
) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "isolation_forest",
                IsolationForest(
                    n_estimators=n_estimators,
                    contamination="auto",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "MotionStage Stage 18E - leave-one-group-out "
            "anomaly/divergence alignment evaluation"
        )
    )

    parser.add_argument(
        "--input",
        default="data/ml/window_features.csv",
    )

    parser.add_argument(
        "--scores-output",
        default="data/ml/anomaly_alignment_scores.csv",
    )

    parser.add_argument(
        "--report",
        default="data/ml/anomaly_alignment_report.json",
    )

    parser.add_argument(
        "--n-estimators",
        type=int,
        default=400,
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    scores_path = Path(args.scores_output)
    report_path = Path(args.report)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Stage 18C dataset not found: {input_path}"
        )

    df = pd.read_csv(input_path)

    required = {
        "window_id",
        "comparison_id",
        "leakage_group_id",
        "overlaps_primary_divergence",
        "overlaps_isolated_divergence",
    }

    missing_required = required - set(df.columns)

    if missing_required:
        raise ValueError(
            "Input dataset is missing required columns: "
            f"{sorted(missing_required)}"
        )

    if df["leakage_group_id"].nunique() < 3:
        raise RuntimeError(
            "At least three leakage groups are required "
            "for this exploratory evaluation."
        )

    candidate_features = select_candidate_features(df)

    if not candidate_features:
        raise RuntimeError(
            "No Stage 18C ML feature columns were found."
        )

    scored_parts: list[pd.DataFrame] = []
    fold_reports: list[dict[str, Any]] = []

    groups = df["leakage_group_id"].astype(str)

    for fold_number, held_out_group in enumerate(
        groups.drop_duplicates(),
        start=1,
    ):
        train_mask = groups != held_out_group
        test_mask = groups == held_out_group

        train = df.loc[train_mask].copy()
        test = df.loc[test_mask].copy()

        fold_features = []

        for column in candidate_features:
            train_values = pd.to_numeric(
                train[column],
                errors="coerce",
            )

            if train_values.isna().all():
                continue

            if train_values.nunique(dropna=True) <= 1:
                continue

            fold_features.append(column)

        if len(fold_features) < 2:
            raise RuntimeError(
                f"Fold {fold_number} ({held_out_group}) has fewer "
                "than two usable training features."
            )

        X_train = train[fold_features]
        X_test = test[fold_features]

        pipeline = build_fold_pipeline(
            n_estimators=args.n_estimators,
            random_state=args.random_state + fold_number,
        )

        pipeline.fit(X_train)

        # score_samples: larger = more normal.
        # Negate so larger MotionStage score = more anomalous.
        anomaly_score = -pipeline.score_samples(X_test)

        test["loco_anomaly_score"] = anomaly_score

        test["loco_within_comparison_percentile"] = (
            test.groupby("comparison_id")["loco_anomaly_score"]
            .rank(
                method="average",
                pct=True,
                ascending=True,
            )
            * 100.0
        )

        test["loco_within_comparison_rank"] = (
            test.groupby("comparison_id")["loco_anomaly_score"]
            .rank(
                method="first",
                ascending=False,
            )
            .astype(int)
        )

        test["evaluation_fold"] = fold_number
        test["held_out_leakage_group"] = held_out_group

        scored_parts.append(test)

        fold_reports.append(
            {
                "fold": fold_number,
                "held_out_leakage_group": held_out_group,
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "train_groups": int(
                    train["leakage_group_id"].nunique()
                ),
                "test_comparisons": (
                    test["comparison_id"]
                    .astype(str)
                    .drop_duplicates()
                    .tolist()
                ),
                "model_feature_count": int(len(fold_features)),
            }
        )

    scored = (
        pd.concat(scored_parts, ignore_index=True)
        .sort_values(
            [
                "comparison_id",
                "comparison_window_start_seconds",
            ]
        )
        .reset_index(drop=True)
    )

    scored["overlaps_primary_divergence"] = (
        scored["overlaps_primary_divergence"]
        .astype(bool)
    )

    scored["overlaps_isolated_divergence"] = (
        scored["overlaps_isolated_divergence"]
        .astype(bool)
    )

    scored["overlaps_any_divergence"] = (
        scored["overlaps_primary_divergence"]
        | scored["overlaps_isolated_divergence"]
    )

    comparison_reports: list[dict[str, Any]] = []

    for comparison_id, group in scored.groupby(
        "comparison_id",
        sort=False,
    ):
        ranked = group.sort_values(
            "loco_anomaly_score",
            ascending=False,
        ).reset_index(drop=True)

        baseline_primary = float(
            ranked["overlaps_primary_divergence"].mean()
        )
        baseline_isolated = float(
            ranked["overlaps_isolated_divergence"].mean()
        )
        baseline_any = float(
            ranked["overlaps_any_divergence"].mean()
        )

        top_results: dict[str, Any] = {}

        for k in (1, 3, 5):
            actual_k = min(k, len(ranked))
            top = ranked.head(actual_k)

            top_results[f"top_{k}"] = {
                "windows_considered": int(actual_k),
                "primary_hits": int(
                    top["overlaps_primary_divergence"].sum()
                ),
                "isolated_hits": int(
                    top["overlaps_isolated_divergence"].sum()
                ),
                "any_divergence_hits": int(
                    top["overlaps_any_divergence"].sum()
                ),
                "any_divergence_hit": bool(
                    top["overlaps_any_divergence"].any()
                ),
            }

        top10_count = max(
            1,
            int(math.ceil(len(ranked) * 0.10)),
        )

        top10 = ranked.head(top10_count)

        top10_any_rate = float(
            top10["overlaps_any_divergence"].mean()
        )

        enrichment = (
            top10_any_rate / baseline_any
            if baseline_any > 0
            else None
        )

        inside = ranked.loc[
            ranked["overlaps_any_divergence"],
            "loco_within_comparison_percentile",
        ]

        outside = ranked.loc[
            ~ranked["overlaps_any_divergence"],
            "loco_within_comparison_percentile",
        ]

        top_window = ranked.iloc[0]

        comparison_reports.append(
            {
                "comparison_id": str(comparison_id),
                "leakage_group_id": str(
                    ranked["leakage_group_id"].iloc[0]
                ),
                "windows": int(len(ranked)),
                "baseline_primary_overlap_fraction":
                    baseline_primary,
                "baseline_isolated_overlap_fraction":
                    baseline_isolated,
                "baseline_any_divergence_fraction":
                    baseline_any,
                "top_k_alignment": top_results,
                "top_10_percent_count": int(top10_count),
                "top_10_percent_any_divergence_fraction":
                    top10_any_rate,
                "top_10_percent_enrichment_over_baseline":
                    enrichment,
                "mean_percentile_inside_any_divergence":
                    safe_mean(inside),
                "mean_percentile_outside_any_divergence":
                    safe_mean(outside),
                "median_percentile_inside_any_divergence":
                    safe_median(inside),
                "median_percentile_outside_any_divergence":
                    safe_median(outside),
                "top_anomaly_window": {
                    "window_id": str(top_window["window_id"]),
                    "start_seconds": float(
                        top_window[
                            "comparison_window_start_seconds"
                        ]
                    ),
                    "end_seconds": float(
                        top_window[
                            "comparison_window_end_seconds"
                        ]
                    ),
                    "score": float(
                        top_window["loco_anomaly_score"]
                    ),
                    "primary_overlap": bool(
                        top_window[
                            "overlaps_primary_divergence"
                        ]
                    ),
                    "isolated_overlap": bool(
                        top_window[
                            "overlaps_isolated_divergence"
                        ]
                    ),
                    "any_divergence_overlap": bool(
                        top_window[
                            "overlaps_any_divergence"
                        ]
                    ),
                },
            }
        )

    overall_baseline_any = float(
        scored["overlaps_any_divergence"].mean()
    )

    top1_hit_rate = float(
        np.mean(
            [
                item["top_k_alignment"]["top_1"][
                    "any_divergence_hit"
                ]
                for item in comparison_reports
            ]
        )
    )

    top3_hit_rate = float(
        np.mean(
            [
                item["top_k_alignment"]["top_3"][
                    "any_divergence_hit"
                ]
                for item in comparison_reports
            ]
        )
    )

    top5_hit_rate = float(
        np.mean(
            [
                item["top_k_alignment"]["top_5"][
                    "any_divergence_hit"
                ]
                for item in comparison_reports
            ]
        )
    )

    top10_rates = [
        item["top_10_percent_any_divergence_fraction"]
        for item in comparison_reports
    ]

    enrichments = [
        item["top_10_percent_enrichment_over_baseline"]
        for item in comparison_reports
        if item["top_10_percent_enrichment_over_baseline"]
        is not None
    ]

    inside_percentiles = scored.loc[
        scored["overlaps_any_divergence"],
        "loco_within_comparison_percentile",
    ]

    outside_percentiles = scored.loc[
        ~scored["overlaps_any_divergence"],
        "loco_within_comparison_percentile",
    ]

    scores_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    scored.to_csv(
        scores_path,
        index=False,
    )

    report = {
        "stage": "18E",
        "evaluation_design": (
            "Leave-one-leakage-group-out unsupervised evaluation"
        ),
        "input_csv": str(input_path),
        "scores_csv": str(scores_path),
        "windows": int(len(scored)),
        "comparisons": int(
            scored["comparison_id"].nunique()
        ),
        "leakage_groups": int(
            scored["leakage_group_id"].nunique()
        ),
        "candidate_feature_columns": int(
            len(candidate_features)
        ),
        "n_estimators": args.n_estimators,
        "random_state": args.random_state,
        "overall": {
            "primary_overlap_fraction": float(
                scored[
                    "overlaps_primary_divergence"
                ].mean()
            ),
            "isolated_overlap_fraction": float(
                scored[
                    "overlaps_isolated_divergence"
                ].mean()
            ),
            "any_divergence_overlap_fraction":
                overall_baseline_any,
            "top_1_comparison_hit_rate":
                top1_hit_rate,
            "top_3_comparison_hit_rate":
                top3_hit_rate,
            "top_5_comparison_hit_rate":
                top5_hit_rate,
            "mean_top_10_percent_any_divergence_fraction":
                float(np.mean(top10_rates)),
            "mean_top_10_percent_enrichment_over_baseline":
                (
                    float(np.mean(enrichments))
                    if enrichments
                    else None
                ),
            "mean_anomaly_percentile_inside_any_divergence":
                safe_mean(inside_percentiles),
            "mean_anomaly_percentile_outside_any_divergence":
                safe_mean(outside_percentiles),
            "median_anomaly_percentile_inside_any_divergence":
                safe_median(inside_percentiles),
            "median_anomaly_percentile_outside_any_divergence":
                safe_median(outside_percentiles),
        },
        "folds": fold_reports,
        "comparisons_detail": comparison_reports,
        "notes": [
            (
                "Deterministic divergence windows are not treated as "
                "ground-truth labels. These metrics describe agreement "
                "between two different MotionStage signals."
            ),
            (
                "All anomaly scores are generated out-of-group: the held-out "
                "leakage group is not used to fit its scoring model."
            ),
            (
                "Within-comparison percentile/rank is preferred for aggregate "
                "interpretation because raw Isolation Forest scores from "
                "different leave-one-group-out folds are not calibrated to "
                "one common absolute scale."
            ),
            (
                "The dataset is still small, so results should be described "
                "as exploratory rather than as validated generalisation."
            ),
        ],
    }

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    overall = report["overall"]

    print("=" * 74)
    print(
        "MotionStage Stage 18E - ML / Deterministic Alignment Evaluation"
    )
    print("=" * 74)
    print(f"Windows                         : {len(scored)}")
    print(
        f"Comparisons                     : "
        f"{scored['comparison_id'].nunique()}"
    )
    print(
        f"Leakage groups                  : "
        f"{scored['leakage_group_id'].nunique()}"
    )
    print(
        f"Candidate ML features           : "
        f"{len(candidate_features)}"
    )
    print(
        f"Baseline any-divergence fraction: "
        f"{overall['any_divergence_overlap_fraction']:.3f}"
    )
    print(
        f"Top-1 comparison hit rate       : "
        f"{overall['top_1_comparison_hit_rate']:.3f}"
    )
    print(
        f"Top-3 comparison hit rate       : "
        f"{overall['top_3_comparison_hit_rate']:.3f}"
    )
    print(
        f"Top-5 comparison hit rate       : "
        f"{overall['top_5_comparison_hit_rate']:.3f}"
    )
    print(
        "Mean top-10% divergence fraction: "
        f"{overall['mean_top_10_percent_any_divergence_fraction']:.3f}"
    )

    enrichment = overall[
        "mean_top_10_percent_enrichment_over_baseline"
    ]

    print(
        "Mean top-10% enrichment          : "
        + (
            f"{enrichment:.3f}x"
            if enrichment is not None
            else "n/a"
        )
    )

    print(
        "Mean percentile inside divergence: "
        f"{overall['mean_anomaly_percentile_inside_any_divergence']:.2f}"
    )
    print(
        "Mean percentile outside divergence: "
        f"{overall['mean_anomaly_percentile_outside_any_divergence']:.2f}"
    )

    print("\nTop anomaly window by comparison:")

    for item in comparison_reports:
        top = item["top_anomaly_window"]

        print(
            f"  - {item['comparison_id']}: "
            f"{top['start_seconds']:.3f}-"
            f"{top['end_seconds']:.3f}s "
            f"(any_overlap="
            f"{top['any_divergence_overlap']})"
        )

    print(f"\nScores CSV : {scores_path}")
    print(f"Report     : {report_path}")

    print("\nInterpretation rule:")
    print(
        "  Higher top-k hit rates, top-10% enrichment above 1.0x, "
        "and higher anomaly percentiles inside divergence windows "
        "indicate useful agreement."
    )
    print(
        "  Treat the result as exploratory because only seven "
        "independent comparison groups are available."
    )

    print("=" * 74)


if __name__ == "__main__":
    main()
