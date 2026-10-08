#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ID_COLUMNS = {
    "comparison_id",
    "summary_path",
    "reference_name",
    "comparison_name",
}


def normalise_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return round(float(value), 10)
    if isinstance(value, (int, float, bool, str)):
        return value
    return str(value)


def signature_for_row(
    row: pd.Series,
    columns: list[str],
) -> str:
    payload = {
        col: normalise_value(row[col])
        for col in columns
    }

    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()

    return hashlib.sha256(raw).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "MotionStage Stage 18B dataset validation"
        )
    )

    parser.add_argument(
        "--input",
        default="data/ml/comparison_features.csv",
    )

    parser.add_argument(
        "--output",
        default="data/ml/comparison_features_validated.csv",
    )

    parser.add_argument(
        "--report",
        default="data/ml/comparison_features_validation.json",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    report_path = Path(args.report)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {input_path}"
        )

    df = pd.read_csv(input_path)

    feature_columns = [
        c
        for c in df.columns
        if c not in ID_COLUMNS
        and not c.startswith("method_")
    ]

    numeric_feature_columns = [
        c
        for c in feature_columns
        if pd.api.types.is_numeric_dtype(df[c])
    ]

    categorical_feature_columns = [
        c
        for c in feature_columns
        if c not in numeric_feature_columns
    ]

    all_null_columns = [
        c
        for c in feature_columns
        if df[c].isna().all()
    ]

    usable_feature_columns = [
        c
        for c in feature_columns
        if c not in all_null_columns
    ]

    constant_columns = [
        c
        for c in usable_feature_columns
        if df[c].nunique(dropna=True) <= 1
    ]

    missing_by_column = {
        c: int(n)
        for c, n in df.isna().sum().items()
        if int(n) > 0
    }

    signature_columns = [
        c
        for c in numeric_feature_columns
        if c not in all_null_columns
    ]

    df["feature_signature"] = df.apply(
        lambda row: signature_for_row(
            row,
            signature_columns,
        ),
        axis=1,
    )

    signature_to_group = {}
    groups = []

    for sig in df["feature_signature"]:
        if sig not in signature_to_group:
            signature_to_group[sig] = (
                f"group_{len(signature_to_group) + 1:03d}"
            )

        groups.append(
            signature_to_group[sig]
        )

    df["leakage_group_id"] = groups

    duplicate_groups = []

    for group_id, group in df.groupby(
        "leakage_group_id",
        sort=False,
    ):
        if len(group) > 1:
            duplicate_groups.append(
                {
                    "leakage_group_id":
                        group_id,

                    "rows":
                        int(len(group)),

                    "comparison_ids":
                        group[
                            "comparison_id"
                        ]
                        .astype(str)
                        .tolist(),

                    "reference_names":
                        (
                            group[
                                "reference_name"
                            ]
                            .astype(str)
                            .tolist()
                            if "reference_name"
                            in group.columns
                            else []
                        ),

                    "comparison_names":
                        (
                            group[
                                "comparison_name"
                            ]
                            .astype(str)
                            .tolist()
                            if "comparison_name"
                            in group.columns
                            else []
                        ),
                }
            )

    cleaned = df.drop(
        columns=all_null_columns
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cleaned.to_csv(
        output_path,
        index=False,
    )

    report = {
        "input_csv":
            str(input_path),

        "output_csv":
            str(output_path),

        "rows":
            int(len(df)),

        "original_feature_columns":
            int(len(feature_columns)),

        "numeric_feature_columns":
            int(
                len(
                    numeric_feature_columns
                )
            ),

        "categorical_feature_columns":
            categorical_feature_columns,

        "all_null_feature_columns_removed":
            all_null_columns,

        "constant_feature_columns_flagged":
            constant_columns,

        "missing_values_by_column":
            missing_by_column,

        "unique_leakage_groups":
            int(
                df[
                    "leakage_group_id"
                ].nunique()
            ),

        "duplicate_feature_groups":
            duplicate_groups,

        "duplicate_rows_total":
            int(
                sum(
                    group["rows"]
                    for group
                    in duplicate_groups
                )
            ),

        "notes": [
            (
                "leakage_group_id groups exact "
                "numeric feature duplicates."
            ),
            (
                "Future train/test splits must "
                "keep a leakage group in one "
                "split only."
            ),
            (
                "No target label has been "
                "created and no model is "
                "trained here."
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

    print("=" * 68)
    print(
        "MotionStage Stage 18B - "
        "Dataset Validation"
    )
    print("=" * 68)

    print(
        f"Rows                     : "
        f"{len(df)}"
    )

    print(
        "Original feature columns : "
        f"{len(feature_columns)}"
    )

    print(
        "Numeric feature columns  : "
        f"{len(numeric_feature_columns)}"
    )

    print(
        "All-null columns removed : "
        f"{len(all_null_columns)}"
    )

    print(
        "Constant columns flagged : "
        f"{len(constant_columns)}"
    )

    print(
        "Unique leakage groups    : "
        f"{df['leakage_group_id'].nunique()}"
    )

    print(
        "Duplicate groups         : "
        f"{len(duplicate_groups)}"
    )

    print(
        f"Validated CSV            : "
        f"{output_path}"
    )

    print(
        f"Validation report        : "
        f"{report_path}"
    )

    if all_null_columns:
        print(
            "\nAll-null columns removed:"
        )

        for c in all_null_columns:
            print(f"  - {c}")

    if constant_columns:
        print(
            "\nConstant columns flagged:"
        )

        for c in constant_columns:
            print(f"  - {c}")

    if duplicate_groups:
        print(
            "\nDuplicate / "
            "leakage-sensitive groups:"
        )

        for group in duplicate_groups:
            print(
                f"  "
                f"{group['leakage_group_id']}: "
                + " | ".join(
                    group["comparison_ids"]
                )
            )

    print("\nImportant:")

    print(
        "  Keep each leakage_group_id "
        "entirely inside one future "
        "split/fold."
    )

    print(
        "  We still need a defensible "
        "target label before "
        "supervised ML."
    )

    print("=" * 68)


if __name__ == "__main__":
    main()
