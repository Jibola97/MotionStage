from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


ALIGNMENT_FEATURES = [
    "left_elbow_angle_deg",
    "right_elbow_angle_deg",
    "left_shoulder_angle_deg",
    "right_shoulder_angle_deg",
    "torso_lean_deg",
]


def standardise_pair(
    sequence_a: np.ndarray,
    sequence_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Standardise both performances using shared
    mean/std values so the feature scales remain
    directly comparable.
    """
    combined = np.vstack(
        [sequence_a, sequence_b]
    )

    mean = np.nanmean(
        combined,
        axis=0,
    )

    std = np.nanstd(
        combined,
        axis=0,
    )

    # Prevent division by zero for near-static features.
    std[std < 1e-8] = 1.0

    sequence_a_z = (
        sequence_a - mean
    ) / std

    sequence_b_z = (
        sequence_b - mean
    ) / std

    return sequence_a_z, sequence_b_z


def dynamic_time_warping(
    sequence_a: np.ndarray,
    sequence_b: np.ndarray,
):
    """
    Classic DTW using Euclidean distance
    between feature vectors.
    """
    local_cost = cdist(
        sequence_a,
        sequence_b,
        metric="euclidean",
    )

    n, m = local_cost.shape

    accumulated_cost = np.full(
        (n + 1, m + 1),
        np.inf,
    )

    accumulated_cost[0, 0] = 0.0

    # Build accumulated-cost matrix.
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            accumulated_cost[i, j] = (
                local_cost[i - 1, j - 1]
                + min(
                    accumulated_cost[i - 1, j],
                    accumulated_cost[i, j - 1],
                    accumulated_cost[i - 1, j - 1],
                )
            )

    # Backtrack to recover optimal alignment.
    i = n
    j = m

    path = []

    while i > 0 and j > 0:
        path.append(
            (i - 1, j - 1)
        )

        options = [
            accumulated_cost[i - 1, j - 1],
            accumulated_cost[i - 1, j],
            accumulated_cost[i, j - 1],
        ]

        move = int(
            np.argmin(options)
        )

        if move == 0:
            i -= 1
            j -= 1
        elif move == 1:
            i -= 1
        else:
            j -= 1

    while i > 0:
        path.append(
            (i - 1, 0)
        )
        i -= 1

    while j > 0:
        path.append(
            (0, j - 1)
        )
        j -= 1

    path.reverse()

    total_cost = accumulated_cost[n, m]

    mean_path_cost = (
        total_cost / len(path)
    )

    return (
        path,
        local_cost,
        total_cost,
        mean_path_cost,
    )


def align_performances(
    reference_csv: str,
    comparison_csv: str,
    output_csv: str,
) -> None:
    reference_path = Path(
        reference_csv
    )

    comparison_path = Path(
        comparison_csv
    )

    output_path = Path(
        output_csv
    )

    if not reference_path.exists():
        raise FileNotFoundError(
            f"Reference file not found: "
            f"{reference_path}"
        )

    if not comparison_path.exists():
        raise FileNotFoundError(
            f"Comparison file not found: "
            f"{comparison_path}"
        )

    reference_df = pd.read_csv(
        reference_path
    )

    comparison_df = pd.read_csv(
        comparison_path
    )

    missing_reference = (
        set(ALIGNMENT_FEATURES)
        - set(reference_df.columns)
    )

    missing_comparison = (
        set(ALIGNMENT_FEATURES)
        - set(comparison_df.columns)
    )

    if missing_reference:
        raise ValueError(
            "Reference performance is missing: "
            f"{sorted(missing_reference)}"
        )

    if missing_comparison:
        raise ValueError(
            "Comparison performance is missing: "
            f"{sorted(missing_comparison)}"
        )

    reference_features = (
        reference_df[
            ALIGNMENT_FEATURES
        ]
        .to_numpy(dtype=float)
    )

    comparison_features = (
        comparison_df[
            ALIGNMENT_FEATURES
        ]
        .to_numpy(dtype=float)
    )

    (
        reference_features_z,
        comparison_features_z,
    ) = standardise_pair(
        reference_features,
        comparison_features,
    )

    (
        path,
        local_cost,
        total_cost,
        mean_path_cost,
    ) = dynamic_time_warping(
        reference_features_z,
        comparison_features_z,
    )

    alignment_rows = []

    for reference_index, comparison_index in path:
        reference_row = (
            reference_df.iloc[
                reference_index
            ]
        )

        comparison_row = (
            comparison_df.iloc[
                comparison_index
            ]
        )

        alignment_rows.append(
            {
                "reference_frame":
                    int(reference_row["frame"]),
                "reference_time":
                    float(
                        reference_row[
                            "timestamp_seconds"
                        ]
                    ),
                "comparison_frame":
                    int(comparison_row["frame"]),
                "comparison_time":
                    float(
                        comparison_row[
                            "timestamp_seconds"
                        ]
                    ),
                "local_cost":
                    float(
                        local_cost[
                            reference_index,
                            comparison_index,
                        ]
                    ),
            }
        )

    alignment_df = pd.DataFrame(
        alignment_rows
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    alignment_df.to_csv(
        output_path,
        index=False,
    )

    reference_duration = float(
        reference_df[
            "timestamp_seconds"
        ].iloc[-1]
    )

    comparison_duration = float(
        comparison_df[
            "timestamp_seconds"
        ].iloc[-1]
    )

    duration_ratio = (
        comparison_duration
        / reference_duration
        if reference_duration > 0
        else np.nan
    )

    print(
        "MotionStage Temporal Alignment"
    )
    print(
        "------------------------------"
    )

    print(
        f"Reference frames: "
        f"{len(reference_df)}"
    )

    print(
        f"Comparison frames: "
        f"{len(comparison_df)}"
    )

    print(
        f"Reference duration: "
        f"{reference_duration:.2f}s"
    )

    print(
        f"Comparison duration: "
        f"{comparison_duration:.2f}s"
    )

    print(
        f"Duration ratio: "
        f"{duration_ratio:.3f}"
    )

    print(
        f"DTW path length: "
        f"{len(path)}"
    )

    print(
        f"Total DTW cost: "
        f"{total_cost:.3f}"
    )

    print(
        f"Mean path cost: "
        f"{mean_path_cost:.3f}"
    )

    print(
        f"Alignment saved to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    align_performances(
        reference_csv=(
            "data/processed/"
            "test_performance_01/"
            "frame_features.csv"
        ),
        comparison_csv=(
            "data/processed/"
            "test_performance_02/"
            "frame_features.csv"
        ),
        output_csv=(
            "data/comparisons/"
            "test_performance_01_vs_02/"
            "dtw_alignment.csv"
        ),
    )
