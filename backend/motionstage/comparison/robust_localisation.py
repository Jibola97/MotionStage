from pathlib import Path

import numpy as np
import pandas as pd

from backend.motionstage.comparison.similarity_engine import (
    calculate_similarity,
)
from backend.motionstage.comparison.detect_divergence import (
    detect_divergence,
)


WARP_BAND_RATIO = 0.15


REGION_ALIGNMENT_FEATURES = {
    "left_arm": [
        "left_elbow_angle_deg",
        "left_shoulder_angle_deg",
    ],

    "right_arm": [
        "right_elbow_angle_deg",
        "right_shoulder_angle_deg",
    ],

    "lower_body": [
        "left_knee_angle_deg",
        "right_knee_angle_deg",
    ],

    "torso": [
        "torso_lean_deg",
    ],
}


def standardise_features(
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
):
    """
    Standardise each alignment feature using
    statistics shared across both performances.
    """

    all_features = sorted(
        {
            feature
            for features in REGION_ALIGNMENT_FEATURES.values()
            for feature in features
        }
    )

    reference_values = (
        reference[
            all_features
        ].to_numpy(dtype=float)
    )

    comparison_values = (
        comparison[
            all_features
        ].to_numpy(dtype=float)
    )

    combined = np.vstack(
        [
            reference_values,
            comparison_values,
        ]
    )

    mean = np.nanmean(
        combined,
        axis=0,
    )

    std = np.nanstd(
        combined,
        axis=0,
    )

    std[
        std < 1e-8
    ] = 1.0

    reference_z = (
        reference_values - mean
    ) / std

    comparison_z = (
        comparison_values - mean
    ) / std

    feature_indices = {
        feature:
            all_features.index(
                feature
            )
        for feature in all_features
    }

    return (
        reference_z,
        comparison_z,
        feature_indices,
    )


def calculate_robust_local_cost(
    reference_z: np.ndarray,
    comparison_z: np.ndarray,
    feature_indices: dict,
) -> np.ndarray:
    """
    Calculate alignment cost for every frame pair.

    Cost is calculated separately for four body
    regions. The worst regional cost is discarded,
    and the remaining three determine alignment.

    This makes alignment robust to a local anomaly
    affecting one body region.
    """

    n = len(reference_z)
    m = len(comparison_z)

    local_cost = np.full(
        (n, m),
        np.inf,
    )

    for i in range(n):

        for j in range(m):

            region_costs = []

            for features in (
                REGION_ALIGNMENT_FEATURES.values()
            ):

                indices = [
                    feature_indices[
                        feature
                    ]
                    for feature in features
                ]

                difference = (
                    reference_z[
                        i,
                        indices,
                    ]
                    - comparison_z[
                        j,
                        indices,
                    ]
                )

                # RMS distance means regions with two
                # features are comparable to regions
                # containing only one feature.
                region_cost = float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                difference
                            )
                        )
                    )
                )

                region_costs.append(
                    region_cost
                )

            region_costs = sorted(
                region_costs
            )

            # Ignore the single worst matching region.
            robust_cost = float(
                np.mean(
                    region_costs[:3]
                )
            )

            local_cost[
                i,
                j,
            ] = robust_cost

    return local_cost


def constrained_dtw(
    local_cost: np.ndarray,
    band_ratio: float,
):
    n, m = local_cost.shape

    accumulated = np.full(
        (n + 1, m + 1),
        np.inf,
    )

    accumulated[
        0,
        0,
    ] = 0.0

    band_radius = max(
        1,
        int(
            round(
                m * band_ratio
            )
        ),
    )

    for i in range(
        1,
        n + 1,
    ):

        if n > 1:

            expected_j = (
                (i - 1)
                * (m - 1)
                / (n - 1)
            ) + 1

        else:

            expected_j = 1.0

        j_min = max(
            1,
            int(
                np.floor(
                    expected_j
                    - band_radius
                )
            ),
        )

        j_max = min(
            m,
            int(
                np.ceil(
                    expected_j
                    + band_radius
                )
            ),
        )

        for j in range(
            j_min,
            j_max + 1,
        ):

            accumulated[
                i,
                j,
            ] = (
                local_cost[
                    i - 1,
                    j - 1,
                ]
                + min(
                    accumulated[
                        i - 1,
                        j,
                    ],
                    accumulated[
                        i,
                        j - 1,
                    ],
                    accumulated[
                        i - 1,
                        j - 1,
                    ],
                )
            )

    if not np.isfinite(
        accumulated[n, m]
    ):
        raise RuntimeError(
            "No robust DTW path found."
        )

    i = n
    j = m

    path = []

    while i > 0 and j > 0:

        path.append(
            (
                i - 1,
                j - 1,
            )
        )

        options = [
            accumulated[
                i - 1,
                j - 1,
            ],

            accumulated[
                i - 1,
                j,
            ],

            accumulated[
                i,
                j - 1,
            ],
        ]

        move = int(
            np.argmin(
                options
            )
        )

        if move == 0:

            i -= 1
            j -= 1

        elif move == 1:

            i -= 1

        else:

            j -= 1

    path.reverse()

    return (
        path,
        float(
            accumulated[n, m]
        ),
    )


def robust_localisation(
    reference_name: str,
    comparison_name: str,
) -> None:

    root = Path(
        "data/processed"
    )

    reference_dir = (
        root
        / reference_name
    )

    comparison_dir = (
        root
        / comparison_name
    )

    reference_features_path = (
        reference_dir
        / "frame_features.csv"
    )

    comparison_features_path = (
        comparison_dir
        / "frame_features.csv"
    )

    reference_keypoints_path = (
        reference_dir
        / "keypoints_normalised.csv"
    )

    comparison_keypoints_path = (
        comparison_dir
        / "keypoints_normalised.csv"
    )

    reference = pd.read_csv(
        reference_features_path
    )

    comparison = pd.read_csv(
        comparison_features_path
    )

    print()
    print(
        "MotionStage Robust Localisation"
    )

    print(
        "-------------------------------"
    )

    print(
        f"Reference:  {reference_name}"
    )

    print(
        f"Comparison: {comparison_name}"
    )

    print(
        f"Reference frames: "
        f"{len(reference)}"
    )

    print(
        f"Comparison frames: "
        f"{len(comparison)}"
    )

    print(
        f"Warp-band ratio: "
        f"{WARP_BAND_RATIO:.2f}"
    )

    (
        reference_z,
        comparison_z,
        feature_indices,
    ) = standardise_features(
        reference,
        comparison,
    )

    print()
    print(
        "Building robust regional cost matrix..."
    )

    local_cost = (
        calculate_robust_local_cost(
            reference_z,
            comparison_z,
            feature_indices,
        )
    )

    print(
        "Running constrained robust DTW..."
    )

    (
        path,
        total_cost,
    ) = constrained_dtw(
        local_cost,
        WARP_BAND_RATIO,
    )

    comparison_id = (
        f"{reference_name}"
        f"_vs_{comparison_name}"
    )

    output_dir = (
        Path(
            "data/comparisons"
        )
        / comparison_id
        / "robust_localisation"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plots_dir = (
        output_dir
        / "plots"
    )

    plots_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    alignment_csv = (
        output_dir
        / "robust_alignment.csv"
    )

    rows = []

    for (
        reference_index,
        comparison_index,
    ) in path:

        ref = reference.iloc[
            reference_index
        ]

        comp = comparison.iloc[
            comparison_index
        ]

        rows.append(
            {
                "reference_frame":
                    int(
                        ref["frame"]
                    ),

                "reference_time":
                    float(
                        ref[
                            "timestamp_seconds"
                        ]
                    ),

                "comparison_frame":
                    int(
                        comp["frame"]
                    ),

                "comparison_time":
                    float(
                        comp[
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

    pd.DataFrame(
        rows
    ).to_csv(
        alignment_csv,
        index=False,
    )

    print(
        f"Robust path length: "
        f"{len(path)}"
    )

    print(
        f"Total robust cost: "
        f"{total_cost:.3f}"
    )

    print(
        f"Mean robust path cost: "
        f"{total_cost / len(path):.3f}"
    )

    print()

    # -------------------------------------------------
    # Score all regions normally along the robust path.
    # -------------------------------------------------

    similarity_json = (
        output_dir
        / "similarity_report.json"
    )

    similarity_csv = (
        output_dir
        / "similarity_by_step.csv"
    )

    calculate_similarity(
        reference_csv=str(
            reference_features_path
        ),

        comparison_csv=str(
            comparison_features_path
        ),

        reference_keypoints_csv=str(
            reference_keypoints_path
        ),

        comparison_keypoints_csv=str(
            comparison_keypoints_path
        ),

        alignment_csv=str(
            alignment_csv
        ),

        output_json=str(
            similarity_json
        ),

        output_csv=str(
            similarity_csv
        ),
    )

    # -------------------------------------------------
    # Local divergence using the robust alignment.
    # -------------------------------------------------

    divergence_csv = (
        output_dir
        / "divergence_timeline.csv"
    )

    divergence_json = (
        output_dir
        / "divergence_report.json"
    )

    divergence_plot = (
        plots_dir
        / "regional_similarity.png"
    )

    detect_divergence(
        similarity_csv=str(
            similarity_csv
        ),

        output_csv=str(
            divergence_csv
        ),

        output_json=str(
            divergence_json
        ),

        output_plot=str(
            divergence_plot
        ),
    )

    print()
    print(
        "Robust localisation complete."
    )

    print(
        f"Output directory: "
        f"{output_dir}"
    )


if __name__ == "__main__":

    robust_localisation(
        reference_name=(
            "test_performance_01"
        ),

        comparison_name=(
            "test_performance_03"
        ),
    )
