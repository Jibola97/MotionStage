import json
from pathlib import Path

import numpy as np
import pandas as pd


REGIONS = {
    "left_arm": {
        "pose_features": [
            "left_elbow_angle_deg",
            "left_shoulder_angle_deg",
        ],
        "joints": [
            "left_shoulder",
            "left_elbow",
            "left_wrist",
        ],
        "speed_features": [
            "left_arm_mean_speed",
        ],
    },

    "right_arm": {
        "pose_features": [
            "right_elbow_angle_deg",
            "right_shoulder_angle_deg",
        ],
        "joints": [
            "right_shoulder",
            "right_elbow",
            "right_wrist",
        ],
        "speed_features": [
            "right_arm_mean_speed",
        ],
    },

    "lower_body": {
        "pose_features": [
            "left_knee_angle_deg",
            "right_knee_angle_deg",
        ],
        "joints": [
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        ],
        "speed_features": [
            "left_leg_mean_speed",
            "right_leg_mean_speed",
        ],
    },

    "torso": {
        "pose_features": [
            "torso_lean_deg",
        ],
        "joints": [
            "left_shoulder",
            "right_shoulder",
            "left_hip",
            "right_hip",
        ],
        "speed_features": [
            "torso_mean_speed",
        ],
    },
}


ANGLE_MAX_DIFFERENCE = {
    "left_elbow_angle_deg": 180.0,
    "right_elbow_angle_deg": 180.0,
    "left_shoulder_angle_deg": 180.0,
    "right_shoulder_angle_deg": 180.0,
    "left_knee_angle_deg": 180.0,
    "right_knee_angle_deg": 180.0,
    "torso_lean_deg": 90.0,
}


# ---------------------------------------------------------
# Regional comparison configuration
# ---------------------------------------------------------

REGION_POSE_WEIGHT = 0.40
REGION_POSITION_WEIGHT = 0.40
REGION_SPEED_WEIGHT = 0.20

# A position error of 1.5 torso lengths or greater
# maps to zero position similarity.
#
# This is provisional and will later be calibrated.
POSITION_MAX_DISTANCE = 1.5

STATIONARY_SPEED_THRESHOLD = 0.05


# ---------------------------------------------------------
# Keep the original overall score for comparison with V1.
# ---------------------------------------------------------

OVERALL_POSE_WEIGHT = 0.60
OVERALL_SPEED_WEIGHT = 0.25
OVERALL_DURATION_WEIGHT = 0.15


def clamp_score(value: float) -> float:
    return float(
        np.clip(
            value,
            0.0,
            100.0,
        )
    )


def mean_or_nan(values) -> float:
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan

    return float(
        np.mean(values)
    )


def angle_similarity(
    reference_value: float,
    comparison_value: float,
    max_difference: float,
) -> float:

    difference = abs(
        float(reference_value)
        - float(comparison_value)
    )

    similarity = (
        1.0
        - difference / max_difference
    ) * 100.0

    return clamp_score(
        similarity
    )


def speed_similarity(
    reference_speed: float,
    comparison_speed: float,
) -> float:

    a = abs(
        float(reference_speed)
    )

    b = abs(
        float(comparison_speed)
    )

    if (
        a < STATIONARY_SPEED_THRESHOLD
        and
        b < STATIONARY_SPEED_THRESHOLD
    ):
        return 100.0

    maximum = max(
        a,
        b,
    )

    if maximum == 0:
        return 100.0

    minimum = min(
        a,
        b,
    )

    return clamp_score(
        minimum / maximum * 100.0
    )


def position_similarity(
    distance: float,
) -> float:
    """
    Distance is measured in torso-length units.

    0 distance -> 100
    1.5 torso lengths or greater -> 0
    """

    similarity = (
        1.0
        - distance / POSITION_MAX_DISTANCE
    ) * 100.0

    return clamp_score(
        similarity
    )


def get_joint_position(
    keypoints: pd.DataFrame,
    frame: int,
    joint_name: str,
) -> np.ndarray:

    try:
        row = keypoints.loc[
            (frame, joint_name)
        ]
    except KeyError:
        return np.array(
            [np.nan, np.nan]
        )

    if isinstance(
        row,
        pd.DataFrame,
    ):
        row = row.iloc[0]

    return np.array(
        [
            float(row["x_norm"]),
            float(row["y_norm"]),
        ]
    )


def calculate_region_position_similarity(
    reference_keypoints: pd.DataFrame,
    comparison_keypoints: pd.DataFrame,
    reference_frame: int,
    comparison_frame: int,
    joints: list[str],
) -> float:

    joint_scores = []

    for joint_name in joints:

        reference_position = (
            get_joint_position(
                reference_keypoints,
                reference_frame,
                joint_name,
            )
        )

        comparison_position = (
            get_joint_position(
                comparison_keypoints,
                comparison_frame,
                joint_name,
            )
        )

        if (
            np.isnan(
                reference_position
            ).any()
            or
            np.isnan(
                comparison_position
            ).any()
        ):
            continue

        distance = float(
            np.linalg.norm(
                reference_position
                - comparison_position
            )
        )

        joint_scores.append(
            position_similarity(
                distance
            )
        )

    return mean_or_nan(
        joint_scores
    )


def calculate_similarity(
    reference_csv: str,
    comparison_csv: str,
    reference_keypoints_csv: str,
    comparison_keypoints_csv: str,
    alignment_csv: str,
    output_json: str,
    output_csv: str,
) -> None:

    reference = pd.read_csv(
        reference_csv
    )

    comparison = pd.read_csv(
        comparison_csv
    )

    reference_keypoints = (
        pd.read_csv(
            reference_keypoints_csv
        )
        .set_index(
            [
                "frame",
                "joint_name",
            ]
        )
    )

    comparison_keypoints = (
        pd.read_csv(
            comparison_keypoints_csv
        )
        .set_index(
            [
                "frame",
                "joint_name",
            ]
        )
    )

    alignment = pd.read_csv(
        alignment_csv
    )

    reference_indexed = (
        reference.set_index("frame")
    )

    comparison_indexed = (
        comparison.set_index("frame")
    )

    rows = []

    for step, alignment_row in alignment.iterrows():

        reference_frame = int(
            alignment_row[
                "reference_frame"
            ]
        )

        comparison_frame = int(
            alignment_row[
                "comparison_frame"
            ]
        )

        ref = reference_indexed.loc[
            reference_frame
        ]

        comp = comparison_indexed.loc[
            comparison_frame
        ]

        row = {
            "alignment_step":
                int(step),

            "reference_frame":
                reference_frame,

            "reference_time":
                float(
                    alignment_row[
                        "reference_time"
                    ]
                ),

            "comparison_frame":
                comparison_frame,

            "comparison_time":
                float(
                    alignment_row[
                        "comparison_time"
                    ]
                ),
        }

        regional_pose_scores = []
        regional_position_scores = []
        regional_speed_scores = []
        regional_composite_scores = []

        for region_name, config in REGIONS.items():

            # -----------------------------------------
            # Pose / angle similarity
            # -----------------------------------------

            pose_scores = []

            for feature in config[
                "pose_features"
            ]:

                score = angle_similarity(
                    ref[feature],
                    comp[feature],
                    ANGLE_MAX_DIFFERENCE[
                        feature
                    ],
                )

                pose_scores.append(
                    score
                )

            region_pose = mean_or_nan(
                pose_scores
            )

            # -----------------------------------------
            # Normalised joint-position similarity
            # -----------------------------------------

            region_position = (
                calculate_region_position_similarity(
                    reference_keypoints,
                    comparison_keypoints,
                    reference_frame,
                    comparison_frame,
                    config["joints"],
                )
            )

            # -----------------------------------------
            # Regional movement-speed similarity
            # -----------------------------------------

            speed_scores = []

            for feature in config[
                "speed_features"
            ]:

                score = speed_similarity(
                    ref[feature],
                    comp[feature],
                )

                speed_scores.append(
                    score
                )

            region_speed = mean_or_nan(
                speed_scores
            )

            # -----------------------------------------
            # Composite regional similarity
            # -----------------------------------------

            region_composite = (
                REGION_POSE_WEIGHT
                * region_pose

                + REGION_POSITION_WEIGHT
                * region_position

                + REGION_SPEED_WEIGHT
                * region_speed
            )

            region_composite = clamp_score(
                region_composite
            )

            row[
                f"{region_name}_pose_similarity"
            ] = region_pose

            row[
                f"{region_name}_position_similarity"
            ] = region_position

            row[
                f"{region_name}_speed_similarity"
            ] = region_speed

            row[
                f"{region_name}_composite_similarity"
            ] = region_composite

            regional_pose_scores.append(
                region_pose
            )

            regional_position_scores.append(
                region_position
            )

            regional_speed_scores.append(
                region_speed
            )

            regional_composite_scores.append(
                region_composite
            )

        # ---------------------------------------------
        # Whole-body step summaries
        # ---------------------------------------------

        row["pose_similarity"] = (
            mean_or_nan(
                regional_pose_scores
            )
        )

        row["position_similarity"] = (
            mean_or_nan(
                regional_position_scores
            )
        )

        row[
            "movement_speed_similarity"
        ] = mean_or_nan(
            regional_speed_scores
        )

        row[
            "regional_composite_similarity"
        ] = mean_or_nan(
            regional_composite_scores
        )

        rows.append(
            row
        )

    step_df = pd.DataFrame(
        rows
    )

    # -------------------------------------------------
    # Performance-level summaries
    # -------------------------------------------------

    pose_similarity = mean_or_nan(
        step_df[
            "pose_similarity"
        ]
    )

    position_similarity = mean_or_nan(
        step_df[
            "position_similarity"
        ]
    )

    movement_speed_similarity = mean_or_nan(
        step_df[
            "movement_speed_similarity"
        ]
    )

    regional_composite_similarity = (
        mean_or_nan(
            step_df[
                "regional_composite_similarity"
            ]
        )
    )

    reference_duration = float(
        reference[
            "timestamp_seconds"
        ].iloc[-1]
    )

    comparison_duration = float(
        comparison[
            "timestamp_seconds"
        ].iloc[-1]
    )

    duration_similarity = (
        min(
            reference_duration,
            comparison_duration,
        )
        / max(
            reference_duration,
            comparison_duration,
        )
        * 100.0
    )

    # Preserve V1 overall score for comparison.
    overall_similarity = (
        OVERALL_POSE_WEIGHT
        * pose_similarity

        + OVERALL_SPEED_WEIGHT
        * movement_speed_similarity

        + OVERALL_DURATION_WEIGHT
        * duration_similarity
    )

    overall_similarity = clamp_score(
        overall_similarity
    )

    report = {
        "overall_similarity":
            round(
                overall_similarity,
                2,
            ),

        "pose_similarity":
            round(
                pose_similarity,
                2,
            ),

        "position_similarity":
            round(
                position_similarity,
                2,
            ),

        "movement_speed_similarity":
            round(
                movement_speed_similarity,
                2,
            ),

        "regional_composite_similarity":
            round(
                regional_composite_similarity,
                2,
            ),

        "duration_similarity":
            round(
                duration_similarity,
                2,
            ),

        "regions": {},

        "regional_weights": {
            "pose":
                REGION_POSE_WEIGHT,

            "position":
                REGION_POSITION_WEIGHT,

            "speed":
                REGION_SPEED_WEIGHT,
        },

        "position_max_distance":
            POSITION_MAX_DISTANCE,

        "alignment_steps":
            int(
                len(step_df)
            ),

        "notes": (
            "Similarity values are baseline indices, "
            "not probabilities. Regional composite "
            "weights and position scaling are "
            "provisional."
        ),
    }

    for region_name in REGIONS:

        report["regions"][
            region_name
        ] = {

            "pose_similarity":
                round(
                    mean_or_nan(
                        step_df[
                            f"{region_name}"
                            "_pose_similarity"
                        ]
                    ),
                    2,
                ),

            "position_similarity":
                round(
                    mean_or_nan(
                        step_df[
                            f"{region_name}"
                            "_position_similarity"
                        ]
                    ),
                    2,
                ),

            "speed_similarity":
                round(
                    mean_or_nan(
                        step_df[
                            f"{region_name}"
                            "_speed_similarity"
                        ]
                    ),
                    2,
                ),

            "composite_similarity":
                round(
                    mean_or_nan(
                        step_df[
                            f"{region_name}"
                            "_composite_similarity"
                        ]
                    ),
                    2,
                ),
        }

    output_json_path = Path(
        output_json
    )

    output_csv_path = Path(
        output_csv
    )

    output_json_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_json_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
        )

    step_df.to_csv(
        output_csv_path,
        index=False,
    )

    print()
    print(
        "MotionStage Similarity Report V2"
    )
    print(
        "--------------------------------"
    )

    print(
        f"Overall baseline:          "
        f"{overall_similarity:6.2f}"
    )

    print(
        f"Pose similarity:           "
        f"{pose_similarity:6.2f}"
    )

    print(
        f"Position similarity:       "
        f"{position_similarity:6.2f}"
    )

    print(
        f"Movement-speed similarity: "
        f"{movement_speed_similarity:6.2f}"
    )

    print(
        f"Regional composite:        "
        f"{regional_composite_similarity:6.2f}"
    )

    print(
        f"Duration similarity:       "
        f"{duration_similarity:6.2f}"
    )

    print()
    print(
        "Regional breakdown"
    )
    print(
        "------------------"
    )

    for region_name in REGIONS:

        region = report[
            "regions"
        ][region_name]

        print(
            f"{region_name:12} "
            f"pose={region['pose_similarity']:6.2f}  "
            f"position={region['position_similarity']:6.2f}  "
            f"speed={region['speed_similarity']:6.2f}  "
            f"composite={region['composite_similarity']:6.2f}"
        )

    print()

    print(
        f"Report: {output_json_path}"
    )

    print(
        f"Step data: {output_csv_path}"
    )


if __name__ == "__main__":
    calculate_similarity(
        reference_csv=(
            "data/processed/"
            "test_performance_01/"
            "frame_features.csv"
        ),

        comparison_csv=(
            "data/processed/"
            "test_performance_03/"
            "frame_features.csv"
        ),

        reference_keypoints_csv=(
            "data/processed/"
            "test_performance_01/"
            "keypoints_normalised.csv"
        ),

        comparison_keypoints_csv=(
            "data/processed/"
            "test_performance_03/"
            "keypoints_normalised.csv"
        ),

        alignment_csv=(
            "data/comparisons/"
            "test_performance_01_vs_test_performance_03/"
            "dtw_alignment.csv"
        ),

        output_json=(
            "data/comparisons/"
            "test_performance_01_vs_test_performance_03/"
            "similarity_report.json"
        ),

        output_csv=(
            "data/comparisons/"
            "test_performance_01_vs_test_performance_03/"
            "similarity_by_step.csv"
        ),
    )
