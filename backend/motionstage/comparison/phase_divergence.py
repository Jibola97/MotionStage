import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backend.motionstage.comparison.similarity_engine import (
    REGIONS,
    ANGLE_MAX_DIFFERENCE,
    REGION_POSE_WEIGHT,
    REGION_POSITION_WEIGHT,
    REGION_SPEED_WEIGHT,
    angle_similarity,
    speed_similarity,
    position_similarity,
    mean_or_nan,
)


WINDOW_SECONDS = 0.75


def interpolate_value(
    df: pd.DataFrame,
    target_time: float,
    column: str,
) -> float:
    """
    Interpolate one feature at a requested timestamp.
    """

    return float(
        np.interp(
            target_time,
            df["timestamp_seconds"].to_numpy(dtype=float),
            df[column].to_numpy(dtype=float),
        )
    )


def build_joint_trajectories(
    keypoints_df: pd.DataFrame,
) -> dict:
    """
    Store each joint trajectory separately for
    efficient interpolation.
    """

    trajectories = {}

    for joint_name, joint_df in keypoints_df.groupby(
        "joint_name"
    ):
        trajectories[joint_name] = (
            joint_df
            .sort_values("timestamp_seconds")
            .reset_index(drop=True)
        )

    return trajectories


def interpolate_joint_position(
    trajectories: dict,
    joint_name: str,
    target_time: float,
) -> np.ndarray:

    if joint_name not in trajectories:
        return np.array(
            [np.nan, np.nan]
        )

    joint_df = trajectories[
        joint_name
    ]

    x = np.interp(
        target_time,
        joint_df[
            "timestamp_seconds"
        ].to_numpy(dtype=float),
        joint_df[
            "x_norm"
        ].to_numpy(dtype=float),
    )

    y = np.interp(
        target_time,
        joint_df[
            "timestamp_seconds"
        ].to_numpy(dtype=float),
        joint_df[
            "y_norm"
        ].to_numpy(dtype=float),
    )

    return np.array(
        [float(x), float(y)]
    )


def get_reference_joint_position(
    indexed_keypoints: pd.DataFrame,
    frame_number: int,
    joint_name: str,
) -> np.ndarray:

    try:
        row = indexed_keypoints.loc[
            (
                frame_number,
                joint_name,
            )
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


def calculate_position_similarity(
    reference_keypoints: pd.DataFrame,
    comparison_trajectories: dict,
    reference_frame: int,
    comparison_time: float,
    joints: list[str],
) -> float:

    scores = []

    for joint_name in joints:

        reference_position = (
            get_reference_joint_position(
                reference_keypoints,
                reference_frame,
                joint_name,
            )
        )

        comparison_position = (
            interpolate_joint_position(
                comparison_trajectories,
                joint_name,
                comparison_time,
            )
        )

        if (
            np.isnan(
                reference_position
            ).any()
            or np.isnan(
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

        scores.append(
            position_similarity(
                distance
            )
        )

    return mean_or_nan(
        scores
    )


def detect_phase_divergence(
    reference_features_csv: str,
    comparison_features_csv: str,
    reference_keypoints_csv: str,
    comparison_keypoints_csv: str,
    output_csv: str,
    output_json: str,
    output_plot: str,
) -> None:

    reference = pd.read_csv(
        reference_features_csv
    )

    comparison = pd.read_csv(
        comparison_features_csv
    )

    reference_keypoints_raw = pd.read_csv(
        reference_keypoints_csv
    )

    comparison_keypoints_raw = pd.read_csv(
        comparison_keypoints_csv
    )

    reference_keypoints = (
        reference_keypoints_raw
        .set_index(
            [
                "frame",
                "joint_name",
            ]
        )
    )

    comparison_trajectories = (
        build_joint_trajectories(
            comparison_keypoints_raw
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

    rows = []

    # -------------------------------------------------
    # Compare each reference frame against the same
    # relative progress point in the comparison video.
    # -------------------------------------------------

    for _, reference_row in reference.iterrows():

        reference_frame = int(
            reference_row["frame"]
        )

        reference_time = float(
            reference_row[
                "timestamp_seconds"
            ]
        )

        progress = (
            reference_time
            / reference_duration
            if reference_duration > 0
            else 0.0
        )

        comparison_time = (
            progress
            * comparison_duration
        )

        row = {
            "reference_frame":
                reference_frame,

            "reference_time":
                reference_time,

            "comparison_time":
                comparison_time,

            "progress":
                progress,
        }

        for region_name, config in REGIONS.items():

            # -----------------------------------------
            # Pose / angle similarity
            # -----------------------------------------

            pose_scores = []

            for feature in config[
                "pose_features"
            ]:

                comparison_value = (
                    interpolate_value(
                        comparison,
                        comparison_time,
                        feature,
                    )
                )

                pose_scores.append(
                    angle_similarity(
                        reference_row[
                            feature
                        ],
                        comparison_value,
                        ANGLE_MAX_DIFFERENCE[
                            feature
                        ],
                    )
                )

            region_pose = mean_or_nan(
                pose_scores
            )

            # -----------------------------------------
            # Body-relative position similarity
            # -----------------------------------------

            region_position = (
                calculate_position_similarity(
                    reference_keypoints,
                    comparison_trajectories,
                    reference_frame,
                    comparison_time,
                    config["joints"],
                )
            )

            # -----------------------------------------
            # Movement-speed similarity
            # -----------------------------------------

            speed_scores = []

            for feature in config[
                "speed_features"
            ]:

                comparison_speed = (
                    interpolate_value(
                        comparison,
                        comparison_time,
                        feature,
                    )
                )

                speed_scores.append(
                    speed_similarity(
                        reference_row[
                            feature
                        ],
                        comparison_speed,
                    )
                )

            region_speed = mean_or_nan(
                speed_scores
            )

            # -----------------------------------------
            # Regional composite
            # -----------------------------------------

            regional_composite = (
                REGION_POSE_WEIGHT
                * region_pose

                + REGION_POSITION_WEIGHT
                * region_position

                + REGION_SPEED_WEIGHT
                * region_speed
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
            ] = regional_composite

        rows.append(
            row
        )

    phase_df = pd.DataFrame(
        rows
    )

    # -------------------------------------------------
    # Calculate complete rolling windows.
    # -------------------------------------------------

    reference_times = (
        phase_df[
            "reference_time"
        ].to_numpy(dtype=float)
    )

    differences = np.diff(
        reference_times
    )

    valid_differences = (
        differences[
            differences > 0
        ]
    )

    if len(valid_differences) == 0:
        raise ValueError(
            "Could not determine reference FPS."
        )

    frame_interval = float(
        np.median(
            valid_differences
        )
    )

    estimated_fps = (
        1.0 / frame_interval
    )

    window_frames = max(
        3,
        int(
            round(
                WINDOW_SECONDS
                * estimated_fps
            )
        ),
    )

    if window_frames % 2 == 0:
        window_frames += 1

    # -------------------------------------------------
    # Rolling similarity independently for each region.
    # -------------------------------------------------

    for region_name in REGIONS:

        composite_column = (
            f"{region_name}"
            "_composite_similarity"
        )

        rolling_column = (
            f"{region_name}"
            "_window_similarity"
        )

        phase_df[
            rolling_column
        ] = (
            phase_df[
                composite_column
            ]
            .rolling(
                window=window_frames,
                center=True,
                min_periods=window_frames,
            )
            .mean()
        )

    # -------------------------------------------------
    # Find strongest sustained regional divergence.
    # -------------------------------------------------

    most_divergent_region = None
    worst_index = None
    worst_score = np.inf

    for region_name in REGIONS:

        column = (
            f"{region_name}"
            "_window_similarity"
        )

        valid = (
            phase_df[column]
            .dropna()
        )

        if valid.empty:
            continue

        candidate_index = (
            valid.idxmin()
        )

        candidate_score = float(
            valid.loc[
                candidate_index
            ]
        )

        if candidate_score < worst_score:

            worst_score = candidate_score
            worst_index = candidate_index
            most_divergent_region = region_name

    if worst_index is None:
        raise ValueError(
            "No valid complete phase windows."
        )

    worst_row = phase_df.loc[
        worst_index
    ]

    half_window = (
        WINDOW_SECONDS
        / 2.0
    )

    reference_center = float(
        worst_row[
            "reference_time"
        ]
    )

    reference_start = max(
        0.0,
        reference_center
        - half_window,
    )

    reference_end = min(
        reference_duration,
        reference_center
        + half_window,
    )

    # Convert reference timestamps to normalized
    # sequence progress, then back into comparison time.
    progress_start = (
        reference_start
        / reference_duration
    )

    progress_end = (
        reference_end
        / reference_duration
    )

    comparison_start = (
        progress_start
        * comparison_duration
    )

    comparison_end = (
        progress_end
        * comparison_duration
    )

    # -------------------------------------------------
    # Analyse all regions in detected window.
    # -------------------------------------------------

    window_df = phase_df[
        (
            phase_df[
                "reference_time"
            ]
            >= reference_start
        )
        &
        (
            phase_df[
                "reference_time"
            ]
            <= reference_end
        )
    ]

    region_scores = {}

    for region_name in REGIONS:

        region_scores[
            region_name
        ] = float(
            window_df[
                f"{region_name}"
                "_composite_similarity"
            ].mean()
        )

    culprit_pose = float(
        window_df[
            f"{most_divergent_region}"
            "_pose_similarity"
        ].mean()
    )

    culprit_position = float(
        window_df[
            f"{most_divergent_region}"
            "_position_similarity"
        ].mean()
    )

    culprit_speed = float(
        window_df[
            f"{most_divergent_region}"
            "_speed_similarity"
        ].mean()
    )

    # -------------------------------------------------
    # Save outputs.
    # -------------------------------------------------

    output_csv_path = Path(
        output_csv
    )

    output_json_path = Path(
        output_json
    )

    output_plot_path = Path(
        output_plot
    )

    for path in [
        output_csv_path,
        output_json_path,
        output_plot_path,
    ]:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    phase_df.to_csv(
        output_csv_path,
        index=False,
    )

    report = {
        "method":
            "phase_preserving",

        "window_seconds_reference":
            WINDOW_SECONDS,

        "reference_window": {
            "start_seconds":
                round(
                    reference_start,
                    3,
                ),

            "end_seconds":
                round(
                    reference_end,
                    3,
                ),
        },

        "comparison_window": {
            "start_seconds":
                round(
                    comparison_start,
                    3,
                ),

            "end_seconds":
                round(
                    comparison_end,
                    3,
                ),
        },

        "progress_window": {
            "start":
                round(
                    progress_start,
                    4,
                ),

            "end":
                round(
                    progress_end,
                    4,
                ),
        },

        "most_divergent_region":
            most_divergent_region,

        "worst_regional_window_similarity":
            round(
                worst_score,
                2,
            ),

        "region_similarity": {
            region:
                round(
                    score,
                    2,
                )

            for region, score
            in region_scores.items()
        },

        "divergent_region_components": {
            "pose_similarity":
                round(
                    culprit_pose,
                    2,
                ),

            "position_similarity":
                round(
                    culprit_position,
                    2,
                ),

            "speed_similarity":
                round(
                    culprit_speed,
                    2,
                ),
        },

        "notes": (
            "Phase-preserving localisation compares "
            "equal relative progress through both "
            "performances and does not use DTW."
        ),
    }

    with output_json_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
        )

    # -------------------------------------------------
    # Plot regional phase similarities.
    # -------------------------------------------------

    plt.figure(
        figsize=(12, 6)
    )

    for region_name in REGIONS:

        plt.plot(
            phase_df[
                "reference_time"
            ],
            phase_df[
                f"{region_name}"
                "_window_similarity"
            ],
            label=region_name.replace(
                "_",
                " ",
            ).title(),
        )

    plt.axvspan(
        reference_start,
        reference_end,
        alpha=0.15,
        label="Largest divergence window",
    )

    plt.ylim(
        0,
        100,
    )

    plt.xlabel(
        "Reference performance time (seconds)"
    )

    plt.ylabel(
        "Regional similarity index"
    )

    plt.title(
        "MotionStage Phase-Preserving Divergence"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_plot_path,
        dpi=150,
    )

    plt.close()

    # -------------------------------------------------
    # Terminal summary.
    # -------------------------------------------------

    print()
    print(
        "MotionStage Phase-Preserving Divergence"
    )

    print(
        "---------------------------------------"
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
        f"Estimated reference FPS: "
        f"{estimated_fps:.2f}"
    )

    print(
        f"Window: "
        f"{WINDOW_SECONDS:.2f}s "
        f"({window_frames} reference frames)"
    )

    print()

    print(
        "Largest regional divergence"
    )

    print(
        "---------------------------"
    )

    print(
        f"Main contributor: "
        f"{most_divergent_region}"
    )

    print(
        f"Regional similarity: "
        f"{worst_score:.2f}"
    )

    print(
        f"Reference: "
        f"{reference_start:.2f}s "
        f"to {reference_end:.2f}s"
    )

    print(
        f"Comparison: "
        f"{comparison_start:.2f}s "
        f"to {comparison_end:.2f}s"
    )

    print(
        f"Progress: "
        f"{progress_start * 100:.1f}% "
        f"to "
        f"{progress_end * 100:.1f}%"
    )

    print()

    print(
        "Divergent-region components"
    )

    print(
        "---------------------------"
    )

    print(
        f"Pose:     "
        f"{culprit_pose:.2f}"
    )

    print(
        f"Position: "
        f"{culprit_position:.2f}"
    )

    print(
        f"Speed:    "
        f"{culprit_speed:.2f}"
    )

    print()

    print(
        "Regional similarity in window"
    )

    print(
        "-----------------------------"
    )

    for region, score in sorted(
        region_scores.items(),
        key=lambda item: item[1],
    ):

        print(
            f"{region:15}"
            f"{score:6.2f}"
        )

    print()

    print(
        f"Report: {output_json_path}"
    )

    print(
        f"Timeline: {output_csv_path}"
    )

    print(
        f"Plot: {output_plot_path}"
    )


if __name__ == "__main__":

    detect_phase_divergence(
        reference_features_csv=(
            "data/processed/"
            "test_performance_01/"
            "frame_features.csv"
        ),

        comparison_features_csv=(
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

        output_csv=(
            "data/comparisons/"
            "test_performance_01_vs_test_performance_03/"
            "phase_divergence_timeline.csv"
        ),

        output_json=(
            "data/comparisons/"
            "test_performance_01_vs_test_performance_03/"
            "phase_divergence_report.json"
        ),

        output_plot=(
            "data/comparisons/"
            "test_performance_01_vs_test_performance_03/"
            "plots/"
            "phase_divergence.png"
        ),
    )
