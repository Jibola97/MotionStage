import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REGION_COLUMNS = {
    "left_arm":
        "left_arm_composite_similarity",

    "right_arm":
        "right_arm_composite_similarity",

    "lower_body":
        "lower_body_composite_similarity",

    "torso":
        "torso_composite_similarity",
}


WINDOW_SECONDS = 0.75


def detect_divergence(
    similarity_csv: str,
    output_csv: str,
    output_json: str,
    output_plot: str,
) -> None:

    similarity_path = Path(
        similarity_csv
    )

    if not similarity_path.exists():
        raise FileNotFoundError(
            f"Similarity CSV not found: "
            f"{similarity_path}"
        )

    df = pd.read_csv(
        similarity_path
    )

    numeric_columns = [
        column
        for column in df.columns
        if column not in {
            "alignment_step",
            "reference_frame",
            "comparison_frame",
        }
    ]

    # Collapse repeated DTW mappings so a reference
    # frame is represented only once on the timeline.
    frame_df = (
        df.groupby(
            "reference_frame",
            as_index=False,
        )[numeric_columns]
        .mean()
        .sort_values(
            "reference_frame"
        )
        .reset_index(
            drop=True
        )
    )

    unique_times = np.sort(
        frame_df[
            "reference_time"
        ].unique()
    )

    time_differences = np.diff(
        unique_times
    )

    valid_differences = (
        time_differences[
            time_differences > 0
        ]
    )

    if len(
        valid_differences
    ) == 0:
        raise ValueError(
            "Could not estimate frame interval."
        )

    median_frame_interval = float(
        np.median(
            valid_differences
        )
    )

    estimated_fps = (
        1.0
        / median_frame_interval
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
    # Calculate a complete rolling window separately
    # for every region.
    # -------------------------------------------------

    for region, column in (
        REGION_COLUMNS.items()
    ):

        frame_df[
            f"{region}_window_similarity"
        ] = (
            frame_df[
                column
            ]
            .rolling(
                window=window_frames,
                center=True,
                min_periods=window_frames,
            )
            .mean()
        )

    # Overall regional composite, useful for display.
    frame_df[
        "overall_regional_similarity"
    ] = frame_df[
        list(
            REGION_COLUMNS.values()
        )
    ].mean(
        axis=1
    )

    frame_df[
        "overall_window_similarity"
    ] = (
        frame_df[
            "overall_regional_similarity"
        ]
        .rolling(
            window=window_frames,
            center=True,
            min_periods=window_frames,
        )
        .mean()
    )

    # -------------------------------------------------
    # Search all body regions for the strongest
    # sustained divergence.
    # -------------------------------------------------

    most_divergent_region = None
    worst_index = None
    worst_score = np.inf

    for region in REGION_COLUMNS:

        column = (
            f"{region}"
            "_window_similarity"
        )

        valid = (
            frame_df[
                column
            ]
            .dropna()
        )

        if valid.empty:
            continue

        region_index = (
            valid.idxmin()
        )

        region_score = float(
            valid.loc[
                region_index
            ]
        )

        if region_score < worst_score:

            worst_score = (
                region_score
            )

            worst_index = (
                region_index
            )

            most_divergent_region = (
                region
            )

    if worst_index is None:
        raise ValueError(
            "No complete divergence "
            "windows available."
        )

    worst_row = frame_df.loc[
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

    reference_max_time = float(
        frame_df[
            "reference_time"
        ].max()
    )

    reference_start = max(
        0.0,
        reference_center
        - half_window,
    )

    reference_end = min(
        reference_max_time,
        reference_center
        + half_window,
    )

    # -------------------------------------------------
    # Convert the reference window back through DTW
    # to the corresponding comparison-video time.
    # -------------------------------------------------

    original_window = df[
        (
            df["reference_time"]
            >= reference_start
        )
        &
        (
            df["reference_time"]
            <= reference_end
        )
    ]

    if original_window.empty:

        comparison_start = np.nan
        comparison_end = np.nan

    else:

        comparison_start = float(
            original_window[
                "comparison_time"
            ].min()
        )

        comparison_end = float(
            original_window[
                "comparison_time"
            ].max()
        )

    # -------------------------------------------------
    # Regional scores inside the detected window.
    # -------------------------------------------------

    reference_window_df = (
        frame_df[
            (
                frame_df[
                    "reference_time"
                ]
                >= reference_start
            )
            &
            (
                frame_df[
                    "reference_time"
                ]
                <= reference_end
            )
        ]
    )

    region_scores = {}

    for region, column in (
        REGION_COLUMNS.items()
    ):

        region_scores[
            region
        ] = float(
            reference_window_df[
                column
            ].mean()
        )

    # Components of the winning region.
    culprit_pose = float(
        reference_window_df[
            f"{most_divergent_region}"
            "_pose_similarity"
        ].mean()
    )

    culprit_position = float(
        reference_window_df[
            f"{most_divergent_region}"
            "_position_similarity"
        ].mean()
    )

    culprit_speed = float(
        reference_window_df[
            f"{most_divergent_region}"
            "_speed_similarity"
        ].mean()
    )

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

    frame_df.to_csv(
        output_csv_path,
        index=False,
    )

    report = {
        "window_seconds":
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
                (
                    round(
                        comparison_start,
                        3,
                    )
                    if np.isfinite(
                        comparison_start
                    )
                    else None
                ),

            "end_seconds":
                (
                    round(
                        comparison_end,
                        3,
                    )
                    if np.isfinite(
                        comparison_end
                    )
                    else None
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
            "Divergence is selected from regional "
            "composite similarity using only complete "
            "rolling windows."
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
    # Regional similarity plot
    # -------------------------------------------------

    plt.figure(
        figsize=(12, 6)
    )

    for region in REGION_COLUMNS:

        plt.plot(
            frame_df[
                "reference_time"
            ],
            frame_df[
                f"{region}"
                "_window_similarity"
            ],
            label=region.replace(
                "_",
                " ",
            ).title(),
        )

    plt.axvspan(
        reference_start,
        reference_end,
        alpha=0.15,
        label=(
            "Largest divergence window"
        ),
    )

    plt.ylim(
        0,
        100,
    )

    plt.xlabel(
        "Reference performance time "
        "(seconds)"
    )

    plt.ylabel(
        "Regional similarity index"
    )

    plt.title(
        "MotionStage Regional Similarity Timeline"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_plot_path,
        dpi=150,
    )

    plt.close()

    print()
    print(
        "MotionStage Divergence Detection V2"
    )

    print(
        "-----------------------------------"
    )

    print(
        f"Estimated FPS: "
        f"{estimated_fps:.2f}"
    )

    print(
        f"Analysis window: "
        f"{WINDOW_SECONDS:.2f}s "
        f"({window_frames} frames)"
    )

    print()
    print(
        "Largest regional divergence"
    )

    print(
        "---------------------------"
    )

    print(
        f"Reference:  "
        f"{reference_start:.2f}s "
        f"to {reference_end:.2f}s"
    )

    print(
        f"Comparison: "
        f"{comparison_start:.2f}s "
        f"to {comparison_end:.2f}s"
    )

    print(
        f"Main contributor: "
        f"{most_divergent_region}"
    )

    print(
        f"Regional window similarity: "
        f"{worst_score:.2f}"
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

    detect_divergence(
        similarity_csv=(
            "data/comparisons/"
            "test_performance_01_vs_test_performance_03/"
            "similarity_by_step.csv"
        ),

        output_csv=(
            "data/comparisons/"
            "test_performance_01_vs_test_performance_03/"
            "divergence_timeline.csv"
        ),

        output_json=(
            "data/comparisons/"
            "test_performance_01_vs_test_performance_03/"
            "divergence_report.json"
        ),

        output_plot=(
            "data/comparisons/"
            "test_performance_01_vs_test_performance_03/"
            "plots/"
            "regional_similarity_timeline.png"
        ),
    )
