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


def detect_comparison_divergence(
    similarity_csv: str,
    output_csv: str,
    output_json: str,
    output_plot: str,
) -> None:

    input_path = Path(
        similarity_csv
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Similarity file not found: "
            f"{input_path}"
        )

    df = pd.read_csv(
        input_path
    )

    # -------------------------------------------------
    # IMPORTANT:
    #
    # We now collapse by COMPARISON frame rather than
    # reference frame.
    #
    # This preserves inserted movements occurring in
    # the comparison performance.
    # -------------------------------------------------

    numeric_columns = [
        column
        for column in df.columns
        if column not in {
            "alignment_step",
            "reference_frame",
            "comparison_frame",
        }
    ]

    comparison_df = (
        df.groupby(
            "comparison_frame",
            as_index=False,
        )[numeric_columns]
        .mean()
        .sort_values(
            "comparison_frame"
        )
        .reset_index(
            drop=True
        )
    )

    # -------------------------------------------------
    # Estimate comparison FPS.
    # -------------------------------------------------

    times = np.sort(
        comparison_df[
            "comparison_time"
        ].unique()
    )

    differences = np.diff(
        times
    )

    valid_differences = (
        differences[
            differences > 0
        ]
    )

    if len(valid_differences) == 0:
        raise ValueError(
            "Could not estimate comparison FPS."
        )

    frame_interval = float(
        np.median(
            valid_differences
        )
    )

    estimated_fps = (
        1.0
        / frame_interval
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
    # Calculate rolling similarity independently
    # for every body region.
    # -------------------------------------------------

    for region, column in (
        REGION_COLUMNS.items()
    ):

        rolling_column = (
            f"{region}"
            "_window_similarity"
        )

        comparison_df[
            rolling_column
        ] = (
            comparison_df[
                column
            ]
            .rolling(
                window=window_frames,
                center=True,
                min_periods=window_frames,
            )
            .mean()
        )

    # -------------------------------------------------
    # Find the strongest sustained regional anomaly.
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
            comparison_df[
                column
            ]
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
            most_divergent_region = region

    if worst_index is None:
        raise ValueError(
            "No complete comparison windows found."
        )

    worst_row = (
        comparison_df.loc[
            worst_index
        ]
    )

    comparison_center = float(
        worst_row[
            "comparison_time"
        ]
    )

    half_window = (
        WINDOW_SECONDS
        / 2.0
    )

    comparison_min_time = float(
        comparison_df[
            "comparison_time"
        ].min()
    )

    comparison_max_time = float(
        comparison_df[
            "comparison_time"
        ].max()
    )

    comparison_start = max(
        comparison_min_time,
        comparison_center
        - half_window,
    )

    comparison_end = min(
        comparison_max_time,
        comparison_center
        + half_window,
    )

    # -------------------------------------------------
    # Determine reference times represented by the
    # selected comparison interval.
    # -------------------------------------------------

    original_window = df[
        (
            df["comparison_time"]
            >= comparison_start
        )
        &
        (
            df["comparison_time"]
            <= comparison_end
        )
    ]

    if original_window.empty:

        reference_start = np.nan
        reference_end = np.nan

    else:

        reference_start = float(
            original_window[
                "reference_time"
            ].min()
        )

        reference_end = float(
            original_window[
                "reference_time"
            ].max()
        )

    # -------------------------------------------------
    # Regional statistics within the selected interval.
    # -------------------------------------------------

    window_df = comparison_df[
        (
            comparison_df[
                "comparison_time"
            ]
            >= comparison_start
        )
        &
        (
            comparison_df[
                "comparison_time"
            ]
            <= comparison_end
        )
    ]

    region_scores = {}

    for region, column in (
        REGION_COLUMNS.items()
    ):

        region_scores[
            region
        ] = float(
            window_df[
                column
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

    comparison_df.to_csv(
        output_csv_path,
        index=False,
    )

    report = {
        "method":
            "robust_dtw_comparison_timeline",

        "window_seconds":
            WINDOW_SECONDS,

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

        "reference_window": {
            "start_seconds":
                (
                    round(
                        reference_start,
                        3,
                    )
                    if np.isfinite(
                        reference_start
                    )
                    else None
                ),

            "end_seconds":
                (
                    round(
                        reference_end,
                        3,
                    )
                    if np.isfinite(
                        reference_end
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
            "Localisation is performed directly on "
            "the comparison-video timeline so repeated "
            "DTW mappings do not average away inserted "
            "comparison movements."
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
    # Plot.
    # -------------------------------------------------

    plt.figure(
        figsize=(12, 6)
    )

    for region in REGION_COLUMNS:

        plt.plot(
            comparison_df[
                "comparison_time"
            ],
            comparison_df[
                f"{region}"
                "_window_similarity"
            ],
            label=region.replace(
                "_",
                " ",
            ).title(),
        )

    plt.axvspan(
        comparison_start,
        comparison_end,
        alpha=0.15,
        label="Largest divergence window",
    )

    plt.ylim(
        0,
        100,
    )

    plt.xlabel(
        "Comparison performance time (seconds)"
    )

    plt.ylabel(
        "Regional similarity index"
    )

    plt.title(
        "MotionStage Comparison-Timeline Divergence"
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
        "MotionStage Comparison-Timeline Divergence"
    )

    print(
        "------------------------------------------"
    )

    print(
        f"Estimated comparison FPS: "
        f"{estimated_fps:.2f}"
    )

    print(
        f"Window: "
        f"{WINDOW_SECONDS:.2f}s "
        f"({window_frames} comparison frames)"
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
        f"Comparison: "
        f"{comparison_start:.2f}s "
        f"to {comparison_end:.2f}s"
    )

    print(
        f"Reference: "
        f"{reference_start:.2f}s "
        f"to {reference_end:.2f}s"
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

    detect_comparison_divergence(
        similarity_csv=(
            "data/comparisons/"
            "test_performance_01_vs_test_performance_03/"
            "robust_localisation/"
            "similarity_by_step.csv"
        ),

        output_csv=(
            "data/comparisons/"
            "test_performance_01_vs_test_performance_03/"
            "robust_localisation/"
            "comparison_divergence_timeline.csv"
        ),

        output_json=(
            "data/comparisons/"
            "test_performance_01_vs_test_performance_03/"
            "robust_localisation/"
            "comparison_divergence_report.json"
        ),

        output_plot=(
            "data/comparisons/"
            "test_performance_01_vs_test_performance_03/"
            "robust_localisation/"
            "plots/"
            "comparison_divergence.png"
        ),
    )
