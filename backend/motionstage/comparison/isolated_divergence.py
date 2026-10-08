import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


WINDOW_SECONDS = 0.75

REGIONS = [
    "left_arm",
    "right_arm",
    "lower_body",
    "torso",
]


def detect_isolated_divergence(
    reference_name: str,
    comparison_name: str,
) -> None:

    comparison_id = (
        f"{reference_name}_vs_{comparison_name}"
    )

    base_dir = (
        Path("data/comparisons")
        / comparison_id
        / "robust_localisation"
    )

    similarity_path = (
        base_dir
        / "similarity_by_step.csv"
    )

    if not similarity_path.exists():
        raise FileNotFoundError(
            "Robust-localisation similarity data "
            f"not found: {similarity_path}"
        )

    df = pd.read_csv(
        similarity_path
    )

    # -------------------------------------------------
    # Build timeline using comparison-video frames.
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

    timeline = (
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
    # Estimate comparison-video FPS.
    # -------------------------------------------------

    times = np.sort(
        timeline[
            "comparison_time"
        ].unique()
    )

    differences = np.diff(
        times
    )

    valid_differences = differences[
        differences > 0
    ]

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
    # Rolling similarity for each body region.
    # -------------------------------------------------

    for region in REGIONS:

        source_column = (
            f"{region}"
            "_composite_similarity"
        )

        rolling_column = (
            f"{region}"
            "_window_similarity"
        )

        timeline[
            rolling_column
        ] = (
            timeline[
                source_column
            ]
            .rolling(
                window=window_frames,
                center=True,
                min_periods=window_frames,
            )
            .mean()
        )

    # -------------------------------------------------
    # Bilateral isolation.
    #
    # Positive right-arm gap:
    # left arm matches better than right arm.
    #
    # Positive left-arm gap:
    # right arm matches better than left arm.
    # -------------------------------------------------

    timeline[
        "right_arm_isolation_gap"
    ] = (
        timeline[
            "left_arm_window_similarity"
        ]
        - timeline[
            "right_arm_window_similarity"
        ]
    )

    timeline[
        "left_arm_isolation_gap"
    ] = (
        timeline[
            "right_arm_window_similarity"
        ]
        - timeline[
            "left_arm_window_similarity"
        ]
    )

    right_valid = (
        timeline[
            "right_arm_isolation_gap"
        ]
        .dropna()
    )

    left_valid = (
        timeline[
            "left_arm_isolation_gap"
        ]
        .dropna()
    )

    if (
        right_valid.empty
        or left_valid.empty
    ):
        raise ValueError(
            "No complete isolation windows "
            "were available."
        )

    right_index = (
        right_valid.idxmax()
    )

    left_index = (
        left_valid.idxmax()
    )

    right_gap = float(
        right_valid.loc[
            right_index
        ]
    )

    left_gap = float(
        left_valid.loc[
            left_index
        ]
    )

    # -------------------------------------------------
    # Automatically decide which arm is most isolated.
    # -------------------------------------------------

    if right_gap >= left_gap:

        most_isolated_region = (
            "right_arm"
        )

        peer_region = (
            "left_arm"
        )

        worst_index = (
            right_index
        )

        isolation_gap = (
            right_gap
        )

    else:

        most_isolated_region = (
            "left_arm"
        )

        peer_region = (
            "right_arm"
        )

        worst_index = (
            left_index
        )

        isolation_gap = (
            left_gap
        )

    worst_row = timeline.loc[
        worst_index
    ]

    comparison_center = float(
        worst_row[
            "comparison_time"
        ]
    )

    half_window = (
        WINDOW_SECONDS / 2.0
    )

    minimum_time = float(
        timeline[
            "comparison_time"
        ].min()
    )

    maximum_time = float(
        timeline[
            "comparison_time"
        ].max()
    )

    comparison_start = max(
        minimum_time,
        comparison_center
        - half_window,
    )

    comparison_end = min(
        maximum_time,
        comparison_center
        + half_window,
    )

    # -------------------------------------------------
    # Recover corresponding reference interval.
    # -------------------------------------------------

    aligned_window = df[
        (
            df[
                "comparison_time"
            ]
            >= comparison_start
        )
        &
        (
            df[
                "comparison_time"
            ]
            <= comparison_end
        )
    ]

    if aligned_window.empty:

        reference_start = np.nan
        reference_end = np.nan

    else:

        reference_start = float(
            aligned_window[
                "reference_time"
            ].min()
        )

        reference_end = float(
            aligned_window[
                "reference_time"
            ].max()
        )

    # -------------------------------------------------
    # Regional statistics inside selected window.
    # -------------------------------------------------

    window = timeline[
        (
            timeline[
                "comparison_time"
            ]
            >= comparison_start
        )
        &
        (
            timeline[
                "comparison_time"
            ]
            <= comparison_end
        )
    ]

    regional_scores = {}

    for region in REGIONS:

        regional_scores[
            region
        ] = float(
            window[
                f"{region}"
                "_composite_similarity"
            ].mean()
        )

    target_similarity = (
        regional_scores[
            most_isolated_region
        ]
    )

    peer_similarity = (
        regional_scores[
            peer_region
        ]
    )

    context_similarity = float(
        np.mean(
            [
                regional_scores[
                    "lower_body"
                ],
                regional_scores[
                    "torso"
                ],
            ]
        )
    )

    pose_similarity = float(
        window[
            f"{most_isolated_region}"
            "_pose_similarity"
        ].mean()
    )

    position_similarity = float(
        window[
            f"{most_isolated_region}"
            "_position_similarity"
        ].mean()
    )

    speed_similarity = float(
        window[
            f"{most_isolated_region}"
            "_speed_similarity"
        ].mean()
    )

    # -------------------------------------------------
    # Save outputs.
    # -------------------------------------------------

    output_json = (
        base_dir
        / "isolated_divergence_report.json"
    )

    output_csv = (
        base_dir
        / "isolated_divergence_timeline.csv"
    )

    output_plot = (
        base_dir
        / "plots"
        / "isolated_divergence.png"
    )

    output_plot.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    timeline.to_csv(
        output_csv,
        index=False,
    )

    report = {
        "method":
            "bilateral_arm_isolation",

        "reference":
            reference_name,

        "comparison":
            comparison_name,

        "window_seconds":
            WINDOW_SECONDS,

        "most_isolated_region":
            most_isolated_region,

        "peer_region":
            peer_region,

        "isolation_gap":
            round(
                isolation_gap,
                2,
            ),

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

        "isolated_region_similarity":
            round(
                target_similarity,
                2,
            ),

        "peer_arm_similarity":
            round(
                peer_similarity,
                2,
            ),

        "context_similarity":
            round(
                context_similarity,
                2,
            ),

        "region_similarity": {
            region:
                round(
                    score,
                    2,
                )

            for region, score
            in regional_scores.items()
        },

        "isolated_region_components": {
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

            "speed_similarity":
                round(
                    speed_similarity,
                    2,
                ),
        },

        "notes": (
            "Isolation measures unilateral arm "
            "divergence relative to the opposite arm. "
            "No fixed anomaly threshold is applied yet."
        ),
    }

    with output_json.open(
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

    for region in REGIONS:

        plt.plot(
            timeline[
                "comparison_time"
            ],
            timeline[
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
        label=(
            "Strongest isolated-arm deviation"
        ),
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
        "MotionStage Isolated Arm Divergence"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_plot,
        dpi=150,
    )

    plt.close()

    # -------------------------------------------------
    # Terminal report.
    # -------------------------------------------------

    print()
    print(
        "MotionStage Isolated Divergence"
    )

    print(
        "-------------------------------"
    )

    print(
        f"Reference:  "
        f"{reference_name}"
    )

    print(
        f"Comparison: "
        f"{comparison_name}"
    )

    print(
        f"Estimated comparison FPS: "
        f"{estimated_fps:.2f}"
    )

    print(
        f"Window: "
        f"{WINDOW_SECONDS:.2f}s "
        f"({window_frames} frames)"
    )

    print()

    print(
        "Strongest isolated-arm deviation"
    )

    print(
        "--------------------------------"
    )

    print(
        f"Region: "
        f"{most_isolated_region}"
    )

    print(
        f"Isolation gap: "
        f"{isolation_gap:.2f}"
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
        f"{most_isolated_region:12} "
        f"{target_similarity:6.2f}"
    )

    print(
        f"{peer_region:12} "
        f"{peer_similarity:6.2f}"
    )

    print(
        f"{'context':12} "
        f"{context_similarity:6.2f}"
    )

    print()

    print(
        "Isolated-region components"
    )

    print(
        "--------------------------"
    )

    print(
        f"Pose:     "
        f"{pose_similarity:.2f}"
    )

    print(
        f"Position: "
        f"{position_similarity:.2f}"
    )

    print(
        f"Speed:    "
        f"{speed_similarity:.2f}"
    )

    print()

    print(
        f"Report: {output_json}"
    )

    print(
        f"Timeline: {output_csv}"
    )

    print(
        f"Plot: {output_plot}"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Detect the strongest isolated unilateral "
            "arm divergence using robust MotionStage "
            "alignment results."
        )
    )

    parser.add_argument(
        "reference",
    )

    parser.add_argument(
        "comparison",
    )

    args = parser.parse_args()

    detect_isolated_divergence(
        reference_name=args.reference,
        comparison_name=args.comparison,
    )
