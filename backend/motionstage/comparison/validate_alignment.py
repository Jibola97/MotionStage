from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FEATURES = [
    "left_elbow_angle_deg",
    "right_elbow_angle_deg",
    "left_shoulder_angle_deg",
    "right_shoulder_angle_deg",
    "torso_lean_deg",
]


def validate_alignment(
    reference_csv: str,
    comparison_csv: str,
    alignment_csv: str,
    output_dir: str,
) -> None:

    reference = pd.read_csv(reference_csv)
    comparison = pd.read_csv(comparison_csv)
    alignment = pd.read_csv(alignment_csv)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_duration = reference["timestamp_seconds"].iloc[-1]
    comparison_duration = comparison["timestamp_seconds"].iloc[-1]

    # -------------------------------------------------
    # 1. Naive comparison
    #
    # Match performances by relative progress through
    # the clip without DTW.
    # -------------------------------------------------

    reference_progress = (
        reference["timestamp_seconds"]
        / reference_duration
    )

    target_comparison_times = (
        reference_progress
        * comparison_duration
    )

    naive_results = {}

    for feature in FEATURES:

        comparison_interpolated = np.interp(
            target_comparison_times,
            comparison["timestamp_seconds"],
            comparison[feature],
        )

        naive_mae = np.mean(
            np.abs(
                reference[feature].to_numpy()
                - comparison_interpolated
            )
        )

        naive_results[feature] = naive_mae

    # -------------------------------------------------
    # 2. DTW-aligned comparison
    # -------------------------------------------------

    reference_indexed = reference.set_index("frame")
    comparison_indexed = comparison.set_index("frame")

    dtw_results = {}

    for feature in FEATURES:

        differences = []

        for _, row in alignment.iterrows():

            ref_frame = int(row["reference_frame"])
            comp_frame = int(row["comparison_frame"])

            ref_value = reference_indexed.loc[
                ref_frame,
                feature,
            ]

            comp_value = comparison_indexed.loc[
                comp_frame,
                feature,
            ]

            differences.append(
                abs(ref_value - comp_value)
            )

        dtw_results[feature] = np.mean(differences)

    # -------------------------------------------------
    # Print comparison
    # -------------------------------------------------

    print()
    print("MotionStage DTW Validation")
    print("--------------------------")

    print(
        f"{'Feature':32}"
        f"{'Naive MAE':>12}"
        f"{'DTW MAE':>12}"
        f"{'Change':>12}"
    )

    print("-" * 68)

    for feature in FEATURES:

        naive = naive_results[feature]
        dtw = dtw_results[feature]

        improvement = (
            (naive - dtw) / naive * 100
            if naive > 0
            else 0
        )

        print(
            f"{feature:32}"
            f"{naive:12.2f}"
            f"{dtw:12.2f}"
            f"{improvement:11.1f}%"
        )

    # -------------------------------------------------
    # 3. DTW time-mapping plot
    # -------------------------------------------------

    plt.figure(figsize=(9, 7))

    plt.plot(
        alignment["reference_time"],
        alignment["comparison_time"],
        label="DTW alignment",
        linewidth=2,
    )

    # What mapping would look like if the only
    # difference were uniform playback speed.
    reference_line = np.linspace(
        0,
        reference_duration,
        200,
    )

    comparison_line = (
        reference_line
        * (
            comparison_duration
            / reference_duration
        )
    )

    plt.plot(
        reference_line,
        comparison_line,
        linestyle="--",
        label="Uniform time scaling",
    )

    plt.xlabel(
        "Performance 01 time (seconds)"
    )

    plt.ylabel(
        "Performance 02 time (seconds)"
    )

    plt.title(
        "MotionStage DTW Temporal Alignment"
    )

    plt.legend()
    plt.tight_layout()

    path_plot = (
        output_dir
        / "dtw_time_mapping.png"
    )

    plt.savefig(
        path_plot,
        dpi=150,
    )

    plt.close()

    # -------------------------------------------------
    # 4. Plot aligned shoulder-angle trajectories
    # -------------------------------------------------

    aligned_reference_left = []
    aligned_comparison_left = []

    aligned_reference_right = []
    aligned_comparison_right = []

    for _, row in alignment.iterrows():

        ref_frame = int(row["reference_frame"])
        comp_frame = int(row["comparison_frame"])

        aligned_reference_left.append(
            reference_indexed.loc[
                ref_frame,
                "left_shoulder_angle_deg",
            ]
        )

        aligned_comparison_left.append(
            comparison_indexed.loc[
                comp_frame,
                "left_shoulder_angle_deg",
            ]
        )

        aligned_reference_right.append(
            reference_indexed.loc[
                ref_frame,
                "right_shoulder_angle_deg",
            ]
        )

        aligned_comparison_right.append(
            comparison_indexed.loc[
                comp_frame,
                "right_shoulder_angle_deg",
            ]
        )

    alignment_steps = np.arange(
        len(alignment)
    )

    plt.figure(figsize=(11, 5))

    plt.plot(
        alignment_steps,
        aligned_reference_left,
        label="Performance 01",
    )

    plt.plot(
        alignment_steps,
        aligned_comparison_left,
        label="Performance 02",
    )

    plt.xlabel("DTW alignment step")
    plt.ylabel("Shoulder angle (degrees)")
    plt.title(
        "Aligned Left-Shoulder Motion"
    )
    plt.legend()
    plt.tight_layout()

    left_plot = (
        output_dir
        / "left_shoulder_aligned.png"
    )

    plt.savefig(
        left_plot,
        dpi=150,
    )

    plt.close()

    plt.figure(figsize=(11, 5))

    plt.plot(
        alignment_steps,
        aligned_reference_right,
        label="Performance 01",
    )

    plt.plot(
        alignment_steps,
        aligned_comparison_right,
        label="Performance 02",
    )

    plt.xlabel("DTW alignment step")
    plt.ylabel("Shoulder angle (degrees)")
    plt.title(
        "Aligned Right-Shoulder Motion"
    )
    plt.legend()
    plt.tight_layout()

    right_plot = (
        output_dir
        / "right_shoulder_aligned.png"
    )

    plt.savefig(
        right_plot,
        dpi=150,
    )

    plt.close()

    print()
    print("Plots saved:")
    print(f"  {path_plot}")
    print(f"  {left_plot}")
    print(f"  {right_plot}")


if __name__ == "__main__":
    validate_alignment(
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
        alignment_csv=(
            "data/comparisons/"
            "test_performance_01_vs_02/"
            "dtw_alignment.csv"
        ),
        output_dir=(
            "data/comparisons/"
            "test_performance_01_vs_02/"
            "plots"
        ),
    )
