from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


FEATURES = {
    "left_elbow_angle_deg": "Left Elbow Angle",
    "right_elbow_angle_deg": "Right Elbow Angle",
    "left_shoulder_angle_deg": "Left Shoulder Angle",
    "right_shoulder_angle_deg": "Right Shoulder Angle",
    "torso_lean_deg": "Torso Lean",
}


def plot_aligned_features(
    reference_csv: str,
    comparison_csv: str,
    alignment_csv: str,
    output_dir: str,
) -> None:

    reference = pd.read_csv(reference_csv)
    comparison = pd.read_csv(comparison_csv)
    alignment = pd.read_csv(alignment_csv)

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference_indexed = (
        reference.set_index("frame")
    )

    comparison_indexed = (
        comparison.set_index("frame")
    )

    for feature, display_name in FEATURES.items():

        reference_values = []
        comparison_values = []

        for _, row in alignment.iterrows():

            reference_frame = int(
                row["reference_frame"]
            )

            comparison_frame = int(
                row["comparison_frame"]
            )

            reference_values.append(
                reference_indexed.loc[
                    reference_frame,
                    feature,
                ]
            )

            comparison_values.append(
                comparison_indexed.loc[
                    comparison_frame,
                    feature,
                ]
            )

        steps = range(
            len(reference_values)
        )

        plt.figure(
            figsize=(11, 5)
        )

        plt.plot(
            steps,
            reference_values,
            label="Performance 01",
        )

        plt.plot(
            steps,
            comparison_values,
            label="Performance 02",
        )

        plt.xlabel(
            "DTW alignment step"
        )

        if feature == "torso_lean_deg":
            plt.ylabel(
                "Torso lean (degrees)"
            )
        else:
            plt.ylabel(
                "Joint angle (degrees)"
            )

        plt.title(
            f"Aligned {display_name}"
        )

        plt.legend()
        plt.tight_layout()

        output_path = (
            output_dir
            / f"{feature}_aligned.png"
        )

        plt.savefig(
            output_path,
            dpi=150,
        )

        plt.close()

        print(
            f"Saved: {output_path}"
        )


if __name__ == "__main__":
    plot_aligned_features(
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
            "plots/all_features"
        ),
    )
