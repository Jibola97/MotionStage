from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_REFERENCE_JOINTS = {
    "left_hip",
    "right_hip",
    "left_shoulder",
    "right_shoulder",
}


def normalise_pose_data(
    input_csv: str,
    output_csv: str,
) -> None:
    input_path = Path(input_csv)
    output_path = Path(output_csv)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {input_path}"
        )

    df = pd.read_csv(input_path)

    # Prefer smoothed coordinates if available.
    x_column = (
        "x_smooth"
        if "x_smooth" in df.columns
        else "x"
    )

    y_column = (
        "y_smooth"
        if "y_smooth" in df.columns
        else "y"
    )

    required_columns = {
        "frame",
        "joint_name",
        x_column,
        y_column,
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    frame_metadata = {}

    torso_lengths = []

    # -------------------------------------------------
    # Pass 1:
    # Calculate hip centre, shoulder centre and
    # torso length for every frame.
    # -------------------------------------------------
    for frame_number, frame_df in df.groupby("frame"):
        joints = frame_df.set_index("joint_name")

        missing_joints = (
            REQUIRED_REFERENCE_JOINTS
            - set(joints.index)
        )

        if missing_joints:
            raise ValueError(
                f"Frame {frame_number} is missing "
                f"reference joints: {sorted(missing_joints)}"
            )

        left_hip = joints.loc["left_hip"]
        right_hip = joints.loc["right_hip"]

        left_shoulder = joints.loc["left_shoulder"]
        right_shoulder = joints.loc["right_shoulder"]

        hip_center_x = (
            left_hip[x_column]
            + right_hip[x_column]
        ) / 2.0

        hip_center_y = (
            left_hip[y_column]
            + right_hip[y_column]
        ) / 2.0

        shoulder_center_x = (
            left_shoulder[x_column]
            + right_shoulder[x_column]
        ) / 2.0

        shoulder_center_y = (
            left_shoulder[y_column]
            + right_shoulder[y_column]
        ) / 2.0

        torso_length = np.hypot(
            shoulder_center_x - hip_center_x,
            shoulder_center_y - hip_center_y,
        )

        frame_metadata[frame_number] = {
            "hip_center_x": hip_center_x,
            "hip_center_y": hip_center_y,
            "shoulder_center_x": shoulder_center_x,
            "shoulder_center_y": shoulder_center_y,
            "torso_length": torso_length,
        }

        if torso_length > 0:
            torso_lengths.append(torso_length)

    if not torso_lengths:
        raise ValueError(
            "Could not calculate a valid torso scale."
        )

    # One scale value for the entire performance.
    body_scale = float(np.median(torso_lengths))

    # -------------------------------------------------
    # Pass 2:
    # Add body-relative coordinates.
    # -------------------------------------------------
    df["hip_center_x"] = df["frame"].map(
        lambda frame: frame_metadata[frame]["hip_center_x"]
    )

    df["hip_center_y"] = df["frame"].map(
        lambda frame: frame_metadata[frame]["hip_center_y"]
    )

    df["frame_torso_length"] = df["frame"].map(
        lambda frame: frame_metadata[frame]["torso_length"]
    )

    df["body_scale"] = body_scale

    # X:
    # positive = right of pelvis
    # negative = left of pelvis
    df["x_norm"] = (
        df[x_column] - df["hip_center_x"]
    ) / body_scale

    # Image Y coordinates increase downward.
    # Negate them so positive normalised Y means upward.
    df["y_norm"] = -(
        df[y_column] - df["hip_center_y"]
    ) / body_scale

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output_path,
        index=False,
    )

    # -------------------------------------------------
    # Diagnostics
    # -------------------------------------------------
    hip_rows = df[
        df["joint_name"].isin(
            ["left_hip", "right_hip"]
        )
    ]

    shoulder_rows = df[
        df["joint_name"].isin(
            ["left_shoulder", "right_shoulder"]
        )
    ]

    hip_midpoints = (
        hip_rows
        .groupby("frame")[["x_norm", "y_norm"]]
        .mean()
    )

    shoulder_midpoints = (
        shoulder_rows
        .groupby("frame")[["x_norm", "y_norm"]]
        .mean()
    )

    shoulder_distance = np.sqrt(
        shoulder_midpoints["x_norm"] ** 2
        + shoulder_midpoints["y_norm"] ** 2
    )

    print("MotionStage Skeleton Normalisation")
    print("----------------------------------")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Coordinate source: {x_column}, {y_column}")
    print(f"Median torso scale: {body_scale:.3f} pixels")
    print(f"Rows processed: {len(df)}")

    print()
    print("Sanity checks")
    print("-------------")

    print(
        "Mean hip-centre X after normalisation: "
        f"{hip_midpoints['x_norm'].mean():.6f}"
    )

    print(
        "Mean hip-centre Y after normalisation: "
        f"{hip_midpoints['y_norm'].mean():.6f}"
    )

    print(
        "Median normalised shoulder-centre distance: "
        f"{shoulder_distance.median():.4f}"
    )


if __name__ == "__main__":
    normalise_pose_data(
        input_csv=(
            "data/processed/"
            "test_performance_01_keypoints_smoothed.csv"
        ),
        output_csv=(
            "data/processed/"
            "test_performance_01_keypoints_normalised.csv"
        ),
    )
