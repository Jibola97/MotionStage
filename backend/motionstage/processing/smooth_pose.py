from pathlib import Path

import pandas as pd
from scipy.signal import savgol_filter


WINDOW_LENGTH = 5
POLYORDER = 2


def smooth_pose_data(
    input_csv: str,
    output_csv: str,
) -> None:
    input_path = Path(input_csv)
    output_path = Path(output_csv)

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)

    required_columns = {
        "frame",
        "timestamp_seconds",
        "joint_name",
        "x",
        "y",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    output_frames = []

    for joint_name, joint_df in df.groupby("joint_name"):
        joint_df = (
            joint_df
            .sort_values("frame")
            .copy()
        )

        if len(joint_df) >= WINDOW_LENGTH:
            joint_df["x_smooth"] = savgol_filter(
                joint_df["x"].to_numpy(),
                window_length=WINDOW_LENGTH,
                polyorder=POLYORDER,
                mode="interp",
            )

            joint_df["y_smooth"] = savgol_filter(
                joint_df["y"].to_numpy(),
                window_length=WINDOW_LENGTH,
                polyorder=POLYORDER,
                mode="interp",
            )
        else:
            joint_df["x_smooth"] = joint_df["x"]
            joint_df["y_smooth"] = joint_df["y"]

        output_frames.append(joint_df)

    smoothed_df = (
        pd.concat(output_frames)
        .sort_values(["frame", "joint_id"])
        .reset_index(drop=True)
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    smoothed_df.to_csv(
        output_path,
        index=False,
    )

    print("MotionStage Pose Smoothing")
    print("--------------------------")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Window length: {WINDOW_LENGTH}")
    print(f"Polynomial order: {POLYORDER}")
    print(f"Rows processed: {len(smoothed_df)}")


if __name__ == "__main__":
    smooth_pose_data(
        "data/processed/test_performance_01_keypoints.csv",
        "data/processed/test_performance_01_keypoints_smoothed.csv",
    )
