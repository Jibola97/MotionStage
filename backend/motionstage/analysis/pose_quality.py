from pathlib import Path

import pandas as pd


def analyse_pose_quality(csv_path: str) -> None:
    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)

    # Supports both the original CSV and the renamed field.
    score_column = (
        "pose_score"
        if "pose_score" in df.columns
        else "confidence"
    )

    total_frames = df["frame"].nunique()
    total_rows = len(df)
    total_joints = df["joint_name"].nunique()

    print("\nMotionStage Pose Quality Report")
    print("-------------------------------")
    print(f"Frames: {total_frames}")
    print(f"Joint observations: {total_rows}")
    print(f"Joint types: {total_joints}")

    print("\nOverall pose-score statistics")
    print("-----------------------------")
    print(
        df[score_column]
        .describe()
        .round(4)
        .to_string()
    )

    joint_summary = (
        df.groupby("joint_name")[score_column]
        .agg(["min", "mean", "median", "max"])
        .sort_values("mean")
        .round(4)
    )

    print("\nPose scores by joint")
    print("--------------------")
    print(joint_summary.to_string())

    frame_min_scores = (
        df.groupby("frame")[score_column]
        .min()
        .sort_values()
    )

    print("\n10 frames containing the weakest joint score")
    print("--------------------------------------------")
    print(frame_min_scores.head(10).round(4).to_string())


if __name__ == "__main__":
    analyse_pose_quality(
        "data/processed/test_performance_01_keypoints.csv"
    )
