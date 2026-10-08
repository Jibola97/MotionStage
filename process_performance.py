import argparse
from pathlib import Path

from backend.motionstage.pose.track_video import process_video
from backend.motionstage.processing.smooth_pose import smooth_pose_data
from backend.motionstage.processing.normalize_pose import normalise_pose_data
from backend.motionstage.features.extract_motion_features import (
    extract_motion_features,
)


def process_performance(video_path: str) -> None:
    input_path = Path(video_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Video not found: {input_path}"
        )

    performance_name = input_path.stem

    output_dir = (
        Path("data")
        / "processed"
        / performance_name
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pose_video = (
        output_dir
        / "pose.mp4"
    )

    keypoints_csv = (
        output_dir
        / "keypoints.csv"
    )

    smoothed_csv = (
        output_dir
        / "keypoints_smoothed.csv"
    )

    normalised_csv = (
        output_dir
        / "keypoints_normalised.csv"
    )

    joint_features_csv = (
        output_dir
        / "joint_features.csv"
    )

    frame_features_csv = (
        output_dir
        / "frame_features.csv"
    )

    print()
    print("=" * 60)
    print("MotionStage Performance Pipeline")
    print("=" * 60)
    print(f"Performance: {performance_name}")
    print(f"Input: {input_path}")
    print(f"Output directory: {output_dir}")
    print()

    # -------------------------------------------------
    # Stage 1 — Pose tracking
    # -------------------------------------------------
    print()
    print("[1/4] POSE TRACKING")
    print("=" * 60)

    process_video(
        input_path=str(input_path),
        output_video_path=str(pose_video),
        output_csv_path=str(keypoints_csv),
    )

    # -------------------------------------------------
    # Stage 2 — Smoothing
    # -------------------------------------------------
    print()
    print("[2/4] POSE SMOOTHING")
    print("=" * 60)

    smooth_pose_data(
        input_csv=str(keypoints_csv),
        output_csv=str(smoothed_csv),
    )

    # -------------------------------------------------
    # Stage 3 — Skeleton normalisation
    # -------------------------------------------------
    print()
    print("[3/4] SKELETON NORMALISATION")
    print("=" * 60)

    normalise_pose_data(
        input_csv=str(smoothed_csv),
        output_csv=str(normalised_csv),
    )

    # -------------------------------------------------
    # Stage 4 — Motion features
    # -------------------------------------------------
    print()
    print("[4/4] MOTION FEATURE EXTRACTION")
    print("=" * 60)

    extract_motion_features(
        input_csv=str(normalised_csv),
        joint_output_csv=str(joint_features_csv),
        frame_output_csv=str(frame_features_csv),
    )

    # -------------------------------------------------
    # Final summary
    # -------------------------------------------------
    print()
    print("=" * 60)
    print("MotionStage Processing Complete")
    print("=" * 60)

    print()
    print(f"Performance: {performance_name}")

    print()
    print("Generated outputs:")
    print(f"  Pose video:            {pose_video}")
    print(f"  Raw keypoints:         {keypoints_csv}")
    print(f"  Smoothed keypoints:    {smoothed_csv}")
    print(f"  Normalised keypoints:  {normalised_csv}")
    print(f"  Joint features:        {joint_features_csv}")
    print(f"  Frame features:        {frame_features_csv}")

    print()
    print("Ready for performance comparison.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run the full MotionStage pipeline "
            "on a single performance video."
        )
    )

    parser.add_argument(
        "video",
        help="Path to the performance video.",
    )

    args = parser.parse_args()

    process_performance(
        args.video
    )
