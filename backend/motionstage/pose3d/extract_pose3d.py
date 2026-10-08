import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import pandas as pd

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


LANDMARK_NAMES = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]


def extract_pose3d(video_path: str, output_csv: str, model_path: str) -> None:
    video_path = Path(video_path)
    output_csv = Path(output_csv)
    model_path = Path(model_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        raise RuntimeError("Invalid FPS reported by video.")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(model_path)
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    rows = []
    detected_frames = 0

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_index = 0

        while True:
            ok, frame = cap.read()

            if not ok:
                break

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb,
            )

            timestamp_ms = int(
                round(
                    frame_index
                    * 1000.0
                    / fps
                )
            )

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms,
            )

            if result.pose_world_landmarks:
                detected_frames += 1

                world_landmarks = (
                    result.pose_world_landmarks[0]
                )

                for joint_id, landmark in enumerate(world_landmarks):
                    rows.append(
                        {
                            "frame": frame_index,
                            "timestamp_seconds": frame_index / fps,
                            "joint_id": joint_id,
                            "joint_name": LANDMARK_NAMES[joint_id],
                            "x_world": landmark.x,
                            "y_world": landmark.y,
                            "z_world": landmark.z,
                            "visibility": landmark.visibility,
                            "presence": landmark.presence,
                        }
                    )

            frame_index += 1

            if frame_index % 30 == 0:
                print(
                    f"Processed {frame_index}/{frame_count} frames"
                )

    cap.release()

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

    print()
    print("=" * 60)
    print("MotionStage Estimated 3D Pose Extraction")
    print("=" * 60)
    print(f"Video: {video_path}")
    print(f"FPS: {fps:.2f}")
    print(f"Frames processed: {frame_index}")
    print(f"Frames with 3D pose: {detected_frames}")
    print(
        "Detection rate: "
        f"{100 * detected_frames / max(frame_index, 1):.2f}%"
    )
    print(f"3D rows written: {len(df)}")
    print(f"Output: {output_csv}")

    if not df.empty:
        print()
        print("Coordinate ranges:")
        print(
            df[
                [
                    "x_world",
                    "y_world",
                    "z_world",
                ]
            ]
            .agg(["min", "max"])
            .round(4)
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("video")
    parser.add_argument("output_csv")

    parser.add_argument(
        "--model",
        default="models/pose_landmarker_full.task",
    )

    args = parser.parse_args()

    extract_pose3d(
        video_path=args.video,
        output_csv=args.output_csv,
        model_path=args.model,
    )
