import csv
from pathlib import Path

import cv2
from rtmlib import Body, draw_skeleton


JOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def process_video(
    input_path: str,
    output_video_path: str,
    output_csv_path: str,
):
    input_path = Path(input_path)
    output_video_path = Path(output_video_path)
    output_csv_path = Path(output_csv_path)

    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("MotionStage Full-Video Pose Tracking")
    print("------------------------------------")
    print(f"Input: {input_path.name}")
    print(f"FPS: {fps:.2f}")
    print(f"Frames: {frame_count}")
    print(f"Resolution: {width}x{height}")
    print()

    # Initialise models ONCE.
    # Do not recreate them for every frame.
    model = Body(
        mode="balanced",
        backend="onnxruntime",
        device="cpu",
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_video_path),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError("Could not create output video.")

    successful_pose_frames = 0
    missing_pose_frames = 0

    with output_csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        csv_writer = csv.writer(csv_file)

        csv_writer.writerow(
            [
                "frame",
                "timestamp_seconds",
                "joint_id",
                "joint_name",
                "x",
                "y",
                "pose_score",
            ]
        )

        frame_index = 0

        while True:
            success, frame = cap.read()

            if not success:
                break

            timestamp = frame_index / fps

            keypoints, scores = model(frame)

            if len(keypoints) > 0:
                successful_pose_frames += 1

                # Controlled v1 footage contains one performer,
                # so we use the first detected person.
                person_keypoints = keypoints[0]
                person_scores = scores[0]

                for joint_id, joint_name in enumerate(JOINT_NAMES):
                    x, y = person_keypoints[joint_id]
                    pose_score = person_scores[joint_id]

                    csv_writer.writerow(
                        [
                            frame_index,
                            round(timestamp, 4),
                            joint_id,
                            joint_name,
                            round(float(x), 3),
                            round(float(y), 3),
                            round(float(pose_score), 4),
                        ]
                    )

                annotated = draw_skeleton(
                    frame.copy(),
                    keypoints,
                    scores,
                    kpt_thr=0.3,
                )

            else:
                missing_pose_frames += 1
                annotated = frame

            writer.write(annotated)

            frame_index += 1

            if frame_index % 10 == 0 or frame_index == frame_count:
                print(
                    f"Processed {frame_index}/{frame_count} frames"
                )

    cap.release()
    writer.release()

    detection_rate = (
        successful_pose_frames / frame_index * 100
        if frame_index > 0
        else 0
    )

    print()
    print("Processing complete")
    print("-------------------")
    print(f"Frames processed: {frame_index}")
    print(f"Pose frames: {successful_pose_frames}")
    print(f"Missing pose frames: {missing_pose_frames}")
    print(f"Detection rate: {detection_rate:.2f}%")
    print(f"Video: {output_video_path}")
    print(f"CSV: {output_csv_path}")


if __name__ == "__main__":
    process_video(
        input_path="data/raw/test_performance_01.mov",
        output_video_path=(
            "data/processed/test_performance_01_pose.mp4"
        ),
        output_csv_path=(
            "data/processed/test_performance_01_keypoints.csv"
        ),
    )
