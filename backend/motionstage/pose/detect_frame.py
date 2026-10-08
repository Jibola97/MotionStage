from pathlib import Path

import cv2
from rtmlib import Body, draw_skeleton


def extract_frame(video_path: str, timestamp_seconds: float):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_number = int(timestamp_seconds * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    success, frame = cap.read()
    cap.release()

    if not success:
        raise ValueError(
            f"Could not read frame at {timestamp_seconds:.2f} seconds."
        )

    return frame, frame_number


def detect_pose(frame):
    model = Body(
        mode="balanced",
        backend="onnxruntime",
        device="cpu",
    )

    keypoints, scores = model(frame)

    return keypoints, scores


def main():
    video_path = "data/raw/test_performance_01.mov"
    output_path = Path("data/processed/first_pose.jpg")

    # Use a middle frame rather than the first frame.
    timestamp_seconds = 2.5

    frame, frame_number = extract_frame(
        video_path,
        timestamp_seconds,
    )

    print(f"Extracted frame: {frame_number}")

    keypoints, scores = detect_pose(frame)

    print(f"Detected people: {len(keypoints)}")

    if len(keypoints) == 0:
        raise RuntimeError("No person detected in the frame.")

    print(f"Keypoints detected: {keypoints.shape}")
    print(f"Confidence scores: {scores.shape}")

    output = draw_skeleton(
        frame.copy(),
        keypoints,
        scores,
        kpt_thr=0.3,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_path), output)

    print(f"Saved pose image to: {output_path}")


if __name__ == "__main__":
    main()
