from pathlib import Path

import cv2


def inspect_video(video_path: str) -> dict:
    path = Path(video_path)

    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    capture = cv2.VideoCapture(str(path))

    if not capture.isOpened():
        raise ValueError(f"Could not open video: {path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    duration = frame_count / fps if fps > 0 else 0

    capture.release()

    return {
        "filename": path.name,
        "fps": round(fps, 2),
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_seconds": round(duration, 2),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect a MotionStage video.")
    parser.add_argument("video", help="Path to the video file")
    args = parser.parse_args()

    metadata = inspect_video(args.video)

    print("\nMotionStage Video Inspection")
    print("----------------------------")

    for key, value in metadata.items():
        print(f"{key}: {value}")
