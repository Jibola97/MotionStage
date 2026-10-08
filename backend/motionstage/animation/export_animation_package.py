#!/usr/bin/env python3
"""MotionStage Stage 19A: export pose3d_world.csv to an animation-ready package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3] if "backend" in Path(__file__).parts else Path.cwd()

REQUIRED = {
    "frame", "timestamp_seconds", "joint_id", "joint_name",
    "x_world", "y_world", "z_world",
}

SOURCE_JOINTS = {
    "head": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_foot": 31,
    "right_foot": 32,
}

SYNTHETIC_JOINTS = ("pelvis", "spine", "chest")

BONES = [
    {"name": "spine_lower", "head": "pelvis", "tail": "spine", "parent": None},
    {"name": "spine_upper", "head": "spine", "tail": "chest", "parent": "spine_lower"},
    {"name": "head", "head": "chest", "tail": "head", "parent": "spine_upper"},
    {"name": "left_upper_arm", "head": "left_shoulder", "tail": "left_elbow", "parent": "spine_upper"},
    {"name": "left_forearm", "head": "left_elbow", "tail": "left_wrist", "parent": "left_upper_arm"},
    {"name": "right_upper_arm", "head": "right_shoulder", "tail": "right_elbow", "parent": "spine_upper"},
    {"name": "right_forearm", "head": "right_elbow", "tail": "right_wrist", "parent": "right_upper_arm"},
    {"name": "left_thigh", "head": "left_hip", "tail": "left_knee", "parent": "spine_lower"},
    {"name": "left_shin", "head": "left_knee", "tail": "left_ankle", "parent": "left_thigh"},
    {"name": "left_foot", "head": "left_ankle", "tail": "left_foot", "parent": "left_shin"},
    {"name": "right_thigh", "head": "right_hip", "tail": "right_knee", "parent": "spine_lower"},
    {"name": "right_shin", "head": "right_knee", "tail": "right_ankle", "parent": "right_thigh"},
    {"name": "right_foot", "head": "right_ankle", "tail": "right_foot", "parent": "right_shin"},
]


def resolve_input(value: str) -> tuple[str, Path]:
    candidate = Path(value).expanduser()
    if candidate.exists():
        path = candidate.resolve()
        name = path.parent.name if path.name == "pose3d_world.csv" else path.stem
        return name, path

    name = value
    path = PROJECT_ROOT / "data" / "processed" / name / "pose3d_world.csv"
    if not path.exists():
        raise FileNotFoundError(f"pose3d_world.csv not found: {path}")
    return name, path


def estimate_fps(series: pd.Series) -> float:
    t = np.sort(pd.to_numeric(series, errors="coerce").dropna().unique())
    if len(t) < 2:
        return 30.0
    dt = np.diff(t)
    dt = dt[dt > 1e-9]
    return float(1.0 / np.median(dt)) if len(dt) else 30.0


def to_blender(row) -> np.ndarray:
    # Same orientation already used by MotionStage's 3D viewer.
    return np.array([
        float(row.x_world),
        float(row.z_world),
        float(-row.y_world),
    ])


def build_frame(frame_df: pd.DataFrame) -> tuple[dict[str, np.ndarray], np.ndarray]:
    points = {int(r.joint_id): to_blender(r) for r in frame_df.itertuples()}
    missing = [name for name, idx in SOURCE_JOINTS.items() if idx not in points]
    if missing:
        raise ValueError("missing required joints: " + ", ".join(missing))

    joints = {name: points[idx].copy() for name, idx in SOURCE_JOINTS.items()}
    pelvis = (joints["left_hip"] + joints["right_hip"]) / 2.0
    chest = (joints["left_shoulder"] + joints["right_shoulder"]) / 2.0
    spine = (pelvis + chest) / 2.0

    joints["pelvis"] = pelvis
    joints["spine"] = spine
    joints["chest"] = chest

    return joints, pelvis


def rvec(v: np.ndarray) -> list[float]:
    return [round(float(x), 7) for x in v]


def main() -> None:
    parser = argparse.ArgumentParser(description="MotionStage Stage 19A animation exporter")
    parser.add_argument("input", help="Performance name or path to pose3d_world.csv")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    performance, input_path = resolve_input(args.input)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else PROJECT_ROOT / "data" / "animation" / performance
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.sort_values(["frame", "joint_id"]).reset_index(drop=True)
    frames = sorted(df["frame"].dropna().astype(int).unique().tolist())
    fps = estimate_fps(df["timestamp_seconds"])

    package_frames = []
    csv_rows = []
    skipped = []
    shoulder_widths = []
    hip_widths = []

    for frame in frames:
        current = df[df["frame"] == frame]
        timestamp = float(current["timestamp_seconds"].iloc[0])
        try:
            joints, pelvis = build_frame(current)
        except ValueError as exc:
            skipped.append({"frame": frame, "timestamp_seconds": timestamp, "reason": str(exc)})
            continue

        root_relative = {name: pos - pelvis for name, pos in joints.items()}
        shoulder_widths.append(float(np.linalg.norm(joints["left_shoulder"] - joints["right_shoulder"])))
        hip_widths.append(float(np.linalg.norm(joints["left_hip"] - joints["right_hip"])))

        package_frames.append({
            "frame": frame,
            "timestamp_seconds": round(timestamp, 7),
            "root_source_position": rvec(pelvis),
            "joints": {name: rvec(pos) for name, pos in root_relative.items()},
        })

        for name, pos in root_relative.items():
            csv_rows.append({
                "frame": frame,
                "timestamp_seconds": round(timestamp, 7),
                "joint_name": name,
                "source_joint_id": SOURCE_JOINTS.get(name, ""),
                "synthetic": name in SYNTHETIC_JOINTS,
                "x": float(pos[0]),
                "y": float(pos[1]),
                "z": float(pos[2]),
            })

    if not package_frames:
        raise RuntimeError("No complete animation frames could be exported.")

    timestamps = [f["timestamp_seconds"] for f in package_frames]
    duration = max(timestamps) - min(timestamps)

    joint_defs = [
        {"name": name, "synthetic": True, "source_joint_id": None}
        for name in SYNTHETIC_JOINTS
    ] + [
        {"name": name, "synthetic": False, "source_joint_id": idx}
        for name, idx in SOURCE_JOINTS.items()
    ]

    package = {
        "schema": "motionstage.animation.v1",
        "performance_name": performance,
        "source": str(input_path),
        "coordinate_system": {
            "x": "x_world",
            "y": "z_world (depth)",
            "z": "-y_world (up)",
            "root_policy": "in_place_pelvis_relative",
            "units": "estimated metres",
        },
        "timing": {
            "fps_estimate": round(fps, 6),
            "frame_count": len(package_frames),
            "duration_seconds": round(duration, 6),
        },
        "scale_reference": {
            "median_shoulder_width": float(np.median(shoulder_widths)),
            "median_hip_width": float(np.median(hip_widths)),
        },
        "joints": joint_defs,
        "bones": BONES,
        "frames": package_frames,
        "notes": [
            "Stage 19A exports pelvis-relative in-place motion.",
            "Bone rotations are deferred to the Blender import/rig stage.",
            "This bridge skeleton is not yet an Unreal mannequin skeleton.",
        ],
    }

    package_path = output_dir / "animation_package.json"
    joints_path = output_dir / "animation_joints.csv"
    manifest_path = output_dir / "animation_manifest.json"

    package_path.write_text(json.dumps(package, indent=2), encoding="utf-8")
    pd.DataFrame(csv_rows).to_csv(joints_path, index=False)

    manifest = {
        "stage": "19A",
        "performance_name": performance,
        "input": str(input_path),
        "fps_estimate": round(fps, 6),
        "source_frames": len(frames),
        "exported_frames": len(package_frames),
        "skipped_frames": skipped,
        "animation_joint_count": len(joint_defs),
        "bone_count": len(BONES),
        "duration_seconds": round(duration, 6),
        "outputs": {
            "animation_package": str(package_path),
            "animation_joints_csv": str(joints_path),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("=" * 72)
    print("MotionStage Stage 19A - Animation Export")
    print("=" * 72)
    print(f"Performance      : {performance}")
    print(f"Estimated FPS    : {fps:.3f}")
    print(f"Source frames    : {len(frames)}")
    print(f"Exported frames  : {len(package_frames)}")
    print(f"Skipped frames   : {len(skipped)}")
    print(f"Animation joints : {len(joint_defs)}")
    print(f"Skeleton bones   : {len(BONES)}")
    print(f"Duration         : {duration:.3f}s")
    print(f"Package          : {package_path}")
    print(f"Joint CSV        : {joints_path}")
    print(f"Manifest         : {manifest_path}")
    print("Coordinate map   : X=x_world, Y=z_world, Z=-y_world")
    print("Root policy      : pelvis-relative / in-place")
    print("=" * 72)


if __name__ == "__main__":
    main()
