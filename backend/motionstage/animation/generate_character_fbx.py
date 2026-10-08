#!/usr/bin/env python3
"""
MotionStage Stage 22B — backend character generator

Runs the proven MotionStage -> bridge rig -> Mixamo Y-Bot -> FBX workflow
through Blender in background mode.

Usage from MotionStage root:
    python backend/motionstage/animation/generate_character_fbx.py test_performance_02
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]

ANIMATION_DIR = (
    PROJECT_ROOT
    / "backend"
    / "motionstage"
    / "animation"
)

PACKAGE_EXPORTER = (
    ANIMATION_DIR
    / "export_animation_package.py"
)

BLENDER_WORKER = (
    ANIMATION_DIR
    / "blender_headless_character_build.py"
)

YBOT_ASSET = (
    PROJECT_ROOT
    / "assets"
    / "characters"
    / "ybot.fbx"
)


def find_blender() -> Path:
    candidates = [
        Path("/Applications/Blender.app/Contents/MacOS/Blender"),
        Path("/Applications/Blender 4.1.app/Contents/MacOS/Blender"),
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    hit = shutil.which("blender")
    if hit:
        return Path(hit).resolve()

    raise FileNotFoundError(
        "Blender executable not found."
    )


def require_file(path: Path, label: str) -> None:
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(
            f"{label} not found: {path}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("performance")
    args = parser.parse_args()

    performance = args.performance.strip()

    if not performance:
        raise ValueError("Performance name cannot be empty.")

    pose_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / performance
        / "pose3d_world.csv"
    )

    require_file(pose_path, "Processed pose3d_world.csv")
    require_file(PACKAGE_EXPORTER, "Stage 19A package exporter")
    require_file(BLENDER_WORKER, "Stage 22B Blender worker")
    require_file(YBOT_ASSET, "Y-Bot asset")

    blender = find_blender()

    animation_dir = (
        PROJECT_ROOT
        / "data"
        / "animation"
        / performance
    )
    animation_dir.mkdir(parents=True, exist_ok=True)

    character_dir = animation_dir / "character"
    character_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 72)
    print("MotionStage Stage 22B — Backend Character Generator")
    print("=" * 72)
    print("Performance :", performance)
    print("Blender     :", blender)
    print("Character   :", character_dir)

    # Refresh the animation package so Stage 22B always uses current pose data.
    subprocess.run(
        [
            sys.executable,
            str(PACKAGE_EXPORTER),
            performance,
            "--output-dir",
            str(animation_dir),
        ],
        cwd=str(PROJECT_ROOT),
        check=True,
    )

    package_path = animation_dir / "animation_package.json"
    require_file(package_path, "Animation package")

    command = [
        str(blender),
        "-b",
        "--factory-startup",
        "--python",
        str(BLENDER_WORKER),
        "--",
        "--project-root",
        str(PROJECT_ROOT),
        "--performance",
        performance,
        "--ybot",
        str(YBOT_ASSET),
        "--output-dir",
        str(character_dir),
    ]

    print()
    print("[22B] Launching Blender headlessly...")
    print()

    completed = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        text=True,
        capture_output=True,
    )

    if completed.stdout:
        print(completed.stdout)

    if completed.returncode != 0:
        if completed.stderr:
            print(completed.stderr, file=sys.stderr)

        raise RuntimeError(
            "Headless Blender character generation failed "
            f"with exit code {completed.returncode}."
        )

    manifest_path = (
        character_dir
        / "character_build_manifest.json"
    )
    fbx_path = character_dir / f"{performance}_ybot.fbx"
    blend_path = character_dir / f"{performance}_ybot.blend"

    require_file(manifest_path, "Character build manifest")
    require_file(fbx_path, "Generated FBX")
    require_file(blend_path, "Generated Blender file")

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    if manifest.get("status") != "completed":
        raise RuntimeError(
            "Character build manifest did not report completed status."
        )

    print()
    print("=" * 72)
    print("STAGE 22B PASS")
    print("=" * 72)
    print("FBX      :", fbx_path)
    print("BLEND    :", blend_path)
    print("Manifest :", manifest_path)

    for item in manifest.get("validation", []):
        print(
            "Frame "
            f"{item['frame']}: "
            f"mean={item['mean_error_degrees']:.3f}°, "
            f"max={item['max_error_degrees']:.3f}°"
        )

    print()
    print(
        "MotionStage generated the animated Y-Bot "
        "without opening the Blender UI."
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
