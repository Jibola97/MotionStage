#!/usr/bin/env python3
"""
MotionStage Stage 22A — Character pipeline preflight

Purpose
-------
Verify that the backend has everything required to automate the proven
Blender retarget/export pipeline before we wire it into FastAPI.

This stage does NOT modify Blender files and does NOT export a character yet.
It:
  1. validates the processed 3D pose input,
  2. runs Stage 19A animation-package generation,
  3. verifies the Blender executable,
  4. verifies the project-owned Y-Bot FBX asset,
  5. writes a preflight manifest.

Usage
-----
From the MotionStage project root:

    python backend/motionstage/animation/character_pipeline_preflight.py test_performance_02

Expected project asset:
    assets/characters/ybot.fbx
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def project_root() -> Path:
    here = Path(__file__).resolve()
    # backend/motionstage/animation/<this file>
    return here.parents[3]


PROJECT_ROOT = project_root()

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

YBOT_ASSET = (
    PROJECT_ROOT
    / "assets"
    / "characters"
    / "ybot.fbx"
)

BLENDER_CANDIDATES = [
    Path("/Applications/Blender.app/Contents/MacOS/Blender"),
    Path("/Applications/Blender 4.1.app/Contents/MacOS/Blender"),
]


def find_blender() -> Path:
    """
    Resolve Blender in a predictable way on macOS, with PATH fallback.
    """
    for candidate in BLENDER_CANDIDATES:
        if candidate.exists():
            return candidate

    path_hit = shutil.which("blender")
    if path_hit:
        return Path(path_hit).resolve()

    raise FileNotFoundError(
        "Blender executable not found. Expected "
        "/Applications/Blender.app/Contents/MacOS/Blender "
        "or a 'blender' executable on PATH."
    )


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found:\n  {path}"
        )

    if not path.is_file():
        raise RuntimeError(
            f"{label} exists but is not a file:\n  {path}"
        )


def run_animation_package_export(
    performance_name: str,
) -> tuple[Path, Path]:
    """
    Generate/refresh Stage 19A files for one processed performance.
    """
    pose3d_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / performance_name
        / "pose3d_world.csv"
    )

    require_file(
        pose3d_path,
        "Processed 3D pose input",
    )
    require_file(
        PACKAGE_EXPORTER,
        "Animation package exporter",
    )

    output_dir = (
        PROJECT_ROOT
        / "data"
        / "animation"
        / performance_name
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 72)
    print("MotionStage Stage 22A — Character Pipeline Preflight")
    print("=" * 72)
    print(f"Performance : {performance_name}")
    print(f"Pose input  : {pose3d_path}")
    print()

    subprocess.run(
        [
            sys.executable,
            str(PACKAGE_EXPORTER),
            performance_name,
            "--output-dir",
            str(output_dir),
        ],
        cwd=str(PROJECT_ROOT),
        check=True,
    )

    package_path = (
        output_dir
        / "animation_package.json"
    )
    manifest_path = (
        output_dir
        / "animation_manifest.json"
    )

    require_file(
        package_path,
        "Animation package",
    )
    require_file(
        manifest_path,
        "Animation manifest",
    )

    return package_path, manifest_path


def validate_animation_package(
    package_path: Path,
    performance_name: str,
) -> dict:
    data = json.loads(
        package_path.read_text(
            encoding="utf-8",
        )
    )

    required = {
        "schema",
        "performance_name",
        "timing",
        "bones",
        "frames",
    }

    missing = required - set(data)
    if missing:
        raise RuntimeError(
            "Animation package is missing required fields: "
            + ", ".join(sorted(missing))
        )

    if data["performance_name"] != performance_name:
        raise RuntimeError(
            "Animation package performance mismatch: "
            f"expected {performance_name!r}, "
            f"got {data['performance_name']!r}"
        )

    frames = data.get("frames") or []
    if not frames:
        raise RuntimeError(
            "Animation package contains no animation frames."
        )

    return data


def write_preflight_manifest(
    performance_name: str,
    blender_path: Path,
    package_path: Path,
    package: dict,
) -> Path:
    out_dir = (
        PROJECT_ROOT
        / "data"
        / "animation"
        / performance_name
    )

    out_path = (
        out_dir
        / "character_pipeline_preflight.json"
    )

    manifest = {
        "stage": "22A",
        "status": "ready",
        "performance_name": performance_name,
        "project_root": str(PROJECT_ROOT),
        "blender_executable": str(blender_path),
        "ybot_asset": str(YBOT_ASSET),
        "animation_package": str(package_path),
        "animation_schema": package.get("schema"),
        "frame_count": (
            package
            .get("timing", {})
            .get("frame_count")
        ),
        "fps_estimate": (
            package
            .get("timing", {})
            .get("fps_estimate")
        ),
        "root_policy": (
            package
            .get("coordinate_system", {})
            .get("root_policy")
        ),
        "next_stage": (
            "22B_headless_blender_character_build"
        ),
    }

    out_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "MotionStage Stage 22A "
            "character-pipeline preflight"
        )
    )
    parser.add_argument(
        "performance",
        help=(
            "Processed performance name, e.g. "
            "test_performance_02"
        ),
    )
    args = parser.parse_args()

    performance_name = args.performance.strip()

    if not performance_name:
        raise ValueError(
            "Performance name cannot be empty."
        )

    package_path, _ = (
        run_animation_package_export(
            performance_name
        )
    )

    package = validate_animation_package(
        package_path,
        performance_name,
    )

    blender_path = find_blender()

    require_file(
        YBOT_ASSET,
        "Project Y-Bot FBX asset",
    )

    preflight_path = write_preflight_manifest(
        performance_name,
        blender_path,
        package_path,
        package,
    )

    print()
    print("=" * 72)
    print("STAGE 22A PASS")
    print("=" * 72)
    print(f"Blender      : {blender_path}")
    print(f"Y-Bot asset  : {YBOT_ASSET}")
    print(f"Package      : {package_path}")
    print(
        "Frames       : "
        f"{package['timing']['frame_count']}"
    )
    print(
        "FPS estimate : "
        f"{package['timing']['fps_estimate']}"
    )
    print(
        "Root policy  : "
        f"{package['coordinate_system']['root_policy']}"
    )
    print(f"Preflight    : {preflight_path}")
    print()
    print(
        "Backend prerequisites are ready for "
        "Stage 22B headless Blender automation."
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
