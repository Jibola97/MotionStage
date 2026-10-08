"""
MotionStage animated-character API.

Exposes the validated headless Blender retarget/export pipeline through
FastAPI while leaving the existing analysis endpoints unchanged.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse


router = APIRouter(
    prefix="/performances",
    tags=["character"],
)

# character_api.py lives at:
# MotionStage/backend/motionstage/api/character_api.py
PROJECT_ROOT = Path(__file__).resolve().parents[3]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ANIMATION_DIR = PROJECT_ROOT / "data" / "animation"

CHARACTER_GENERATOR = (
    PROJECT_ROOT
    / "backend"
    / "motionstage"
    / "animation"
    / "generate_character_fbx.py"
)

VALID_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


def validate_performance_name(performance_name: str) -> str:
    if not VALID_NAME.fullmatch(performance_name):
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid performance name. "
                "Use letters, numbers, underscores and hyphens only."
            ),
        )
    return performance_name


def character_paths(performance_name: str) -> dict[str, Path]:
    character_dir = (
        ANIMATION_DIR
        / performance_name
        / "character"
    )

    return {
        "directory": character_dir,
        "fbx": character_dir / f"{performance_name}_ybot.fbx",
        "blend": character_dir / f"{performance_name}_ybot.blend",
        "manifest": character_dir / "character_build_manifest.json",
        "pose3d": (
            PROCESSED_DIR
            / performance_name
            / "pose3d_world.csv"
        ),
    }


def load_manifest(manifest_path: Path) -> dict:
    if not manifest_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Character build manifest has not been generated.",
        )

    try:
        return json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Character build manifest contains invalid JSON: {exc}",
        )


def character_response(
    performance_name: str,
    manifest: dict,
    status: str = "completed",
) -> dict:
    paths = character_paths(performance_name)

    return {
        "status": status,
        "performance_name": performance_name,
        "frame_count": manifest.get("frame_count"),
        "root_policy": manifest.get("root_policy"),
        "validation": manifest.get("validation", []),
        "fbx_url": f"/performances/{performance_name}/character/fbx",
        "blend_url": f"/performances/{performance_name}/character/blend",
        "manifest_url": f"/performances/{performance_name}/character/manifest",
        "outputs": {
            "fbx": (
                str(paths["fbx"].relative_to(PROJECT_ROOT))
                if paths["fbx"].exists()
                else None
            ),
            "blend": (
                str(paths["blend"].relative_to(PROJECT_ROOT))
                if paths["blend"].exists()
                else None
            ),
            "manifest": (
                str(paths["manifest"].relative_to(PROJECT_ROOT))
                if paths["manifest"].exists()
                else None
            ),
        },
    }


@router.get("/{performance_name}/character")
def get_character_status(performance_name: str):
    validate_performance_name(performance_name)
    paths = character_paths(performance_name)

    if not paths["manifest"].exists():
        return {
            "status": "not_generated",
            "performance_name": performance_name,
            "pose3d_ready": paths["pose3d"].exists(),
            "fbx_url": None,
        }

    manifest = load_manifest(paths["manifest"])

    build_status = (
        "completed"
        if (
            manifest.get("status") == "completed"
            and paths["fbx"].exists()
        )
        else "incomplete"
    )

    return character_response(
        performance_name,
        manifest,
        status=build_status,
    )


@router.post("/{performance_name}/character/generate")
def generate_character(
    performance_name: str,
    force: bool = False,
):
    validate_performance_name(performance_name)
    paths = character_paths(performance_name)

    if not paths["pose3d"].exists():
        raise HTTPException(
            status_code=409,
            detail={
                "message": (
                    f"3D pose data is not ready for '{performance_name}'."
                ),
                "required_file": str(
                    paths["pose3d"].relative_to(PROJECT_ROOT)
                ),
                "next_step": (
                    "Generate pose3d_world.csv first, then retry "
                    "character generation."
                ),
            },
        )

    if not CHARACTER_GENERATOR.exists():
        raise HTTPException(
            status_code=500,
            detail=(
                "Character generator worker not found: "
                f"{CHARACTER_GENERATOR}"
            ),
        )

    if (
        not force
        and paths["manifest"].exists()
        and paths["fbx"].exists()
    ):
        manifest = load_manifest(paths["manifest"])
        if manifest.get("status") == "completed":
            return character_response(
                performance_name,
                manifest,
                status="already_generated",
            )

    completed = subprocess.run(
        [
            sys.executable,
            str(CHARACTER_GENERATOR),
            performance_name,
        ],
        cwd=str(PROJECT_ROOT),
        text=True,
        capture_output=True,
        check=False,
    )

    if completed.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Character generation failed.",
                "exit_code": completed.returncode,
                "stdout_tail": (
                    completed.stdout[-6000:]
                    if completed.stdout
                    else ""
                ),
                "stderr_tail": (
                    completed.stderr[-6000:]
                    if completed.stderr
                    else ""
                ),
            },
        )

    missing = [
        key
        for key in ("manifest", "fbx", "blend")
        if not paths[key].exists()
    ]

    if missing:
        raise HTTPException(
            status_code=500,
            detail={
                "message": (
                    "Character generation finished, "
                    "but expected outputs are missing."
                ),
                "missing_outputs": missing,
            },
        )

    manifest = load_manifest(paths["manifest"])

    if manifest.get("status") != "completed":
        raise HTTPException(
            status_code=500,
            detail={
                "message": (
                    "Character worker finished but manifest "
                    "did not report completed status."
                ),
                "manifest": manifest,
            },
        )

    return character_response(
        performance_name,
        manifest,
        status="completed",
    )


@router.get("/{performance_name}/character/manifest")
def get_character_manifest(performance_name: str):
    validate_performance_name(performance_name)
    return load_manifest(
        character_paths(performance_name)["manifest"]
    )


@router.get("/{performance_name}/character/fbx")
def download_character_fbx(performance_name: str):
    validate_performance_name(performance_name)
    path = character_paths(performance_name)["fbx"]

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="Animated character FBX has not been generated.",
        )

    return FileResponse(
        path=str(path),
        filename=(
            f"MotionStage_{performance_name}_"
            "animated_character.fbx"
        ),
        media_type="application/octet-stream",
    )


@router.get("/{performance_name}/character/blend")
def download_character_blend(performance_name: str):
    validate_performance_name(performance_name)
    path = character_paths(performance_name)["blend"]

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="Character Blender snapshot has not been generated.",
        )

    return FileResponse(
        path=str(path),
        filename=(
            f"MotionStage_{performance_name}_"
            "character_project.blend"
        ),
        media_type="application/octet-stream",
    )
