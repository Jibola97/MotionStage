import os

# Headless plotting for FastAPI/server execution.
os.environ["MPLBACKEND"] = "Agg"

import json
import re
import shutil
import subprocess
from pathlib import Path

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
)

from fastapi.responses import FileResponse

from fastapi.middleware.cors import CORSMiddleware
from compare_performances import compare_performances
from process_performance import process_performance
from character_api import router as character_router


app = FastAPI(
    title="MotionStage API",
    description=(
        "Pose-based performance comparison "
        "and motion-divergence analysis."
    ),
    version="0.5.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
COMPARISONS_DIR = PROJECT_ROOT / "data" / "comparisons"
ML_DIR = PROJECT_ROOT / "data" / "ml"

RAW_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


VALID_NAME = re.compile(
    r"^[A-Za-z0-9_-]+$"
)

ALLOWED_VIDEO_EXTENSIONS = {
    ".mov",
    ".mp4",
    ".m4v",
}

POSE_JOINT_SCORE_THRESHOLD = 0.30
POSE_FRAME_JOINT_COVERAGE = 0.50
POSE_MIN_VALID_FRAME_RATIO = 0.50
POSE_MIN_VALID_FRAMES = 3


def assess_pose_quality(
    keypoints_path: Path,
) -> dict:
    import csv
    import math

    if not keypoints_path.exists():
        raise FileNotFoundError(
            f"Pose keypoints not found: {keypoints_path}"
        )

    frame_counts = {}

    with keypoints_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)

        required = {
            "frame",
            "joint_name",
            "pose_score",
        }

        fields = set(reader.fieldnames or [])
        missing = required - fields

        if missing:
            raise ValueError(
                "Pose-quality check cannot run; "
                "keypoints.csv is missing columns: "
                + ", ".join(sorted(missing))
            )

        for row in reader:
            frame = str(row["frame"])

            counts = frame_counts.setdefault(
                frame,
                {
                    "total": 0,
                    "confident": 0,
                },
            )

            counts["total"] += 1

            try:
                score = float(row["pose_score"])
            except (TypeError, ValueError):
                continue

            if (
                math.isfinite(score)
                and score >= POSE_JOINT_SCORE_THRESHOLD
            ):
                counts["confident"] += 1

    total_frames = len(frame_counts)
    valid_frames = 0

    for counts in frame_counts.values():
        total = counts["total"]

        if total <= 0:
            continue

        coverage = counts["confident"] / total

        if coverage >= POSE_FRAME_JOINT_COVERAGE:
            valid_frames += 1

    valid_frame_ratio = (
        valid_frames / total_frames
        if total_frames
        else 0.0
    )

    passed = (
        valid_frames >= POSE_MIN_VALID_FRAMES
        and valid_frame_ratio >= POSE_MIN_VALID_FRAME_RATIO
    )

    return {
        "passed": passed,
        "total_frames": total_frames,
        "valid_frames": valid_frames,
        "valid_frame_ratio": valid_frame_ratio,
    }

def find_raw_video(
    performance_name: str,
) -> Path:
    """
    Locate the original uploaded video for a processed
    MotionStage performance.
    """

    candidates = sorted(
        path
        for path in RAW_DIR.glob(
            f"{performance_name}.*"
        )
        if path.suffix.lower()
        in ALLOWED_VIDEO_EXTENSIONS
    )

    if not candidates:
        raise FileNotFoundError(
            "Could not locate raw video for "
            f"{performance_name!r}."
        )

    return candidates[0]


def generate_pose3d_assets(
    reference_name: str,
    comparison_name: str,
    summary_path: Path,
) -> dict:
    """
    Run the isolated MediaPipe 3D worker.

    FastAPI remains in the normal MotionStage
    environment. MediaPipe runs through
    .venv_pose3d/bin/python.
    """

    pose3d_python = (
        PROJECT_ROOT
        / ".venv_pose3d"
        / "bin"
        / "python"
    )

    extractor_script = (
        PROJECT_ROOT
        / "backend"
        / "motionstage"
        / "pose3d"
        / "extract_pose3d.py"
    )

    viewer_script = (
        PROJECT_ROOT
        / "backend"
        / "motionstage"
        / "pose3d"
        / "compare_pose3d_highlight.py"
    )

    model_path = (
        PROJECT_ROOT
        / "models"
        / "pose_landmarker_full.task"
    )

    required_paths = {
        "3D Python environment": pose3d_python,
        "3D extractor": extractor_script,
        "3D comparison viewer": viewer_script,
        "MediaPipe model": model_path,
        "comparison summary": summary_path,
    }

    for label, path in required_paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{label} not found: {path}"
            )

    reference_video = find_raw_video(
        reference_name
    )

    comparison_video = find_raw_video(
        comparison_name
    )

    reference_output_dir = (
        PROCESSED_DIR / reference_name
    )

    comparison_output_dir = (
        PROCESSED_DIR / comparison_name
    )

    reference_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference_pose3d = (
        reference_output_dir
        / "pose3d_world.csv"
    )

    comparison_pose3d = (
        comparison_output_dir
        / "pose3d_world.csv"
    )

    comparison_dir = (
        COMPARISONS_DIR
        / (
            f"{reference_name}_vs_"
            f"{comparison_name}"
        )
    )

    comparison_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    viewer_path = (
        comparison_dir
        / "pose3d_highlight.html"
    )

    print()
    print("=" * 72)
    print(
        "MotionStage 3D Worker"
    )
    print("=" * 72)

    print()
    print(
        "[3D 1/3] Extracting reference pose"
    )

    subprocess.run(
        [
            str(pose3d_python),
            str(extractor_script),
            str(reference_video),
            str(reference_pose3d),
            "--model",
            str(model_path),
        ],
        cwd=str(PROJECT_ROOT),
        check=True,
    )

    print()
    print(
        "[3D 2/3] Extracting comparison pose"
    )

    subprocess.run(
        [
            str(pose3d_python),
            str(extractor_script),
            str(comparison_video),
            str(comparison_pose3d),
            "--model",
            str(model_path),
        ],
        cwd=str(PROJECT_ROOT),
        check=True,
    )

    print()
    print(
        "[3D 3/3] Building divergence viewer"
    )

    subprocess.run(
        [
            str(pose3d_python),
            str(viewer_script),
            str(reference_pose3d),
            str(comparison_pose3d),
            str(summary_path),
            "--output",
            str(viewer_path),
        ],
        cwd=str(PROJECT_ROOT),
        check=True,
    )

    print()
    print(
        "MotionStage 3D assets complete."
    )
    print(
        f"Viewer: {viewer_path}"
    )
    print("=" * 72)

    return {
        "reference_pose3d": str(
            reference_pose3d.relative_to(
                PROJECT_ROOT
            )
        ),
        "comparison_pose3d": str(
            comparison_pose3d.relative_to(
                PROJECT_ROOT
            )
        ),
        "viewer_path": str(
            viewer_path.relative_to(
                PROJECT_ROOT
            )
        ),
    }


def generate_ml_analysis(
    reference_name: str,
    comparison_name: str,
    summary_path: Path,
) -> dict:
    """
    Score one completed comparison with the Stage 18D
    Isolation Forest model.

    The ML worker runs in .venv_pose3d because that
    environment contains scikit-learn and joblib.
    """

    ml_python = (
        PROJECT_ROOT
        / ".venv_pose3d"
        / "bin"
        / "python"
    )

    ml_script = (
        PROJECT_ROOT
        / "backend"
        / "motionstage"
        / "ml"
        / "score_comparison_anomaly.py"
    )

    model_path = (
        ML_DIR
        / "isolation_forest_model.joblib"
    )

    comparison_dir = (
        COMPARISONS_DIR
        / (
            f"{reference_name}_vs_"
            f"{comparison_name}"
        )
    )

    comparison_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        comparison_dir
        / "ml_anomaly_summary.json"
    )

    required_paths = {
        "ML Python environment":
            ml_python,

        "ML scoring worker":
            ml_script,

        "Isolation Forest model":
            model_path,

        "comparison summary":
            summary_path,
    }

    for label, path in required_paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{label} not found: {path}"
            )

    print()
    print("=" * 72)
    print(
        "MotionStage ML Worker"
    )
    print("=" * 72)

    subprocess.run(
        [
            str(ml_python),
            str(ml_script),
            reference_name,
            comparison_name,
            "--summary",
            str(summary_path),
            "--model",
            str(model_path),
            "--output",
            str(output_path),
        ],
        cwd=str(PROJECT_ROOT),
        check=True,
    )

    if not output_path.exists():
        raise FileNotFoundError(
            "ML worker completed but did not "
            f"create: {output_path}"
        )

    with output_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


@app.get(
    "/ml/anomaly/{reference}/{comparison}"
)
def get_ml_anomaly(
    reference: str,
    comparison: str,
):
    validate_performance_name(reference)
    validate_performance_name(comparison)

    path = (
        COMPARISONS_DIR
        / f"{reference}_vs_{comparison}"
        / "ml_anomaly_summary.json"
    )

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "ML anomaly analysis has not been "
                "generated for this comparison."
            ),
        )

    return load_summary(path)


def validate_performance_name(
    name: str,
) -> None:

    if not VALID_NAME.fullmatch(name):
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid performance name. "
                "Use letters, numbers, underscores "
                "and hyphens only."
            ),
        )


def get_summary_path(
    reference: str,
    comparison: str,
) -> Path:

    return (
        COMPARISONS_DIR
        / f"{reference}_vs_{comparison}"
        / "final_summary.json"
    )


def load_summary(
    path: Path,
) -> dict:

    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail="Comparison result not found.",
        )

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail=(
                "The comparison summary exists "
                "but contains invalid JSON."
            ),
        )


def allocate_performance_name(
    base_name: str,
) -> str:
    # Return a collision-safe internal performance ID.

    validate_performance_name(base_name)

    candidate = base_name
    counter = 1

    while True:
        processed_exists = (
            PROCESSED_DIR / candidate
        ).exists()

        raw_exists = any(
            path.is_file()
            for path in RAW_DIR.glob(
                f"{candidate}.*"
            )
        )

        if not processed_exists and not raw_exists:
            return candidate

        candidate = (
            f"{base_name}_{counter:02d}"
        )
        counter += 1

def process_uploaded_video(
    file: UploadFile,
) -> dict:

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file has no filename.",
        )

    original_name = Path(
        file.filename
    ).name

    extension = (
        Path(original_name)
        .suffix
        .lower()
    )

    if extension not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported video format. "
                "Use MOV, MP4 or M4V."
            ),
        )

    base_performance_name = (
        Path(original_name).stem
    )

    validate_performance_name(
        base_performance_name
    )

    performance_name = (
        allocate_performance_name(
            base_performance_name
        )
    )

    raw_path = (
        RAW_DIR
        / f"{performance_name}{extension}"
    )

    processed_dir = (
        PROCESSED_DIR
        / performance_name
    )

    try:
        with raw_path.open(
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer,
            )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Could not save uploaded video: "
                f"{exc}"
            ),
        )

    finally:
        file.file.close()

    try:
        process_performance(
            str(raw_path)
        )

    except Exception as exc:

        if processed_dir.exists():
            shutil.rmtree(
                processed_dir,
                ignore_errors=True,
            )

        if raw_path.exists():
            raw_path.unlink(
                missing_ok=True
            )

        raise HTTPException(
            status_code=500,
            detail=(
                "MotionStage processing failed: "
                f"{exc}"
            ),
        )


    pose_quality = assess_pose_quality(
        processed_dir / "keypoints.csv"
    )

    print(
        "[POSE QUALITY]",
        performance_name,
        f"{pose_quality['valid_frames']}/"
        f"{pose_quality['total_frames']} "
        "valid frames "
        f"({pose_quality['valid_frame_ratio'] * 100:.1f}%)",
    )

    if not pose_quality["passed"]:
        if processed_dir.exists():
            shutil.rmtree(
                processed_dir,
                ignore_errors=True,
            )

        if raw_path.exists():
            raw_path.unlink(
                missing_ok=True
            )

        raise HTTPException(
            status_code=422,
            detail=(
                "No reliable person pose was detected. "
                f"Only {pose_quality['valid_frames']} of "
                f"{pose_quality['total_frames']} frames "
                "contained enough confident body joints. "
                "Keep the person's body clearly visible "
                "and try again."
            ),
        )

    expected_outputs = {
        "pose_video":
            processed_dir / "pose.mp4",

        "raw_keypoints":
            processed_dir / "keypoints.csv",

        "smoothed_keypoints":
            processed_dir / "keypoints_smoothed.csv",

        "normalised_keypoints":
            processed_dir / "keypoints_normalised.csv",

        "joint_features":
            processed_dir / "joint_features.csv",

        "frame_features":
            processed_dir / "frame_features.csv",
    }

    missing_outputs = [
        name
        for name, path
        in expected_outputs.items()
        if not path.exists()
    ]

    if missing_outputs:
        raise HTTPException(
            status_code=500,
            detail={
                "message":
                    "Processing completed but "
                    "expected outputs are missing.",

                "missing_outputs":
                    missing_outputs,
            },
        )

    return {
        "performance_name":
            performance_name,

        "original_filename":
            original_name,

        "outputs": {
            name:
                str(
                    path.relative_to(
                        PROJECT_ROOT
                    )
                )

            for name, path
            in expected_outputs.items()
        },
    }


@app.get("/")
def root():

    return {
        "name":
            "MotionStage API",

        "version":
            "0.5.0",

        "status":
            "running",
    }


@app.get("/health")
def health():

    return {
        "status":
            "healthy",
    }

@app.get(
    "/pose3d/viewer/{reference}/{comparison}"
)
def get_pose3d_viewer(
    reference: str,
    comparison: str,
):
    validate_performance_name(reference)
    validate_performance_name(comparison)

    viewer_path = (
        COMPARISONS_DIR
        / f"{reference}_vs_{comparison}"
        / "pose3d_highlight.html"
    )

    if not viewer_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "3D viewer has not been generated "
                "for this comparison."
            ),
        )

    return FileResponse(
        path=viewer_path,
        media_type="text/html",
    )
    


@app.get(
    "/comparisons/{reference}/{comparison}"
)
def get_comparison(
    reference: str,
    comparison: str,
):

    validate_performance_name(reference)
    validate_performance_name(comparison)

    return load_summary(
        get_summary_path(
            reference,
            comparison,
        )
    )


@app.post(
    "/comparisons/{reference}/{comparison}/run"
)
def run_comparison(
    reference: str,
    comparison: str,
):

    validate_performance_name(reference)
    validate_performance_name(comparison)

    reference_dir = (
        PROCESSED_DIR
        / reference
    )

    comparison_dir = (
        PROCESSED_DIR
        / comparison
    )

    if not reference_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Reference performance "
                f"'{reference}' was not found."
            ),
        )

    if not comparison_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Comparison performance "
                f"'{comparison}' was not found."
            ),
        )

    try:
        compare_performances(
            reference_name=reference,
            comparison_name=comparison,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "MotionStage comparison failed: "
                f"{exc}"
            ),
        )

    summary_path = get_summary_path(
        reference,
        comparison,
    )

    result = load_summary(
        summary_path
    )

    ml_result = {
        "available": False,
    }

    try:
        ml_result = generate_ml_analysis(
            reference_name=reference,
            comparison_name=comparison,
            summary_path=summary_path,
        )

    except Exception as exc:
        print()
        print("=" * 72)
        print(
            "WARNING: ML analysis could "
            "not be generated."
        )
        print(str(exc))
        print("=" * 72)

        ml_result = {
            "available": False,
            "error": str(exc),
        }

    return {
        "status":
            "completed",

        "reference":
            reference,

        "comparison":
            comparison,

        "result":
            result,

        "ml":
            ml_result,
    }


@app.post("/performances/upload")
def upload_performance(
    file: UploadFile = File(...),
):

    processed = process_uploaded_video(
        file
    )

    return {
        "status":
            "processed",

        **processed,

        "ready_for_comparison":
            True,
    }


@app.post("/analyse")
def analyse_videos(
    reference_video: UploadFile = File(...),
    comparison_video: UploadFile = File(...),
):

    # Prevent ambiguous naming before doing
    # expensive pose processing.
    if (
        reference_video.filename
        and comparison_video.filename
        and Path(reference_video.filename).stem
        == Path(comparison_video.filename).stem
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Reference and comparison videos "
                "must have different filenames."
            ),
        )

    print()
    print("=" * 72)
    print("MotionStage End-to-End Analysis")
    print("=" * 72)

    print()
    print("[1/3] PROCESSING REFERENCE VIDEO")
    print("=" * 72)

    reference = process_uploaded_video(
        reference_video
    )

    print()
    print("[2/3] PROCESSING COMPARISON VIDEO")
    print("=" * 72)

    comparison = process_uploaded_video(
        comparison_video
    )

    reference_name = (
        reference[
            "performance_name"
        ]
    )

    comparison_name = (
        comparison[
            "performance_name"
        ]
    )

    print()
    print("[3/3] COMPARING PERFORMANCES")
    print("=" * 72)

    try:
        compare_performances(
            reference_name=
                reference_name,

            comparison_name=
                comparison_name,
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Both videos were processed, "
                "but MotionStage comparison failed: "
                f"{exc}"
            ),
        )

    summary_path = get_summary_path(
        reference_name,
        comparison_name,
    )

    result = load_summary(
        summary_path
    )

    ml_result = {
        "available": False,
    }

    try:
        ml_result = generate_ml_analysis(
            reference_name=reference_name,
            comparison_name=comparison_name,
            summary_path=summary_path,
        )

    except Exception as exc:
        print()
        print("=" * 72)
        print(
            "WARNING: ML analysis could "
            "not be generated."
        )
        print(str(exc))
        print("=" * 72)

        ml_result = {
            "available": False,
            "error": str(exc),
        }

    pose3d_result = {
        "available": False,
        "viewer_url": None,
    }

    try:
        pose3d_assets = generate_pose3d_assets(
            reference_name=reference_name,
            comparison_name=comparison_name,
            summary_path=summary_path,
        )

        pose3d_result = {
            "available": True,
            **pose3d_assets,
            "viewer_url": (
                "http://127.0.0.1:8000"
                f"/pose3d/viewer/"
                f"{reference_name}/"
                f"{comparison_name}"
            ),
        }

    except Exception as exc:
        print()
        print("=" * 72)
        print(
            "WARNING: 3D analysis could "
            "not be generated."
        )
        print(str(exc))
        print("=" * 72)

        pose3d_result = {
            "available": False,
            "viewer_url": None,
            "error": str(exc),
        }

    print()
    print("=" * 72)
    print(
        "MotionStage End-to-End "
        "Analysis Complete"
    )
    print("=" * 72)

    return {
        "status":
            "completed",

        "reference": {
            "performance_name":
                reference_name,

            "original_filename":
                reference[
                    "original_filename"
                ],
        },

        "comparison": {
            "performance_name":
                comparison_name,

            "original_filename":
                comparison[
                    "original_filename"
                ],
        },

        "result":
            result,

        "ml":
            ml_result,

        "pose3d":
            pose3d_result,
    }


# Stage 22C — character generation API
app.include_router(character_router)
