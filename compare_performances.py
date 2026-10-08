import argparse
import json
from pathlib import Path

from backend.motionstage.comparison.align_performances import (
    align_performances,
)
from backend.motionstage.comparison.similarity_engine import (
    calculate_similarity,
)
from backend.motionstage.comparison.detect_divergence import (
    detect_divergence,
)
from backend.motionstage.comparison.plot_aligned_features import (
    plot_aligned_features,
)
from backend.motionstage.comparison.robust_localisation import (
    robust_localisation,
)
from backend.motionstage.comparison.isolated_divergence import (
    detect_isolated_divergence,
)


def load_json(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def compare_performances(
    reference_name: str,
    comparison_name: str,
) -> None:

    processed_root = Path(
        "data/processed"
    )

    reference_dir = (
        processed_root
        / reference_name
    )

    comparison_dir = (
        processed_root
        / comparison_name
    )

    # -------------------------------------------------
    # Required input files
    # -------------------------------------------------

    reference_features = (
        reference_dir
        / "frame_features.csv"
    )

    comparison_features = (
        comparison_dir
        / "frame_features.csv"
    )

    reference_keypoints = (
        reference_dir
        / "keypoints_normalised.csv"
    )

    comparison_keypoints = (
        comparison_dir
        / "keypoints_normalised.csv"
    )

    required_files = {
        "reference features":
            reference_features,

        "comparison features":
            comparison_features,

        "reference keypoints":
            reference_keypoints,

        "comparison keypoints":
            comparison_keypoints,
    }

    for label, path in (
        required_files.items()
    ):

        if not path.exists():
            raise FileNotFoundError(
                f"Missing {label}: {path}"
            )

    # -------------------------------------------------
    # Output structure
    # -------------------------------------------------

    comparison_id = (
        f"{reference_name}"
        f"_vs_{comparison_name}"
    )

    output_dir = (
        Path("data/comparisons")
        / comparison_id
    )

    plots_dir = (
        output_dir
        / "plots"
    )

    aligned_plots_dir = (
        plots_dir
        / "aligned_features"
    )

    robust_dir = (
        output_dir
        / "robust_localisation"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plots_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Standard constrained-DTW outputs
    alignment_csv = (
        output_dir
        / "dtw_alignment.csv"
    )

    similarity_json = (
        output_dir
        / "similarity_report.json"
    )

    similarity_csv = (
        output_dir
        / "similarity_by_step.csv"
    )

    divergence_csv = (
        output_dir
        / "divergence_timeline.csv"
    )

    divergence_json = (
        output_dir
        / "divergence_report.json"
    )

    divergence_plot = (
        plots_dir
        / "similarity_timeline.png"
    )

    # Final combined result
    final_summary_path = (
        output_dir
        / "final_summary.json"
    )

    print()
    print("=" * 72)
    print("MotionStage Complete Performance Analysis")
    print("=" * 72)

    print(
        f"Reference:  {reference_name}"
    )

    print(
        f"Comparison: {comparison_name}"
    )

    print()

    # =================================================
    # 1. CONSTRAINED DTW
    # =================================================

    print("[1/6] GLOBAL TEMPORAL ALIGNMENT")
    print("=" * 72)

    align_performances(
        reference_csv=str(
            reference_features
        ),

        comparison_csv=str(
            comparison_features
        ),

        output_csv=str(
            alignment_csv
        ),
    )

    # =================================================
    # 2. GLOBAL / REGIONAL SIMILARITY
    # =================================================

    print()
    print("[2/6] GLOBAL SIMILARITY ANALYSIS")
    print("=" * 72)

    calculate_similarity(
        reference_csv=str(
            reference_features
        ),

        comparison_csv=str(
            comparison_features
        ),

        reference_keypoints_csv=str(
            reference_keypoints
        ),

        comparison_keypoints_csv=str(
            comparison_keypoints
        ),

        alignment_csv=str(
            alignment_csv
        ),

        output_json=str(
            similarity_json
        ),

        output_csv=str(
            similarity_csv
        ),
    )

    # =================================================
    # 3. OVERALL DIVERGENCE
    # =================================================

    print()
    print("[3/6] OVERALL DIVERGENCE")
    print("=" * 72)

    detect_divergence(
        similarity_csv=str(
            similarity_csv
        ),

        output_csv=str(
            divergence_csv
        ),

        output_json=str(
            divergence_json
        ),

        output_plot=str(
            divergence_plot
        ),
    )

    # =================================================
    # 4. DIAGNOSTIC FEATURE PLOTS
    # =================================================

    print()
    print("[4/6] ALIGNED FEATURE PLOTS")
    print("=" * 72)

    plot_aligned_features(
        reference_csv=str(
            reference_features
        ),

        comparison_csv=str(
            comparison_features
        ),

        alignment_csv=str(
            alignment_csv
        ),

        output_dir=str(
            aligned_plots_dir
        ),
    )

    # =================================================
    # 5. ROBUST LOCALISATION
    # =================================================

    print()
    print("[5/6] ROBUST REGIONAL LOCALISATION")
    print("=" * 72)

    robust_localisation(
        reference_name,
        comparison_name,
    )

    # =================================================
    # 6. ISOLATED ARM DIVERGENCE
    # =================================================

    print()
    print("[6/6] ISOLATED ARM DIVERGENCE")
    print("=" * 72)

    detect_isolated_divergence(
        reference_name,
        comparison_name,
    )

    # =================================================
    # LOAD GENERATED REPORTS
    # =================================================

    global_similarity = load_json(
        similarity_json
    )

    overall_divergence = load_json(
        divergence_json
    )

    robust_similarity_path = (
        robust_dir
        / "similarity_report.json"
    )

    robust_divergence_path = (
        robust_dir
        / "divergence_report.json"
    )

    isolated_divergence_path = (
        robust_dir
        / "isolated_divergence_report.json"
    )

    robust_similarity = load_json(
        robust_similarity_path
    )

    robust_divergence = load_json(
        robust_divergence_path
    )

    isolated_divergence = load_json(
        isolated_divergence_path
    )

    # =================================================
    # FINAL PRODUCT-LEVEL SUMMARY
    # =================================================

    final_summary = {
        "reference":
            reference_name,

        "comparison":
            comparison_name,

        "global_similarity": {
            "overall_similarity":
                global_similarity.get(
                    "overall_similarity"
                ),

            "pose_similarity":
                global_similarity.get(
                    "pose_similarity"
                ),

            "position_similarity":
                global_similarity.get(
                    "position_similarity"
                ),

            "movement_speed_similarity":
                global_similarity.get(
                    "movement_speed_similarity"
                ),

            "regional_composite_similarity":
                global_similarity.get(
                    "regional_composite_similarity"
                ),

            "duration_similarity":
                global_similarity.get(
                    "duration_similarity"
                ),

            "regions":
                global_similarity.get(
                    "regions",
                    {},
                ),
        },

        "overall_divergence": {
            "most_divergent_region":
                overall_divergence.get(
                    "most_divergent_region"
                ),

            "reference_window":
                overall_divergence.get(
                    "reference_window"
                ),

            "comparison_window":
                overall_divergence.get(
                    "comparison_window"
                ),

            "regional_window_similarity":
                overall_divergence.get(
                    "worst_regional_window_similarity"
                ),

            "region_similarity":
                overall_divergence.get(
                    "region_similarity"
                ),
        },

        "robust_localisation": {
            "similarity":
                robust_similarity,

            "divergence":
                robust_divergence,
        },

        "isolated_arm_divergence": {
            "region":
                isolated_divergence.get(
                    "most_isolated_region"
                ),

            "isolation_gap":
                isolated_divergence.get(
                    "isolation_gap"
                ),

            "comparison_window":
                isolated_divergence.get(
                    "comparison_window"
                ),

            "reference_window":
                isolated_divergence.get(
                    "reference_window"
                ),

            "isolated_region_similarity":
                isolated_divergence.get(
                    "isolated_region_similarity"
                ),

            "peer_arm_similarity":
                isolated_divergence.get(
                    "peer_arm_similarity"
                ),

            "context_similarity":
                isolated_divergence.get(
                    "context_similarity"
                ),

            "components":
                isolated_divergence.get(
                    "isolated_region_components"
                ),
        },

        "methods": {
            "global_alignment":
                "constrained_dtw",

            "regional_localisation":
                "robust_region_resistant_dtw",

            "isolated_arm_detection":
                "bilateral_arm_isolation",
        },

        "interpretation_notes": [
            (
                "Similarity values are prototype "
                "indices rather than probabilities."
            ),

            (
                "No fixed isolated-divergence anomaly "
                "threshold has been established."
            ),

            (
                "Overall divergence and isolated-arm "
                "divergence answer different questions "
                "and should be interpreted separately."
            ),
        ],

        "outputs": {
            "alignment":
                str(alignment_csv),

            "similarity_report":
                str(similarity_json),

            "overall_divergence_report":
                str(divergence_json),

            "robust_directory":
                str(robust_dir),

            "final_summary":
                str(final_summary_path),
        },
    }

    with final_summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            final_summary,
            file,
            indent=2,
        )

    # =================================================
    # CLEAN TERMINAL SUMMARY
    # =================================================

    print()
    print("=" * 72)
    print("MotionStage Analysis Complete")
    print("=" * 72)

    print()
    print(
        f"Reference:  {reference_name}"
    )

    print(
        f"Comparison: {comparison_name}"
    )

    print()
    print("GLOBAL SIMILARITY")
    print("-----------------")

    print(
        f"Overall:  "
        f"{global_similarity['overall_similarity']:.2f}"
    )

    print(
        f"Pose:     "
        f"{global_similarity['pose_similarity']:.2f}"
    )

    print(
        f"Position: "
        f"{global_similarity['position_similarity']:.2f}"
    )

    print(
        f"Speed:    "
        f"{global_similarity['movement_speed_similarity']:.2f}"
    )

    print(
        f"Duration: "
        f"{global_similarity['duration_similarity']:.2f}"
    )

    print()
    print("ISOLATED ARM DEVIATION")
    print("----------------------")

    isolated_region = (
        isolated_divergence[
            "most_isolated_region"
        ]
    )

    isolation_gap = (
        isolated_divergence[
            "isolation_gap"
        ]
    )

    isolated_window = (
        isolated_divergence[
            "comparison_window"
        ]
    )

    print(
        f"Region: "
        f"{isolated_region}"
    )

    print(
        f"Isolation gap: "
        f"{isolation_gap:.2f}"
    )

    print(
        "Comparison window: "
        f"{isolated_window['start_seconds']:.2f}s "
        "to "
        f"{isolated_window['end_seconds']:.2f}s"
    )

    print()
    print(
        f"Final summary: "
        f"{final_summary_path}"
    )

    print()

    print(
        "Note: similarity values are prototype "
        "indices, not probabilities."
    )

    print(
        "No fixed anomaly threshold has been "
        "established."
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Run the complete MotionStage "
            "performance-comparison pipeline."
        )
    )

    parser.add_argument(
        "reference",
        help=(
            "Processed reference performance name, "
            "e.g. test_performance_01"
        ),
    )

    parser.add_argument(
        "comparison",
        help=(
            "Processed comparison performance name, "
            "e.g. test_performance_03"
        ),
    )

    args = parser.parse_args()

    compare_performances(
        reference_name=args.reference,
        comparison_name=args.comparison,
    )
