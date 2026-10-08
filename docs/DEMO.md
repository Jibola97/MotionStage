# MotionStage Demo

This guide walks through a representative end-to-end MotionStage comparison.

For full installation and environment setup, see the main [README](../README.md).

## 1. Prerequisites

Before starting the demo, ensure that:

- the main MotionStage Python environment is installed
- the dedicated `.venv_pose3d` environment is installed
- frontend dependencies have been installed
- the MediaPipe pose model is available locally at `models/pose_landmarker_full.task`
- the Y-Bot FBX is available locally at `assets/characters/ybot.fbx` if character export is required

Neither runtime asset is distributed through this repository.

## 2. Prepare Two Videos

Prepare two short full-body performance videos:

- `reference_demo.mp4`
- `comparison_demo.mp4`

The reference is the target movement. The comparison is the performance MotionStage evaluates.

For reliable processing:

- keep the full body visible
- avoid major occlusion
- keep the performer clearly visible
- use MP4, MOV or M4V

MotionStage may reject videos where reliable body pose cannot be detected.

## 3. Start the Backend

From the MotionStage project root, activate the main environment:

```bash
source .venv/bin/activate
```

Start FastAPI:

```bash
PYTHONPATH="$PWD:$PWD/backend/motionstage/api" \
python -m uvicorn backend.motionstage.api.main:app --reload --port 8000
```

The backend should be available at `http://127.0.0.1:8000`.

Keep this terminal running.

## 4. Start the Frontend

Open a second terminal:

```bash
cd ~/Desktop/MotionStage/frontend
npm run dev
```

Open `http://localhost:3000` in the browser.

## 5. Upload the Performances

In the MotionStage interface:

1. upload `reference_demo.mp4` as the reference performance
2. upload `comparison_demo.mp4` as the comparison performance
3. wait for both videos to load successfully

## 6. Run the Analysis

Select **Analyse performances**.

MotionStage processes both videos through pose extraction, quality validation, smoothing, normalisation, feature extraction, alignment and comparison.

## 7. Review Similarity Results

The main analysis reports:

- Overall Similarity Index
- Pose similarity
- Position similarity
- Movement Speed similarity
- Duration similarity

These metrics provide complementary views of the comparison rather than reducing the result to a single pose score.

![MotionStage analysis](images/motionstage-analysis.png)

## 8. Inspect Movement Divergence

Review the divergence section to inspect:

- the strongest regional divergence
- the detected divergence time window
- body-region similarity scores
- isolated-arm divergence signals
- the comparison timeline

![MotionStage divergence](images/motionstage-divergence.png)

The regional breakdown provides additional body-region detail:

![MotionStage regional analysis](images/motionstage-regional.png)

## 9. Review Exploratory Anomaly Analysis

MotionStage also runs an Isolation Forest over movement-difference windows.

This provides an independent exploratory signal and remains separate from the deterministic similarity and divergence analysis.

![MotionStage anomaly analysis](images/motionstage-anomaly.png)

## 10. Open the 3D Comparison

Open the 3D divergence viewer from the analysis interface.

The viewer displays synchronised reference and comparison skeletons and follows both performances using relative progress.

Detected divergence regions can be highlighted during relevant movement windows.

![MotionStage 3D viewer](images/motionstage-pose3d.png)

## 11. Generate an Animated Character

To test the character-animation pipeline, select **Generate animated FBX**.

MotionStage then:

1. exports the analysed motion package
2. launches Blender headlessly
3. imports the local Y-Bot character
4. retargets the comparison-performance motion
5. validates the retarget
6. saves a Blender project
7. exports an animated FBX

A successful run displays **FBX ready** in the interface.

The generated Blender project and animated FBX can then be accessed through the frontend.

## Expected Demo Outcome

A successful MotionStage demo demonstrates the complete path from two human-performance videos to:

- processed pose trajectories
- multiple similarity measures
- temporal divergence localisation
- body-region divergence analysis
- exploratory anomaly windows
- synchronised 3D comparison
- retargeted character animation
- animated FBX export

## Demo Interface

![MotionStage interface](images/motionstage-hero.png)
