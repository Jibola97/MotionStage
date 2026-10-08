"use client";

import {
  ChangeEvent,
  Dispatch,
  SetStateAction,
  useEffect,
  useRef,
  useState,
} from "react";

import {
  analysePerformances,
  characterBlendDownloadUrl,
  characterFbxDownloadUrl,
  generateCharacter,
  getCharacterStatus,
  MotionStageCharacterResult,
  MotionStageResult,
} from "@/lib/motionstage-api";


type SelectedVideo = {
  file: File;
  url: string;
};

type TimeWindow = {
  start_seconds: number;
  end_seconds: number;
};

type RegionSimilarity = {
  pose_similarity: number;
  position_similarity: number;
  speed_similarity: number;
  composite_similarity: number;
};

type RegionName =
  | "left_arm"
  | "right_arm"
  | "lower_body"
  | "torso";

type FinalSummary = {
  reference: string;
  comparison: string;

  global_similarity: {
    overall_similarity: number;
    pose_similarity: number;
    position_similarity: number;
    movement_speed_similarity: number;
    regional_composite_similarity: number;
    duration_similarity: number;
    regions: Record<
      RegionName,
      RegionSimilarity
    >;
  };

  overall_divergence: {
    most_divergent_region: RegionName;
    reference_window: TimeWindow;
    comparison_window: TimeWindow;
    regional_window_similarity: number;
    region_similarity: Record<
      RegionName,
      number
    >;
  };

  robust_localisation: {
    similarity: {
      overall_similarity: number;
      pose_similarity: number;
      position_similarity: number;
      movement_speed_similarity: number;
      regional_composite_similarity: number;
      duration_similarity: number;
      regions: Record<
        RegionName,
        RegionSimilarity
      >;
    };
    regional_weights: {
      pose: number;
      position: number;
      speed: number;
    };
    position_max_distance: number;
    alignment_steps: number;
    notes: string;
  };

  divergence?: {
    window_seconds: number;
    reference_window: TimeWindow;
    comparison_window: TimeWindow;
    most_divergent_region: RegionName;
    worst_regional_window_similarity: number;
    region_similarity: Record<
      RegionName,
      number
    >;
    divergent_region_components: {
      pose_similarity: number;
      position_similarity: number;
      speed_similarity: number;
    };
    notes: string;
  };

  isolated_arm_divergence: {
    region: "left_arm" | "right_arm";
    isolation_gap: number;
    comparison_window: TimeWindow;
    reference_window: TimeWindow;
    isolated_region_similarity: number;
    peer_arm_similarity: number;
    context_similarity: number;
    components: {
      pose_similarity: number;
      position_similarity: number;
      speed_similarity: number;
    };
  };

  methods: {
    global_alignment: string;
    regional_localisation: string;
    isolated_arm_detection: string;
  };

  interpretation_notes: string[];

  outputs: Record<string, string>;
};


const REGION_LABELS: Record<
  RegionName,
  string
> = {
  left_arm: "Left arm",
  right_arm: "Right arm",
  lower_body: "Lower body",
  torso: "Torso",
};

const LAST_ANALYSIS_STORAGE_KEY =
  "motionstage:last-analysis";


export default function Home() {
  const [referenceVideo, setReferenceVideo] =
    useState<SelectedVideo | null>(null);

  const [comparisonVideo, setComparisonVideo] =
    useState<SelectedVideo | null>(null);

  const [analysisResult, setAnalysisResult] =
    useState<MotionStageResult | null>(null);

  const [analysisError, setAnalysisError] =
    useState<string | null>(null);

  const [isAnalysing, setIsAnalysing] =
    useState(false);

  const [analysisElapsed, setAnalysisElapsed] =
    useState(0);

  const bothVideosSelected =
    referenceVideo !== null &&
    comparisonVideo !== null;

  const summary =
    analysisResult?.result
      ? (analysisResult.result as FinalSummary)
      : null;



  useEffect(() => {
    try {
      const saved =
        window.localStorage.getItem(
          LAST_ANALYSIS_STORAGE_KEY
        );

      if (!saved) {
        return;
      }

      const parsed =
        JSON.parse(saved) as MotionStageResult;

      if (
        parsed &&
        parsed.reference &&
        parsed.comparison &&
        parsed.result
      ) {
        setAnalysisResult(parsed);
      } else {
        window.localStorage.removeItem(
          LAST_ANALYSIS_STORAGE_KEY
        );
      }
    } catch {
      window.localStorage.removeItem(
        LAST_ANALYSIS_STORAGE_KEY
      );
    }
  }, []);

  useEffect(() => {
    if (!isAnalysing) {
      return;
    }

    const timer = window.setInterval(() => {
      setAnalysisElapsed(
        (seconds) => seconds + 1
      );
    }, 1000);

    return () => {
      window.clearInterval(timer);
    };
  }, [isAnalysing]);


  function clearAnalysis() {
    setAnalysisResult(null);
    setAnalysisError(null);

    window.localStorage.removeItem(
      LAST_ANALYSIS_STORAGE_KEY
    );
  }


  async function handleAnalyse() {
    if (
      !referenceVideo ||
      !comparisonVideo
    ) {
      return;
    }

    setAnalysisElapsed(0);
    setIsAnalysing(true);
    setAnalysisError(null);
    setAnalysisResult(null);

    window.localStorage.removeItem(
      LAST_ANALYSIS_STORAGE_KEY
    );

    try {
      const result =
        await analysePerformances(
          referenceVideo.file,
          comparisonVideo.file,
        );

      setAnalysisResult(result);

      window.localStorage.setItem(
        LAST_ANALYSIS_STORAGE_KEY,
        JSON.stringify(result),
      );
    } catch (error) {
      if (error instanceof Error) {
        setAnalysisError(error.message);
      } else {
        setAnalysisError(
          "An unexpected error occurred."
        );
      }
    } finally {
      setIsAnalysing(false);
    }
  }


  return (
    <main className="min-h-screen bg-[#07090d] text-white">
      <header className="border-b border-white/10">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-5 lg:px-8">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-white text-sm font-bold text-black">
              M
            </div>

            <div>
              <p className="text-lg font-semibold tracking-tight">
                MotionStage
              </p>

              <p className="text-xs text-white/40">
                Motion intelligence
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2 rounded-full border border-emerald-400/20 bg-emerald-400/5 px-3 py-1.5 text-xs text-emerald-300">
            <span className="h-2 w-2 rounded-full bg-emerald-400" />
            Analysis engine ready
          </div>
        </div>
      </header>


      <section className="mx-auto max-w-7xl px-6 pb-24 pt-20 lg:px-8">
        <div className="mx-auto max-w-3xl text-center">
          <p className="mb-4 text-sm font-medium uppercase tracking-[0.25em] text-cyan-400">
            Computer Vision Motion Analysis
          </p>

          <h1 className="text-5xl font-semibold tracking-tight sm:text-6xl">
            Compare human movement

            <span className="block bg-gradient-to-r from-cyan-300 to-violet-400 bg-clip-text text-transparent">
              frame by frame.
            </span>
          </h1>

          <p className="mx-auto mt-6 max-w-2xl text-base leading-7 text-white/55 sm:text-lg">
            Upload a reference performance and a comparison
            performance. MotionStage analyses pose, timing,
            position and movement speed to identify where
            the performances diverge.
          </p>
        </div>


        <div className="mt-16 grid gap-6 lg:grid-cols-2">
          <UploadCard
            number="01"
            title="Reference performance"
            description={
              "The movement you want to use as the target performance."
            }
            selectedVideo={referenceVideo}
            setSelectedVideo={setReferenceVideo}
            locked={isAnalysing}
            onSelectionChange={clearAnalysis}
          />

          <UploadCard
            number="02"
            title="Comparison performance"
            description={
              "The performance you want MotionStage to evaluate."
            }
            selectedVideo={comparisonVideo}
            setSelectedVideo={setComparisonVideo}
            locked={isAnalysing}
            onSelectionChange={clearAnalysis}
          />
        </div>


        <div className="mt-8 flex flex-col items-center">
          <button
            type="button"
            onClick={handleAnalyse}
            disabled={
              !bothVideosSelected ||
              isAnalysing
            }
            className="
              min-w-56
              rounded-xl
              bg-white
              px-8
              py-3.5
              text-sm
              font-semibold
              text-black
              transition
              enabled:hover:bg-cyan-100
              disabled:cursor-not-allowed
              disabled:opacity-40
            "
          >
            {isAnalysing ? (
              <span className="flex items-center justify-center gap-3">
                <span className="h-4 w-4 animate-spin rounded-full border-2 border-black/25 border-t-black" />

                Analysing performances
              </span>
            ) : (
              "Analyse performances"
            )}
          </button>

          {isAnalysing && (
            <div role="status" aria-live="polite" className="mt-6 w-full max-w-2xl rounded-2xl border border-cyan-400/15 bg-cyan-400/[0.04] p-6">
              <div className="flex items-start gap-4">
                <div className="mt-1 flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-cyan-400/15 bg-cyan-400/5">
                  <span className="h-4 w-4 animate-pulse rounded-full bg-cyan-300" />
                </div>

                <div>
                  <p className="font-medium text-white/90">
                    MotionStage is analysing your performances
                  </p>

                  <p className="mt-2 text-sm leading-6 text-white/45">
                    The computer-vision pipeline is processing both videos
                    and comparing their movement data. Keep this tab open
                    until the analysis completes.
                  </p>

                  <p className="mt-3 text-xs font-medium uppercase tracking-[0.15em] text-cyan-300/70">
                    Elapsed time: {analysisElapsed}s
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>


        {analysisError && (
          <div role="alert" className="mx-auto mt-6 max-w-2xl rounded-2xl border border-red-400/20 bg-red-400/5 p-5">
            <p className="text-sm font-semibold text-red-300">
              Analysis failed
            </p>

            <p className="mt-2 text-sm leading-6 text-red-200/70">
              {analysisError}
            </p>
          </div>
        )}


        {analysisResult && summary && (
          <ResultsDashboard
            analysisResult={analysisResult}
            summary={summary}
            comparisonVideo={comparisonVideo}
          />
        )}


        {!summary && (
          <div className="mt-20 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <MetricPreview
              label="Pose"
              value="Joint angles"
            />

            <MetricPreview
              label="Position"
              value="Body alignment"
            />

            <MetricPreview
              label="Motion"
              value="Movement speed"
            />

            <MetricPreview
              label="Localisation"
              value="Body-region divergence"
            />
          </div>
        )}
      </section>
    </main>
  );
}



function ResultsDashboard({
  analysisResult,
  summary,
  comparisonVideo,
}: {
  analysisResult: MotionStageResult;
  summary: FinalSummary;
  comparisonVideo: SelectedVideo | null;
}) {
  const [comparisonDuration, setComparisonDuration] =
    useState<number | null>(null);

  const global =
    summary.global_similarity;

  const divergence =
    summary.overall_divergence;

  const divergenceDetails =
    summary.divergence;

  const isolated =
    summary.isolated_arm_divergence;

  const ml =
    analysisResult.ml;

  const pose3d =
    analysisResult.pose3d;

  const regionEntries =
    (
      [
        "left_arm",
        "right_arm",
        "lower_body",
        "torso",
      ] as RegionName[]
    ).map((region) => ({
      region,
      label: REGION_LABELS[region],
      value:
        global.regions[region]
          .composite_similarity,
    }));

  const globalSignals = [
    {
      label: "Pose",
      value: global.pose_similarity,
    },
    {
      label: "Position",
      value: global.position_similarity,
    },
    {
      label: "Movement speed",
      value:
        global.movement_speed_similarity,
    },
    {
      label: "Duration",
      value: global.duration_similarity,
    },
  ];

  const lowestGlobalSignal =
    globalSignals.reduce(
      (lowest, current) =>
        current.value < lowest.value
          ? current
          : lowest
    );

  const mlTopWindowEnd =
    ml?.available &&
    ml.top_window
      ? ml.top_window.end_seconds
      : 0;

  const fallbackDuration =
    Math.max(
      divergence.comparison_window
        .end_seconds,
      isolated.comparison_window
        .end_seconds,
      mlTopWindowEnd,
      1
    );

  const timelineDuration =
    comparisonDuration &&
    Number.isFinite(comparisonDuration) &&
    comparisonDuration > 0
      ? comparisonDuration
      : fallbackDuration;


  return (
    <section className="mt-12">
      {comparisonVideo && (
        <video
          src={comparisonVideo.url}
          preload="metadata"
          className="hidden"
          onLoadedMetadata={(event) => {
            const duration =
              event.currentTarget.duration;

            if (
              Number.isFinite(duration) &&
              duration > 0
            ) {
              setComparisonDuration(
                duration
              );
            }
          }}
        />
      )}


      <div className="rounded-3xl border border-emerald-400/20 bg-emerald-400/[0.035] p-6 sm:p-8">
        <div className="flex flex-col gap-6 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <div className="flex items-center gap-2">
              <span className="h-2 w-2 rounded-full bg-emerald-400" />

              <p className="text-sm font-semibold text-emerald-300">
                Analysis complete
              </p>
            </div>

            <p className="mt-3 text-sm text-white/45">
              {analysisResult.reference.original_filename}
              <span className="mx-2 text-white/20">
                vs
              </span>
              {analysisResult.comparison.original_filename}
            </p>
          </div>

          <div className="sm:text-right">
            <p className="text-xs uppercase tracking-[0.18em] text-white/30">
              Overall similarity index
            </p>

            <p className="mt-1 text-5xl font-semibold tracking-tight text-white">
              {formatScore(
                global.overall_similarity
              )}
            </p>

            <p className="mt-1 text-xs text-white/30">
              Composite score / 100
            </p>
          </div>
        </div>
      </div>


      <div className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <ResultMetricCard
          label="Pose"
          value={global.pose_similarity}
          description="Joint-angle similarity"
        />

        <ResultMetricCard
          label="Position"
          value={global.position_similarity}
          description="Body-position similarity"
        />

        <ResultMetricCard
          label="Movement speed"
          value={
            global.movement_speed_similarity
          }
          description="Motion-speed similarity"
        />

        <ResultMetricCard
          label="Duration"
          value={global.duration_similarity}
          description="Performance-duration similarity"
        />
      </div>


      <CharacterExportCard
        performanceName={
          analysisResult.comparison.performance_name
        }
      />


      <div className="mt-6 grid gap-6 lg:grid-cols-[1.15fr_0.85fr]">
        <div className="rounded-3xl border border-white/10 bg-white/[0.025] p-6 sm:p-8">
          <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
            <div>
              <p className="text-xs uppercase tracking-[0.18em] text-white/30">
                Primary divergence
              </p>

              <h2 className="mt-3 text-2xl font-semibold">
                {REGION_LABELS[
                  divergence.most_divergent_region
                ]}
              </h2>

              <p className="mt-2 max-w-xl text-sm leading-6 text-white/45">
                This is the body region with the lowest
                regional similarity in the strongest
                detected divergence window.
              </p>
            </div>

            <div className="rounded-2xl border border-white/10 bg-black/20 px-5 py-4 sm:text-right">
              <p className="text-xs uppercase tracking-[0.15em] text-white/30">
                Window similarity
              </p>

              <p className="mt-1 text-2xl font-semibold text-white">
                {formatScore(
                  divergence.regional_window_similarity
                )}
              </p>
            </div>
          </div>


          <div className="mt-8 grid gap-4 sm:grid-cols-2">
            <TimeWindowCard
              label="Reference window"
              window={
                divergence.reference_window
              }
            />

            <TimeWindowCard
              label="Comparison window"
              window={
                divergence.comparison_window
              }
            />
          </div>


          {divergenceDetails && (
            <div className="mt-8">
              <p className="text-xs uppercase tracking-[0.18em] text-white/30">
                Divergence components
              </p>

              <div className="mt-4 grid gap-3 sm:grid-cols-3">
                <MiniMetric
                  label="Pose"
                  value={
                    divergenceDetails
                      .divergent_region_components
                      .pose_similarity
                  }
                />

                <MiniMetric
                  label="Position"
                  value={
                    divergenceDetails
                      .divergent_region_components
                      .position_similarity
                  }
                />

                <MiniMetric
                  label="Speed"
                  value={
                    divergenceDetails
                      .divergent_region_components
                      .speed_similarity
                  }
                />
              </div>
            </div>
          )}
        </div>


        <div className="rounded-3xl border border-white/10 bg-white/[0.025] p-6 sm:p-8">
          <p className="text-xs uppercase tracking-[0.18em] text-white/30">
            Regional similarity
          </p>

          <h2 className="mt-3 text-xl font-semibold">
            Body-region breakdown
          </h2>

          <p className="mt-2 text-sm leading-6 text-white/45">
            Composite similarity combines pose,
            position and movement-speed information.
          </p>


          <div className="mt-7 space-y-5">
            {regionEntries.map(
              ({
                region,
                label,
                value,
              }) => (
                <RegionBar
                  key={region}
                  label={label}
                  value={value}
                />
              )
            )}
          </div>
        </div>
      </div>


      <div className="mt-6 rounded-3xl border border-white/10 bg-white/[0.025] p-6 sm:p-8">
        <div className="flex flex-col gap-3">
          <p className="text-xs uppercase tracking-[0.18em] text-cyan-300/70">
            Movement divergence
          </p>

          <h2 className="text-2xl font-semibold">
            Where the performances diverged
          </h2>

          <p className="max-w-3xl text-sm leading-6 text-white/45">
            The body map shows the strongest regional divergence,
            while the comparison timeline places the detected
            windows in their actual position within the comparison video.
          </p>
        </div>


        <div className="mt-8 grid gap-6 lg:grid-cols-[0.72fr_1.28fr]">
          <BodyRegionVisual
            primaryRegion={
              divergence.most_divergent_region
            }
            isolatedRegion={
              isolated.region
            }
          />

          <div className="rounded-2xl border border-white/10 bg-black/20 p-5 sm:p-6">
            <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
              <div>
                <p className="text-xs uppercase tracking-[0.16em] text-white/30">
                  Comparison timeline
                </p>

                <p className="mt-2 text-sm text-white/50">
                  Detected movement windows
                </p>
              </div>

              <p className="text-xs text-white/30">
                Video duration:{" "}
                {formatSeconds(
                  timelineDuration
                )}
              </p>
            </div>


            <div className="mt-7 space-y-7">
              <TimelineLane
                label={`Primary — ${
                  REGION_LABELS[
                    divergence
                      .most_divergent_region
                  ]
                }`}
                window={
                  divergence.comparison_window
                }
                duration={timelineDuration}
                accent="primary"
              />

              <TimelineLane
                label={`Isolated — ${
                  REGION_LABELS[
                    isolated.region
                  ]
                }`}
                window={
                  isolated.comparison_window
                }
                duration={timelineDuration}
                accent="isolated"
              />
            </div>


            <div className="mt-6 flex justify-between text-[11px] text-white/25">
              <span>0s</span>
              <span>
                {formatSeconds(
                  timelineDuration * 0.25
                )}
              </span>
              <span>
                {formatSeconds(
                  timelineDuration * 0.5
                )}
              </span>
              <span>
                {formatSeconds(
                  timelineDuration * 0.75
                )}
              </span>
              <span>
                {formatSeconds(
                  timelineDuration
                )}
              </span>
            </div>
          </div>
        </div>
      </div>


      {ml?.available && ml.top_window ? (
        <div className="mt-6 rounded-3xl border border-amber-300/15 bg-amber-300/[0.025] p-6 sm:p-8">
          <div className="flex flex-col gap-5 lg:flex-row lg:items-start lg:justify-between">
            <div>
              <p className="text-xs uppercase tracking-[0.18em] text-amber-200/60">
                Exploratory movement signal
              </p>

              <h2 className="mt-3 text-2xl font-semibold">
                Unusual movement-difference windows
              </h2>

              <p className="mt-2 max-w-3xl text-sm leading-6 text-white/45">
                MotionStage uses an unsupervised Isolation Forest as a second,
                independent signal. Higher anomaly scores indicate windows that
                differ more unusually from the movement patterns seen during
                the model's training data.
              </p>
            </div>

            <div className="rounded-2xl border border-amber-300/15 bg-black/20 px-5 py-4 lg:text-right">
              <p className="text-xs uppercase tracking-[0.15em] text-white/30">
                Highest-ranked anomaly window
              </p>

              <p className="mt-1 text-2xl font-semibold text-amber-100">
                {formatSeconds(
                  ml.top_window.start_seconds
                )}{" "}
                →{" "}
                {formatSeconds(
                  ml.top_window.end_seconds
                )}
              </p>

              <p className="mt-1 text-xs text-white/35">
                Rank #{ml.top_window.rank} ·{" "}
                {ml.top_window.percentile.toFixed(
                  1
                )}
                th percentile
              </p>
            </div>
          </div>


          <div className="mt-7 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <MLMetricCard
              label="Windows analysed"
              value={String(
                ml.windows_analyzed ?? "—"
              )}
              description="Aligned movement windows"
            />

            <MLMetricCard
              label="Flagged windows"
              value={String(
                ml.flagged_windows ?? "—"
              )}
              description="Model anomaly threshold"
            />

            <MLMetricCard
              label="Top anomaly score"
              value={ml.top_window.anomaly_score.toFixed(
                4
              )}
              description="Higher = more unusual"
            />

            <MLMetricCard
              label="Model features"
              value={String(
                ml.model_feature_count ?? "—"
              )}
              description={ml.model ?? "Isolation Forest"}
            />
          </div>


          <div className="mt-7 rounded-2xl border border-white/10 bg-black/20 p-5 sm:p-6">
            <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
              <div>
                <p className="text-xs uppercase tracking-[0.16em] text-white/30">
                  Top anomaly windows
                </p>

                <p className="mt-2 text-sm text-white/50">
                  Ranked independently from the deterministic divergence detector
                </p>
              </div>

              <p className="text-xs text-white/30">
                Comparison duration:{" "}
                {formatSeconds(
                  timelineDuration
                )}
              </p>
            </div>


            <div className="mt-6 space-y-4">
              {(ml.top_windows ?? []).map(
                (window) => (
                  <MLAnomalyWindowRow
                    key={window.window_id}
                    window={window}
                    duration={timelineDuration}
                  />
                )
              )}
            </div>
          </div>


          <div className="mt-5 flex flex-wrap gap-2">
            {ml.top_window
              .overlaps_primary_divergence && (
              <span className="rounded-full border border-cyan-300/20 bg-cyan-300/[0.06] px-3 py-1 text-xs text-cyan-200/80">
                Top anomaly window overlaps primary divergence
              </span>
            )}

            {ml.top_window
              .overlaps_isolated_divergence && (
              <span className="rounded-full border border-violet-300/20 bg-violet-300/[0.06] px-3 py-1 text-xs text-violet-200/80">
                Top anomaly window overlaps isolated-arm divergence
              </span>
            )}

            {!ml.top_window
              .overlaps_primary_divergence &&
              !ml.top_window
                .overlaps_isolated_divergence && (
                <span className="rounded-full border border-white/10 bg-white/[0.035] px-3 py-1 text-xs text-white/40">
                  Top anomaly window is outside the deterministic divergence windows
                </span>
              )}
          </div>


          <p className="mt-5 max-w-4xl text-xs leading-5 text-white/30">
            This anomaly signal is exploratory and unsupervised. It is not a
            pass/fail judgement of movement quality; it is shown alongside the
            deterministic MotionStage analysis as an additional movement signal.
          </p>
        </div>
      ) : ml && !ml.available ? (
        <div className="mt-6 rounded-2xl border border-amber-300/15 bg-amber-300/[0.035] p-5">
          <p className="text-sm font-semibold text-amber-200/90">
            Movement anomaly analysis unavailable
          </p>

          <p className="mt-2 text-sm leading-6 text-white/40">
            The main MotionStage analysis completed successfully, but the
            optional movement-anomaly analysis could not be generated for this run.
          </p>
        </div>
      ) : null}


      {pose3d?.available && pose3d.viewer_url ? (
        <div className="mt-6 rounded-3xl border border-cyan-400/15 bg-white/[0.025] p-6 sm:p-8">
          <div className="flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">
            <div>
              <p className="text-xs uppercase tracking-[0.18em] text-cyan-300/70">
                Interactive 3D analysis
              </p>

              <h2 className="mt-3 text-2xl font-semibold">
                Explore the movement divergence in 3D
              </h2>

              <p className="mt-2 max-w-3xl text-sm leading-6 text-white/45">
                The reference and comparison skeletons are synchronised by
                performance progress. MotionStage highlights the detected
                divergence region during the relevant movement windows.
              </p>
            </div>

            <a
              href={pose3d.viewer_url}
              target="_blank"
              rel="noreferrer"
              className="inline-flex shrink-0 items-center justify-center rounded-xl border border-white/10 bg-white/5 px-4 py-2.5 text-sm font-medium text-white/70 transition hover:bg-white/10 hover:text-white"
            >
              Open full-screen viewer
            </a>
          </div>

          <div className="mt-7 overflow-hidden rounded-2xl border border-white/10 bg-[#07090d]">
            <iframe
              src={pose3d.viewer_url}
              title="MotionStage interactive 3D divergence viewer"
              className="h-[760px] w-full bg-[#07090d]"
              loading="lazy"
              allowFullScreen
            />
          </div>

          <div className="mt-4 flex flex-col gap-2 text-xs text-white/30 sm:flex-row sm:items-center sm:justify-between">
            <span>
              Drag inside the viewer to rotate the 3D scene.
            </span>

            <span>
              Use Play, Pause or the timeline to inspect the motion.
            </span>
          </div>
        </div>
      ) : pose3d && !pose3d.available ? (
        <div className="mt-6 rounded-2xl border border-amber-300/15 bg-amber-300/[0.035] p-5">
          <p className="text-sm font-semibold text-amber-200/90">
            Interactive 3D view unavailable
          </p>

          <p className="mt-2 text-sm leading-6 text-white/40">
            The main MotionStage analysis completed successfully, but the
            optional 3D viewer could not be generated for this run.
          </p>
        </div>
      ) : null}


      <div className="mt-6 grid gap-6 lg:grid-cols-[1.15fr_0.85fr]">
        <div className="rounded-3xl border border-white/10 bg-white/[0.025] p-6 sm:p-8">
          <p className="text-xs uppercase tracking-[0.18em] text-white/30">
            Interpretation
          </p>

          <h2 className="mt-3 text-xl font-semibold">
            Analysis signals at a glance
          </h2>

          <div className="mt-6 space-y-4">
            <InterpretationRow
              number="01"
              title="Lowest global similarity component"
              text={`${lowestGlobalSignal.label} returned ${formatScore(
                lowestGlobalSignal.value
              )}, making it the lowest of the four global similarity components in this comparison.`}
            />

            <InterpretationRow
              number="02"
              title="Strongest regional divergence"
              text={`${REGION_LABELS[
                divergence.most_divergent_region
              ]} was selected in the comparison window ${formatSeconds(
                divergence.comparison_window.start_seconds
              )}–${formatSeconds(
                divergence.comparison_window.end_seconds
              )}, with a regional window similarity of ${formatScore(
                divergence.regional_window_similarity
              )}.`}
            />

            <InterpretationRow
              number="03"
              title="Isolated arm signal"
              text={`${REGION_LABELS[
                isolated.region
              ]} showed an isolation gap of ${isolated.isolation_gap.toFixed(
                2
              )} relative to the peer-arm and body-context comparison used by the isolated-arm detector.`}
            />
          </div>
        </div>


        <div className="rounded-3xl border border-violet-400/15 bg-violet-400/[0.035] p-6 sm:p-8">
          <p className="text-xs uppercase tracking-[0.18em] text-violet-200/50">
            Reading the visualisation
          </p>

          <div className="mt-5 space-y-5">
            <LegendItem
              accent="primary"
              title="Primary divergence"
              text="The lowest regional similarity found inside the strongest detected divergence window."
            />

            <LegendItem
              accent="isolated"
              title="Isolated arm deviation"
              text="A separate bilateral signal that checks one arm against the opposite arm and surrounding body context."
            />

            <div className="border-t border-white/10 pt-5">
              <p className="text-xs leading-5 text-white/40">
                These signals answer different questions, so the
                primary divergence region and isolated-arm region
                do not need to be the same.
              </p>
            </div>
          </div>
        </div>
      </div>


      <div className="mt-6 rounded-3xl border border-white/10 bg-white/[0.025] p-6 sm:p-8">
        <div className="flex flex-col gap-5 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <p className="text-xs uppercase tracking-[0.18em] text-white/30">
              Isolated arm deviation
            </p>

            <h2 className="mt-3 text-xl font-semibold">
              {REGION_LABELS[
                isolated.region
              ]}
            </h2>

            <p className="mt-2 max-w-2xl text-sm leading-6 text-white/45">
              MotionStage also checks whether one arm
              deviates more strongly than the opposite
              arm and surrounding body context.
            </p>
          </div>

          <div className="rounded-2xl border border-violet-400/15 bg-violet-400/5 px-6 py-4">
            <p className="text-xs uppercase tracking-[0.15em] text-violet-200/50">
              Isolation gap
            </p>

            <p className="mt-1 text-3xl font-semibold text-violet-200">
              {isolated.isolation_gap.toFixed(
                2
              )}
            </p>
          </div>
        </div>


        <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <InfoCard
            label="Comparison window"
            value={`${formatSeconds(
              isolated.comparison_window
                .start_seconds
            )} – ${formatSeconds(
              isolated.comparison_window
                .end_seconds
            )}`}
          />

          <InfoCard
            label="Region similarity"
            value={formatScore(
              isolated.isolated_region_similarity
            )}
          />

          <InfoCard
            label="Peer arm similarity"
            value={formatScore(
              isolated.peer_arm_similarity
            )}
          />

          <InfoCard
            label="Context similarity"
            value={formatScore(
              isolated.context_similarity
            )}
          />
        </div>
      </div>


      <div className="mt-6 rounded-2xl border border-white/10 bg-white/[0.02] p-5">
        <p className="text-xs leading-5 text-white/35">
          Similarity scores are comparative indices, not probabilities. Use them to
          compare movement patterns within MotionStage rather than as absolute
          accuracy or performance grades. The anomaly model is exploratory and
          has no fixed pass/fail threshold.
        </p>
      </div>
    </section>
  );
}



function BodyRegionVisual({
  primaryRegion,
  isolatedRegion,
}: {
  primaryRegion: RegionName;
  isolatedRegion: RegionName;
}) {
  function regionStroke(
    region: RegionName
  ) {
    if (region === primaryRegion) {
      return "stroke-cyan-300";
    }

    if (region === isolatedRegion) {
      return "stroke-violet-300";
    }

    return "stroke-white/20";
  }


  return (
    <div className="rounded-2xl border border-white/10 bg-black/20 p-5 sm:p-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-xs uppercase tracking-[0.16em] text-white/30">
            Body-region focus
          </p>

          <p className="mt-2 text-sm text-white/50">
            Subject-facing schematic
          </p>
        </div>

        <div className="text-right text-[11px] text-white/25">
          <p>L = subject left</p>
          <p>R = subject right</p>
        </div>
      </div>


      <div className="mt-5 flex justify-center">
        <svg
          viewBox="0 0 240 330"
          role="img"
          aria-label="Body-region divergence visualisation"
          className="h-72 w-full max-w-[260px]"
        >
          <circle
            cx="120"
            cy="42"
            r="24"
            className="fill-white/5 stroke-white/25"
            strokeWidth="5"
          />

          <line
            x1="120"
            y1="72"
            x2="120"
            y2="184"
            className={regionStroke(
              "torso"
            )}
            strokeWidth="34"
            strokeLinecap="round"
          />

          <line
            x1="105"
            y1="92"
            x2="49"
            y2="174"
            className={regionStroke(
              "left_arm"
            )}
            strokeWidth="18"
            strokeLinecap="round"
          />

          <line
            x1="135"
            y1="92"
            x2="191"
            y2="174"
            className={regionStroke(
              "right_arm"
            )}
            strokeWidth="18"
            strokeLinecap="round"
          />

          <line
            x1="107"
            y1="190"
            x2="82"
            y2="295"
            className={regionStroke(
              "lower_body"
            )}
            strokeWidth="21"
            strokeLinecap="round"
          />

          <line
            x1="133"
            y1="190"
            x2="158"
            y2="295"
            className={regionStroke(
              "lower_body"
            )}
            strokeWidth="21"
            strokeLinecap="round"
          />

          <text
            x="35"
            y="206"
            className="fill-white/30 text-[13px]"
          >
            L
          </text>

          <text
            x="197"
            y="206"
            className="fill-white/30 text-[13px]"
          >
            R
          </text>
        </svg>
      </div>


      <div className="mt-4 space-y-3 border-t border-white/10 pt-5">
        <LegendItem
          accent="primary"
          title={`${REGION_LABELS[
            primaryRegion
          ]} — primary`}
          text="Strongest regional divergence"
        />

        <LegendItem
          accent="isolated"
          title={`${REGION_LABELS[
            isolatedRegion
          ]} — isolated`}
          text="Strongest isolated-arm signal"
        />
      </div>
    </div>
  );
}



function CharacterExportCard({
  performanceName,
}: {
  performanceName: string;
}) {
  const [character, setCharacter] =
    useState<MotionStageCharacterResult | null>(null);

  const [isCheckingStatus, setIsCheckingStatus] =
    useState(true);

  const [isGenerating, setIsGenerating] =
    useState(false);

  const [elapsed, setElapsed] =
    useState(0);

  const [error, setError] =
    useState<string | null>(null);


  useEffect(() => {
    let cancelled = false;

    setCharacter(null);
    setError(null);
    setIsCheckingStatus(true);

    getCharacterStatus(performanceName)
      .then((result) => {
        if (!cancelled) {
          setCharacter(result);
        }
      })
      .catch((statusError) => {
        if (!cancelled) {
          setError(
            statusError instanceof Error
              ? statusError.message
              : "Could not check character export status."
          );
        }
      })
      .finally(() => {
        if (!cancelled) {
          setIsCheckingStatus(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [performanceName]);


  useEffect(() => {
    if (!isGenerating) {
      return;
    }

    const timer = window.setInterval(() => {
      setElapsed((seconds) => seconds + 1);
    }, 1000);

    return () => {
      window.clearInterval(timer);
    };
  }, [isGenerating]);


  const characterReady =
    character?.status === "completed" ||
    character?.status === "already_generated";

  const pose3dUnavailable =
    character?.status === "not_generated" &&
    character?.pose3d_ready === false;

  const validation =
    character?.validation ?? [];

  const worstValidation =
    validation.length > 0
      ? validation.reduce(
          (worst, current) =>
            current.max_error_degrees >
            worst.max_error_degrees
              ? current
              : worst
        )
      : null;


  async function handleGenerate(
    force: boolean
  ) {
    setElapsed(0);
    setIsGenerating(true);
    setError(null);

    try {
      const result =
        await generateCharacter(
          performanceName,
          force,
        );

      setCharacter(result);
    } catch (generationError) {
      setError(
        generationError instanceof Error
          ? generationError.message
          : "Character export failed."
      );
    } finally {
      setIsGenerating(false);
    }
  }


  return (
    <div className="mt-6 rounded-3xl border border-violet-400/20 bg-violet-400/[0.035] p-6 sm:p-8">
      <div className="flex flex-col gap-6 lg:flex-row lg:items-start lg:justify-between">
        <div className="max-w-3xl">
          <p className="text-xs uppercase tracking-[0.18em] text-violet-300/70">
            Animated character export
          </p>

          <h2 className="mt-3 text-2xl font-semibold">
            Export this performance as an animated character
          </h2>

          <p className="mt-2 text-sm leading-6 text-white/45">
            MotionStage retargets the analysed comparison performance onto the
            Y-Bot character and prepares a downloadable FBX while preserving
            the movement captured by the motion-analysis pipeline.
          </p>

          <p className="mt-3 text-xs text-white/30">
            Source performance: {performanceName}
          </p>
        </div>

        <div
          className={`inline-flex w-fit items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-medium ${
            isGenerating
              ? "border-cyan-300/20 bg-cyan-300/[0.06] text-cyan-200"
              : isCheckingStatus
                ? "border-white/10 bg-white/[0.035] text-white/45"
                : characterReady
                  ? "border-emerald-300/20 bg-emerald-300/[0.06] text-emerald-200"
                  : error
                    ? "border-red-300/20 bg-red-300/[0.05] text-red-200"
                    : "border-white/10 bg-white/[0.035] text-white/45"
          }`}
        >
          <span
            className={`h-2 w-2 rounded-full ${
              isGenerating
                ? "animate-pulse bg-cyan-300"
                : isCheckingStatus
                  ? "animate-pulse bg-white/35"
                  : characterReady
                    ? "bg-emerald-300"
                    : error
                      ? "bg-red-300"
                      : "bg-white/25"
            }`}
          />

          {isGenerating
            ? "Generating export"
            : isCheckingStatus
              ? "Checking export"
              : characterReady
                ? "FBX ready"
                : character?.status === "incomplete"
                  ? "Export incomplete"
                  : error
                    ? "Export unavailable"
                    : "Ready to generate"}
        </div>
      </div>


      {isCheckingStatus && (
        <div role="status" aria-live="polite" className="mt-6 rounded-2xl border border-white/10 bg-black/20 p-5">
          <div className="flex items-start gap-4">
            <span className="mt-1 h-4 w-4 shrink-0 animate-spin rounded-full border-2 border-white/20 border-t-white/70" />

            <div>
              <p className="text-sm font-medium text-white/80">
                Checking for an existing character export
              </p>

              <p className="mt-2 text-sm leading-6 text-white/40">
                MotionStage is checking whether this performance already has
                a completed animated-character build.
              </p>
            </div>
          </div>
        </div>
      )}


      {isGenerating && (
        <div role="status" aria-live="polite" className="mt-6 rounded-2xl border border-cyan-300/15 bg-black/20 p-5">
          <div className="flex items-start gap-4">
            <span className="mt-1 h-4 w-4 shrink-0 animate-spin rounded-full border-2 border-cyan-200/25 border-t-cyan-200" />

            <div>
              <p className="text-sm font-medium text-white/85">
                Preparing animated character
              </p>

              <p className="mt-2 text-sm leading-6 text-white/40">
                MotionStage is retargeting and validating the performance.
                Keep this page open until the export is ready.
              </p>

              <p className="mt-3 text-xs uppercase tracking-[0.15em] text-cyan-200/60">
                Elapsed time: {elapsed}s
              </p>
            </div>
          </div>
        </div>
      )}


      {pose3dUnavailable && !isCheckingStatus && (
        <div className="mt-6 rounded-2xl border border-amber-300/20 bg-amber-300/[0.04] p-5">
          <p className="text-sm font-semibold text-amber-200">
            3D motion data is not ready
          </p>

          <p className="mt-2 text-sm leading-6 text-white/45">
            Complete the 3D motion-processing step for this performance before
            generating an animated character.
          </p>
        </div>
      )}


      {error && (
        <div role="alert" className="mt-6 rounded-2xl border border-red-400/20 bg-red-400/[0.05] p-5">
          <p className="text-sm font-semibold text-red-300">
            Character export unavailable
          </p>

          <p className="mt-2 break-words text-sm leading-6 text-red-200/70">
            {error}
          </p>

          <p className="mt-3 text-xs leading-5 text-white/30">
            Your MotionStage analysis results are unaffected.
          </p>
        </div>
      )}


      {characterReady && (
        <>
          <div className="mt-6 rounded-2xl border border-emerald-300/15 bg-emerald-300/[0.035] p-4">
            <div className="flex items-center gap-3">
              <span className="h-2.5 w-2.5 rounded-full bg-emerald-300" />

              <p className="text-sm font-medium text-emerald-100/90">
                Animated character export is ready to download.
              </p>
            </div>
          </div>

          <div className="mt-4 grid gap-4 sm:grid-cols-3">
            <div className="rounded-2xl border border-white/10 bg-black/20 p-5">
              <p className="text-xs uppercase tracking-[0.15em] text-white/30">
                Animation frames
              </p>

              <p className="mt-2 text-2xl font-semibold text-white">
                {character?.frame_count ?? "—"}
              </p>
            </div>

            <div className="rounded-2xl border border-white/10 bg-black/20 p-5">
              <p className="text-xs uppercase tracking-[0.15em] text-white/30">
                Motion mode
              </p>

              <p className="mt-2 break-words text-sm font-medium text-white/75">
                {formatRootPolicy(
                  character?.root_policy
                )}
              </p>
            </div>

            <div className="rounded-2xl border border-white/10 bg-black/20 p-5">
              <p className="text-xs uppercase tracking-[0.15em] text-white/30">
                Retarget validation
              </p>

              <p className="mt-2 text-2xl font-semibold text-emerald-200">
                {worstValidation
                  ? `${worstValidation.max_error_degrees.toFixed(3)}°`
                  : "—"}
              </p>

              <p className="mt-1 text-xs text-white/30">
                Maximum sampled error
              </p>
            </div>
          </div>
        </>
      )}


      <div className="mt-6 flex flex-wrap gap-3">
        {!characterReady && !pose3dUnavailable && (
          <button
            type="button"
            disabled={
              isGenerating ||
              isCheckingStatus
            }
            onClick={() => handleGenerate(false)}
            className="rounded-xl bg-white px-5 py-3 text-sm font-semibold text-black transition hover:bg-violet-100 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {isGenerating
              ? "Generating FBX…"
              : isCheckingStatus
                ? "Checking export…"
                : "Generate animated FBX"}
          </button>
        )}

        {characterReady && (
          <>
            <a
              href={characterFbxDownloadUrl(
                performanceName
              )}
              className="inline-flex items-center justify-center rounded-xl bg-white px-5 py-3 text-sm font-semibold text-black transition hover:bg-violet-100"
            >
              Download animated FBX
            </a>

            <a
              href={characterBlendDownloadUrl(
                performanceName
              )}
              className="inline-flex items-center justify-center rounded-xl border border-white/10 bg-white/[0.04] px-5 py-3 text-sm font-medium text-white/70 transition hover:bg-white/[0.08] hover:text-white"
            >
              Download Blender project
            </a>

            <button
              type="button"
              disabled={isGenerating}
              onClick={() => handleGenerate(true)}
              className="rounded-xl border border-violet-300/15 bg-violet-300/[0.04] px-5 py-3 text-sm font-medium text-violet-100/80 transition hover:bg-violet-300/[0.08] disabled:cursor-not-allowed disabled:opacity-40"
            >
              {isGenerating
                ? "Regenerating…"
                : "Regenerate FBX"}
            </button>
          </>
        )}
      </div>


      <p className="mt-5 max-w-4xl text-xs leading-5 text-white/30">
        The exported FBX is generated from the comparison performance shown
        in this analysis. Exporting or regenerating a character does not alter
        the similarity scores or analysis results.
      </p>
    </div>
  );
}


function MLMetricCard({
  label,
  value,
  description,
}: {
  label: string;
  value: string;
  description: string;
}) {
  return (
    <div className="rounded-2xl border border-white/[0.08] bg-black/20 p-5">
      <p className="text-xs uppercase tracking-[0.15em] text-white/30">
        {label}
      </p>

      <p className="mt-2 text-2xl font-semibold text-white">
        {value}
      </p>

      <p className="mt-1 text-xs leading-5 text-white/35">
        {description}
      </p>
    </div>
  );
}



function MLAnomalyWindowRow({
  window,
  duration,
}: {
  window: NonNullable<
    NonNullable<
      MotionStageResult["ml"]
    >["top_window"]
  >;
  duration: number;
}) {
  const startPercent =
    clampPercent(
      (
        window.start_seconds /
        duration
      ) * 100
    );

  const endPercent =
    clampPercent(
      (
        window.end_seconds /
        duration
      ) * 100
    );

  const widthPercent =
    Math.max(
      1.5,
      endPercent - startPercent
    );


  return (
    <div className="rounded-xl border border-white/[0.07] bg-white/[0.02] p-4">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-3">
          <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg border border-amber-300/15 bg-amber-300/[0.06] text-[11px] font-semibold text-amber-100/80">
            {window.rank}
          </span>

          <div>
            <p className="text-sm font-medium text-white/70">
              {formatSeconds(
                window.start_seconds
              )}{" "}
              →{" "}
              {formatSeconds(
                window.end_seconds
              )}
            </p>

            <p className="mt-0.5 text-xs text-white/30">
              Score{" "}
              {window.anomaly_score.toFixed(
                4
              )}{" "}
              ·{" "}
              {window.percentile.toFixed(
                1
              )}
              th percentile
            </p>
          </div>
        </div>

        <div className="flex flex-wrap gap-2">
          {window.anomaly_flag && (
            <span className="rounded-full border border-amber-300/15 bg-amber-300/[0.05] px-2.5 py-1 text-[11px] text-amber-100/70">
              Flagged
            </span>
          )}

          {window.overlaps_primary_divergence && (
            <span className="rounded-full border border-cyan-300/15 bg-cyan-300/[0.05] px-2.5 py-1 text-[11px] text-cyan-200/70">
              Primary overlap
            </span>
          )}

          {window.overlaps_isolated_divergence && (
            <span className="rounded-full border border-violet-300/15 bg-violet-300/[0.05] px-2.5 py-1 text-[11px] text-violet-200/70">
              Isolated overlap
            </span>
          )}
        </div>
      </div>

      <div className="relative mt-4 h-2 overflow-hidden rounded-full bg-white/[0.06]">
        <div
          className="absolute top-0 h-full rounded-full bg-amber-300 shadow-[0_0_18px_rgba(252,211,77,0.22)]"
          style={{
            left: `${startPercent}%`,
            width: `${widthPercent}%`,
          }}
        />
      </div>
    </div>
  );
}



function TimelineLane({
  label,
  window,
  duration,
  accent,
}: {
  label: string;
  window: TimeWindow;
  duration: number;
  accent: "primary" | "isolated";
}) {
  const startPercent =
    clampPercent(
      (
        window.start_seconds /
        duration
      ) * 100
    );

  const endPercent =
    clampPercent(
      (
        window.end_seconds /
        duration
      ) * 100
    );

  const widthPercent =
    Math.max(
      1.5,
      endPercent - startPercent
    );

  const barClass =
    accent === "primary"
      ? "bg-cyan-400 shadow-[0_0_18px_rgba(34,211,238,0.25)]"
      : "bg-violet-400 shadow-[0_0_18px_rgba(167,139,250,0.25)]";


  return (
    <div>
      <div className="flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
        <p className="text-sm font-medium text-white/70">
          {label}
        </p>

        <p className="text-xs text-white/35">
          {formatSeconds(
            window.start_seconds
          )}{" "}
          →{" "}
          {formatSeconds(
            window.end_seconds
          )}
        </p>
      </div>

      <div className="relative mt-3 h-3 overflow-hidden rounded-full bg-white/[0.06]">
        <div
          className={`absolute top-0 h-full rounded-full ${barClass}`}
          style={{
            left: `${startPercent}%`,
            width: `${widthPercent}%`,
          }}
        />
      </div>
    </div>
  );
}



function InterpretationRow({
  number,
  title,
  text,
}: {
  number: string;
  title: string;
  text: string;
}) {
  return (
    <div className="flex gap-4 rounded-2xl border border-white/[0.08] bg-black/20 p-4">
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border border-white/10 bg-white/[0.04] text-[11px] font-medium text-white/35">
        {number}
      </div>

      <div>
        <p className="text-sm font-medium text-white/75">
          {title}
        </p>

        <p className="mt-1.5 text-sm leading-6 text-white/40">
          {text}
        </p>
      </div>
    </div>
  );
}



function LegendItem({
  accent,
  title,
  text,
}: {
  accent: "primary" | "isolated";
  title: string;
  text: string;
}) {
  const dotClass =
    accent === "primary"
      ? "bg-cyan-400"
      : "bg-violet-400";


  return (
    <div className="flex items-start gap-3">
      <span
        className={`mt-1.5 h-2.5 w-2.5 shrink-0 rounded-full ${dotClass}`}
      />

      <div>
        <p className="text-sm font-medium text-white/70">
          {title}
        </p>

        <p className="mt-1 text-xs leading-5 text-white/35">
          {text}
        </p>
      </div>
    </div>
  );
}



function clampPercent(
  value: number
) {
  return Math.max(
    0,
    Math.min(100, value)
  );
}



function ResultMetricCard({
  label,
  value,
  description,
}: {
  label: string;
  value: number;
  description: string;
}) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/[0.025] p-5">
      <p className="text-xs uppercase tracking-[0.18em] text-white/30">
        {label}
      </p>

      <p className="mt-4 text-3xl font-semibold tracking-tight text-white">
        {formatScore(value)}
      </p>

      <p className="mt-2 text-xs text-white/35">
        {description}
      </p>
    </div>
  );
}



function TimeWindowCard({
  label,
  window,
}: {
  label: string;
  window: TimeWindow;
}) {
  return (
    <div className="rounded-2xl border border-white/10 bg-black/20 p-5">
      <p className="text-xs uppercase tracking-[0.15em] text-white/30">
        {label}
      </p>

      <p className="mt-3 text-lg font-medium text-white/85">
        {formatSeconds(
          window.start_seconds
        )}
        {" → "}
        {formatSeconds(
          window.end_seconds
        )}
      </p>
    </div>
  );
}



function MiniMetric({
  label,
  value,
}: {
  label: string;
  value: number;
}) {
  return (
    <div className="rounded-xl border border-white/10 bg-black/20 p-4">
      <p className="text-xs text-white/35">
        {label}
      </p>

      <p className="mt-2 text-lg font-semibold text-white/85">
        {formatScore(value)}
      </p>
    </div>
  );
}



function RegionBar({
  label,
  value,
}: {
  label: string;
  value: number;
}) {
  const width =
    Math.max(
      0,
      Math.min(100, value)
    );

  return (
    <div>
      <div className="flex items-center justify-between gap-4 text-sm">
        <span className="text-white/65">
          {label}
        </span>

        <span className="font-medium text-white/85">
          {formatScore(value)}
        </span>
      </div>

      <div className="mt-2 h-2 overflow-hidden rounded-full bg-white/5">
        <div
          className="h-full rounded-full bg-gradient-to-r from-cyan-400 to-violet-400"
          style={{
            width: `${width}%`,
          }}
        />
      </div>
    </div>
  );
}



function InfoCard({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="rounded-2xl border border-white/10 bg-black/20 p-5">
      <p className="text-xs uppercase tracking-[0.15em] text-white/30">
        {label}
      </p>

      <p className="mt-3 text-lg font-medium text-white/80">
        {value}
      </p>
    </div>
  );
}



function UploadCard({
  number,
  title,
  description,
  selectedVideo,
  setSelectedVideo,
  locked,
  onSelectionChange,
}: {
  number: string;
  title: string;
  description: string;

  selectedVideo:
    | SelectedVideo
    | null;

  setSelectedVideo:
    Dispatch<
      SetStateAction<
        SelectedVideo | null
      >
    >;

  locked: boolean;

  onSelectionChange: () => void;
}) {
  const inputRef =
    useRef<HTMLInputElement | null>(null);

  const [error, setError] =
    useState<string | null>(null);


  function handleFile(
    event: ChangeEvent<HTMLInputElement>
  ) {
    if (locked) {
      return;
    }

    const file =
      event.target.files?.[0];

    if (!file) {
      return;
    }

    setError(null);

    const extension =
      file.name
        .split(".")
        .pop()
        ?.toLowerCase();

    const allowedExtensions = [
      "mov",
      "mp4",
      "m4v",
    ];

    if (
      !extension ||
      !allowedExtensions.includes(
        extension
      )
    ) {
      setError(
        "Please choose a MOV, MP4 or M4V video."
      );

      event.target.value = "";

      return;
    }

    if (selectedVideo) {
      URL.revokeObjectURL(
        selectedVideo.url
      );
    }

    const url =
      URL.createObjectURL(file);

    setSelectedVideo({
      file,
      url,
    });

    onSelectionChange();
  }


  function removeVideo() {
    if (locked) {
      return;
    }

    if (selectedVideo) {
      URL.revokeObjectURL(
        selectedVideo.url
      );
    }

    setSelectedVideo(null);
    setError(null);
    onSelectionChange();

    if (inputRef.current) {
      inputRef.current.value = "";
    }
  }


  return (
    <div className="rounded-3xl border border-white/10 bg-white/[0.035] p-6 shadow-2xl shadow-black/20">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-xs font-medium tracking-[0.2em] text-white/30">
            PERFORMANCE {number}
          </p>

          <h2 className="mt-3 text-xl font-semibold">
            {title}
          </h2>

          <p className="mt-2 max-w-md text-sm leading-6 text-white/45">
            {description}
          </p>
        </div>

        <div className="rounded-full border border-white/10 px-3 py-1 text-xs text-white/40">
          Required
        </div>
      </div>


      {!selectedVideo ? (
        <div className="mt-8 flex min-h-56 flex-col items-center justify-center rounded-2xl border border-dashed border-white/15 bg-black/20 px-6 text-center">
          <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-2xl border border-white/10 bg-white/5">
            <svg
              width="22"
              height="22"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.6"
              className="text-white/60"
            >
              <path d="M12 16V4" />
              <path d="M7 9l5-5 5 5" />
              <path d="M5 20h14" />
            </svg>
          </div>

          <p className="font-medium text-white/80">
            Upload performance video
          </p>

          <p className="mt-2 text-sm text-white/35">
            MOV, MP4 or M4V
          </p>

          <input
            ref={inputRef}
            type="file"
            accept=".mov,.mp4,.m4v,video/quicktime,video/mp4"
            onChange={handleFile}
            disabled={locked}
            className="hidden"
          />

          <button
            type="button"
            disabled={locked}
            onClick={() =>
              inputRef.current?.click()
            }
            className="mt-5 rounded-lg border border-white/10 bg-white/5 px-4 py-2 text-sm font-medium text-white/70 transition hover:bg-white/10 hover:text-white disabled:cursor-not-allowed disabled:opacity-30"
          >
            Choose video
          </button>

          {error && (
            <p className="mt-4 text-sm text-red-400">
              {error}
            </p>
          )}
        </div>
      ) : (
        <div className="mt-8 overflow-hidden rounded-2xl border border-white/10 bg-black">
          <div className="relative aspect-video bg-black">
            <video
              src={selectedVideo.url}
              controls
              playsInline
              className="h-full w-full object-contain"
            />
          </div>

          <div className="flex items-center justify-between gap-4 border-t border-white/10 bg-white/[0.025] p-4">
            <div className="min-w-0">
              <p className="truncate text-sm font-medium text-white/80">
                {selectedVideo.file.name}
              </p>

              <p className="mt-1 text-xs text-white/35">
                {formatFileSize(
                  selectedVideo.file.size
                )}
              </p>
            </div>

            <div className="flex shrink-0 gap-2">
              <button
                type="button"
                disabled={locked}
                onClick={() =>
                  inputRef.current?.click()
                }
                className="rounded-lg border border-white/10 px-3 py-2 text-xs font-medium text-white/60 transition hover:bg-white/5 hover:text-white disabled:cursor-not-allowed disabled:opacity-30"
              >
                Replace
              </button>

              <button
                type="button"
                disabled={locked}
                onClick={removeVideo}
                className="rounded-lg border border-red-400/15 bg-red-400/5 px-3 py-2 text-xs font-medium text-red-300 transition hover:bg-red-400/10 disabled:cursor-not-allowed disabled:opacity-30"
              >
                Remove
              </button>
            </div>
          </div>

          <input
            ref={inputRef}
            type="file"
            accept=".mov,.mp4,.m4v,video/quicktime,video/mp4"
            onChange={handleFile}
            disabled={locked}
            className="hidden"
          />
        </div>
      )}
    </div>
  );
}



function MetricPreview({
  label,
  value,
}: {
  label: string;
  value: string;
}) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/[0.025] p-5">
      <p className="text-xs uppercase tracking-[0.18em] text-white/30">
        {label}
      </p>

      <p className="mt-3 text-sm font-medium text-white/70">
        {value}
      </p>
    </div>
  );
}



function formatFileSize(
  bytes: number
) {
  if (bytes < 1024 * 1024) {
    return `${(
      bytes / 1024
    ).toFixed(1)} KB`;
  }

  return `${(
    bytes /
    (1024 * 1024)
  ).toFixed(1)} MB`;
}



function formatRootPolicy(
  value?: string | null
) {
  if (!value) {
    return "—";
  }

  if (
    value === "in_place_pelvis_relative" ||
    value === "pelvis-relative / in-place"
  ) {
    return "In-place · pelvis-relative";
  }

  return value
    .replaceAll("_", " ")
    .replace(/\b\w/g, (character) =>
      character.toUpperCase()
    );
}


function formatScore(
  value: number
) {
  return value.toFixed(1);
}


function formatSeconds(
  value: number
) {
  return `${value.toFixed(2)}s`;
}
