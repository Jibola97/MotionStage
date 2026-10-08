export type MotionStagePose3D = {
  available: boolean;
  reference_pose3d?: string;
  comparison_pose3d?: string;
  viewer_path?: string;
  viewer_url: string | null;
  error?: string;
};

export type MotionStageMLWindow = {
  window_id: string;
  start_seconds: number;
  end_seconds: number;
  anomaly_score: number;
  percentile: number;
  rank: number;
  anomaly_flag: boolean;
  overlaps_primary_divergence: boolean;
  overlaps_isolated_divergence: boolean;
};

export type MotionStageML = {
  available: boolean;
  model?: string;
  model_stage?: string;
  reference?: string;
  comparison?: string;
  windows_analyzed?: number;
  window_seconds?: number;
  stride_seconds?: number;
  model_feature_count?: number;
  flagged_windows?: number;
  top_window?: MotionStageMLWindow;
  top_windows?: MotionStageMLWindow[];
  scores_path?: string;
  summary_path?: string;
  interpretation?: string;
  error?: string;
};


export type MotionStageResult = {
  status: string;

  reference: {
    performance_name: string;
    original_filename: string;
  };

  comparison: {
    performance_name: string;
    original_filename: string;
  };

  result: Record<string, unknown>;
  ml?: MotionStageML;
  pose3d?: MotionStagePose3D;
};

const API_BASE_URL =
  process.env.NEXT_PUBLIC_MOTIONSTAGE_API_URL?.replace(
    /\/$/,
    "",
  ) || "http://127.0.0.1:8000";

export async function analysePerformances(
  referenceVideo: File,
  comparisonVideo: File,
): Promise<MotionStageResult> {
  const formData = new FormData();

  formData.append(
    "reference_video",
    referenceVideo,
  );

  formData.append(
    "comparison_video",
    comparisonVideo,
  );

  const response = await fetch(
    `${API_BASE_URL}/analyse`,
    {
      method: "POST",
      body: formData,
    },
  );

  if (!response.ok) {
    let message =
      `Analysis failed with HTTP ${response.status}.`;

    try {
      const errorBody = await response.json();

      if (errorBody?.detail) {
        message =
          typeof errorBody.detail === "string"
            ? errorBody.detail
            : JSON.stringify(errorBody.detail);
      }
    } catch {
      // Keep fallback message.
    }

    throw new Error(message);
  }

  return response.json();
}

export type MotionStageCharacterValidation = {
  frame: number;
  mean_error_degrees: number;
  max_error_degrees: number;
};

export type MotionStageCharacterResult = {
  status: string;
  performance_name: string;
  pose3d_ready?: boolean;
  frame_count?: number | null;
  root_policy?: string | null;
  validation?: MotionStageCharacterValidation[];
  fbx_url?: string | null;
  blend_url?: string | null;
  manifest_url?: string | null;
  outputs?: {
    fbx?: string | null;
    blend?: string | null;
    manifest?: string | null;
  };
};


async function readApiError(
  response: Response,
  fallback: string,
): Promise<string> {
  try {
    const errorBody = await response.json();
    const detail = errorBody?.detail;

    if (typeof detail === "string") {
      return detail;
    }

    if (
      detail &&
      typeof detail === "object"
    ) {
      const message =
        typeof detail.message === "string"
          ? detail.message
          : fallback;

      const nextStep =
        typeof detail.next_step === "string"
          ? detail.next_step
          : null;

      return nextStep
        ? `${message} ${nextStep}`
        : message;
    }
  } catch {
    // Keep fallback message.
  }

  return fallback;
}


export async function getCharacterStatus(
  performanceName: string,
): Promise<MotionStageCharacterResult> {
  const response = await fetch(
    `${API_BASE_URL}/performances/${encodeURIComponent(
      performanceName,
    )}/character`,
  );

  if (!response.ok) {
    throw new Error(
      await readApiError(
        response,
        `Character status failed with HTTP ${response.status}.`,
      ),
    );
  }

  return response.json();
}


export async function generateCharacter(
  performanceName: string,
  force = false,
): Promise<MotionStageCharacterResult> {
  const response = await fetch(
    `${API_BASE_URL}/performances/${encodeURIComponent(
      performanceName,
    )}/character/generate?force=${force ? "true" : "false"}`,
    {
      method: "POST",
    },
  );

  if (!response.ok) {
    throw new Error(
      await readApiError(
        response,
        `Character generation failed with HTTP ${response.status}.`,
      ),
    );
  }

  return response.json();
}


export function characterFbxDownloadUrl(
  performanceName: string,
) {
  return `${API_BASE_URL}/performances/${encodeURIComponent(
    performanceName,
  )}/character/fbx`;
}


export function characterBlendDownloadUrl(
  performanceName: string,
) {
  return `${API_BASE_URL}/performances/${encodeURIComponent(
    performanceName,
  )}/character/blend`;
}

