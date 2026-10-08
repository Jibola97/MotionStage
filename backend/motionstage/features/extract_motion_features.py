from pathlib import Path

import numpy as np
import pandas as pd


BODY_REGIONS = {
    "left_arm": [
        "left_shoulder",
        "left_elbow",
        "left_wrist",
    ],
    "right_arm": [
        "right_shoulder",
        "right_elbow",
        "right_wrist",
    ],
    "torso": [
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
    ],
    "left_leg": [
        "left_hip",
        "left_knee",
        "left_ankle",
    ],
    "right_leg": [
        "right_hip",
        "right_knee",
        "right_ankle",
    ],
}

BODY_JOINTS = sorted(
    {
        joint
        for joints in BODY_REGIONS.values()
        for joint in joints
    }
)


def calculate_angle(
    point_a: np.ndarray,
    vertex: np.ndarray,
    point_c: np.ndarray,
) -> float:
    """
    Calculate the angle A-B-C in degrees,
    where B is the vertex.
    """
    vector_a = point_a - vertex
    vector_c = point_c - vertex

    denominator = (
        np.linalg.norm(vector_a)
        * np.linalg.norm(vector_c)
    )

    if denominator == 0:
        return np.nan

    cosine_angle = np.dot(
        vector_a,
        vector_c,
    ) / denominator

    cosine_angle = np.clip(
        cosine_angle,
        -1.0,
        1.0,
    )

    return float(
        np.degrees(
            np.arccos(cosine_angle)
        )
    )


def get_point(
    frame_df: pd.DataFrame,
    joint_name: str,
) -> np.ndarray:
    joint = frame_df[
        frame_df["joint_name"] == joint_name
    ]

    if joint.empty:
        return np.array([np.nan, np.nan])

    row = joint.iloc[0]

    return np.array(
        [
            row["x_norm"],
            row["y_norm"],
        ],
        dtype=float,
    )


def calculate_joint_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add velocity, speed and acceleration
    to each joint trajectory.
    """
    joint_outputs = []

    for joint_name, joint_df in df.groupby(
        "joint_name"
    ):
        joint_df = (
            joint_df
            .sort_values("frame")
            .copy()
        )

        time = joint_df[
            "timestamp_seconds"
        ].to_numpy(dtype=float)

        x = joint_df[
            "x_norm"
        ].to_numpy(dtype=float)

        y = joint_df[
            "y_norm"
        ].to_numpy(dtype=float)

        if len(joint_df) < 3:
            joint_df["vx_norm_s"] = np.nan
            joint_df["vy_norm_s"] = np.nan
            joint_df["speed_norm_s"] = np.nan
            joint_df["ax_norm_s2"] = np.nan
            joint_df["ay_norm_s2"] = np.nan
            joint_df["acceleration_norm_s2"] = np.nan

            joint_outputs.append(joint_df)
            continue

        vx = np.gradient(x, time)
        vy = np.gradient(y, time)

        speed = np.hypot(
            vx,
            vy,
        )

        ax = np.gradient(
            vx,
            time,
        )

        ay = np.gradient(
            vy,
            time,
        )

        acceleration = np.hypot(
            ax,
            ay,
        )

        joint_df["vx_norm_s"] = vx
        joint_df["vy_norm_s"] = vy
        joint_df["speed_norm_s"] = speed

        joint_df["ax_norm_s2"] = ax
        joint_df["ay_norm_s2"] = ay
        joint_df[
            "acceleration_norm_s2"
        ] = acceleration

        joint_outputs.append(joint_df)

    return (
        pd.concat(joint_outputs)
        .sort_values(
            ["frame", "joint_id"]
        )
        .reset_index(drop=True)
    )


def calculate_frame_features(
    joint_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Produce one feature row per video frame.
    """
    frame_rows = []

    for frame_number, frame_df in joint_df.groupby(
        "frame"
    ):
        timestamp = float(
            frame_df[
                "timestamp_seconds"
            ].iloc[0]
        )

        left_shoulder = get_point(
            frame_df,
            "left_shoulder",
        )
        right_shoulder = get_point(
            frame_df,
            "right_shoulder",
        )

        left_elbow = get_point(
            frame_df,
            "left_elbow",
        )
        right_elbow = get_point(
            frame_df,
            "right_elbow",
        )

        left_wrist = get_point(
            frame_df,
            "left_wrist",
        )
        right_wrist = get_point(
            frame_df,
            "right_wrist",
        )

        left_hip = get_point(
            frame_df,
            "left_hip",
        )
        right_hip = get_point(
            frame_df,
            "right_hip",
        )

        left_knee = get_point(
            frame_df,
            "left_knee",
        )
        right_knee = get_point(
            frame_df,
            "right_knee",
        )

        left_ankle = get_point(
            frame_df,
            "left_ankle",
        )
        right_ankle = get_point(
            frame_df,
            "right_ankle",
        )

        # -----------------------------------------
        # Joint angles
        # -----------------------------------------
        left_elbow_angle = calculate_angle(
            left_shoulder,
            left_elbow,
            left_wrist,
        )

        right_elbow_angle = calculate_angle(
            right_shoulder,
            right_elbow,
            right_wrist,
        )

        left_shoulder_angle = calculate_angle(
            left_elbow,
            left_shoulder,
            left_hip,
        )

        right_shoulder_angle = calculate_angle(
            right_elbow,
            right_shoulder,
            right_hip,
        )

        left_knee_angle = calculate_angle(
            left_hip,
            left_knee,
            left_ankle,
        )

        right_knee_angle = calculate_angle(
            right_hip,
            right_knee,
            right_ankle,
        )

        # -----------------------------------------
        # Torso orientation
        # -----------------------------------------
        shoulder_center = (
            left_shoulder
            + right_shoulder
        ) / 2.0

        hip_center = (
            left_hip
            + right_hip
        ) / 2.0

        torso_vector = (
            shoulder_center
            - hip_center
        )

        # Signed lean angle relative to vertical.
        torso_lean_deg = float(
            np.degrees(
                np.arctan2(
                    torso_vector[0],
                    torso_vector[1],
                )
            )
        )

        row = {
            "frame": int(frame_number),
            "timestamp_seconds": timestamp,
            "left_elbow_angle_deg":
                left_elbow_angle,
            "right_elbow_angle_deg":
                right_elbow_angle,
            "left_shoulder_angle_deg":
                left_shoulder_angle,
            "right_shoulder_angle_deg":
                right_shoulder_angle,
            "left_knee_angle_deg":
                left_knee_angle,
            "right_knee_angle_deg":
                right_knee_angle,
            "torso_lean_deg":
                torso_lean_deg,
        }

        # -----------------------------------------
        # Regional motion summaries
        # -----------------------------------------
        for region_name, joints in BODY_REGIONS.items():
            region_rows = frame_df[
                frame_df[
                    "joint_name"
                ].isin(joints)
            ]

            row[
                f"{region_name}_mean_speed"
            ] = float(
                region_rows[
                    "speed_norm_s"
                ].mean()
            )

        body_rows = frame_df[
            frame_df[
                "joint_name"
            ].isin(BODY_JOINTS)
        ]

        row["whole_body_mean_speed"] = float(
            body_rows[
                "speed_norm_s"
            ].mean()
        )

        # Motion intensity proxy.
        # This is NOT physical energy.
        row[
            "movement_intensity_proxy"
        ] = float(
            np.mean(
                np.square(
                    body_rows[
                        "speed_norm_s"
                    ].to_numpy()
                )
            )
        )

        frame_rows.append(row)

    return pd.DataFrame(frame_rows)


def extract_motion_features(
    input_csv: str,
    joint_output_csv: str,
    frame_output_csv: str,
) -> None:
    input_path = Path(input_csv)
    joint_output_path = Path(
        joint_output_csv
    )
    frame_output_path = Path(
        frame_output_csv
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {input_path}"
        )

    df = pd.read_csv(input_path)

    required_columns = {
        "frame",
        "timestamp_seconds",
        "joint_id",
        "joint_name",
        "x_norm",
        "y_norm",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing)}"
        )

    joint_features = (
        calculate_joint_features(df)
    )

    frame_features = (
        calculate_frame_features(
            joint_features
        )
    )

    joint_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joint_features.to_csv(
        joint_output_path,
        index=False,
    )

    frame_features.to_csv(
        frame_output_path,
        index=False,
    )

    print(
        "MotionStage Motion Feature Extraction"
    )
    print(
        "-------------------------------------"
    )

    print(
        f"Input: {input_path}"
    )

    print(
        f"Joint features: "
        f"{joint_output_path}"
    )

    print(
        f"Frame features: "
        f"{frame_output_path}"
    )

    print(
        f"Joint rows: "
        f"{len(joint_features)}"
    )

    print(
        f"Frames: "
        f"{len(frame_features)}"
    )

    print()
    print("Movement summary")
    print("----------------")

    speed_summary = (
        joint_features
        .groupby("joint_name")[
            "speed_norm_s"
        ]
        .agg(
            ["mean", "median", "max"]
        )
        .sort_values(
            "mean",
            ascending=False,
        )
        .round(3)
    )

    print(
        speed_summary.head(8).to_string()
    )

    print()
    print("Angle ranges")
    print("------------")

    angle_columns = [
        column
        for column in frame_features.columns
        if column.endswith(
            "_angle_deg"
        )
    ]

    for column in angle_columns:
        minimum = (
            frame_features[column].min()
        )

        maximum = (
            frame_features[column].max()
        )

        print(
            f"{column}: "
            f"{minimum:.1f}° "
            f"to {maximum:.1f}°"
        )


if __name__ == "__main__":
    extract_motion_features(
        input_csv=(
            "data/processed/"
            "test_performance_01_keypoints_normalised.csv"
        ),
        joint_output_csv=(
            "data/processed/"
            "test_performance_01_joint_features.csv"
        ),
        frame_output_csv=(
            "data/processed/"
            "test_performance_01_frame_features.csv"
        ),
    )
