import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CONNECTIONS = [
    # torso
    (11, 12),
    (11, 23),
    (12, 24),
    (23, 24),

    # left arm
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),

    # right arm
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),

    # left leg
    (23, 25),
    (25, 27),
    (27, 29),
    (29, 31),
    (27, 31),

    # right leg
    (24, 26),
    (26, 28),
    (28, 30),
    (30, 32),
    (28, 32),

    # head / shoulders
    (0, 7),
    (0, 8),
    (7, 11),
    (8, 12),
]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("csv_path")
    parser.add_argument(
        "--frame",
        type=int,
        default=150,
    )
    parser.add_argument(
        "--output",
        default="pose3d_validation.png",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv_path)

    df = pd.read_csv(csv_path)

    frame_df = (
        df[df["frame"] == args.frame]
        .sort_values("joint_id")
        .copy()
    )

    if frame_df.empty:
        raise ValueError(
            f"No data found for frame {args.frame}"
        )

    if len(frame_df) != 33:
        raise ValueError(
            f"Expected 33 landmarks, found {len(frame_df)}"
        )

    points = {
        int(row.joint_id): (
            float(row.x_world),
            float(row.y_world),
            float(row.z_world),
        )
        for row in frame_df.itertuples()
    }

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    # Reorient MediaPipe coordinates for a more intuitive
    # human-view display:
    # horizontal = X
    # depth      = Z
    # vertical   = -Y
    for a, b in CONNECTIONS:
        if a not in points or b not in points:
            continue

        xa, ya, za = points[a]
        xb, yb, zb = points[b]

        ax.plot(
            [xa, xb],
            [za, zb],
            [-ya, -yb],
            linewidth=2,
        )

    xs = frame_df["x_world"]
    ys = frame_df["z_world"]
    zs = -frame_df["y_world"]

    ax.scatter(
        xs,
        ys,
        zs,
        s=25,
    )

    ax.set_xlabel("X — left / right")
    ax.set_ylabel("Z — depth")
    ax.set_zlabel("Y — vertical")

    ax.set_title(
        f"MotionStage Estimated 3D Pose — Frame {args.frame}"
    )

    # Keep axes reasonably proportional.
    all_values = pd.concat(
        [
            xs,
            ys,
            zs,
        ]
    )

    span = (
        all_values.max()
        - all_values.min()
    )

    span = max(float(span), 1.0)

    x_mid = (
        xs.max()
        + xs.min()
    ) / 2

    y_mid = (
        ys.max()
        + ys.min()
    ) / 2

    z_mid = (
        zs.max()
        + zs.min()
    ) / 2

    half = span / 2

    ax.set_xlim(
        x_mid - half,
        x_mid + half,
    )

    ax.set_ylim(
        y_mid - half,
        y_mid + half,
    )

    ax.set_zlim(
        z_mid - half,
        z_mid + half,
    )

    ax.view_init(
        elev=12,
        azim=-90,
    )

    plt.tight_layout()

    output = Path(args.output)

    fig.savefig(
        output,
        dpi=180,
    )

    print()
    print("=" * 60)
    print("MotionStage 3D Pose Validation")
    print("=" * 60)
    print(f"CSV: {csv_path}")
    print(f"Frame: {args.frame}")
    print(f"Landmarks: {len(frame_df)}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
