import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


CONNECTIONS = [
    # shoulders / torso
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

    # right leg
    (24, 26),
    (26, 28),
    (28, 30),
    (30, 32),

    # head
    (0, 7),
    (0, 8),
    (7, 11),
    (8, 12),
]


def prepare_frame(frame_df):
    frame_df = (
        frame_df
        .sort_values("joint_id")
        .copy()
    )

    points = {
        int(row.joint_id): (
            float(row.x_world),
            float(row.z_world),
            float(-row.y_world),
        )
        for row in frame_df.itertuples()
    }

    line_x = []
    line_y = []
    line_z = []

    for a, b in CONNECTIONS:
        if a not in points or b not in points:
            continue

        ax, ay, az = points[a]
        bx, by, bz = points[b]

        line_x.extend([ax, bx, None])
        line_y.extend([ay, by, None])
        line_z.extend([az, bz, None])

    timestamp = float(
        frame_df["timestamp_seconds"].iloc[0]
    )

    skeleton = go.Scatter3d(
        x=line_x,
        y=line_y,
        z=line_z,
        mode="lines",
        line=dict(
            width=6,
            color="#45d7f0",
        ),
        hoverinfo="skip",
        name="Skeleton",
    )

    joints = go.Scatter3d(
        x=frame_df["x_world"],
        y=frame_df["z_world"],
        z=-frame_df["y_world"],
        mode="markers",
        marker=dict(
            size=5,
            color="#ffffff",
            line=dict(
                width=1,
                color="#45d7f0",
            ),
        ),
        customdata=list(
            zip(
                frame_df["joint_name"],
                frame_df["joint_id"],
            )
        ),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Joint ID: %{customdata[1]}<br>"
            "X: %{x:.3f}<br>"
            "Depth: %{y:.3f}<br>"
            "Height: %{z:.3f}<br>"
            "<extra></extra>"
        ),
        name="Landmarks",
    )

    return skeleton, joints, timestamp


def main():
    parser = argparse.ArgumentParser(
        description="Interactive MotionStage 3D pose viewer."
    )

    parser.add_argument(
        "csv_path",
        help="Path to pose3d_world.csv",
    )

    parser.add_argument(
        "--output",
        default="pose3d_viewer.html",
        help="Output HTML path",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    output_path = Path(args.output)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found: {csv_path}"
        )

    df = pd.read_csv(csv_path)

    required = {
        "frame",
        "timestamp_seconds",
        "joint_id",
        "joint_name",
        "x_world",
        "y_world",
        "z_world",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    frame_numbers = sorted(
        df["frame"].unique().astype(int)
    )

    if not frame_numbers:
        raise ValueError("No frames found.")

    # Global display ranges so the view does not
    # resize from frame to frame.
    display_x = df["x_world"]
    display_y = df["z_world"]
    display_z = -df["y_world"]

    def padded_range(series, padding=0.15):
        minimum = float(series.min())
        maximum = float(series.max())
        span = maximum - minimum

        if span <= 0:
            span = 1.0

        return [
            minimum - span * padding,
            maximum + span * padding,
        ]

    x_range = padded_range(display_x)
    y_range = padded_range(display_y)
    z_range = padded_range(display_z)

    first_number = frame_numbers[0]

    first_df = df[
        df["frame"] == first_number
    ]

    first_skeleton, first_joints, first_time = (
        prepare_frame(first_df)
    )

    animation_frames = []

    for frame_number in frame_numbers:
        frame_df = df[
            df["frame"] == frame_number
        ]

        skeleton, joints, timestamp = (
            prepare_frame(frame_df)
        )

        animation_frames.append(
            go.Frame(
                name=str(frame_number),
                data=[
                    skeleton,
                    joints,
                ],
                layout=go.Layout(
                    title=dict(
                        text=(
                            "MotionStage Estimated 3D Pose"
                            f"<br><sup>"
                            f"Frame {frame_number} · "
                            f"{timestamp:.2f} seconds"
                            "</sup>"
                        )
                    )
                ),
            )
        )

    slider_steps = []

    for frame_number in frame_numbers:
        slider_steps.append(
            {
                "method": "animate",
                "args": [
                    [str(frame_number)],
                    {
                        "mode": "immediate",
                        "frame": {
                            "duration": 0,
                            "redraw": True,
                        },
                        "transition": {
                            "duration": 0,
                        },
                    },
                ],
                "label": "",
            }
        )

    fig = go.Figure(
        data=[
            first_skeleton,
            first_joints,
        ],
        frames=animation_frames,
    )

    fig.update_layout(
        title=dict(
            text=(
                "MotionStage Estimated 3D Pose"
                f"<br><sup>"
                f"Frame {first_number} · "
                f"{first_time:.2f} seconds"
                "</sup>"
            ),
            x=0.5,
        ),

        paper_bgcolor="#07090d",
        plot_bgcolor="#07090d",
        font=dict(
            color="#e5e7eb",
        ),

        scene=dict(
            bgcolor="#07090d",

            xaxis=dict(
                title="Left / right",
                range=x_range,
                gridcolor="#242830",
                zerolinecolor="#3a3f49",
            ),

            yaxis=dict(
                title="Depth",
                range=y_range,
                gridcolor="#242830",
                zerolinecolor="#3a3f49",
            ),

            zaxis=dict(
                title="Vertical",
                range=z_range,
                gridcolor="#242830",
                zerolinecolor="#3a3f49",
            ),

            aspectmode="data",

            camera=dict(
                eye=dict(
                    x=0.0,
                    y=-2.4,
                    z=0.25,
                )
            ),

            uirevision="motionstage-pose3d",
        ),

        margin=dict(
            l=0,
            r=0,
            t=90,
            b=80,
        ),

        showlegend=False,

        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.5,
                "xanchor": "center",
                "y": -0.05,
                "yanchor": "top",

                "buttons": [
                    {
                        "label": "▶ Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "fromcurrent": True,
                                "frame": {
                                    "duration": 33,
                                    "redraw": True,
                                },
                                "transition": {
                                    "duration": 0,
                                },
                            },
                        ],
                    },
                    {
                        "label": "⏸ Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "mode": "immediate",
                                "frame": {
                                    "duration": 0,
                                    "redraw": False,
                                },
                                "transition": {
                                    "duration": 0,
                                },
                            },
                        ],
                    },
                ],
            }
        ],

        sliders=[
            {
                "active": 0,
                "x": 0.08,
                "len": 0.84,
                "y": 0.02,
                "pad": {
                    "t": 30,
                    "b": 10,
                },
                "currentvalue": {
                    "prefix": "Timeline ",
                    "visible": True,
                    "font": {
                        "size": 12,
                    },
                },
                "steps": slider_steps,
            }
        ],
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.write_html(
        output_path,
        include_plotlyjs=True,
        full_html=True,
        auto_open=False,
    )

    print()
    print("=" * 60)
    print("MotionStage Interactive 3D Viewer")
    print("=" * 60)
    print(f"Source: {csv_path}")
    print(f"Frames: {len(frame_numbers)}")
    print(
        "Duration: "
        f"{df['timestamp_seconds'].max():.2f}s"
    )
    print(f"Output: {output_path}")
    print()
    print("Viewer controls:")
    print("  Drag        = rotate")
    print("  Scroll      = zoom")
    print("  Slider      = scrub through frames")
    print("  Play/Pause  = animate motion")


if __name__ == "__main__":
    main()
