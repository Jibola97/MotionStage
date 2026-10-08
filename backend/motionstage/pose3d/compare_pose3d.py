import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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

    # right leg
    (24, 26),
    (26, 28),
    (28, 30),
    (30, 32),

    # head / shoulders
    (0, 7),
    (0, 8),
    (7, 11),
    (8, 12),
]


def load_pose(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Pose CSV not found: {path}")

    df = pd.read_csv(path)

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
            f"{path} is missing columns: {sorted(missing)}"
        )

    return df


def frame_numbers(df):
    return sorted(df["frame"].unique().astype(int))


def frame_data(df, frame_number, colour):
    current = (
        df[df["frame"] == frame_number]
        .sort_values("joint_id")
        .copy()
    )

    if current.empty:
        raise ValueError(f"No data for frame {frame_number}")

    points = {
        int(row.joint_id): (
            float(row.x_world),
            float(row.z_world),
            float(-row.y_world),
        )
        for row in current.itertuples()
    }

    line_x = []
    line_y = []
    line_z = []

    for start, end in CONNECTIONS:
        if start not in points or end not in points:
            continue

        x1, y1, z1 = points[start]
        x2, y2, z2 = points[end]

        line_x.extend([x1, x2, None])
        line_y.extend([y1, y2, None])
        line_z.extend([z1, z2, None])

    skeleton = go.Scatter3d(
        x=line_x,
        y=line_y,
        z=line_z,
        mode="lines",
        line=dict(
            width=7,
            color=colour,
        ),
        hoverinfo="skip",
        showlegend=False,
    )

    joints = go.Scatter3d(
        x=current["x_world"],
        y=current["z_world"],
        z=-current["y_world"],
        mode="markers",
        marker=dict(
            size=5,
            color="#ffffff",
            line=dict(
                width=1,
                color=colour,
            ),
        ),
        customdata=list(
            zip(
                current["joint_name"],
                current["joint_id"],
            )
        ),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Joint ID: %{customdata[1]}<br>"
            "X: %{x:.3f}<br>"
            "Depth: %{y:.3f}<br>"
            "Vertical: %{z:.3f}<br>"
            "<extra></extra>"
        ),
        showlegend=False,
    )

    timestamp = float(
        current["timestamp_seconds"].iloc[0]
    )

    return skeleton, joints, timestamp


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


def main():
    parser = argparse.ArgumentParser(
        description=(
            "MotionStage synchronized reference/comparison "
            "3D pose viewer."
        )
    )

    parser.add_argument("reference_csv")
    parser.add_argument("comparison_csv")

    parser.add_argument(
        "--output",
        default="pose3d_comparison.html",
    )

    args = parser.parse_args()

    reference = load_pose(args.reference_csv)
    comparison = load_pose(args.comparison_csv)

    ref_frames = frame_numbers(reference)
    comp_frames = frame_numbers(comparison)

    if not ref_frames or not comp_frames:
        raise ValueError("Both inputs must contain pose frames.")

    # Use the longer sequence as the number of animation steps.
    # Each sequence is mapped by relative progress.
    steps = max(
        len(ref_frames),
        len(comp_frames),
    )

    combined_x = pd.concat(
        [
            reference["x_world"],
            comparison["x_world"],
        ],
        ignore_index=True,
    )

    combined_y = pd.concat(
        [
            reference["z_world"],
            comparison["z_world"],
        ],
        ignore_index=True,
    )

    combined_z = pd.concat(
        [
            -reference["y_world"],
            -comparison["y_world"],
        ],
        ignore_index=True,
    )

    x_range = padded_range(combined_x)
    y_range = padded_range(combined_y)
    z_range = padded_range(combined_z)

    def map_index(progress, frames):
        index = round(
            progress * (len(frames) - 1)
        )

        return frames[index]

    first_ref_frame = ref_frames[0]
    first_comp_frame = comp_frames[0]

    ref_skeleton, ref_joints, ref_time = frame_data(
        reference,
        first_ref_frame,
        "#22d3ee",
    )

    comp_skeleton, comp_joints, comp_time = frame_data(
        comparison,
        first_comp_frame,
        "#a78bfa",
    )

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[
            [
                {"type": "scene"},
                {"type": "scene"},
            ]
        ],
        column_widths=[0.5, 0.5],
        horizontal_spacing=0.03,
        subplot_titles=(
            "REFERENCE",
            "COMPARISON",
        ),
    )

    fig.add_trace(
        ref_skeleton,
        row=1,
        col=1,
    )

    fig.add_trace(
        ref_joints,
        row=1,
        col=1,
    )

    fig.add_trace(
        comp_skeleton,
        row=1,
        col=2,
    )

    fig.add_trace(
        comp_joints,
        row=1,
        col=2,
    )

    animation_frames = []
    slider_steps = []

    for step in range(steps):
        progress = (
            step / (steps - 1)
            if steps > 1
            else 0.0
        )

        ref_frame = map_index(
            progress,
            ref_frames,
        )

        comp_frame = map_index(
            progress,
            comp_frames,
        )

        ref_skeleton, ref_joints, ref_time = frame_data(
            reference,
            ref_frame,
            "#22d3ee",
        )

        comp_skeleton, comp_joints, comp_time = frame_data(
            comparison,
            comp_frame,
            "#a78bfa",
        )

        progress_percent = progress * 100

        frame_name = str(step)

        animation_frames.append(
            go.Frame(
                name=frame_name,
                traces=[0, 1, 2, 3],
                data=[
                    ref_skeleton,
                    ref_joints,
                    comp_skeleton,
                    comp_joints,
                ],
                layout=go.Layout(
                    title=dict(
                        text=(
                            "MotionStage 3D Performance Comparison"
                            "<br><sup>"
                            f"Progress {progress_percent:.1f}%"
                            " · "
                            f"Reference {ref_time:.2f}s"
                            " · "
                            f"Comparison {comp_time:.2f}s"
                            "</sup>"
                        )
                    )
                ),
            )
        )

        slider_steps.append(
            {
                "method": "animate",
                "args": [
                    [frame_name],
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
                "label": f"{progress_percent:.0f}%",
            }
        )

    fig.frames = animation_frames

    axis_style = dict(
        backgroundcolor="#07090d",
        gridcolor="#242830",
        zerolinecolor="#3a3f49",
        showbackground=True,
    )

    camera = dict(
        eye=dict(
            x=0.0,
            y=-2.4,
            z=0.25,
        )
    )

    fig.update_layout(
        title=dict(
            text=(
                "MotionStage 3D Performance Comparison"
                "<br><sup>"
                "Progress 0.0%"
                f" · Reference {ref_time:.2f}s"
                f" · Comparison {comp_time:.2f}s"
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
            xaxis=dict(
                title="Left / right",
                range=x_range,
                **axis_style,
            ),
            yaxis=dict(
                title="Depth",
                range=y_range,
                **axis_style,
            ),
            zaxis=dict(
                title="Vertical",
                range=z_range,
                **axis_style,
            ),
            aspectmode="data",
            camera=camera,
            uirevision="reference-camera",
        ),

        scene2=dict(
            xaxis=dict(
                title="Left / right",
                range=x_range,
                **axis_style,
            ),
            yaxis=dict(
                title="Depth",
                range=y_range,
                **axis_style,
            ),
            zaxis=dict(
                title="Vertical",
                range=z_range,
                **axis_style,
            ),
            aspectmode="data",
            camera=camera,
            uirevision="comparison-camera",
        ),

        margin=dict(
            l=10,
            r=10,
            t=100,
            b=105,
        ),

        showlegend=False,

        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.5,
                "xanchor": "center",
                "y": -0.04,
                "yanchor": "top",

                "buttons": [
                    {
                        "label": "▶ Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "fromcurrent": True,
                                "mode": "immediate",
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
                "x": 0.06,
                "len": 0.88,
                "y": 0.02,

                "currentvalue": {
                    "prefix": "Relative progress: ",
                    "font": {
                        "size": 13,
                    },
                },

                "pad": {
                    "t": 35,
                    "b": 5,
                },

                "steps": slider_steps,
            }
        ],
    )

    output = Path(args.output)

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.write_html(
        output,
        include_plotlyjs=True,
        full_html=True,
        auto_open=False,
    )

    print()
    print("=" * 64)
    print("MotionStage Synchronized 3D Comparison")
    print("=" * 64)

    print()
    print("REFERENCE")
    print(f"Frames:   {len(ref_frames)}")
    print(
        "Duration: "
        f"{reference['timestamp_seconds'].max():.2f}s"
    )

    print()
    print("COMPARISON")
    print(f"Frames:   {len(comp_frames)}")
    print(
        "Duration: "
        f"{comparison['timestamp_seconds'].max():.2f}s"
    )

    print()
    print(f"Animation steps: {steps}")
    print("Synchronization: relative performance progress")
    print(f"Output: {output}")

    print()
    print("Controls:")
    print("  Play / Pause       = synchronized playback")
    print("  Timeline           = synchronized scrubbing")
    print("  Drag each panel    = rotate in 3D")
    print("  Scroll             = zoom")


if __name__ == "__main__":
    main()
