import argparse
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


REFERENCE_COLOUR = "#22d3ee"
COMPARISON_COLOUR = "#a78bfa"

PRIMARY_COLOUR = "#f59e0b"
ISOLATED_COLOUR = "#f472b6"

JOINT_COLOUR = "#ffffff"


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


REGION_JOINTS = {
    "left_arm": {
        11, 13, 15, 17, 19, 21
    },

    "right_arm": {
        12, 14, 16, 18, 20, 22
    },

    "lower_body": {
        23, 24,
        25, 26,
        27, 28,
        29, 30,
        31, 32,
    },

    "torso": {
        11, 12, 23, 24
    },
}


def load_pose(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Pose CSV not found: {path}"
        )

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
            f"{path} missing columns: "
            f"{sorted(missing)}"
        )

    return df


def load_summary(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Summary not found: {path}"
        )

    with path.open() as handle:
        return json.load(handle)


def get_divergence(summary):
    divergence = (
        summary.get("divergence")
        or summary.get("overall_divergence")
    )

    if not divergence:
        raise ValueError(
            "No divergence data found in final_summary.json"
        )

    return divergence


def in_window(timestamp, window):
    if not window:
        return False

    start = float(
        window.get("start_seconds", 0)
    )

    end = float(
        window.get("end_seconds", start)
    )

    return start <= timestamp <= end


def region_connections(region):
    joints = REGION_JOINTS.get(
        region,
        set(),
    )

    return [
        (a, b)
        for a, b in CONNECTIONS
        if a in joints and b in joints
    ]


def build_lines(points, connections):
    x = []
    y = []
    z = []

    for start, end in connections:
        if (
            start not in points
            or end not in points
        ):
            continue

        x1, y1, z1 = points[start]
        x2, y2, z2 = points[end]

        x.extend([x1, x2, None])
        y.extend([y1, y2, None])
        z.extend([z1, z2, None])

    return x, y, z


def frame_data(
    df,
    frame_number,
    base_colour,
    primary_region=None,
    primary_active=False,
    isolated_region=None,
    isolated_active=False,
):
    current = (
        df[df["frame"] == frame_number]
        .sort_values("joint_id")
        .copy()
    )

    if current.empty:
        raise ValueError(
            f"No data for frame {frame_number}"
        )

    timestamp = float(
        current["timestamp_seconds"].iloc[0]
    )

    points = {
        int(row.joint_id): (
            float(row.x_world),
            float(row.z_world),
            float(-row.y_world),
        )
        for row in current.itertuples()
    }

    base_x, base_y, base_z = build_lines(
        points,
        CONNECTIONS,
    )

    base_trace = go.Scatter3d(
        x=base_x,
        y=base_y,
        z=base_z,
        mode="lines",
        line=dict(
            width=6,
            color=base_colour,
        ),
        hoverinfo="skip",
        showlegend=False,
    )

    joints_trace = go.Scatter3d(
        x=current["x_world"],
        y=current["z_world"],
        z=-current["y_world"],
        mode="markers",
        marker=dict(
            size=5,
            color=JOINT_COLOUR,
            line=dict(
                width=1,
                color=base_colour,
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

        # Explicitly hide inactive highlights so Plotly
    # cannot retain a highlight from the previous frame.
    highlight_x = [None]
    highlight_y = [None]
    highlight_z = [None]
    highlight_colour = PRIMARY_COLOUR
    highlight_active = False

    if isolated_active and isolated_region:
        highlight_active = True
        highlight_colour = ISOLATED_COLOUR

        (
            highlight_x,
            highlight_y,
            highlight_z,
        ) = build_lines(
            points,
            region_connections(
                isolated_region
            ),
        )

    elif primary_active and primary_region:
        highlight_active = True
        highlight_colour = PRIMARY_COLOUR

        (
            highlight_x,
            highlight_y,
            highlight_z,
        ) = build_lines(
            points,
            region_connections(
                primary_region
            ),
        )
        highlight_colour = PRIMARY_COLOUR

        (
            highlight_x,
            highlight_y,
            highlight_z,
        ) = build_lines(
            points,
            region_connections(
                primary_region
            ),
        )

    highlight_trace = go.Scatter3d(
        x=highlight_x,
        y=highlight_y,
        z=highlight_z,
        mode="lines",
        line=dict(
            width=14,
            color=highlight_colour,
        ),
        visible=highlight_active,
        hoverinfo="skip",
        showlegend=False,
    )

    return (
        base_trace,
        joints_trace,
        highlight_trace,
        timestamp,
    )


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
            "MotionStage 3D comparison viewer "
            "with divergence highlighting."
        )
    )

    parser.add_argument("reference_csv")
    parser.add_argument("comparison_csv")
    parser.add_argument("summary_json")

    parser.add_argument(
        "--output",
        default="pose3d_highlight.html",
    )

    args = parser.parse_args()

    reference = load_pose(
        args.reference_csv
    )

    comparison = load_pose(
        args.comparison_csv
    )

    summary = load_summary(
        args.summary_json
    )

    divergence = get_divergence(
        summary
    )

    isolated = summary.get(
        "isolated_arm_divergence"
    )

    primary_region = divergence.get(
        "most_divergent_region"
    )

    primary_ref_window = divergence.get(
        "reference_window"
    )

    primary_comp_window = divergence.get(
        "comparison_window"
    )

    isolated_region = None
    isolated_comp_window = None

    if isolated:
        isolated_region = isolated.get(
            "region"
        )

        isolated_comp_window = isolated.get(
            "comparison_window"
        )

    ref_frames = sorted(
        reference["frame"]
        .unique()
        .astype(int)
    )

    comp_frames = sorted(
        comparison["frame"]
        .unique()
        .astype(int)
    )

    steps = max(
        len(ref_frames),
        len(comp_frames),
    )

    combined_x = pd.concat(
        [
            reference["x_world"],
            comparison["x_world"],
        ]
    )

    combined_y = pd.concat(
        [
            reference["z_world"],
            comparison["z_world"],
        ]
    )

    combined_z = pd.concat(
        [
            -reference["y_world"],
            -comparison["y_world"],
        ]
    )

    x_range = padded_range(combined_x)
    y_range = padded_range(combined_y)
    z_range = padded_range(combined_z)

    def mapped_frame(
        progress,
        frames,
    ):
        index = round(
            progress
            * (len(frames) - 1)
        )

        return frames[index]

    first_ref = ref_frames[0]
    first_comp = comp_frames[0]

    (
        ref_base,
        ref_joints,
        ref_highlight,
        ref_time,
    ) = frame_data(
        reference,
        first_ref,
        REFERENCE_COLOUR,
    )

    (
        comp_base,
        comp_joints,
        comp_highlight,
        comp_time,
    ) = frame_data(
        comparison,
        first_comp,
        COMPARISON_COLOUR,
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
        subplot_titles=(
            "REFERENCE",
            "COMPARISON",
        ),
        horizontal_spacing=0.03,
    )

    for trace in [
        ref_base,
        ref_joints,
        ref_highlight,
    ]:
        fig.add_trace(
            trace,
            row=1,
            col=1,
        )

    for trace in [
        comp_base,
        comp_joints,
        comp_highlight,
    ]:
        fig.add_trace(
            trace,
            row=1,
            col=2,
        )

    frames = []
    slider_steps = []

    for step in range(steps):
        progress = (
            step / (steps - 1)
            if steps > 1
            else 0.0
        )

        ref_frame = mapped_frame(
            progress,
            ref_frames,
        )

        comp_frame = mapped_frame(
            progress,
            comp_frames,
        )

        ref_rows = reference[
            reference["frame"]
            == ref_frame
        ]

        comp_rows = comparison[
            comparison["frame"]
            == comp_frame
        ]

        ref_timestamp = float(
            ref_rows[
                "timestamp_seconds"
            ].iloc[0]
        )

        comp_timestamp = float(
            comp_rows[
                "timestamp_seconds"
            ].iloc[0]
        )

        # Keep the reference skeleton visually stable.
        # Divergence highlighting is shown on the comparison performance.
        ref_primary_active = False

        comp_primary_active = in_window(
            comp_timestamp,
            primary_comp_window,
        )

        comp_isolated_active = in_window(
            comp_timestamp,
            isolated_comp_window,
        )

        (
            ref_base,
            ref_joints,
            ref_highlight,
            ref_time,
        ) = frame_data(
            reference,
            ref_frame,
            REFERENCE_COLOUR,
            primary_region=primary_region,
            primary_active=ref_primary_active,
        )

        (
            comp_base,
            comp_joints,
            comp_highlight,
            comp_time,
        ) = frame_data(
            comparison,
            comp_frame,
            COMPARISON_COLOUR,
            primary_region=primary_region,
            primary_active=comp_primary_active,
            isolated_region=isolated_region,
            isolated_active=comp_isolated_active,
        )

        status = "No active divergence window"

        if comp_isolated_active:
            status = (
                "ISOLATED ARM DEVIATION"
                f" · {isolated_region}"
            )

        elif comp_primary_active:
            status = (
                "PRIMARY DIVERGENCE"
                f" · {primary_region}"
            )

        frame_name = str(step)

        frames.append(
            go.Frame(
                name=frame_name,
                traces=[
                    0, 1, 2,
                    3, 4, 5,
                ],
                data=[
                    ref_base,
                    ref_joints,
                    ref_highlight,
                    comp_base,
                    comp_joints,
                    comp_highlight,
                ],
                layout=go.Layout(
                    title=dict(
                        text=(
                            "MotionStage 3D Divergence Viewer"
                            "<br><sup>"
                            f"Progress {progress * 100:.1f}%"
                            " · "
                            f"Reference {ref_time:.2f}s"
                            " · "
                            f"Comparison {comp_time:.2f}s"
                            "<br>"
                            f"{status}"
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
                "label": "",
            }
        )

    # Keep the slider fully frame-accurate while showing only
    # five clean progress labels.
    if slider_steps:
        progress_labels = (
            (0.00, "0%"),
            (0.25, "25%"),
            (0.50, "50%"),
            (0.75, "75%"),
            (1.00, "100%"),
        )

        last_index = len(slider_steps) - 1

        for fraction, label in progress_labels:
            index = round(fraction * last_index)
            slider_steps[index]["label"] = label

    fig.frames = frames

    axis_style = dict(
        backgroundcolor="#07090d",
        gridcolor="#242830",
        zerolinecolor="#3a3f49",
        showbackground=True,
    )

    camera = dict(
        eye=dict(
            x=0,
            y=-2.4,
            z=0.25,
        )
    )

    fig.update_layout(
        title=dict(
            text=(
                "MotionStage 3D Divergence Viewer"
                "<br><sup>"
                "Orange = primary divergence"
                " · Pink = isolated-arm deviation"
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
            t=110,
            b=110,
        ),

        showlegend=False,

        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.5,
                "xanchor": "center",
                "y": -0.04,

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
                "x": 0.06,
                "len": 0.88,
                "y": 0.02,

                "currentvalue": {
                    "prefix": (
                        "Relative progress: "
                    )
                },

                "steps": slider_steps,
            }
        ],
    )

    output = Path(
        args.output
    )

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
    print("=" * 65)
    print(
        "MotionStage 3D Divergence Highlight Viewer"
    )
    print("=" * 65)

    print()
    print("Primary divergence:")
    print(
        f"  Region: {primary_region}"
    )
    print(
        "  Reference: "
        f"{primary_ref_window}"
    )
    print(
        "  Comparison: "
        f"{primary_comp_window}"
    )

    print()
    print("Isolated arm deviation:")

    if isolated:
        print(
            f"  Region: {isolated_region}"
        )
        print(
            "  Comparison: "
            f"{isolated_comp_window}"
        )
    else:
        print("  None")

    print()
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
