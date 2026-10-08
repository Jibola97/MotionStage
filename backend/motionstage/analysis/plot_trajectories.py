from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


JOINTS = [
    "left_wrist",
    "right_wrist",
    "left_elbow",
    "right_elbow",
    "left_hip",
    "right_hip",
    "left_ankle",
    "right_ankle",
]


def plot_trajectories(csv_path: str, output_dir: str) -> None:
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    for joint in JOINTS:
        joint_df = df[df["joint_name"] == joint]

        if joint_df.empty:
            continue

        plt.figure(figsize=(10, 5))

        plt.plot(
            joint_df["timestamp_seconds"],
            joint_df["x"],
            label="X position",
        )

        plt.plot(
            joint_df["timestamp_seconds"],
            joint_df["y"],
            label="Y position",
        )

        plt.title(f"{joint.replace('_', ' ').title()} Trajectory")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Pixel coordinate")
        plt.legend()
        plt.tight_layout()

        output_path = output_dir / f"{joint}_trajectory.png"
        plt.savefig(output_path, dpi=150)
        plt.close()

        print(f"Saved: {output_path}")


if __name__ == "__main__":
    plot_trajectories(
        "data/processed/test_performance_01_keypoints.csv",
        "data/processed/trajectory_plots",
    )
