# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
"""Plotting functions for the noise study."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from data_utils import (
    DMC_ANALYSIS_DIR,
    load_eval_stats,
    load_object_model,
)
from plot_utils import (
    TBP_COLORS,
    axes3d_set_aspect_equal,
    init_matplotlib_style,
    violinplot,
)

init_matplotlib_style()

# Shared random number generator.
rng = np.random.RandomState(0)

# Directories to save plots and tables to.
OUT_DIR = DMC_ANALYSIS_DIR / "fig3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

all_info = {
    "dist_agent_1lm_none": {
        "run_name": "dist_agent_1lm_none",
        "label": "none",
        "noise_params": {},
    },
    "dist_agent_1lm_loc": {
        "run_name": "dist_agent_1lm_loc",
        "label": "loc",
        "noise_params": {"location": 0.002},
    },
    "dist_agent_1lm_hsv": {
        "run_name": "dist_agent_1lm_hsv",
        "label": "hsv",
        "noise_params": {"features": {"hsv": 0.1}},
    },
    "dist_agent_1lm_hsv5": {
        "run_name": "dist_agent_1lm_hsv5",
        "label": "large hsv",
        "noise_params": {"features": {"hsv": 0.5}},
    },
    "dist_agent_1lm_loc_hsv": {
        "run_name": "dist_agent_1lm_loc_hsv",
        "label": "loc + hsv",
        "noise_params": {"location": 0.002, "features": {"hsv": 0.1}},
    },
    "dist_agent_1lm_loc_hsv5": {
        "run_name": "dist_agent_1lm_loc_hsv5",
        "label": "loc + large hsv",
        "noise_params": {"location": 0.002, "features": {"hsv": 0.5}},
    },
    "dist_agent_1lm_std": {
        "run_name": "dist_agent_1lm_std",
        "label": "std",
        "noise_params": {
            "location": 0.002,
            "features": {
                "pose_vectors": 2.0,
                "hsv": 0.1,
                "principal_curvatures_log": 0.1,
                "pose_fully_defined": 0.01,
            },
        },
    },
    "dist_agent_1lm_std_hsv5": {
        "run_name": "dist_agent_1lm_std_hsv5",
        "label": "std + large hsv",
        "noise_params": {
            "location": 0.002,
            "features": {
                "pose_vectors": 2.0,
                "hsv": 0.5,
                "principal_curvatures_log": 0.1,
                "pose_fully_defined": 0.01,
            },
        },
    },
}


def plot_performance():
    """Plot the performance of the distant agent with different levels of noise."""
    # Initialize output paths.
    out_dir = OUT_DIR / "performance"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load the stats.
    experiments = [
        "dist_agent_1lm_none",
        "dist_agent_1lm_loc",
        "dist_agent_1lm_hsv",
        "dist_agent_1lm_hsv5",
        "dist_agent_1lm_loc_hsv",
        "dist_agent_1lm_loc_hsv5",
        "dist_agent_1lm_std",
        "dist_agent_1lm_std_hsv5",
    ]
    dataframes = [load_eval_stats(exp) for exp in experiments]
    infos = [all_info[exp] for exp in experiments]
    xticklabels = [info["label"] for info in infos]

    accuracy, rotation_error = [], []
    for i, df in enumerate(dataframes):
        sub_df = df[df.primary_performance.isin(["correct", "correct_mlh"])]
        accuracy.append(100 * len(sub_df) / len(df))
        rotation_error.append(np.degrees(sub_df.rotation_error))

    # Initialize the plot.
    fig, ax1 = plt.subplots(1, 1, figsize=(5, 4))
    ax2 = ax1.twinx()

    # Params
    bar_width = 0.4
    violin_width = 0.4
    gap = 0.02
    xticks = np.arange(len(experiments)) * 1.3
    bar_positions = xticks - bar_width / 2 - gap / 2
    violin_positions = xticks + violin_width / 2 + gap / 2
    median_style = dict(color="lightgray", lw=1, ls="-")

    # Plot accuracy bars
    ax1.bar(
        bar_positions,
        accuracy,
        color=TBP_COLORS["blue"],
        width=bar_width,
    )
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("% Correct")

    # Plot rotation error violins
    violinplot(
        rotation_error,
        violin_positions,
        width=violin_width,
        color=TBP_COLORS["purple"],
        showextrema=False,
        showmedians=True,
        median_style=median_style,
        bw_method=0.1,
        ax=ax2,
    )

    ax2.set_yticks([0, 45, 90, 135, 180])
    ax2.set_ylim(0, 180)
    ax2.set_ylabel("Rotation Error (deg)")

    ax1.set_xticks(xticks)

    ax1.set_xticklabels(xticklabels, rotation=45, ha="right")
    # Adjust x-axis tick label positions to the right

    ax1.spines["right"].set_visible(True)
    ax2.spines["right"].set_visible(True)

    fig.tight_layout()
    fig.savefig(out_dir / "performance.png")
    fig.savefig(out_dir / "performance.svg")
    plt.show()


def draw_icons():
    """Draw the mug model with different levels of noise applied."""
    # Initialize output paths.
    out_dir = OUT_DIR / "icons"
    out_dir.mkdir(parents=True, exist_ok=True)

    experiments = [
        "dist_agent_1lm_none",
        "dist_agent_1lm_loc",
        "dist_agent_1lm_hsv",
        "dist_agent_1lm_hsv5",
        "dist_agent_1lm_loc_hsv",
        "dist_agent_1lm_loc_hsv5",
        "dist_agent_1lm_std",
        "dist_agent_1lm_std_hsv5",
    ]
    n_experiments = len(experiments)
    axes_height = axes_width = 2

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(2 * axes_width, 2 * axes_height),
        subplot_kw={"projection": "3d"},
    )
    model = load_object_model("dist_agent_1lm", "mug")
    model = model - [0, 1.5, 0]

    for i in range(n_experiments):
        ax = axes.flatten()[i]
        exp = experiments[i]
        info = all_info[exp]
        noise_params = info["noise_params"]
        label = info["label"]

        obj = model.copy()
        if "location" in noise_params:
            noise = rng.normal(0, noise_params["location"], obj.pos.shape)
            obj.pos = obj.pos + noise

        if "features" in noise_params:
            if "hsv" in noise_params["features"]:
                hsv_std = noise_params["features"]["hsv"]

                # Convert RGB to HSV
                rgb = model.rgba[:, :3]
                hsv = np.zeros_like(rgb)
                for j in range(len(rgb)):
                    hsv[j] = matplotlib.colors.rgb_to_hsv(rgb[j])

                # Add HSV noise
                noise = rng.normal(0, hsv_std, hsv.shape)
                hsv = hsv + noise
                hsv = np.clip(hsv, 0, 1)

                # Convert back to RGB
                rgba = np.ones((len(hsv), 4))
                for j in range(len(hsv)):
                    rgba[j, :3] = matplotlib.colors.hsv_to_rgb(hsv[j])
                obj.rgba = rgba

        ax.scatter(obj.x, obj.y, obj.z, c=obj.rgba, alpha=0.45, edgecolors="none", s=2)
        ax.set_title(label)
        axes3d_set_aspect_equal(ax)
        ax.axis("off")
        ax.view_init(100, -90)

    fig.savefig(out_dir / "icons.png")
    fig.savefig(out_dir / "icons.svg")
    plt.show()


plot_performance()
draw_icons()
