# Uniform plotting aesthetics for all notebooks

import colorsys
import os
from collections import defaultdict
from typing import Any

import matplotlib as mpl  # noqa: F401
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt  # noqa: F401
import pyrootutils
import seaborn as sns
from IPython.display import display
from matplotlib.axes._axes import Axes
from seaborn.palettes import _ColorPalette as ColorPalette

PROJECT_ROOT = pyrootutils.find_root(
    search_from=os.path.abspath(""), indicator=".project-root"
)


FIGURES_DIR = PROJECT_ROOT / "notebooks" / "figures"

PAPER_WIDTH_IN = 5.5

rcs = {
    "font.size": 10.0,
    "axes.labelsize": "small",
    "axes.titlesize": "small",
    "xtick.labelsize": "x-small",
    "ytick.labelsize": "x-small",
}


def darken(
    color: str | tuple[float, float, float] | dict[str, Any] | ColorPalette,
    by: float = 0.2,
):
    """
    Darken a color by provided amount.
    """

    def _darken_color(c: str | tuple[float, float, float], by: float):
        by = min(max(0, by), 1)
        pct_darken = 1 - by

        if isinstance(c, str):
            c = sns.color_palette([c])[0]

        for c_i in c:
            if c_i > 1:
                c_i /= 255
        c_hls = colorsys.rgb_to_hls(c[0], c[1], c[2])
        # Darken the color by reducing the lightness

        c_hls = (
            c_hls[0],  # hue
            c_hls[1] * pct_darken,  # lightness
            c_hls[2],  # saturation
        )
        # Convert back to RGB
        c_rgb = colorsys.hls_to_rgb(c_hls[0], c_hls[1], c_hls[2])
        return c_rgb

    if isinstance(color, dict):
        # If color is a dictionary, assume it's a palette
        # and darken each color in the palette
        return {k: _darken_color(v, by) for k, v in color.items()}
    elif isinstance(color, ColorPalette):
        colors = [_darken_color(c, by) for c in color]
        return ColorPalette(colors)
    else:
        return _darken_color(color, by)


# For heatmaps, correlations, -1 to 1 scales, etc
CMAP_HEATMAP = "vlag"

# For any plots where color differentiates sample type
PALETTE_SAMPLE_TYPE = {
    "positive": darken("#ffcc66"),
    "negative": "#5c5cff",
    "unknown": "#C41E3A",
}

# For any plots where color differentiates model
PALETTE_MODEL = darken(
    {
        "gpt-4.1-nano": sns.cubehelix_palette(start=0.5, rot=-0.5, n_colors=4)[0],
        "gpt-4.1-mini": sns.cubehelix_palette(start=0.5, rot=-0.5, n_colors=4)[1],
        "gpt-4.1": sns.cubehelix_palette(start=0.5, rot=-0.5, n_colors=4)[2],
        "o4-mini": sns.color_palette("YlOrBr", n_colors=2)[0],
        "o3": sns.color_palette("YlOrBr", n_colors=2)[1],
        "gemma-3-1b": sns.cubehelix_palette(n_colors=5)[0],
        "gemma-3-4b": sns.cubehelix_palette(n_colors=5)[1],
        "gemma-3-12b": sns.cubehelix_palette(n_colors=5)[2],
        "gemma-3-27b": sns.cubehelix_palette(n_colors=5)[3],
        "DSR1-7B": sns.color_palette("cool", n_colors=2)[0],
    }
)

PALETTE_SCORE = {
    "Weighted F1": sns.color_palette("terrain", n_colors=2, desat=0.8)[0],
    "Macro F1": sns.color_palette("terrain", n_colors=2, desat=0.8)[1],
}

PALETTE_STRAGETY = {
    "rule-based": "#942822",
    "heuristic": "#FFE7CE",
    "code": sns.color_palette("gist_earth", n_colors=4)[2],
    "unknown": sns.color_palette("gist_earth", n_colors=4)[3],
}

PALETTES = {
    "model": PALETTE_MODEL,
    "sample_type": PALETTE_SAMPLE_TYPE,
    "score": PALETTE_SCORE,
    "strategy": PALETTE_STRAGETY,
}

MODEL_COLOR = "#4CA970"

# For marking at-chance baselines
COLOR_AT_CHANCE = "#ff0000"  # Red
ALPHA_AT_CHANCE = 0.5

# Bar Chart settings
BAR_EDGE_COLOR = "black"
BAR_EDGE_WIDTH = 0.8


def display_palette(palette: dict[str, Any] | ColorPalette):
    if isinstance(palette, ColorPalette):
        display(palette)
    else:
        colors = list(palette.values())
        display(sns.color_palette(colors))


def filter_by_alpha(
    keys: list[str],
    ax,
    palette: dict[str, Any] | None = None,
    alpha=0.3,
    highlight: str | list[str] | None = None,
):
    alphas = defaultdict(lambda: alpha)
    if highlight is not None:
        if isinstance(highlight, str):
            highlight = [highlight]
        for h in highlight:
            alphas[h] = 1

    if palette is None:
        # look through all the palettes in PALETTE; find one whose keys match
        # the keys passed in; if found, use that palette; otherwise, throw an error
        for palette_name, palette_dict in PALETTES.items():
            if all(k in palette_dict for k in keys):
                palette = palette_dict
                break
        else:
            raise ValueError(f"No matching palette found for keys: {keys}")

    face_colors = ax.collections[0].get_facecolors()
    face_colors[:, 3] = alpha
    for key in keys:
        key_color = palette[key]

        # Get indices of face_colors whose first 3 values match the model color
        indices = [
            i
            for i, color in enumerate(face_colors)
            if (color[0], color[1], color[2]) == key_color[:3]
        ]
        for i in indices:
            face_colors[i][3] = alphas[key]
    ax.collections[0].set_facecolor(face_colors)


def legend_format(
    ax: Axes | sns.FacetGrid,
    keys: list[str] | None = None,
    title: str | None = None,
    **kwargs,
):
    if isinstance(ax, sns.FacetGrid):
        fg = ax
        ax = fg.ax

        _ = fg.legend.remove()

    # Legend Formatting
    handles, labels = ax.get_legend_handles_labels()

    if keys is not None:
        if "gpt-4.1" in keys:
            spacing_locs = [3, 6]
        else:
            ValueError(f"Unsure how to space legend for {keys=}")

        spacer = mpatches.Patch(alpha=0, linewidth=0)
        for sloc in spacing_locs:
            handles.insert(sloc, spacer)
            labels.insert(sloc, "")

    _ = ax.legend(handles, labels)
    _ = ax.get_legend().set_frame_on(False)

    if title is not None:
        _ = ax.get_legend().set_title(title)

    if "loc" not in kwargs:
        kwargs["loc"] = "upper left"
    if "bbox_to_anchor" not in kwargs:
        kwargs["bbox_to_anchor"] = (1, 1)
    _ = sns.move_legend(ax, **kwargs)
