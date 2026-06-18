"""Plotting helpers shared by paper-oriented examples."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any, TypeAlias

PathLikeStr: TypeAlias = str | PathLike[str]


def configure_matplotlib(
    output_file: PathLikeStr,
    show: bool = False,
) -> tuple[Any, float]:
    """Configure matplotlib.

    Args:
        output_file: Path to the output image file. Line transparency will be adjusted based on the file extension.
        show: Whether to display the plot interactively.

    Returns:
        `matplotlib.pyplot` instance and the recommended line transparency.
    """
    import matplotlib

    if not show:
        matplotlib.use("Agg")

    import matplotlib.font_manager as font_manager
    import matplotlib.pyplot as plt

    preferred_font = "Times New Roman"
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    font_family = (
        preferred_font if preferred_font in available_fonts else "DejaVu Serif"
    )

    plt.rcParams["font.family"] = font_family
    plt.rcParams["font.size"] = 20
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["axes.xmargin"] = 0.01
    plt.rcParams["axes.ymargin"] = 0.01
    plt.rcParams["legend.fancybox"] = False
    plt.rcParams["legend.framealpha"] = 1
    plt.rcParams["legend.edgecolor"] = "black"
    plt.rcParams["mathtext.default"] = "default"

    output_file = Path(output_file)
    line_alpha = 1.0 if output_file.suffix.lower() == ".eps" else 0.5
    return plt, line_alpha
