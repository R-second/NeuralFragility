"""Plotting helpers shared by paper-oriented examples."""
from pathlib import Path


def configure_paper_matplotlib(output_file, show=False):
    import matplotlib

    if not show:
        matplotlib.use("Agg")

    import matplotlib.font_manager as font_manager
    import matplotlib.pyplot as plt

    preferred_font = "Times New Roman"
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    font_family = preferred_font if preferred_font in available_fonts else "DejaVu Serif"

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
