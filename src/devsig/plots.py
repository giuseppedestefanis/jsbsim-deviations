"""Figure style for the paper: print-first, one hue per job, thin marks, recessive axes."""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# validated categorical slots (light surface): blue, orange, aqua; text and grid tokens
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8985", "#e6e5e1"
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
COL1, COL2 = 3.3, 5.0  # inches: a single-column journal text block is about 5 in wide; both are "full width" here


def style():
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
        "axes.edgecolor": GRID, "axes.linewidth": 0.8, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "xtick.color": INK2, "ytick.color": INK2,
        "axes.labelcolor": INK2, "text.color": INK, "legend.frameon": False, "lines.linewidth": 1.4,
        "figure.dpi": 150, "savefig.dpi": 300, "pdf.fonttype": 42, "axes.axisbelow": True,
    })


def seq_cmap():
    return mpl.colors.LinearSegmentedColormap.from_list("seqblue", SEQ)


def save(fig, path):
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(path.replace(".pdf", ".png"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
