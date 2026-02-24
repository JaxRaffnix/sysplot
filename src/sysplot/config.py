"""
Global configuration for sysplot.

This module defines global settings that are applied
*automatically at import time* when ``sysplot`` is imported.

Importing this module has the following effects:

- A Seaborn theme suitable for publication (``context="paper"``) is activated
- Matplotlib ``rcParams`` are updated to enforce consistent figure layout,
  font sizes, line widths, and rendering resolution
- Global constants defining figure size, font size, and styling parameters
  are made available to the rest of the package

Note:
    These values are intended to be read-only. Runtime modifications are
    possible but not part of the stable public API and may cause
    inconsistent results.

Constants:
    LANGUAGE (str):
        Default language for labels and filenames. Supported values are
        ``"de"`` and ``"en"``.
    FIGURE_SIZE (tuple[float, float]):
        Default figure dimensions in inches as ``(width, height)``.
    FONT_SIZE (int):
        Base font size for all text elements.
    LINEWIDTH (float):
        Default line width for plotted lines.
    MARKERSIZE (float):
        Default marker size for data points.
    POLES_ZEROS_MARKERSIZE (float):
        Marker size for pole-zero diagrams.
    ARROWSTYLE (str):
        Matplotlib arrow style used for annotations.
"""

import seaborn as sns
import matplotlib.pyplot as plt

# ___________________________________________________________________
# Constants


LANGUAGE = "de"         # default language for labels and filenames. Supports "de" and "en".
FIGURE_SIZE = (7, 5)    # figure size in inches: (width, height)
FONT_SIZE = 11
LINEWIDTH = 1
MARKERSIZE = 6
POLES_ZEROS_MARKERSIZE = 10
ARROWSTYLE = '-|>'


# ___________________________________________________________________
#  Public Style Invokation

def apply_config():
    """
    Sets seaborn theme and MatplotLib Params.

    Ensures consistent figure style by using the config constants and a seaborn theme.
    """
    # TODO: this breaks when called before a plot_stem call. get_linestyle_for_color does not work with this

    #! must be called before rcParams update
    sns.set_theme(context="paper", style="whitegrid") 

    # latex document uses font avant sfdefault 
    # ? maybe add the font or a free verision of it? 
    plt.rcParams.update({
        # Layout and resolution
        "figure.constrained_layout.use": True,
        "figure.figsize": FIGURE_SIZE,
        "savefig.dpi": 300,
        "figure.dpi": 150,

        # Axes behavior
        "axes.autolimit_mode": "data",
        "axes.xmargin": 0,
        "axes.formatter.limits": (-9, 9),
        "legend.loc": "best",
        "ytick.labelleft": True,

        # Font sizes
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,

        # Line defaults
        "lines.linewidth": LINEWIDTH,
        "lines.markersize": MARKERSIZE,

        # tick config
        "xtick.direction": "in",
        "ytick.direction": "in",

        # latex font rendering
        # "text.usetex" : True,
        # 'font.family' : 'sans-serif',
        # "text.latex.preamble" : r"\usepackage{avant} \usepackage{sansmath} \sansmath"
    })

apply_config()
