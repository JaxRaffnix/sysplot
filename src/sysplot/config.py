from dataclasses import dataclass, replace
from typing import Literal, Tuple

import seaborn as sns
import matplotlib.pyplot as plt

from .styles import _custom_cycler


# ___________________________________________________________________
# Global Default Configuration


@dataclass(slots=True)
class SysplotConfig:
    """
    Global configuration for sysplot visualizations.

    For details, please refer to https://matplotlib.org/stable/users/explain/customizing.html and https://seaborn.pydata.org/tutorial/aesthetics.html
    """
    # Figure Layout
    figure_size: tuple[float, float] = (7.0, 5.0)
    max_fig_size_factor: int = 2
    figure_dpi: int = 150
    savefig_dpi: int = 300
    figure_fmt: str = "pdf"
    constrained_layout: bool = True

    # Typography
    font_size: int = 11
    use_latex: bool = False
    font_family: str = "sans-serif"

    # Plots, Lines and Markers
    linewidth: float = 1
    highlight_linewidth: float = 2
    markersize: float = 6
    poles_zeros_markersize: float = 10

    # Axes and Ticks
    tick_direction: Literal["in", "out", "inout"] = "in"
    xmargin: float = 0.0
    formatter_limits: Tuple[int, int] = (-9, 9)

    # Seaborn Theme
    seaborn_context: Literal["paper", "notebook", "talk", "poster"] = "paper"
    seaborn_style: Literal["white", "whitegrid", "darkgrid", "ticks"] = "whitegrid"

    # For My Custom Functions
    arrowstyle: str = "-|>"

    def validate(self) -> None:
        if self.font_size <= 0:
            raise ValueError("font_size must be positive.")
        if self.linewidth <= 0:
            raise ValueError("linewidth must be positive.")
        if self.figure_dpi <= 0 or self.savefig_dpi <= 0:
            raise ValueError("DPI values must be positive.")

# Global active configuration
_config = SysplotConfig()


def get_config() -> SysplotConfig:
    """Return current active configuration (read-only usage recommended)."""
    return _config


def reset_config() -> None:
    """Reset configuration to library defaults."""
    global _config
    _config = SysplotConfig()
    _apply_rcparams()


# ___________________________________________________________________
#  Apply My Config


# TODO: this breaks when called before a plot_stem call. get_linestyle_for_color does not work with this

def apply_config(
    config: SysplotConfig | None = None,
    **overrides,
):
    """
    Globally apply plotting configuration with rcParams and seaborn theme.

    Either pass a full SysPlotConfig instance,
    or override individual parameters via keyword arguments.

    The seaborn theme is loaded first, then the rcParams are updated to potentially override the seaborn values.

    Note:
        With the default seaborn theme whitegrid, the patche edgecolor is white by default. This means a default plt arrow, e.g. plt.annotate(arrowprops) will draw the arrow in white, which will not be visble. Alywas explictily pass `arrowprops=dict(arrowstyle="-|>", color="black")` to plt.annotate() to fix that.
    """
    global _config

    if config is not None:
        if not isinstance(config, SysplotConfig):
            raise TypeError("config must be a SysplotConfig instance.")
        _config = config
    else:
        # validate override keys
        valid_fields = SysplotConfig.__dataclass_fields__.keys()
        invalid = set(overrides) - set(valid_fields)
        if invalid:
            raise ValueError(f"Invalid config field(s): {invalid}")

        if overrides:
            _config = replace(_config, **overrides)

    _config.validate()
    _apply_rcparams()



def _apply_rcparams() -> None:
    """Internal: map SysPlotConfig → matplotlib rcParams."""

    #! must be called before rcParams update
    sns.set_theme(
        context=_config.seaborn_context,
        style=_config.seaborn_style,
    )

    plt.rcParams.update({
        # Layout
        "figure.constrained_layout.use": _config.constrained_layout,
        "figure.figsize": _config.figure_size,
        "figure.dpi": _config.figure_dpi,
        "savefig.dpi": _config.savefig_dpi,

        # Axes
        "axes.autolimit_mode": "data",
        "axes.xmargin": _config.xmargin,
        "axes.formatter.limits": _config.formatter_limits,
        "legend.loc": "best",
        "ytick.labelleft": True,

        # Font sizes
        "font.size": _config.font_size,
        "axes.labelsize": _config.font_size,
        "axes.titlesize": _config.font_size,
        "legend.fontsize": _config.font_size,
        "xtick.labelsize": _config.font_size,
        "ytick.labelsize": _config.font_size,

        # Lines
        "lines.linewidth": _config.linewidth,
        "lines.markersize": _config.markersize,

        # Ticks
        "xtick.direction": _config.tick_direction,
        "ytick.direction": _config.tick_direction,

        # use my custom cycler for color and linestyle
        'axes.prop_cycle': _custom_cycler,

        #? import latex font?
        "font.family": _config.font_family,
        "text.usetex": _config.use_latex,
        # "text.latex.preamble" : r"\usepackage{avant} \usepackage{sansmath} \sansmath"
    })
