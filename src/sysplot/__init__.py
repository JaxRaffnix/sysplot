"""
sysplot
=======
Sysplot provides centralized plotting utilities for reproducible,
publication-quality figures in system theory and control engineering.

It extends Matplotlib with consistent figure styling, configuration management,
specialized helpers for annotating and improving visual clarity, and
high-level plotting functions for Bode plots, Nyquist diagrams, and pole-zero maps.

The project source code and documentation are available on GitHub: https://jaxraffnix.github.io/sysplot.
"""


# ___________________________________________________________________
#  Import Handling


# Initialize global sysplot first
from .config import SysplotConfig, apply_config, get_config, reset_config

# Export public API
from .figures import get_figsize, save_current_figure
from .axes import emphasize_coord_lines, repeat_axis_ticks, set_xmargin, add_origin
from .styles import PlotStyle, styles, get_style
from .plotting import plot_stem, plot_nyquist, plot_bode, plot_poles_zeros, plot_unit_circle, plot_filter_tolerance
from .ticks import set_major_ticks, set_minor_log_ticks, add_tick_line
from .angles import plot_angle
from .signals import heaviside

__all__ = [
  "SysplotConfig",
  "apply_config",
  "get_config",
  "reset_config",
  "get_figsize",
  "save_current_figure",
  "emphasize_coord_lines",
  "repeat_axis_ticks",
  "set_xmargin",
  "get_style",
  "plot_stem",
  "plot_nyquist",
  "plot_bode",
  "plot_poles_zeros",
  "plot_angle",
  "plot_unit_circle",
  "plot_filter_tolerance",
  "set_major_ticks",
  "set_minor_log_ticks",
  "add_tick_line",
  "heaviside",
  "PlotStyle",
  "styles",
  "add_origin",
]




# ___________________________________________________________________
#  Global To dos.


# TODO: default values from config not inside function declaration, but rather inside the function body.
# TODO: maybe allow config params change only via the apply_config function, not direct public access?
