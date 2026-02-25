"""
sysplot
=======
Public API for the sysplot module.

Centralized plotting utilities for reproducible, publication-quality
figures in system theory and control engineering.

The package provides:

- Consistent figure sizing and saving
- Global style and configuration management
- Axis and tick helpers
- Control-theory visualizations:
  Bode plots, Nyquist diagrams, and pole-zero maps

Source code and documentation are hosted on GitHub at https://jaxraffnix.github.io/sysplot.
"""


# ___________________________________________________________________
#  Import Handling


# Initialize global sysplot first
from .config import LANGUAGE, FIGURE_SIZE, FONT_SIZE, LINEWIDTH, MARKERSIZE, POLES_ZEROS_MARKERSIZE, ARROWSTYLE, apply_config

# Export public API
from .figures import get_figsize, save_current_figure
from .axes import highlight_axes, repeat_axis_ticks
from .styles import get_style, get_next_style
from .plotting import plot_stem, plot_nyquist, plot_bode, plot_poles_zeros, plot_unit_circle, plot_filter_tolerance
from .ticks import set_major_ticks, set_minor_log_ticks, add_tick_line
from .angles import plot_angle

__all__ = [
  "LANGUAGE", "FIGURE_SIZE", "FONT_SIZE", "LINEWIDTH",
  "MARKERSIZE", "POLES_ZEROS_MARKERSIZE", "ARROWSTYLE",
  "apply_config",
  "get_figsize",
  "save_current_figure",
  "highlight_axes",
  "repeat_axis_ticks",
  "get_style",
  "get_next_style",
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
]


# ___________________________________________________________________
#  Global To dos.


# TODO: default values from config not inside function declaration, but rather inside the function body.
# TODO: maybe allow config params change only via the apply_config function, not direct public access?
