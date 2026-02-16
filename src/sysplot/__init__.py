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

Source code and documentation are hosted on GitHub at https://github.com/JaxRaffnix/sysplot.
"""

# Initialize global sysplot first
from .config import LANGUAGE, FIGURE_SIZE, FONT_SIZE, LINEWIDTH, MARKERSIZE, POLES_ZEROS_MARKERSIZE, ARROWSTYLE, apply_config

# Export public API
from .figures import get_figsize, save_current_figure
from .axes import highlight_axes, set_symmetric_axis_limits
from .styles import get_style
from .plotting import plot_stem, plot_nyquist, plot_bode, plot_poles_zeros
from .ticks import set_major_tick_labels, set_minor_log_ticks

__all__ = [
    "LANGUAGE", "FIGURE_SIZE", "FONT_SIZE", "LINEWIDTH",
    "MARKERSIZE", "POLES_ZEROS_MARKERSIZE", "ARROWSTYLE",
    "apply_config",
    "get_figsize",
    "save_current_figure",
    "highlight_axes",
    "set_symmetric_axis_limits",
    "get_style",
    "plot_stem",
    "plot_nyquist",
    "plot_bode",
    "plot_poles_zeros",
    "set_major_tick_labels",
    "set_minor_log_ticks",
]

# TODO: default values from config not inside function declaration, but rather inside the function body.
# TODO: maybe allow config params change only via the apply_config function, not direct public access?
