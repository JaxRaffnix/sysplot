import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.axis import Axis, XAxis
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.figure import Figure

from .figures import get_figsize
from .config import get_config


# ___________________________________________________________________
#  Axis Highlight


def emphasize_coord_lines(
    fig: Figure | None = None, 
    zorder: int = 1, 
    linewidth: float|None = None,
    color: str|None = None
) -> None:
    """Draw origin guide lines on each 2D axes in a figure.

    Adds one horizontal line at ``y=0`` and one vertical line at ``x=0`` for
    every 2D axes in the target figure. Existing lines with the same internal
    IDs are not duplicated.

    Args:
        fig: Matplotlib figure to modify. If ``None``, uses
            ``matplotlib.pyplot.gcf()``.
        zorder: Drawing order of the added guide lines.
        linewidth: Line width of the guide lines. If ``None``, uses the
            project configuration value.
        color: Line color. If ``None``, uses ``matplotlib.rcParams['grid.color']``.

    Note:
        3D axes are not currently supported.

    Example:
        :ref:`sphx_glr__auto_examples_highlight_axes.py`
    """
    color = color or mpl.rcParams['grid.color'] 
    linewidth = linewidth or get_config().highlight_linewidth

    if fig is None:
        fig = plt.gcf()

    if not hasattr(fig, "canvas"):
        raise TypeError("emphasize_coord_lines() expected a Matplotlib figure as argument or previously created figure.")
    
    
    # TODO: 3D axes are currently not supported and will raise an error.

    for ax in fig.axes:
        if isinstance(ax, Axes3D): # Skip 3D axes
            raise TypeError("emphasize_coord_lines  () currently does not support 3D axes.")
        if not any(line.get_gid() == 'coord_x' for line in ax.lines):
            ax.axhline(0, color=color, linewidth=linewidth, zorder=zorder, gid='coord_x')
        if not any(line.get_gid() == 'coord_y' for line in ax.lines):
            ax.axvline(0, color=color, linewidth=linewidth, zorder=zorder, gid='coord_y')


# ___________________________________________________________________
#  Axis Modifiers

def repeat_axis_ticks(fig: Figure | None = None, ) -> None:
    """Show tick labels on all axes of a figure.

    Useful when working with shared axes layouts where Matplotlib hides some
    tick labels by default.

    Args:
        fig: Matplotlib figure to modify. If ``None``, uses
            ``matplotlib.pyplot.gcf()``.

    Example:
        :ref:`sphx_glr__auto_examples_repeat_axis_ticks.py`
    """
    if fig is None:
        fig = plt.gcf()

    for ax in fig.axes:
        ax.tick_params(labelbottom=True, labelleft=True)


def add_origin(ax: Axes|None = None) -> None:
    """Ensure the origin is included in axes autoscaling.

    Adds an invisible scatter point at ``(0, 0)`` so autoscaling includes the
    origin without changing the visible plot.

    Args:
        ax: Target axes. If ``None``, uses ``matplotlib.pyplot.gca()``.

    Example:
        :ref:`sphx_glr__auto_examples_add_origin.py`
    """
    if ax is None:
        ax = plt.gca()
    ax.scatter(0, 0, alpha=0, color="gray", facecolors='none', edgecolors='none')


def set_xmargin(ax: Axes|None = None, use_margin: bool = True) -> None:
    """Toggle x-axis margins for an axes.

    Applies either Matplotlib's default ``axes.xmargin`` value or the current
    project-configured ``axes.xmargin`` value, then refreshes autoscaling.

    Args:
        ax: Target axes to modify. If ``None``, uses
            ``matplotlib.pyplot.gca()``.
        use_margin: If ``True``, applies Matplotlib's default x-margin.
            If ``False``, applies the currently active/project x-margin.

    Example:
        :ref:`sphx_glr__auto_examples_set_xmargin.py`
    """
    if ax is None:
        ax = plt.gca()
    if not isinstance(use_margin, bool):
        raise TypeError(f"use_margin must be a bool, got {type(use_margin)}")
    
    default_margin = mpl.rcParamsDefault['axes.xmargin']
    project_margin = mpl.rcParams['axes.xmargin']

    margin = default_margin if use_margin else project_margin
    ax.margins(x=margin)
    ax.autoscale_view()
