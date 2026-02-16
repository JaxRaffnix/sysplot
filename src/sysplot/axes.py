import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.axis import Axis, XAxis
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.figure import Figure

from .figures import get_figsize
from .config import LINEWIDTH

def _ensure_two_axes(ax: Axes | None = None) -> np.ndarray:
    """Return an array of two axes for two-panel plots.

    If ``ax`` is ``None``, this function will either check the current figure for exactly two axes or create a new 1x2 figure and
    return the axes. If ``ax`` is provided, it must either be an array-like
    of exactly two axes.

    Args:
        ax (Axes or array-like, optional): Single axis, array of axes, or None.
            If None, uses from the current figure. As a failsafe, it creates axes

    Returns:
        np.ndarray: Array of two Matplotlib Axes objects.

    Raises:
        ValueError: If the number of axes is not exactly 2.
    """
    if ax is None:
        fig = plt.gcf()
        axes = fig.get_axes()
        if len(axes) == 2:
            return np.array(axes)
        elif len(axes) == 0:
            _, axes = plt.subplots(1, 2, figsize=get_figsize(1, 2))
            return np.array(axes)
        else:
            raise ValueError(f"Expected 0 or 2 axes in current figure, got {len(axes)}")
    
    ax = np.atleast_1d(ax)
    if len(ax) != 2:
        raise ValueError("ax must be array-like with exactly two axes.")
    return ax

def highlight_axes(
    fig: Figure | None = None, 
    linewidth: float = LINEWIDTH*2, 
    zorder: int = 1, 
    color: str = mpl.rcParams['grid.color']
) -> None:
    """Draws horizontal (y=0) and vertical (x=0) lines on each 2D axes to
    emphasize the coordinate system origin.

    Args:
        fig (Figure, optional): Matplotlib figure to modify. If None,
            the current figure (``plt.gcf()``) is used.
        linewidth (float, optional): Thickness of the coordinate axes lines.
            Defaults to ``LINEWIDTH``.
        zorder (int, optional): Drawing order for the coordinate axes.
            Default is 1 (below most plot elements).
        color (str, optional): Color of the coordinate axes. Default is the Matplotlib grid color.

    Raises:
        TypeError: If fig is not a valid Matplotlib Figure.
        ValueError: If linewidth <= 0 or if 3D axes are encountered.

    Note:
        - 3D axes are currently not supported and will raise an error.

    Examples:
        >>> fig, ax = plt.subplots()
        >>> highlight_axes(fig)
        >>> ax.plot([-2, 2], [-1, 1])
    """
    if fig is None:
        fig = plt.gcf()

    if not hasattr(fig, "canvas"):
        raise TypeError("highlight_axes() expected a Matplotlib figure as argument or previously created figure.")
    if linewidth <= 0:
        raise ValueError(f"linewidth must be > 0, got {linewidth}")
    
    # TODO: 3D axes are currently not supported and will raise an error.

    for ax in fig.axes:
        if isinstance(ax, Axes3D): # Skip 3D axes
            raise TypeError("highlight_axes() currently does not support 3D axes.")
        if not any(line.get_gid() == 'coord_x' for line in ax.lines):
            ax.axhline(0, color=color, linewidth=linewidth, zorder=zorder, gid='coord_x')
        if not any(line.get_gid() == 'coord_y' for line in ax.lines):
            ax.axvline(0, color=color, linewidth=linewidth, zorder=zorder, gid='coord_y')


# ___________________________________________________________________
#  Axis Modifiers

def set_symmetric_axis_limits(
    axis: Axis | None = None, 
    margin: None | float = 0.0
) -> None:
    """Set symmetric axis limits centered around zero.

    Adjusts the given axis so that its limits are symmetric about zero,
    based on the current maximum absolute value of the axis. Useful for plots where zero is a
    meaningful reference point.

    Args:
        axis (Axis, optional): Matplotlib axis (XAxis or YAxis) to modify.
            If None, uses the x-axis of the current axes (``plt.gca().xaxis``).
        margin (float, optional): Additional margin to add beyond the data
            range. If ``None``, uses Matplotlib's default x-margin setting. Use ``0`` for no margin.

    Raises:
        TypeError: If axis is not a Matplotlib Axis instance.

    Note:
        - If ``set_symmetric_axis_limits()`` is used on both x and y axis of a plot, it is not possible to use ``ax.axis("equal")`` afterwards, as this would override the symmetric limits.

    Example:
        >>> fig, ax = plt.subplots()
        >>> ax.plot([-3, 5], [1, 2])
        >>> set_symmetric_axis_limits(ax.xaxis)  # Sets limits to [-5, 5]

        >>> fig, ax = plt.subplots()
        >>> ax.plot([1, 2], [-4, 3])
        >>> set_symmetric_axis_limits(axis=ax.yaxis, margin=None)  # Sets limits to [-4, 4] + default margin

        >>> fig, ax = plt.subplots()
        >>> ax.plot([-2, 2], [-1, 1])
        >>> set_symmetric_axis_limits(ax.xaxis, margin=1)  # sets limits to [-3, 3]
    """
    if axis is None:
        axis = plt.gca().xaxis
    if not isinstance(axis, Axis):
        raise TypeError(f"ax must be a Matplotlib Axis instance, got {type(axis).__name__}")

    if margin is None:
        margin = float(mpl.rcParamsDefault['axes.xmargin'])

    limits = axis.get_view_interval()
    max_limit = max(abs(limits[0]), abs(limits[1]))
    max_limit = max_limit * (1 + margin)
    if isinstance(axis, XAxis):
        axis.axes.set_xlim(-max_limit, max_limit)
    else:
        axis.axes.set_ylim(-max_limit, max_limit)
