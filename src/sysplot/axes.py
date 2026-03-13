import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.figure import Figure

from .config import get_config


# ___________________________________________________________________
#  Axis Highlight


def emphasize_coord_lines(
    fig: Figure | None = None,
    **kwargs,
) -> None:
    """Draw coordinate axes lines at the origin on all 2D axes in a figure.

    Adds one horizontal line at ``y=0`` and one vertical line at ``x=0`` for
    every 2D axes in the target figure.

    Args:
        fig: Matplotlib figure to modify. If ``None``, uses
            ``matplotlib.pyplot.gcf()``.
        **kwargs: Additional keyword arguments forwarded to
            :meth:`matplotlib.axes.Axes.axhline` and
            :meth:`matplotlib.axes.Axes.axvline`.

            ``color`` controls both coordinate lines unless overridden.
            If omitted, the default is ``rcParams["grid.color"]``.

    Note:
        3D axes are not currently supported.

    .. minigallery:: sysplot.emphasize_coord_lines
        :add-heading:
    """
    color = kwargs.pop("color", mpl.rcParams["grid.color"])
    linewidth = kwargs.pop("linewidth", get_config().highlight_linewidth)
    zorder = kwargs.pop("zorder", get_config().zorder_emphasized_grid)

    if fig is None:
        fig = plt.gcf()

    if not hasattr(fig, "canvas"):
        raise TypeError(
            "emphasize_coord_lines() expected a Matplotlib figure as argument or previously created figure."
        )

    # TODO: 3D axes are currently not supported and will raise an error.

    for ax in fig.axes:
        if isinstance(ax, Axes3D):  # Skip 3D axes
            raise TypeError(
                "emphasize_coord_lines  () currently does not support 3D axes."
            )
        if not any(line.get_gid() == "coord_x" for line in ax.lines):
            ax.axhline(
                0,
                color=color,
                linewidth=linewidth,
                zorder=zorder,
                gid="coord_x",
                **kwargs,
            )
        if not any(line.get_gid() == "coord_y" for line in ax.lines):
            ax.axvline(
                0,
                color=color,
                linewidth=linewidth,
                zorder=zorder,
                gid="coord_y",
                **kwargs,
            )


# ___________________________________________________________________
#  Axis Modifiers


def restore_tick_labels(
    fig: Figure | None = None,
) -> None:
    """Show tick labels on all axes of a figure.

    Useful when working with shared axes layouts where Matplotlib hides some
    tick labels by default. Instead of restoring the axis labels one by one, this function applies the necessary tick parameters to every axes in the figure so that all tick labels are visible.

    Args:
        fig: Matplotlib figure to modify. If ``None``, uses
            ``matplotlib.pyplot.gcf()``.

    .. minigallery:: sysplot.restore_tick_labels
        :add-heading:
    """
    if fig is None:
        fig = plt.gcf()

    for ax in fig.axes:
        ax.tick_params(labelbottom=True, labelleft=True)


def add_origin(ax: Axes | None = None) -> None:
    """Ensure the origin is included in axes autoscaling.

    Adds an invisible scatter point at ``(0, 0)`` so autoscaling includes the
    origin without changing the visible plot or advancing any style cyclers.

    Args:
        ax: Target axes. If ``None``, uses ``matplotlib.pyplot.gca()``.

    .. minigallery:: sysplot.add_origin
        :add-heading:
    """
    if ax is None:
        ax = plt.gca()

    #! using scatter with alpha=0 and no edgecolors/facecolors to avoid advancing style cycler or affecting the plot in any way.
    ax.scatter(0, 0, alpha=0, color="gray", facecolors="none", edgecolors="none")


def set_xmargin(ax: Axes | None = None, use_margin: bool = True) -> None:
    """Toggle x-axis margins for an axes.

    Applies either Matplotlib's default ``axes.xmargin`` value or the current
    project-configured ``axes.xmargin`` value, then refreshes autoscaling.

    Args:
        ax: Target axes to modify. If ``None``, uses
            ``matplotlib.pyplot.gca()``.
        use_margin: If ``True``, applies Matplotlib's default x-margin.
            If ``False``, applies the currently active/project x-margin.

    .. minigallery:: sysplot.set_xmargin
        :add-heading:
    """
    if ax is None:
        ax = plt.gca()
    if not isinstance(use_margin, bool):
        raise TypeError(f"use_margin must be a bool, got {type(use_margin)}")

    default_margin = mpl.rcParamsDefault["axes.xmargin"]
    project_margin = get_config().xmargin  # = 0 per default

    margin = default_margin if use_margin else project_margin
    ax.margins(x=margin)
    ax.autoscale_view()
