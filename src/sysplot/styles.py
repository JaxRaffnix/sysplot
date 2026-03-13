import matplotlib as mpl
from matplotlib.axes import Axes
from cycler import cycler

from typing import TypedDict
from typing import cast, Union


# ___________________________________________________________________
#  Styles


ColorTypeHint = Union[
    str, tuple[float, float, float], tuple[float, float, float, float]
]


class PlotStyle(TypedDict):
    """Type Hint for a style entry in the sysplot custom cycler ::`custom_styles`."""

    color: ColorTypeHint
    linestyle: str | tuple[int, ...]


# Global style configuration
_DEFAULT_COLORS = mpl.rcParamsDefault["axes.prop_cycle"].by_key()["color"]
_LINE_STYLES = [
    "-",  # solid
    "--",  # dashed
    "-.",  # dash-dot
    ":",  # dotted
    (0, (3, 1, 1, 1)),  # densely dashdotted
    (0, (5, 5)),  # dashed
    (0, (1, 1)),  # densely dotted
    (0, (5, 1)),  # densely dashed
    (0, (3, 5, 1, 5)),  # dashdotted
    (0, (1, 10)),  # loosely dotted
]
# Validate configuration
if len(_DEFAULT_COLORS) != len(_LINE_STYLES):
    raise ValueError(
        f"Color palette length ({len(_DEFAULT_COLORS)}) does not match number of line styles ({len(_LINE_STYLES)})."
    )

# Build and apply custom style cycler
_custom_cycler = cycler(color=_DEFAULT_COLORS) + cycler(linestyle=_LINE_STYLES)
custom_styles = list(_custom_cycler)


def _get_linestyle_for_color(color):
    target = mpl.colors.to_rgba(color)
    for style in custom_styles:
        if mpl.colors.to_rgba(style["color"]) == target:
            return style["linestyle"]

    # return default value if color is not found
    return "-"


def get_style(
    index: int | None = None,
    ax: Axes | None = None,
) -> PlotStyle:
    """Return a style entry from the sysplot custom cycler ::`custom_styles`.

    Use one of two modes:
    - ``index`` for deterministic access to a specific style.
    - ``ax`` to consume the next style from an axes color cycle.

    Args:
        index: Fixed style index in ``custom_styles``.
        ax: Target axes to read the next style from.

    Returns:
        PlotStyle: Dictionary with ``color`` and ``linestyle``.

    .. minigallery:: sysplot.get_style
        :add-heading:
    """
    if index is not None and ax is not None:
        raise ValueError("Cannot specify both index and ax. Please provide only one.")

    # index is provided
    if index is not None:
        if not isinstance(index, int):
            raise TypeError("Style index must be integer.")
        n = len(custom_styles)
        if not (0 <= index < n):
            raise IndexError(f"Style index out of range [0, {n - 1}].")
        return cast(PlotStyle, custom_styles[index].copy())

    # ax is provided
    if ax is not None:
        color = ax._get_lines.get_next_color()
        linestyle = _get_linestyle_for_color(color)

        return cast(PlotStyle, {"color": color, "linestyle": linestyle})

    raise ValueError("Either index or ax must be provided.")
