import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from cycler import cycler

from typing import TypedDict
from typing import cast, Union, Tuple


# ___________________________________________________________________
#  Styles


ColorTypeHint = Union[
    str,
    Tuple[float, float, float],
    Tuple[float, float, float, float]
]

class PlotStyle(TypedDict):
    color: ColorTypeHint
    linestyle: Union[str, tuple[int, ...]]

# TODO: do these:
#? implement default gray filled value with transparency for areas?


# Global style configuration
_DEFAULT_COLORS = mpl.rcParamsDefault['axes.prop_cycle'].by_key()['color']
_LINE_STYLES = [
    '-',                    # solid
    '--',                   # dashed
    '-.',                   # dash-dot
    ':',                    # dotted
    (0, (3, 1, 1, 1)),     # densely dashdotted
    (0, (5, 5)),           # dashed
    (0, (1, 1)),           # densely dotted
    (0, (5, 1)),           # densely dashed
    (0, (3, 5, 1, 5)),     # dashdotted
    (0, (1, 10)),          # loosely dotted
]
# Validate configuration
if len(_DEFAULT_COLORS) != len(_LINE_STYLES):
    raise ValueError(f"Color palette length ({len(_DEFAULT_COLORS)}) does not match number of line styles ({len(_LINE_STYLES)}).")

# Build and apply custom style cycler
_custom_cycler = cycler(color=_DEFAULT_COLORS) + cycler(linestyle=_LINE_STYLES)
styles = list(_custom_cycler)


def _get_linestyle_for_color(color):
    target = mpl.colors.to_rgba(color)
    for style in styles:
        if mpl.colors.to_rgba(style["color"]) == target:
            return style["linestyle"]
    raise ValueError(f"Color {color} not found in custom cycler.")


def get_style(
    index: int|None = None, 
    ax: Axes|None=None, 
) -> PlotStyle:
    """Return a style entry from the sysplot custom cycler.

    Use one of two modes:

    - ``index`` for deterministic access to a specific style.
    - ``ax`` to consume the next style from an axes color cycle.

    When ``index`` is used, the current axes cycler is also advanced once to
    keep behavior aligned with plotting calls.

    Args:
        index: Fixed style index in ``styles``.
        ax: Target axes to read the next style from.

    Returns:
        PlotStyle: Dictionary with ``color`` and ``linestyle``.

    Example:
        :ref:`sphx_glr__auto_examples_get_style.py`
    """
    if index is not None and ax is not None:
        raise ValueError("Cannot specify both index and ax. Please provide only one.")
    
    # index is provided
    if index is not None:
        if not isinstance(index, int):
            raise TypeError("Style index must be integer.")
        n = len(styles)
        if not (0 <= index < n):
            raise IndexError(f"Style index out of range [0, {n-1}].")
        ax = plt.gca()
        ax._get_lines.get_next_color()  # Advance the color cycler to keep it in sync with the style index
        return cast(PlotStyle, styles[index].copy())
    
    # ax is provided        
    if ax is not None:
        color = ax._get_lines.get_next_color()
        try:
            linestyle = _get_linestyle_for_color(color)
        except ValueError:
            linestyle = "-"

        return cast(PlotStyle, {"color": color, "linestyle": linestyle})

    raise ValueError("Either index or ax must be provided.")  

# ___________________________________________________________________
#  Custom Markers


# Minimal flip table: only define one direction
_FLIPPED_MARKERS_BASE = {
    '^': 'v',  # up triangle → down triangle
    '>': '<',  # right triangle → left triangle
}

# Auto-generated full flip table (bidirectional mapping)
FLIPPED_MARKERS = {
    **_FLIPPED_MARKERS_BASE,
    **{v: k for k, v in _FLIPPED_MARKERS_BASE.items()}
}


def _is_directional_marker(marker: str) -> bool:
    """Check if a marker is directional and can be flipped.

    Determines whether a marker symbol has directional meaning (e.g., triangles)
    and exists in the ``FLIPPED_MARKERS`` mapping. Used to validate marker
    choices for plots with directional emphasis.

    Args:
        marker (str): Single-character marker symbol to check (e.g., '^', 'v',
            'o', 's').

    Returns:
        bool: True if the marker is directional and has a defined flip, False
            otherwise.
    """
    if not isinstance(marker, str) or len(marker) != 1:
        raise ValueError(
            f"Marker must be a single-character string, got {marker!r}"
        )
    return marker in FLIPPED_MARKERS
