import matplotlib as mpl
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
#? implement default color palette?
#? implement default gray filled value with transparency for areas?

# TODO: add default arrows style

# Global style configuration
_DEFAULT_COLORS = mpl.rcParamsDefault['axes.prop_cycle'].by_key()['color']
"""list[str]: Color palette extracted from current Matplotlib color cycle."""

LINE_STYLES = [
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
"""list: Line style patterns for plot cycling.

Each entry is either a string shorthand ('-', '--', etc.) or a tuple defining
custom dash patterns as (offset, (on, off, on, off, ...)).
"""

# Validate configuration
if len(_DEFAULT_COLORS) != len(LINE_STYLES):
    raise ValueError(f"Color palette length ({len(_DEFAULT_COLORS)}) does not match number of line styles ({len(LINE_STYLES)}).")

# Build and apply custom style cycler
_custom_cycler = cycler(color=_DEFAULT_COLORS) + cycler(linestyle=LINE_STYLES)
_styles = list(_custom_cycler)
mpl.rcParams['axes.prop_cycle'] = _custom_cycler


def _get_linestyle_for_color(color):
    target = mpl.colors.to_rgba(color)
    for style in _styles:
        style_color = mpl.colors.to_rgba(style["color"])
        if style_color == target:
            return style["linestyle"]
    raise ValueError(f"Color {color} not found in custom cycler.")


def get_next_style(ax, index=None) -> PlotStyle:
    """return the style for the next plot element and advances the cycler."""
    # TODO: find better name, update docstring

    if index is not None:
        #TODO: potentially advance the internal color cycler even for manual index selection, to keep the style cycling consistent with the plt.plot() behavrior.
        # ax._get_lines.get_next_color()    
        return get_style(index)

    # index is manually passed
    color = ax._get_lines.get_next_color()
    try:
        linestyle = _get_linestyle_for_color(color)
    except ValueError:
        # If no matching linestyle is found, default to solid line
        linestyle = "-"

    return {"color": color, "linestyle": linestyle}


def get_style(index: int) -> PlotStyle:
    """Get style by index from custom cycler.

    Args:
        index (int): Index of the desired style. Cycles through available styles.

    Returns:
        PlotStyle: A dictionary containing 'color' and 'linestyle' for the given index.
    """
    if not isinstance(index, int):
        raise TypeError("Style index must be integer.")

    n = len(_styles)
    if not (0 <= index < n):
        raise IndexError(f"Style index out of range [0, {n-1}].")

    return cast(PlotStyle, _styles[index].copy())

# class StyleManager:
#     """
#     Stateful style manager for deterministic cycling.

#     Default behavior:
#         - Each call to `next()` advances the internal style index.

#     Manual override:
#         - `get(index)` returns a specific style without advancing.
#     """
#     def __init__(self):
#         self._index = 0
#         self._n = len(_styles)

#     def reset(self) -> None:
#         """Reset internal style index to 0."""
#         self._index = 0

#     def next(self) -> PlotStyle:
#         """
#         Return next style and advance internal index.
#         """
#         style = _styles[self._index]
#         self._index = (self._index + 1) % self._n
#         return cast(PlotStyle, style.copy())

#     def get(self, index: int) -> PlotStyle:
#         """
#         Return specific style without modifying internal index.
#         """

#         # the resulting style is either invoked dynamically or a manual index value is requested by the user.
#         # even for user request the index should be advanced.
#         if not isinstance(index, int):
#             raise TypeError("Style index must be integer.")

#         if not (0 <= index < self._n):
#             raise IndexError(f"Style index out of range [0, {self._n-1}].")

#         return cast(PlotStyle, _styles[index].copy())


# def get_style_manager(ax) -> StyleManager:
#     if not hasattr(ax, "_style_manager"):
#         ax._style_manager = StyleManager()
#     return ax._style_manager


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
"""dict[str, str]: Bidirectional mapping of directional markers to their flipped counterparts.

Used for creating symmetric stem plots where markers point away from a baseline.
Supports: '^' ↔ 'v' and '>' ↔ '<'.

Example:
    >>> FLIPPED_MARKERS['^']
    'v'
    >>> FLIPPED_MARKERS['<']
    '>'
"""


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

    Raises:
        ValueError: If marker is not a single-character string.

    Example:
        >>> _is_directional_marker('^')
        True
        >>> _is_directional_marker('o')  # Circle has no direction
        False
        >>> _is_directional_marker('>')
        True
    """
    if not isinstance(marker, str) or len(marker) != 1:
        raise ValueError(
            f"Marker must be a single-character string, got {marker!r}"
        )
    return marker in FLIPPED_MARKERS
