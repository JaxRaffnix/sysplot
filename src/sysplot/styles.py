import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.axes import Axes
from cycler import cycler

from typing import cast

# TODO: implement default color palette?
# TODO implement default gray filled area with transparency?

# Global style configuration
COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']
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
if len(COLORS) < len(LINE_STYLES):
    raise ValueError(
        f"Not enough colors in the palette for the number of line styles. "
        f"Expected at least {len(LINE_STYLES)} colors, got {len(COLORS)}."
    )

# Build and apply custom style cycler
_custom_cycler = cycler(color=COLORS) + cycler(linestyle=LINE_STYLES)
_styles = list(_custom_cycler)
mpl.rcParams['axes.prop_cycle'] = _custom_cycler


def get_style(index: int) -> dict[ str, str | tuple[int, ...]]:
    """Return a (color, linestyle) dictionary for a given style index.

    Retrieves plotting style attributes from the internal style cycle. Useful
    for manually controlling plot appearance or synchronizing styles across
    multiple plotting calls.

    Args:
        index (int): Index into the internal style list. Must be a non-negative
            integer within the valid range ``[0, len(_styles)-1]``.

    Returns:
        dict[str, str | tuple[int, ...]]: Dictionary with keys:
            - 'color' (str): Matplotlib color string (e.g., "C0", "#1f77b4", "red").
            - 'linestyle' (str | tuple[int, ...]): Line style, either a standard 
              string ("-", "--", ":") or a custom dash pattern tuple.

    Raises:
        TypeError: If index is not an integer.
        IndexError: If index is outside the valid range.

    Example:
        >>> plt.plot([1, 2, 3], [1, 4, 9], **get_style(0))
        
        >>> # Get multiple styles
        >>> for i in range(3):
        ...     plt.plot([1, 2, 3], [i, i+1, i+2], **get_style(i))
    """
    if not isinstance(index, int):
        raise TypeError(f"Style index must be an integer, got {type(index).__name__}.")
    
    if index < 0 or index >= len(_styles):
        raise IndexError(
            f"Style index {index} out of range [0, {len(_styles)-1}]."
        )
    
    style = _styles[index]
    return {
        "color": style["color"],
        "linestyle": style["linestyle"],
    }


def _get_style_then_advance(index: int | None, ax: Axes) ->  dict[ str, str | tuple[int, ...]]:
    """Get style and advance the per-axes style counter.

    Internal utility that manages style cycling on a per-axes basis. Each axes
    object maintains its own style counter to ensure consistent progression
    even when mixing automatic and manual style selection.

    The counter advances regardless of whether ``index`` is provided (manual)
    or ``None`` (automatic from counter).

    Args:
        index (int | None): Desired style index. If None, uses the next style
            from the axes' internal counter. If provided, wraps to valid range.
        ax (Axes): Matplotlib axes object that stores the style counter.

    Returns:
        dict[str, str | tuple[int, ...]]: Dictionary with keys:
            - 'color' (str): Matplotlib color string (e.g., "C0", "#1f77b4", "red").
            - 'linestyle' (str | tuple[int, ...]): Line style, either a standard 
              string ("-", "--", ":") or a custom dash pattern tuple.

    Note:
        This function mutates the axes object by setting ``ax._style_idx``.
    """
    # Get current counter value (default to -1 if not yet set)
    current = getattr(ax, "_style_idx", -1)
    
    if index is None:
        # Automatic: advance from current position
        style_index = (current + 1) % len(_styles)
    else:
        # Manual: use provided index with wraparound safety
        style_index = index % len(_styles)
    
    # Update axes-local counter for next call
    setattr(ax, "_style_idx", style_index)

    # Ensure type checker sees an int (index could have been None initially)
    style_index = cast(int, style_index)
    return get_style(style_index)


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
