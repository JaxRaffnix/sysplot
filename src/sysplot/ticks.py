import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter, Locator
from matplotlib.axis import XAxis, YAxis, Axis
import numpy as np
import math

import warnings
from typing import Callable, Sequence, Literal, cast

from .config import get_config


# ___________________________________________________________________
#  Custom Minor Axis Ticks


def set_minor_log_ticks(
    axis: XAxis | YAxis | None = None,
    tick_direction: Literal["in", "out", "inout"] | None = None,
    base: float = 10.0,
) -> None:
    """Add unlabeled minor ticks at every subdivision of a logarithmic axis.

    Places minor ticks between decades and styles them to match the grid color.
    Tick direction defaults to :attr:`~sysplot.SysplotConfig.tick_direction`.

    Note:
        Minor ticks are always placed at decade intervals, regardless of where
        the major ticks fall.

    Args:
        axis: Axis to modify. Defaults to the current x-axis.
        tick_direction: Direction of minor ticks (``"in"``, ``"out"``, or
            ``"inout"``). Defaults to
            :attr:`~sysplot.SysplotConfig.tick_direction`.
        base: Logarithm base for the axis scale. Must be > 1.

    .. minigallery:: sysplot.set_minor_log_ticks
        :add-heading:
    """
    if axis is None:
        axis = plt.gca().xaxis
    if not isinstance(axis, (XAxis, YAxis)):
        raise TypeError(f"'axis' must be XAxis or YAxis, got {type(axis).__name__}")
    if axis.get_scale() != "log":
        raise ValueError("set_minor_log_ticks() requires a log-scaled axis.")
    axis_name: Literal["x", "y"] = cast(Literal["x", "y"], axis.axis_name)
    if axis_name not in ("x", "y"):
        raise ValueError(f"axis_name must be 'x' or 'y', got {axis_name!r}")
    if not np.isfinite(base) or base <= 1.0:
        raise ValueError(f"logarithm base must be > 1, got {base!r}")

    tick_direction = get_config().tick_direction if tick_direction is None else tick_direction
    tick_color = mpl.rcParams["grid.color"]

    # Configure minor tick locator
    axis.set_minor_locator(LogLocator(
        base=base,
        subs="all",  # Place ticks at all subdivisions (controls decade behavior)
        numticks=30,  # Maximum number of minor ticks (prevents overcrowding)
    ))
    axis.set_minor_formatter(NullFormatter())  # No labels for minor ticks

    # TODO: minor ticks at every decade, even if major are at different spacing.

    # Apply tick styling
    axis.axes.tick_params(
        axis=axis_name, 
        which="minor",
        direction=tick_direction,
        color=tick_color,
        bottom=True,
        left=True,    
    )


# ___________________________________________________________________
#  Custom Axis Ticks


def _generate_ticks(
    step: float,
    mode: str,
    axis_min: float,
    axis_max: float,
) -> np.ndarray:
    """Compute tick positions from step size, mode, and axis limits.

    Args:
        step: Spacing between consecutive ticks.
        mode: Placement strategy — ``"repeating"``, ``"single"``, or
            ``"symmetric"``.
        axis_min: Current axis minimum value.
        axis_max: Current axis maximum value.

    Returns:
        Array of tick positions within the axis limits.
    """
    if mode == "repeating":
        # Generate ticks at all multiples of step within axis limits
        t_max = int(np.floor(axis_max / abs(step)))
        t_min = int(np.ceil(axis_min / abs(step)))
        ticks = np.arange(t_min, t_max + 1) * abs(step)     
    elif mode == "single":
        ticks = np.array([0.0, step])
    elif mode == "symmetric":
        ticks = np.array([-step, 0.0, step])
    else:
        raise ValueError(f"Invalid mode: {mode!r}")
    
    # Filter ticks to stay inside axis limits
    mask = (ticks >= axis_min) & (ticks <= axis_max)
    return ticks[mask]


class _FractionalLocator(Locator):
    """Tick locator for fractional unit spacing with adaptive denominator refinement.

    Places ticks at multiples of ``unit * numerator / denominator``. If fewer
    than two ticks fall in the visible range, the denominator is doubled
    automatically until sufficient ticks appear or the maximum is reached.
    """
    def __init__(
        self,
        unit: float = 1.0,
        numerator: int = 1,
        denominator: int = 1,
        mode: str = "repeating",
    ):
        """Initialize the fractional locator.

        Args:
            unit: Physical value per label unit.
            numerator: Numerator for the fractional step.
            denominator: Denominator for the fractional step.
            mode: Tick placement mode.
        """
        self.unit = unit
        self.numerator = numerator
        self.denominator = denominator
        self.mode = mode
        self.MAX_DEN = 2048

    def __call__(self) -> Sequence[float]:
        """Compute tick locations for the associated axis.

        Doubles the denominator if fewer than 2 ticks are visible.

        Returns:
            Array of tick positions.
        """
        axis = self.axis
        if axis is None:
            raise RuntimeError("Locator is not attached to an axis.")
        
        axis_min, axis_max = axis.get_view_interval()
        
        warned = False
        denominator = self.denominator
        numerator = self.numerator
        
        while True:
            step = self.unit * numerator / denominator
            ticks = _generate_ticks(step, self.mode, axis_min, axis_max)

            # Safety check: prevent excessive tick count
            if ticks.size >= Locator.MAXTICKS:
                raise ValueError(
                    f"Too many ticks ({ticks.size}) would be placed on the axis. "
                    "Adjust 'unit', 'numerator', or 'denominator'."
                )
            
            # Found valid solution (at least 2 ticks)
            if ticks.size > 1:
                return ticks.tolist()
            
            # Refine denominator to get more ticks
            if not warned:
                warnings.warn(
                    "Insufficient ticks in axis range. Auto-adjusting denominator...",
                    UserWarning
                )
                warned = True
            denominator *= 2
            
            # Abort if denominator becomes unreasonably large
            if denominator > self.MAX_DEN:
                raise RuntimeError(
                    f"Failed to place multiple ticks: maximum denominator "
                    f"({self.MAX_DEN}) reached. Check axis limits and unit settings."
                )


def _get_formatter(
    label: str,
    unit: float,
    denominator: int,
    locator: Callable[[], Sequence[float]],
    mode: str,
) -> Callable:
    """Create a FuncFormatter for fractional tick labels.

    Converts numeric tick values into LaTeX-formatted fractional labels
    (e.g., ``π/4``, ``3π/2``).

    Args:
        label: Base label text (e.g., ``"$\\pi$"``).
        unit: Physical value corresponding to one label unit.
        denominator: Denominator used for the fractional step.
        locator: Locator callable to dynamically fetch current tick positions.
        mode: Tick placement mode.

    Returns:
        Formatter callable with signature ``(value, pos) -> str``.
    """
    def formatter(value, pos):
        """Format a single tick value as a reduced fraction label."""
        ticks = locator()  # Recompute ticks dynamically

        # In non-repeating modes, only label explicitly placed ticks
        if mode != "repeating":    
            if value not in ticks:
                return ""
            
        # Convert value to fractional representation
        frac = value / unit
        numerator = round(frac * denominator)
        
        # Skip labels for non-exact fractional values (floating-point safety)
        if not np.isclose(frac * denominator, numerator):
            return ""
        
        # Reduce fraction to simplest form
        gcd = math.gcd(numerator, denominator)
        numerator //= gcd
        denominator_scaled = denominator // gcd
        
        # Format label based on numerator/denominator values
        if numerator == 0:
            return "0"
        if denominator_scaled == 1:
            if numerator == 1:
                return f"{label}"
            if numerator == -1:
                return f"-{label}"
            return f"{numerator} {label}"
        return f"{numerator}/{denominator_scaled} {label}"
    
    return formatter


def _ensure_latex_math(text: str) -> str:
    """Ensure text is wrapped in LaTeX math mode delimiters ``$``.

    Args:
        text: Text to check and potentially wrap.

    Returns:
        Text guaranteed to be wrapped in ``$...$``.
    """
    if isinstance(text, str) and not (text.startswith("$") and text.endswith("$")):
        warnings.warn(
            f"Auto-wrapped '{text}' with $...$ for LaTeX math mode.",
            UserWarning
        )
        text = f"${text}$"

    return text


def set_major_ticks(
    label: str,
    unit: float = 1.0,
    axis: XAxis | YAxis | None = None,
    mode: Literal["single", "symmetric", "repeating"] = "repeating",
    numerator: int = 1,
    denominator: int = 1,
) -> None:
    r"""Set major tick labels as reduced fractions of a unit.

    Places ticks at ``step = unit * numerator / denominator`` and formats
    labels as LaTeX fractions (e.g., ``π/2``, ``3π/4``). Labels are reduced
    to their simplest form automatically.

    The ``mode`` controls where ticks are placed:

    - ``"repeating"``: all integer multiples of ``step`` within the axis range.
    - ``"single"``: only ``0`` and ``step``.
    - ``"symmetric"``: only ``-step``, ``0``, and ``step``.

    If the label text is not already wrapped in ``$...$``, it is auto-wrapped
    with a warning. 

    Note:
        If insufficient ticks fall within the visible range, the denominator is
        automatically increased until at least two ticks are visible or a maximum
        denominator is reached.

    Args:
        label: Base label text (e.g., ``r"\pi"`` or ``"$\\pi$"``).
        unit: Physical value for one label unit (e.g., ``np.pi``). Must be
            finite and non-zero.
        axis: Axis to modify. Defaults to the current x-axis.
        mode: Tick placement strategy.
        numerator: Numerator of the fractional step. Must be a positive int.
        denominator: Denominator of the fractional step. Must be a positive int.

    .. minigallery:: sysplot.set_major_ticks
        :add-heading:
    """
    if not isinstance(label, str):
        raise TypeError(f"label must be a string, got {type(label)}")
    if not np.isfinite(unit) or unit == 0:
        raise ValueError(f"unit must be a finite non-zero number, got {unit!r}")
    if not isinstance(numerator, int) or not isinstance(denominator, int):
        raise TypeError("numerator and denominator must be integers")
    if numerator <= 0 or denominator <= 0:
        raise ValueError("numerator and denominator must be positive integers")
    if mode not in ("single", "symmetric", "repeating"):
        raise ValueError(f"mode must be 'single', 'symmetric', or 'repeating', got {mode!r}")
    if axis is None:
        axis = plt.gca().xaxis
    if not isinstance(axis, (XAxis, YAxis)):
        raise TypeError(f"'axis' must be XAxis or YAxis, got {type(axis).__name__}")

    # Create and set custom locator
    locator = _FractionalLocator(
        unit=unit, 
        numerator=numerator, 
        denominator=denominator, 
        mode=mode
    )
    axis.set_major_locator(locator)

    # Create and set custom formatter
    label = _ensure_latex_math(label)
    formatter = _get_formatter(label, unit, denominator, locator, mode)
    axis.set_major_formatter(FuncFormatter(formatter))


# ___________________________________________________________________
#  Manaully Add a Tick


def add_tick_line(
    value: float,
    label: str,
    axis: XAxis | YAxis | None = None,
    color: str | None = None,
    linewidth: float | None = None,
    fontsize: float | None = None,
    offset: float = 0.0,
) -> None:
    """Draw a dotted reference line and label at a specific axis value.

    Adds a labeled tick at ``value`` without modifying the existing major tick
    locator. Useful for highlighting key values such as resonance frequencies
    or time constants.

    Args:
        value: Tick position in data coordinates.
        label: Text to display at the tick.
        axis: Target axis (``ax.xaxis`` or ``ax.yaxis``). Defaults to the
            current x-axis.
        color: Line and text color. Defaults to ``rcParams['text.color']``.
        linewidth: Reference line width. Defaults to
            ``rcParams['grid.linewidth']``.
        fontsize: Label font size. Defaults to ``rcParams['font.size']``.
        offset: Fractional offset along the perpendicular axis for the label
            (axis coordinates). Negative values place the label below/left.

    .. minigallery:: sysplot.add_tick_line
        :add-heading:
    """
    if axis is None:
        axis = plt.gca().xaxis
    if not isinstance(axis, (XAxis, YAxis)):
        raise TypeError(f"axis must be ax.xaxis or ax.yaxis, got {type(axis)}")
    if not isinstance(label, str):
        raise TypeError(f"label must be a string, got {type(label)}")
    if not np.isfinite(value):
        raise ValueError(f"value must be a finite number, got {value!r}")
    if linewidth is not None:
        if linewidth <= 0:
            raise ValueError(f"linewidth must be a positive number, got {linewidth!r}")
    if fontsize is not None:
        if fontsize <= 0:
            raise ValueError(f"fontsize must be a positive number, got {fontsize!r}")
        
    color = color or mpl.rcParams["text.color"]
    linewidth = linewidth if linewidth is not None else mpl.rcParams["grid.linewidth"]
    fontsize = fontsize if fontsize is not None else mpl.rcParams["font.size"]

    ax = axis.axes

    if isinstance(axis, XAxis):
        ax.axvline(value, color=color, linestyle=":", linewidth=linewidth, zorder=0)
        ax.text(
            x=value,
            y=offset,
            s=label,
            color=color,
            fontsize=fontsize,
            va="bottom",
            ha="center",
            transform=ax.get_xaxis_transform(),
        )
    elif isinstance(axis, YAxis):
        ax.axhline(value, color=color, linestyle=":", linewidth=linewidth, zorder=0)
        ax.text(
            x=offset,
            y=value,
            s=label,
            color=color,
            fontsize=fontsize,
            va="center",
            ha="left",
            transform=ax.get_yaxis_transform(),
        )
    