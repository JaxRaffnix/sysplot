import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter, Locator
from matplotlib.axis import XAxis, YAxis, Axis
import numpy as np
import math

import warnings
from typing import Callable, Sequence, Literal, cast


# ___________________________________________________________________
#  Custom Minor Axis Ticks


def set_minor_log_ticks(
    axis: XAxis | YAxis | None = None,
    tick_direction: Literal["in", "out", "inout"] | None = None,
    base: float = 10.0,
) -> None:
    """Add minor ticks to a logarithmic axis.

    Places unlabeled minor ticks between decades for readability and styles
    them to match the grid color. 

    Note:
        If major ticks are not at every decade, minor ticks are still placed at decade intervals, not between major ticks. This behavior may be adjusted in future versions.

    Args:
        axis: Axis to modify. Defaults to the current x-axis.
        tick_direction: Tick direction ("in", "out", or "inout"). If None,
            uses Matplotlib rcParams for the selected axis.
        base: Logarithm base for the axis scale. If None, defaults to 10.0.

    Example:
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> import sysplot as ssp
        >>>
        >>> x = np.logspace(0, 3, 200)
        >>> y = 1 / x
        >>> fig, ax = plt.subplots()
        >>> ax.set_xscale("log")
        >>> ax.plot(x, y)
        >>> ssp.set_minor_log_ticks(ax.xaxis)
        >>> plt.show()
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
    if tick_direction is None:
        tick_direction = mpl.rcParams[f"{axis_name}tick.direction"]
    if tick_direction not in ("in", "out", "inout"):
        raise ValueError(f"tick_direction must be 'in', 'out', or 'inout', got {tick_direction!r}")
    if not np.isfinite(base) or base <= 1.0:
        raise ValueError(f"logarithm base must be > 1, got {base!r}")

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
    axis_max: float
) -> np.ndarray:
    """Compute tick positions based on step size, mode, and axis limits.

    Internal helper for calculating where major ticks should be placed
    based on the requested spacing and placement strategy.

    Args:
        step (float): Spacing between consecutive ticks.
        mode (str): Placement strategy - one of:
            - ``"repeating"``: Ticks at all multiples of step within limits
            - ``"single"``: Ticks at [0, step]
            - ``"symmetric"``: Ticks at [-step, 0, step]
        axis_min (float): Current axis minimum value.
        axis_max (float): Current axis maximum value.

    Returns:
        np.ndarray: Array of tick positions that fall inside axis limits.
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
    """Matplotlib tick locator for fractional unit spacing with adaptive refinement.

    Computes tick positions based on fractional multiples of a unit value.
    If too few ticks fall within the axis range, automatically doubles the
    denominator until sufficient ticks are visible or a maximum is reached.

    This is particularly useful for axes labeled with fractions of π, time
    units, or other physical quantities.

    Attributes:
        unit (float): Physical value corresponding to one label unit.
        numerator (int): Numerator of the fractional step.
        denominator (int): Denominator of the fractional step.
        mode (str): Tick placement mode ("repeating", "single", "symmetric").
        MAX_DEN (int): Maximum denominator before raising an error (2048).
    """
    def __init__(
        self, 
        unit: float = 1.0, 
        numerator: int = 1, 
        denominator: int = 1, 
        mode: str = "repeating"
    ):
        """Initialize the fractional locator.

        Args:
            unit (float, optional): Physical value per label unit. Default is 1.0.
            numerator (int, optional): Numerator for step fraction. Default is 1.
            denominator (int, optional): Denominator for step fraction. Default is 1.
            mode (str, optional): Tick placement mode. Default is "repeating".
        """
        self.unit = unit
        self.numerator = numerator
        self.denominator = denominator
        self.mode = mode
        self.MAX_DEN = 2048

    def __call__(self) -> Sequence[float]:
        """Compute tick locations for the associated axis.

        Automatically refines the denominator if initial tick spacing would
        result in fewer than 2 visible ticks.

        Returns:
            np.ndarray: Array of tick positions.
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
    mode: str
):
    """Create a formatter for fractional tick labels.

    Returns a Matplotlib formatter callable that converts numeric tick values
    into LaTeX-formatted fractional labels (e.g., "π/4", "3π/2").

    Args:
        label (str): Base label text (e.g., "$\\pi$" or "$t$").
        unit (float): Physical value corresponding to one label unit.
        denominator (int): Denominator used for fractional step calculation.
        locator (callable): Locator callable to dynamically fetch current ticks.
        mode (str): Tick placement mode ("repeating", "single", "symmetric").

    Returns:
        callable: Formatter function with signature ``(value, pos) -> str``.
    """
    def formatter(value, pos):
        """Format a tick value as a fractional label.

        Args:
            value (float): Numeric tick position.
            pos (int): Tick index (unused but required by Matplotlib).

        Returns:
            str: LaTeX-formatted label string or empty string if tick
                should not be labeled.
        """
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
    """Ensure text is wrapped in LaTeX math mode delimiters `$`.
    Args:
        text (str): Text to check and potentially wrap.

    Returns:
        str: Text guaranteed to be wrapped in `$...$`.
    """
    if isinstance(text, str) and not (text.startswith("$") and text.endswith("$")):
        warnings.warn(
            f"Auto-wrapped '{text}' with $...$ for LaTeX math mode.",
            UserWarning
        )
        text = f"${text}$"

    return text


def set_major_tick_labels(
    label: str,
    unit: float = 1.0,
    axis: XAxis | YAxis | None = None,
    mode: Literal["single", "symmetric", "repeating"] = "repeating",
    numerator: int = 1,
    denominator: int = 1,
) -> None:
    r"""Set major ticks with fractional labels.

    Formats tick labels as multiples of a unit (for example, π). The function
    places ticks at ``step = unit * numerator / denominator`` spacing. Labels are
    formatted in LaTeX math mode and reduced to simplest fractions.

    Ticks are placed according to ``mode``:
        - ``"single"``: ticks at ``0`` and ``step`` only.
        - ``"symmetric"``: ticks at ``-step``, ``0``, and ``step``.
        - ``"repeating"``: ticks at all integer multiples of ``step`` within
          the visible axis limits.

    Note:
        If insufficient ticks fall within the visible range, the denominator is
        automatically doubled until at least two ticks are visible or a maximum
        denominator is reached.

    Args:
        label: Base label text (e.g., ``r"\pi"``). If not wrapped in
            ``$...$``, it is auto-wrapped with a warning.
        unit: Physical value corresponding to one label unit
            (e.g., ``np.pi``). Must be finite and non-zero.
        axis: Axis to modify. Defaults to the current x-axis.
        mode: Tick placement strategy: ``"single"``, ``"symmetric"``, or
            ``"repeating"``.
        numerator: Numerator for the fractional step. Must be a positive int.
        denominator: Denominator for the fractional step. Must be a positive int.

    Example:
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> import sysplot as ssp
        >>>
        >>> x = np.linspace(0, 2 * np.pi, 200)
        >>> y = np.sin(x)
        >>> fig, ax = plt.subplots()
        >>> ax.plot(x, y)
        >>> ssp.set_major_tick_labels(label=r"\pi", unit=np.pi, denominator=2, axis=ax.xaxis)
        >>> ax.set_xlabel(r"$t$ [rad]")
        >>> plt.show()
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


def add_second_tick(
    value: float,
    label: str,
    axis: XAxis | YAxis | None = None,
    color: str | None = None,
    linewidth: float | None = None,
    fontsize: float | None = None,
    offset: float = 0.0,
) -> None:
    """Add a labeled reference tick without changing the major tick locator.

    Draws a dotted reference line at ``value`` and places a text label offset
    from the axis. Works for both linear and log axes.

    Note:
        This is useful when you want to highlight a specific value (e.g., resonance frequency, time constant) without altering the existing major tick configuration.

    Args:
        value: Tick position in data coordinates.
        label: Text label to display at the tick.
        axis: Target axis (``ax.xaxis`` or ``ax.yaxis``). Defaults to current x-axis.
        color: Line and text color. Defaults to grid color.
        linewidth: Line width. Defaults to ``1.5 * rcParams['grid.linewidth']``.
        fontsize: Font size for label. Defaults to ``rcParams['font.size']``.
        offset: Fractional offset in axis coordinates. Negative offsets place
            labels below (x-axis) or left (y-axis).
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

    color = color or str(mpl.rcParams["grid.color"])
    linewidth = linewidth if linewidth is not None else 1.5 * mpl.rcParams["grid.linewidth"]
    fontsize = fontsize if fontsize is not None else mpl.rcParams["font.size"]

    ax = axis.axes  # get the parent axes

    if isinstance(axis, XAxis):     # vertical line at x=value
        ax.axvline(value, color=color, linestyle=":", linewidth=linewidth, zorder=0)
        ax.text(
            x=value,
            y=offset,
            s=label,
            fontsize=fontsize,
            va="bottom",
            ha="center",
            transform=ax.get_xaxis_transform(),
        )
    elif isinstance(axis, YAxis):       # horizontal line at y=value
        ax.axhline(value, color=color, linestyle=":", linewidth=linewidth, zorder=0)
        ax.text(
            x=offset,
            y=value,
            s=label,
            fontsize=fontsize,
            va="center",
            ha="left",
            transform=ax.get_yaxis_transform(),
        )
    