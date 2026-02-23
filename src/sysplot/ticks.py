import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter, Locator
from matplotlib.axis import XAxis, YAxis
import numpy as np
import math

import warnings
from typing import Union, Literal, cast

# ___________________________________________________________________
#  Custom Minor Axis Ticks


def set_minor_log_ticks(
    axis: Union[None, XAxis, YAxis] = None, 
    tick_direction: str = "in", 
    base: float = 10.0
) -> None:
    """Add minor ticks to a logarithmic axis for enhanced readability.

    Places unlabeled minor tick marks between major ticks (usually decades) on a
    logarithmic scale. Tick marks use the same color as grid lines for
    visual consistency.

    Note:
        If major ticks are not at every decade, minor ticks are still placed
        at decade intervals, not between major ticks. This behavior may be
        adjusted in future versions.

    Args:
        axis (XAxis | YAxis, optional): Axis to modify. If None, uses the
            current x-axis (``plt.gca().xaxis``). Use ``ax.xaxis`` or
            ``ax.yaxis`` when supplying an axis.
        tick_direction (str, optional): Direction of minor tick marks.
            One of ``"in"``, ``"out"``, or ``"inout"``. Default is ``"in"``.
        base (float, optional): Logarithm base for the axis scale.
            Default is 10.0.

    Raises:
        TypeError: If axis is not XAxis or YAxis.
        ValueError: If axis is not logarithmically scaled.
        ValueError: If tick_direction is not valid.
        ValueError: If base is <= 1.

    Example:
        >>> fig, ax = plt.subplots()
        >>> ax.set_xscale('log')
        >>> ax.plot([1, 10, 100], [1, 2, 3])
        >>> set_minor_log_ticks(ax.xaxis)
        
        >>> # Both axes with outward ticks
        >>> set_minor_log_ticks(ax.xaxis, tick_direction="out")
        >>> set_minor_log_ticks(ax.yaxis, tick_direction="out")
    """
    if axis is None:
        axis = plt.gca().xaxis

    if not isinstance(axis, (XAxis, YAxis)):
        raise TypeError(f"'axis' must be XAxis or YAxis, got {type(axis).__name__}")
    if axis.get_scale() != "log":
        raise ValueError("set_minor_log_ticks() requires a log-scaled axis.")
    if tick_direction not in ("in", "out", "inout"):
        raise ValueError(f"tick_direction must be 'in', 'out', or 'inout', got {tick_direction!r}")
    if base <= 1.0:
        raise ValueError(f"logarithm base must be > 1, got {base}")

    tick_color = mpl.rcParams['grid.color']
    axis_name: Literal["x", "y"] = cast(Literal["x", "y"], axis.axis_name)
    if axis_name not in ("x", "y"):
        raise ValueError(f"axis_name must be 'x' or 'y', got {axis_name!r}")

    # Configure minor tick locator
    axis.set_minor_locator(LogLocator(
        base=base,
        subs="all",  # Place ticks at all subdivisions (controls decade behavior)
        numticks=30  # Maximum number of minor ticks (prevents overcrowding)
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

    Raises:
        ValueError: If mode is not recognized.
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


class FractionalLocator(Locator):
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

    def __call__(self):
        """Compute tick locations for the associated axis.

        Automatically refines the denominator if initial tick spacing would
        result in fewer than 2 visible ticks.

        Returns:
            np.ndarray: Array of tick positions.

        Raises:
            ValueError: If too many ticks would be placed (> MAXTICKS).
            RuntimeError: If maximum denominator is reached without finding
                a valid tick configuration.
        """
        axis = self.axis
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
                return ticks
            
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
    locator: callable, 
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

    Note:
        Fractions are automatically reduced using ``math.gcd``.
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
    """Ensure text is wrapped in LaTeX math mode delimiters.

    Checks if the input string is already wrapped in `$...$` and adds
    delimiters if missing. Issues a warning when auto-wrapping occurs.

    Args:
        text (str): Text to check and potentially wrap.

    Returns:
        str: Text guaranteed to be wrapped in `$...$`.

    Example:
        >>> _ensure_latex_math("\\pi")
        '$\\pi$'  # Warning issued
        >>> _ensure_latex_math("$\\pi$")
        '$\\pi$'  # No warning
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
    numerator: int = 1,
    denominator: int = 1,
    mode: Literal["single", "symmetric", "repeating"] = "repeating",
    axis: Union[None, XAxis, YAxis] = None,
) -> None:
    r"""Set custom major ticks with LaTeX-formatted fractional labels.

    Configures an axis to display ticks at integer or fractional multiples
    of a physical unit (e.g., π, time constants, wavelengths). Fractions
    are automatically reduced and formatted in LaTeX math mode.

    The function supports three placement modes, where ``step=unit * numerator / denominator``:

    - **"single"**: Ticks at ``[0, +step]``  
    - **"symmetric"**: Ticks at ``[-step, 0, +step]``  
    - **"repeating"**: Ticks at all multiples of ``step`` within axis limits  

    If insufficient ticks fall within the visible range, the denominator is
    automatically doubled until at least two ticks are visible or a maximum
    denominator is reached.

    Args:
        label (str): Base label for tick formatting (e.g., ``"\\pi"``, ``"t"``). 
            If not wrapped in ``$...$``, it will be automatically LaTeX-wrapped
            with a warning.
        unit (float, optional): Physical value corresponding to one label unit.
            For example, use ``np.pi`` for π-based axes. Default is 1.0.
        numerator (int, optional): Numerator for fractional step size.
            Must be positive. Default is 1.
        denominator (int, optional): Denominator for fractional step size.
            Must be positive. Default is 1.
        mode (Literal["single", "symmetric", "repeating"], optional): 
            Tick placement strategy. Default is ``"repeating"``.
        axis (XAxis | YAxis, optional): Axis to modify. If None, uses the
            current x-axis (``plt.gca().xaxis``). Use ``ax.xaxis`` or
            ``ax.yaxis`` when supplying an axis.

    Raises:
        ValueError: If unit is zero.
        ValueError: If numerator or denominator is not positive.
        ValueError: If mode is not one of the allowed values.
        TypeError: If axis is not XAxis or YAxis.

    Note:
        - Fractions are automatically reduced using ``math.gcd``.
        - Formatter uses exact fractional arithmetic to avoid floating-point
          errors in label placement.
        - 3D axes (ZAxis) are not yet supported.

    Example:
        >>> # Phase axis with π/4 increments
        >>> fig, ax = plt.subplots()
        >>> ax.plot(omega, phase)
        >>> set_major_tick_labels(
        ...     label=r"\pi", 
        ...     unit=np.pi, 
        ...     denominator=4,
        ...     axis=ax.yaxis
        ... )
        
        >>> # Time axis with 0.5s increments
        >>> set_major_tick_labels(
        ...     label="s", 
        ...     unit=1.0, 
        ...     numerator=1, 
        ...     denominator=2,
        ...     axis=ax.xaxis
        ... )
    """
    # Validate inputs
    if unit == 0:
        raise ValueError("unit must not be zero")
    if numerator <= 0 or denominator <= 0:
        raise ValueError("numerator and denominator must be positive integers")
    if mode not in ("single", "symmetric", "repeating"):
        raise ValueError(f"mode must be 'single', 'symmetric', or 'repeating', got {mode!r}")
    if axis is None:
        axis = plt.gca().xaxis
    if not isinstance(axis, (XAxis, YAxis)):
        raise TypeError(f"'axis' must be XAxis or YAxis, got {type(axis).__name__}")

    # Create and set custom locator
    locator = FractionalLocator(
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
