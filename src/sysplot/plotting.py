import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from typing import Union

from .config import MARKERSIZE, ARROWSTYLE, POLES_ZEROS_MARKERSIZE
from .styles import PlotStyle, _get_next_style, _is_directional_marker, FLIPPED_MARKERS
from .axes import add_origin, set_xmargin
from .ticks import set_major_tick_labels, set_minor_log_ticks

from .config import FIGURE_SIZE


# ___________________________________________________________________
#  Pole-Zero Plot

def plot_poles_zeros(
    poles: Union[complex, list[complex], np.ndarray, None] = None,
    zeros: Union[complex, list[complex], np.ndarray, None] = None,
    label: str | None = None,
    ax: Axes|None = None,
    style_index: int | None = None,
    markersize: float = POLES_ZEROS_MARKERSIZE,
    show_origin: bool = True,
    enable_xmargin: bool = True
) -> None:
    """
    Plot poles and zeros on a complex plane.

    This function visualizes the poles and zeros on a complex plane diagram. Poles are marked with 'x' markers and zeros with
    hollow circles. If show_origin is True, the origin (0, 0) is marked with a transparent point to ensure it is visible in the plot.

    Args:
        poles (np.ndarray): Array of complex pole locations. Can be empty if there are no poles to plot.
        zeros (np.ndarray): Array of complex zero locations. Can be empty if there are no zeros to plot.
        label (str | None, optional): Label for the poles and zeros in the legend. Default is None.
        ax (Axes | None, optional): Matplotlib axes object to plot on. If None, uses the current axes from plt.gca(). Default is None.
        style_index (int, optional): Index for selecting line style and color from the style cycle. Default is 0.
        markersize (float, optional): Size of the markers for poles and zeros. Default is MARKERSIZE.
        show_origin (bool, optional): If True, shows the origin (0, 0) in the plot. Default is True.
        enable_xmargin (bool, optional): If True, enables automatic x-axis margin to ensure poles/zeros near the edges are fully visible. Default is True.

    Returns:
        None

    Example:
        >>> poles = np.array([-0.5 + 0.5j, -0.5 - 0.5j])
        >>> zeros = -1
        >>> fig, ax = plt.subplots()
        >>> plot_poles_zeros(poles, zeros, ax=ax)    

        Plot multiple subplots with shared axes:

        >>> fig, axes = plt.subplots(1, 3, sharex=True, sharey=True)
        >>> poles = [[1 + 1j, 1 - 1j], [2 + 1j, 2 - 1j], [3 + 3j, -3 - 3j]]
        >>> for i, (ax, pole) in enumerate(zip(axes, poles)):
        ...     plot_poles_zeros(poles=pole, ax=ax, style_index=i, show_origin=True)
        ...     ax.set_xlabel("Real")
        ...     ax.set_ylabel("Imaginary")
        ...     ax.set_title(f"System {i}")
        ... set_symmetric_axis_limits(axes[2].xaxis, margin=None) # manually select the axis with the largest range
    """
    if ax is not None and not isinstance(ax, Axes):
        raise TypeError(f"ax must be a matplotlib Axes object, got {type(ax)}")
    if markersize <= 0:
        raise ValueError(f"markersize must be positive, got {markersize}")

    poles = np.atleast_1d(poles) if poles is not None else np.array([])
    zeros = np.atleast_1d(zeros) if zeros is not None else np.array([])

    if poles.size == 0 and zeros.size == 0:
        raise ValueError("At least one of poles or zeros must be non-empty to plot.")

    if ax is None:
        ax = plt.gca()

    if style_index is not None and (not isinstance(style_index, int) or style_index < 0):
        raise ValueError(f"style_index must be a non-negative integer or None, got {style_index}")
    
    style = _get_next_style(ax, style_index)

    # poles
    if poles.size > 0:
        ax.scatter(
            np.real(poles),
            np.imag(poles),
            marker='x',
            **style,
            s=markersize**2,
            label=label
        )

    # zeros
    if zeros.size > 0:
        ax.scatter(
            np.real(zeros),
            np.imag(zeros),
            marker='o',
            facecolors='none',
            **style,
            s=markersize**2,
            label=label
        )    

    if show_origin:
        add_origin(ax)

    if enable_xmargin:
        set_xmargin(ax, use_margin=enable_xmargin)


# ___________________________________________________________________
#  Stem Plot

def plot_stem_segment(x, y, ax, bottom, label, marker, markersize, show_baseline, style: PlotStyle):
    color, linestyle = style["color"], style["linestyle"]

    markerline, stemline, baseline = ax.stem(x, y, bottom=bottom, label=label)
    plt.setp(markerline, color=color, marker=marker, markersize=markersize)
    plt.setp(stemline, color=color, linestyle=linestyle)
    if show_baseline:
        plt.setp(baseline, color=color, linestyle=linestyle)
    else:
        baseline.set_visible(False)

    return markerline, stemline, baseline


def plot_stem(
    x,
    y,
    ax=None,
    label=None,
    bottom: float = 0,
    marker="o",
    markersize=MARKERSIZE,
    show_baseline=False,
    style_index: None|int=None,
    markers_outwards=False,
):
    """Plot a styled stem plot with optional outward-pointing markers.

    This is a convenience wrapper around ``Axes.stem`` that:
      * Selects color and linestyle from a style cycle (either Matplotlib's
        internal cycle or a custom style from ``_styles``).
      * Optionally flips directional markers below the baseline so that they
        point away from it (e.g. '^' above, 'v' below).
      * Optionally draws or hides the baseline.

    When ``markers_outwards`` is ``False``, all stems are drawn with the same
    marker. When ``True``, the data is split into values above and below
    ``bottom``. Stems above use ``marker``, while stems below use the flipped
    marker from ``FLIPPED_MARKERS``.

    Note:
        This function relies on the private Matplotlib API
        ``ax._get_lines.get_next_color()`` to retrieve the style cycle when
        ``style_index`` is ``None``. This may break in future Matplotlib
        versions.

    Args:
        x (array-like): X-coordinates of the stems. Must have the same length
            as ``y``.
        y (array-like): Y-values of the stems.
        ax (matplotlib.axes.Axes | None, optional): Axes to plot on. If
            ``None``, uses the current axes returned by ``plt.gca()``.
        label (str | None, optional): Label for the plotted data, used in the
            legend. Only applied to the "up" stems.
        bottom (float, optional): Baseline value from which stems originate.
            Default is ``0``.
        marker (str, optional): Matplotlib marker style for the "up" stems
            (e.g. ``"o"``, ``"^"``, ``"v"``). Default is ``"o"``.
        markersize (float, optional): Size of the markers. Default is ``6``.
        show_baseline (bool, optional): If ``True``, the baseline returned by
            ``Axes.stem`` is styled and shown; otherwise it is hidden.
            Default is ``False``.
        style_index (int | None, optional): Optional index into the internal
            ``_styles`` list. If ``None``, the color is taken from
            ``ax._get_lines.get_next_color()`` and the linestyle is determined
            by ``_get_linestyle_for_color``. Default is ``None``.
        markers_outwards (bool, optional): If ``True`` and ``marker`` is
            directional, markers below ``bottom`` are flipped using
            ``FLIPPED_MARKERS`` so they point away from the baseline.
            Default is ``False``.

    Returns:
        tuple[list, list, list]:
            A 3-tuple ``(markerlines, stemlines, baselines)`` where:

            * ``markerlines`` is a list of Line2D objects for the markers
              (1 element if ``markers_outwards`` is ``False``, 2 elements
              otherwise).
            * ``stemlines`` is a list of LineCollection objects for the stems
              (1 or 2 elements).
            * ``baselines`` is a list of baseline Line2D objects (same length
              as ``markerlines``).

    Raises:
        KeyError: If ``markers_outwards`` is ``True`` and ``marker`` does not
            exist in ``FLIPPED_MARKERS``.

    """
    if _is_directional_marker(marker) and markers_outwards and marker not in FLIPPED_MARKERS:
        raise ValueError(
            f"markers_outwards=True requires a directional marker with a defined flip in FLIPPED_MARKERS, got '{marker}'"
        )

    # TODO: add example to docstrinng
    # TODO: add necessary input vallidation.
    y = np.asarray(y)
    x = np.asarray(x)

    if ax is None:
        ax = plt.gca()

    # plt.stem() always need manaul style info for marker and stemline
    style = _get_next_style(ax, style_index)

    if markers_outwards:
        up_stems = np.where(y >= bottom, y, np.nan)
    else:
        up_stems = y

    markerline_up, stemline_up, baseline_up = plot_stem_segment(
        x=x, y=up_stems, ax=ax, bottom=bottom, label=label, marker=marker, markersize=markersize, show_baseline=show_baseline, style=style
    )

    if markers_outwards:
        down_stems = np.where(y < bottom, y, np.nan)
        flipped_marker = FLIPPED_MARKERS[marker]

        markerline_down, stemline_down, baseline_down = plot_stem_segment(
            x=x, y=down_stems, ax=ax, bottom=bottom, label=None, marker=flipped_marker, markersize=markersize, show_baseline=show_baseline, style=style
        )

    if markers_outwards:
        return [markerline_up, markerline_down], [stemline_up, stemline_down], [baseline_up, baseline_down]
    else:
        return [markerline_up], [stemline_up], [baseline_up]


# def plot_stem(
#     x,
#     y,
#     style_index: int | None = None,
#     ax: Axes | None = None,
#     label: str | None = None,
#     marker: str = 'o',
#     markers_outwards: bool = False,
#     baseline: float = 0.0,
#     show_baseline: bool = True,
#     markersize: float = MARKERSIZE,
# ) -> tuple:
#     """Plot a styled stem plot with optional directional marker flipping.

#     Creates a stem plot where markers can automatically flip direction based on
#     their position relative to the baseline. This is useful for visualizing
#     discrete signals or impulse responses with clear directional emphasis.

#     The function automatically advances the global style cycler for consistent
#     multi-plot styling.

#     Args:
#         x (array-like): X-coordinates of the samples. Must match length of ``y``.
#         y (array-like): Y-values of the samples. Must match length of ``x``.
#         style_index (int | None, optional): Style index from ``get_style()``.
#             If None, uses the next style from the current cycler. Default is None.
#         ax (Axes, optional): Matplotlib axes to plot on. If None, uses current
#             axes (``plt.gca()``). Default is None.
#         label (str | None, optional): Legend label for the plot. Default is None.
#         marker (str, optional): Marker symbol for data points. Default is 'o'.
#         markers_outwards (bool, optional): If True, flips directional markers
#             ('^', 'v') below the baseline to point away from it. Requires a
#             directional marker. Default is False.
#         baseline (float, optional): Baseline value where stems originate.
#             Default is 0.0.
#         show_baseline (bool, optional): If True, renders the baseline as a
#             horizontal line. Default is True.
#         markersize (float, optional): Size of the markers. Default is MARKERSIZE.

#     Returns:
#         tuple: Four-element tuple containing:
#             - markerlines (list[Line2D]): Marker objects (1 or 2 elements)
#             - stemlines (list[LineCollection]): Stem line collections (1 or 2)
#             - baseline_line (Line2D | None): Baseline object or None

#     Raises:
#         ValueError: If x and y have different shapes.
#         ValueError: If markers_outwards=True but marker is non-directional.

#     Example:
#         >>> # Basic stem plot
#         >>> x = [1, 2, 3, 4]
#         >>> y = [2, -1, 3, -2]
#         >>> markers, stems, baseline = plot_stem(x, y)
        
#         >>> # With outward-pointing markers
#         >>> markers, stems, baseline = plot_stem(
#         ...     x, y, marker='^', markers_outwards=True
#         ... )
#     """
#     x = np.asarray(x)
#     y = np.asarray(y)

#     if x.shape != y.shape:
#         raise ValueError(f"x and y must have the same shape, got {x.shape} vs {y.shape}")
#     if markers_outwards and not _is_directional_marker(marker):
#         raise ValueError(
#             f"markers_outwards=True requires a directional marker ('^', 'v'), "
#             f"got non-directional marker '{marker}'"
#         )
#     if ax is not None and not isinstance(ax, Axes):
#         raise TypeError(f"ax must be a matplotlib Axes object, got {type(ax)}")
    
#     if ax is None:
#         ax = plt.gca()

#     style_manager = get_style_manager(ax)
#     if style_index is not None:
#         style = style_manager.get(style_index)
#     else:
#         style = style_manager.next()

#     if not markers_outwards:        # Case 1: No outward flipping → everything uses the same marker
#         markerline, stems, baseline_line = _stem_segment(
#             ax, x, y, baseline, label, marker, markersize, style, show_baseline
#         )
#         markerlines = [markerline]
#         stemlines = [stems]
#     else:           # Case 2: Outward flipping enabled → split data by baseline
#         y_up = np.where(y >= baseline, y, np.nan)
#         y_down = np.where(y < baseline, y, np.nan)
#         flipped = FLIPPED_MARKERS[marker]

#         # Upper segment (original marker pointing up/down)
#         m_up, s_up, baseline_line = _stem_segment(
#             ax, x, y_up, baseline, label, marker, markersize, style, show_baseline
#         )

#         # Lower segment (flipped marker)
#         m_down, s_down, _ = _stem_segment(
#             ax, x, y_down, baseline, None, flipped, markersize, style, show_baseline
#         )

#         markerlines = [m_up, m_down]
#         stemlines = [s_up, s_down]

#     if baseline_line is not None:
#         baseline_line.set_visible(show_baseline)

#     return markerlines, stemlines, baseline_line


# ___________________________________________________________________
#  Custom Nyquist Plot


def _nyquist_segment(
    ax: Axes,
    x: np.ndarray,
    y: np.ndarray,
    arrow_index: int,
    style: PlotStyle,
    label: str | None = None,
    alpha: float | None = None,
    arrow_size: int = 15
) -> None:
    """Plot one Nyquist segment (forward or mirrored) with a directional arrow.

    Internal helper that draws a single curve segment for Nyquist plots and
    adds a directional arrow to indicate frequency progression. Used for both
    the main curve and its complex conjugate mirror.

    Args:
        ax (Axes): Matplotlib axes to draw on.
        x (array-like): Real parts of the frequency response curve.
        y (array-like): Imaginary parts of the frequency response curve.
        arrow_index (int): Index where the arrow should be placed (must be
            less than len(x) - 1).
        style (PlotStyle): The plotting style to use for the segment.
        label (str | None, optional): Legend label for the curve. Default is None.
        alpha (float | None, optional): Transparency level (0-1). Default is None.
        arrow_size (int, optional): Size of the arrow marker (mutation_scale).
            Default is 15.

    Returns:
        None
    """
    
    ax.plot(x, y, label=label, alpha=alpha, **style)

    ax.annotate(
        '',
        xy=(x[arrow_index + 1], y[arrow_index + 1]),
        xytext=(x[arrow_index], y[arrow_index]),
        arrowprops=dict(
            arrowstyle=ARROWSTYLE,
            color=style["color"],
            lw=0,  # No shaft line
            alpha=alpha,
            mutation_scale=arrow_size
        ),
    )


def plot_nyquist(
    real,
    imag,
    ax: Axes | None = None,
    style_index: int = 0,
    label: str | None = None,
    mirror: bool = True,
    arrow_position: float = 0.33,
    alpha: float = 0.5,
    equal_axes: bool = True
) -> None:
    """Plot a Nyquist diagram with directional arrows and optional mirror curve.

    Creates a Nyquist plot showing the frequency response in the complex plane.
    Automatically sets equal axis scaling and adds directional arrows to indicate
    increasing frequency. Optionally plots the complex conjugate (mirror) curve.

    Important:
        Input arrays should extend beyond critical points (e.g., origin, crossings)
        to ensure proper visualization of curve endpoints.

    Args:
        real (array-like): Real parts of the frequency response H(jω).
        imag (array-like): Imaginary parts of the frequency response H(jω).
        ax (Axes, optional): Matplotlib axes to plot on. If None, uses current
            axes (``plt.gca()``). Default is None.
        style_index (int, optional): Index for ``get_style()`` to select color
            and linestyle. Default is 0.
        label (str | None, optional): Legend label for the main curve.
            Default is None.
        mirror (bool, optional): If True, plots the complex conjugate
            mirror curve (negative imaginary part). Default is True.
        arrow_position (float, optional): Relative position (0-1) along the
            curve's arc length where the arrow is placed. 0 is start, 1 is end.
            Default is 0.33.
        alpha (float, optional): Transparency level for the mirror curve (0-1).
            Default is 0.5.
        equal_axes (bool, optional): If True, sets equal scaling for x and y
            axes. Default is True.

    Raises:
        ValueError: If real and imag are not 1D arrays with matching shapes.
        ValueError: If arrays contain fewer than 2 points.
        ValueError: If arrow_position is not in [0, 1].
        ValueError: If style_index is not a non-negative integer.

    Example:
        >>> # Basic Nyquist plot
        >>> omega = np.logspace(-2, 2, 500)
        >>> H = 1 / (1 + 1j * omega)
        >>> plot_nyquist(H.real, H.imag)
        
        >>> # Without mirror curve
        >>> plot_nyquist(H.real, H.imag, mirror=False, arrow_position=0.5)
    """
    real = np.asarray(real)
    imag = np.asarray(imag)

    if real.ndim != 1 or imag.ndim != 1:
        raise ValueError(
            f"real and imag must be 1D arrays, got shapes {real.shape} and {imag.shape}"
        )
    if real.shape != imag.shape:
        raise ValueError(
            f"real and imag must have the same shape, got {real.shape} vs {imag.shape}"
        )
    if len(real) < 2:
        raise ValueError(
            "real and imag must contain at least 2 points to plot a curve"
        )
    if not (0 <= arrow_position <= 1):
        raise ValueError(f"arrow_position must be in [0, 1], got {arrow_position}")
    if not isinstance(style_index, int) or style_index < 0:
        raise ValueError(
            f"style_index must be a non-negative integer, got {style_index}"
        )
    if ax is not None and not isinstance(ax, Axes):
        raise TypeError(f"ax must be a matplotlib Axes object, got {type(ax)}")

    # Calculate arrow position based on arc length
    dist = np.insert(np.cumsum(np.abs(np.diff(real + 1j * imag))), 0, 0)
    total = dist[-1]
    arrow_idx = np.argmin(np.abs(dist - total * arrow_position))
    arrow_idx =  int(min(arrow_idx, len(real) - 2))

    if ax is None:
        ax = plt.gca()

    if style_index is None:
        style = {}
    else:
        style = _get_next_style(ax, style_index)

    # Plot main curve
    _nyquist_segment(ax, real, imag, arrow_idx, style, label)

    # Plot mirror curve (complex conjugate)
    if mirror:
        _nyquist_segment(ax, real, -imag, arrow_idx, style, alpha=alpha)

    if equal_axes:
        ax.axis("equal")


# ___________________________________________________________________
#  Custom Bode Plot


def plot_bode(
    mag,
    phase,
    omega,
    axes: np.ndarray | None = None,
    style_index: int | None = None,
    label: str | None = None,
    dB: bool = True,
    minor_ticks: bool = True,
    tick_denominator: int = 4,
    tick_numerator: int = 1
) -> np.ndarray:
    """Plot Bode magnitude and phase diagrams from frequency response data.

    Creates a two-panel Bode plot showing magnitude (in dB or linear scale) and
    phase (in radians with π-based tick labels) versus frequency on a logarithmic
    scale. Automatically advances the global style cycler for consistent styling.

    Phase is unwrapped using ``numpy.unwrap()`` for continuous display
    across ±π boundaries.

    Args:
        mag (array-like): Magnitude response values ``|H(jω)|``. Must match length
            of ``omega``.
        phase (array-like): Phase response in radians.
            Must match length of ``omega``.
        omega (array-like): Frequency vector in rad/s. Must be positive and
            match length of ``mag`` and ``phase``.
        axes (array-like of Axes, optional): Two axes for [magnitude, phase]
            subplots. If None, creates a new 1x2 subplot figure. Default is None.
        style_index (int | None, optional): Style index from ``get_style()``.
            If None, uses next style from current cycler. Default is None.
        label (str, optional): Legend label for both plots. Default is None.
        dB (bool, optional): If True, plots magnitude in decibels (20·log₁₀).
            If False, plots linear magnitude. Default is True.
        minor_ticks (bool, optional): If True, adds logarithmic minor ticks to
            frequency axes. Default is True.
        tick_denominator (int, optional): Denominator for phase tick fractions
            (e.g., 4 for π/4 increments). Default is 4.
        tick_numerator (int, optional): Numerator for phase tick fractions
            (e.g., 1 for π/4). Default is 1.

    Returns:
        np.ndarray: Array of two Matplotlib Axes: ``[magnitude_ax, phase_ax]``.

    Raises:
        ValueError: If omega is empty or contains non-positive values.
        ValueError: If mag or phase length doesn't match omega length.
        ValueError: If style_index is not a non-negative integer or None.

    Example:
        >>> # Create frequency response
        >>> omega = np.logspace(-1, 2, 300)
        >>> H = 1 / (1 + 1j * omega)
        >>> mag = np.abs(H)
        >>> phase = np.angle(H)
        
        >>> # Plot Bode diagram
        >>> axes = plot_bode(mag, phase, omega, label='Low-pass filter')
        >>> axes[0].set_ylabel('Magnitude (dB)')
        >>> axes[1].set_ylabel('Phase (rad)')
        >>> plt.show()
        
        >>> # With custom phase tick spacing (π increments)
        >>> axes = plot_bode(mag, phase, omega, tick_denominator=1)
    """
    omega = np.atleast_1d(omega)
    mag = np.atleast_1d(mag)
    phase = np.atleast_1d(phase)

    # Validate inputs
    if omega.size == 0:
        raise ValueError("Frequency vector `omega` must be non-empty")
    if mag.size != omega.size:
        raise ValueError(
            f"`mag` length ({mag.size}) does not match `omega` length ({omega.size})"
        )
    if phase.size != omega.size:
        raise ValueError(
            f"`phase` length ({phase.size}) does not match `omega` length ({omega.size})"
        )
    if np.any(omega <= 0):
        raise ValueError("All frequency values in `omega` must be positive")
    if style_index is not None and (not isinstance(style_index, int) or style_index < 0):
        raise ValueError(
            f"style_index must be a non-negative integer or None, got {style_index}"
        )
    if axes is None:
        fig = plt.gcf()
        axes = fig.get_axes()
    if axes is None:
        _, axes = plt.subplots(1, 2, sharex=True, figsize=FIGURE_SIZE)
    else:
        if isinstance(axes, Axes):
            axes = np.array([axes])
        else:
            axes = np.array(axes).ravel()
        if axes.size != 2 or not all(isinstance(ax, Axes) for ax in axes):
            raise TypeError("`axes` must be array-like of exactly two matplotlib Axes objects")
    mag_ax, phase_ax = axes

    # unwrap freq around pi
    phase = np.unwrap(phase)

    # Convert magnitude to dB if requested
    if dB:
        mag = 20 * np.log10(mag)

    if style_index is None:
        style = {}
    else:
        style = _get_next_style(mag_ax, style_index)

    # Magnitude plot
    mag_ax.plot(omega, mag, label=label, **style)
    mag_ax.set_xscale("log")

    # Phase plot
    phase_ax.plot(omega, phase, label=label, **style)
    phase_ax.set_xscale("log")
    set_major_tick_labels(
        label=r"$\pi$",
        unit=np.pi,
        denominator=tick_denominator,
        numerator=tick_numerator,
        axis=phase_ax.yaxis
    )

    if minor_ticks:
        set_minor_log_ticks(axis=mag_ax.xaxis)
        set_minor_log_ticks(axis=phase_ax.xaxis)

    return np.array([mag_ax, phase_ax])
