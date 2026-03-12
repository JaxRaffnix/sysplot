import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.axes import Axes
import numpy as np

from typing import Sequence, Union

from .figures import get_figsize
from .config import get_config
from .styles import PlotStyle, get_style
from .axes import add_origin, set_xmargin
from .ticks import set_major_ticks, set_minor_log_ticks


# ___________________________________________________________________
#  Pole-Zero Plot

def plot_poles_zeros(
    poles: complex | list[complex] | np.ndarray | None = None,
    zeros: complex | list[complex] | np.ndarray | None = None,
    label: str | None = None,
    ax: Axes | None = None,
    markersize: float | None = None,
    show_origin: bool = True,
    enable_xmargin: bool = True,
    **kwargs,
) -> None:
    """Plot poles and zeros on the complex plane.

    Poles are drawn as ``×`` markers and zeros as hollow circles. Both share
    the same color and linestyle from the active style cycle. Either poles or zeros can be omitted by passing ``None`` or an empty list/array.

    Args:
        poles: Complex pole locations. Accepts a scalar, list, or array.
        zeros: Complex zero locations. Accepts a scalar, list, or array.
        label: Legend label. Applied to poles if present, otherwise to zeros.
        ax: Axes to plot on. Defaults to the current axes.
        markersize: Marker size. Defaults to
            :attr:`~sysplot.SysplotConfig.poles_zeros_markersize`.
        show_origin: If ``True``, marks the origin to keep it in view.
        enable_xmargin: If ``True``, adds a small x-axis margin so markers
            near the edges are not clipped.

    .. minigallery:: sysplot.plot_poles_zeros
        :add-heading:
    """
    if ax is not None and not isinstance(ax, Axes):
        raise TypeError(f"'ax' must be a matplotlib Axes or None, got {type(ax).__name__!r}")
    if not isinstance(show_origin, bool):
        raise TypeError(f"'show_origin' must be a bool, got {type(show_origin).__name__!r}")
    if not isinstance(enable_xmargin, bool):
        raise TypeError(f"'enable_xmargin' must be a bool, got {type(enable_xmargin).__name__!r}")
    markersize = get_config().poles_zeros_markersize if markersize is None else markersize
    if not isinstance(markersize, (int, float)) or markersize <= 0:
        raise ValueError(f"'markersize' must be a positive number, got {markersize!r}")

    poles = np.atleast_1d(poles) if poles is not None else np.array([])
    zeros = np.atleast_1d(zeros) if zeros is not None else np.array([])

    if poles.size == 0 and zeros.size == 0:
        raise ValueError("At least one of poles or zeros must be non-empty to plot.")

    if ax is None:
        ax = plt.gca()

    # TODO: in the docs/example/plot_pole_zero: show the different markers when plotting multiple pole-zero calls.
    
    # get style from axis, but allow user overwrite
    style = get_style(ax=ax)
    color = kwargs.pop("color", style["color"])
    linestyle = kwargs.pop("linestyle", style["linestyle"])
    style_kwargs = dict(color=color, linestyle=linestyle)


    # poles
    if poles.size > 0:
        ax.scatter(
            np.real(poles),
            np.imag(poles),
            marker='x',
            s=markersize**2,
            label=label,
            **style_kwargs,
            **kwargs
        )

    # zeros
    if zeros.size > 0:
        ax.scatter(
            np.real(zeros),
            np.imag(zeros),
            marker='o',
            facecolors='none',
            s=markersize**2,
            label=None if poles.size > 0 else label,
            **style_kwargs,
            **kwargs
        )    

    if show_origin:
        add_origin(ax)

    if enable_xmargin:
        set_xmargin(ax, use_margin=enable_xmargin)


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
    """Return ``True`` if ``marker`` has a defined flip in ``FLIPPED_MARKERS``."""
    return isinstance(marker, str) and marker in FLIPPED_MARKERS


# ___________________________________________________________________
#  Stem Plot

def _plot_stem_segment(
    x, y, 
    ax, 
    bottom, 
    label, 
    marker, markersize, 
    show_baseline, 
    color, linestyle,
    kwargs
):

    markerline, stemline, baseline = ax.stem(x, y, bottom=bottom, label=label, **kwargs)
    plt.setp(markerline, color=color, marker=marker, markersize=markersize)
    plt.setp(stemline, color=color, linestyle=linestyle)
    if show_baseline:
        plt.setp(baseline, color=color, linestyle=linestyle)
    else:
        baseline.set_visible(False)

    return markerline, stemline, baseline


def plot_stem(
    x: np.ndarray,
    y: np.ndarray,
    ax: Axes | None = None,
    label: str | None = None,
    bottom: float = 0.0,
    marker: str = "o",
    markersize: float | None = None,
    show_baseline: bool = True,
    markers_outwards: bool = False,
    continous_baseline: bool = False,
    **kwargs,
) -> tuple[list, list, list]:
    """Plot vertical stems with optional outward-pointing markers.

    Wraps ``Axes.stem`` with automatic style cycling. When
    ``markers_outwards=True``, stems above ``bottom`` use ``marker`` and
    stems below use its directional opposite from ``FLIPPED_MARKERS``
    (e.g., ``^`` becomes ``v``).

    Note:
        Use ``^`` as the ``marker`` to get outward-pointing arrows on both
        sides of the baseline.

    Args:
        x: X-coordinates.
        y: Y-values. Must have the same length as ``x``.
        ax: Axes to plot on. Defaults to the current axes.
        label: Legend label. Applied to the above-baseline stems only.
        bottom: Baseline value. Default is ``0``.
        marker: Matplotlib marker style for stems at or above ``bottom``.
        markersize: Marker size. Defaults to
            :attr:`~sysplot.SysplotConfig.markersize`.
        show_baseline: If ``True``, draw and style the baseline.
        markers_outwards: If ``True``, flip the marker for stems below
            ``bottom``. Requires ``marker`` to be in ``FLIPPED_MARKERS``.
        continous_baseline: If ``True``, draw a full-width ``axhline``
            baseline. Requires ``show_baseline=True``.

    Returns:
        Tuple of ``(markerlines, stemlines, baselines)`` — each a list of
        1 or 2 objects depending on whether ``markers_outwards`` was used.

    .. minigallery:: sysplot.plot_stem
        :add-heading:
    """
    if markers_outwards and marker not in FLIPPED_MARKERS:
        raise ValueError(
            f"'markers_outwards=True' requires a directional marker "
            f"({list(FLIPPED_MARKERS.keys())}), got {marker!r}"
        )
    if continous_baseline and not show_baseline:
        raise ValueError("'continous_baseline=True' requires 'show_baseline=True'.")
    markersize = get_config().markersize if markersize is None else markersize

    # TODO: rename outwards to flip_around_baseline. beacuse using marker="v" will point inwards with their counterpart,
    # TODO: add example to docstrinng
    # TODO: add necessary input vallidation.
    y = np.asarray(y)
    x = np.asarray(x)

    if ax is None:
        ax = plt.gca()

    # plt.stem() always need manaul style info for marker and stemline
    # get style from axis, but allow user overwrite
    style = get_style(ax=ax)
    color = kwargs.pop("color", style["color"])
    linestyle = kwargs.pop("linestyle", style["linestyle"])
    # style_kwargs = dict(color=color, linestyle=linestyle)

    if markers_outwards:
        up_stems = np.where(y >= bottom, y, np.nan)
    else:
        up_stems = y

    markerline_up, stemline_up, baseline_up = _plot_stem_segment(
        x=x, y=up_stems, 
        ax=ax, 
        bottom=bottom, 
        label=label, 
        marker=marker, markersize=markersize, 
        show_baseline=show_baseline, 
        color=color, linestyle=linestyle,
        kwargs=kwargs
    )

    if markers_outwards:
        down_stems = np.where(y < bottom, y, np.nan)
        flipped_marker = FLIPPED_MARKERS[marker]

        markerline_down, stemline_down, baseline_down = _plot_stem_segment(
            x=x, y=down_stems, 
            ax=ax, 
            bottom=bottom, 
            label=None, 
            marker=flipped_marker, markersize=markersize, 
            show_baseline=show_baseline, 
            color=color, linestyle=linestyle,
            kwargs=kwargs
        )
    else:
        markerline_down = stemline_down = baseline_down = None

    if continous_baseline and show_baseline:
        ax.axhline(bottom, color=color, linestyle=linestyle, **kwargs)

    if markers_outwards:
        return [markerline_up, markerline_down], [stemline_up, stemline_down], [baseline_up, baseline_down]
    else:
        return [markerline_up], [stemline_up], [baseline_up]
    

# ___________________________________________________________________
#  Custom Nyquist Plot


def _nyquist_segment(
    ax: Axes,
    x: np.ndarray,
    y: np.ndarray,
    arrow_index: int,
    style_kwargs: dict,
    kwargs: dict,
    label: str | None = None,
    alpha: float | None = None,
    arrow_size: int = 15,
) -> None:
    """Draw one Nyquist curve segment with a directional arrow."""
    
    ax.plot(x, y, label=label, alpha=alpha, **style_kwargs, **kwargs)

    ax.annotate(
        '',
        xy=(x[arrow_index + 1], y[arrow_index + 1]),
        xytext=(x[arrow_index], y[arrow_index]),
        arrowprops=dict(
            arrowstyle=get_config().arrowstyle,
            color=style_kwargs["color"],
            lw=0,  # No shaft line
            alpha=alpha,
            mutation_scale=arrow_size
        ),
    )


def plot_nyquist(
    real: np.ndarray,
    imag: np.ndarray,
    ax: Axes | None = None,
    label: str | None = None,
    mirror: bool = True,
    arrow_position: float | None = None,
    alpha: float | None = None,
    equal_axes: bool = True,
    arrow_size: int | None = None,
    **kwargs,
) -> None:
    """Plot a Nyquist diagram with a directional arrow and optional mirror curve.

    Draws the frequency response in the complex plane. A directional arrow
    marks the direction of increasing frequency. The complex conjugate curve
    (mirror) is rendered at reduced opacity when ``mirror=True``.

    Args:
        real: Real parts of the frequency response ``H(jω)``.
        imag: Imaginary parts of the frequency response ``H(jω)``.
        ax: Axes to plot on. Defaults to the current axes.
        label: Legend label for the main curve.
        mirror: If ``True``, also plots the complex conjugate curve.
        arrow_position: Fractional arc-length position (0–1) for the direction
            arrow. Defaults to
            :attr:`~sysplot.SysplotConfig.nyquist_arrow_position`.
        alpha: Transparency of the mirror curve. Defaults to
            :attr:`~sysplot.SysplotConfig.nyquist_mirror_alpha`.
        equal_axes: If ``True``, sets equal axis scaling.
        arrow_size: Arrow head size (``mutation_scale``). Defaults to
            :attr:`~sysplot.SysplotConfig.nyquist_arrow_size`.

    .. minigallery:: sysplot.plot_nyquist
        :add-heading:
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
    arrow_position = arrow_position if arrow_position is not None else get_config().nyquist_arrow_position
    alpha = alpha if alpha is not None else get_config().nyquist_mirror_alpha
    arrow_size = arrow_size if arrow_size is not None else get_config().nyquist_arrow_size
    if not (0.0 <= arrow_position <= 1.0):
        raise ValueError(f"'arrow_position' must be in [0, 1], got {arrow_position!r}")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"'alpha' must be in [0, 1], got {alpha!r}")
    if ax is not None and not isinstance(ax, Axes):
        raise TypeError(f"'ax' must be a matplotlib Axes or None, got {type(ax).__name__!r}")

    # Calculate arrow position based on arc length
    dist = np.insert(np.cumsum(np.abs(np.diff(real + 1j * imag))), 0, 0)
    total = dist[-1]
    arrow_idx = np.argmin(np.abs(dist - total * arrow_position))
    arrow_idx =  int(min(arrow_idx, len(real) - 2))

    if ax is None:
        ax = plt.gca()

    # get style from axis, but allow user overwrite
    style = get_style(ax=ax)
    color = kwargs.pop("color", style["color"])
    linestyle = kwargs.pop("linestyle", style["linestyle"])
    style_kwargs = dict(color=color, linestyle=linestyle)

    # Plot main curve
    _nyquist_segment(ax, real, imag, arrow_idx, style_kwargs, kwargs, label, arrow_size=arrow_size)

    # Plot mirror curve (complex conjugate)
    if mirror:
        _nyquist_segment(ax, real, -imag, arrow_idx, style_kwargs, kwargs, label=None, alpha=alpha, arrow_size=arrow_size)

    if equal_axes:
        ax.axis("equal")


# ___________________________________________________________________
#  Custom Bode Plot

def _normalize_axes(
        candidate: Axes | Sequence[Axes] | np.ndarray | None,
    ) -> np.ndarray:
        if candidate is None:
            return np.array([], dtype=object)
        if isinstance(candidate, Axes):
            return np.array([candidate], dtype=object)

        normalized = np.asarray(candidate, dtype=object).ravel()
        if not all(isinstance(ax, Axes) for ax in normalized):
            raise TypeError("`axes` must contain matplotlib Axes objects")
        return normalized


def plot_bode(
    mag,
    phase,
    omega,
    axes: Axes | Sequence[Axes] | np.ndarray | None = None,
    label: str | None = None,
    mag_to_dB: bool = True,
    minor_ticks: bool = True,
    tick_denominator: int = 4,
    tick_numerator: int = 1,
    x_to_log: bool = True,
    phase_is_rad: bool = True,
    **kwargs
) -> np.ndarray:
    """Plot a two-panel Bode magnitude and phase diagram.

    Creates magnitude and phase subplots from frequency response arrays.
    Phase is unwrapped with ``numpy.unwrap`` for continuous display across
    ±π boundaries and labelled as multiples of π when ``phase_is_rad=True``.

    Note:
        To show phase in degrees, pass ``np.degrees(phase)`` and set
        ``phase_is_rad=False``.

    Args:
        mag: Magnitude response values ``|H(jω)|``.
        phase: Phase response in radians (or degrees when
            ``phase_is_rad=False``).
        omega: Frequency vector in rad/s.
        axes: Two Axes for ``[magnitude, phase]`` subplots. If ``None``,
            creates a new 1×2 figure.
        label: Legend label for both subplots.
        mag_to_dB: If ``True``, plots magnitude in dB (20·log₁₀).
        minor_ticks: If ``True``, adds log-scale minor ticks to the frequency
            axis.
        tick_denominator: Denominator for π-based phase tick fractions
            (e.g., ``4`` gives π/4 increments). Default is ``4``.
        tick_numerator: Numerator for π-based phase tick fractions. Default
            is ``1``.
        x_to_log: If ``True``, uses a logarithmic frequency axis.
        phase_is_rad: If ``True``, labels the phase axis in multiples of π.
            If ``False``, the axis uses default numeric labels.
        **kwargs: Additional keyword arguments forwarded to both ``ax.plot``
            calls.

    Returns:
        Array of two Matplotlib Axes: ``[magnitude_ax, phase_ax]``.

    .. minigallery:: sysplot.plot_bode
        :add-heading:
    """
    omega = np.atleast_1d(omega)
    mag = np.atleast_1d(mag)
    phase = np.atleast_1d(phase)

    if phase_is_rad and (tick_denominator <= 0 or tick_numerator <= 0):
        raise ValueError("tick_denominator and tick_numerator must be positive integers when phase_is_rad is True")

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

    resolved_axes = _normalize_axes(axes)
    if resolved_axes.size == 0 and plt.get_fignums():
        resolved_axes = _normalize_axes(plt.gcf().get_axes())
    if resolved_axes.size == 0:
        _, created_axes = plt.subplots(1, 2, sharex=True, figsize=get_figsize(1, 2))
        resolved_axes = _normalize_axes(created_axes)

    if resolved_axes.size != 2:
        raise ValueError(
            f"plot_bode requires exactly 2 axes, got {resolved_axes.size}"
        )

    mag_ax, phase_ax = resolved_axes

    # unwrap freq around pi
    phase = np.unwrap(phase)

    # Convert magnitude to dB if requested
    if mag_to_dB:
        mag = _get_db(mag)

    # get style from axis, but allow user overwrite
    style = get_style(ax=mag_ax)
    color = kwargs.pop("color", style["color"])
    linestyle = kwargs.pop("linestyle", style["linestyle"])
    style_kwargs = dict(color=color, linestyle=linestyle)

    # Magnitude plot
    mag_ax.plot(omega, mag, label=label, **style_kwargs, **kwargs)

    # Phase plot
    phase_ax.plot(omega, phase, label=label, **style_kwargs, **kwargs)      
    if phase_is_rad:  
        set_major_ticks(
            label=r"$\pi$",
            unit=np.pi,
            denominator=tick_denominator,
            numerator=tick_numerator,
            axis=phase_ax.yaxis
        )
    else:
        pass
        # TODO: set ticks to integer multiples of 90/45

    if x_to_log:
        mag_ax.set_xscale("log")
        phase_ax.set_xscale("log")

    if minor_ticks:
        set_minor_log_ticks(axis=mag_ax.xaxis)
        set_minor_log_ticks(axis=phase_ax.xaxis)

    return np.array([mag_ax, phase_ax])


# ___________________________________________________________________
#  Unit Circle

def plot_unit_circle(
    ax: Axes | None = None,
    origin: tuple[float, float] = (0.0, 0.0),
    equal_axes: bool = True,
    **kwargs,
) -> None:
    """Draw a unit circle on the axes.

    Draws a circle of radius 1 centered at ``origin``. Line color, style, and
    width default to the grid style from Matplotlib's rcParams.

    Args:
        ax: Axes to draw on. Defaults to the current axes.
        origin: Center of the circle as ``(x, y)``. Default is ``(0, 0)``.
        equal_axes: If ``True``, sets equal axis scaling so the circle
            appears round. Default is ``True``.
        kwargs: Additional keyword arguments forwarded to ``ax.plot``. E.g. color, linestyle, linewidth, zorder.

    .. minigallery:: sysplot.plot_unit_circle
        :add-heading:
    """
    if ax is not None and not isinstance(ax, Axes):
        raise TypeError(f"'ax' must be a matplotlib Axes or None, got {type(ax).__name__!r}")
    if not (hasattr(origin, '__len__') and len(origin) == 2):
        raise ValueError(f"'origin' must be a 2-element sequence, got {origin!r}")
    if not isinstance(equal_axes, bool):
        raise TypeError(f"'equal_axes' must be a bool, got {type(equal_axes).__name__!r}")

    color= kwargs.pop("color", mpl.rcParams['grid.color'])
    linestyle = kwargs.pop("linestyle", mpl.rcParams['grid.linestyle'])
    linewidth = kwargs.pop("linewidth", mpl.rcParams['grid.linewidth'])
    zorder = kwargs.pop("zorder", get_config().zorder_grid)

    if ax is None:
        ax = plt.gca()
        
    theta = np.linspace(0, 2 * np.pi, 200)
    x = origin[0] + np.cos(theta)
    y = origin[1] + np.sin(theta)
    ax.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth, zorder=zorder, **kwargs)

    if equal_axes:
        ax.axis("equal")


# ___________________________________________________________________
#  Filter Tolerance

def _get_db(A: float | np.ndarray, is_power: bool = False):
    """
    Convert linear amplitude or power to dB.
    Returns -inf for zero or negative values.
    
    Parameters
    ----------
    A : float or np.ndarray
        Linear amplitude or power.
    is_power : bool
        True if A is a power value, False if amplitude.

    Returns
    -------
    float or np.ndarray
        Value(s) in dB.
    """
    A = np.asarray(A)  # allow scalar or array

    # Avoid log of zero or negative values
    mask = A <= 0
    db = np.empty_like(A, dtype=float)

    if is_power:
        db[~mask] = 10 * np.log10(A[~mask])
    else:
        db[~mask] = 20 * np.log10(A[~mask])

    db[mask] = -np.inf
    return db if isinstance(A, np.ndarray) else db.item()


def plot_filter_tolerance(
    ax: Axes,
    bands: list[dict],
    A_pass: float,
    A_stop: float,
    w_max: float,
    mag_to_db: bool = False,
    show_mask: bool = True,
    show_arrows: bool = True,
    show_labels: bool = False,
    set_ticks: bool = True,
    arrow_y: float = -0.2,
    alpha: float | None = None,
) -> None:
    """Draw a filter power tolerance mask on an existing axes.

    Shades the forbidden regions of a filter specification. Each entry in
    ``bands`` is a ``dict`` with the following keys:

    - ``"type"``: ``"pass"``, ``"stop"``, or ``"transition"``
    - ``"w0"``: lower frequency bound
    - ``"w1"``: upper frequency bound
    - ``"label"`` (optional): band annotation text
    - ``"w0_label"`` / ``"w1_label"`` (optional): tick label overrides

    Note:
        Axis y-limits must be set before calling this function, as the
        forbidden-region patches are sized from the current limits.

    Args:
        ax: Axes to draw on.
        bands: List of band specification dicts (see above).
        A_pass: Lower passband power bound (``A_D²``).
        A_stop: Upper stopband power bound (``A_S²``).
        w_max: Maximum frequency for the x-axis.
        mag_to_db: If ``True``, converts power bounds to dB before plotting.
        show_mask: If ``True``, shades the forbidden regions.
        show_arrows: If ``True``, draws double-headed arrows with band labels
            below the axis.
        show_labels: If ``True``, adds legend labels to the shaded patches.
        set_ticks: If ``True``, sets frequency and amplitude tick marks from
            the band definitions.
        arrow_y: Vertical position of band arrows in axes coordinates.
            Default is ``-0.2``.
        alpha: Opacity of shaded regions. Defaults to
            :attr:`~sysplot.SysplotConfig.filter_tolerance_alpha`.

    .. minigallery:: sysplot.plot_filter_tolerance
        :add-heading:
    """
    if not isinstance(ax, Axes):
        raise TypeError(f"'ax' must be a matplotlib Axes, got {type(ax).__name__!r}")
    if not isinstance(bands, list) or not bands:
        raise ValueError("'bands' must be a non-empty list of dicts.")
    if not isinstance(w_max, (int, float)) or w_max <= 0:
        raise ValueError(f"'w_max' must be a positive number, got {w_max!r}")
    alpha = alpha if alpha is not None else get_config().filter_tolerance_alpha

    # TODO: make axis limits dynamic and update when they change after the function is called.

    #! Axis limits must be fixed before drawing patches
    ax.set_xlim(0, w_max)
    y_max = ax.get_ylim()[1]
    y_min = ax.get_ylim()[0]

    one_value = 1
    one_label = r'$1$'
    if mag_to_db:
        A_pass = _get_db(A_pass, is_power=True)
        A_stop = _get_db(A_stop, is_power=True)
        one_value = _get_db(1, is_power=True)
        one_label = r'$0$'

    # ------------------------------------------------------------------
    # Set ticks

    if set_ticks:
        # draw frequency ticks
        xticks = []
        xticklabels = []
        for i, band in enumerate(bands):
            for w_point in ["w0", "w1"]:
                w = band[w_point]
                # Avoid duplicate ticks
                if w not in xticks:
                    xticks.append(w)
                    # Use label if available, else default LaTeX
                    tick_label = band.get(f"{w_point}_label", rf"$\omega_{i}_{w_point[-1]}$")
                    xticklabels.append(tick_label)

        # Draw frequency ticks
        ax.set_xticks(ticks=xticks, labels=xticklabels)

        # Draw amplitude ticks
        ax.set_yticks(ticks=[y_min, A_stop, A_pass, one_value], labels=[f'{y_min}', r'$A_S^2$', r'$A_D^2$', one_label])

    # ------------------------------------------------------------------
    # Draw band masks
    if show_mask:
        for band in bands:
            w0, w1 = band["w0"], band["w1"]
            width = w1 - w0

            rect = None
            rect2 = None

            if band["type"] == "pass":
                # Forbidden below Ad
                rect = patches.Rectangle(
                    (w0, y_min),
                    width,
                    A_pass - y_min,
                    alpha=alpha,
                    color="gray",
                    linewidth=0,
                    label=band.get("label") if show_labels else None
                )

                # Forbidden above 1
                rect2 = patches.Rectangle(
                    (w0, one_value),
                    width,
                    y_max - one_value,
                    alpha=alpha,
                    color="gray",
                    linewidth=0,
                )

            elif band["type"] == "stop":
                # Forbidden above As
                rect = patches.Rectangle(
                    (w0, A_stop),
                    width,
                    one_value - A_stop,
                    alpha=alpha,
                    color="gray",
                    linewidth=0,
                    label=band.get("label") if show_labels else None
                )

                # Forbidden above 1
                rect2 = patches.Rectangle(
                    (w0, one_value),
                    width,
                    y_max - one_value,
                    alpha=alpha,
                    color="gray",
                    linewidth=0,
                )

            elif band["type"] == "transition":
                # Forbidden above 1
                rect2 = patches.Rectangle(
                    (w0, one_value),
                    width,
                    y_max - one_value,
                    alpha=alpha,
                    color="gray",
                    linewidth=0,
                    label=band.get("label") if show_labels else None
                )

            else:
                raise ValueError(f"Unknown band type: {band['type']}")

            if rect is not None:
                ax.add_patch(rect)
            if rect2 is not None:
                ax.add_patch(rect2)

    # ------------------------------------------------------------------
    # Draw arrows and labels
    if show_arrows:
        for band in bands:
            w0, w1 = band["w0"], band["w1"]
            label = band.get("label", band["type"])

            ax.annotate(
                "",
                xy=(w0, arrow_y),
                xytext=(w1, arrow_y),
                xycoords=("data", "axes fraction"),
                arrowprops=dict(arrowstyle="<|-|>", linewidth=1, color="black"),
            )

            ax.text(
                (w0 + w1) / 2,
                arrow_y - 0.05,
                label,
                ha="center",
                va="top",
                transform=ax.get_xaxis_transform(),
            )