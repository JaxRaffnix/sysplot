Concepts
========

This section explains the main design concepts behind the ``sysplot`` module.

Plot Style
----------


Matplotlib uses a *property cycler* to determine the default styling of new
plot elements. Each call to ``plot()`` (or a related function) advances this
cycler and assigns the next color automatically.

``sysplot`` modifies this behavior slightly by providing a custom cycler that
controls both color **and** linestyle. The goal is to keep styling consistent
and distinct even in black/white figures while still remaining compatible with
the default behavior of Matplotlib and Seaborn. 

This behavior allows multiple lines to be plotted without manually specifying
styles.

However, different plotting functions use **independent cyclers**. For example,
``scatter()`` does not share the same style cycle as ``plot()``.

.. literalinclude:: examples/matplotlib_cycler.py
   :language: python



Another subtle behavior occurs when color or linestyle properties are manually
specified. In that case the scatter cycler **may not advance**, which
can lead to inconsistent styling.

The ``sysplot`` solution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``sysplot`` is designed for producing **consistent, publication-quality plots**
for system theory and control engineering. Users should not worry about modifying
colors or linestyles manually, but rather access the configured styles from the cycler.

Also, many higher-level plotting functions internally call multiple Matplotlib
commands. For example:

* ``plot_nyquist()`` may call ``plot()`` multiple times
* ``plot_poles_zeros()`` uses ``scatter()`` to draw markers

From the user's perspective, these should represent  **a single logical plot**
and therefore show only one style.

To achieve this, ``sysplot`` provides the public user function ``get_style()``.
This function returns a dictionary containing a color and linestyle derived
from the configured style cycler.

Example return value::

    {
        "color": "#1f77b4",
        "linestyle": "-"
    }

Two usage patterns are supported.

1. Retrieve a style by index
"""""""""""""""""""""""""""""""""""

A specific style can be retrieved directly from the cycler::

    style = get_style(index=2)
    ax.plot(x, y, **style)

This returns the style at the specified position and still advances the internal
cycler. Especially useflul if you want two ``plot()`` call to have the same style.

1. Retrieve the next style for an axis
""""""""""""""""""""""""""""""""""""""""

Alternatively, the next style can be determined for a specific axis::

    style = get_style(ax=ax)
    ax.scatter(x, y, **style)

In this case ``sysplot`` determines the next style that would normally be used
for ``plot()`` and returns it as a dictionary.

This allows functions such as ``scatter()`` to remain consistent with the
line-style cycle used by ``plot()``.

System Modelling
----------------

A recommended workflow for modelling systems is shown below. Following this
structure makes it convenient to use the plotting utilities provided by
``sysplot`` (see the next section). Other approaches are also possible, as long
as the resulting data is compatible with the plotting function arguments.

The example below defines a second-order system with a root and computes its
frequency response.

.. literalinclude:: examples/modeling.py


Plotting
--------

Assuming the variables from the previous section are defined,
``sysplot`` provides several convenience functions for common control-
engineering visualizations.

- :func:`plot_bode` – Plot Bode magnitude and phase diagrams from frequency response data.
- :func:`plot_nyquist` – Plot a Nyquist diagram with directional arrows and optional mirror curve.
- :func:`plot_poles_zeros` – Plot poles and zeros on the complex plane.
- :func:`plot_stem` – Plot a styled stem plot with optional outward-pointing markers.
- :func:`plot_angle` – Draw and label the angle between two vectors.


Figure Styling
--------------

To improve clarity and readability of figures, ``sysplot`` provides several
helpers for adding reference elements and adjusting axes behavior. Using these
tools is recommended when appropriate.

- :func:`emphasize_coord_lines` – Draw origin guide lines on all 2D axes of a figure.
- :func:`add_origin` – Ensure the origin is included in the axes autoscaling.
- :func:`plot_unit_circle` – Plot a unit circle on the current axes.
- :func:`plot_filter_tolerance` – Draw generic filter power constraints.

Axis and tick helpers:

- :func:`set_major_ticks` – Set major ticks with fractional labels.
- :func:`set_minor_log_ticks` – Add minor ticks at decade intervals on logarithmic axes.
- :func:`set_xmargin` – Enable or disable the automatic x-axis margin.
- :func:`repeat_axis_ticks` – Show tick labels on all axes of a figure.
- :func:`add_tick_line` – Add a labeled reference tick without modifying the major tick locator.


Figure Configuration
--------------------

Global plotting behavior can be configured using
:class:`SysplotConfig`.

The configuration is applied using:

- :func:`apply_config` – Apply ``sysplot`` styling globally via Seaborn and
  Matplotlib ``rcParams``.
