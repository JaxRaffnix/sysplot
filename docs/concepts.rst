Concepts
========

This page explains the core ideas behind sysplot and how they are used in practice.

Plot Styling
-------------

Matplotlib uses a *property cycler* to assign default styles to new plot
elements. Every call to ``plot()`` (or a related function) advances that
cycler. This means multiple lines can be plotted without manually specifying
styles.

sysplot extends this cycler to include both color and linestyle. The 
goal is consistent styling, including black-and-white print contexts, while
remaining compatible with Matplotlib and Seaborn defaults.

However, different plotting functions use independent cyclers. For example,
``scatter()`` does not follow the cycler from ``plot()``, as you can see 
in this :ref:`matplotlib_cycler Example <sphx_glr__auto_examples_matplotlib_cycler.py>`.

Another subtle behavior appears when the cycler contains both color and
linestyle (as configured in sysplot by :func:`sysplot.apply_config`): the internal cycle
may still advance even if you override only one property (for example,
``color=...``). This can produce unexpected style offsets later in a figure.

The sysplot solution
^^^^^^^^^^^^^^^^^^^^^^^^

sysplot is designed to produce consistent, publication-quality plots for
system theory and control engineering. Instead of manually assigning color and linestyle
values, users should easily access the styles from the configured cycler.

Also, many higher-level plotting helpers internally call multiple Matplotlib
commands. For example:

* :func:`sysplot.plot_nyquist` and :func:`sysplot.plot_stem` may call
  ``plot()`` multiple times.
* :func:`sysplot.plot_poles_zeros` uses ``scatter()`` to draw markers.

From the user's perspective, these should represent a single logical plot
and therefore show only one style. Additionally, the ``scatter()`` and ``plot()`` might sometimes share
their style cycler and advance each other.

To support this, sysplot provides :func:`sysplot.get_style`, which returns
a style dictionary derived from the configured cycler.

Example return value::

    {
        "color": "#1f77b4",
        "linestyle": "-"
    }

1. Retrieve a style by index
""""""""""""""""""""""""""""""""

A specific style can be retrieved directly from the cycler::

    style = get_style(index=2)
    ax.plot(x, y, **style)

This returns the style at the specified position and still advances the internal
cycler. This is useful if you want explicit style control or if multiple
elements should intentionally share a style.

1. Retrieve the next style for an axis
"""""""""""""""""""""""""""""""""""""""""

Alternatively, the next style can be determined for a specific axis::

    style = get_style(ax=ax)
    ax.scatter(x, y, **style)

In this mode, sysplot determines the next style that would be used by
``plot()`` on that axis and returns it as a dictionary.
This helps keep functions such as ``scatter()`` visually consistent with the
line-style progression used by ``plot()``.

Both usage patterns are shown in this :ref:`example <sphx_glr__auto_examples_get_style.py>`.

System Modelling
----------------

A recommended workflow for modeling systems is shown below. Following this
structure makes it convenient to pass data into sysplot plotters. Other
approaches are also valid as long as the resulting arrays match the expected
function arguments.

The example below defines a second-order system with a root and computes its
frequency response.

.. literalinclude:: examples/modelling.py
    :language: python


Plot Functions
----------------

Assuming the variables from the previous section are defined,
sysplot provides convenience functions for common control-engineering
visualizations.

* :func:`sysplot.plot_bode` — Plot Bode magnitude and phase diagrams from
  frequency response data.
* :func:`sysplot.plot_nyquist` — Plot a Nyquist diagram with directional
  arrows and optional mirror curve.
* :func:`sysplot.plot_poles_zeros` — Plot poles and zeros on the complex plane.
* :func:`sysplot.plot_stem` — Plot a styled stem plot with optional outward-pointing markers.
* :func:`sysplot.plot_angle` — Draw and label the angle between two vectors.


Figure Styling
--------------

To improve clarity and readability of figures, sysplot provides several
helpers for adding reference elements and adjusting axes behavior. Using these
tools is recommended whenever appropriate.

* :func:`sysplot.emphasize_coord_lines` — Draw coordinate origin lines on all 2D
  axes of a figure.
* :func:`sysplot.add_origin` — Ensure the origin is included in autoscaling.
* :func:`sysplot.plot_unit_circle` — Plot a unit circle on the current axes.
* :func:`sysplot.plot_filter_tolerance` — Draw generic filter power constraints.
* :func:`sysplot.set_minor_log_ticks` — Add minor ticks at decade intervals on logarithmic axes.

Often times you want to show the connection between your data and system parameters.
The :func:`sysplot.set_major_ticks` is especially useful for this, as it allows you to set ticks at specific values with custom labels. You can even adjust the numerator and denominator of the labels to show fractional values, or limit the ticks
to only show the parameter once on the axis. If you wish to show a parameter without chaning the major ticks, you can use :func:`sysplot.add_tick_line` to add a labeled reference line.

Since most plots will show a time continous signal on the x-axis, the x-margin has 
been set to ``0`` when callign :func:`sysplot.apply_config`. To disable this, use :func:`sysplot.set_xmargin` on a specific axis with ``use_margin=True`` to restore default Matplotlib behavior.

To repeat tick labels on all axes of a figure with shared axes, use :func:`sysplot.repeat_axis_ticks`. 

All of these functions are demonstrated in this :ref:`example <sphx_glr__auto_examples_quick_start.py>`.

Figure Configuration
--------------------

An oppinonated design choice is used for the sysplot module. It leverages seaborn styles and Matploblibs defaults, but makes opnionated changes to these defaults. To activate these changes, the user must call :func:`sysplot.apply_config`. Changes can be configuredby  using
:class:`sysplot.SysplotConfig`. 

A common workflow is:

1. Import the module: ``import sysplot as ssp``.
2. Call :func:`sysplot.apply_config` once near the top of your file.
3. Build figures and plot data.
4. Use :func:`sysplot.get_style` when you need style synchronization across
   functions such as ``plot()`` and ``scatter()``.

