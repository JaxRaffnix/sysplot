.. _concepts:

Concepts
========

This page explains the core ideas behind sysplot and how they are used in practice.

Plot Cyclers
-------------

Matplotlib uses a *property cycler* to assign default styles to new plot
elements. Every call to ``plot()`` (or a related function) advances that
cycler. This means multiple distinguishable lines can be plotted without manually specifying
styles.

sysplot extends this cycler to include both color and linestyle. The 
goal is consistent styling, including black-and-white print contexts, while
remaining compatible with Matplotlib and Seaborn defaults.

However, different plotting functions use independent cyclers. For example,
``scatter()`` does not follow the cycler from ``plot()``, as you can see 
in this

.. minigallery:: examples/matplotlib_cycler.py

Another subtle behavior appears when the cycler contains both color and
linestyle -- as configured in sysplot by :func:`sysplot.apply_config` -- the internal cycle
may still advance even if you override only one property (e.g.
``color=...``), but not both (``color=..., linesle=...``). This can produce unexpected style offsets later in a figure.

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
and therefore show only one style. Additionally, all plots from sysplot should
be on the same cycler with the plot() function. Some users may also want that 
``scatter()`` and ``plot()`` share their style cycler and advance each other.

**To support this, sysplot provides** :func:`sysplot.get_style`, **which returns
a style dictionary derived from the configured cycler.** The return value may look like this::

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

2. Retrieve the next style for an axis
"""""""""""""""""""""""""""""""""""""""""

Alternatively, the next style can be determined for a specific axis::

    style = get_style(ax=ax)
    ax.scatter(x, y, **style)

In this mode, sysplot determines the next style that would be used by
``plot()`` on that axis and returns it as a dictionary.
This helps keep functions such as ``scatter()`` visually consistent with the
line-style progression used by ``plot()``.

Both usage patterns are shown here:

.. minigallery:: examples/get_style.py

Recommended System Modelling
-------------------------------

A recommended workflow for modeling systems is shown below that uses numpy and control library. Following this
structure makes it convenient to pass data into sysplot plotters. Other
approaches are also valid as long as the resulting arrays match the expected
function arguments.

The example below defines a second-order system with a root and computes its
frequency response.

.. minigallery:: examples/modelling.py


Plotting Functions
---------------------

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


Annotating the Figure
----------------------

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

To repeat tick labels on all axes of a figure with shared axes, use :func:`sysplot.restore_tick_labels`. 

All of these functions are demonstrated here

.. minigallery:: examples/quick_start.py

Configuration
-----------------------------

An oppinonated design choice is used for the sysplot module. It leverages seaborn styles and Matploblibs defaults, but makes opnionated changes to these defaults. To activate these changes, the user must call :func:`sysplot.apply_config`. Changes can be configuredby using
:class:`sysplot.SysplotConfig`. Please refer to the documentation for these functions and classes for more details or check the following example

.. minigallery:: examples/apply_config.py

Why this package exists
------------------------

Sysplot originated from work at Hochschule Karlsruhe, where the
diagrams used in the lecture *System and Signal Theory* by Prof. 
Dr.-Ing. Manfred Strohrmann were revised.

A central requirement was that all figures should share a *consistent
visual style* and meet a *high publication standard*. To achieve this,
a global configuration system was introduced to control styling,
figure dimensions, and export behavior.

Because many diagrams appear repeatedly in the lecture material,
common plotting tasks were automated. Additional utilities were created
to improve the visual clarity of figures.
