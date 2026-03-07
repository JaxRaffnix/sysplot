"""
Set Major Ticks
=====================================

:func:`sysplot.set_major_ticks` replaces numeric tick labels with reduced
fractions of a unit. The tick step is ``unit * numerator / denominator``.
The ``mode`` parameter controls placement: ``"repeating"`` fills the visible
range, ``"single"`` places ticks at ``0`` and ``step`` only, and
``"symmetric"`` places ticks at ``-step``, ``0``, and ``step``.
"""

import numpy as np
import matplotlib.pyplot as plt
import sysplot as ssp

ssp.apply_config()

x = np.linspace(-2 * np.pi, 2 * np.pi, 600)

fig, axes = plt.subplots(1, 3, figsize=ssp.get_figsize(ncols=3))

# repeating, denominator=2 → ticks every pi/2
ax = axes[0]
ax.plot(x, np.sin(x))
ssp.set_major_ticks(label=r"$\pi$", unit=np.pi, mode="repeating", numerator=1, denominator=2, axis=ax.xaxis)
ax.set(title='mode="repeating", denom=2', xlabel=r"$t$ [rad]", ylabel="amplitude")

# single, numerator=1, denominator=1 → ticks at 0 and pi
ax = axes[1]
ax.plot(x, np.sin(x))
ssp.set_major_ticks(label=r"$\pi$", unit=np.pi, mode="single", numerator=1, denominator=1, axis=ax.xaxis)
ax.set(title='mode="single", denom=1', xlabel=r"$t$ [rad]")

# symmetric, numerator=3, denominator=4 → ticks at -3pi/4, 0, 3pi/4
ax = axes[2]
ax.plot(x, np.sin(x))
ssp.set_major_ticks(label=r"$\pi$", unit=np.pi, mode="symmetric", numerator=3, denominator=4, axis=ax.xaxis)
ax.set(title='mode="symmetric", num=3, denom=4', xlabel=r"$t$ [rad]")

plt.show()

