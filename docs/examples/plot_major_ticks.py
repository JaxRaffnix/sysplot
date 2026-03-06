"""
Major Ticks Example
=====================================

:func:`sysplot.set_major_ticks` replaces default numeric tick labels with
fractional multiples of a unit. Here the x-axis of a sine plot is labelled
in multiples of :math:`\\pi/2`, producing ticks at
0, :math:`\\pi/2`, :math:`\\pi`, :math:`3\\pi/2`, :math:`2\\pi`.
"""

import numpy as np
import matplotlib.pyplot as plt
import sysplot as ssp

ssp.apply_config()

x = np.linspace(0, 2 * np.pi, 400)
y = np.sin(x)

fig, ax = plt.subplots()
ax.plot(x, y)

# shows x-axis ticks at integer mutiples of pi/2 with labels 0, pi/2, pi, 3/2pi, ...
ssp.set_major_ticks(
    label=r"$\pi$",
    unit=np.pi,
    numerator=1,
    denominator=2,
    axis=ax.xaxis,
)

ax.set_title("Sine with pi ticks")
ax.set_xlabel(r"$t$ [rad]")
ax.set_ylabel("amplitude")

plt.show()

