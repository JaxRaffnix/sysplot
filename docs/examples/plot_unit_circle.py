"""
Plot Unit Circle
=====================================

:func:`sysplot.plot_unit_circle` draws a circle of radius 1 centered at a
given origin. Line color, style, and width default to the grid style from
Matplotlib's rcParams. Combined with :func:`sysplot.plot_poles_zeros`, it
provides the standard stability boundary for discrete-time pole-zero maps.
"""

import numpy as np
import matplotlib.pyplot as plt
import sysplot as ssp

ssp.apply_config()

# Discrete-time poles: some inside, some outside the unit circle
poles_stable = np.array([0.5 + 0.4j, 0.5 - 0.4j, -0.3])
poles_unstable = np.array([0.85 + 0.55j, 0.85 - 0.55j])

fig, ax = plt.subplots(figsize=ssp.get_figsize())

ssp.plot_unit_circle(ax=ax)
ssp.plot_poles_zeros(poles=poles_stable, label="Stable", ax=ax)
ssp.plot_poles_zeros(poles=poles_unstable, label="Unstable", ax=ax)

ax.set(title="Unit circle with pole-zero map", xlabel=r"Re[$z$]", ylabel=r"Im[$z$]")
ax.legend()
plt.show()
