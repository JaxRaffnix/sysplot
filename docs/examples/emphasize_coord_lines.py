"""
Emphasize Coordinate Lines Example
=====================================

:func:`sysplot.emphasize_coord_lines` draws the x = 0 and y = 0 lines with
increased visual weight, providing clear reference axes. T
"""

import matplotlib.pyplot as plt
import sysplot as ssp

ssp.apply_config()

fig, ax = plt.subplots()

# TODO: show zorder and linewidth and color arguments

# the coordinate axes are empphazised
ssp.emphasize_coord_lines(fig)

plt.plot([-2, 2], [-1, 1])

ax.set(title="Emphasize Coord Lines", xlabel="x", ylabel="y")

plt.show()
