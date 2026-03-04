"""
Emphasize Coordinate Lines Example
=====================================

Demonstrates a short example of the emphasize_coord_lines function.
"""

import matplotlib.pyplot as plt
import sysplot as ssp

ssp.apply_config()

fig, ax = plt.subplots()

# the coordinate axes are empphazised
ssp.emphasize_coord_lines(fig)

plt.plot([-2, 2], [-1, 1])

ax.set(title="Emphasize Coord Lines", xlabel="x", ylabel="y")

plt.show()
