"""
Angle Plot Example
=====================================

:func:`sysplot.plot_angle` draws a labeled arc between two vectors sharing a
common center, annotating the angle between them. The result is a figure with
two line segments from the origin and an arc with a :math:`\\theta` label
indicating the enclosed angle.
"""

import matplotlib.pyplot as plt
import sysplot as ssp

ssp.apply_config()

# TODO: show off more options for plot_angle():
# with equal axes on an otherwise distored plot
# different sizes, units and textposition 
# different colors in textkwargs 

center = (0.0, 0.0)
p1 = (1.0, 0.0)
p2 = (0.6, 0.8)

fig, ax = plt.subplots()
ax.plot(*zip(center, p1), label="vector 1")
ax.plot(*zip(center, p2), label="vector 2")

# shows the arc between the two vectors center->p1, center->p2 with an annotation text
ssp.plot_angle(
    center, p1, p2, 
    text=r"$\theta$", 
    ax=ax,
    equal_axes=True,
)

ax.set(title="Angle annotation", xlabel="x", ylabel="y")
ax.legend()

plt.show()
