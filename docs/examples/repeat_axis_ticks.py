"""
Repeat Axis Ticks Example
=====================================

Demonstrates a short example of the repeat_axis_ticks function.
"""

import matplotlib.pyplot as plt
import sysplot as ssp

ssp.apply_config()

fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)

# shows the axis ticks labels on every subplot, even when axes are shared.
ssp.repeat_axis_ticks(fig)

for a in ax.flatten():
    a.plot([-2, 2], [-1, 1])
    a.set(xlabel="x", ylabel="y")

plt.suptitle("Repeat Axis Ticks")

plt.show()
