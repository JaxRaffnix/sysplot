"""
Add Origin Example
=====================================

Demonstrates a short example of the add_origin function.
"""

import matplotlib.pyplot as plt
import sysplot as ssp

ssp.apply_config()

fig, ax = plt.subplots()

# makes the origin (0,0) visible in the plot even if no data points are plotted there
ssp.add_origin(ax)

plt.plot([2, 4], [1, 3])

ax.set(title="Add Origin Example", xlabel="x", ylabel="y")

plt.show()
