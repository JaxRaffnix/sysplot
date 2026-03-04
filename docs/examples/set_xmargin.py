"""
X-Margin Example
=====================================

Demonstrates a short example of the set_xmargin function.
"""

import matplotlib.pyplot as plt
import sysplot as ssp

ssp.apply_config()

fig, ax = plt.subplots()

# reenables x-axis margins around the data 
ssp.set_xmargin(ax, use_margin=True)

plt.plot([2, 4], [1, 3])

ax.set(title="X-Margin Example", xlabel="x", ylabel="y")

plt.show()
