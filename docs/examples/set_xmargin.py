"""
X-Margin Example
=====================================

:func:`sysplot.apply_config` sets the x-margin to zero by default so that
data fills the full axis width. :func:`sysplot.set_xmargin` re-enables
horizontal padding on individual axes when whitespace around the data is
preferred.
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
