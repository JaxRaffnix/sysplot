"""Plot Stem
=====================================

:func:`sysplot.plot_stem` is a styled wrapper around ``Axes.stem``.
It applies the active style cycle automatically and supports two key options:

- ``directional_markers=True`` flips the marker direction for stems below the
  baseline (e.g., ``^`` above, ``v`` below).
- ``show_baseline=False`` hides the horizontal baseline.
"""

import numpy as np
import matplotlib.pyplot as plt
import sysplot as ssp

ssp.apply_config()

n = np.arange(-8, 9)
y = np.sinc(n / 4)

fig, axes = plt.subplots(1, 3, figsize=ssp.get_figsize(ncols=3))

# Default: round markers, baseline shown
ssp.plot_stem(n, y, ax=axes[0])
axes[0].set(title="default", xlabel="n", ylabel="y[n]")

# Outward-pointing triangular markers
ssp.plot_stem(n, y, ax=axes[1], marker="^", directional_markers=True)
axes[1].set(title="directional_markers=True", xlabel="n")

# No baseline
ssp.plot_stem(n, y, ax=axes[2], show_baseline=False)
axes[2].set(title="show_baseline=False", xlabel="n")

plt.show()
