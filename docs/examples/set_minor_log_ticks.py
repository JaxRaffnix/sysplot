"""
Set Minor Log Ticks
=====================================

:func:`sysplot.set_minor_log_ticks` adds unlabeled minor ticks at every
subdivision of a logarithmic axis. Tick direction is controlled by
``tick_direction`` and defaults to :attr:`~sysplot.SysplotConfig.tick_direction`.
"""

import numpy as np
import matplotlib.pyplot as plt
import sysplot as ssp

ssp.apply_config()

x = np.logspace(0, 3, 300)
y = 1 / x

fig, axes = plt.subplots(1, 3, figsize=ssp.get_figsize(ncols=3))

for ax, direction in zip(axes, ["in", "out", "inout"]):
    ax.set_xscale("log")
    ax.plot(x, y)
    ssp.set_minor_log_ticks(axis=ax.xaxis, tick_direction=direction)
    ax.set(title=f"tick_direction={direction!r}", xlabel="frequency", ylabel="magnitude")

plt.show()
