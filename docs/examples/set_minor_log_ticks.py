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

axes[0].set_xscale("log")
axes[0].plot(x, y)
ssp.set_minor_log_ticks(axis=axes[0].xaxis, tick_direction="in")
axes[0].set(title='tick_direction="in"', xlabel="frequency", ylabel="magnitude")

axes[1].set_xscale("log")
axes[1].plot(x, y)
ssp.set_minor_log_ticks(axis=axes[1].xaxis, tick_direction="out")
axes[1].set(title='tick_direction="out"', xlabel="frequency")

axes[2].set_xscale("log")
axes[2].plot(x, y)
ssp.set_minor_log_ticks(axis=axes[2].xaxis, tick_direction="inout")
axes[2].set(title='tick_direction="inout"', xlabel="frequency")

plt.show()
