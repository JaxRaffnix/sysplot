"""
Minor Log Ticks Example
=====================================

:func:`sysplot.set_minor_log_ticks` adds minor tick marks at every integer
power of ten on a logarithmic axis. The result is a log-scale magnitude plot
where decade boundaries are clearly marked between major labels.
"""

import numpy as np
import matplotlib.pyplot as plt
import sysplot as ssp

ssp.apply_config()

x = np.logspace(0, 3, 300)
y = 1 / x

fig, ax = plt.subplots()
ax.set_xscale("log")
ax.plot(x, y)

# shows minor ticks at every integer power of 10, i.e. 1, 10, 100, etc.
ssp.set_minor_log_ticks(axis=ax.xaxis)

ax.set_title("Log axis with minor ticks")
ax.set_xlabel("frequency")
ax.set_ylabel("magnitude")

plt.show()
