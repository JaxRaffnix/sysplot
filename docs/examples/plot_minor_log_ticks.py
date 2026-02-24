"""
Minor Log Ticks Example
=====================================

Demonstrates a short example of the set_minor_log_ticks function.
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

ssp.set_minor_log_ticks(axis=ax.xaxis)

ax.set_title("Log axis with minor ticks")
ax.set_xlabel("frequency")
ax.set_ylabel("magnitude")

plt.show()
