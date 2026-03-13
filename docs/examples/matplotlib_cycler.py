"""Matplotlib Cycler Behavior
=====================================

Matplotlib's ``plot`` and ``scatter`` maintain **independent** style cyclers,
so their colors are not synchronised. Manually setting ``color=`` on one call
advances only that call's slot without influencing the other. This background
is useful for understanding why sysplot's :func:`sysplot.get_style` is the
preferred way to obtain consistent styles across different plot commands.
"""

import numpy as np
import matplotlib.pyplot as plt

# TODO: this needs work. both scatter and plot must be on the same axis to show the result.

x = np.array([0, 1])

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle("plot() and scatter() cycle independently")

# ── plot() cycler ─────────────────────────────────────────────────────────────
ax1.plot(x, x * 0 + 3, label="plot()  → C0")           # C0
ax1.plot(x, x * 0 + 2, color="gray",
         label="plot(color='gray')  → C1 slot skipped") # C1 consumed, draws gray
ax1.plot(x, x * 0 + 1, label="plot()  → C2")           # C2
ax1.plot(x, x * 0 + 0, label="plot()  → C3")           # C3

ax1.set(title="plot() cycle", xlim=(-0.2, 1.2), ylim=(-0.5, 3.5))
ax1.legend(ncols=2)

# ── scatter() cycler — resets to C0 independently ────────────────────────────
ax2.scatter(x, x * 0 + 3, s=60, label="scatter()  → C0")           # C0
ax2.scatter(x, x * 0 + 2, color="gray", s=60,
            label="scatter(color='gray')  → C1 slot skipped")       # C1 consumed, draws gray
ax2.scatter(x, x * 0 + 1, s=60, label="scatter()  → C2")           # C2
ax2.scatter(x, x * 0 + 0, s=60, label="scatter()  → C3")           # C3

ax2.set(title="scatter() cycle (independent)", xlim=(-0.2, 1.2), ylim=(-0.5, 3.5))
ax2.legend(ncols=2)

plt.show()
