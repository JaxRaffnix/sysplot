"""Matplotlib cycler behavior.

This example highlights three ideas:
1. ``plot`` and ``scatter`` use independent style progression.
2. Manually setting ``color`` creates a custom style entry.
3. A ``scatter`` call does not affect the next ``plot`` style.
"""

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-2, 2, 10)


def y(i):
    return np.sin(x + i * 0.2) + 0.15 * i


fig, ax = plt.subplots()

# plot cycle
ax.plot(x, y(0), label="Plot 0")
ax.plot(x, y(1), color="gray", label="Plot manual")
ax.plot(x, y(2), label="Plot 2")

# scatter cycle (independent from plot cycle)
ax.scatter(x, -y(0), label="Scatter 0")
ax.scatter(x, -y(1), color="gray", label="Scatter manual")
ax.scatter(x, -y(2), label="Scatter 2")
ax.scatter(x, -y(3), label="Scatter 3")

# plot cycle continues from its own history
ax.plot(x, y(3), label="Plot 3")

ax.legend()

plt.show()