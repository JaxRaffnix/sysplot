"""
Explanation for Matplotlib Cyclers
=====================================

Demonstrates the plotting cycler behavior in Matplotlib.
"""

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-2, 2, 10)

def y(i):
    return np.sin(x + i * 0.2) + 0.15 * i

fig, ax = plt.subplots()

plt.plot(x, y(0), label="Plot 0")               # -> plot style 0


plt.plot(x, y(1), color="gray", label="Plot 1") # -> custom style
plt.plot(x, y(2), label="Plot 2")               # -> style 1

plt.scatter(x, -y(0), label="Scatter 0")                       # -> scatter style 0
plt.scatter(x, -y(1), color="gray", linestyle="-", label="Scatter 1")                       # -> custom style
plt.scatter(x, -y(2), label="Scatter 2")                       # -> scatter style 1

plt.plot(x, y(3), label="Plot 3")                          # -> plot style 2



plt.show()