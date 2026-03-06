"""
Save Current Figure
=====================================

:func:`sysplot.save_current_figure` saves the active Matplotlib figure to a
file next to the calling script. It uses a structured naming convention
``Bild_{chapter}_{number}_{script}{_suffix}.{fmt}`` and creates the output
directory ``{folder}/{language}/`` automatically. The file format defaults to
the value in :class:`sysplot.SysplotConfig`.
"""

import matplotlib.pyplot as plt
import numpy as np
import sysplot as ssp

ssp.apply_config()

x = np.linspace(0, 2 * np.pi, 200)

fig, ax = plt.subplots(figsize=ssp.get_figsize())
ax.plot(x, np.sin(x), label="sin(x)")
ax.plot(x, np.cos(x), label="cos(x)")
ax.set(title="Example Figure", xlabel="x", ylabel="y")
ax.legend()

# Saved as: images/en/Bild_1_2_save_current_figure.pdf
path = ssp.save_current_figure(chapter=1, number=2, language="en")
print(f"Saved to: {path}")

plt.show()

# With an optional suffix to distinguish variants
fig, ax = plt.subplots(figsize=ssp.get_figsize())
ax.plot(x, np.sin(x))
ax.set(title="Variant A", xlabel="x", ylabel="y")

# Saved as: images/en/Bild_1_2_save_current_figure_a.pdf
path = ssp.save_current_figure(chapter=1, number=2, language="en", suffix="a")
print(f"Saved to: {path}")

plt.show()
