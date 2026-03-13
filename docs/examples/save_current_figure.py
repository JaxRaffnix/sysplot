"""Save Current Figure
=====================================

:func:`sysplot.save_current_figure` saves the active Matplotlib figure to a
file next to the calling script. It uses a structured naming convention
``Bild_{chapter}_{number}_{script}{_suffix}.{fmt}`` and creates the output
directory ``{folder}/{language}/`` automatically. The file format, output
folder, and transparency all default to values in :class:`sysplot.SysplotConfig`.
"""

import matplotlib.pyplot as plt
import numpy as np
import sysplot as ssp

ssp.apply_config()

x = np.linspace(0, 2 * np.pi, 200)


def make_fig(title):
    fig, ax = plt.subplots(figsize=ssp.get_figsize())
    ax.plot(x, np.sin(x))
    ax.set(title=title, xlabel="x", ylabel="y")


# Default: fmt and folder from SysplotConfig
# → images/en/Bild_1_1_save_current_figure.pdf
make_fig("Default (pdf)")
path = ssp.save_current_figure(chapter=1, number=1, language="en")
print(f"Default:     {path}")

# Custom format: save as PNG
# → images/en/Bild_1_2_save_current_figure.png
make_fig("PNG format")
path = ssp.save_current_figure(chapter=1, number=2, language="en", fmt="png")
print(f"PNG:         {path}")

# Transparent background (useful for slides/reports)
# → images/en/Bild_1_3_save_current_figure.pdf
make_fig("Transparent background")
path = ssp.save_current_figure(chapter=1, number=3, language="en", transparent=True)
print(f"Transparent: {path}")

# Custom output folder
# → figures/en/Bild_1_4_save_current_figure.pdf
make_fig("Custom folder")
path = ssp.save_current_figure(chapter=1, number=4, language="en", folder="figures")
print(f"Custom folder: {path}")

# Suffix to distinguish variants of the same figure number
# → images/de/Bild_2_1_save_current_figure_v2.pdf
make_fig("German variant with suffix")
path = ssp.save_current_figure(chapter=2, number=1, language="de", suffix="v2")
print(f"Suffix:      {path}")

plt.show()

