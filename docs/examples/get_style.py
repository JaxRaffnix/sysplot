"""
Get Style Example
=====================================

Demonstrates style selection with ``get_style`` using both fixed index
and axes-driven cycling.
"""

import matplotlib.pyplot as plt
import sysplot as ssp

ssp.apply_config()

fig, ax = plt.subplots()

style0 = ssp.get_style(index=0)
style1 = ssp.get_style(index=1)

ax.plot([0, 1, 2], [0, 1, 0], label="index=0", **style0)        
ax.plot([0, 1, 2], [0.2, 0.6, 0.2], label="index=1", **style1)
ax.scatter([0, 0.5, 1, 1.5, 2], [0.4, 0.3, 0.2, 0.3, 0.4], label="ax cycle", **ssp.get_style(ax=ax)) # -> style 2
ax.plot([0, 1, 2], [0.4, 0.2, 0.4], label="ax cycle",) # -> style 3

ax.set(title="get_style()", xlabel="x", ylabel="y")
ax.legend()
plt.show()
