"""
Minimum Example
=====================================

Demonstrates a short example of the core functionality of the sysplot module.
"""

import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
import sysplot as ssp

ssp.apply_config()  # apply default configuration 

# Generate frequency response
omega = np.logspace(-2, 2, 300)
system = ctrl.tf([6.25], [1, 3 , 6.25])
mag, phase, _ = ctrl.frequency_response(system, omega)

# Create Bode plot
fig, axes = plt.subplots(1, 2, figsize=ssp.get_figsize(1, 2))
ssp.plot_bode(mag, phase, omega, axes=axes)

# Labels
axes[0].set(title="Magnitude", xlabel=r"$\omega$ [rad/s]", ylabel="dB")
axes[1].set(title="Phase", xlabel=r"$\omega$ [rad/s]", ylabel="rad")
plt.show()