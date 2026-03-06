"""
System Modelling Example
=====================================

Shows the recommended way to define a transfer function and derive the data
neded by sysplot's plotting functions: magnitude and phase arrays from
``ctrl.frequency_response``, and poles and zeros from ``ctrl.poles``/
``ctrl.zeros``. No figure is produced — this example focuses on the data
model.
"""


import numpy as np
import control as ctrl

# Second-order system with root:
# H(s) = ωₙ² (s + z) / (s² + 2ζωₙ s + ωₙ²)

omega_n = 2.5   # natural frequency [rad/s]
zeta = 0.6      # damping ratio (<1 → underdamped)
z = 1.0         # zero location at s = -1

system = ctrl.TransferFunction(
    [omega_n**2, omega_n**2 * z],
    [1, 2*zeta*omega_n, omega_n**2],
)

# Frequency grid
omega = np.logspace(-3, 3, 4000)

# Frequency response
mag, phase, _ = ctrl.frequency_response(system, omega)

# Convert to complex frequency response
H = mag * np.exp(1j * phase)

# System poles and zeros
poles = ctrl.poles(system)
zeros = ctrl.zeros(system)