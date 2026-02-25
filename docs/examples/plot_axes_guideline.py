"""
Implementation of the Axis Guideline
=====================================

The four different possibilities of axis labels and ticks are shown.
"""

import matplotlib.pyplot as plt
import numpy as np
import control as ctrl

import sysplot as ssp

ssp.apply_config()


t0 = np.linspace(0, 5, 500)
y0 = np.cos(t0)

T1 = 2
w1 = np.logspace(-1, 2, 500)
system = ctrl.TransferFunction([1], [T1, 1])
mag, phase, _ = ctrl.frequency_response(system, w1)
phase = np.unwrap(phase)

T2 = 2
OMEGA_G = 2 * np.pi / T2
w2 = np.linspace(-4*OMEGA_G, 4*OMEGA_G, 2000)  # rad/s
y2 = np.abs(T2 * np.sinc(w2 / OMEGA_G))

T3 = 1.5
x3 = np.linspace(-2, 5, 500)
y3 = np.heaviside(x3, 1) - np.heaviside(x3 - T3, 1)

fig, axes = plt.subplots(2, 2, figsize=ssp.get_figsize(2, 2))
ssp.highlight_axes(fig)

axes[0, 0].plot(t0, y0)
axes[0, 0].set_xlabel("Zeit t / s")
axes[0, 0].set_ylabel(r"Spannung $u(t)$ / V")
axes[0, 0].set_title("Physikalisches Signal")

axes[0, 1].semilogx(w1, phase)
ssp.set_major_ticks(r"$\pi$", unit=np.pi, denominator=4, mode="repeating", axis=axes[0, 1].yaxis)
ssp.set_minor_log_ticks(axes[0, 1].xaxis)
axes[0, 1].set_ylabel(r"Phase $\varphi(\omega)$ / rad")
axes[0, 1].set_xlabel(r"Kreisfrequenz $\omega$ / rad/s")
axes[0, 1].set_title("Signal in Radian")

axes[1, 0].plot(w2 / OMEGA_G, y2 / T2, label=r"$|T \cdot \mathrm{sinc}(\frac{\omega}{\omega_G})|$")
axes[1, 0].set_ylabel(r"Spektrum $X(\omega)$ / T")
axes[1, 0].set_xlabel(r"Kreisfrequenz $\omega$ / $\omega_G$")
axes[1, 0].set_title("Normiertes Signal")
axes[1, 0].legend()

axes[1, 1].plot(x3, y3, label=r"$\sigma(t) - \sigma(t - t_0)$")
ssp.set_major_ticks(r"$t_0$", unit=T3, axis=axes[1, 1].xaxis, mode="symmetric")
axes[1, 1].set_ylabel("Signal $y(t)$")
axes[1, 1].set_xlabel(r"Zeit $t$")
axes[1, 1].set_title("Signal mit freiem Parameter")
axes[1, 1].legend()

# ssp.save_current_figure(language=ssp.LANGUAGE, chapter=0, number=0)
plt.show()