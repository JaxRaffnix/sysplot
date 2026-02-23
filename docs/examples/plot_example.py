"""
Implementation of the SysPlot module
=====================================

Demonstrates implementation of different functions provided by the sysplot module
"""


import numpy as np
import matplotlib.pyplot as plt
import control as ctrl

# custom import
import sysplot as ssp

ssp.apply_config()


# ___________________________________________________________________
#  Localization


LANGUAGE = ssp.LANGUAGE
LOCALIZATION = {
    "de": {
        "amplitude": "Amplitude",
        "phase": "Phase",
        "frequency": r"Kreisfrequenz $\omega$ / rad/s",
        "magnitude": r"Betrag $A(\omega)$ / dB",
        "phase_w": r"Phase $\phi(\omega)$ / rad"
    },
    "en": {
        "amplitude": "Amplitude",
        "phase": "Phase",
        "frequency": r"Frequency $\omega$ / rad/s",
        "magnitude": r"Magnitude $A(\omega)$ / dB",
        "phase_w": r"Phase $\phi(\omega)$ / rad"
    }
}
text = LOCALIZATION[LANGUAGE]


# ___________________________________________________________________
#  Modelling


# sinus
x = np.linspace(-1, 5, 50)
T = 1
y = np.sin(2*np.pi / T * x)

# first order system
K = 1.5
T = 0.8
system = ctrl.TransferFunction([K], [T, 1])
omega = np.logspace(-3, 2, 400)
mag, phase, _ = ctrl.frequency_response(system, omega)
H = mag * np.exp(1j * phase)


# ___________________________________________________________________
#  Plotting


# stem plot with highlighted parameter
fig0, ax0 = plt.subplots(figsize=ssp.get_figsize())
ssp.highlight_axes(fig0)

ssp.plot_stem(x, y, ax=ax0)

ssp.set_major_tick_labels(label=r"$T$", unit=1, mode="symmetric", axis=ax0.xaxis)
ax0.set_xlabel(r"Zeit $t$")
ax0.set_ylabel(r"$\sin(\frac{2\pi}{T} t)$")

# ssp.save_current_figure(chapter=0, number=0, language=LANGUAGE)

# poles and zeros plot
fig1, ax1 = plt.subplots(figsize=ssp.get_figsize())
ssp.highlight_axes(fig1)

ssp.plot_poles_zeros(poles=[-1/T], zeros=[], ax=ax1, label=r"$H(s)$")
ax1.legend()

# ssp.save_current_figure(chapter=0, number=1, language=LANGUAGE)

# nyquist plot
fig2, ax2 = plt.subplots(figsize=ssp.get_figsize())
ssp.highlight_axes(fig2)

ssp.plot_nyquist(np.real(H), np.imag(H), ax=ax2, label=r"$H(j\omega)$")
ax2.set_xlabel(r"Re{$H(j\omega)$}")
ax2.set_ylabel(r"Im{$H(j\omega)$}")
ax2.legend()

# ssp.save_current_figure(chapter=0, number=2, language=LANGUAGE)


# bode plot
fig3, ax3 = plt.subplots(nrows=1, ncols=2, figsize=ssp.get_figsize(nrows=1, ncols=2))
ssp.highlight_axes(fig3)

ssp.plot_bode(mag, phase, omega, axes=ax3, label=r"$A(\omega)$")

ax3[0].set_title(text["amplitude"])
ax3[0].set_xlabel(text["frequency"])
ax3[0].set_ylabel(text["magnitude"])
ax3[0].legend()

ax3[1].set_title(text["phase"])
ax3[1].set_xlabel(text["frequency"])
ax3[1].set_ylabel(text["phase_w"])
ax3[1].legend()

# ssp.save_current_figure(chapter=0, number=3, language=LANGUAGE)


# ___________________________________________________________________
#  Save and show


plt.show()