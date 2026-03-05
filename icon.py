import sysplot as ssp
import matplotlib.pyplot as plt
import seaborn as sns
import control as ctrl
import numpy as np

ssp.apply_config(
    font_size=15,
    linewidth=3,
    seaborn_style="ticks",
    poles_zeros_markersize=20
    # seaborn_context="talk"
)


omega_n = 2.5   # natural frequency [rad/s]
zeta = 0.6      # damping ratio (< 1 for underdamped, gives resonance peak)
z = 1.0         # zero location at s = -1

system = ctrl.TransferFunction([1, 0], [1, 2, 3])
omega = np.logspace(-3, 3, 4000)
mag, phase, _ = ctrl.frequency_response(system, omega)
H = mag * np.exp(1j * phase)

poles = ctrl.poles(system)
zeros = ctrl.zeros(system)

fig, ax = plt.subplots()
ssp.emphasize_coord_lines(fig)
ssp.plot_poles_zeros(poles=poles, zeros=zeros)
ssp.plot_unit_circle(ax=ax)

fig, axes = plt.subplots(1, 2, figsize=ssp.get_figsize(1, 2), sharex=True)
ssp.plot_bode(
    mag, phase, omega, 
    axes=axes,
    minor_ticks=True,
    mag_to_dB=True, x_to_log=True,
    tick_numerator=1, tick_denominator=4
)

fig, ax = plt.subplots()
ssp.plot_nyquist(
    real=np.real(H),
    imag=np.imag(H),
    ax=ax,
    mirror=True,           # show complex conjugate
    arrow_position=0.5, 
    arrow_size=35,
    label=r"$H(j\omega)$"
)
ax.legend()

plt.show()