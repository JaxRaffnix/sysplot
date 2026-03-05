import sysplot as ssp
import matplotlib.pyplot as plt
import seaborn as sns
import control as ctrl
import numpy as np

ssp.apply_config(
    font_size=15,
    linewidth=2,
    seaborn_style="ticks",
    # seaborn_context="talk"
)


omega_n = 2.5   # natural frequency [rad/s]
zeta = 0.6      # damping ratio (< 1 for underdamped, gives resonance peak)
z = 1.0         # zero location at s = -1

system = ctrl.TransferFunction([omega_n**2, omega_n**2 * z], [1, 2*zeta*omega_n, omega_n**2])
omega = np.logspace(-3, 3, 4000)
mag, phase, _ = ctrl.frequency_response(system, omega)
H = mag * np.exp(1j * phase)

fig, ax = plt.subplots()
ssp.emphasize_coord_lines(fig)

ssp.plot_nyquist(
    real=np.real(H),
    imag=np.imag(H),
    ax=ax,
    mirror=True,           # show complex conjugate
    arrow_position=0.5, 
    label=r"$H(j\omega)$"
)

ax.legend()

plt.show()