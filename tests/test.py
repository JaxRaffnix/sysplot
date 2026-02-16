# example.py — test script for ssp.py

import numpy as np
import matplotlib.pyplot as plt
import control as ctrl

# Import custom moduel
import sysplot as ssp
from mpl_toolkits.mplot3d import Axes3D

# ---------------------------------------------------------------------
# Helper depending on LANGUAGE
# ---------------------------------------------------------------------
if ssp.LANGUAGE == "de":
    xlabel = "Zeit t [s]"
    ylabel = "Amplitude"
    title  = "Beispielplot"
else:
    xlabel = "Time t [s]"
    ylabel = "Amplitude"
    title  = "Example Plot"


def test_get_style(save: bool = True):
    """Test manual application of color + linestyle via get_style()."""
    x = np.linspace(-2, 2, 400)

    fig, ax = plt.subplots(figsize=ssp.FIGURE_SIZE)
    ssp.highlight_axes(fig)

    for i in range(10):
        y = np.sin(x + i * 0.2) + 0.15 * i
        ax.plot(x, y, **ssp.get_style(i), label=f"Style index: {i}")

    ax.set_title("Manual Style Access (get_style)")
    ax.legend()
    if save:
        ssp.save_current_figure(chapter=0, number=2, folder="test_images", language=ssp.LANGUAGE)

def test_dynamic_subplots(save: bool = True):
    """Subplot test: verify dynamic figure size helper."""
    x = np.linspace(-2, 2, 400)
    nrows, ncols = 2, 3

    figsize = ssp.get_figsize(nrows=nrows, ncols=ncols)
    fig3, axs = plt.subplots(nrows, ncols, figsize=figsize)
    ssp.highlight_axes(fig3)

    for idx, ax in enumerate(axs.flat):
        ax.plot(x, np.exp(-(idx+1) * x**2))
        ax.set_title(f"Subplot {idx+1}")

    fig3.suptitle("Dynamic Figure Size (get_figsize)")
    if save:
        ssp.save_current_figure(chapter=0, number=3, folder="test_images", language=ssp.LANGUAGE)

def test_stem(save: bool = True):
    """Stem test: Custom Stem plotter."""
    x4 = np.arange(0, 10, 1)
    y4 = np.random.rand(10)
    
    fig4, ax4 = plt.subplots(figsize=ssp.FIGURE_SIZE)
    ssp.highlight_axes(fig4)
    ssp.plot_stem(x4, y4-0.5, marker="^", markers_outwards=True)
    ssp.plot_stem(x4+0.5, y4, baseline=0.25, show_baseline=False)
    
    if save:
        ssp.save_current_figure(chapter=0, number=4, folder="test_images", language=ssp.LANGUAGE)

def test_nyquist_plot(save: bool = True):
    """Nyquist plot test: Verify Nyquist plot with arrow positioning."""
    omega = np.logspace(-3, 8, 2000)

    system = ctrl.tf([1, 100], [1, 10])
    mag, phase, _ = ctrl.frequency_response(system, omega)
    H = mag * np.exp(1j * phase)

    fig5, ax5 = plt.subplots(figsize=ssp.FIGURE_SIZE)
    ssp.highlight_axes(fig5)
    ssp.plot_nyquist(np.real(H), np.imag(H), arrow_position=0.4, style_index=0)
    ax5.set_title("Nyquist Plot with Arrow Positioning")

    if save:
        ssp.save_current_figure(chapter=0, number=5, folder="test_images", language=ssp.LANGUAGE)


def test_bode_plot(save: bool = True):
    """Bode plot test: Verify Bode plot with dB option."""
    omega = np.logspace(-1, 8, 2000)
    system = ctrl.tf([1, 100], [1, 10])
    mag, phase, _ = ctrl.frequency_response(system, omega)
    
    fig6, ax6 = plt.subplots(1, 2, figsize=ssp.get_figsize(1, 2))
    ssp.highlight_axes(fig6)
    ssp.plot_bode(mag, phase, omega, axes=ax6)
    fig6.suptitle("Bode Plot in dB")

    if save:
        ssp.save_current_figure(chapter=0, number=6, folder="test_images", language=ssp.LANGUAGE)


def test_minor_ticks(save: bool = True):
    x = np.logspace(-2, 8, 200)
    y = x*x

    fig, ax = plt.subplots()
    ssp.highlight_axes(fig)
    plt.semilogx(x, y)
    ssp.set_minor_log_ticks(axis=ax.xaxis)

    if save:
        ssp.save_current_figure(chapter=0, number=7, folder="test_images", language=ssp.LANGUAGE)


def test_major_ticks(save: bool = True):
    """Test major ticks helper function."""
    x = np.linspace(-4, 20, 2000)
    y = np.sin(x) * 5

    fig, ax = plt.subplots()
    ssp.highlight_axes(fig)
    plt.plot(x, y)
    ssp.set_major_tick_labels(label=r"$\pi$", unit=np.pi, mode="single", axis=ax.xaxis)
    ssp.set_major_tick_labels(label=r"t", unit=2, denominator=5, numerator=2, axis=ax.yaxis)

    if save:
        ssp.save_current_figure(chapter=0, number=8, folder="test_images", language=ssp.LANGUAGE)

def test_highlight_axes(save: bool = True):
    """Test highlight_axes function."""
    x = np.linspace(-10, 10, 400)
    y = np.linspace(-10, 10, 400)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))

    fig = plt.figure()
    # ssp.highlight_axes(fig)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z)
    # ssp.highlight_axes(fig)

    if save:
        ssp.save_current_figure(chapter=0, number=9, folder="test_images", language=ssp.LANGUAGE)

def test_plot_poles_zeros(save: bool = True):
    """Test plot_poles_zeros function."""
    poles = np.array([-1 + 1j, -1 - 1j, -2])
    zeros = -1

    fig, ax = plt.subplots()
    ssp.highlight_axes(fig)
    ssp.plot_poles_zeros(poles, zeros, ax=ax)
    ax.set_title("Poles and Zeros Plot")

    if save:
        ssp.save_current_figure(chapter=0, number=10, folder="test_images", language=ssp.LANGUAGE)

# ---------------------------------------------------------------------
# Manual main() for toggling tests on/off
# ---------------------------------------------------------------------
def main():
    """Simple main to toggle tests and control saving of figures."""

    SAVE_IMAGES = False

    test_get_style(save=SAVE_IMAGES)
    test_dynamic_subplots(save=SAVE_IMAGES)
    test_stem(save=SAVE_IMAGES)
    test_nyquist_plot(save=SAVE_IMAGES)
    test_bode_plot(save=SAVE_IMAGES)
    test_minor_ticks(save=SAVE_IMAGES)
    test_major_ticks(save=SAVE_IMAGES)
    test_highlight_axes(save=SAVE_IMAGES)
    test_plot_poles_zeros(save=SAVE_IMAGES)

    # Display all figures at the end
    plt.show()


if __name__ == "__main__":
    main()
