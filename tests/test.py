"""Pytest suite for sysplot."""

import os

import control as ctrl
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

import sysplot as ssp

matplotlib.use("Agg")

# ---------------------------------------------------------------------
# Helper depending on LANGUAGE
# ---------------------------------------------------------------------
if ssp.LANGUAGE == "de":
    xlabel = "Zeit t [s]"
    ylabel = "Amplitude"
    title = "Beispielplot"
else:
    xlabel = "Time t [s]"
    ylabel = "Amplitude"
    title = "Example Plot"


@pytest.fixture(scope="session")
def save_images() -> bool:
    """Control whether tests save images.

    Enable by setting SYSPLOT_SAVE_IMAGES=1 in the environment.
    """
    return os.getenv("SYSPLOT_SAVE_IMAGES", "0") == "1"


def test_get_style(save_images: bool):
    """Test manual application of color + linestyle via get_style()."""
    x = np.linspace(-2, 2, 400)

    fig, ax = plt.subplots(figsize=ssp.FIGURE_SIZE)
    ssp.highlight_axes(fig)

    for i in range(10):
        y = np.sin(x + i * 0.2) + 0.15 * i
        ax.plot(x, y, **ssp.get_style(i), label=f"Style index: {i}")

    ax.set_title("Manual Style Access (get_style)")
    ax.legend()
    if save_images:
        ssp.save_current_figure(chapter=0, number=2, folder="test_images", language=ssp.LANGUAGE)
    plt.close(fig)

def test_dynamic_subplots(save_images: bool):
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
    if save_images:
        ssp.save_current_figure(chapter=0, number=3, folder="test_images", language=ssp.LANGUAGE)
    plt.close(fig3)

def test_stem(save_images: bool):
    """Stem test: Custom Stem plotter."""
    x4 = np.arange(0, 10, 1)
    rng = np.random.default_rng(0)
    y4 = rng.random(10)
    
    fig4, ax4 = plt.subplots(figsize=ssp.FIGURE_SIZE)
    ssp.highlight_axes(fig4)
    ssp.plot_stem(x4, y4-0.5, marker="^", markers_outwards=True)
    ssp.plot_stem(x4+0.5, y4, baseline=0.25, show_baseline=False)
    
    if save_images:
        ssp.save_current_figure(chapter=0, number=4, folder="test_images", language=ssp.LANGUAGE)
    plt.close(fig4)

def test_nyquist_plot(save_images: bool):
    """Nyquist plot test: Verify Nyquist plot with arrow positioning."""
    omega = np.logspace(-3, 8, 2000)

    system = ctrl.tf([1, 100], [1, 10])
    mag, phase, _ = ctrl.frequency_response(system, omega)
    H = mag * np.exp(1j * phase)

    fig5, ax5 = plt.subplots(figsize=ssp.FIGURE_SIZE)
    ssp.highlight_axes(fig5)
    ssp.plot_nyquist(np.real(H), np.imag(H), arrow_position=0.4, style_index=0)
    ax5.set_title("Nyquist Plot with Arrow Positioning")

    if save_images:
        ssp.save_current_figure(chapter=0, number=5, folder="test_images", language=ssp.LANGUAGE)
    plt.close(fig5)


def test_bode_plot(save_images: bool):
    """Bode plot test: Verify Bode plot with dB option."""
    omega = np.logspace(-1, 8, 2000)
    system = ctrl.tf([1, 100], [1, 10])
    mag, phase, _ = ctrl.frequency_response(system, omega)
    
    fig6, ax6 = plt.subplots(1, 2, figsize=ssp.get_figsize(1, 2))
    ssp.highlight_axes(fig6)
    ssp.plot_bode(mag, phase, omega, axes=ax6)
    fig6.suptitle("Bode Plot in dB")

    if save_images:
        ssp.save_current_figure(chapter=0, number=6, folder="test_images", language=ssp.LANGUAGE)
    plt.close(fig6)


def test_minor_ticks(save_images: bool):
    x = np.logspace(-2, 8, 200)
    y = x*x

    fig, ax = plt.subplots()
    ssp.highlight_axes(fig)
    plt.semilogx(x, y)
    ssp.set_minor_log_ticks(axis=ax.xaxis)

    if save_images:
        ssp.save_current_figure(chapter=0, number=7, folder="test_images", language=ssp.LANGUAGE)
    plt.close(fig)


def test_major_ticks(save_images: bool):
    """Test major ticks helper function."""
    x = np.linspace(-4, 20, 2000)
    y = np.sin(x) * 5

    fig, ax = plt.subplots()
    ssp.highlight_axes(fig)
    plt.plot(x, y)
    ssp.set_major_tick_labels(label=r"$\pi$", unit=np.pi, mode="single", axis=ax.xaxis)
    ssp.set_major_tick_labels(label=r"t", unit=2, denominator=5, numerator=2, axis=ax.yaxis)

    if save_images:
        ssp.save_current_figure(chapter=0, number=8, folder="test_images", language=ssp.LANGUAGE)
    plt.close(fig)

def test_highlight_axes(save_images: bool):
    """Test highlight_axes function."""
    x = np.linspace(-10, 10, 400)
    y = np.linspace(-10, 10, 400)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z)

    if save_images:
        ssp.save_current_figure(chapter=0, number=9, folder="test_images", language=ssp.LANGUAGE)
    plt.close(fig)

def test_plot_poles_zeros(save_images: bool):
    """Test plot_poles_zeros function."""
    poles = np.array([-1 + 1j, -1 - 1j, -2])
    zeros = -1

    fig, ax = plt.subplots()
    ssp.highlight_axes(fig)
    ssp.plot_poles_zeros(poles, zeros, ax=ax)
    ax.set_title("Poles and Zeros Plot")

    if save_images:
        ssp.save_current_figure(chapter=0, number=10, folder="test_images", language=ssp.LANGUAGE)
    plt.close(fig)
