"""Pytest suite for sysplot."""

import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest
import control as ctrl

import sysplot as ssp

matplotlib.use("Agg")

# ---------------------------------------------------------------------
# LANGUAGE-DEPENDENT LABELS
# ---------------------------------------------------------------------
if ssp.LANGUAGE == "de":
    xlabel = "Zeit t [s]"
    ylabel = "Amplitude"
    title = "Beispielplot"
else:
    xlabel = "Time t [s]"
    ylabel = "Amplitude"
    title = "Example Plot"

# ---------------------------------------------------------------------
# Pytest fixture to enable image saving
# ---------------------------------------------------------------------
@pytest.fixture(scope="session")
def save_images() -> bool:
    """Always save images during tests."""
    return True


# ---------------------------------------------------------------------
# Helper function to extract color/linestyle from a Line2D
# ---------------------------------------------------------------------
def _get_marker_style(line):
    return line.get_color()

def _get_line_style(line):
    return line.get_linestyle()


# ---------------------------------------------------------------------
# Style / Cycler / Stem Plot Tests
# ---------------------------------------------------------------------

def test_get_style(save_images: bool):
    x = np.linspace(-2, 2, 400)
    fig, ax = plt.subplots(figsize=ssp.FIGURE_SIZE)
    ssp.highlight_axes(fig)

    for i in range(10):
        y = np.sin(x + i * 0.2) + 0.15 * i
        ax.plot(x, y, **ssp.get_style(i), label=f"Style index: {i}")

    ax.set_title("Manual Style Access (get_style)")
    ax.legend()
    if save_images:
        ssp.save_current_figure(chapter=0, number=6, folder="test_images", language=ssp.LANGUAGE)
    plt.close(fig)




def test_stem_advances_once_per_call(save_images: bool):
    fig, ax = plt.subplots()
    x = np.arange(5)
    y = np.ones(5)

    stem1, markers1, _ = ssp.plot_stem(x, y, ax=ax)
    stem2, markers2, _ = ssp.plot_stem(x + 1, y, ax=ax)

    # for m in markers1[0]:
    #     assert _get_marker_style(m) == style0["color"]

    # for s in stem1[0]:
    #     assert _get_line_style(s) == style0["linestyle"]

    # assert _get_marker_style(markers1[0]) == style0["color"]
    # assert _get_line_style(stem1[0][0]) == style0["linestyle"]
    # assert _get_marker_style(markers2[0][0]) == style1["color"]
    # assert _get_line_style(stem2[0][0]) == style1["linestyle"]

    if save_images:
        ssp.save_current_figure(chapter=0, number=0, folder="test_images", language=ssp.LANGUAGE)
    plt.close(fig)


def test_stem_and_plot_interaction(save_images: bool):
    fig, ax = plt.subplots()
    x = np.arange(5)
    y = np.ones(5)

    _, markers1, _ = ssp.plot_stem(x, y, ax=ax)
    line = ax.plot(x, y)[0]
    _, markers2, _ = ssp.plot_stem(x + 1, y, ax=ax)

    # assert _get_marker_style(markers1[0][0]) == style0["color"]
    # assert _get_line_style(markers1[0]) == style0["linestyle"]
    # assert _get_marker_style(line) == style1["color"]
    # assert _get_line_style(line) == style1["linestyle"]
    # assert _get_marker_style(markers2[0][0]) == style2["color"]
    # assert _get_line_style(markers2[0]) == style2["linestyle"]

    if save_images:
        ssp.save_current_figure(chapter=0, number=1, folder="test_images", language=ssp.LANGUAGE)
    plt.close(fig)


def test_style_index_does_not_advance(save_images: bool):
    fig, ax = plt.subplots()
    x = np.arange(5)
    y = np.ones(5)

    _, markers_fixed, _ = ssp.plot_stem(x, y, ax=ax, style_index=5)
    _, markers_auto, _ = ssp.plot_stem(x + 1, y, ax=ax)

    # assert _get_marker_style(markers_fixed[0]) == style5["color"]
    # assert _get_line_style(markers_fixed[0]) == style5["linestyle"]
    # assert _get_marker_style(markers_auto[0]) == style0["color"]
    # assert _get_line_style(markers_auto[0]) == style0["linestyle"]

    if save_images:
        ssp.save_current_figure(chapter=0, number=2, folder="test_images", language=ssp.LANGUAGE)
    plt.close(fig)


def test_axes_independent(save_images):
    fig, (ax1, ax2) = plt.subplots(2)
    x = np.arange(5)
    y = np.ones(5)

    _, m1, _ = ssp.plot_stem(x, y, ax=ax1)
    _, m2, _ = ssp.plot_stem(x, y, ax=ax2)

    # assert _get_marker_style(m1[0]) == style0["color"]
    # assert _get_line_style(m1[0]) == style0["linestyle"]
    # assert _get_marker_style(m2[0]) == style0["color"]
    # assert _get_line_style(m2[0]) == style0["linestyle"]

    if save_images:
        ssp.save_current_figure(chapter=0, number=3, folder="test_images", language=ssp.LANGUAGE)
    plt.close(fig)


def test_shared_manager_across_axes(save_images: bool):
    fig, (ax1, ax2) = plt.subplots(2)

    x = np.arange(5)
    y = np.ones(5)

    _, m1, _ = ssp.plot_stem(x, y, ax=ax1)
    _, m2, _ = ssp.plot_stem(x, y, ax=ax2)

    # assert _get_marker_style(m1[0]) == style0["color"]
    # assert _get_line_style(m1[0]) == style0["linestyle"]
    # assert _get_marker_style(m2[0]) == style1["color"]
    # assert _get_line_style(m2[0]) == style1["linestyle"]

    if save_images:
        ssp.save_current_figure(chapter=0, number=4, folder="test_images", language=ssp.LANGUAGE)
    plt.close(fig)


def test_no_double_advance(save_images):
    fig, ax = plt.subplots()

    ssp.plot_stem([0], [1], ax=ax)

    if save_images:
        ssp.save_current_figure(chapter=0, number=5, folder="test_images", language=ssp.LANGUAGE)
    plt.close(fig)


# ---------------------------------------------------------------------
# Image generation / standard plotting tests
# ---------------------------------------------------------------------



def test_dynamic_subplots(save_images: bool):
    x = np.linspace(-2, 2, 400)
    nrows, ncols = 2, 3
    figsize = ssp.get_figsize(nrows=nrows, ncols=ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    ssp.highlight_axes(fig)

    for idx, ax in enumerate(axs.flat):
        ax.plot(x, np.exp(-(idx + 1) * x**2))
        ax.set_title(f"Subplot {idx+1}")

    fig.suptitle("Dynamic Figure Size (get_figsize)")
    if save_images:
        ssp.save_current_figure(chapter=0, number=7, folder="test_images", language=ssp.LANGUAGE)
    plt.close(fig)


def test_stem_save_image(save_images: bool):
    x = np.arange(0, 10, 1)
    rng = np.random.default_rng(0)
    y = rng.random(10)

    fig, ax = plt.subplots(figsize=ssp.FIGURE_SIZE)
    ssp.highlight_axes(fig)
    ssp.plot_stem(x, y - 0.5, marker="^", markers_outwards=True)
    ssp.plot_stem(x + 0.5, y, bottom=0.25, show_baseline=False)

    if save_images:
        ssp.save_current_figure(chapter=0, number=8, folder="test_images", language=ssp.LANGUAGE)
    plt.close(fig)


def test_nyquist_plot(save_images: bool):
    omega = np.logspace(-3, 8, 2000)
    system = ctrl.tf([1, 100], [1, 10])
    mag, phase, _ = ctrl.frequency_response(system, omega)
    H = mag * np.exp(1j * phase)

    fig, ax = plt.subplots(figsize=ssp.FIGURE_SIZE)
    ssp.highlight_axes(fig)
    ssp.plot_nyquist(np.real(H), np.imag(H), arrow_position=0.4, style_index=0)
    ax.set_title("Nyquist Plot with Arrow Positioning")

    if save_images:
        ssp.save_current_figure(chapter=0, number=9, folder="test_images", language=ssp.LANGUAGE)
    plt.close(fig)


def test_bode_plot(save_images: bool):
    omega = np.logspace(-1, 8, 2000)
    system = ctrl.tf([1, 100], [1, 10])
    mag, phase, _ = ctrl.frequency_response(system, omega)

    fig, ax = plt.subplots(1, 2, figsize=ssp.get_figsize(1, 2))
    ssp.highlight_axes(fig)
    ssp.plot_bode(mag, phase, omega, axes=ax)
    fig.suptitle("Bode Plot in dB")

    if save_images:
        ssp.save_current_figure(chapter=0, number=10, folder="test_images", language=ssp.LANGUAGE)
    plt.close(fig)


def test_minor_ticks(save_images: bool):
    x = np.logspace(-2, 8, 200)
    y = x * x

    fig, ax = plt.subplots()
    ssp.highlight_axes(fig)
    plt.semilogx(x, y)
    ssp.set_minor_log_ticks(axis=ax.xaxis)

    if save_images:
        ssp.save_current_figure(chapter=0, number=11, folder="test_images", language=ssp.LANGUAGE)
    plt.close(fig)


def test_major_ticks(save_images: bool):
    x = np.linspace(-4, 20, 2000)
    y = np.sin(x) * 5

    fig, ax = plt.subplots()
    ssp.highlight_axes(fig)
    plt.plot(x, y)
    ssp.set_major_tick_labels(label=r"$\pi$", unit=np.pi, mode="single", axis=ax.xaxis)
    ssp.set_major_tick_labels(label=r"t", unit=2, denominator=5, numerator=2, axis=ax.yaxis)

    if save_images:
        ssp.save_current_figure(chapter=0, number=12, folder="test_images", language=ssp.LANGUAGE)
    plt.close(fig)


def test_highlight_axes(save_images: bool):
    x = np.linspace(-10, 10, 400)
    y = np.linspace(-10, 10, 400)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z)

    if save_images:
        ssp.save_current_figure(chapter=0, number=13, folder="test_images", language=ssp.LANGUAGE)
    plt.close(fig)


def test_plot_poles_zeros(save_images: bool):
    poles = np.array([-1 + 1j, -1 - 1j, -2])
    zeros = -1

    fig, ax = plt.subplots()
    ssp.highlight_axes(fig)
    ssp.plot_poles_zeros(poles, zeros, ax=ax)
    ax.set_title("Poles and Zeros Plot")

    if save_images:
        ssp.save_current_figure(chapter=0, number=14, folder="test_images", language=ssp.LANGUAGE)
    plt.close(fig)


def test_multiple_pole_zero_diagrams(save_images: bool):
    fig, ax = plt.subplots(1, 1, figsize=ssp.get_figsize(1, 1))
    ssp.highlight_axes(fig)

    pole_sets = [
        np.array([-1 + 1j, -1 - 1j, -2]) + 0,
        np.array([-2 + 2j, -2 - 2j]) + 3,
        np.array([-3]) + 6,
        np.array([-1, -2, -3]) + 9,
        np.array([-2 + 1j, -2 - 1j, -4]) + 12
    ]
    zero_sets = [
        np.array([1]) + 0,
        np.array([2]) + 3,
        np.array([]) + 6,
        np.array([0]) + 9,
        np.array([1, 2]) + 12
    ]

    for i, (poles, zeros) in enumerate(zip(pole_sets, zero_sets)):
        ssp.plot_poles_zeros(poles, zeros, ax=ax, label=f"Poles and Zeros: {i}")

    ax.set_title("Poles/Zeros")
    ax.legend()
    fig.suptitle("Multiple Pole-Zero Diagrams")
    if save_images:
        ssp.save_current_figure(chapter=0, number=15, folder="test_images", language=ssp.LANGUAGE)
    plt.close(fig)


def test_plot_then_stem_interaction(save_images: bool):
    x = np.arange(5)
    y = np.arange(5)
    fig, ax = plt.subplots()

    markerlines, stemlines, baseline = ssp.plot_stem(x, y + 1, ax=ax, markers_outwards=False)
    line1, = ax.plot(x, y)    

    if save_images:
        ssp.save_current_figure(chapter=0, number=16, folder="test_images", language=ssp.LANGUAGE)

    assert(markerlines[0].get_color() != line1.get_color())

    plt.close(fig)


def test_stem_then_plot_interaction(save_images: bool):
    x = np.arange(5)
    y = np.arange(5)
    fig, ax = plt.subplots()

    ssp.plot_stem(x, y, ax=ax, marker="^", markers_outwards=True)
    plt.plot(x, y)

    if save_images:
        ssp.save_current_figure(chapter=0, number=17, folder="test_images", language=ssp.LANGUAGE)
    plt.close(fig)