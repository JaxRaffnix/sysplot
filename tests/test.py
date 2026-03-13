"""Assertion-focused pytest suite for sysplot."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import sysplot as ssp


LANGUAGE = "en"


@pytest.fixture(autouse=True)
def reset_sysplot_config() -> None:
    """Reset global config before and after each test for isolation."""
    ssp.reset_config()
    yield
    ssp.reset_config()
    plt.close("all")


@pytest.fixture(scope="session")
def save_images() -> bool:
    """Keep image saving enabled in tests."""
    return True


def test_sysplot_config_validation() -> None:
    cfg = ssp.SysplotConfig()
    assert cfg.figure_size == (7.0, 5.0)
    assert cfg.seaborn_style == "whitegrid"

    with pytest.raises(ValueError):
        ssp.SysplotConfig(font_size=0).validate()

    with pytest.raises(ValueError):
        ssp.SysplotConfig(linewidth=0).validate()


def test_apply_config_overrides_update_active_config_and_rcparams() -> None:
    ssp.apply_config(
        figure_size=(8.0, 4.0),
        font_size=13,
        linewidth=2.5,
        markersize=7,
        tick_direction="out",
        seaborn_style="ticks",
    )

    cfg = ssp.get_config()
    assert cfg.figure_size == (8.0, 4.0)
    assert cfg.font_size == 13
    assert cfg.linewidth == 2.5
    assert cfg.seaborn_style == "ticks"

    assert tuple(map(float, plt.rcParams["figure.figsize"])) == (8.0, 4.0)
    assert plt.rcParams["font.size"] == 13
    assert plt.rcParams["lines.linewidth"] == 2.5
    assert plt.rcParams["xtick.direction"] == "out"


def test_apply_config_accepts_config_instance() -> None:
    config = ssp.SysplotConfig(font_size=9, tick_direction="inout")
    ssp.apply_config(config=config)

    assert ssp.get_config() is config
    assert plt.rcParams["font.size"] == 9
    assert plt.rcParams["ytick.direction"] == "inout"


def test_apply_config_rejects_invalid_field() -> None:
    with pytest.raises(ValueError, match="Invalid config field"):
        ssp.apply_config(not_a_real_field=1)


def test_styles_and_get_style_index() -> None:
    style = ssp.get_style(index=0)
    assert set(style.keys()) == {"color", "linestyle"}
    assert style["color"] == ssp.custom_styles[0]["color"]
    assert style["linestyle"] == ssp.custom_styles[0]["linestyle"]


def test_get_style_with_axis_advances_cycler() -> None:
    fig, ax = plt.subplots()
    first = ssp.get_style(ax=ax)
    second = ssp.get_style(ax=ax)

    assert first["color"] != second["color"]

    plt.close(fig)


def test_plot_stem_directional_markers_flips_marker(save_images: bool) -> None:
    fig, ax = plt.subplots()

    markerlines, stemlines, baselines = ssp.plot_stem(
        x=np.array([0, 1]),
        y=np.array([1, -1]),
        ax=ax,
        marker="^",
        directional_markers=True,
        show_baseline=True,
    )

    assert len(markerlines) == 2
    assert len(stemlines) == 2
    assert len(baselines) == 2
    assert markerlines[0].get_marker() == "^"
    assert markerlines[1].get_marker() == "v"
    assert baselines[0].get_visible()

    if save_images:
        out = ssp.save_current_figure(chapter=0, number=1, language=LANGUAGE, fmt="png")
        assert Path(out).exists()

    plt.close(fig)


def test_get_figsize_uses_config_and_caps_with_nmax() -> None:
    ssp.apply_config(figure_size=(3.0, 2.0), figure_size_nmax=2)

    assert ssp.get_figsize(nrows=1, ncols=3) == (6.0, 2.0)
    assert ssp.get_figsize(nrows=3, ncols=1) == (3.0, 4.0)

    with pytest.raises(ValueError):
        ssp.get_figsize(nrows=0, ncols=1)


def test_save_current_figure_creates_expected_file(save_images: bool) -> None:
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    if save_images:
        out = ssp.save_current_figure(
            chapter=0,
            number=2,
            language=LANGUAGE,
            suffix="assert",
            fmt="png",
        )
        out_path = Path(out)
        assert out_path.exists()
        assert out_path.suffix == ".png"
        assert f"{Path(__file__).stem}" in out_path.name

    plt.close(fig)


def test_plot_poles_zeros_requires_non_empty_data() -> None:
    fig, ax = plt.subplots()

    with pytest.raises(ValueError, match="At least one of poles or zeros"):
        ssp.plot_poles_zeros([], [], ax=ax)

    plt.close(fig)


def test_highlight_axes_adds_coordinate_lines_once() -> None:
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    ssp.emphasize_coord_lines(fig)
    ssp.emphasize_coord_lines(fig)

    coord_x = [line for line in ax.lines if line.get_gid() == "coord_x"]
    coord_y = [line for line in ax.lines if line.get_gid() == "coord_y"]
    assert len(coord_x) == 1
    assert len(coord_y) == 1

    plt.close(fig)


def test_restore_tick_labels_reenables_tick_labels() -> None:
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    ssp.restore_tick_labels(fig)

    for ax in axs:
        ax.plot([0, 1], [0, 1])
        ax.tick_params(labelbottom=False, labelleft=False)
    fig.canvas.draw()

    for ax in axs:
        assert all(label.get_visible() for label in ax.get_xticklabels())
        assert all(label.get_visible() for label in ax.get_yticklabels())

    plt.close(fig)


def test_set_xmargin_toggles_margin() -> None:
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])

    ssp.set_xmargin(ax=ax, use_margin=False)
    x_margin, _ = ax.margins()
    assert x_margin == 0

    ssp.set_xmargin(ax=ax, use_margin=True)
    x_margin_after, _ = ax.margins()
    assert x_margin_after >= 0

    plt.close(fig)


def test_plot_angle_adds_annotation_patch() -> None:
    fig, ax = plt.subplots()
    ssp.plot_angle(
        center=(0, 0),
        point1=(1, 0),
        point2=(0, 1),
        text=r"$\theta$",
        ax=ax,
    )

    assert len(ax.patches) == 1

    plt.close(fig)


def test_plot_angle_returns_correct_angle() -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    fig.canvas.draw()  # finalise transforms so pixel-space angles are correct

    # 90° between the +x and +y unit vectors
    angle_90 = ssp.plot_angle(
        center=(0.0, 0.0),
        point1=(1.0, 0.0),
        point2=(0.0, 1.0),
        text=r"$\theta$",
        ax=ax,
    )
    assert angle_90 == pytest.approx(90.0, abs=0.5)

    # 45° between (1,0) and (1,1)
    angle_45 = ssp.plot_angle(
        center=(0.0, 0.0),
        point1=(1.0, 0.0),
        point2=(1.0, 1.0),
        text=r"$\alpha$",
        ax=ax,
    )
    assert angle_45 == pytest.approx(45.0, abs=0.5)

    plt.close(fig)


def test_plot_nyquist_draws_main_and_mirror_curves() -> None:
    fig, ax = plt.subplots()
    real = np.array([1.0, 0.8, 0.4, 0.1])
    imag = np.array([0.0, 0.3, 0.2, 0.1])

    ssp.plot_nyquist(real, imag, ax=ax, mirror=True, arrow_position=0.5)

    assert len(ax.lines) == 2

    plt.close(fig)


def test_plot_bode_returns_two_axes_and_log_xscale() -> None:
    omega = np.logspace(-1, 2, 50)
    mag = 1 / np.sqrt(1 + omega**2)
    phase = -np.arctan(omega)
    fig, axarr = plt.subplots(1, 2)

    returned_fig, axes = ssp.plot_bode(mag=mag, phase=phase, omega=omega, axes=axarr)

    assert returned_fig is fig
    assert len(axes) == 2
    assert axes[0].get_xscale() == "log"
    assert axes[1].get_xscale() == "log"

    plt.close(fig)


def test_plot_bode_creates_and_returns_figure_and_axes() -> None:
    omega = np.logspace(-1, 2, 50)
    mag = 1 / np.sqrt(1 + omega**2)
    phase = -np.arctan(omega)

    fig, axes = ssp.plot_bode(mag=mag, phase=phase, omega=omega)

    assert fig is axes[0].figure
    assert fig is axes[1].figure
    assert len(axes) == 2

    plt.close(fig)


def test_plot_unit_circle_draws_curve() -> None:
    fig, ax = plt.subplots()
    start_line_count = len(ax.lines)

    ssp.plot_unit_circle(ax=ax)

    assert len(ax.lines) == start_line_count + 1

    plt.close(fig)


def test_plot_filter_tolerance_sets_limits_and_adds_masks() -> None:
    fig, ax = plt.subplots()
    ax.set_ylim(0, 1.2)
    bands = [
        {"type": "pass", "w0": 0.0, "w1": 1.0},
        {"type": "stop", "w0": 1.5, "w1": 2.5},
    ]

    ssp.plot_filter_tolerance(
        ax=ax,
        bands=bands,
        A_pass=0.9,
        A_stop=0.2,
        w_max=3.0,
        show_mask=True,
        show_arrows=False,
        set_ticks=True,
    )

    x0, x1 = ax.get_xlim()
    assert x0 == 0
    assert x1 == 3.0
    assert len(ax.patches) > 0

    plt.close(fig)


def test_tick_helpers_apply_locators_and_lines() -> None:
    fig, ax = plt.subplots()
    x = np.logspace(0, 2, 100)
    y = np.sin(np.log10(x))
    ax.plot(x, y)
    ax.set_xscale("log")

    ssp.set_minor_log_ticks(axis=ax.xaxis)
    ssp.set_major_ticks(label=r"$\\pi$", unit=np.pi, axis=ax.yaxis, denominator=2)

    line_count_before = len(ax.lines)
    text_count_before = len(ax.texts)
    ssp.add_tick_line(value=10.0, label="w_c", axis=ax.xaxis)

    assert ax.xaxis.get_minor_locator() is not None
    assert ax.yaxis.get_major_locator() is not None
    assert len(ax.lines) == line_count_before + 1
    assert len(ax.texts) == text_count_before + 1

    plt.close(fig)


def test_heaviside_matches_expected_values() -> None:
    x = np.array([-1.0, 0.0, 2.0])
    y = ssp.heaviside(x, default_value=0.5)

    assert np.allclose(y, np.array([0.0, 0.5, 1.0]))
