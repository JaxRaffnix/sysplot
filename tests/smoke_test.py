"""Smoke test for validating a published sysplot install.

This file supports two execution modes:
1. ``uv run pytest tests/smoke_test.py``
2. ``python tests/smoke_test.py``
"""

from importlib.metadata import version

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import sysplot as ssp


def run_smoke_test() -> None:
	"""Run a minimal end-to-end check of the public plotting API."""
	package_version = version("sysplot")
	assert package_version

	ssp.apply_config()

	x = np.linspace(0.0, 1.0, 10)
	fig, ax = plt.subplots()

	style = ssp.get_style(ax=ax)
	assert "color" in style
	assert "linestyle" in style

	line, = ax.plot(x, x**2, **style)
	assert len(ax.lines) == 1
	assert line.get_color()

	fig.canvas.draw()
	plt.close(fig)


def test_smoke_import_and_plotting() -> None:
	"""Pytest entrypoint for smoke testing."""
	run_smoke_test()


if __name__ == "__main__":
	run_smoke_test()
	print("sysplot smoke test succeeded")