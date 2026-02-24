# sysplot

Centralized plotting utilities for reproducible, publication-quality figures in system theory and control engineering.

## Features

- **Consistent figure sizing** — Automatically compute figure dimensions for single and multi-subplot layouts
- **Global style management** — Apply color and line styles uniformly across plots
- **Control-theory visualizations** — Specialized plotters for Bode diagrams, Nyquist plots, and pole-zero maps
- **Axis helpers** — Highlight axes, set symmetric limits, and configure custom tick labels
- **Configurable output** — Save figures with automatic filename generation and language support (English/German)

## Installation

Install from PyPI:

```bash
pip install sysplot
```

Or alternatively, if you already have this repository cloned, you can access it from another project with:

```bash
pip install -e relative//path/to/sysplot
```

## Quick start

Example usage for creatinng a bode plot:
```python
import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
import sysplot as ssp

# Generate frequency response
omega = np.logspace(-1, 8, 2000)
system = ctrl.tf([1, 100], [1, 10])
mag, phase, _ = ctrl.frequency_response(system, omega)

# Create a figure with automatic sizing
fig, axes = plt.subplots(figsize=ssp.get_figsize(nrows=1, ncols=2))
ssp.highlight_axes(fig)

# Plot the Bode diagram
ssp.plot_bode(mag, phase, omega, axes=axes)
fig.suptitle("Bode Plot in dB")
axes[0].set_xlabel("Frequency [rad/s]")
axes[0].set_ylabel("Amplitude [dB]")
axes[1].set_ylabel("Phase [deg]")

# Save the figure
ssp.save_current_figure(chapter=1, number=1, folder="figures")
plt.show()
```

![Bode Plot](docs/auto_examples/images/sphx_glr_plot_example_004.png)

## TO DO

- Make test script current useful and use `assert`
- publish to pypi
- add example images to docstrings
- update example images to match the code
- update installing.rst to match code and image
- plot poles zeros should always show origin but not be symmetric around it.
- config constants should be set in the apply func, not be global variables.
- idea: create custom EngineeringFigure
- maybe add default arrow with text plotter? default linewidth, arrow size, ...

```python
fig = EngineeringFigure()
fig.plot_poles(poles)
fig.plot_zeros(zeros)
```

## Development

Run tests:

```powershell
pytest tests/test.py
```

Enable/Disable saving images during tests:

```powershell
$SYSPLOT_SAVE_IMAGES = 1 # or 0 to disable
```

## Documentation

See the [docs](https://jaxraffnix.github.io/sysplot) for detailed usage instructions, API reference, and examples.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on setup, testing, and submitting changes.

## License

MIT License – see [LICENSE](LICENSE) for details.
