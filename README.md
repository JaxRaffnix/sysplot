<p align="center">
  <img src="docs\_static\icon.svg" width="120">
</p>

# Sysplot

Centralized plotting utilities for reproducible, publication-quality figures in system theory and control engineering.

It is designed to be used as an extension to Matplotlib and works best when combined with Numpy and Control.

See the [docs](https://jaxraffnix.github.io/sysplot) for detailed usage instructions, API reference, and examples.

## Features

An implementation of the features can be seen here: [docs/examples/quick_start.py](docs/examples/quick_start.py).

- **Default Configuration** — consistent styling for publication-quality plots
  - Default values for plot element properties
  - adjusted seaborn theme
  - activated Contraint layout
  - removed xmargin for time continous plots
- **Global style management** — Apply color and line styles uniformly across plots
  
  - Custom plot style cycler with color and linestyle
  - synchronised style cycler for all plot functions
- **Control-theory visualizations** — Specialized plotters systems
  - pole zero plots, includes origin in axis space, reenables xmargin
  - stem plots with toggleable baseline, continous base line stretching larger than the data range, autmatic switchting between directional markers to always face outside
  - nyquist plot with mirrored around real axis, arroww indicating direction, axes with euqal aspect ratio
  - bode plot with dB conversion, axis log scaling, automatic phase unwrapping, automatic phase axis in radian with pi labels
  - unit circle plot, optionally sets euqal aspect ratio for axes, default same colors as grid lines
  - filter tolerance plotter to show off forbidden areas for filter design. optionally created labeld arrows underneath the x axis or creates legend entries
- **Figure Manipulation**
  - Automatically compute figure dimensions
  - helper function to save figure with automatic filenames support
- **Tick Manipulation**
  - indicate integer steps for base in log axis with minor ticks
  - set major ticks at multiples of a arbiraty unit and show with custom label. steps between ticks can be controlled as a fraction with given numerator and denonimaotr. automatically reduces to greatest common devisor. Support symmetric mode where ticks is only placed at unit and -unit. label will always be displayed as latex string.
  - manually add an additional tick with dotted gridline. usefuil for log axis so the base**n is not interfered.
- **Axis helpers** — Highlight axes, set symmetric limits, and configure custom tick labels
  - highlight axes with emphasized gridline
  - wrapper to repeat axis and ticks in shared axes
  - wrapper to add origin to plot without making the plot call point visible
- **Angle Plotter** — Plot angles between lines
  - A very thin wrapper around `matplotlib.patches.Arc` to plot angles between lines. Automatically calculates the angle and direction between 3 points and plots an arc with an optional label.

## Installation

`sysplot` is available on PyPI. This assumes you are workiing on Windows and have [uv](https://docs.astral.sh/uv/getting-started/installation/) installed. Python 3.9 or higher is required.

```bash
uv add sysplot
```

Or with pip:

```bash
pip install sysplot
```

## Minimum Example

```python
import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
import sysplot as ssp

# apply default configuration 
ssp.apply_config() 

# Generate frequency response
omega = np.logspace(-2, 2, 300)
system = ctrl.tf([6.25], [1, 3 , 6.25])
mag, phase, _ = ctrl.frequency_response(system, omega)

# Create Bode plot
fig, axes = plt.subplots(1, 2, figsize=ssp.get_figsize(1, 2))

# ** sysplot is used here **
ssp.plot_bode(mag, phase, omega, axes=axes)

# Labels
axes[0].set(title="Magnitude", xlabel=r"$\omega$ [rad/s]", ylabel="dB")
axes[1].set(title="Phase", xlabel=r"$\omega$ [rad/s]", ylabel="rad")
plt.show()
```

![Bode Plot](docs/_static/minimum_example.png)

## Contributing

See [CONTRIBUTING](CONTRIBUTING.md) for dev setup, code requirements, and contribution guidelines.

## License

MIT License – see [LICENSE](LICENSE) for details.

## Ideas / To Do

- there is still an issue with plot_pole_zero with no linestyle not upadting the cycler index
- maybe add default arrow with text plotter? default linewidth, arrow size, ...
- publish to pypi
- automatically link gallery examples from function reference
- update install guide. test with laptop.
- update, improve, synchronize the intro in readme, init, index files.
- update contributing guide with more detailed instructions for testing, documentation, and code style.
- explain the syspot features in the examples. phase in pi/2, mirrores nyquist, arrows, markers for pole and zero, ....

# copilot promt

for my function: plot_angle()

use google style docstring. improve docstring. improve parameter types. add necessary input validation. 
dont list the raised expections in the docstring.s
if a default argument is reponsible for the asthetics of the plot, then it should have a default value in `SysplotConfig`. the function parameter should be default None and be loaded from the config inside the function body.
creaate a file in docs\examples to show the usage of this function. Dont have a code examples in the docstring. for examples, use `import sysplot as ssp`.
