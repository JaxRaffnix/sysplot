# sysplot

Centralized plotting utilities for reproducible, publication-quality figures in system theory and control engineering.

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
  - helper function to save figure with automatic filenames and language support
- **Tick Manipulation**
  - indicate integer steps for base in log axis with minor ticks
  - set major ticks at multiples of a arbiraty unit and show with custom label. steps between ticks can be controlled as a fraction with given numerator and denonimaotr. automatically reduces to greatest common devisor. Support symmetric mode where ticks is only placed at unit and -unit. label will always be displayed as latex string.
  - manually add an additional tick with dotted gridline. usefuil for log axis so the base**n is not interfered.
- **Axis helpers** — Highlight axes, set symmetric limits, and configure custom tick labels
  - highlight axes with emphasized gridline
  - wrapper to repeat axis and ticks in shared axes
  - wrapper to add origin to plot without making the plot call point visible
- **Angle Plotter** - Plot angles between lines
  - copied from matplolib example.

## Installation

Install from PyPI:

With uv: (link)

for first time project setup:

```bash
uv venv create
.venv\Scripts\Activate
uv init 
```

add sysplot with uv
```bash
uv add sysplot
```

without uv

```bash
pip install sysplot
```

## Quick start

Example usage for creatinng a bode plot with the following feature:


```python
import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
import sysplot as ssp

ssp.apply_config()  # apply default configuration 

# Generate frequency response
omega = np.logspace(-2, 2, 300)
system = ctrl.tf([2.5 **2], [1, 2*0.6*2.5 , 2.5 **2])
mag, phase, _ = ctrl.frequency_response(system, omega)

# Create Bode plot
fig, axes = plt.subplots(1, 2, figsize=ssp.get_figsize(1, 2))
ssp.plot_bode(mag, phase, omega, axes=axes)

axes[0].set(title="Magnitude", xlabel=r"$\omega$ [rad/s]", ylabel="dB")
axes[1].set(title="Phase", xlabel=r"$\omega$ [rad/s]", ylabel="rad")
plt.show()
```

![Bode Plot](docs/_static/minimum_example.png)

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

## Documentation

See the [docs](https://jaxraffnix.github.io/sysplot) for detailed usage instructions, API reference, and examples.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on setup, testing, and submitting changes.

## License

MIT License – see [LICENSE](LICENSE) for details.
