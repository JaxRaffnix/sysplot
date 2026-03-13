<p align="center">
  <img src="docs\_static\wide.svg" width="120">
</p>

![PyPI](https://img.shields.io/pypi/v/sysplot)

# Sysplot

Sysplot provides centralized plotting utilities for reproducible, publication-quality figures in system theory and control engineering.

It extends Matplotlib with consistent figure styling, configuration management, specialized helpers for annotating and improving visual clarity, and high-level plotting functions for Bode plots, Nyquist diagrams, and pole-zero maps.

The project documentation is available on GitHub: https://jaxraffnix.github.io/sysplot.

## Installation

Sysplot is available on PyPI. [Python 3.11](https://www.python.org/downloads/) or higher is required. Install with [uv](https://docs.astral.sh/uv/):

```bash
uv add sysplot
```

Or with pip:

```bash
pip install sysplot
```

## Minimum Example

A single call to sysplot.plot_bode() produces a Bode plot; with magnitude in dB, phase unwrapped in multiples of 2𝜋, logarithmic frequency axis, and minor decade ticks included automatically.

![Bode Plot](docs\_auto_examples\images\sphx_glr_minimum_example_001.png)

```python
import numpy as np
import matplotlib.pyplot as plt
import control as ctrl
import sysplot as ssp

ssp.apply_config() # apply sysplot style 

# Generate frequency response
omega = np.logspace(-2, 2, 300)
system = ctrl.tf([6.25], [1, 3 , 6.25])
mag, phase, _ = ctrl.frequency_response(system, omega)

ssp.plot_bode(mag, phase, omega)    # ** sysplot is used here **
plt.show()
```

## Contributing

See [CONTRIBUTING](CONTRIBUTING.md) for dev setup, code requirements, and contribution guidelines.

## License

MIT License – see [LICENSE](LICENSE) for details.

## Ideas / To Do

- publish to pypi with github workflow
  - add code formatter, unused import cheker, type checker. ruff, ty
- explain what happens with the style cycler when using plt.stem() with and without linestyles, basefmt, 
- add sorting order to api reference by name

- manually keep minimum python version up to date in readme, docs, from toml.