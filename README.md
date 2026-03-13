<p align="center">
  <img src="docs/_static/wide.svg" width="400">
</p>

![PyPI](https://img.shields.io/pypi/v/sysplot)
![CI](https://github.com/JaxRaffnix/sysplot/actions/workflows/ci.yaml/badge.svg)
![Docs](https://github.com/JaxRaffnix/sysplot/actions/workflows/docs.yaml/badge.svg)
![Publish](https://github.com/JaxRaffnix/sysplot/actions/workflows/publish.yaml/badge.svg)

# Sysplot

Sysplot provides centralized plotting utilities for reproducible, publication-quality figures in system theory and control engineering.

It extends Matplotlib with consistent figure styling, configuration management, specialized helpers for annotating and improving visual clarity, and high-level plotting functions for Bode plots, Nyquist diagrams, and pole-zero maps.

The project documentation is available via GitHub [Pages](https://jaxraffnix.github.io/sysplot).

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

After you defined the magnitude, phase and frequency data for your system, a single call to
`sysplot.plot_bode` is all you need to generate a Bode plot. This will include a custom
seaborn theme, magnitude in dB, phase unwrapped in multiples of $2\pi$, phase tick labels in 
fractional multiples of $\frac{\pi}{2}$, and a logarithmic frequency axis with minor decade ticks included automatically.

![Bode Plot](docs/_auto_examples/images/sphx_glr_minimum_example_001.png)

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

axes[0].set(xlabel="rad/s", ylabel="dB", title="Bode Plot")
axes[1].set(xlabel="rad/s", ylabel="rad/s", title="Phase Plot")
plt.show()
```

## Contributing

See [CONTRIBUTING](CONTRIBUTING.md) for dev setup, code requirements, and contribution guidelines.

## License

MIT License – see [LICENSE](LICENSE) for details.

## TODO

- Add pytest for gallery examples to CI workflow.
- Allow set_major_ticks() formatting to be customized in the config. Dont force users to accept "num/dennum" without LaTeX formatting.
- Restore_tick_labels() sounds like it works together with set_major_ticks() and add_tick_line. Find a better name.