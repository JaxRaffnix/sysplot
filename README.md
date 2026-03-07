<p align="center">
  <img src="docs\_static\icon.svg" width="120">
</p>

# Sysplot

Sysplot provides centralized plotting utilities for reproducible, publication-quality figures in system theory and control engineering.

It extends Matplotlib with consistent figure styling, configuration management, specialized helpers for annotating and improving visual clarity, and high-level plotting functions for Bode plots, Nyquist diagrams, and pole-zero maps.

The project documentation is available on GitHub: https://jaxraffnix.github.io/sysplot.

## Installation

Sysplot is available on PyPI. Python 3.9 or higher is required.

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

- there is still an issue with plot_pole_zero with no linestyle not upadting the cycler index
- maybe add default arrow with text plotter? default linewidth, arrow size, ...
- publish to pypi
- update install guide. test with laptop.
- explain what happens with the style cycler when using plt.stem() with and without linestyles, basefmt, 
- maybe plotting functions that create a new figure if none exists should return the figure and axis? 
- add sorting order to api reference 

# Copilot promt

for my function: `<function_name>()`

- use google style docstring. 
- improve, clarify and simplify the docstring
- improve parameter type annotaion. 
- add necessary input validation. 
- dont list the raised expections in the docstring.
- Dont have a code examples in the docstring. for examples
- if a default argument is reponsible for the asthetics of the plot, then it should have a default value in `SysplotConfig`. the function parameter should be default None and be loaded from the config inside the function body.
- create a file in docs\examples to show the usage of this function. 
  - the example in docs/examples must have a reST docstring in the beginning of the file. there, explain waht the called sysplot function does and its features in a very short way
  - if a function settings changes the astehtics, show their effect.
  - refernce the gallery file in the docstring with
```
.. minigallery:: sysplot.<function_name>
  :add-heading:
```