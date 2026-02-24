 # Contributing to sysplot

Thanks for your interest in contributing. This guide covers setup, workflows, and expectations.

>[!TIP]
> To use the sysplot package from another project, you can access your local develop repo with `pip install -e relative//path/to/sysplot`. This allows you to work on sysplot and test changes in your project without needing to publish to PyPI.

## Quick start

1) install in editable mode with development dependencies:

```bash
git clone https://github.com/JaxRaffnix/sysplot.git
cd sysplot
```

2) Create and activate a virtual environment.

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

3) Install the package in editable mode with dev dependencies.

```bash
pip install -e .[dev]
```

If your environment does not support extras, install from requirements instead:

```bash
pip install -r requirements.txt
```

## Running tests

Tests are written with pytest.

```bash
pytest
```

By default, image files are not saved. To save images during tests:

```bash
$env:SYSPLOT_SAVE_IMAGES = "1"
pytest
```

## Documentation

Build the docs with Sphinx:

```bash
cd docs
make html
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

## Code style

- Use Google style docstrings for all public functions and classes.
- Keep changes focused and minimal.
- Prefer clear, small functions over long, complex blocks.
- Add tests for behavior changes and new features.

## Submitting changes

1) Create a new branch for your work.
2) Ensure tests pass locally.
3) Update documentation if behavior or APIs change.
4) Open a pull request with a short summary and rationale.

## Reporting issues

When reporting bugs, please include:

- A minimal repro script or steps.
- Expected vs. actual behavior.
- Your Python version and OS.
- Any relevant error output.
