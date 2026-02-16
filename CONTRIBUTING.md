 # Contributing to sysplot

Thanks for your interest in contributing. This guide covers setup, workflows, and expectations.

## Quick start

1) Create and activate a virtual environment.

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2) Install the package in editable mode with dev dependencies.

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

## Code style

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
