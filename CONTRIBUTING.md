# Contributing to `sysplot`

Thanks for your interest in contributing. This guide covers:

- Development setup.
- `sysplot` install.
- Running tests.
- Building documentation.
- Code requirements.


## Install `sysplot` and Setup Development Environment

This assumes you are workiing on Windows and have [uv](https://docs.astral.sh/uv/getting-started/installation/) installed. Python 3.9 or higher is required.

```bash
git clone https://github.com/JaxRaffnix/sysplot.git
cd sysplot
uv venv
.venv\Scripts\Activate.ps1
uv pip install -e ".[dev,docs]"
```

## Code Requirements

- Use Google style docstrings for all public functions and classes.
- Dont list Raised Errors in Docstring
- Every public function should have a gallery example. Reference to this example in the docstring. 
- Prefer clear, small functions over long, complex blocks.
- Add tests for behavior changes and new features.

## Local Commands

Run tests:

```bash
$env:SYSPLOT_SAVE_IMAGES = "1" # to save images during tests
pytest tests/test.py
```

Build the docs with Sphinx (on Windows):

```bash
cd docs
.\make html
```

## Submitting changes

1) Create a new branch for your work.
2) Ensure tests pass locally.
3) Update documentation if behavior or APIs change.
4) Open a pull request with a short summary and rationale.

## Reporting issues

When reporting bugs, please include:

- A minimal script or steps to reproduce the issue.
- Expected vs. actual behavior.
- Your Python version and OS.
- Any relevant error output.
