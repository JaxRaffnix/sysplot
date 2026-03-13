# Contributing to sysplot

Thanks for your interest in contributing. This guide covers:

- Development setup.
- sysplot install.
- Running tests.
- Building documentation.
- Code requirements.


## Install Development Environment

This assumes you have [uv](https://docs.astral.sh/uv/getting-started/installation/) installed. Python 3.11 or higher is required.

```bash
git clone https://github.com/JaxRaffnix/sysplot.git
cd sysplot
uv sync --extra dev --extra docs
```

### Running Tests

Tests are run with pytest. To disable saving of images during tests, set the environment variable SYSPLOT_DISABLE_SAVE to "1". Reset to "0".

```bash
$env:SYSPLOT_DISABLE_SAVE = "1"
uv run pytest 
```

### Building Documentation

The html documentation is generated with Sphinx, Github Workflow creates the webpage. To build yours locally, (e.g. to test changes to the docs) run the following commands in the docs directory:

```bash
cd docs
.\make html # this is for windows, on linux or macOS use `make html`
```

### Publishing

To update the version number, use bumpver. Select either --major, --minor, or --patch depending on the nature of the changes.

```bash
uv run bumpver update --patch

uv run ruff check --fix
uv run ruff format

uv run ty src

uv build
uv publish
```

<!-- TODO: renew uv lock. how to publish package and changes -->

## Code Requirements

- Use Google style docstrings for all public functions and classes.
- Dont list Raised Errors in Docstring
- Every public function should have a gallery example. Reference to this example in the docstring. 
- Prefer clear, small functions over long, complex blocks.
- Add tests for behavior changes and new features.

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
