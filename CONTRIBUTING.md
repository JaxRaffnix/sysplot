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

### Adding Dependencies

If the dependency is necessary for the sysplot package, use 

```bash
uv add <package>
```

If a package should be added for all develeopers (e.g. a new test feature), add it to the optionals:

```bash
uv add --optional dev <package>
uv add --optional docs <package>
```

If a package is only needed for your personal environement, add it with to your dependecy goup:
```bash
uv add <package> --dev 
```

### Running Tests

Tests are run with pytest. To disable saving of images during tests, set the environment variable SYSPLOT_DISABLE_SAVE to "1". Reset to "0".

```bash
uv run pytest 
```

### Building Documentation

The html documentation is generated with Sphinx, Github Workflow creates the webpage. To build yours locally, (e.g. to test changes to the docs) run the following commands in the docs directory:

```bash
cd docs
# this is for windows, on linux or macOS use `make` instead.
.\make clean
.\make html 
```

### Publishing

This repository includes a GitHub workflow to publish to PyPI when changes are merged to main. The [documentation](https://jaxraffnix.github.io/sysplot) webpage is also updated automatically.

To update the version number, use bumpver. Select either --major, --minor, or --patch depending on the nature of the changes.

```bash
uv sync --extra dev --extra docs

uv lock --refresh

uv run ruff check --fix
uv run ruff format

uv run ty check

uv version --bump minor # or patch, major
uv build --no-sources
# uv publish publish with workflow instead

git tag -a vx.y.z -m vx.y.z
git push --tags
```

## Code Requirements

- Use Google style docstrings for all public functions and classes.
- Dont list Raised Errors in Docstring
- Every public function should have a gallery example. Reference to this example in the docstring with a minigallery. 
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
