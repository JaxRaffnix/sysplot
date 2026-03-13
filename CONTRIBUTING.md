# Contributing to Sysplot

Thanks for your interest in contributing. This page covers:

- How to report issues.
- Setting up a development environment.
- Developing with sysplot.
- Preparing your work for merge requests.

## Reporting Issues

You've discovered a bug or something else you want to change? Great!
Please open an issue with the following information:

- A minimal script or steps to reproduce the issue.
- Expected vs. actual behavior.
- Your Python version and OS.
- Any relevant error output.


## Install a Development Environment

I recommend using [uv](https://docs.astral.sh/uv/getting-started/installation/). Also, Python 3.11 or higher is required.

```bash
git clone https://github.com/JaxRaffnix/sysplot.git
cd sysplot
uv sync --extra dev --extra docs
```
## Working on Sysplot

If you want to work directly on sysplot, please set up a development environmnet for yourself. The following sections covers
often used commands for development. When you are finished, please submit a pull request.

### Adding Dependencies

If you are increasing the scope of sysplot and require a new dependency, add it to the project with:

```bash
uv add <package>
```

If you instead only use this package for yourself in your environment, add it to your dependency group:

```bash
uv add <package> --dev 
```

If a package should be added for all develeopers (e.g. a new CI feature), add it to the optionals:

```bash
uv add --optional dev <package>
uv add --optional docs <package>
```

> [!Important]
> When you are done, please update the lock file and refresh the dependencies for all groups:

```bash
uv sync --extra dev --extra docs
uv lock --refresh
```

### Running Code Tests

Use pytest to check for breaking changes. Please also add your own tests for new features or behavior changes.
Ruff is used for linting and formatting, and ty is used for type checking. The same code will be run in the CI.

```bash
uv run pytest 

uv run ruff check --fix
uv run ruff format

uv run ty check

uv build --no-sources
```

### Building Documentation

The HTML documentation is generated with Sphinx. While the GitHub workflow creates the webpage, you can also generate
the documentation locally to test changes to the docs. To do this, run the following commands in the docs directory:

```bash
cd docs

# this is for windows, on linux or macOS use `make` instead.
.\make clean
.\make html 
```

### Publishing

The package is hosted on PyPI and the version number is exclusively handled in the pyproject.toml file. 
To update the version number, select either `major`, `minor`, or `patch` depending on the nature of the changes. 
This snippet shows how to publish the new version to [TestPyPi](https://test.pypi.org/project/sysplot/).

```powershell
uv version # current version
uv version --bump minor # or patch, major

uv build --no-sources
uv publish --token <token> --repository testpypi
```

To test your new version on [TestPyPi](https://test.pypi.org/project/sysplot/), download the package with

```bash
pip install sysplot --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple
```

---------

For the live version, select your SemVer with either `` or manually udpate the `pyproject.toml` file,
then run to launch the GitHub workflow:

```powershell
uv version --bump <major|minor|patch>
$version = python -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])"
$version
git add .
git commit -m "Prepared for release $version"
git push

git tag -a "v$version" -m "Release v$version"
git push --tags
```

## Preparing for Merge Requests

### Code Requirements

- Use Google style docstrings for all public functions and classes.
- Dont list Raised Errors in Docstring
- Every public function should have a gallery example. Reference to this example in the docstring with a minigallery. 
- Prefer clear, small functions over long, complex blocks.
- Add tests for behavior changes and new features.

### Submitting changes

1) Create a new branch for your work.
2) Ensure tests pass locally.
3) Update documentation if behavior or APIs change.
4) Open a pull request with a short summary and rationale.


