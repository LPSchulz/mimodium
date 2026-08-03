<p align="center">
  <img src="docs/_static/brand/mimodium-logo.svg" alt="Mimodium" width="720">
</p>

# Mimodium

Mimodium is a Python research library for reproducible cell-free massive MIMO
system-level simulations. It provides Dagreon task components for scenario
generation, propagation and channel modeling, signal-processing algorithms,
evaluation metrics, and visualization.

The documentation is available on
[Read the Docs](https://mimodium.readthedocs.io/).

## Installation

Mimodium requires Python 3.12 or newer. Install the published package with:

```bash
python -m pip install mimodium
```

To install Mimodium from a source checkout, run:

```bash
python -m pip install .
```

## Development

Create and activate a virtual environment from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
```

Then install Mimodium and its development dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Check formatting and lint the package, tests, and documentation with:

```bash
ruff format --check src tests docs
ruff check src tests docs
```

## Testing

Run the complete test suite with:

```bash
pytest
```

## Building the Documentation

Build a local HTML preview with:

```bash
sphinx-build -b html docs docs/_build/html
```

The generated documentation is written to `docs/_build/html`.

For release validation, build the documentation with warnings treated as
errors:

```bash
sphinx-build -E -W --keep-going -b html docs docs/_build/html
```

## License and Citation

Mimodium is licensed under `GPL-3.0-or-later`; see [LICENSE](LICENSE).

If you use Mimodium in a research publication, cite the software version and
exact commit using [CITATION.cff](CITATION.cff).

Mimodium is pre-release research software. Its API and numerical behavior may
continue to evolve; pin exact versions and commits for reproducible studies.
