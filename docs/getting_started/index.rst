Getting Started
===============

Installation
------------

Mimodium requires Python 3.12 or newer.

Installation via pip
~~~~~~~~~~~~~~~~~~~~

Install the published package from PyPI with:

.. code-block:: bash

   python -m pip install mimodium

Pip installs Mimodium's runtime dependencies, including Dagreon,
automatically.

Installation from source
~~~~~~~~~~~~~~~~~~~~~~~~

From a Mimodium source checkout, install the package in editable mode with:

.. code-block:: bash

   python -m pip install -e .

This is useful when modifying Mimodium locally. To include the development
tools for testing, formatting, and building the documentation, install the
``dev`` extra:

.. code-block:: bash

   python -m pip install -e ".[dev]"
