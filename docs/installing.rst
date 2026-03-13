Getting Started
=====================

Use sysplot to create consistent, publication-quality figures for control systems analysis. The package provides utilities for figure sizing, styling, and visualizations like Bode plots and Nyquist diagrams.

Install sysplot
-------------------

sysplot is available on `PyPI <https://pypi.org/project/sysplot/>`_. With `Python 3.11 <https://www.python.org/downloads/>`_ or higher, you can 
install the package with `uv <https://docs.astral.sh/uv/>`_ or pip:

.. tab-set::

   .. tab-item:: uv

      .. code-block:: bash

         uv add sysplot

   .. tab-item:: pip

      .. code-block:: bash

         pip install sysplot

If you already have the sysplot repository cloned locally on your machine and want to use it in another project, install it in editable mode:

.. tab-set::

   .. tab-item:: uv

      .. code-block:: bash

         uv pip install -e relative/path/to/sysplot

   .. tab-item:: pip

      .. code-block:: bash


         pip install -e path/to/sysplot

After installing, confirm that sysplot is available by running:

.. code-block:: python

   import sysplot
   print(sysplot.__version__)

If no error is raised, the installation was successful.

Examples
-----------------

To get a quick overview of the module, check out either of these examples. The quick start example
covers most features provided by sysplot, while the minimum example shows how to create a Bode plot with a single call to :func:`sysplot.plot_bode`.

.. minigallery::
   examples/minimum_example.py
   examples/quick_start.py

Development Installation
-----------------------------

If you want to contribute to sysplot, please clone the repository and set up a development environment. You can refer to the `CONTRIBUTING.md <https://github.com/JaxRaffnix/sysplot/blob/main/CONTRIBUTING.md>`_ file for guidelines on code style, running tests, and building the documentation.

.. tab-set::

   .. tab-item:: uv (Windows)

      .. code-block:: bash

         git clone https://github.com/JaxRaffnix/sysplot.git
         cd sysplot
         uv sync --extra dev --extra docs

   .. tab-item:: pip (Windows)

      .. code-block:: bash

         git clone https://github.com/JaxRaffnix/sysplot.git
         cd sysplot
         python -m venv .venv
         .venv\Scripts\Activate
         pip install -e ".[dev,docs]"

   .. tab-item:: pip (Linux / macOS)

      .. code-block:: bash

         git clone https://github.com/JaxRaffnix/sysplot.git
         cd sysplot
         python -m venv .venv
         source .venv/bin/activate
         pip install -e ".[dev,docs]"

Next Steps
----------

- :ref:`concepts` — understand the design principles behind sysplot.
- :ref:`api` — full API reference for all public functions.
