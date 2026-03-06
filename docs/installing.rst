Getting Started
=====================

Use sysplot to create consistent, publication-quality figures for control systems analysis. The package provides utilities for figure sizing, styling, and visualizations like Bode plots and Nyquist diagrams.

Install sysplot
-------------------

sysplot is available on PyPI. We assume you have `Python 3.9 <https://www.python.org/downloads/>`_ or higher installed.

.. tab-set::

   .. tab-item:: uv

      .. code-block:: bash

         uv add sysplot

   .. tab-item:: pip

      .. code-block:: bash

         pip install sysplot

If you already have the development repository cloned, you can install it in editable mode to use it in other local projects:

.. tab-set::

   .. tab-item:: uv

      .. code-block:: bash

         uv pip install -e relative/path/to/sysplot

   .. tab-item:: pip

      .. code-block:: bash

         pip install -e relative/path/to/sysplot

Minimum Example
-----------------

After installing, you can import sysplot in your Python code and start using it to create various types of plots for control systems analysis. The :ref:`minimum example <sphx_glr__auto_examples_minimum_example.py>` demonstrates:

- Using the sysplot style.
- Modelling a transfer function.
- Generating a Bode plot with these features
    - Phase axis in radian with pi/2 ticks
    - frequency in log scale
    - magnitude in dB
    - minor log ticks every decade


If you are instead interested in a more comprehensive example that covers additional features, see the :ref:`quick start example <sphx_glr__auto_examples_quick_start.py>`.

Development Installation
-----------------------------

If you want to contribute to the development of sysplot, you can clone the repository and set up a development environment. This allows you to make changes to the code and test them locally. Please refer to the `Contributing.md <https://github.com/JaxRaffnix/sysplot/blob/main/CONTRIBUTING.md>`_ file for guidelines and a how-to for running tests or creating documentation.

.. tab-set::

   .. tab-item:: uv

      .. code-block:: bash

         git clone https://github.com/JaxRaffnix/sysplot.git
         cd sysplot
         uv venv
         .venv\Scripts\Activate
         uv sync --extra dev --extra docs

   .. tab-item:: pip

      .. code-block:: bash

         git clone https://github.com/JaxRaffnix/sysplot.git
         cd sysplot
         python -m venv .venv
         .venv\Scripts\Activate  # this works on windows
         pip install -e ".[dev,docs]"
