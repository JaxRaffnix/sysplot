Getting Started
=====================

Use ``sysplot`` to create consistent, publication-quality figures for control systems analysis. The package provides utilities for figure sizing, styling, and visualizations like Bode plots and Nyquist diagrams.

Install sysplot
-------------------

``sysplot`` is available on PyPI.

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

         uv pip install -e relative//path/to/sysplot

   .. tab-item:: pip

      .. code-block:: bash

         pip install -e relative//path/to/sysplot

Minimum Example
-----------------

After installing, you can import sysplot in your Python code and start using it to create various types of plots for control systems analysis. For example, you can run the minimum example script:

.. literalinclude:: examples/minimum_example.py
   :language: python
   :caption: Minimum example

which produces the following figure:

.. image:: _static/minimum_example.png
   :alt: Minimum example output

For a more comprehensive example, please check out the quick start file: `examples/quick_start.py <examples/quick_start.py>`_.

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