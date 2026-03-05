Getting Started
=====================

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

If you have already cloned the repository and want to use it from another local project, install it in editable mode:

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
   :caption: Minimum sysplot example

which produces the following figure:

.. image:: _static/minimum_example.png
   :alt: Minimum example output

For a more comprehensive example, please check out the quick start file: `examples/quick_start.py <examples/quick_start.py>`_.

Development Installation
-----------------------------

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
         .venv\Scripts\Activate
         pip install -e ".[dev,docs]"